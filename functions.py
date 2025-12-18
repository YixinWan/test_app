import numpy as np
from skimage import color
import json
import os
from scipy.optimize import minimize
import itertools
import cv2
from skimage.segmentation import slic, find_boundaries
from skimage.color import rgb2gray
from skimage.filters import sobel, gaussian
from skimage.morphology import dilation, square

# -------------------------辅助函数------------------------------------------
def rgb_to_cmy(rgb):
    """
    将 RGB 值转换为 CMY 值。
    
    输入:
      - rgb: RGB 值列表或数组，例如 [255, 128, 0]
    
    输出:
      - cmy: CMY 值的 numpy 数组，范围 [0, 1]
    """
    return 1 - np.array(rgb) / 255.0

def cmy_to_rgb(cmy):
    """
    将 CMY 值转换为 RGB 值。
    
    输入:
      - cmy: CMY 值的数组或列表，范围 [0, 1]
    
    输出:
      - rgb: RGB 值的整数数组，范围 [0, 255]
    """
    return np.clip((1 - cmy) * 255, 0, 255).astype(int)

def smooth_image(img):
    """
    对图像进行平滑处理，使用均值漂移滤波。
    
    输入:
      - img: 输入图像，numpy 数组 (H, W, 3)，RGB 格式，uint8 类型
    
    输出:
      - 平滑后的图像，numpy 数组，与输入相同的形状和类型
    """
    return cv2.pyrMeanShiftFiltering(img, 10, 20)


# -------------------------计算混色建议函数------------------------------------------
def suggest_mix(target_rgb, palette_source, paint_colors=None, max_candidates=6):
    """
    给定目标 RGB 值和一个颜料调色盘（字典或 my_palette.json 路径），返回候选颜料名称与对应的混合权重。

    输入:
      - target_rgb: 可迭代对象，目标颜色的 RGB 值，例如 [255, 128, 0]
      - palette_source: 字典 {name: [r,g,b], ...} 或者指向 my_palette.json 的文件路径字符串。
      - paint_colors: 可选，完整的颜料色库（当 palette_source 为空时作为后备）。
      - max_candidates: 从调色盘中选取最接近的候选颜色数（默认 6）。

    输出:
      - top_colors: 列表，形如 [(name, [r,g,b]), ...]（按顺序为权重对应顺序）
      - weights: numpy 数组，对应于 top_colors 的比例（归一化和过滤掉很小的权重）

    注: 该函数是独立且自包含的，内部使用Lab空间作为色差度量，在CMY空间进行线性混色模拟，并尝试 1~4 色的线性混合优化。
    """
    # 规范化并加载 palette
    if isinstance(palette_source, str):
        try:
            if os.path.exists(palette_source):
                with open(palette_source, 'r', encoding='utf-8') as f:
                    palette = json.load(f)
            else:
                palette = {}
        except Exception:
            palette = {}
    elif isinstance(palette_source, dict):
        palette = palette_source
    else:
        palette = {}

    if not palette:
        palette = paint_colors or {}

    # 保证 palette 是 dict
    if not isinstance(palette, dict):
        palette = {}

    # 计算 Lab 色差的辅助函数（在函数内部自包含，方便迁移）
    def delta_e_local(rgb1, rgb2):
        lab1 = color.rgb2lab(np.array([[rgb1]])/255.0)[0, 0]
        lab2 = color.rgb2lab(np.array([[rgb2]])/255.0)[0, 0]
        return np.linalg.norm(lab1 - lab2)

    # 选择最接近的候选颜色
    try:
        sorted_items = sorted(palette.items(), key=lambda item: delta_e_local(target_rgb, item[1]))
    except Exception:
        sorted_items = list(palette.items())

    candidate_colors = sorted_items[:max_candidates]

    # 单色优先检查
    best_loss = 1e9
    best_colors = None
    best_weights = None

    for name, rgb_paint in candidate_colors:
        loss = delta_e_local(target_rgb, rgb_paint)
        if loss < best_loss:
            best_loss = loss
            best_colors = [(name, rgb_paint)]
            best_weights = np.array([1.0])

    # 如果单色已经足够接近则直接返回（阈值可调）
    if best_loss < 3:
        return best_colors, best_weights

    rng = np.random.default_rng(42)

    # 尝试 2~4 色组合的线性混合（CMY 空间混合）
    for n in range(2, 5):
        for comb in itertools.combinations(candidate_colors, n):
            palette_cmy = np.array([rgb_to_cmy(c[1]) for c in comb])

            def loss(w):
                mixed_cmy = np.dot(w, palette_cmy) # 线性混合模拟颜料混色 CMYmix​=w1​⋅CMY1​+w2​⋅CMY2​+...+wn​⋅CMYn​
                mixed_rgb = cmy_to_rgb(mixed_cmy)
                lab1 = color.rgb2lab(np.array([[target_rgb]])/255.0)[0, 0]
                lab2 = color.rgb2lab(np.array([[mixed_rgb]])/255.0)[0, 0]
                return np.linalg.norm(lab1 - lab2)

            N = len(comb)
            cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
            bounds = [(0, 1)] * N

            for _ in range(6): # 多次（6次）随机初始化以避免局部最优
                w0 = rng.random(N)
                w0 /= w0.sum()
                try:
                    res = minimize(loss, w0, bounds=bounds, constraints=cons, method='SLSQP')
                except Exception:
                    continue
                if res.success and res.fun < best_loss:
                    best_loss = res.fun
                    best_weights = res.x
                    best_colors = comb

        if best_loss < 2:
            break

    # 如果没有找到（极少情况），回退到最接近的单色
    if best_colors is None or best_weights is None:
        return [(sorted_items[0][0], sorted_items[0][1])], np.array([1.0])

    # 过滤掉极小权重并返回
    filtered = [(c, w) for c, w in zip(best_colors, best_weights) if w > 0.01]
    if filtered:
        top_colors, weights = zip(*filtered)
        return list(top_colors), np.array(weights)
    else:
        return list(best_colors), np.array(best_weights)

#-----------------------生成分步调色数据函数-------------------------------------------
def generate_steps_from_mix(top_colors, weights, max_total=15):
    """
    给定 suggest_mix 的输出（top_colors 和 weights），生成分步调色数据。内部集成小整数近似。

    参数:
      - top_colors: 列表，形如 [(name,[r,g,b]), ...]，应按权重降序排列
      - weights: 与 top_colors 对应的权重数组/列表（不必归一化）
      - max_total: 所有份数之和的上限（默认 15）

    返回:
      - 步骤列表，每项为字典：
        {
          'step_num': 步骤编号（从1开始）,
          'parts': 当前步骤各颜色份数列表,
          'names': 当前步骤各颜色名称列表,
          'rgbs': 当前步骤各颜色RGB列表,
          'mixed_hex': 混合后的HEX字符串 '#rrggbb'
        }

    说明：主色（第一个）份数固定，次要颜料逐份增加（1份, 2份, ...）
    """
    if not top_colors or weights is None:
        return []

    w = np.array(weights, dtype=float)
    if w.size == 0:
        return []

    # 归一化权重
    w_normalized = w / w.sum()

    # ========== 内联的小整数近似逻辑 ==========
    counts_best = None
    counts_best_err = float('inf')
    k = w_normalized.size

    # 穷举所有可能的份数组合（每项从1到max_total，且总和不超过max_total）
    import itertools as _it
    for combo in _it.product(range(1, max_total + 1), repeat=k):
        counts = np.array(combo, dtype=float)
        total = counts.sum()
        # 检查总和约束
        if total > max_total:
            continue
        fracs = counts / total
        err = np.sum((fracs - w_normalized) ** 2)  # 最小二乘误差
        if err < counts_best_err:
            counts_best_err = err
            counts_best = counts.astype(int).tolist()
            # 若误差已经非常小则可以提前退出
            if counts_best_err < 1e-6:
                break

    # 若没有找到满足约束的组合，降低上限并重试
    if counts_best is None:
        for relaxed_total in range(max_total - 1, 0, -1):
            for combo in _it.product(range(1, relaxed_total + 1), repeat=k):
                counts = np.array(combo, dtype=float)
                total = counts.sum()
                if total > relaxed_total:
                    continue
                fracs = counts / total
                err = np.sum((fracs - w_normalized) ** 2)
                if err < counts_best_err:
                    counts_best_err = err
                    counts_best = counts.astype(int).tolist()
            if counts_best is not None:
                break

    # 若仍未找到，退回到简单缩放方案
    if counts_best is None:
        total = min(max_total, max(1, int(round(1.0 / np.min(w_normalized)))))
        raw = (w_normalized * total).round().astype(int)
        raw[raw < 1] = 1
        if raw.sum() > max_total:
            # 超过限制时，按比例缩放
            scale = raw.sum() / max_total
            raw = (raw / scale).round().astype(int)
            raw[raw < 1] = 1
        counts_best = raw.tolist()

    # ========== 生成步骤 ==========
    steps = []
    primary_count = int(counts_best[0])
    secondary_counts = [int(c) for c in counts_best[1:]]

    step_num = 0

    # 逐步加入次要颜料：主色份数固定，其他颜料逐个增加
    for sec_idx, sec_target in enumerate(secondary_counts):
        for sec_cur in range(1, int(sec_target) + 1):
            step_num += 1
            parts = [primary_count]
            color_indices = [0]

            # 前 sec_idx 种次要颜料使用其目标份数
            for i in range(sec_idx):
                parts.append(int(secondary_counts[i]))
                color_indices.append(i + 1)

            # 当前次要颜料（第 sec_idx 种）使用 sec_cur
            parts.append(int(sec_cur))
            color_indices.append(sec_idx + 1)

            names = [top_colors[i][0] for i in color_indices]
            rgbs = [top_colors[i][1] for i in color_indices]

            # 计算混合 RGB（使用 CMY 模型）
            parts_arr = np.array(parts, dtype=float)
            parts_w = parts_arr / parts_arr.sum()
            palette_cmy = np.array([rgb_to_cmy(rgb) for rgb in rgbs])
            mixed_cmy = np.dot(parts_w, palette_cmy)
            mixed_rgb = cmy_to_rgb(mixed_cmy)
            mixed_hex = "#{:02x}{:02x}{:02x}".format(*mixed_rgb)

            steps.append({
                'step_num': step_num,
                'parts': parts,
                'names': names,
                'rgbs': rgbs,
                'mixed_hex': mixed_hex,
            })

    return steps

#-----------------------细分色块函数（SLIC）-------------------------------------------
def slic_color_blocks(image: np.ndarray, target_segments: int,
                      *, compactness: float = 8.0, sigma: float = 0.8) -> tuple:
    """
    用 SLIC 进行“按颜色为主”的超像素分割（封装函数，便于复用）。

    输入
    - image: 原图 (H,W,3) RGB, uint8 [0,255]
    - target_segments: 目标色块数量（“粗细”控制：值越大越细，块越多越小）
    - compactness: 形状紧凑度（越大形状更规整，默认 8.0，偏向颜色主导）
    - sigma: 分割前的高斯平滑（去噪，默认 0.8）

    输出 (labels, palette)
    - labels: (H,W) int，从 1 开始的色块 ID（0 保留为背景未用）
    - palette: dict[int, (r,g,b)]，每个 label 对应原图中的均值 RGB（uint8），用于前端重建均色图与点击取色

    """
    # 适度保边缘平滑，减少纹理噪点，但不破坏大边界
    sm = smooth_image(image)

    # 执行 SLIC，颜色主导 + 空间约束（start_label=1 方便下游）
    labels = slic(
        sm, n_segments=int(max(1, target_segments)), compactness=float(compactness),
        sigma=float(sigma), start_label=1
    )

    # 计算每个 label 的原图均值颜色 -> palette
    palette = {}
    uniq = np.unique(labels)
    for lab in uniq:
        if lab == 0:
            continue
        mask = labels == lab
        mean_color = np.mean(image[mask], axis=0)
        # 转为 uint8 三元组
        rgb = tuple(int(np.clip(round(c), 0, 255)) for c in mean_color)
        palette[int(lab)] = rgb

    return labels, palette

#-----------------------粗分色块函数-------------------------------------------
def coarse_color_blocks(
    image: np.ndarray,
    *,
    slic_segments: int = 5000,
    color_merge_thresh: float = 30.0,
    force_merge_thresh: float = 0.01,
    absorb_area_thresh: int = 800,
) -> tuple:
    """
    生成“粗分色块（铺色指导图）”：结构线约束 + 颜色相似 + 小面积合并。

    输入参数（核心可迁移）：
    - image: 原图 (H,W,3) RGB uint8
    - slic_segments: 初始 SLIC 超像素数（越大越细）
    - color_merge_thresh: 区域颜色合并阈值（越小越保守）
    - force_merge_thresh: 无结构线处强制合并阈值（线强 < 此值直接合并）
    - absorb_area_thresh: 小区域面积阈值（像素），小于此阈值尝试被吸收

    输出：
    - labels: (H,W) int 连续标签，从 1 开始
    - palette: dict[label] -> (r,g,b) uint8
    """

    # 内部推荐默认（无需在函数签名暴露）
    slic_compactness = 12.0
    edge_sigma = 1.0
    edge_merge_thresh = 0.25
    absorb_color_thresh = 70.0
    absorb_edge_protect_thresh = 0.3

    # 预处理与结构线
    smooth = smooth_image(image)
    # 调整结构线平滑强度（默认 1.0，与 extract_structure_lines 一致）
    gray = rgb2gray(smooth)
    grad = sobel(gray)
    grad = gaussian(grad, sigma=float(edge_sigma))
    grad = dilation(grad, square(2))
    edge_strength = np.clip(grad / max(grad.max(), 1e-8), 0, 1)

    # 初始 SLIC 分割
    segments = slic(
        smooth,
        n_segments=int(max(1, slic_segments)),
        compactness=float(slic_compactness),
        start_label=1
    )

    # Union-Find 初始化（按区域均值颜色）
    h, w = segments.shape
    parent = {}
    region_color = {}
    uniq = np.unique(segments)
    for lab in uniq:
        parent[int(lab)] = int(lab)
        mask = segments == lab
        region_color[int(lab)] = np.mean(image[mask], axis=0)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # 结构线约束 + 颜色合并
    for y in range(h - 1):
        for x in range(w - 1):
            a = int(segments[y, x])
            b = int(segments[y, x + 1])
            c = int(segments[y + 1, x])
            for u, v in ((a, b), (a, c)):
                if u == v:
                    continue
                cu = region_color[find(u)]
                cv = region_color[find(v)]
                color_dist = float(np.linalg.norm(cu - cv))
                edge_val = float(edge_strength[y, x])
                if edge_val < float(force_merge_thresh):
                    union(u, v)
                    continue
                if color_dist < float(color_merge_thresh) and edge_val < float(edge_merge_thresh):
                    union(u, v)

    # 构建合并后的标签
    merged = np.zeros_like(segments)
    label_map = {}
    new_label = 1
    for y in range(h):
        for x in range(w):
            root = find(int(segments[y, x]))
            if root not in label_map:
                label_map[root] = new_label
                new_label += 1
            merged[y, x] = label_map[root]

    # 小面积吸收式进一步合并
    labels = merged.copy()

    # 预计算属性
    props = {}
    uniq2 = np.unique(labels)
    for lab in uniq2:
        if lab == 0:
            continue
        mask = labels == lab
        props[int(lab)] = {
            "area": int(mask.sum()),
            "mean_color": np.mean(image[mask], axis=0),
            "mask": mask,
        }

    def find_neighbors(label):
        mask = labels == label
        dilated = dilation(mask, square(3))
        neighbor_labels = np.unique(labels[dilated])
        return [int(l) for l in neighbor_labels if int(l) != int(label) and int(l) != 0]

    def edge_between(mask_a, mask_b):
        boundary = find_boundaries(mask_a, mode="outer") & mask_b
        if boundary.sum() == 0:
            return 0.0
        return float(np.mean(edge_strength[boundary]))

    for lab, p in props.items():
        if p["area"] >= int(absorb_area_thresh):
            continue
        neighbors = find_neighbors(lab)
        if not neighbors:
            continue
        best_target = None
        best_score = np.inf
        for nb in neighbors:
            np_ = props.get(nb)
            if np_ is None:
                continue
            if np_["area"] <= p["area"]:
                continue
            eval = edge_between(p["mask"], np_["mask"])
            if eval > float(absorb_edge_protect_thresh):
                continue
            cdist = float(np.linalg.norm(p["mean_color"] - np_["mean_color"]))
            if cdist > float(absorb_color_thresh):
                continue
            score = cdist - 1e-4 * np_["area"]
            if score < best_score:
                best_score = score
                best_target = nb
        if best_target is not None:
            labels[p["mask"]] = best_target

    # 重排标签为连续 [1..K]
    uniq3 = np.unique(labels)
    uniq3 = [int(l) for l in uniq3 if int(l) != 0]
    remap = {lab: i + 1 for i, lab in enumerate(sorted(uniq3))}
    final_labels = np.zeros_like(labels)
    for lab, new_lab in remap.items():
        final_labels[labels == lab] = new_lab

    # palette 计算（用原图均值颜色）
    palette = {}
    for lab in sorted(remap.values()):
        mask = final_labels == lab
        mean_color = np.mean(image[mask], axis=0)
        rgb = tuple(int(np.clip(round(c), 0, 255)) for c in mean_color)
        palette[int(lab)] = rgb

    return final_labels, palette

