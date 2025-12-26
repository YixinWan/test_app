from typing import Dict, Tuple

import numpy as np
import cv2
from skimage.segmentation import slic, find_boundaries
from skimage.morphology import dilation, square
from .line_art import line_art


def smooth_image(img: np.ndarray) -> np.ndarray:
    """对图像进行平滑处理，使用均值漂移滤波。"""
    return cv2.pyrMeanShiftFiltering(img, 10, 20)




def slic_color_blocks(image: np.ndarray, target_segments: int,
                      *, compactness: float = 8.0, sigma: float = 0.8) -> tuple:
    """用 SLIC 进行“按颜色为主”的超像素分割。

    返回 (labels, palette):
    - labels: (H,W) int，从 1 开始的色块 ID
    - palette: dict[label] -> (r,g,b)
    """
    sm = smooth_image(image)

    labels = slic(
        sm,
        n_segments=int(max(1, target_segments)),
        compactness=float(compactness),
        sigma=float(sigma),
        start_label=1,
    )

    palette: Dict[int, Tuple[int, int, int]] = {}
    uniq = np.unique(labels)
    for lab in uniq:
        if lab == 0:
            continue
        mask = labels == lab
        mean_color = np.mean(image[mask], axis=0)
        rgb = tuple(int(np.clip(round(c), 0, 255)) for c in mean_color)
        palette[int(lab)] = rgb

    return labels, palette


def coarse_color_blocks(
    image: np.ndarray,
    *,
    slic_segments: int = 70,
    color_merge_thresh: float = 40.0,
    force_merge_thresh: float = 0.01,
    absorb_area_thresh: int = 10000,
) -> tuple:
    """生成“粗分色块（铺色指导图）”：结构线约束 + 颜色相似 + 小面积合并。

    返回 (labels, palette):
    - labels: (H,W) int 连续标签，从 1 开始
    - palette: dict[label] -> (r,g,b)
    """

    slic_compactness = 12.0
    edge_sigma = 1.0
    edge_merge_thresh = 0.25
    absorb_color_thresh = 70.0
    absorb_edge_protect_thresh = 0.3

    smooth = smooth_image(image)
    # 使用新的素描法生成的线稿（白底黑线，uint8），再转换为 [0,1] 的“边缘强度”(边缘=1)
    # 将 edge_sigma 粗略映射为高斯核大小（21 对应 ~1.0 的柔和度）
    est_k = max(3, int(round(edge_sigma * 10)) * 2 + 1)
    sketch = line_art(smooth, k_size=est_k)
    edge_strength = 1.0 - (sketch.astype(np.float32) / 255.0)

    segments = slic(
        smooth,
        n_segments=int(max(1, slic_segments)),
        compactness=float(slic_compactness),
        start_label=1,
    )

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

    labels = merged.copy()

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

    uniq3 = np.unique(labels)
    uniq3 = [int(l) for l in uniq3 if int(l) != 0]
    remap = {lab: i + 1 for i, lab in enumerate(sorted(uniq3))}
    final_labels = np.zeros_like(labels)
    for lab, new_lab in remap.items():
        final_labels[labels == lab] = new_lab

    palette: Dict[int, Tuple[int, int, int]] = {}
    for lab in sorted(remap.values()):
        mask = final_labels == lab
        mean_color = np.mean(image[mask], axis=0)
        rgb = tuple(int(np.clip(round(c), 0, 255)) for c in mean_color)
        palette[int(lab)] = rgb

    return final_labels, palette
