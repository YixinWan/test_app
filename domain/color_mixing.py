import os
import json
import itertools
from typing import Dict, List, Tuple

import numpy as np
from skimage import color
from scipy.optimize import minimize


# -------------------------颜色空间与调色核心算法------------------------------------------

def rgb_to_cmy(rgb):
    """RGB -> CMY，输入 [0,255]，输出 [0,1]."""
    return 1 - np.array(rgb) / 255.0


def cmy_to_rgb(cmy):
    """CMY -> RGB，输入 [0,1]，输出 [0,255] int."""
    return np.clip((1 - cmy) * 255, 0, 255).astype(int)


def suggest_mix(target_rgb, palette_source, paint_colors=None, max_candidates=6):
    """给定目标 RGB 值和一个颜料调色盘，返回候选颜料名称与对应的混合权重。

    参数
    ------
    target_rgb: Iterable[int]
        目标颜色的 RGB 值，例如 [255, 128, 0]
    palette_source: dict 或 str
        - dict 形式: {name: [r,g,b], ...}
        - str 形式: 指向 my_palette.json 的文件路径
    paint_colors: dict, optional
        备用颜料色库（当 palette_source 为空时作为后备）。
    max_candidates: int
        从调色盘中选取最接近的候选颜色数（默认 6）。

    返回
    ------
    top_colors: List[(name, [r,g,b])]
        颜色列表（顺序与返回的权重对应）
    weights: np.ndarray
        对应于 top_colors 的比例（归一化且过滤掉很小的权重）
    """
    # 规范化并加载 palette
    if isinstance(palette_source, str):
        try:
            if os.path.exists(palette_source):
                with open(palette_source, "r", encoding="utf-8") as f:
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

    if not isinstance(palette, dict):
        palette = {}

    def delta_e_local(rgb1, rgb2):
        lab1 = color.rgb2lab(np.array([[rgb1]]) / 255.0)[0, 0]
        lab2 = color.rgb2lab(np.array([[rgb2]]) / 255.0)[0, 0]
        return np.linalg.norm(lab1 - lab2)

    try:
        sorted_items = sorted(palette.items(), key=lambda item: delta_e_local(target_rgb, item[1]))
    except Exception:
        sorted_items = list(palette.items())

    candidate_colors = sorted_items[:max_candidates]

    best_loss = 1e9
    best_colors = None
    best_weights = None

    # 单色优先
    for name, rgb_paint in candidate_colors:
        loss = delta_e_local(target_rgb, rgb_paint)
        if loss < best_loss:
            best_loss = loss
            best_colors = [(name, rgb_paint)]
            best_weights = np.array([1.0])

    if best_loss < 3:
        return best_colors, best_weights

    rng = np.random.default_rng(42)

    # 多色线性混合（CMY 空间）
    for n in range(2, 5):
        for comb in itertools.combinations(candidate_colors, n):
            palette_cmy = np.array([rgb_to_cmy(c[1]) for c in comb])

            def loss(w):
                mixed_cmy = np.dot(w, palette_cmy)
                mixed_rgb = cmy_to_rgb(mixed_cmy)
                lab1 = color.rgb2lab(np.array([[target_rgb]]) / 255.0)[0, 0]
                lab2 = color.rgb2lab(np.array([[mixed_rgb]]) / 255.0)[0, 0]
                return np.linalg.norm(lab1 - lab2)

            N = len(comb)
            cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1})
            bounds = [(0, 1)] * N

            for _ in range(6):
                w0 = rng.random(N)
                w0 /= w0.sum()
                try:
                    res = minimize(loss, w0, bounds=bounds, constraints=cons, method="SLSQP")
                except Exception:
                    continue
                if res.success and res.fun < best_loss:
                    best_loss = res.fun
                    best_weights = res.x
                    best_colors = comb

        if best_loss < 2:
            break

    if best_colors is None or best_weights is None:
        return [(sorted_items[0][0], sorted_items[0][1])], np.array([1.0])

    filtered = [(c, w) for c, w in zip(best_colors, best_weights) if w > 0.01]
    if filtered:
        top_colors, weights = zip(*filtered)
        return list(top_colors), np.array(weights)
    else:
        return list(best_colors), np.array(best_weights)


def generate_steps_from_mix(top_colors, weights, max_total=15):
    """给定 suggest_mix 的输出，生成分步调色数据。

    返回步骤列表：
    [
      {
        'step_num': int,
        'parts': [...],
        'names': [...],
        'rgbs': [...],
        'mixed_hex': '#rrggbb'
      },
      ...
    ]
    """
    if not top_colors or weights is None:
        return []

    w = np.array(weights, dtype=float)
    if w.size == 0:
        return []

    w_normalized = w / w.sum()

    counts_best = None
    counts_best_err = float("inf")
    k = w_normalized.size

    import itertools as _it

    for combo in _it.product(range(1, max_total + 1), repeat=k):
        counts = np.array(combo, dtype=float)
        total = counts.sum()
        if total > max_total:
            continue
        fracs = counts / total
        err = np.sum((fracs - w_normalized) ** 2)
        if err < counts_best_err:
            counts_best_err = err
            counts_best = counts.astype(int).tolist()
            if counts_best_err < 1e-6:
                break

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

    if counts_best is None:
        total = min(max_total, max(1, int(round(1.0 / np.min(w_normalized)))))
        raw = (w_normalized * total).round().astype(int)
        raw[raw < 1] = 1
        if raw.sum() > max_total:
            scale = raw.sum() / max_total
            raw = (raw / scale).round().astype(int)
            raw[raw < 1] = 1
        counts_best = raw.tolist()

    steps = []
    primary_count = int(counts_best[0])
    secondary_counts = [int(c) for c in counts_best[1:]]

    step_num = 0

    for sec_idx, sec_target in enumerate(secondary_counts):
        for sec_cur in range(1, int(sec_target) + 1):
            step_num += 1
            parts = [primary_count]
            color_indices = [0]

            for i in range(sec_idx):
                parts.append(int(secondary_counts[i]))
                color_indices.append(i + 1)

            parts.append(int(sec_cur))
            color_indices.append(sec_idx + 1)

            names = [top_colors[i][0] for i in color_indices]
            rgbs = [top_colors[i][1] for i in color_indices]

            parts_arr = np.array(parts, dtype=float)
            parts_w = parts_arr / parts_arr.sum()
            palette_cmy = np.array([rgb_to_cmy(rgb) for rgb in rgbs])
            mixed_cmy = np.dot(parts_w, palette_cmy)
            mixed_rgb = cmy_to_rgb(mixed_cmy)
            mixed_hex = "#{:02x}{:02x}{:02x}".format(*mixed_rgb)

            steps.append({
                "step_num": step_num,
                "parts": parts,
                "names": names,
                "rgbs": rgbs,
                "mixed_hex": mixed_hex,
            })

    return steps
