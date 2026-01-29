import numpy as np
import cv2
from typing import Tuple, List, Dict, Optional


def detect_hue_blocks(
    img: np.ndarray,
    *,
    input_bgr: bool = False,
    min_saturation: float = 0.1,
    min_value: float = 0.1,
    bin_size_deg: int = 2,
    smooth_sigma_deg: int = 6,
    min_prominence_ratio: float = 0.05,
    coverage_mode: str = "valley",
    min_ratio: float = 0.05,
) -> Dict[str, np.ndarray | List[Dict[str, float]]]:
    """
    基于色相直方图的波峰自动检测与色块划分。

    输入:
        img: RGB 或 BGR 图像 (H, W, 3)
        input_bgr: 若为 True，表示 img 为 BGR；默认认为是 RGB
        min_saturation: 忽略低饱和度像素 (灰度)
        min_value: 忽略低明度像素 (太暗)
        bin_size_deg: 直方图角度分辨率（度）
        smooth_sigma_deg: 平滑的高斯核标准差（度）
        min_prominence_ratio: 峰值显著性相对阈值
        coverage_mode: 边界方式（目前保留参数，默认 'valley'）
        min_ratio: 最小像素占比阈值（过滤微小色块）

    输出:
        dict 包含：
            - bins_deg: 每个 bin 的中心角度（度）
            - hist_counts: 原始数量直方图
            - smooth_density: 平滑后的密度曲线
            - peaks: list[dict]
            - blocks: list[dict]，每块包含 {start_deg, end_deg, center_deg, pixel_count, percentage, ratio}
    """
    if img is None:
        raise ValueError("图片未加载")

    # HSV转换与有效像素筛选 (OpenCV HSV: H in [0,180])
    if input_bgr:
        hsv_cv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    else:
        hsv_cv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h_cv, s_cv, v_cv = cv2.split(hsv_cv)
    h = h_cv.astype(np.float32) * 2.0            # 转为 0-360°
    s = s_cv.astype(np.float32) / 255.0          # 0-1
    v = v_cv.astype(np.float32) / 255.0          # 0-1
    valid_mask = (s >= float(min_saturation)) & (v >= float(min_value))
    valid_hues = h[valid_mask]

    if valid_hues.size == 0:
        return {
            "bins_deg": np.array([], dtype=np.float32),
            "hist_counts": np.array([], dtype=np.int32),
            "smooth_density": np.array([], dtype=np.float32),
            "peaks": [],
            "blocks": [],
        }

    # 构建直方图（数量与密度）
    edges = np.arange(0, 360 + bin_size_deg, bin_size_deg, dtype=np.float32)
    counts, edges = np.histogram(valid_hues, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2.0
    density = counts.astype(np.float32) / valid_hues.size

    # 环形平滑（高斯卷积，考虑360°环绕）
    sigma_bins = max(1, int(round(smooth_sigma_deg / bin_size_deg)))
    k_radius = max(1, 3 * sigma_bins)  # ~3σ
    x = np.arange(-k_radius, k_radius + 1, dtype=np.float32)
    gauss = np.exp(-(x**2) / (2.0 * sigma_bins**2))
    gauss /= gauss.sum()
    dens_ext = np.concatenate([density, density, density])
    smooth_ext = np.convolve(dens_ext, gauss, mode="same")
    smooth = smooth_ext[density.size:2*density.size]

    # 辅助函数：局部峰与谷检测
    def find_local_extrema(arr: np.ndarray) -> Tuple[List[int], List[int]]:
        maxima: List[int] = []
        minima: List[int] = []
        n = arr.size
        for i in range(n):
            prev = arr[(i - 1) % n]
            curr = arr[i]
            nextv = arr[(i + 1) % n]
            if curr > prev and curr > nextv:
                maxima.append(i)
            if curr < prev and curr < nextv:
                minima.append(i)
        return maxima, minima

    peaks_idx, _mins_idx = find_local_extrema(smooth)
    if len(peaks_idx) == 0:
        peaks_idx = [int(np.argmax(smooth))]

    # 峰显著性筛选（相对最大高度）
    max_height = float(np.max(smooth))
    min_prom = max_height * float(min_prominence_ratio)
    peaks_idx = [i for i in peaks_idx if smooth[i] >= min_prom]
    if len(peaks_idx) == 0:
        peaks_idx = [int(np.argmax(smooth))]

    # 计算峰间谷底作为边界（环形）
    peaks_idx_sorted = sorted(peaks_idx)
    blocks: List[Dict[str, float]] = []
    peaks: List[Dict[str, float]] = []
    n = smooth.size

    def interval_min_idx(a: int, b: int) -> int:
        # 返回 [a,b] 或环绕区间内的最小值位置
        if a <= b:
            vals = smooth[a:b + 1]
            idxs = np.arange(a, b + 1)
        else:
            vals = np.concatenate([smooth[a:n], smooth[0:b + 1]])
            idxs = np.concatenate([np.arange(a, n), np.arange(0, b + 1)])
        rel_min = int(np.argmin(vals))
        return int(idxs[rel_min])

    # 确定每个峰的左右边界（谷底）
    for k, p in enumerate(peaks_idx_sorted):
        p_next = peaks_idx_sorted[(k + 1) % len(peaks_idx_sorted)]
        p_prev = peaks_idx_sorted[(k - 1) % len(peaks_idx_sorted)]
        right_min = interval_min_idx(p, p_next)
        left_min = interval_min_idx(p_prev, p)
        peaks.append({
            "idx": float(p),
            "hue_deg": float(centers[p]),
            "height": float(smooth[p]),
            "left_idx": float(left_min),
            "right_idx": float(right_min),
        })

    # 构造色块区间（谷底作为边界，覆盖完整分布）
    for pk in peaks:
        start_idx = int(pk["left_idx"]) if isinstance(pk["left_idx"], float) else pk["left_idx"]
        end_idx = int(pk["right_idx"]) if isinstance(pk["right_idx"], float) else pk["right_idx"]
        if start_idx <= end_idx:
            mask_bins = np.arange(start_idx, n if end_idx == n else end_idx + 1)
        else:
            mask_bins = np.concatenate([np.arange(start_idx, n), np.arange(0, end_idx + 1)])
        pixel_count = int(np.sum(counts[mask_bins]))
        ratio = pixel_count / float(valid_hues.size)
        percentage = ratio * 100.0
        start_deg = float(centers[start_idx] - bin_size_deg/2)
        end_deg = float(centers[end_idx] + bin_size_deg/2)
        start_deg = (start_deg + 360.0) % 360.0
        end_deg = (end_deg + 360.0) % 360.0
        blocks.append({
            "start_deg": start_deg,
            "end_deg": end_deg,
            "center_deg": float(centers[int(pk["idx"]) if isinstance(pk["idx"], float) else pk["idx"]]),
            "pixel_count": float(pixel_count),
            "percentage": float(percentage),
            "ratio": float(ratio),
        })

    # 过滤与排序：按像素比例降序
    blocks = [b for b in blocks if b["ratio"] >= float(min_ratio)]
    blocks.sort(key=lambda b: b["ratio"], reverse=True)

    return {
        "bins_deg": centers,
        "hist_counts": counts,
        "smooth_density": smooth,
        "peaks": peaks,
        "blocks": blocks,
    }


def clean_mask(
    mask: np.ndarray,
    *,
    open_size: int = 3,
    close_size: int = 7,
    iterations_open: int = 1,
    iterations_close: int = 1,
    min_area: int = 150,
) -> np.ndarray:
    """
    对二值掩码做形态学开闭并移除小连通域，得到更干净、边界更清晰的区域。

    输入:
        mask: 8位单通道二值图 (0/255)
        open_size: 开运算核大小
        close_size: 闭运算核大小
        iterations_open: 开运算迭代次数
        iterations_close: 闭运算迭代次数
        min_area: 连通域最小面积阈值
    输出:
        cleaned: 清理后的二值掩码
    """
    if mask is None or mask.size == 0:
        return mask
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    m1 = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open, iterations=iterations_open)
    m2 = cv2.morphologyEx(m1, cv2.MORPH_CLOSE, k_close, iterations=iterations_close)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m2, connectivity=8)
    cleaned = np.zeros_like(m2)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= int(min_area):
            cleaned[labels == i] = 255
    return cleaned


def _hue_range_mask(hsv_img: np.ndarray, start_deg: float, end_deg: float) -> np.ndarray:
    """在 HSV 图上生成指定色相范围的原始二值掩码 (uint8)。"""
    # OpenCV: H in [0,180], S/V in [0,255]
    lower_cv = np.array([start_deg/2.0, 0, 0], dtype=np.float32)
    upper_cv = np.array([end_deg/2.0, 255, 255], dtype=np.float32)
    lower_cv_i = lower_cv.astype(np.uint8)
    upper_cv_i = upper_cv.astype(np.uint8)
    if start_deg <= end_deg:
        mask = cv2.inRange(hsv_img, lower_cv_i, upper_cv_i)
    else:
        # 跨越0°的区间：拆分两段再取并集
        mask1 = cv2.inRange(hsv_img, np.array([0, 0, 0], dtype=np.uint8), upper_cv_i)
        mask2 = cv2.inRange(hsv_img, lower_cv_i, np.array([180, 255, 255], dtype=np.uint8))
        mask = cv2.bitwise_or(mask1, mask2)
    return mask


def segment_hue_masks(
    img: np.ndarray,
    *,
    input_bgr: bool = False,
    min_saturation: float = 0.1,
    min_value: float = 0.1,
    bin_size_deg: int = 2,
    smooth_sigma_deg: int = 6,
    min_prominence_ratio: float = 0.04,
    min_ratio: float = 0.05,
    open_size: int = 3,
    close_size: int = 7,
    iterations_open: int = 1,
    iterations_close: int = 1,
    min_area: Optional[int] = None,
) -> Tuple[List[np.ndarray], List[Dict[str, float]]]:
    """
    对输入图像按 H 的分布自动划分色块，并对每块生成清理后的二值掩码。

    输出：
        (masks, blocks)
        - masks: List[uint8 mask]，已按块大小(比例)降序排列
        - blocks: 对应的块元数据列表
    """
    # 1) 自动检测色块区间
    result = detect_hue_blocks(
        img,
        input_bgr=input_bgr,
        min_saturation=min_saturation,
        min_value=min_value,
        bin_size_deg=bin_size_deg,
        smooth_sigma_deg=smooth_sigma_deg,
        min_prominence_ratio=min_prominence_ratio,
        min_ratio=min_ratio,
    )
    blocks = result["blocks"]
    if len(blocks) == 0:
        return [], []

    # 2) 逐块生成掩码并清理
    if input_bgr:
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    else:
        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    if min_area is None:
        h, w = img.shape[:2]
        min_area = max(150, int(0.0005 * (h * w)))  # 自适应下限

    masks: List[np.ndarray] = []
    for blk in blocks:
        raw_mask = _hue_range_mask(hsv_img, blk["start_deg"], blk["end_deg"])
        cleaned_mask = clean_mask(
            raw_mask,
            open_size=open_size,
            close_size=close_size,
            iterations_open=iterations_open,
            iterations_close=iterations_close,
            min_area=min_area,
        )
        masks.append(cleaned_mask)

    # 已按 blocks 的比例降序生成，因此 masks 已排序
    return masks, blocks


# 模块导出（便于在 app.py 中调用）
__all__ = [
    "detect_hue_blocks",
    "segment_hue_masks",
    "clean_mask",
]