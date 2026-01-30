"""
亮部筛选 (Light) 模块

核心功能：对一张输入图，依据 HSV 的 V 与 S 的分布共同筛选亮部；
在每个色块图内，计算连通域的 V/S 高分位阈值，输出亮部白掩码。

封装目标：
- 处理单个色块图，返回亮部白掩码 (uint8, 0/255)
- 批量处理 static/color_blocks/ 下的色块图，将结果保存到 static/light_details/，文件名一一对应

注意：本模块不做任何可视化，仅提供可调用的函数，便于在 app.py 中集成。
"""

from typing import List, Dict, Tuple
import os
import glob
import numpy as np
import cv2


def bright_mask_for_block(
    block_bgr: np.ndarray,
    *,
    v_percentile: float = 0.80,
    s_percentile: float = 0.10,
    morph_kernel_size: int = 1,
    min_region_area: int = 10,
) -> Tuple[np.ndarray, List[Tuple[int, float, float]]]:
    """
    对单个“色块图”计算亮部白掩码。

    约定：色块图的非背景区域即“色块区域”。背景通常为纯黑或近似黑。

    输入:
        - block_bgr: HxWx3 BGR 图像（OpenCV 读取的格式）
    参数:
        - v_percentile: V 通道的分位数阈值（更亮）
        - s_percentile: S 通道的分位数阈值（更饱和）
        - morph_kernel_size: 亮部掩码的开运算核大小（去噪）
        - min_region_area: 连通域最小面积，过滤过小的亮部区域

    返回:
        - white_mask: HxW uint8 掩码（亮部=255，背景=0）
        - thresholds: list[(label_id, thr_v, thr_s)] 每个连通域的 V/S 阈值
    """
    if block_bgr is None or block_bgr.size == 0:
        raise ValueError("block_bgr is empty")

    # 以“非近似黑”作为色块区域掩码（对保存为黑背景的块图适配）
    # 阈值 8：允许轻微压缩/噪声；可按需调整
    block_gray = cv2.cvtColor(block_bgr, cv2.COLOR_BGR2GRAY)
    region_mask = (block_gray > 8).astype(np.uint8)
    if int(np.count_nonzero(region_mask)) == 0:
        # 无有效像素，返回全黑掩码
        return np.zeros(block_gray.shape, dtype=np.uint8), []

    hsv = cv2.cvtColor(block_bgr, cv2.COLOR_BGR2HSV)
    v_full = hsv[:, :, 2].astype(np.float32)
    s_full = hsv[:, :, 1].astype(np.float32)

    # 在区域内做连通域标记
    num, labels = cv2.connectedComponents(region_mask, connectivity=8)

    white_mask = np.zeros_like(block_gray, dtype=np.uint8)
    thresholds: List[Tuple[int, float, float]] = []

    for lab in range(1, int(num)):
        comp_mask = (labels == lab)
        v_vals = v_full[comp_mask]
        s_vals = s_full[comp_mask]
        if v_vals.size == 0 or s_vals.size == 0:
            continue

        thr_v = float(np.quantile(v_vals, v_percentile))
        thr_s = float(np.quantile(s_vals, s_percentile))
        thresholds.append((lab, thr_v, thr_s))

        # 亮部条件：V >= thr_v 且 S <= thr_s
        comp_bright = comp_mask & (v_full >= thr_v) & (s_full <= thr_s)

        # 形态学开运算去噪
        if morph_kernel_size and morph_kernel_size > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
            comp_bright = cv2.morphologyEx(comp_bright.astype(np.uint8), cv2.MORPH_OPEN, k).astype(bool)

        if int(np.count_nonzero(comp_bright)) >= int(min_region_area):
            white_mask[comp_bright] = 255

    return white_mask, thresholds


def process_light_color_blocks_directory(
    input_dir: str,
    output_dir: str,
    *,
    v_percentile: float = 0.80,
    s_percentile: float = 0.50,
    morph_kernel_size: int = 1,
    min_region_area: int = 10,
) -> List[Dict[str, str]]:
    """
    扫描 input_dir 下的色块图（如 .png/.jpg），逐一生成亮部白掩码，
    并保存到 output_dir 中，文件名保持一致。

    返回处理记录列表：[{"src": path, "dst": path, "ok": True/False}]
    """
    os.makedirs(output_dir, exist_ok=True)

    # 仅处理常见图片后缀
    patterns = ["*.png", "*.jpg", "*.jpeg", "*.webp"]
    files: List[str] = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(input_dir, p)))
    files = sorted(files)

    results: List[Dict[str, str]] = []
    for src_path in files:
        img = cv2.imread(src_path, cv2.IMREAD_COLOR)
        if img is None:
            results.append({"src": src_path, "dst": "", "ok": False, "reason": "imread failed"})
            continue

        mask, _thr = bright_mask_for_block(
            img,
            v_percentile=v_percentile,
            s_percentile=s_percentile,
            morph_kernel_size=morph_kernel_size,
            min_region_area=min_region_area,
        )

        fname = os.path.basename(src_path)
        dst_path = os.path.join(output_dir, fname)
        ok = cv2.imwrite(dst_path, mask)
        results.append({"src": src_path, "dst": dst_path, "ok": bool(ok)})

    return results


# 模块导出（便于在 app.py 中调用）
__all__ = [
    "bright_mask_for_block",
    "process_light_color_blocks_directory",
]