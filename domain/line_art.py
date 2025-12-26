import numpy as np
import cv2


def line_art(image: np.ndarray, k_size: int = 41) -> np.ndarray:
    """根据“灰度化 -> 反相 -> 高斯模糊 -> 颜色减淡（Color Dodge） -> 对比度拉升”生成素描风格线稿。

    输入:
    - image: HxW 或 HxWx3 图像。
        • 若为 uint8，取值应为 [0,255]
        • 若为 float/其他类型，会先裁剪到 [0,1] 再映射到 uint8

    参数:
    - k_size: 高斯模糊核大小（建议使用奇数，>=3）。会自动纠正为 >=3 的奇数。

    返回:
    - sketch_raw: HxW uint8 灰度图，白底黑线（适合直接保存/展示），并做了适度对比度拉升让黑更黑、白更白。
    """
    # 1) 灰度化（确保输入为 uint8）
    if image.dtype != np.uint8:
        img_u8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    else:
        img_u8 = image

    if img_u8.ndim == 3 and img_u8.shape[2] == 3:
        # 约定外部传入为 RGB
        gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)
    elif img_u8.ndim == 2:
        gray = img_u8
    else:
        raise ValueError("image must be HxW or HxWx3")

    # 2) 反相
    inv_gray = 255 - gray

    # 3) 高斯模糊（核大小转为 >=3 的奇数）
    k = int(k_size)
    if k < 3:
        k = 3
    if k % 2 == 0:
        k += 1
    blurred = cv2.GaussianBlur(inv_gray, (k, k), 0)

    # 4) 颜色减淡（Color Dodge）
    # 使用 denom = 255 - blurred，OpenCV 的 divide 会做饱和处理
    denom = cv2.subtract(np.full_like(blurred, 255), blurred)
    sketch_raw = cv2.divide(gray, denom, scale=256)

    # 5) 对比度拉升（百分位线性拉伸）：让黑更黑、白更白
    # 说明：对直方图的低/高百分位进行裁剪并线性拉伸到 [0,255]
    def _boost_contrast_u8(img_u8: np.ndarray, black_clip: float = 2.0, white_clip: float = 98.0) -> np.ndarray:
        if img_u8.dtype != np.uint8:
            raise ValueError("_boost_contrast_u8 expects uint8 image")
        # 计算低/高百分位阈值
        p_low = float(np.percentile(img_u8, black_clip))
        p_high = float(np.percentile(img_u8, white_clip))
        if p_high <= p_low:
            # 极端情况，返回原图
            return img_u8
        img_f = img_u8.astype(np.float32)
        img_f = np.clip(img_f, p_low, p_high)
        img_f = (img_f - p_low) / (p_high - p_low) * 255.0
        return img_f.astype(np.uint8)

    sketch_raw = _boost_contrast_u8(sketch_raw, black_clip=2.0, white_clip=98.0)

    return sketch_raw


__all__ = ["line_art"]