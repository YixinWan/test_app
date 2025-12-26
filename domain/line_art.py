from typing import Any

import numpy as np
from skimage.color import rgb2gray
from skimage.filters import sobel, gaussian
from skimage.morphology import dilation, footprint_rectangle


def line_art(
    image: np.ndarray,
    *,
    edge_sigma: float = 0.2,
    dilation_size: int = 2,
) -> np.ndarray:
    """将彩色图像生成线稿强度图（0-1）。

    处理步骤：
    1) 转灰度 rgb2gray
    2) Sobel 梯度
    3) 高斯平滑（sigma=edge_sigma）
    4) 形态学膨胀（footprint_rectangle((dilation_size, dilation_size))）
    5) 归一化到 [0,1]

    参数:
    - image: HxWx3 RGB 图像 (uint8/float 均可)
    - edge_sigma: 高斯平滑的 sigma
    - dilation_size: 膨胀核大小（square(size)）

    返回:
    - edge_strength: HxW 浮点数组，范围 [0,1]
    """
    gray = rgb2gray(image)
    grad = sobel(gray)
    
    # 1. 适度平滑，压制高频噪点（保守一些，避免抹掉细轮廓）
    grad = gaussian(grad, sigma=float(edge_sigma))
    
    # 2. 归一化
    grad = grad / max(float(grad.max()), 1e-8)
    
    # 3. 软阈值抬升：去掉弱噪点，同时保留细线
    noise_thresh = 0.07
    grad = np.clip((grad - noise_thresh) / (1.0 - noise_thresh), 0, 1)
    
    # 4. 轻度对比增强：让线条更黑但不过度压暗
    contrast_gain = 2.4
    grad = np.clip(grad * contrast_gain, 0, 1)

    # 5. 先轻度膨胀，再做一次加权抬升以连粗主线
    grad = dilation(grad, footprint_rectangle((int(dilation_size), int(dilation_size))))

    # 6. 对膨胀后的结果再做一次平滑抬升，压低孤立小块
    #    连续区域被膨胀后数值更高，乘以 gain 后更黑；孤立噪点因面积小被扩散稀释，变得更淡
    contrast_gain_2 = 1.4
    grad = np.clip(grad * contrast_gain_2, 0, 1)
    
    # 7. 最终取反：黑白反转（让原本黑线变白，白底变黑，或反之）
    return grad


__all__ = ["line_art"]