from typing import Any

import numpy as np
from skimage.color import rgb2gray
from skimage.filters import sobel, gaussian
from skimage.morphology import dilation, square


def line_art(
    image: np.ndarray,
    *,
    edge_sigma: float = 1.0,
    dilation_size: int = 2,
) -> np.ndarray:
    """将彩色图像生成线稿强度图（0-1）。

    处理步骤：
    1) 转灰度 rgb2gray
    2) Sobel 梯度
    3) 高斯平滑（sigma=edge_sigma）
    4) 形态学膨胀（square(dilation_size)）
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
    grad = gaussian(grad, sigma=float(edge_sigma))
    grad = dilation(grad, square(int(dilation_size)))
    edge_strength = np.clip(grad / max(float(grad.max()), 1e-8), 0, 1)
    return edge_strength


__all__ = ["line_art"]