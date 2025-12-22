"""兼容层：旧的 functions.py 保留对外 API，但内部实现迁移至 domain 包。

目的是：
- 让现有代码（包括 app.py 或未来脚本）继续从 `functions` 导入；
- 同时将真实实现拆分到 `domain.color_mixing` / `domain.segmentation` 中，结构更清晰。
"""

from domain import (
    rgb_to_cmy,
    cmy_to_rgb,
    suggest_mix,
    generate_steps_from_mix,
    smooth_image,
    slic_color_blocks,
    coarse_color_blocks,
)

#-----------------------生成分步调色数据函数-------------------------------------------
__all__ = [
    "rgb_to_cmy",
    "cmy_to_rgb",
    "suggest_mix",
    "generate_steps_from_mix",
    "smooth_image",
    "slic_color_blocks",
    "coarse_color_blocks",
]

