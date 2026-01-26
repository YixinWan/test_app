from .color_mixing import rgb_to_cmy, cmy_to_rgb, suggest_mix, generate_steps_from_mix
from .segmentation import smooth_image, slic_color_blocks, coarse_color_blocks
from .gray_fade import generate_gray_fade_sequence

__all__ = [
    "rgb_to_cmy",
    "cmy_to_rgb",
    "suggest_mix",
    "generate_steps_from_mix",
    "smooth_image",
    "slic_color_blocks",
    "coarse_color_blocks",
    "generate_gray_fade_sequence",
]
