import os
import uuid
from typing import List, Optional

import numpy as np
import cv2


def generate_gray_fade_sequence(
        image: np.ndarray,
        steps: int = 8,
        output_root: Optional[str] = None,
        prefix: Optional[str] = None,
) -> List[str]:
        """Generate a sequence of images that gradually fade the original image
        directly to black (no grayscale desaturation), saving them under static/gray_fade.

        Inputs/contract:
        - image: HxWx3 RGB numpy array (uint8 in [0,255] preferred). If not uint8, it will be clipped and converted.
        - steps: total number of frames (>= 2). The sequence covers: color -> black progressively.
        - output_root: optional absolute directory to save frames. Defaults to <repo>/test_app/static/gray_fade.
        - prefix: deprecated; kept for backward compatibility but ignored. Frames are saved directly under output_root.

        Returns:
        - List of relative paths from test_app root, like ["static/gray_fade/0.png", ...]

        Notes:
        - This function assumes the project structure where this module lives in test_app/domain.
            It resolves the static directory relative to this file if output_root is not provided.
        """
        if image is None:
            raise ValueError("image is None")

        if steps is None or steps < 2:
            steps = 2

        # Ensure uint8 RGB
        if image.dtype != np.uint8:
            img_u8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        else:
            img_u8 = image

        if img_u8.ndim != 3 or img_u8.shape[2] != 3:
            raise ValueError("image must be HxWx3 RGB")

        # Work in float32 for fading
        img_f = img_u8.astype(np.float32)

        # Resolve key paths
        domain_dir = os.path.dirname(os.path.abspath(__file__))
        app_root = os.path.dirname(domain_dir)  # test_app
        # Resolve output root: <repo>/test_app/static/gray_fade
        if output_root is None:
            output_root = os.path.join(app_root, "static", "gray_fade")

        os.makedirs(output_root, exist_ok=True)

        # Save frames directly under output_root (no per-invocation prefix folder)
        out_dir = output_root
        os.makedirs(out_dir, exist_ok=True)

        rel_outputs: List[str] = []

        # For step i in [0, steps-1], s in [0,1]
        # Direct fade: original color -> black (brightness scales 1->0)
        for i in range(steps):
            s = 0.0 if steps == 1 else i / float(steps - 1)
            # Brightness linearly drops from 1 to 0
            bright = 1.0 - s
            out_f = np.clip(img_f * bright, 0.0, 255.0)
            out_u8 = out_f.astype(np.uint8)

            # Save as PNG, keep RGB to be consistent with project convention (OpenCV writes expects BGR)
            # Naming requirement: original is 0.png, then 1.png ... increasing gray to black
            filename = f"{i}.png"
            abs_path = os.path.join(out_dir, filename)
            bgr = cv2.cvtColor(out_u8, cv2.COLOR_RGB2BGR)
            cv2.imwrite(abs_path, bgr)

            # Relative path from test_app root: static/gray_fade/...
            rel_path = os.path.relpath(abs_path, start=app_root)
            # Normalize to posix style
            rel_path = rel_path.replace("\\", "/")
            rel_outputs.append(rel_path)

        return rel_outputs


__all__ = ["generate_gray_fade_sequence"]
