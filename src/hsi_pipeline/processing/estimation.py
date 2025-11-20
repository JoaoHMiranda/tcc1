"""Global center estimation helpers."""

from __future__ import annotations

import os
from typing import Tuple

import cv2
import numpy as np

from ..config import MedianGuessSettings
from ..features.segmentation import largest_component_mask


def estimate_global_from_median(
    msc_dir: str,
    height: int,
    width: int,
    median_cfg: MedianGuessSettings,
) -> Tuple[float, float, float]:
    files = sorted(
        [os.path.join(msc_dir, f) for f in os.listdir(msc_dir) if f.lower().endswith(".png")]
    )
    if not files:
        return (np.nan, np.nan, np.nan)
    step = max(1, median_cfg.sample_step)
    sample = files[::step][: median_cfg.max_samples]
    stack = []
    for path in sample:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        stack.append(gray)
    if not stack:
        return (np.nan, np.nan, np.nan)
    med = np.median(np.stack(stack, axis=0), axis=0).astype(np.uint8)
    med = cv2.GaussianBlur(
        med, (median_cfg.kernel_size, median_cfg.kernel_size), median_cfg.blur_sigma
    )
    _, circle = largest_component_mask(med, min_area_frac=median_cfg.min_area_fraction)
    if circle is not None:
        cx, cy, r = circle
        return float(cx), float(cy), float(r)

    base_len = min(height, width)
    fallback_radius = base_len * 0.33
    return (float(width) / 2.0, float(height) / 2.0, fallback_radius)
