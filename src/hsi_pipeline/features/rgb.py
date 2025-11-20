"""Utilities to scale and save pseudo-RGB combinations."""

from __future__ import annotations

import numpy as np
from PIL import Image


def scale_robust(img2d: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    p1, p99 = np.percentile(img2d, [p_low, p_high])
    if p99 - p1 < 1e-12:
        p1, p99 = float(np.min(img2d)), float(np.max(img2d))
    band = img2d
    if p99 > p1:
        band = (band - p1) / (p99 - p1)
    else:
        band = band - band.min()
        denom = band.max() - band.min()
        band = band / denom if denom > 1e-12 else band
    return np.clip(band, 0.0, 1.0)


def save_rgb_from_channels(ch_r: np.ndarray, ch_g: np.ndarray, ch_b: np.ndarray, out_path: str):
    r = (np.clip(ch_r, 0, 1) * 255.0).astype(np.uint8)
    g = (np.clip(ch_g, 0, 1) * 255.0).astype(np.uint8)
    b = (np.clip(ch_b, 0, 1) * 255.0).astype(np.uint8)
    Image.fromarray(np.dstack([r, g, b])).save(out_path, format="PNG")
