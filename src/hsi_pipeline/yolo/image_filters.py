"""Image enhancement helpers for YOLO training inputs."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def apply_clahe_lab(bgr: np.ndarray, clip_limit: float = 2.0, tile_grid: int = 8) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def unsharp_mask(bgr: np.ndarray, sigma: float = 1.0, amount: float = 0.5) -> np.ndarray:
    if amount <= 0:
        return bgr
    blur = cv2.GaussianBlur(bgr, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return cv2.addWeighted(bgr, 1.0 + amount, blur, -amount, 0)


def enhance_pca_rgb(
    image_path: Path,
    output_path: Path,
    clip_limit: float = 2.0,
    tile_grid: int = 8,
    sharpen_sigma: float = 1.0,
    sharpen_amount: float = 0.5,
) -> Path:
    """Apply light corrections (CLAHE + unsharp mask) to PCA pseudo-RGB."""
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Não foi possível ler a imagem: {image_path}")
    corrected = apply_clahe_lab(bgr, clip_limit=clip_limit, tile_grid=tile_grid)
    corrected = unsharp_mask(corrected, sigma=sharpen_sigma, amount=sharpen_amount)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), corrected)
    return output_path


__all__ = ["enhance_pca_rgb"]
