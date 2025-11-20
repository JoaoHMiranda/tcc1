"""Helpers to build the pixel dataset used by band-selection methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ...config import BandSelectionConfig


@dataclass
class PixelSampleDataset:
    """Hold sampled pixels (ROI vs fundo) and metadata."""

    X: np.ndarray
    y: np.ndarray
    kept_band_indices: Sequence[int]
    orig_band_indices: Sequence[int]
    wavelengths: Sequence[float]

    def feature_positions(self) -> Dict[int, int]:
        return {band_idx: pos for pos, band_idx in enumerate(self.kept_band_indices)}

    def subset(self, kept_subset: Sequence[int]) -> "PixelSampleDataset":
        positions = [self.feature_positions().get(idx) for idx in kept_subset]
        positions = [p for p in positions if p is not None]
        if not positions:
            raise ValueError("Nenhuma banda solicitada existe no dataset.")
        X_sub = self.X[:, positions]
        kept_sub = [self.kept_band_indices[p] for p in positions]
        orig_sub = [self.orig_band_indices[p] for p in positions]
        wavs_sub = [self.wavelengths[p] for p in positions]
        return PixelSampleDataset(
            X=X_sub,
            y=self.y,
            kept_band_indices=kept_sub,
            orig_band_indices=orig_sub,
            wavelengths=wavs_sub,
        )


def build_circular_masks(
    shape: Tuple[int, int],
    cx: float,
    cy: float,
    radius: float,
    cfg: BandSelectionConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    H, W = shape
    y_axis, x_axis = np.ogrid[:H, :W]
    dist_sq = (x_axis - float(cx)) ** 2 + (y_axis - float(cy)) ** 2
    base_radius = max(radius, 1.0)
    roi_radius = max(base_radius * max(cfg.roi_radius_scale, 1e-3), 4.0)
    roi_mask = dist_sq <= roi_radius**2
    inner = max(base_radius * max(cfg.inner_background_scale, 1.01), roi_radius + 1.0)
    outer = max(base_radius * max(cfg.outer_background_scale, inner + 0.1), inner + 1.0)
    bg_mask = np.logical_and(dist_sq >= inner**2, dist_sq <= outer**2)
    if not np.any(bg_mask):
        bg_mask = np.logical_not(roi_mask)
    return roi_mask, bg_mask


def sample_coords(mask: np.ndarray, max_pixels: int, rng: np.random.Generator) -> Optional[np.ndarray]:
    coords = np.column_stack(np.nonzero(mask))
    if coords.size == 0:
        return None
    if coords.shape[0] <= max_pixels:
        return coords
    choice = rng.choice(coords.shape[0], size=max_pixels, replace=False)
    return coords[choice]


def gather_samples(bands: Sequence[np.ndarray], coords: np.ndarray) -> np.ndarray:
    rows = coords.shape[0]
    cols = len(bands)
    data = np.empty((rows, cols), dtype=np.float32)
    rr = coords[:, 0]
    cc = coords[:, 1]
    for idx, band in enumerate(bands):
        data[:, idx] = np.asarray(band, dtype=np.float32)[rr, cc]
    return data


def prepare_pixel_dataset(
    bands: Sequence[np.ndarray],
    kept_band_indices: Sequence[int],
    orig_band_indices: Sequence[int],
    wavelengths: Sequence[float],
    cx: float,
    cy: float,
    radius: float,
    cfg: BandSelectionConfig,
) -> Tuple[Optional[PixelSampleDataset], Optional[str]]:
    rng = np.random.default_rng(cfg.random_state)
    roi_mask, bg_mask = build_circular_masks(bands[0].shape, cx, cy, radius, cfg)
    roi_coords = sample_coords(roi_mask, cfg.sample_pixels_per_class, rng)
    bg_coords = sample_coords(bg_mask, cfg.sample_pixels_per_class, rng)
    if roi_coords is None or roi_coords.shape[0] < cfg.min_pixels_per_class:
        return None, "Quantidade insuficiente de pixels na ROI para seleção de bandas."
    if bg_coords is None or bg_coords.shape[0] < cfg.min_pixels_per_class:
        return None, "Quantidade insuficiente de pixels no fundo para seleção de bandas."
    roi_data = gather_samples(bands, roi_coords)
    bg_data = gather_samples(bands, bg_coords)
    X = np.vstack([roi_data, bg_data])
    y = np.concatenate(
        [
            np.ones(roi_data.shape[0], dtype=np.uint8),
            np.zeros(bg_data.shape[0], dtype=np.uint8),
        ]
    )
    dataset = PixelSampleDataset(
        X=X,
        y=y,
        kept_band_indices=list(kept_band_indices),
        orig_band_indices=list(orig_band_indices),
        wavelengths=list(wavelengths),
    )
    return dataset, None
