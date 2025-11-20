"""SNV computations split away from the monolithic pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING

import numpy as np

from ...features.rgb import save_rgb_from_channels, scale_robust
from ..utils import advance_progress, make_band_filename

if TYPE_CHECKING:
    from ...config import VariantOutputSettings
    from ...pipeline.progress import PipelineProgress


@dataclass
class SNVResult:
    snv_cache: List[np.ndarray]
    snv_band_sums: np.ndarray
    a_map_reflectance: np.ndarray
    b_map_reflectance: np.ndarray
    images_generated: int


def run_snv_stage(
    reflectance_cache: List[np.ndarray],
    mean_px: np.ndarray,
    std_px: np.ndarray,
    band_sums: np.ndarray,
    kept: List[int],
    wavs_kept: List[float],
    delta: int,
    variant_dir: Optional[str],
    variant_setting: Optional["VariantOutputSettings"],
    progress: Optional["PipelineProgress"] = None,
) -> SNVResult:
    """Runs the SNV normalization and optional visualization export."""

    if not reflectance_cache:
        return SNVResult([], np.zeros(0), mean_px, np.ones_like(mean_px), 0)

    H, W = mean_px.shape
    Bk = len(reflectance_cache)
    ref_spec = band_sums / (H * W)
    ref_mean = float(ref_spec.mean())
    ref_centered = ref_spec - ref_mean
    denom_ref = float(np.sum(ref_centered * ref_centered)) or 1e-12

    Dacc_R = np.zeros((H, W), dtype=np.float64)
    snv_band_sums = np.zeros(Bk, dtype=np.float64)
    snv_cache: List[np.ndarray] = []
    images_generated = 0
    variant_enabled = bool(variant_dir and variant_setting and variant_setting.plot)

    for idx in range(Bk):
        Rb = reflectance_cache[idx]
        Dacc_R += Rb * ref_centered[idx]
        snv = (Rb - mean_px) / std_px
        snv_cache.append(snv)
        left_idx = max(0, idx - delta)
        right_idx = min(Bk - 1, idx + delta)
        left = snv_cache[left_idx]
        if right_idx < len(snv_cache):
            right = snv_cache[right_idx]
        else:
            right = (reflectance_cache[right_idx] - mean_px) / std_px
        if variant_enabled:
            save_rgb_from_channels(
                scale_robust(left),
                scale_robust(snv),
                scale_robust(right),
                os.path.join(variant_dir, make_band_filename(idx, wavs_kept[idx])),
            )
            advance_progress(progress, "variants")
            images_generated += 1
        snv_band_sums[idx] = float(snv.sum())

    b_map_R = Dacc_R / denom_ref
    b_map_R[np.abs(b_map_R) < 1e-6] = 1e-6
    a_map_R = mean_px - b_map_R * ref_mean
    return SNVResult(snv_cache, snv_band_sums, a_map_R, b_map_R, images_generated)
