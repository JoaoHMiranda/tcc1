"""MSC stages extracted from the original pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING

import numpy as np

from ...features.rgb import save_rgb_from_channels, scale_robust
from ...processing.reflectance import ReflectanceBandProvider
from ..utils import advance_progress, make_band_filename

if TYPE_CHECKING:
    from ...config import VariantOutputSettings
    from ...pipeline.progress import PipelineProgress


@dataclass
class MSCResult:
    snv_msc_cache: List[np.ndarray]
    a_map_snv: np.ndarray
    b_map_snv: np.ndarray
    images_snv_msc: int


def run_reflectance_msc_stage(
    provider: ReflectanceBandProvider,
    kept: List[int],
    wavs_kept: List[float],
    delta: int,
    variant_dir: Optional[str],
    variant_setting: Optional["VariantOutputSettings"],
    a_map_reflectance: np.ndarray,
    b_map_reflectance: np.ndarray,
    progress: Optional["PipelineProgress"] = None,
) -> int:
    """Generates MSC variants based on raw reflectance bands."""

    variant_enabled = bool(variant_dir and variant_setting and variant_setting.plot)
    if not variant_enabled:
        return 0

    images_generated = 0
    Bk = len(kept)
    for idx in range(Bk):
        left_band = kept[max(0, idx - delta)]
        right_band = kept[min(Bk - 1, idx + delta)]
        save_rgb_from_channels(
            scale_robust((provider.get(left_band) - a_map_reflectance) / b_map_reflectance),
            scale_robust((provider.get(kept[idx]) - a_map_reflectance) / b_map_reflectance),
            scale_robust((provider.get(right_band) - a_map_reflectance) / b_map_reflectance),
            os.path.join(variant_dir, make_band_filename(idx, wavs_kept[idx])),
        )
        advance_progress(progress, "variants")
        images_generated += 1
    return images_generated


def run_snv_msc_stage(
    snv_cache: List[np.ndarray],
    snv_band_sums: np.ndarray,
    kept: List[int],
    wavs_kept: List[float],
    delta: int,
    variant_dir: Optional[str],
    variant_setting: Optional["VariantOutputSettings"],
    progress: Optional["PipelineProgress"] = None,
) -> MSCResult:
    """Applies MSC on top of SNV-normalized bands."""

    if not snv_cache:
        return MSCResult([], np.array([]), np.array([]), 0)

    H, W = snv_cache[0].shape
    Bk = len(snv_cache)
    ref_spec_snv = snv_band_sums / (H * W)
    ref_mean_snv = float(ref_spec_snv.mean())
    ref_centered_snv = ref_spec_snv - ref_mean_snv
    denom_snv = float(np.sum(ref_centered_snv * ref_centered_snv)) or 1e-12
    Dacc_SNV = np.zeros((H, W), dtype=np.float64)
    for idx in range(Bk):
        Dacc_SNV += snv_cache[idx] * ref_centered_snv[idx]
    b_map_SNV = Dacc_SNV / denom_snv
    b_map_SNV[np.abs(b_map_SNV) < 1e-6] = 1e-6
    a_map_SNV = -b_map_SNV * ref_mean_snv

    snv_msc_cache: List[np.ndarray] = []
    images_generated = 0
    variant_enabled = bool(variant_dir and variant_setting and variant_setting.plot)

    for idx in range(Bk):
        snv_msc = (snv_cache[idx] - a_map_SNV) / b_map_SNV
        snv_msc_cache.append(snv_msc)
        if variant_enabled:
            left_idx = max(0, idx - delta)
            right_idx = min(Bk - 1, idx + delta)
            left = snv_msc_cache[left_idx]
            if right_idx < len(snv_msc_cache):
                right = snv_msc_cache[right_idx]
            else:
                right = (snv_cache[right_idx] - a_map_SNV) / b_map_SNV
            save_rgb_from_channels(
                scale_robust(left),
                scale_robust(snv_msc),
                scale_robust(right),
                os.path.join(variant_dir, make_band_filename(idx, wavs_kept[idx])),
            )
            advance_progress(progress, "variants")
            images_generated += 1

    return MSCResult(snv_msc_cache, a_map_SNV, b_map_SNV, images_generated)
