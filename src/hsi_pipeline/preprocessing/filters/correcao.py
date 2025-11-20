"""Correção básica (reflectância) gerada como um estágio independente."""

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
class CorrecaoResult:
    reflectance_cache: List[np.ndarray]
    mean_px: np.ndarray
    std_px: np.ndarray
    band_sums: np.ndarray
    images_generated: int


def run_correcao_stage(
    provider: ReflectanceBandProvider,
    kept: List[int],
    wavs_kept: List[float],
    delta: int,
    variant_dir: Optional[str],
    variant_setting: Optional["VariantOutputSettings"],
    progress: Optional["PipelineProgress"] = None,
    step_task: Optional[str] = None,
) -> CorrecaoResult:
    """Computes reflectance bands and optionally saves RGB composites."""

    H, W = provider.height, provider.width
    Bk = len(kept)
    sum_px = np.zeros((H, W), dtype=np.float64)
    sum_sq_px = np.zeros((H, W), dtype=np.float64)
    band_sums = np.zeros(Bk, dtype=np.float64)
    reflectance_cache: List[np.ndarray] = []
    images_generated = 0
    variant_enabled = bool(variant_dir and variant_setting and variant_setting.plot)

    for idx, band in enumerate(kept):
        Rb = provider.get(band)
        reflectance_cache.append(Rb)
        if variant_enabled:
            left_band = kept[max(0, idx - delta)]
            right_band = kept[min(Bk - 1, idx + delta)]
            save_rgb_from_channels(
                scale_robust(provider.get(left_band)),
                scale_robust(Rb),
                scale_robust(provider.get(right_band)),
                os.path.join(variant_dir, make_band_filename(idx, wavs_kept[idx])),
            )
            advance_progress(progress, "variants")
            images_generated += 1
        if step_task:
            advance_progress(progress, step_task)
        sum_px += Rb
        sum_sq_px += Rb * Rb
        band_sums[idx] = float(Rb.sum())

    mean_px = sum_px / Bk
    std_px = np.sqrt(np.maximum(sum_sq_px / Bk - mean_px * mean_px, 1e-10))
    return CorrecaoResult(reflectance_cache, mean_px, std_px, band_sums, images_generated)
