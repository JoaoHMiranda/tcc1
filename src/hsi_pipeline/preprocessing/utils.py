"""Shared helpers used across preprocessing modules."""

from __future__ import annotations

import os
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..config.config import VariantOutputSettings
    from ..pipeline.progress import PipelineProgress


VARIANT_SUBDIRS = {
    "correcao_snv_msc": "correcao_snv_msc_rgb_bands",
}
VARIANT_ORDER = tuple(VARIANT_SUBDIRS.keys())


def prepare_variant_dirs(base_dir: str, toggles) -> Dict[str, Optional[str]]:
    dirs: Dict[str, Optional[str]] = {}
    for key, subdir in VARIANT_SUBDIRS.items():
        setting = getattr(toggles, key, None)
        enabled = bool(getattr(setting, "enabled", setting))
        if enabled:
            path = os.path.join(base_dir, subdir)
            os.makedirs(path, exist_ok=True)
            dirs[key] = path
        else:
            dirs[key] = None
    return dirs


def make_band_filename(idx: int, wavelength_nm: float) -> str:
    return f"band_{idx:03d}_{int(round(wavelength_nm))}nm.png"


def relative_paths_for_variant(
    enabled: bool, subdir: str, wavs_kept: List[float], prefix: Optional[str] = None
) -> List[str]:
    if not enabled:
        return ["" for _ in range(len(wavs_kept))]
    rel_dir = os.path.join(prefix, subdir) if prefix else subdir
    return [os.path.join(rel_dir, make_band_filename(i, wavs_kept[i])) for i in range(len(wavs_kept))]


def advance_progress(progress: Optional["PipelineProgress"], key: str, step: int = 1):
    if progress is not None:
        progress.advance(key, step)
