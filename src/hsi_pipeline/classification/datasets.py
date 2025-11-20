"""Dataset discovery utilities for the classification step."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List


def discover_samples(source_root: Path) -> Dict[str, List[Path]]:
    """Return mapping sample_name → list of PCA PNGs."""

    samples: Dict[str, List[Path]] = {}
    for sample_dir in sorted(source_root.glob("*")):
        if not sample_dir.is_dir():
            continue
        files = sorted(sample_dir.glob("pseudo_rgb/pca/*.png"))
        if files:
            samples[sample_dir.name] = files
    if not samples:
        raise FileNotFoundError(f"Nenhum PNG encontrado em {source_root}/*/pseudo_rgb/pca/")
    return samples


__all__ = ["discover_samples"]
