"""Helpers for trimming and keeping spectral bands."""

from __future__ import annotations

from typing import List, Tuple

from .resources import DatasetResources


def prepare_kept_bands(resources: DatasetResources, trimming) -> Tuple[List[int], List[float]]:
    trim_left = max(0, int(trimming.left))
    trim_right = max(0, int(trimming.right))
    start = trim_left
    end = max(start, resources.total_bands - trim_right - 1)
    if end < start:
        raise ValueError("Recorte espectral inválido (fim antes do início).")
    kept = list(range(start, end + 1))
    wavs_kept = [resources.wavelengths[i] for i in kept]
    return kept, wavs_kept


__all__ = ["prepare_kept_bands"]
