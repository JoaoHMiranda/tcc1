"""Dataset resource loading for preprocessing."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from ..config import PipelineConfig
from ..data.envi_io import discover_set, open_envi_memmap
from ..data.paths import resolve_out_base
from ..processing.reflectance import ReflectanceBandProvider


@dataclass
class DatasetResources:
    provider: ReflectanceBandProvider
    folder: str
    out_base: str
    out_root: str
    base_name: str
    height: int
    width: int
    total_bands: int
    wavelengths: List[float]


def resolve_wavelengths(meta_main, total_bands: int) -> List[float]:
    wavs = meta_main.get("wavelengths")
    if wavs is None or len(wavs) != total_bands:
        return np.linspace(400.0, 800.0, total_bands).tolist()
    if isinstance(wavs, np.ndarray):
        return wavs.astype(float).tolist()
    if isinstance(wavs, Sequence):
        return [float(v) for v in wavs]
    return np.linspace(400.0, 800.0, total_bands).tolist()


def load_dataset_resources(config: PipelineConfig) -> Tuple[DatasetResources, float]:
    folder = os.path.abspath(config.folder)
    out_base, out_root, base = resolve_out_base(folder, config.out_root)
    stage_start = time.perf_counter()
    hdr_main, raw_main, hdr_dark, raw_dark, hdr_white, raw_white = discover_set(folder)
    mm_main, meta_main = open_envi_memmap(hdr_main, raw_main)
    mm_dark, meta_dark = open_envi_memmap(hdr_dark, raw_dark)
    mm_white, meta_white = open_envi_memmap(hdr_white, raw_white)
    provider = ReflectanceBandProvider(
        mm_main,
        meta_main,
        mm_dark,
        meta_dark,
        mm_white,
        meta_white,
        meta_main["lines"],
        meta_main["samples"],
        cache_size=config.cache_size_bands,
    )
    elapsed = time.perf_counter() - stage_start
    resources = DatasetResources(
        provider=provider,
        folder=folder,
        out_base=out_base,
        out_root=out_root,
        base_name=base,
        height=meta_main["lines"],
        width=meta_main["samples"],
        total_bands=meta_main["bands"],
        wavelengths=resolve_wavelengths(meta_main, meta_main["bands"]),
    )
    return resources, elapsed


__all__ = ["DatasetResources", "load_dataset_resources"]
