"""Orchestration helpers for running the selection stage standalone."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

from ..config import PipelineConfig
from ..data.envi_io import discover_set, open_envi_memmap
from ..data.paths import resolve_out_base
from ..pipeline.progress import PipelineProgress
from ..processing.reflectance import ReflectanceBandProvider
from .selection import run_band_selection_pipeline


def prepare_reflectance_stack(
    config: PipelineConfig, folder: str
) -> Tuple[List[np.ndarray], List[int], List[float], Tuple[int, int]]:
    hdr_main, raw_main, hdr_dark, raw_dark, hdr_white, raw_white = discover_set(folder)
    mm_main, meta_main = open_envi_memmap(hdr_main, raw_main)
    mm_dark, meta_dark = open_envi_memmap(hdr_dark, raw_dark)
    mm_white, meta_white = open_envi_memmap(hdr_white, raw_white)
    H, W, B = meta_main["lines"], meta_main["samples"], meta_main["bands"]
    wavs = meta_main.get("wavelengths")
    if wavs is None or len(wavs) != B:
        wavs = np.linspace(400.0, 800.0, B).tolist()

    trim_left = max(0, int(config.trimming.left))
    trim_right = max(0, int(config.trimming.right))
    start = trim_left
    end = max(start, B - trim_right - 1)
    if end < start:
        raise ValueError("Recorte espectral inválido.")
    kept = list(range(start, end + 1))
    wavs_kept = [wavs[i] for i in kept]

    provider = ReflectanceBandProvider(
        mm_main,
        meta_main,
        mm_dark,
        meta_dark,
        mm_white,
        meta_white,
        H,
        W,
        cache_size=config.cache_size_bands,
    )

    reflectance_cache: List[np.ndarray] = []
    for band in kept:
        reflectance_cache.append(provider.get(band))
    return reflectance_cache, kept, wavs_kept, (H, W)


def run_selection_for_dataset(config: PipelineConfig, dataset_folder: Path, progress=None) -> str:
    folder = str(dataset_folder)
    out_base, _, base = resolve_out_base(folder, config.out_root)
    reflectance_cache, kept, wavs_kept, shape = prepare_reflectance_stack(config, folder)
    H, W = shape
    radius = min(H, W) * 0.33
    cx, cy = W / 2.0, H / 2.0
    selection_dir = os.path.join(out_base, "selecao")
    os.makedirs(selection_dir, exist_ok=True)
    run_band_selection_pipeline(
        config=config.band_selection,
        bands=reflectance_cache,
        kept_band_indices=list(range(len(kept))),
        orig_band_indices=kept,
        wavelengths=wavs_kept,
        center=(cx, cy),
        radius=radius,
        out_dir=selection_dir,
        dataset_name=base,
        progress=progress,
    )
    return selection_dir


def run_selection(config: PipelineConfig, dataset_paths: Sequence[Path]) -> None:
    if not getattr(config, "enabled", True):
        print("[skip] Seleção inteira desativada (config.enabled=false).")
        return
    if not config.band_selection.enabled:
        print("[skip] band_selection.enabled=false → nenhuma seleção será executada.")
        return
    datasets = list(dataset_paths)
    if not datasets:
        print("[warn] Nenhum dataset para executar a seleção.")
        return
    with PipelineProgress() as progress:
        progress.create_task(
            "selection",
            f"[cyan]Selecionando bandas em {len(datasets)} conjunto(s)",
            len(datasets),
        )
        for idx, dataset in enumerate(datasets, start=1):
            progress.start_dataset(dataset.name, idx, len(datasets))
            progress.log(f"Iniciando seleção em {dataset}", style="cyan")
            run_selection_for_dataset(config, dataset, progress=progress)
            progress.log(f"[ok] Finalizado: {dataset}", style="green")
            progress.advance("selection")


__all__ = [
    "run_selection",
    "run_selection_for_dataset",
]
