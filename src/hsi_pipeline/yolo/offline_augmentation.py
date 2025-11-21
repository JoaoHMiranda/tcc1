"""Offline pseudo-RGB augmentation helpers for YOLO training."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Sequence

from hsi_pipeline.config import OfflineAugmentationConfig, YoloTrainingConfig
from hsi_pipeline.tools.generate_pca_augmented import augment_sample


def _count_images(datasets: Sequence[Path], pseudo_root: Path, method: str) -> int:
    total = 0
    for dataset_dir in datasets:
        method_dir = dataset_dir / pseudo_root / method
        if method_dir.exists():
            total += sum(1 for _ in method_dir.rglob("*.png"))
    return total


def _count_samples(datasets: Sequence[Path]) -> int:
    total = 0
    for dataset_dir in datasets:
        # Se já estamos no nível das amostras (contém pseudo_rgb diretamente), conte 1.
        if (dataset_dir / "pseudo_rgb").exists():
            total += 1
            continue
        # Caso contrário, conte subpastas (nível dataset -> amostras).
        total += sum(1 for sample_dir in dataset_dir.iterdir() if sample_dir.is_dir())
    return total


def _iter_sample_dirs(datasets: Sequence[Path]) -> Sequence[Path]:
    """Yield diretórios de amostras, suportando 'datasets' no nível dataset ou amostra."""
    for dataset_dir in datasets:
        if (dataset_dir / "pseudo_rgb").exists():
            # Já é uma amostra
            yield dataset_dir
            continue
        for sample_dir in sorted(d for d in dataset_dir.iterdir() if d.is_dir()):
            yield sample_dir


def run_offline_augmentations(
    config: YoloTrainingConfig,
    datasets: Sequence[Path],
    progress=None,
) -> Dict[str, int]:
    aug_cfg: OfflineAugmentationConfig = getattr(config, "offline_augmentation", None)
    if not aug_cfg or not aug_cfg.enabled:
        return {"generated": 0, "per_sample": 0, "samples": 0, "base_count": 0}

    pseudo_root = Path(config.pseudo_root)
    source_method = aug_cfg.source_method or config.pseudo_method
    output_method = aug_cfg.output_method or config.pseudo_method
    base_count = _count_images(datasets, pseudo_root, source_method)
    n_samples = max(_count_samples(datasets), 1)

    per_sample = max(1, aug_cfg.per_sample)
    if aug_cfg.target_total_images:
        needed = max(0, aug_cfg.target_total_images - base_count)
        if needed > 0:
            per_sample = max(per_sample, math.ceil(needed / n_samples))

    if progress:
        progress.log("Gerando imagens sintéticas offline...", style="yellow")

    total_generated = 0
    samples_used = 0
    for sample_dir in _iter_sample_dirs(datasets):
        n = augment_sample(sample_dir, output_method, per_sample, source_method)
        if n > 0:
            total_generated += n
            samples_used += 1

    if progress and total_generated:
        progress.log(
            f"Imagens geradas: {total_generated} ({per_sample} por amostra) em {samples_used} amostras.",
            style="green",
        )
    elif progress and not total_generated:
        progress.log("Nenhuma imagem sintética gerada (não foram encontradas bases com labels).", style="yellow")

    return {
        "generated": total_generated,
        "per_sample": per_sample,
        "samples": samples_used,
        "base_count": base_count,
    }


__all__ = ["run_offline_augmentations"]
