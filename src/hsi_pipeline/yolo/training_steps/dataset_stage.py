"""Dataset preparation stage for YOLO training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from ...config import YoloTrainingConfig
from ...pipeline.progress import PipelineProgress
from ...tools.restore_doentes_labels import restore_doentes_labels
from .. import dataset
from ..fs_utils import next_available_path


@dataclass
class DatasetStageResult:
    dataset_dir: Path
    data_yaml: Path
    counts: Dict[str, int]
    dataset_reports: Dict[str, str]
    training_root: Path


def log_progress(progress: PipelineProgress | None, message: str, style: str = "green") -> None:
    if progress:
        progress.log(message, style=style)


def prepare_dataset_stage(config: YoloTrainingConfig, progress: PipelineProgress | None) -> DatasetStageResult:
    if not config.out_root:
        raise ValueError("Defina 'out_root' no JSON ou via --output-root para localizar os pseudo-RGBs.")
    out_root = dataset.resolve_path(config.out_root)
    if out_root is None:
        raise ValueError("out_root inválido após resolução.")
    training_root = dataset.resolve_path(config.training_root) or out_root

    log_progress(progress, "Descobrindo datasets", style="cyan")
    datasets = dataset.discover_datasets(out_root, config)
    log_progress(progress, f"{len(datasets)} dataset(s) encontrados.", style="cyan")

    # As amostras de "doentes" não trazem labels por padrão.
    if out_root.name.lower() == "doentes":
        created, _ = restore_doentes_labels(out_root, active_samples=[d.name for d in datasets])
        if created:
            log_progress(progress, f"{len(created)} label(s) padrão recriadas automaticamente.", style="yellow")

    log_progress(progress, "Coletando pseudo-RGBs", style="cyan")
    records = dataset.collect_records(out_root, datasets, config)
    log_progress(progress, f"{len(records)} imagens com rótulos coletadas.", style="cyan")

    records_split = dataset.split_records(records, config)
    dataset_dir = next_available_path(training_root / config.dataset_output_dir)
    total_records = sum(len(items) for items in records_split.values())
    if progress:
        progress.create_task(
            "dataset",
            "[green]Preparando dataset YOLO",
            total=max(total_records, 1),
        )
    counts = dataset.materialize_dataset(
        records_split=records_split,
        dataset_dir=dataset_dir,
        clean=config.clean_output,
        label_extension=config.label_extension,
        enhance_images=config.enhance_pseudo_rgb,
        progress=progress,
        task_id="dataset",
    )
    if progress:
        progress.complete("dataset")

    data_yaml = dataset.write_data_yaml(dataset_dir, list(config.classes))
    dataset_reports = dataset.write_dataset_reports(records_split, counts, dataset_dir)
    return DatasetStageResult(
        dataset_dir=dataset_dir,
        data_yaml=data_yaml,
        counts=counts,
        dataset_reports=dataset_reports,
        training_root=training_root,
    )


__all__ = ["DatasetStageResult", "prepare_dataset_stage"]
