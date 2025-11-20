"""Final reporting stage for YOLO training."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

from ...config import YoloTrainingConfig
from ..artifacts import copy_training_reports, write_summary_json
from ..metrics import (
    copy_plots,
    extract_detection_metrics,
    write_training_metrics_csv,
    write_training_summary_txt,
)


def build_report_paths(dataset_reports: Dict[str, str]) -> List[Path]:
    return [
        Path(dataset_reports["dataset_index_csv"]),
        Path(dataset_reports["counts_csv"]),
        Path(dataset_reports["counts_txt"]),
    ]


def finalize_training_reports(
    config: YoloTrainingConfig,
    dataset_result,
    training_info: Dict[str, object],
    eval_info: Optional[Dict[str, object]],
    model_info: Optional[Dict[str, object]],
):
    dataset_dir = dataset_result.dataset_dir
    data_yaml = dataset_result.data_yaml
    counts = dataset_result.counts
    dataset_reports = dataset_result.dataset_reports

    reference_metrics = eval_info.get("metrics") if eval_info else training_info.get("metrics")
    final_metrics = extract_detection_metrics(reference_metrics)
    metrics_csv = write_training_metrics_csv(final_metrics, dataset_dir)
    training_txt = write_training_summary_txt(
        dataset_dir=dataset_dir,
        counts=counts,
        metrics=final_metrics,
        training_info=training_info,
        eval_metrics=extract_detection_metrics(eval_info.get("metrics")) if eval_info else None,
        data_yaml=data_yaml,
        model_info=model_info,
    )

    plots_root = dataset_dir / "plots"
    copied_plots = copy_plots([Path(p) for p in training_info.get("plots", [])], plots_root)
    if eval_info:
        copied_plots += copy_plots([Path(p) for p in eval_info.get("plots", [])], plots_root)

    summary = {
        "config": asdict(config),
        "dataset_dir": str(dataset_dir),
        "data_yaml": str(data_yaml),
        "counts": counts,
        "training": training_info,
        "evaluation": eval_info or {},
        "dataset_index_csv": dataset_reports["dataset_index_csv"],
        "counts_csv": dataset_reports["counts_csv"],
        "counts_txt": dataset_reports["counts_txt"],
        "training_metrics_csv": str(metrics_csv),
        "training_summary_txt": str(training_txt),
        "model_artifacts": model_info,
        "plots": [str(p) for p in copied_plots],
    }
    summary_json = write_summary_json(dataset_dir, summary)
    report_files = build_report_paths(dataset_reports) + [
        Path(metrics_csv),
        Path(training_txt),
        summary_json,
        data_yaml,
    ]
    if model_info and model_info.get("model_dir"):
        copied_reports = copy_training_reports(Path(model_info["model_dir"]), report_files + copied_plots)
        model_info["reports"] = [str(path) for path in copied_reports]
        summary["model_artifacts"] = model_info
    return summary


__all__ = ["finalize_training_reports"]
