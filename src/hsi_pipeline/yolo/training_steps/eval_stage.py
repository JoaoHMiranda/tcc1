"""Evaluation stage for YOLO models."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from ultralytics import YOLO

from ...config import YoloTrainingConfig
from ..metrics import collect_plot_files


def evaluate_yolo_model(
    config: YoloTrainingConfig,
    data_yaml: Path,
    runs_dir: Path,
    weight_path: Path,
) -> Dict[str, object]:
    model = YOLO(weight_path)
    val_results = model.val(
        data=str(data_yaml),
        imgsz=config.imgsz,
        device=config.device or "cpu",
        split="val",
        project=str(runs_dir),
        name=f"{config.experiment_name}_eval",
        exist_ok=True,
    )
    return {
        "save_dir": str(val_results.save_dir),
        "metrics": getattr(val_results, "metrics", {}),
        "plots": [str(p) for p in collect_plot_files(Path(val_results.save_dir))],
    }


__all__ = ["evaluate_yolo_model"]
