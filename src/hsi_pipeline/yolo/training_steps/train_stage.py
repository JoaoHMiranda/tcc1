"""Training stage that wraps Ultralytics YOLO training."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from ultralytics import YOLO

from ...config import YoloTrainingConfig
from ..metrics import collect_plot_files


def train_yolo_model(
    config: YoloTrainingConfig,
    data_yaml: Path,
    runs_dir: Path,
) -> Dict[str, object]:
    model = YOLO(config.model)
    train_args = {
        "data": str(data_yaml),
        "epochs": config.epochs,
        "imgsz": config.imgsz,
        "batch": config.batch,
        "patience": config.patience,
        "project": str(runs_dir),
        "name": config.experiment_name,
        "exist_ok": True,
        "amp": config.amp,
    }
    if config.device:
        train_args["device"] = config.device
    if config.augmentations:
        train_args.update(
            {
                "fliplr": config.augmentations.fliplr,
                "flipud": config.augmentations.flipud,
                "degrees": config.augmentations.degrees,
            }
        )
    train_args.update(config.train_extra_args or {})
    results = model.train(**train_args)
    return {
        "save_dir": str(results.save_dir),
        "metrics": getattr(results, "metrics", {}),
        "plots": [str(p) for p in collect_plot_files(Path(results.save_dir))],
        "train_args": train_args,
    }


__all__ = ["train_yolo_model"]
