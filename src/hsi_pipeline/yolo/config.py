"""Dataclasses and loaders for YOLO training and evaluation configs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


def _strip_values(data: Dict[str, Any]) -> Dict[str, Any]:
    cleaned: Dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, dict) and "value" in value:
            cleaned[key] = value["value"]
        else:
            cleaned[key] = value
    return cleaned


@dataclass
class YoloTrainConfig:
    model: str = "src/hsi_pipeline/yolo/modelos/yolo12x.pt"
    input_root: str = "hsi_modificado/doentes/"
    output_root: str = "yolo"
    classes: List[str] = field(default_factory=lambda: ["staphylococcus"])
    imgsz: int = 256
    epochs: int = 200
    patience: int = 50
    batch: int = 16
    device: str = "0"
    train_extra_args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class YoloEvalConfig:
    model: str
    data_yaml: str
    imgsz: int = 256
    batch: int = 16
    device: str | None = "0"
    split: str = "test"
    output_root: str = "yolo/avaliacoes"


def load_yolo_train_config(path: str | Path) -> YoloTrainConfig:
    cfg_path = Path(path).expanduser().resolve()
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    return YoloTrainConfig(**_strip_values(data))


def load_yolo_eval_config(path: str | Path) -> YoloEvalConfig:
    cfg_path = Path(path).expanduser().resolve()
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    return YoloEvalConfig(**_strip_values(data))


__all__ = [
    "YoloTrainConfig",
    "YoloEvalConfig",
    "load_yolo_train_config",
    "load_yolo_eval_config",
]
