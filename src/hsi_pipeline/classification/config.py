"""Configuration loader for the classification/inference stage."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass
class ClassificationConfig:
    model: str
    source_root: str
    output_root: str
    imgsz: int = 1280
    device: str | None = "0"


DEFAULT_CLASSIFICATION_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "classificar.json"


def resolve_value(mapping: Mapping[str, Any], key: str, default: Any) -> Any:
    value = mapping.get(key, default)
    if isinstance(value, Mapping) and "value" in value:
        return value["value"]
    return value


def load_classification_config(path: str | Path) -> ClassificationConfig:
    cfg_path = Path(path).expanduser().resolve()
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    model = str(resolve_value(data, "model", ""))
    source_root = str(resolve_value(data, "source_root", ""))
    output_root = str(resolve_value(data, "output_root", "resultados"))
    imgsz = int(resolve_value(data, "imgsz", 1280))
    device_value = resolve_value(data, "device", "0")
    device = None if device_value in ("", None) else str(device_value)
    return ClassificationConfig(
        model=model,
        source_root=source_root,
        output_root=output_root,
        imgsz=imgsz,
        device=device,
    )


__all__ = ["ClassificationConfig", "load_classification_config", "DEFAULT_CLASSIFICATION_CONFIG"]
