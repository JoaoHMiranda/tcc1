"""Config dataclass and loader for YOLO-based classification."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass
class ClassificationConfig:
    model: str
    source_root: str = "hsi_modificado/classificar"
    output_root: str = "resultados"
    imgsz: int = 256
    device: str | None = "0"
    conf: float = 0.5


def _resolve_value(mapping: Mapping[str, Any], key: str, default: Any) -> Any:
    value = mapping.get(key, default)
    if isinstance(value, Mapping) and "value" in value:
        return value["value"]
    return value


def load_classification_config(path: str | Path) -> ClassificationConfig:
    cfg_path = Path(path).expanduser().resolve()
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    model = str(_resolve_value(data, "model", ""))
    source_root = str(_resolve_value(data, "source_root", "hsi_modificado/classificar"))
    output_root = str(_resolve_value(data, "output_root", "resultados"))
    imgsz = int(_resolve_value(data, "imgsz", 256))
    conf = float(_resolve_value(data, "conf", 0.5))
    device_value = _resolve_value(data, "device", "0")
    device = None if device_value in ("", None) else str(device_value)
    return ClassificationConfig(
        model=model,
        source_root=source_root,
        output_root=output_root,
        imgsz=imgsz,
        device=device,
        conf=conf,
    )


__all__ = ["ClassificationConfig", "load_classification_config"]
