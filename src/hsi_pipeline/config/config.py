"""Typed configuration for the HSI preprocessing pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union


@dataclass
class TrimConfig:
    left: int = 10
    right: int = 10


@dataclass
class MedianGuessSettings:
    sample_step: int = 4
    max_samples: int = 60
    blur_sigma: float = 1.2
    kernel_size: int = 5
    min_area_fraction: float = 0.02


@dataclass
class VariantOutputSettings:
    enabled: bool = False
    plot: bool = False


@dataclass
class OutputToggleConfig:
    correcao: VariantOutputSettings = field(default_factory=VariantOutputSettings)
    correcao_snv: VariantOutputSettings = field(default_factory=VariantOutputSettings)
    correcao_msc: VariantOutputSettings = field(default_factory=VariantOutputSettings)
    correcao_snv_msc: VariantOutputSettings = field(default_factory=VariantOutputSettings)
    export_metadata: bool = True


@dataclass
class PipelineConfig:
    """Master configuration of the preprocessing pipeline."""

    folder: str = "/home/joaoh/programacao/TCC1/minha/ATCC27_240506-161129"
    out_root: Optional[str] = None
    enabled: bool = True
    cpu_workers: Optional[int] = None
    delta_bands: int = 1
    cache_size_bands: int = 64
    trimming: TrimConfig = field(default_factory=TrimConfig)
    median_guess: MedianGuessSettings = field(default_factory=MedianGuessSettings)
    toggles: OutputToggleConfig = field(default_factory=OutputToggleConfig)
    dataset_filters: Dict[str, bool] = field(default_factory=dict)


def strip_descriptions(node: Any) -> Any:
    if isinstance(node, dict):
        if set(node.keys()).issubset({"value", "description"}) and "value" in node:
            return node["value"]
        cleaned: Dict[str, Any] = {}
        for key, value in node.items():
            if key == "description":
                continue
            cleaned[key] = strip_descriptions(value)
        return cleaned
    if isinstance(node, list):
        return [strip_descriptions(item) for item in node]
    return node


def update_dataclass(instance, values: Dict[str, Any]):
    for key, value in values.items():
        if hasattr(instance, key):
            setattr(instance, key, value)
    return instance


def load_config_from_mapping(data: Dict[str, Any]) -> PipelineConfig:
    trimmed = strip_descriptions(data)
    base = PipelineConfig()
    base.folder = trimmed.get("folder", base.folder)
    base.out_root = trimmed.get("out_root", base.out_root)
    base.enabled = trimmed.get("enabled", base.enabled)
    base.cpu_workers = trimmed.get("cpu_workers", base.cpu_workers)
    base.delta_bands = trimmed.get("delta_bands", base.delta_bands)
    base.cache_size_bands = trimmed.get("cache_size_bands", base.cache_size_bands)

    trimming = trimmed.get("trimming", {})
    base.trimming = TrimConfig(
        left=trimming.get("left", base.trimming.left),
        right=trimming.get("right", base.trimming.right),
    )

    med = trimmed.get("median_guess", {})
    base.median_guess = update_dataclass(MedianGuessSettings(), med)

    toggles_cfg = trimmed.get("toggles", {})
    base.toggles = OutputToggleConfig()

    def _coerce_variant_settings(value) -> VariantOutputSettings:
        if isinstance(value, dict):
            return update_dataclass(VariantOutputSettings(), value)
        if isinstance(value, bool):
            return VariantOutputSettings(enabled=value, plot=value)
        return VariantOutputSettings()

    for field_name in ("correcao", "correcao_snv", "correcao_msc", "correcao_snv_msc"):
        val = toggles_cfg.get(field_name)
        if val is not None:
            setattr(base.toggles, field_name, _coerce_variant_settings(val))
    export_meta = toggles_cfg.get("export_metadata")
    if export_meta is not None:
        base.toggles.export_metadata = bool(export_meta)

    dataset_filters = trimmed.get("dataset_filters")
    if isinstance(dataset_filters, dict):
        base.dataset_filters = dataset_filters

    return base


def load_config_from_json(path: Union[str, Path]) -> PipelineConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return load_config_from_mapping(raw)


__all__ = [
    "PipelineConfig",
    "TrimConfig",
    "MedianGuessSettings",
    "VariantOutputSettings",
    "OutputToggleConfig",
    "load_config_from_json",
    "load_config_from_mapping",
]
