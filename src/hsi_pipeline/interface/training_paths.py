"""Path helpers for the YOLO training CLI."""

from __future__ import annotations

from hsi_pipeline.data.paths import GlobalPathConfig


def apply_paths(config, paths: GlobalPathConfig, output_override: str | None, models_override: str | None):
    config.out_root = output_override or paths.output_root or config.out_root
    config.models_root = models_override or paths.models_root or config.models_root


__all__ = ["apply_paths"]
