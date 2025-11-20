"""Utility helpers shared across selection modules."""

from __future__ import annotations

from typing import List, Optional, Sequence

import pandas as pd

from ...config import BandSelectionConfig
from .dataset import PixelSampleDataset


def method_config(cfg: BandSelectionConfig, name: str):
    methods = getattr(cfg, "methods", None)
    if methods is None:
        return None
    return getattr(methods, name, None)


def enabled_selection_methods(cfg: BandSelectionConfig) -> List[str]:
    order = ["anova", "t_test", "kruskal", "random_forest", "vip_pls_da", "pca"]
    enabled: List[str] = []
    methods = getattr(cfg, "methods", None)
    if methods is None:
        return enabled
    for name in order:
        method_cfg = getattr(methods, name, None)
        if method_cfg and getattr(method_cfg, "enabled", False):
            enabled.append(name)
    return enabled


def resolve_active_method(cfg: BandSelectionConfig, enabled: Sequence[str]) -> Optional[str]:
    if not enabled:
        return None
    if not cfg.active_method:
        return enabled[0]
    if cfg.active_method in enabled:
        return cfg.active_method
    return enabled[0]


def base_dataframe(dataset: PixelSampleDataset) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "band_index_kept": dataset.kept_band_indices,
            "orig_band_index": dataset.orig_band_indices,
            "wavelength_nm": dataset.wavelengths,
        }
    )


def save_dataframe(df: pd.DataFrame, path: str) -> str:
    df.to_csv(path, index=False)
    return path


def band_list_from_df(df: pd.DataFrame, top_k: int) -> List[int]:
    return df["band_index_kept"].head(top_k).astype(int).tolist()
