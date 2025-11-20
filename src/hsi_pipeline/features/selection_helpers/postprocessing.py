"""Helpers for summarizing selection outputs and PCA on selected bands."""

from __future__ import annotations

import math
import os
from typing import Dict, List

from ..selection_methods import band_list_from_df, run_pca, save_dataframe


def summarize_selected_bands(dataset, df, selected_band_indices: List[int]) -> List[Dict[str, object]]:
    summary_rows: List[Dict[str, object]] = []
    position_map = dataset.feature_positions()
    for idx in selected_band_indices:
        pos = position_map.get(idx)
        if pos is None:
            continue
        rank_series = df.loc[df["band_index_kept"] == idx, "rank"]
        rank = int(rank_series.iloc[0]) if not rank_series.empty else math.nan
        summary_rows.append(
            {
                "band_index_kept": idx,
                "orig_band_index": dataset.orig_band_indices[pos],
                "wavelength_nm": dataset.wavelengths[pos],
                "rank": rank,
            }
        )
    return summary_rows


def pca_on_selected(dataset, selected_band_indices: List[int], config, out_dir: str):
    if not selected_band_indices:
        return {}
    try:
        subset = dataset.subset(selected_band_indices)
        rank_df, components_df, variance_df = run_pca(subset, config.methods.pca)
        return {
            "ranking_csv": save_dataframe(rank_df, os.path.join(out_dir, "ranking_pca_selected.csv")),
            "components_csv": save_dataframe(
                components_df, os.path.join(out_dir, "pca_components_selected.csv")
            ),
            "variance_csv": save_dataframe(variance_df, os.path.join(out_dir, "pca_variance_selected.csv")),
        }
    except Exception as exc:  # pragma: no cover
        return {"error": str(exc)}


def select_top_bands(df, config):
    return band_list_from_df(df, config.top_k_bands)


__all__ = ["summarize_selected_bands", "pca_on_selected", "select_top_bands"]
