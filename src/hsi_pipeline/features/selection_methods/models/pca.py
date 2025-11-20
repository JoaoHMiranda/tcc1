"""PCA helper used both directly and on selected bands."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from ....config import PCASelectionSettings
from ..dataset import PixelSampleDataset
from ..utils import base_dataframe


def run_pca(
    dataset: PixelSampleDataset,
    method_cfg: Optional[PCASelectionSettings] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if method_cfg and method_cfg.n_components is not None:
        n_components = int(method_cfg.n_components)
    else:
        n_components = max(1, min(10, dataset.X.shape[1], dataset.X.shape[0]))
    n_components = max(1, min(n_components, dataset.X.shape[1], dataset.X.shape[0]))
    whiten = bool(method_cfg.whiten) if method_cfg else False
    pca = PCA(n_components=n_components, whiten=whiten)
    pca.fit(dataset.X)
    variance_df = pd.DataFrame(
        {
            "component": np.arange(1, n_components + 1),
            "explained_variance_ratio": pca.explained_variance_ratio_,
        }
    )
    components_list: List[Dict[str, object]] = []
    for comp_idx, loadings in enumerate(pca.components_, start=1):
        for band_pos, loading in enumerate(loadings):
            components_list.append(
                {
                    "component": comp_idx,
                    "band_index_kept": dataset.kept_band_indices[band_pos],
                    "orig_band_index": dataset.orig_band_indices[band_pos],
                    "wavelength_nm": dataset.wavelengths[band_pos],
                    "loading": float(loading),
                }
            )
    components_df = pd.DataFrame(components_list)
    score = np.zeros(len(dataset.kept_band_indices))
    for comp_idx, loading in enumerate(pca.components_):
        score += np.abs(loading) * pca.explained_variance_ratio_[comp_idx]
    ranking_df = base_dataframe(dataset).copy()
    ranking_df["contribution_score"] = score
    ranking_df = ranking_df.sort_values("contribution_score", ascending=False)
    ranking_df.insert(0, "rank", range(1, len(ranking_df) + 1))
    return ranking_df.reset_index(drop=True), components_df, variance_df
