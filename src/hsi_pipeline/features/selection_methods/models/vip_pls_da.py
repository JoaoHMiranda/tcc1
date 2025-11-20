"""VIP PLS-DA ranking."""

from __future__ import annotations

import math
from typing import List

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression

from ....config import VipPLSDASelectionSettings
from ..dataset import PixelSampleDataset
from ..utils import base_dataframe


def run_vip_scores(
    dataset: PixelSampleDataset,
    method_cfg: VipPLSDASelectionSettings,
) -> pd.DataFrame:
    if method_cfg.n_components is not None:
        comps = int(method_cfg.n_components)
    else:
        comps = max(1, min(5, dataset.X.shape[1], dataset.X.shape[0] - 1))
    comps = max(1, min(comps, dataset.X.shape[1], max(1, dataset.X.shape[0] - 1)))
    pls = PLSRegression(n_components=comps)
    pls.fit(dataset.X, dataset.y)
    t = pls.x_scores_
    w = pls.x_weights_
    q = pls.y_loadings_
    p = dataset.X.shape[1]
    vip_scores = np.zeros(p)
    sst = np.sum(np.square(t), axis=0) * np.square(q.reshape(-1))
    denom = np.sum(sst) or 1.0
    for j in range(p):
        weight = np.square(w[j, :])
        vip_scores[j] = math.sqrt(p * np.sum(sst * weight) / denom)
    df = base_dataframe(dataset).copy()
    df["vip_score"] = vip_scores
    df = df.sort_values("vip_score", ascending=False)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df.reset_index(drop=True)
