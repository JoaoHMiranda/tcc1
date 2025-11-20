"""Two-sample t-test ranking."""

from __future__ import annotations

import math
import warnings
from typing import List

import pandas as pd
from scipy import stats

from ..dataset import PixelSampleDataset
from ..utils import base_dataframe


def run_t_test(dataset: PixelSampleDataset, alpha: float) -> pd.DataFrame:
    mask_a = dataset.y == 0
    mask_b = dataset.y == 1
    Xa = dataset.X[mask_a]
    Xb = dataset.X[mask_b]
    stats_vals: List[float] = []
    p_values: List[float] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for col in range(dataset.X.shape[1]):
            if Xa.shape[0] == 0 or Xb.shape[0] == 0:
                stats_vals.append(math.nan)
                p_values.append(math.nan)
                continue
            stat, p_val = stats.ttest_ind(Xa[:, col], Xb[:, col], equal_var=False)
            stats_vals.append(float(stat))
            p_values.append(float(p_val))
    df = base_dataframe(dataset).copy()
    df["statistic"] = stats_vals
    df["p_value"] = p_values
    df["significant"] = df["p_value"] < alpha
    df = df.sort_values("p_value", ascending=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df.reset_index(drop=True)
