"""ANOVA ranking implementation."""

from __future__ import annotations

import math
import warnings
from typing import List

import numpy as np
import pandas as pd
from scipy import stats

from ..dataset import PixelSampleDataset
from ..stats_common import ConstantInputWarning
from ..utils import base_dataframe


def run_anova(dataset: PixelSampleDataset, alpha: float) -> pd.DataFrame:
    unique_labels = np.unique(dataset.y)
    groups = [dataset.X[dataset.y == label] for label in unique_labels]
    stats_vals: List[float] = []
    p_values: List[float] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        warnings.simplefilter("ignore", category=ConstantInputWarning)
        for col in range(dataset.X.shape[1]):
            samples = [grp[:, col] for grp in groups if grp.shape[0] > 0]
            if len(samples) < 2:
                stats_vals.append(math.nan)
                p_values.append(math.nan)
                continue
            stat, p_val = stats.f_oneway(*samples)
            stats_vals.append(float(stat))
            p_values.append(float(p_val))
    df = base_dataframe(dataset).copy()
    df["statistic"] = stats_vals
    df["p_value"] = p_values
    df["significant"] = df["p_value"] < alpha
    df = df.sort_values("p_value", ascending=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df.reset_index(drop=True)
