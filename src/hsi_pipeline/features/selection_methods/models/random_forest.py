"""Random Forest based feature ranking."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ....config import RandomForestSelectionSettings
from ..dataset import PixelSampleDataset
from ..utils import base_dataframe


DEFAULT_SELECTION_REASON = (
    "Random Forest Feature Importance foi definido como padrão por capturar relações "
    "não lineares e tolerar ruído típico de HSI biomédico sem pressupor distribuição."
)


def run_random_forest(
    dataset: PixelSampleDataset,
    method_cfg: RandomForestSelectionSettings,
    random_state: int,
) -> pd.DataFrame:
    clf = RandomForestClassifier(
        n_estimators=method_cfg.n_estimators,
        random_state=random_state,
        n_jobs=method_cfg.n_jobs,
        class_weight=method_cfg.class_weight,
        max_features=method_cfg.max_features,
        bootstrap=method_cfg.bootstrap,
    )
    clf.fit(dataset.X, dataset.y)
    importances = clf.feature_importances_
    df = base_dataframe(dataset).copy()
    df["importance"] = importances
    df = df.sort_values("importance", ascending=False)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df.reset_index(drop=True)
