"""Optional lightweight classification run after band selection."""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from ...config import ClassificationConfig
from .dataset import PixelSampleDataset


DEFAULT_CLASSIFICATION_REASON = (
    "Mantivemos o SVM linear como classificador principal por funcionar bem em HSI "
    "com poucas amostras e margens amplas, exigindo pouca calibração."
)


def enabled_classifiers(cfg: ClassificationConfig) -> List[str]:
    order = ["svm_linear", "random_forest", "pls_da"]
    enabled: List[str] = []
    for name in order:
        if getattr(cfg.toggles, name, False):
            enabled.append(name)
    return enabled


def resolve_active_classifier(cfg: ClassificationConfig, enabled: Sequence[str]) -> Optional[str]:
    if not enabled:
        return None
    if not cfg.active_method:
        return enabled[0]
    if cfg.active_method in enabled:
        return cfg.active_method
    return enabled[0]


def classifier_from_name(name: str, cfg: ClassificationConfig):
    if name == "svm_linear":
        return SVC(
            kernel="linear",
            class_weight="balanced",
            probability=True,
            random_state=cfg.random_state,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=600,
            random_state=cfg.random_state,
            n_jobs=-1,
            class_weight="balanced",
            max_features="sqrt",
            bootstrap=True,
        )
    if name == "pls_da":
        comps = 2
        return PLSRegression(n_components=comps)
    raise ValueError(f"Classificador desconhecido: {name}")


def run_classification(
    dataset: PixelSampleDataset,
    selected_band_indices: Optional[List[int]],
    cfg: ClassificationConfig,
    out_dir: str,
) -> Optional[Dict[str, object]]:
    enabled = enabled_classifiers(cfg)
    if not cfg.enabled or not enabled:
        return None
    active = resolve_active_classifier(cfg, enabled)
    if active is None:
        return None
    feature_positions = dataset.feature_positions()
    if selected_band_indices:
        cols = [feature_positions[idx] for idx in selected_band_indices if idx in feature_positions]
    else:
        cols = list(range(dataset.X.shape[1]))
    if not cols:
        return None
    X_sel = dataset.X[:, cols]
    y = dataset.y
    if len(np.unique(y)) < 2:
        return None
    X_train, X_test, y_train, y_test = train_test_split(
        X_sel,
        y,
        test_size=cfg.test_size,
        stratify=y,
        random_state=cfg.random_state,
    )
    classifier = classifier_from_name(active, cfg)
    if active == "pls_da":
        classifier.fit(X_train, y_train)
        y_score = classifier.predict(X_test).ravel()
        y_pred = (y_score >= 0.5).astype(int)
    else:
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        if hasattr(classifier, "predict_proba"):
            y_score = classifier.predict_proba(X_test)[:, 1]
        else:
            y_score = classifier.decision_function(X_test)
    metrics = {
        "active_classifier": active,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_score))
    except ValueError:
        metrics["roc_auc"] = math.nan
    metrics["n_train"] = int(X_train.shape[0])
    metrics["n_test"] = int(X_test.shape[0])
    metrics["n_features"] = int(len(cols))
    metrics["selected_bands"] = selected_band_indices or []
    path = os.path.join(out_dir, "classification_metrics.csv")
    pd.DataFrame([metrics]).to_csv(path, index=False)
    metrics["metrics_path"] = path
    metrics["reason"] = DEFAULT_CLASSIFICATION_REASON
    return metrics
