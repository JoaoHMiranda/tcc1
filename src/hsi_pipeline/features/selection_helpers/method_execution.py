"""Execution helpers for selection methods."""

from __future__ import annotations

import os
from typing import Dict, Sequence

from ..selection_methods import (
    DEFAULT_SELECTION_REASON,
    method_config,
    save_dataframe,
    run_anova,
    run_t_test,
    run_kruskal,
    run_random_forest,
    run_vip_scores,
    run_pca,
)
from ...config import BandSelectionConfig


def execute_method(method: str, dataset, config: BandSelectionConfig, out_dir: str) -> Dict[str, object]:
    method_cfg = method_config(config, method)
    if not method_cfg:
        return {"error": f"Configuração ausente para o método {method}."}
    if method == "anova":
        df = run_anova(dataset, method_cfg.alpha)
        path = save_dataframe(df, os.path.join(out_dir, "ranking_anova.csv"))
        return {"df": df, "path": path}
    if method == "t_test":
        df = run_t_test(dataset, method_cfg.alpha)
        path = save_dataframe(df, os.path.join(out_dir, "ranking_t_test.csv"))
        return {"df": df, "path": path}
    if method == "kruskal":
        df = run_kruskal(dataset, method_cfg.alpha)
        path = save_dataframe(df, os.path.join(out_dir, "ranking_kruskal.csv"))
        return {"df": df, "path": path}
    if method == "random_forest":
        df = run_random_forest(dataset, method_cfg, config.random_state)
        path = save_dataframe(df, os.path.join(out_dir, "ranking_random_forest.csv"))
        return {"df": df, "path": path, "reason": DEFAULT_SELECTION_REASON}
    if method == "vip_pls_da":
        df = run_vip_scores(dataset, method_cfg)
        path = save_dataframe(df, os.path.join(out_dir, "ranking_vip_pls_da.csv"))
        return {"df": df, "path": path}
    if method == "pca":
        ranking_df, components_df, variance_df = run_pca(dataset, method_cfg)
        rank_path = save_dataframe(ranking_df, os.path.join(out_dir, "ranking_pca.csv"))
        save_dataframe(components_df, os.path.join(out_dir, "pca_components.csv"))
        save_dataframe(variance_df, os.path.join(out_dir, "pca_variance.csv"))
        return {"df": ranking_df, "path": rank_path}
    return {"error": f"Método desconhecido: {method}"}


def execute_enabled_methods(
    config: BandSelectionConfig,
    dataset,
    out_dir: str,
    enabled_methods: Sequence[str],
) -> Dict[str, Dict[str, object]]:
    outputs: Dict[str, Dict[str, object]] = {}
    for method in enabled_methods:
        try:
            outputs[method] = execute_method(method, dataset, config, out_dir)
        except Exception as exc:  # pragma: no cover
            outputs[method] = {"error": str(exc)}
    return outputs


__all__ = ["execute_method", "execute_enabled_methods"]
