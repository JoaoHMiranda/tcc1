"""Band selection and lightweight classification utilities."""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import BandSelectionConfig
from .selection_methods import (
    enabled_selection_methods,
    prepare_pixel_dataset,
    resolve_active_method,
    run_classification,
    write_selection_summary,
)
from .selection_helpers.method_execution import execute_enabled_methods
from .selection_helpers.postprocessing import (
    summarize_selected_bands,
    pca_on_selected,
    select_top_bands,
)




def run_band_selection_pipeline(
    config: BandSelectionConfig,
    bands: Sequence[np.ndarray],
    kept_band_indices: Sequence[int],
    orig_band_indices: Sequence[int],
    wavelengths: Sequence[float],
    center: Tuple[float, float],
    radius: float,
    out_dir: str,
    dataset_name: str,
) -> Dict[str, object]:
    os.makedirs(out_dir, exist_ok=True)
    summary: Dict[str, object] = {"dataset": dataset_name, "enabled": config.enabled}
    if not config.enabled:
        summary["message"] = "Etapa desativada."
        write_selection_summary(out_dir, summary)
        return summary
    start = time.perf_counter()
    dataset, error = prepare_pixel_dataset(
        bands,
        kept_band_indices,
        orig_band_indices,
        wavelengths,
        center[0],
        center[1],
        radius,
        config,
    )
    if dataset is None:
        summary["message"] = error or "Não foi possível montar as amostras."
        write_selection_summary(out_dir, summary)
        return summary
    enabled_methods = enabled_selection_methods(config)
    if len(enabled_methods) != 1:
        raise ValueError(
            "Ative exatamente um método de seleção de bandas; o método escolhido será usado pelo PCA."
        )
    active_method = resolve_active_method(config, enabled_methods)
    method_outputs = execute_enabled_methods(config, dataset, out_dir, enabled_methods)
    selected_band_indices: List[int] = []
    selected_summary: List[Dict[str, object]] = []
    if active_method and method_outputs.get(active_method, {}).get("df") is not None:
        df = method_outputs[active_method]["df"]
        selected_band_indices = select_top_bands(df, config)
        selected_summary = summarize_selected_bands(dataset, df, selected_band_indices)
    pca_selected_summary = pca_on_selected(dataset, selected_band_indices, config, out_dir)
    classification_summary = run_classification(
        dataset,
        selected_band_indices,
        config.classification,
        out_dir,
    )
    elapsed = time.perf_counter() - start
    summary.update(
        {
            "elapsed_s": elapsed,
            "active_method": active_method,
            "selected_band_indices": selected_band_indices,
            "selected_bands": selected_summary,
            "selection_files": {
                method: info.get("path") for method, info in method_outputs.items() if info.get("path")
            },
            "classification": classification_summary,
        }
    )
    if pca_selected_summary:
        summary["pca_on_selected_bands"] = pca_selected_summary
    if active_method and method_outputs.get(active_method, {}).get("reason"):
        summary["method_reason"] = method_outputs[active_method]["reason"]
    write_selection_summary(out_dir, summary)
    return summary


__all__ = ["run_band_selection_pipeline"]
