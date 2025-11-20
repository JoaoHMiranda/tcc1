"""Band selection and lightweight classification utilities."""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

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
from .selection_helpers.postprocessing import summarize_selected_bands, pca_on_selected, select_top_bands
from .selection_methods.classification import enabled_classifiers




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
    progress=None,
) -> Dict[str, object]:
    os.makedirs(out_dir, exist_ok=True)
    summary: Dict[str, object] = {"dataset": dataset_name, "enabled": config.enabled}
    task_key = f"selection_{dataset_name}"
    if not config.enabled:
        summary["message"] = "Etapa desativada."
        write_selection_summary(out_dir, summary)
        if progress:
            progress.complete(task_key)
        return summary
    start = time.perf_counter()
    enabled_methods = enabled_selection_methods(config)
    method_count = len(enabled_methods)
    active_classifiers = enabled_classifiers(config.classification) if config.classification else []
    classification_steps = len(active_classifiers) if config.classification and config.classification.enabled else 0
    total_steps = max(
        1,
        1  # dataset
        + max(1, method_count)  # métodos (pelo menos 1 passo)
        + 1  # top-k/summary
        + 1  # PCA
        + (classification_steps or 1)  # classificação (ou placeholder)
        + 1,  # write summary
    )
    if progress:
        progress.create_task(task_key, f"[cyan]Seleção de bandas ({dataset_name})", total_steps)
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
    if progress:
        progress.advance(task_key)
    if dataset is None:
        summary["message"] = error or "Não foi possível montar as amostras."
        write_selection_summary(out_dir, summary)
        if progress:
            progress.complete(task_key)
        return summary
    if len(enabled_methods) != 1:
        raise ValueError(
            "Ative exatamente um método de seleção de bandas; o método escolhido será usado pelo PCA."
        )
    active_method = resolve_active_method(config, enabled_methods)
    method_outputs = execute_enabled_methods(config, dataset, out_dir, enabled_methods)
    if progress:
        progress.advance(task_key, max(1, method_count))
    selected_band_indices: List[int] = []
    selected_summary: List[Dict[str, object]] = []
    if active_method and method_outputs.get(active_method, {}).get("df") is not None:
        df = method_outputs[active_method]["df"]
        selected_band_indices = select_top_bands(df, config)
        selected_summary = summarize_selected_bands(dataset, df, selected_band_indices)
    if progress:
        progress.advance(task_key)
    pca_selected_summary = pca_on_selected(dataset, selected_band_indices, config, out_dir)
    if progress:
        progress.advance(task_key)
    classification_summary = run_classification(
        dataset,
        selected_band_indices,
        config.classification,
        out_dir,
    )
    if progress:
        progress.advance(task_key, max(1, classification_steps))
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
    if progress:
        progress.complete(task_key)
    return summary


__all__ = ["run_band_selection_pipeline"]
