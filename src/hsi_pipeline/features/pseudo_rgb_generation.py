"""Pseudo-RGB generation helpers (manual, PCA and linear combos)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.decomposition import PCA

from ..config import PseudoRGBConfig
from .rgb import save_rgb_from_channels, scale_robust


@dataclass
class PseudoResult:
    method: str
    outputs: List[str]
    details: Dict[str, object]


def resolve_band_position(identifier: int, kept_indices: Sequence[int]) -> int:
    if identifier in kept_indices:
        return kept_indices.index(identifier)
    if 0 <= int(identifier) < len(kept_indices):
        return int(identifier)
    raise ValueError(f"Índice de banda inválido: {identifier}")


def make_output_dir(base_dir: str, name: str) -> str:
    path = os.path.join(base_dir, name)
    os.makedirs(path, exist_ok=True)
    return path


def save_rgb(channels: Sequence[np.ndarray], path: str) -> str:
    save_rgb_from_channels(channels[0], channels[1], channels[2], path)
    return path


def generate_manual(
    config: PseudoRGBConfig,
    bands: Sequence[np.ndarray],
    kept: Sequence[int],
    wavs: Sequence[float],
    out_dir: str,
) -> PseudoResult:
    if len(config.manual.band_indices) < 3:
        raise ValueError("manual.band_indices precisa conter ao menos 3 entradas.")
    indices = [
        resolve_band_position(idx, kept)
        for idx in config.manual.band_indices[:3]
    ]
    channels = [scale_robust(bands[idx]) for idx in indices]
    out_path = os.path.join(out_dir, config.manual.output_name)
    save_rgb(channels, out_path)
    details = [
        {
            "kept_index": idx,
            "orig_band_index": kept[idx],
            "wavelength_nm": wavs[idx],
        }
        for idx in indices
    ]
    return PseudoResult("manual", [out_path], {"bands": details})


def generate_pca(
    config: PseudoRGBConfig,
    bands: Sequence[np.ndarray],
    out_dir: str,
) -> PseudoResult:
    stack = np.stack(bands, axis=-1)
    H, W, B = stack.shape
    flat = stack.reshape(-1, B).astype(np.float32)
    if config.pca.standardize:
        mean = flat.mean(axis=0)
        std = flat.std(axis=0)
        std[std < 1e-12] = 1.0
        flat = (flat - mean) / std
    n_samples = flat.shape[0]
    max_pixels = max(1000, int(config.pca.max_pixels))
    if n_samples > max_pixels:
        idx = np.random.default_rng(config.pca.random_state).choice(n_samples, max_pixels, replace=False)
        sample = flat[idx]
    else:
        sample = flat
    pca = PCA(n_components=3, whiten=config.pca.whiten, random_state=config.pca.random_state)
    pca.fit(sample)
    transformed = pca.transform(flat)
    transformed = transformed.reshape(H, W, 3)
    channels = [scale_robust(transformed[:, :, i]) for i in range(3)]
    out_path = os.path.join(out_dir, "pca_rgb.png")
    save_rgb(channels, out_path)
    return PseudoResult(
        "pca",
        [out_path],
        {
            "explained_variance": pca.explained_variance_ratio_.tolist(),
        },
    )


def channel_from_weights(
    weights: List[Dict[str, float]],
    bands: Sequence[np.ndarray],
    kept: Sequence[int],
) -> np.ndarray:
    channel = np.zeros_like(bands[0], dtype=np.float32)
    for item in weights:
        band_identifier = item.get("kept_index")
        if band_identifier is None and "orig_band_index" in item:
            try:
                band_identifier = kept.index(int(item["orig_band_index"]))
            except ValueError:
                band_identifier = None
        if band_identifier is None and "band" in item:
            band_identifier = item["band"]
        if band_identifier is None:
            continue
        idx = resolve_band_position(int(band_identifier), kept)
        weight = float(item.get("weight", 1.0))
        channel += weight * bands[idx]
    return channel


def generate_linear(
    config: PseudoRGBConfig,
    bands: Sequence[np.ndarray],
    kept: Sequence[int],
    out_dir: str,
) -> PseudoResult:
    outputs: List[str] = []
    details: List[Dict[str, object]] = []
    recipes = config.linear.recipes or []
    for recipe in recipes:
        name = recipe.get("name", "recipe")
        channels_def = recipe.get("channels", {})
        r_weights = channels_def.get("r", [])
        g_weights = channels_def.get("g", [])
        b_weights = channels_def.get("b", [])
        r = scale_robust(channel_from_weights(r_weights, bands, kept))
        g = scale_robust(channel_from_weights(g_weights, bands, kept))
        b = scale_robust(channel_from_weights(b_weights, bands, kept))
        recipe_dir = make_output_dir(out_dir, name)
        out_path = os.path.join(recipe_dir, f"{name}.png")
        save_rgb((r, g, b), out_path)
        outputs.append(out_path)
        details.append({
            "name": name,
            "description": recipe.get("description", ""),
        })
    return PseudoResult("linear", outputs, {"recipes": details})


def run_pseudo_rgb_stage(
    config: PseudoRGBConfig,
    bands: Sequence[np.ndarray],
    kept: Sequence[int],
    wavs: Sequence[float],
    out_base: str,
) -> Dict[str, object]:
    base_dir = os.path.join(out_base, config.output_dir_name)
    os.makedirs(base_dir, exist_ok=True)
    enabled_methods = [m for m in ("manual", "pca", "linear") if getattr(config.toggles, m, False)]
    summary: Dict[str, object] = {
        "enabled": config.enabled,
        "methods": [],
        "active_method": config.active_method,
    }
    if not enabled_methods:
        return summary
    active_method = config.active_method or enabled_methods[0]
    method_results: List[PseudoResult] = []
    for method in enabled_methods:
        method_dir = make_output_dir(base_dir, method)
        if method == "manual":
            result = generate_manual(config, bands, kept, wavs, method_dir)
        elif method == "pca":
            result = generate_pca(config, bands, method_dir)
        else:
            result = generate_linear(config, bands, kept, method_dir)
        method_results.append(result)
        summary["methods"].append(
            {
                "method": method,
                "outputs": [os.path.relpath(path, out_base) for path in result.outputs],
                "details": result.details,
            }
        )
    summary["active_method"] = active_method if active_method in enabled_methods else enabled_methods[0]
    summary["base_dir"] = os.path.relpath(base_dir, out_base)
    return summary


__all__ = ["run_pseudo_rgb_stage"]
