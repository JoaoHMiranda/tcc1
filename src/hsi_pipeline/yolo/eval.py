"""Evaluation routine for YOLO models with detection metrics and channel plots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from ultralytics import YOLO

from .config import YoloEvalConfig
from .utils import (
    next_run_name,
    resolve_model_path,
    resolve_dataset_for_model,
    enforce_offline_mode,
)


def _channel_stats(images_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    means, stds, mins, maxs = [], [], [], []
    pngs = sorted(images_dir.glob("*.png"))
    for path in pngs:
        img = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        means.append(img.reshape(-1, 3).mean(axis=0))
        stds.append(img.reshape(-1, 3).std(axis=0))
        mins.append(img.reshape(-1, 3).min(axis=0))
        maxs.append(img.reshape(-1, 3).max(axis=0))
    if not means:
        raise RuntimeError(f"Nenhuma imagem encontrada em {images_dir}")
    means_arr = np.stack(means).mean(axis=0)
    stds_arr = np.stack(stds).mean(axis=0)
    mins_arr = np.stack(mins).min(axis=0)
    maxs_arr = np.stack(maxs).max(axis=0)
    return means_arr, stds_arr, mins_arr, maxs_arr


def _plot_channel_stats(out_dir: Path, means: np.ndarray, stds: np.ndarray) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    channels = ["R", "G", "B"]
    x = np.arange(len(channels))
    plt.figure(figsize=(4, 3))
    plt.bar(x, means, yerr=stds, capsize=5, color=["#d9534f", "#5cb85c", "#5bc0de"])
    plt.xticks(x, channels)
    plt.ylabel("Média (0-1)")
    plt.title("Estatísticas por canal (teste)")
    plt.tight_layout()
    out_path = out_dir / "channel_stats.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def _to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def run_yolo_evaluation(config: YoloEvalConfig) -> Dict[str, Any]:
    enforce_offline_mode()
    model_path = resolve_model_path(config.model)
    data_yaml = resolve_dataset_for_model(config.data_yaml, model_path)
    output_root = Path(config.output_root).expanduser().resolve()
    runs_root = output_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    base_name = model_path.stem
    eval_name = next_run_name(runs_root, f"eval_{base_name}")

    print(f"[info] Avaliando {model_path} com data={data_yaml} split={config.split}")
    model = YOLO(str(model_path))
    results = model.val(
        data=str(data_yaml),
        imgsz=config.imgsz,
        batch=config.batch,
        device=config.device,
        split=config.split,
        project=str(runs_root),
        name=eval_name,
        exist_ok=False,
        save_json=True,
        plots=True,
    )

    metrics_raw = {
        "map50": getattr(results.box, "map50", None),
        "map": getattr(results.box, "map", None),
        "maps": getattr(results.box, "maps", None),
        "results_dir": str(results.save_dir),
    }
    metrics = {k: _to_serializable(v) for k, v in metrics_raw.items()}

    dataset_root = data_yaml.parent
    images_test = dataset_root / "images" / config.split
    means, stds, mins, maxs = _channel_stats(images_test)
    plots_dir = results.save_dir
    stats_plot = _plot_channel_stats(plots_dir, means, stds)

    channel_stats = {
        "mean": means.tolist(),
        "std": stds.tolist(),
        "min": mins.tolist(),
        "max": maxs.tolist(),
        "plot": str(stats_plot),
    }

    summary = {
        "model": str(model_path),
        "data_yaml": str(data_yaml),
        "split": config.split,
        "imgsz": config.imgsz,
        "batch": config.batch,
        "device": config.device,
        "metrics": metrics,
        "channel_stats": channel_stats,
        "run_dir": str(results.save_dir),
    }

    summary_path = Path(results.save_dir) / "evaluation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[ok] Avaliação concluída. Resumo em {summary_path}")
    return summary


__all__ = ["run_yolo_evaluation"]
