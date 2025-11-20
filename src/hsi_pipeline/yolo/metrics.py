"""Metrics utilities for YOLO12 training and evaluation."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


def normalize_metrics(metrics: object) -> Dict[str, object]:
    if metrics is None:
        return {}
    if isinstance(metrics, dict):
        return metrics
    if hasattr(metrics, "to_dict"):
        try:
            return metrics.to_dict()  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - best effort
            pass
    if hasattr(metrics, "__dict__"):
        return {k: v for k, v in vars(metrics).items() if not k.startswith("_")}
    return {"value": str(metrics)}


def extract_detection_metrics(metrics: object) -> Dict[str, object]:
    """Flatten Ultralytics detection metrics into a serializable dict."""
    base = normalize_metrics(metrics)
    if not metrics:
        return base
    box = getattr(metrics, "box", None)
    if box:
        for key in ("map", "map50", "map75", "mp", "mr"):
            value = getattr(box, key, None)
            if value is not None:
                base[f"box/{key}"] = value
        maps = getattr(box, "maps", None)
        if maps is not None:
            try:
                base["box/maps"] = [float(x) for x in maps]
            except Exception:  # pragma: no cover - tolerant parsing
                pass
    speed = getattr(metrics, "speed", None)
    if isinstance(speed, dict):
        for key, value in speed.items():
            base[f"speed/{key}"] = value
    return base


def write_training_metrics_csv(metrics: Dict[str, object], dataset_dir: Path) -> Path:
    csv_path = dataset_dir / "training_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            writer.writerow([key, value])
    return csv_path


def append_section(lines: List[str], title: str, rows: Sequence[str]) -> None:
    lines.append(title)
    for row in rows:
        lines.append(f"- {row}")
    lines.append("")


def write_training_summary_txt(
    dataset_dir: Path,
    counts: Dict[str, int],
    metrics: Dict[str, object],
    training_info: Dict[str, object],
    eval_metrics: Optional[Dict[str, object]] = None,
    data_yaml: Optional[Path] = None,
    model_info: Optional[Dict[str, object]] = None,
) -> Path:
    txt_path = dataset_dir / "training_summary.txt"
    lines: List[str] = [
        "Resumo do treinamento YOLO12",
        f"Save dir: {training_info.get('save_dir', 'N/A')}",
        f"Data YAML: {data_yaml}" if data_yaml else "",
        "",
    ]
    lines = [l for l in lines if l]  # drop empties

    append_section(lines, "Contagem por partição:", [f"{split}: {value}" for split, value in counts.items()])

    if training_info.get("train_args"):
        args = training_info["train_args"]
        arg_rows = [f"{k}={v}" for k, v in sorted(args.items())]
        append_section(lines, "Parâmetros de treino:", arg_rows)

    if metrics:
        append_section(lines, "Métricas (treino):", [f"{k}: {v}" for k, v in metrics.items()])

    if eval_metrics:
        append_section(lines, "Métricas (val):", [f"{k}: {v}" for k, v in eval_metrics.items()])

    if training_info.get("plots"):
        append_section(lines, "Plots salvos:", [str(p) for p in training_info["plots"]])

    if model_info and model_info.get("artifacts"):
        append_section(lines, "Artefatos exportados:", [str(p) for p in model_info["artifacts"]])

    with txt_path.open("w", encoding="utf-8") as fp:
        fp.write("\n".join(lines))
    return txt_path


def collect_plot_files(run_dir: Path, patterns: Optional[Iterable[str]] = None) -> List[Path]:
    """Gather plot files from a Ultralytics run directory (default: PNGs)."""
    if not run_dir.exists():
        return []
    patterns = list(patterns or ["*.png"])
    plots: List[Path] = []
    for pattern in patterns:
        plots.extend([p for p in run_dir.glob(pattern) if p.is_file()])
    return sorted({p for p in plots})


def copy_plots(plots: Iterable[Path], target_dir: Path) -> List[Path]:
    copied: List[Path] = []
    target_dir.mkdir(parents=True, exist_ok=True)
    for plot in plots:
        path = Path(plot)
        if not path.exists():
            continue
        dest = target_dir / path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        with path.open("rb") as src, dest.open("wb") as dst:
            dst.write(src.read())
        copied.append(dest)
    return copied


__all__ = [
    "collect_plot_files",
    "copy_plots",
    "extract_detection_metrics",
    "normalize_metrics",
    "write_training_metrics_csv",
    "write_training_summary_txt",
]
