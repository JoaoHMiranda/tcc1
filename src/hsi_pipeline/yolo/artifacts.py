"""Artifacts and metrics handling for YOLO12 training."""

from __future__ import annotations

import csv
import json
import pickle
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from ultralytics import YOLO

from ..config import YoloTrainingConfig
from .dataset import resolve_path
from .export_requirements import missing_dependency_message
from .fs_utils import next_available_path


def copy_into(dir_path: Path, source: Path) -> Optional[Path]:
    if not source.exists():
        return None
    dir_path.mkdir(parents=True, exist_ok=True)
    target = dir_path / source.name
    shutil.copy2(source, target)
    return target


def copy_training_reports(model_dir: Path, files: Sequence[Path]) -> List[Path]:
    """Mirror dataset/report artifacts inside the model directory."""
    copied: List[Path] = []
    reports_dir = model_dir / "reports"
    for file_path in files:
        if not file_path:
            continue
        path = Path(file_path)
        copied_file = copy_into(reports_dir, path)
        if copied_file:
            copied.append(copied_file)
    return copied


def write_model_artifacts_csv(artifacts: List[Path], model_dir: Path) -> Path:
    csv_path = model_dir / "model_artifacts.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["file", "type"])
        for path in artifacts:
            writer.writerow([str(path), path.suffix])
    return csv_path


def write_model_summary_txt(model_dir: Path, artifacts: List[Path]) -> Path:
    txt_path = model_dir / "model_summary.txt"
    lines = [f"Artefatos exportados ({len(artifacts)} arquivos):"]
    for path in artifacts:
        lines.append(f"- {path.name}")
    with txt_path.open("w", encoding="utf-8") as fp:
        fp.write("\n".join(lines))
    return txt_path


def export_model_artifacts(
    training_info: Dict[str, object],
    config: YoloTrainingConfig,
) -> Optional[Dict[str, object]]:
    models_root = resolve_path(config.models_root)
    if models_root is None:
        raise ValueError(
            "Defina 'models_root' (em configs/global_paths.json ou --models-root) para salvar os artefatos em 'modelos/'."
        )
    version_name = Path(config.model).stem
    model_dir = next_available_path(models_root / version_name)
    model_dir.mkdir(parents=True, exist_ok=True)

    save_dir = training_info.get("save_dir")
    if not save_dir:
        return None
    weights_dir = Path(save_dir) / "weights"
    best_pt = weights_dir / "best.pt"
    artifacts: List[Path] = []
    if best_pt.exists():
        dest = model_dir / "best.pt"
        shutil.copy2(best_pt, dest)
        artifacts.append(dest)
        try:
            # Torch 2.6+ usa weights_only=True por padrão; forçamos False porque confiamos no checkpoint local.
            state = torch.load(best_pt, map_location="cpu", weights_only=False)
            pkl_path = model_dir / "best.pkl"
            with pkl_path.open("wb") as fp:
                pickle.dump(state, fp)
            artifacts.append(pkl_path)
        except Exception as exc:  # pragma: no cover - best effort
            print(f"[warn] Não foi possível salvar pickle do modelo: {exc}")
        export_formats = list(config.export_formats or [])
        if not export_formats:
            export_formats = ["onnx", "torchscript"]
        yolo_model = YOLO(best_pt)
        export_dir = model_dir / "exports"
        for fmt in export_formats:
            req_error = missing_dependency_message(fmt)
            if req_error:
                print(f"[warn] Pulando exportação {fmt}: {req_error}")
                continue
            try:
                exported = yolo_model.export(
                    format=fmt,
                    imgsz=config.imgsz,
                    device=config.device or "cpu",
                    project=str(export_dir),
                    name=fmt,
                    exist_ok=True,
                )
                artifacts.append(Path(exported))
            except Exception as exc:  # pragma: no cover - best effort
                print(f"[warn] Falha ao exportar formato {fmt}: {exc}")
    last_pt = weights_dir / "last.pt"
    if last_pt.exists():
        dest = model_dir / "last.pt"
        shutil.copy2(last_pt, dest)
        artifacts.append(dest)
    if not artifacts:
        return None
    write_model_artifacts_csv(artifacts, model_dir)
    write_model_summary_txt(model_dir, artifacts)
    return {
        "model_dir": str(model_dir),
        "artifacts": [str(p) for p in artifacts],
    }


def write_summary_json(dataset_dir: Path, summary: Dict[str, object]) -> Path:
    json_path = dataset_dir / "training_summary.json"
    with json_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)
    return json_path


__all__ = [
    "copy_training_reports",
    "export_model_artifacts",
    "write_model_artifacts_csv",
    "write_model_summary_txt",
    "write_summary_json",
]
