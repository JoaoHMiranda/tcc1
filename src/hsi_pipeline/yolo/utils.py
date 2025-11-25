"""Shared helpers for YOLO workflows."""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import List


def next_run_name(root: Path, base_name: str) -> str:
    pattern = re.compile(rf"^{re.escape(base_name)}_(\d+)$")
    existing: List[int] = []
    if root.exists():
        for child in root.iterdir():
            if child.is_dir():
                m = pattern.match(child.name)
                if m:
                    existing.append(int(m.group(1)))
    next_idx = max(existing) + 1 if existing else 1
    return f"{base_name}_{next_idx}"


def collect_samples(input_root: Path) -> List[Path]:
    samples = []
    for child in sorted(input_root.iterdir()):
        if not child.is_dir():
            continue
        rgb_dir = child / "correcao_snv_msc_rgb_bands"
        if rgb_dir.is_dir():
            samples.append(child)
    return samples


def copy_split(sample: Path, split: str, dataset_root: Path) -> int:
    images_dir = dataset_root / "images" / split
    labels_dir = dataset_root / "labels" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir = sample / "correcao_snv_msc_rgb_bands"
    count = 0
    for img_path in sorted(rgb_dir.glob("*.png")):
        label_path = img_path.with_suffix(".txt")
        out_img = images_dir / img_path.name
        out_lbl = labels_dir / label_path.name
        shutil.copy2(img_path, out_img)
        if label_path.exists():
            shutil.copy2(label_path, out_lbl)
        else:
            # cria label vazio para permitir treino mesmo sem anotações
            out_lbl.write_text("", encoding="utf-8")
            print(f"[warn:{split}] label ausente, criando vazio: {out_lbl}")
        count += 1
    return count


def write_data_yaml(dataset_root: Path, classes: List[str]) -> Path:
    yaml_path = dataset_root / "data.yaml"
    names_block = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(classes))
    yaml_content = (
        f"path: {dataset_root}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"nc: {len(classes)}\n"
        "names:\n"
        f"{names_block}\n"
    )
    yaml_path.write_text(yaml_content, encoding="utf-8")
    return yaml_path


def enforce_offline_mode() -> None:
    """Force Ultralytics/YOLO to run offline (no downloads)."""
    os.environ["YOLO_OFFLINE"] = "1"
    os.environ["ULTRALYTICS_OFFLINE"] = "1"
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["HF_HUB_OFFLINE"] = "1"


def find_model_by_name(name: str) -> Path:
    """Search for a model file under yolo/modelos matching the given name."""
    roots = [
        Path("yolo/modelos").expanduser().resolve(),
        Path("src/hsi_pipeline/yolo/modelos").expanduser().resolve(),
    ]
    for modelos_root in roots:
        direct = modelos_root / name
        if direct.exists():
            return direct
        for p in modelos_root.rglob(name):
            if p.is_file():
                return p
    raise FileNotFoundError(f"Modelo não encontrado: {name} em {', '.join(str(r) for r in roots)}")


def resolve_latest_model() -> Path:
    """Pick the newest best.pt in yolo/modelos."""
    modelos_root = Path("yolo/modelos").expanduser().resolve()
    candidates = []
    for p in modelos_root.rglob("best.pt"):
        try:
            mtime = p.stat().st_mtime
        except OSError:
            continue
        candidates.append((mtime, p))
    if not candidates:
        raise FileNotFoundError(f"Nenhum modelo encontrado em {modelos_root}")
    candidates.sort(reverse=True)
    return candidates[0][1]


def resolve_model_path(model_str: str) -> Path:
    """Resolve model by path, name under yolo/modelos, or 'auto' (newest best.pt)."""
    if model_str == "auto":
        return resolve_latest_model()
    path = Path(model_str)
    if path.exists():
        return path.expanduser().resolve()
    return find_model_by_name(model_str)


def resolve_dataset_for_model(data_yaml_str: str, model_path: Path) -> Path:
    """Resolve dataset path. If 'auto', match the dataset folder to the model run name."""
    if data_yaml_str != "auto":
        resolved = Path(data_yaml_str).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"data.yaml não encontrado: {resolved}")
        return resolved
    run_name = model_path.parent.name
    dataset_yaml = Path("yolo/datasets") / run_name / "data.yaml"
    dataset_yaml = dataset_yaml.expanduser().resolve()
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"data.yaml não encontrado para o modelo {model_path} em {dataset_yaml}")
    return dataset_yaml


__all__ = [
    "next_run_name",
    "collect_samples",
    "copy_split",
    "write_data_yaml",
    "find_model_by_name",
    "resolve_latest_model",
    "resolve_model_path",
    "resolve_dataset_for_model",
]
