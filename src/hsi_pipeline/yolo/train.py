"""Training routine for YOLO using correcao_snv_msc datasets."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Tuple

from ultralytics import YOLO

from .config import YoloTrainConfig
from .utils import (
    next_run_name,
    collect_samples,
    copy_split,
    write_data_yaml,
    resolve_model_path,
    enforce_offline_mode,
)


def run_yolo_training(config: YoloTrainConfig) -> Tuple[str, Path]:
    enforce_offline_mode()
    input_root = Path(config.input_root).expanduser().resolve()
    if not input_root.exists():
        raise FileNotFoundError(f"Pasta de entrada não encontrada: {input_root}")
    model_path = resolve_model_path(config.model)
    base_name = model_path.stem
    output_root = Path(config.output_root).expanduser().resolve()
    datasets_root = output_root / "datasets"
    runs_root = output_root / "runs"
    models_root = output_root / "modelos"
    runs_root.mkdir(parents=True, exist_ok=True)
    models_root.mkdir(parents=True, exist_ok=True)

    run_name = next_run_name(models_root, base_name)
    dataset_root = datasets_root / run_name
    dataset_root.mkdir(parents=True, exist_ok=True)

    samples = collect_samples(input_root)
    if len(samples) < 3:
        raise RuntimeError("Esperava ao menos 3 amostras (ATCC) para dividir em train/val/test.")

    splits = {"train": samples[0], "val": samples[1], "test": samples[2]}
    for split, sample in splits.items():
        n = copy_split(sample, split, dataset_root)
        if n == 0:
            raise RuntimeError(f"Nenhuma imagem com label encontrada para {split} em {sample}.")
        print(f"[ok] {split}: {n} imagens copiadas de {sample}")

    data_yaml = write_data_yaml(dataset_root, config.classes)

    model = YOLO(str(model_path))
    args = {
        "data": str(data_yaml),
        "epochs": config.epochs,
        "imgsz": config.imgsz,
        "batch": config.batch,
        "patience": config.patience,
        "device": config.device,
        "project": str(runs_root),
        "name": run_name,
        "exist_ok": False,
    }
    args.update(config.train_extra_args or {})
    print(f"[info] Iniciando treinamento {run_name} com {config.model}")
    model.train(**args)

    best_src = runs_root / run_name / "weights" / "best.pt"
    models_dir = models_root / run_name
    models_dir.mkdir(parents=True, exist_ok=True)
    if best_src.exists():
        shutil.copy2(best_src, models_dir / "best.pt")
        print(f"[ok] best.pt copiado para {models_dir}")
    else:
        print("[warn] best.pt não encontrado após o treino.")

    return run_name, models_dir


__all__ = ["run_yolo_training"]
