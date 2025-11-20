"""Dataset preparation utilities for YOLO12 training."""

from __future__ import annotations

import csv
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..config import YoloTrainingConfig
from . import image_filters


def resolve_path(path: str | os.PathLike[str] | None) -> Optional[Path]:
    if path is None:
        return None
    return Path(path).expanduser().resolve()


def discover_datasets(out_root: Path, config: YoloTrainingConfig) -> List[Path]:
    candidates = [d for d in sorted(out_root.iterdir()) if d.is_dir()]
    if config.datasets:
        target = set(config.datasets)

        # Se o próprio diretório de saída for um dos alvos, listen seus subdiretórios
        if len(target) == 1 and out_root.name in target:
            return candidates

        filtered = [d for d in candidates if d.name in target]
        missing = target.difference({d.name for d in filtered})
        if missing:
            raise FileNotFoundError(f"Datasets não encontrados em {out_root}: {sorted(missing)}")
        return filtered
    return candidates


def find_label(
    image_path: Path,
    dataset_dir: Path,
    config: YoloTrainingConfig,
    pseudo_root: Path,
    annotations_root: Optional[Path],
) -> Optional[Path]:
    if config.labels_follow_images:
        candidate = image_path.with_suffix(config.label_extension)
        if candidate.exists():
            return candidate
    if annotations_root:
        rel = image_path.relative_to(pseudo_root)
        search_roots = [
            annotations_root / dataset_dir.parent.name,
            annotations_root / dataset_dir.name,
            annotations_root,
        ]
        for root in search_roots:
            candidate = root / rel
            candidate = candidate.with_suffix(config.label_extension)
            if candidate.exists():
                return candidate
    return None


def collect_records(
    out_root: Path,
    datasets: Sequence[Path],
    config: YoloTrainingConfig,
) -> List[Dict[str, object]]:
    annotations_root = resolve_path(config.annotations_root)
    records: List[Dict[str, object]] = []
    missing_labels: List[Path] = []
    for dataset_dir in datasets:
        pseudo_root = dataset_dir / config.pseudo_root
        method_dirs: List[tuple[Path, Path, Optional[str]]] = []
        # Apenas um modo ativo: PCA puro.
        method_dirs.append((pseudo_root / config.pseudo_method, pseudo_root, None))
        seen_paths = set()
        for method_dir, method_root, forced_split in method_dirs:
            if not method_dir.exists():
                continue
            for image_path in sorted(method_dir.rglob("*.png")):
                if not image_path.is_file():
                    continue
                if image_path in seen_paths:
                    continue
                seen_paths.add(image_path)
                label_path = find_label(image_path, dataset_dir, config, method_root, annotations_root)
                if label_path is None:
                    if config.missing_label_policy == "skip":
                        continue
                    missing_labels.append(image_path)
                    continue
                records.append(
                    {
                        "image": image_path,
                        "label": label_path,
                        "dataset": dataset_dir.name,
                        "force_split": forced_split,
                    }
                )
    if missing_labels:
        lookup_hint = f"procure no diretório das imagens ou em {annotations_root}" if annotations_root else "adicione as labels ao lado das imagens"
        sample = missing_labels[0]
        raise FileNotFoundError(
            f"{len(missing_labels)} label(s) não encontrada(s); exemplo: {sample}. ({lookup_hint})"
        )
    return records


def split_records(records: Sequence[Dict[str, Path]], config: YoloTrainingConfig) -> Dict[str, List[Dict[str, Path]]]:
    if not records:
        raise RuntimeError("Nenhuma imagem pseudo-RGB com rótulo foi encontrada para o treinamento YOLO.")
    fractions = np.array([config.split.train, config.split.val, config.split.test], dtype=float)
    total = fractions.sum()
    fractions = fractions / total if total > 0 else np.array([1.0, 0.0, 0.0])
    rng = np.random.default_rng(config.random_state)

    forced: Dict[str, List[Dict[str, Path]]] = {"train": [], "val": [], "test": []}
    leftover: List[Dict[str, Path]] = []
    for rec in records:
        forced_split = rec.get("force_split")
        if forced_split in forced:
            forced[forced_split].append(rec)
        else:
            leftover.append(rec)

    shuffled = list(leftover)
    rng.shuffle(shuffled)
    n_total = len(shuffled)
    n_train = max(0, int(round(n_total * fractions[0])))
    n_val = max(0, int(round(n_total * fractions[1])))
    if n_train + n_val > n_total:
        n_val = max(0, n_total - n_train)
    n_test = n_total - n_train - n_val

    train_records = forced["train"] + shuffled[:n_train]
    val_records = forced["val"] + shuffled[n_train : n_train + n_val]
    test_records = forced["test"] + shuffled[n_train + n_val : n_train + n_val + n_test]

    # Se ainda não houver treino, faça fallback para não ficar sem base.
    if not train_records:
        if val_records:
            train_records, val_records = val_records, []
        elif test_records:
            train_records, test_records = test_records, []
    # Garante que exista pelo menos uma imagem de validação quando houver dados,
    # evitando falha do Ultralytics ao carregar o split val vazio.
    # Aqui não removemos itens de train/test, apenas duplicamos um registro para val.
    if not val_records and (train_records or test_records):
        candidate = train_records[0] if train_records else test_records[0]
        val_records.append(candidate)
    return {"train": train_records, "val": val_records, "test": test_records}


def prepare_dataset_dirs(base_dir: Path, clean: bool) -> Dict[str, Path]:
    if clean and base_dir.exists():
        shutil.rmtree(base_dir)
    (base_dir / "images").mkdir(parents=True, exist_ok=True)
    (base_dir / "labels").mkdir(parents=True, exist_ok=True)
    mapping: Dict[str, Path] = {}
    for split in ("train", "val", "test"):
        mapping[f"images_{split}"] = base_dir / "images" / split
        mapping[f"labels_{split}"] = base_dir / "labels" / split
        mapping[f"images_{split}"].mkdir(parents=True, exist_ok=True)
        mapping[f"labels_{split}"].mkdir(parents=True, exist_ok=True)
    return mapping


def materialize_dataset(
    records_split: Dict[str, List[Dict[str, object]]],
    dataset_dir: Path,
    clean: bool,
    label_extension: str,
    enhance_images: bool = False,
    progress=None,
    task_id=None,
) -> Dict[str, int]:
    dirs = prepare_dataset_dirs(dataset_dir, clean)
    counts: Dict[str, int] = {}
    for split, recs in records_split.items():
        count = 0
        for record in recs:
            image_stem = record["image"].stem
            dataset_prefix = record.get("dataset") or ""
            unique_stem = f"{dataset_prefix}_{image_stem}" if dataset_prefix else image_stem
            image_target = dirs[f"images_{split}"] / f"{unique_stem}{record['image'].suffix}"
            label_name = f"{unique_stem}{label_extension}"
            if enhance_images:
                image_filters.enhance_pca_rgb(record["image"], image_target)
            else:
                shutil.copy2(record["image"], image_target)
            shutil.copy2(record["label"], dirs[f"labels_{split}"] / label_name)
            count += 1
            if progress is not None and task_id is not None:
                progress.advance(task_id)
        counts[split] = count
    return counts


def write_data_yaml(dataset_dir: Path, classes: Sequence[str]) -> Path:
    yaml_path = dataset_dir / "data.yaml"
    with yaml_path.open("w", encoding="utf-8") as fp:
        fp.write(f"path: {dataset_dir.resolve()!s}\n")
        fp.write("train: images/train\n")
        fp.write("val: images/val\n")
        fp.write("test: images/test\n")
        fp.write("names:\n")
        for idx, name in enumerate(classes):
            fp.write(f"  {idx}: {name}\n")
    return yaml_path


def write_dataset_reports(
    records_split: Dict[str, List[Dict[str, Path]]],
    counts: Dict[str, int],
    dataset_dir: Path,
) -> Dict[str, str]:
    dataset_index = dataset_dir / "dataset_index.csv"
    with dataset_index.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["split", "dataset", "image", "label"])
        writer.writeheader()
        for split, recs in records_split.items():
            for record in recs:
                writer.writerow(
                    {
                        "split": split,
                        "dataset": record["dataset"],
                        "image": str(record["image"]),
                        "label": str(record["label"]),
                    }
                )
    counts_csv = dataset_dir / "dataset_split_counts.csv"
    with counts_csv.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["split", "count"])
        for split, value in counts.items():
            writer.writerow([split, value])
    counts_txt = dataset_dir / "dataset_split_counts.txt"
    with counts_txt.open("w", encoding="utf-8") as fp:
        fp.write("Contagem de imagens por partição:\n")
        for split, value in counts.items():
            fp.write(f"- {split}: {value}\n")
    return {
        "dataset_index_csv": str(dataset_index),
        "counts_csv": str(counts_csv),
        "counts_txt": str(counts_txt),
    }


__all__ = [
    "resolve_path",
    "discover_datasets",
    "collect_records",
    "split_records",
    "materialize_dataset",
    "write_data_yaml",
    "write_dataset_reports",
]
