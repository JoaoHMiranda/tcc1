"""Run YOLO predictions on correcao_snv_msc images and produce CSV/TXT reports."""

from __future__ import annotations

import csv
import json
import os
import time
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from ultralytics import YOLO

from .config import ClassificationConfig
from ..yolo.utils import next_run_name, resolve_model_path, enforce_offline_mode


def _discover_samples(source_root: Path) -> List[Path]:
    samples: List[Path] = []
    for child in sorted(source_root.iterdir()):
        rgb_dir = child / "correcao_snv_msc_rgb_bands"
        if rgb_dir.is_dir():
            samples.append(child)
    return samples


def _prepare_run_dir(output_root: Path, model_path: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    base_name = model_path.stem
    run_name = next_run_name(output_root, base_name)
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _save_summary_txt(path: Path, summary: Dict[str, object]) -> None:
    # montar lista de amostras com prob média e máximo
    sample_lines: List[str] = []
    for sample in summary.get("samples", []):
        sample_lines.append(
            f"- {sample['sample']}: imagens={sample['images']} detecoes={sample.get('detections', 0)} "
            f"imgs_com_det={sample.get('images_with_detection', 0)} "
            f"prob_med={sample['prob_mean']:.3f} prob_max={sample['prob_max']:.3f} "
            f"tempo_s={sample.get('duration_sec', 0):.2f}"
        )
    if not sample_lines:
        sample_lines.append("- (nenhuma amostra encontrada)")

    lines = [
        f"Modelo: {summary['model']}",
        f"Fonte: {summary['source_root']}",
        f"Run: {summary['run_dir']}",
        f"Início: {summary.get('started_at', '')}",
        f"Término: {summary.get('ended_at', '')}",
        f"Duração (s): {summary.get('duration_sec', 0):.2f}",
        f"Total de imagens: {summary['total_images']}",
        f"Imagens com detecção: {summary['images_with_detection']}",
        f"Probabilidade média (staphylococcus): {summary['prob_mean']:.3f}",
        f"Probabilidade mínima: {summary['prob_min']:.3f}",
        f"Probabilidade máxima: {summary['prob_max']:.3f}",
        "",
        "Por amostra:",
        *sample_lines,
        "",
        "Nota: probabilidade = maior confiança de detecção da classe staphylococcus em cada imagem.",
        f"Resumo por amostra: {summary.get('sample_summary_csv', '')}",
        f"Detecções detalhadas: {summary.get('detections_csv', '')}",
        f"Probabilidades por imagem: {summary.get('summary_csv', '')}",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_classification(config: ClassificationConfig) -> Tuple[str, Path]:
    enforce_offline_mode()
    started_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    t0 = time.perf_counter()
    model_path = resolve_model_path(config.model)
    source_root = Path(config.source_root).expanduser().resolve()
    output_root = Path(config.output_root).expanduser().resolve()

    samples = _discover_samples(source_root)
    if not samples:
        raise RuntimeError(f"Nenhuma amostra encontrada em {source_root} com correcao_snv_msc_rgb_bands.")

    run_dir = _prepare_run_dir(output_root, model_path)
    model = YOLO(str(model_path))
    class_names = model.names or {}

    detection_rows: List[Dict[str, object]] = []
    prob_rows: List[Dict[str, object]] = []
    sample_image_counts: Dict[str, int] = {}
    sample_image_with_det: Dict[str, int] = {}
    sample_det_counts: Dict[str, int] = {}
    sample_start_ts: Dict[str, str] = {}
    sample_duration: Dict[str, float] = {}

    total_images = 0
    images_with_det = 0

    for sample in samples:
        sample_start = time.perf_counter()
        sample_start_ts[sample.name] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        images_dir = sample / "correcao_snv_msc_rgb_bands"
        image_paths = sorted(images_dir.glob("*.png"))
        if not image_paths:
            continue
        sample_image_counts[sample.name] = len(image_paths)
        print(f"[info] Classificando {len(image_paths)} imagens de {sample.name}")
        results = model.predict(
            source=[str(p) for p in image_paths],
            imgsz=config.imgsz,
            device=config.device,
            conf=config.conf,
            project=str(run_dir),
            name=sample.name,
            exist_ok=False,
            save=True,
            save_txt=True,
        )
        for res in results:
            total_images += 1
            img_path = Path(res.path)
            sample_name = sample.name
            img_name = img_path.name
            boxes = res.boxes
            staph_probs: List[float] = []
            if boxes is not None and boxes.xyxy is not None:
                for xyxy, cls_tensor, conf_tensor in zip(boxes.xyxy, boxes.cls, boxes.conf):
                    cls_id = int(cls_tensor)
                    conf = float(conf_tensor)
                    cls_name = class_names.get(cls_id, str(cls_id))
                    if cls_name.lower().startswith("staph") or cls_id == 0:
                        staph_probs.append(conf)
                        sample_det_counts[sample_name] = sample_det_counts.get(sample_name, 0) + 1
                    detection_rows.append(
                        {
                            "sample": sample_name,
                            "image": img_name,
                            "class_id": cls_id,
                            "class_name": cls_name,
                            "confidence": conf,
                            "xmin": float(xyxy[0]),
                            "ymin": float(xyxy[1]),
                            "xmax": float(xyxy[2]),
                            "ymax": float(xyxy[3]),
                        }
                    )
            prob = max(staph_probs) if staph_probs else 0.0
            if prob > 0:
                images_with_det += 1
                sample_image_with_det[sample_name] = sample_image_with_det.get(sample_name, 0) + 1
            prob_rows.append(
                {
                    "sample": sample_name,
                    "image": img_name,
                    "staphylococcus_prob": prob,
                }
            )
        # copiar imagens anotadas para pasta agregada
        sample_output_dir = run_dir / sample.name
        if sample_output_dir.exists():
            dest_dir = run_dir / "imagens" / sample.name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for img in sample_output_dir.glob("*.jpg"):
                shutil.copy2(img, dest_dir / img.name)
        sample_duration[sample.name] = time.perf_counter() - sample_start

    detections_path = run_dir / "detections.csv"
    if detection_rows:
        pd.DataFrame(detection_rows).to_csv(detections_path, index=False)
    else:
        # create empty with headers
        with detections_path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(
                fp,
                fieldnames=["sample", "image", "class_id", "class_name", "confidence", "xmin", "ymin", "xmax", "ymax"],
            )
            writer.writeheader()

    summary_df = pd.DataFrame(prob_rows)
    summary_path = run_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)

    # agregados por amostra
    sample_summary = pd.DataFrame(prob_rows)
    if not sample_summary.empty:
        sample_summary = (
            sample_summary.groupby("sample")["staphylococcus_prob"]
            .agg(["count", "mean", "min", "max"])
            .reset_index()
            .rename(
                columns={
                    "count": "images",
                    "mean": "prob_mean",
                    "min": "prob_min",
                    "max": "prob_max",
                }
            )
        )
    else:
        sample_summary = pd.DataFrame(columns=["sample", "images", "prob_mean", "prob_min", "prob_max"])
    # anexar métricas extras por amostra
    sample_summary["detections"] = sample_summary["sample"].map(sample_det_counts).fillna(0).astype(int)
    sample_summary["images_with_detection"] = sample_summary["sample"].map(sample_image_with_det).fillna(0).astype(int)
    sample_summary["duration_sec"] = sample_summary["sample"].map(sample_duration).fillna(0.0).astype(float)
    sample_summary_path = run_dir / "sample_summary.csv"
    sample_summary.to_csv(sample_summary_path, index=False)

    probs = summary_df["staphylococcus_prob"].to_numpy() if not summary_df.empty else np.array([0.0])
    sample_records = sample_summary.to_dict(orient="records")
    summary = {
        "model": str(model_path),
        "source_root": str(source_root),
        "run_dir": str(run_dir),
        "total_images": int(total_images),
        "images_with_detection": int(images_with_det),
        "prob_mean": float(np.mean(probs)),
        "prob_min": float(np.min(probs)),
        "prob_max": float(np.max(probs)),
        "sample_summary_csv": str(sample_summary_path),
        "detections_csv": str(detections_path),
        "summary_csv": str(summary_path),
        "samples": sample_records,
        "started_at": started_at,
        "ended_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "duration_sec": float(time.perf_counter() - t0),
    }
    summary_json = run_dir / "summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _save_summary_txt(run_dir / "summary.txt", summary)

    return summary["run_dir"], run_dir


__all__ = ["run_classification"]
