"""Inference orchestration for the classification step."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

from ultralytics import YOLO

from .config import ClassificationConfig
from .datasets import discover_samples
from .reporting import write_csv_reports, write_text_report
try:
    from ..pipeline.progress import PipelineProgress  # type: ignore
except Exception:  # pragma: no cover
    PipelineProgress = None  # type: ignore


@dataclass
class SampleInferenceResult:
    sample_name: str
    summary_csv: Path
    detections_csv: Path
    report_txt: Path


def resolve_cli_path(base: str | Path | None, default: str) -> Path:
    if base is None:
        base = default
    return Path(base).expanduser().resolve()


def run_classification_inference(
    config: ClassificationConfig,
    *,
    model_path: Path | None = None,
    source_root: Path | None = None,
    output_root: Path | None = None,
    imgsz: int | None = None,
    device: str | None = None,
    project_root: Path | None = None,
    progress: "PipelineProgress | None" = None,
) -> List[SampleInferenceResult]:
    project_root = project_root or Path.cwd()
    model = resolve_cli_path(model_path, config.model)
    source = resolve_cli_path(source_root, config.source_root)
    output = resolve_cli_path(output_root, config.output_root)
    output.mkdir(parents=True, exist_ok=True)
    imgsz_value = int(imgsz or config.imgsz)
    device_value = device if device is not None else config.device

    if not os.environ.get("ULTRALYTICS_HOME"):
        os.environ["ULTRALYTICS_HOME"] = str(project_root / "train-yolo")
    if not model.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {model}")

    samples = discover_samples(source)
    total_samples = len(samples)
    if progress:
        progress.log(
            f"Inferindo {total_samples} amostra(s) a partir de {source} -> {output}",
            style="cyan",
        )
        progress.create_task(
            "classification",
            f"[blue]Classificação YOLO em {total_samples} amostra(s)",
            total_samples or 1,
        )
    model_instance = YOLO(model)
    outputs: List[SampleInferenceResult] = []
    for sample_name, files in samples.items():
        sample_out = output / sample_name
        sample_out.mkdir(parents=True, exist_ok=True)
        if progress:
            progress.log(f"Processando {sample_name} ({len(files)} arquivo[s])", style="white")
        results = model_instance.predict(
            source=[str(f) for f in files],
            project=str(sample_out),
            name="pca_inference",
            exist_ok=True,
            imgsz=imgsz_value,
            save_txt=True,
            save_conf=True,
            device=device_value,
            verbose=False,
        )
        summary_csv, details_csv, rows = write_csv_reports(results, sample_out, relative_to=project_root)
        report_txt = write_text_report(summary_csv, details_csv, rows, sample_out, relative_to=project_root)
        outputs.append(
            SampleInferenceResult(
                sample_name=sample_name,
                summary_csv=summary_csv,
                detections_csv=details_csv,
                report_txt=report_txt,
            )
        )
        if progress:
            progress.advance("classification")
    return outputs


__all__ = ["SampleInferenceResult", "run_classification_inference"]
