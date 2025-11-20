"""High-level orchestration for YOLO12 training."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from ..config import YoloTrainingConfig
from ..pipeline.progress import PipelineProgress
from .artifacts import export_model_artifacts
from .fs_utils import next_available_path
from .training_steps import (
    DatasetStageResult,
    evaluate_yolo_model,
    finalize_training_reports,
    prepare_dataset_stage,
    train_yolo_model,
)


def run_yolo_training(config: YoloTrainingConfig) -> Dict[str, object]:
    with PipelineProgress() as progress:
        progress.create_task(
            "yolo_main",
            "[magenta]Treinamento YOLO12 (dataset → treino → export → val)",
            4,
        )
        dataset_result = prepare_dataset_stage(config, progress)
        progress.advance("yolo_main")
        dataset_dir = dataset_result.dataset_dir
        data_yaml = dataset_result.data_yaml
        runs_dir = next_available_path(dataset_result.training_root / config.runs_dir)
        runs_dir.mkdir(parents=True, exist_ok=True)

        progress.log("Treinando YOLO12", style="magenta")
        training_info = train_yolo_model(config, data_yaml, runs_dir)
        progress.log("Treino finalizado.", style="magenta")
        progress.advance("yolo_main")

        progress.log("Exportando artefatos", style="yellow")
        model_info = export_model_artifacts(training_info, config)
        progress.log("Exportação concluída.", style="yellow")
        progress.advance("yolo_main")

        eval_info: Dict[str, object] = {}
        if config.run_validation and model_info and model_info.get("model_dir"):
            best_pt = Path(model_info["model_dir"]) / "best.pt"
            last_pt = Path(model_info["model_dir"]) / "last.pt"
            weight_path = best_pt if best_pt.exists() else last_pt
            if weight_path.exists():
                progress.log("Avaliando modelo (val)", style="blue")
                eval_info = evaluate_yolo_model(config, data_yaml, runs_dir, weight_path)
                progress.log("Avaliação concluída.", style="blue")
        progress.complete("yolo_main")
    summary = finalize_training_reports(config, dataset_result, training_info, eval_info, model_info)
    return summary


__all__ = ["run_yolo_training"]
