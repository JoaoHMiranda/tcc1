#!/usr/bin/env python3
"""Roda inferência em datasets `classificar` usando um modelo YOLO12 treinado."""

from __future__ import annotations

from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[3]

from hsi_pipeline.classification import (
    load_classification_config,
    run_classification_inference,
)
from .classification_parser import build_parser
from .classification_paths import resolve_override


def main(argv: List[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = load_classification_config(args.config)
    results = run_classification_inference(
        cfg,
        model_path=resolve_override(args.model),
        source_root=resolve_override(args.source),
        output_root=resolve_override(args.output),
        imgsz=args.imgsz,
        device=args.device,
        project_root=ROOT,
    )
    print("[ok] Inferência concluída.")
    for res in results:
        print(f"[ok] {res.sample_name}: {res.summary_csv}")


if __name__ == "__main__":
    main()
