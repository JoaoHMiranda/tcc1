#!/usr/bin/env python3
"""CLI wrapper for YOLO evaluation."""

import os
# Force offline before any Ultralytics import
os.environ["YOLO_OFFLINE"] = "1"
os.environ["ULTRALYTICS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["WANDB_DISABLED"] = "true"

from pathlib import Path
import sys
import argparse

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hsi_pipeline.yolo.config import load_yolo_eval_config
from hsi_pipeline.yolo.eval import run_yolo_evaluation
from hsi_pipeline.yolo.utils import enforce_offline_mode

DEFAULT_CONFIG = ROOT / "configs" / "yolo_avaliacao.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Avalia um modelo YOLO nas imagens correcao_snv_msc.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="JSON de configuração da avaliação.")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    enforce_offline_mode()
    cfg = load_yolo_eval_config(args.config)
    run_yolo_evaluation(cfg)


if __name__ == "__main__":
    main()
