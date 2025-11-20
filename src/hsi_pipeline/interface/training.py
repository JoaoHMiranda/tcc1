#!/usr/bin/env python3
"""Entry-point to train YOLO12 using pseudo-RGB datasets."""

from __future__ import annotations

import sys
from pathlib import Path

from .training_parser import build_parser, ROOT
from .training_env import ensure_models_in_train_dir, configure_ultralytics_home
from .training_paths import apply_paths

SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hsi_pipeline.config import load_yolo_training_config_from_json
from hsi_pipeline.data.paths import load_paths_config
from hsi_pipeline.yolo import run_yolo_training


def main(argv: list[str] | None = None):
    configure_ultralytics_home()
    parser = build_parser()
    args = parser.parse_args(argv)
    ensure_models_in_train_dir(ROOT)
    config = load_yolo_training_config_from_json(args.config)
    paths = load_paths_config(args.paths)
    apply_paths(config, paths, args.output_root, args.models_root)
    if not config.enabled:
        print("[skip] Treinamento YOLO desativado via config.enabled=false.")
        return
    try:
        run_yolo_training(config)
    finally:
        ensure_models_in_train_dir(ROOT)


if __name__ == "__main__":
    main()
