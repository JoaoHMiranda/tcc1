"""Parser helpers for the YOLO training CLI."""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "configs" / "yolo_treinamento.json"
DEFAULT_PATHS = ROOT / "configs" / "global_paths.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Treina YOLO12 usando as imagens pseudo-RGB geradas pelo pipeline.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Arquivo JSON com os parâmetros do treinamento.")
    parser.add_argument("--paths", default=str(DEFAULT_PATHS), help="JSON com input/output globais.")
    parser.add_argument("--output-root", help="Sobrescreve a pasta base com os datasets processados (out_root).")
    parser.add_argument("--models-root", help="Sobrescreve a pasta onde os artefatos treinados serão gravados.")
    return parser


__all__ = ["build_parser", "DEFAULT_CONFIG", "DEFAULT_PATHS", "ROOT"]
