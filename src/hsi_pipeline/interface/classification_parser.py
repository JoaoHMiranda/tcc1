"""Argument parser for the classification CLI."""

from __future__ import annotations

import argparse

from hsi_pipeline.classification import DEFAULT_CLASSIFICATION_CONFIG


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Executa inferência no conjunto classificar.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CLASSIFICATION_CONFIG),
        help="JSON com as configurações da inferência (modelo, pastas, imgsz, device).",
    )
    parser.add_argument("--model", help="Sobrescreve o caminho do modelo YOLO12 treinado.")
    parser.add_argument("--source", help="Sobrescreve a raiz do dataset classificar já processado.")
    parser.add_argument("--output", help="Diretório onde salvar os relatórios e predições.")
    parser.add_argument("--imgsz", type=int, help="Resolução de inferência (default vem do JSON).")
    parser.add_argument("--device", help="Dispositivo para inferência (ex.: 0, 0,1 ou cpu).")
    return parser


__all__ = ["build_parser"]
