#!/usr/bin/env python3
"""Entry-point for running the preprocessing stage."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hsi_pipeline.pipeline.cli import run_preprocess
from hsi_pipeline.config import load_config_from_json
from hsi_pipeline.data.paths import GlobalPathConfig, load_paths_config


DEFAULT_CONFIG = ROOT / "configs" / "preprocessamento.json"
DEFAULT_PATHS = ROOT / "configs" / "global_paths.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Executa o pipeline de pré-processamento.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="JSON com os parâmetros da etapa.")
    parser.add_argument(
        "--paths",
        default=str(DEFAULT_PATHS),
        help="JSON com os diretórios globais de entrada/saída.",
    )
    parser.add_argument("--input-root", help="Sobrescreve a pasta de entrada do JSON global.")
    parser.add_argument("--output-root", help="Sobrescreve a pasta de saída do JSON global.")
    return parser


def iter_dataset_configs(
    base_config,
    paths: GlobalPathConfig,
    input_override: str | None,
    output_override: str | None,
):
    if input_override or output_override:
        cfg = replace(base_config)
        cfg.folder = input_override or cfg.folder or paths.input_root
        cfg.out_root = output_override or cfg.out_root or paths.output_root
        if not cfg.folder:
            raise ValueError("Nenhum diretório de entrada definido (use --input-root ou global_paths.json).")
        yield None, cfg
        return
    filters = getattr(base_config, "dataset_filters", {}) or {}

    def is_enabled(name: Optional[str]) -> bool:
        if name is None:
            return True
        if name in filters:
            return bool(filters[name])
        return bool(filters.get("default", True))

    if paths.dataset_pairs:
        for pair in paths.dataset_pairs:
            if not pair.input_root:
                continue
            cfg = replace(base_config)
            cfg.folder = pair.input_root
            cfg.out_root = pair.output_root or cfg.out_root or paths.output_root
            if not cfg.folder:
                raise ValueError("Par de dataset sem 'input_root'.")
            if is_enabled(pair.name):
                yield pair.name, cfg
        return
    cfg = replace(base_config)
    cfg.folder = cfg.folder or paths.input_root
    cfg.out_root = cfg.out_root or paths.output_root
    if not cfg.folder:
        raise ValueError("Nenhum diretório de entrada definido (use --input-root ou global_paths.json).")
    if is_enabled(None):
        yield None, cfg


def main(argv: list[str] | None = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config_from_json(args.config)
    paths = load_paths_config(args.paths)
    if not getattr(config, "enabled", True):
        print("[skip] Pré-processamento desativado via config.enabled=false.")
        return
    for name, cfg in iter_dataset_configs(config, paths, args.input_root, args.output_root):
        label = f"[{name}] " if name else ""
        print(f"{label}Iniciando pré-processamento em {cfg.folder}")
        run_preprocess(cfg)


if __name__ == "__main__":
    main()
