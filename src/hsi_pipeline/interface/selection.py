#!/usr/bin/env python3
"""Entry-point for running only the band selection stage."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hsi_pipeline.pipeline.cli import resolve_dataset_paths
from hsi_pipeline.config import load_config_from_json
from hsi_pipeline.data.paths import GlobalPathConfig, load_paths_config
from hsi_pipeline.features.selection_runner import run_selection
from hsi_pipeline.pipeline.cpu import configure_cpu_workers

DEFAULT_CONFIG = ROOT / "configs" / "selecao.json"
DEFAULT_PATHS = ROOT / "configs" / "global_paths.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Executa apenas a etapa de seleção/redução de bandas.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="JSON específico da etapa de seleção.")
    parser.add_argument("--paths", default=str(DEFAULT_PATHS), help="JSON com input/output globais.")
    parser.add_argument("--input-root", help="Sobrescreve a entrada do arquivo de paths.")
    parser.add_argument("--output-root", help="Sobrescreve a saída do arquivo de paths.")
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
    configured = configure_cpu_workers(getattr(config, "cpu_workers", None))
    if configured:
        print(f"[info] Limitando threads de CPU para {configured}.")
    paths = load_paths_config(args.paths)
    if not getattr(config, "enabled", True):
        print("[skip] Seleção desativada via config.enabled=false.")
        return
    if not getattr(config.band_selection, "enabled", True):
        print("[skip] band_selection.enabled=false → nenhuma operação realizada.")
        return
    for name, cfg in iter_dataset_configs(config, paths, args.input_root, args.output_root):
        datasets = list(resolve_dataset_paths(cfg.folder))
        label = f"[{name}] " if name else ""
        print(f"{label}Executando seleção em {cfg.folder}")
        run_selection(cfg, datasets)


if __name__ == "__main__":
    main()
