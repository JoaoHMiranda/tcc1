"""Simple CLI to run the HSI pipeline using a JSON configuration."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Sequence

from ..config import PipelineConfig, load_config_from_json
from .cpu import configure_cpu_workers
from .pipeline import process_folder
from .progress import PipelineProgress

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "preprocessamento.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HSI circle detection pipeline (usa um arquivo de configuração JSON)."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Caminho para o JSON de configuração. Execute sem argumentos para usar o padrão.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> PipelineConfig:
    parser = build_parser()
    args = parser.parse_args(argv)
    return load_config_from_json(args.config)


def has_hdr_files(path: Path) -> bool:
    return any(path.glob("*.hdr"))


def resolve_dataset_paths(folder: str) -> Iterable[Path]:
    root = Path(folder).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Pasta {root} não existe.")
    if root.is_dir() and has_hdr_files(root):
        return [root]
    if root.is_dir():
        children = [
            child for child in sorted(root.iterdir()) if child.is_dir() and has_hdr_files(child)
        ]
        if children:
            return children
    raise RuntimeError(
        f"Não foram encontrados arquivos .hdr em {root} nem em subpastas diretas. "
        "Verifique se o caminho está correto."
    )


def run_preprocess(config: PipelineConfig):
    if not getattr(config, "enabled", True):
        print("[skip] Pré-processamento desativado (config.enabled=false).")
        return
    configured = configure_cpu_workers(getattr(config, "cpu_workers", None))
    if configured:
        print(f"[info] Limitando threads de CPU para {configured}.")
    datasets = list(resolve_dataset_paths(config.folder))
    with PipelineProgress() as progress:
        progress.create_task(
            "preprocess",
            f"[cyan]Pré-processando {len(datasets)} conjunto(s)",
            len(datasets),
        )
        for idx, dataset in enumerate(datasets, start=1):
            progress.start_dataset(dataset.name, idx, len(datasets))
            current_cfg = replace(config, folder=str(dataset))
            progress.log(f"Iniciando processamento em {dataset}", style="yellow")
            process_folder(current_cfg, progress=progress)
            progress.advance("preprocess")


def main(argv: Sequence[str] | None = None):
    config = parse_args(argv)
    run_preprocess(config)


if __name__ == "__main__":
    main()
