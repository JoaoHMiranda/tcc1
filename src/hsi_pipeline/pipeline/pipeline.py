"""End-to-end pipeline orchestration."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

import pandas as pd

from ..config import PipelineConfig, VariantOutputSettings
from ..preprocessing import (
    load_dataset_resources,
    prepare_kept_bands,
    run_correcao_stage,
    run_snv_stage,
    run_snv_msc_stage,
)
from ..preprocessing.utils import (
    VARIANT_ORDER,
    VARIANT_SUBDIRS,
    prepare_variant_dirs,
    advance_progress,
)

if TYPE_CHECKING:
    from .progress import PipelineProgress


def _ensure_full_image_labels(variant_dir: Optional[str]) -> int:
    """Create full-image YOLO labels (class 0) for each PNG if missing."""
    if not variant_dir:
        return 0
    created = 0
    for png in Path(variant_dir).glob("*.png"):
        txt = png.with_suffix(".txt")
        if txt.exists():
            continue
        txt.write_text("0 0.5 0.5 1.0 1.0\n", encoding="utf-8")
        created += 1
    return created


def process_folder(config: PipelineConfig, progress: Optional["PipelineProgress"] = None) -> str:
    timings: Dict[str, float] = {}
    images_per_variant = {variant: 0 for variant in VARIANT_ORDER}
    total_start = time.perf_counter()
    if progress:
        progress.log("Descobrindo arquivos ENVI...", style="cyan")

    resources, discovery_time = load_dataset_resources(config)
    timings["descoberta_e_memmap"] = discovery_time
    folder = resources.folder
    out_base = resources.out_base
    base = resources.base_name
    H, W, B = resources.height, resources.width, resources.total_bands

    variant_dirs = prepare_variant_dirs(out_base, config.toggles)
    variant_settings = {}
    for key in VARIANT_ORDER:
        setting = getattr(config.toggles, key, None)
        if setting is None:
            setting = VariantOutputSettings()
        variant_settings[key] = setting

    kept, wavs_kept = prepare_kept_bands(resources, config.trimming)
    Bk = len(kept)

    if progress:
        progress.log(f"Dataset preparado com {B} bandas (mantendo {Bk}).", style="cyan")

    delta = max(0, int(config.delta_bands))
    active_variants = [
        key for key in VARIANT_ORDER if variant_dirs[key] and variant_settings[key].plot
    ]
    total_steps = (
        1  # descoberta
        + Bk  # correcao
        + Bk  # snv
        + Bk  # snv_msc
        + 1  # relatorios
    )
    if progress:
        progress.create_task("steps", "[cyan]Pré-processamento", total_steps)
        progress.advance("steps")  # descoberta já concluída
    if progress and active_variants:
        progress.create_task(
            "variants",
            f"[cyan]Gerando composições RGB ({len(active_variants)} variações)",
            len(active_variants) * Bk,
        )

    stage_start = time.perf_counter()

    correcao_result = run_correcao_stage(
        resources.provider,
        kept,
        wavs_kept,
        delta,
        variant_dirs.get("correcao"),
        variant_settings.get("correcao"),
        progress,
        step_task="steps",
    )
    snv_result = run_snv_stage(
        correcao_result.reflectance_cache,
        correcao_result.mean_px,
        correcao_result.std_px,
        correcao_result.band_sums,
        kept,
        wavs_kept,
        delta,
        variant_dirs.get("correcao_snv"),
        variant_settings.get("correcao_snv"),
        progress,
        step_task="steps",
    )
    msc_result = run_snv_msc_stage(
        snv_result.snv_cache,
        snv_result.snv_band_sums,
        kept,
        wavs_kept,
        delta,
        variant_dirs.get("correcao_snv_msc"),
        variant_settings.get("correcao_snv_msc"),
        progress,
        step_task="steps",
    )
    snv_msc_cache = msc_result.snv_msc_cache
    images_per_variant["correcao_snv_msc"] = msc_result.images_snv_msc
    labels_created = _ensure_full_image_labels(variant_dirs.get("correcao_snv_msc"))

    timings["geracao_variacoes"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()

    band_indices = list(range(Bk))
    metadata = {
        "band_index_kept": band_indices,
        "orig_band_index": kept,
        "wavelength_nm": wavs_kept,
    }
    metadata_df = pd.DataFrame(metadata)

    export_steps = 3  # metadata CSV, image report CSV, summary TXT
    if progress:
        progress.create_task("exports", "[green]Salvando relatórios", export_steps)

    # band_metadata.csv
    if config.toggles.export_metadata:
        metadata_df.to_csv(os.path.join(out_base, "band_metadata.csv"), index=False)
    advance_progress(progress, "exports")

    # image_report.csv
    write_image_report(
        out_base=out_base,
        dataset=base,
        metadata_df=metadata_df,
    )
    advance_progress(progress, "exports")

    timings["exportacao_relatorios"] = time.perf_counter() - stage_start
    timings["total"] = time.perf_counter() - total_start

    write_summary_txt(
        path=os.path.join(out_base, f"{base}_summary.txt"),
        dataset=base,
        folder=folder,
        out_base=out_base,
        config=config,
        timings=timings,
        images_per_variant=images_per_variant,
    )
    advance_progress(progress, "exports")

    if progress:
        progress.advance("steps")
        progress.log(
            f"Concluído: H={H} W={W} B_total={B} → B_kept={Bk} | saída: {out_base}",
            style="green",
        )
    else:
        print(
            f"[OK] processado: H={H} W={W} B_total={B} → B_kept={Bk} "
            f"(trim {config.trimming.left}/{config.trimming.right})"
        )
        print(f"Saída base: {out_base}")
    return out_base


def write_image_report(
    out_base: str,
    dataset: str,
    metadata_df: pd.DataFrame,
) -> None:
    rows: List[Dict[str, object]] = []
    has_orig = "orig_band_index" in metadata_df.columns
    for idx in metadata_df.index:
        row = {
            "dataset": dataset,
            "band_index_kept": metadata_df.loc[idx, "band_index_kept"],
            "wavelength_nm": metadata_df.loc[idx, "wavelength_nm"],
        }
        if has_orig:
            row["orig_band_index"] = metadata_df.loc[idx, "orig_band_index"]
        rows.append(row)
    output_path = os.path.join(out_base, f"{dataset}_image_report.csv")
    pd.DataFrame(rows).to_csv(output_path, index=False)


def write_summary_txt(
    path: str,
    dataset: str,
    folder: str,
    out_base: str,
    config: PipelineConfig,
    timings: Dict[str, float],
    images_per_variant: Dict[str, int],
) -> None:
    lines = [
        f"Dataset: {dataset}",
        f"Pasta origem: {folder}",
        f"Saída: {out_base}",
        "",
        "Tempo por etapa:",
    ]
    stage_labels = {
        "descoberta_e_memmap": "Descoberta e memmap",
        "geracao_variacoes": "Geração de variações",
        "exportacao_relatorios": "Exportação de relatórios",
        "total": "Tempo total",
    }
    for key in (
        "descoberta_e_memmap",
        "geracao_variacoes",
        "exportacao_relatorios",
        "total",
    ):
        if key in timings:
            lines.append(f"- {stage_labels[key]}: {timings[key]:.2f} s")
    lines.append("")

    active_variants = [v for v in VARIANT_SUBDIRS if getattr(config.toggles, v, False)]
    if active_variants:
        lines.append("Imagens geradas por variação:")
        for variant in active_variants:
            lines.append(f"- {variant}: {images_per_variant.get(variant, 0)}")
        lines.append("")

    with open(path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines))
