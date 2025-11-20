"""End-to-end pipeline orchestration."""

from __future__ import annotations

import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

from ..config import PipelineConfig, VariantOutputSettings
from ..processing.estimation import estimate_global_from_median
from ..features.selection import run_band_selection_pipeline
from ..features.pseudo_rgb_generation import run_pseudo_rgb_stage
from ..preprocessing import (
    load_dataset_resources,
    prepare_kept_bands,
    run_correcao_stage,
    run_snv_stage,
    run_reflectance_msc_stage,
    run_snv_msc_stage,
)
from ..preprocessing.utils import (
    VARIANT_ORDER,
    VARIANT_SUBDIRS,
    prepare_variant_dirs,
    select_median_dir,
    advance_progress,
)

if TYPE_CHECKING:
    from .progress import PipelineProgress



def _initial_circle(
    variant_dirs: Dict[str, Optional[str]],
    preferred: str,
    H: int,
    W: int,
    median_cfg: PipelineConfig,
) -> Tuple[float, float, float]:
    median_dir = select_median_dir(variant_dirs, preferred)
    if median_dir and os.path.isdir(median_dir):
        cxg, cyg, rg = estimate_global_from_median(
            median_dir, H, W, median_cfg.median_guess
        )
    else:
        cxg, cyg, rg = W / 2.0, H / 2.0, min(H, W) * 0.33
    if np.isnan(cxg) or np.isnan(cyg) or np.isnan(rg):
        cxg, cyg, rg = W / 2.0, H / 2.0, min(H, W) * 0.33
    return cxg, cyg, rg


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
    msc_variant_enabled = bool(
        variant_dirs.get("correcao_msc") and variant_settings["correcao_msc"].plot
    )
    total_steps = (
        1  # descoberta
        + Bk  # correcao
        + Bk  # snv
        + (Bk if msc_variant_enabled else 1)  # msc imagens (ou passo único)
        + Bk  # snv_msc
        + 1  # selecao
        + 1  # pseudo
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
    images_per_variant["correcao"] = correcao_result.images_generated

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
    images_per_variant["correcao_snv"] = snv_result.images_generated

    images_per_variant["correcao_msc"] = run_reflectance_msc_stage(
        resources.provider,
        kept,
        wavs_kept,
        delta,
        variant_dirs.get("correcao_msc"),
        variant_settings.get("correcao_msc"),
        snv_result.a_map_reflectance,
        snv_result.b_map_reflectance,
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
    reflectance_cache = correcao_result.reflectance_cache
    snv_cache = snv_result.snv_cache

    timings["geracao_variacoes"] = time.perf_counter() - stage_start

    selected_indices: List[int] = []
    if config.band_selection.enabled:
        cx_sel, cy_sel, rg_sel = _initial_circle(
            variant_dirs, config.median_source_variant, H, W, config
        )
        selection_dir = os.path.join(out_base, "selecao")
        os.makedirs(selection_dir, exist_ok=True)
        selection_summary = run_band_selection_pipeline(
            config=config.band_selection,
            bands=snv_msc_cache if snv_msc_cache else reflectance_cache,
            kept_band_indices=list(range(len(kept))),
            orig_band_indices=kept,
            wavelengths=wavs_kept,
            center=(cx_sel, cy_sel),
            radius=rg_sel,
            out_dir=selection_dir,
            dataset_name=base,
            progress=progress,
        )
        selected_indices = selection_summary.get("selected_band_indices") or []
        selected_indices = sorted({idx for idx in selected_indices if 0 <= idx < len(kept)})
        if selected_indices:
            reflectance_cache = [reflectance_cache[idx] for idx in selected_indices]
            if snv_cache:
                snv_cache = [snv_cache[idx] for idx in selected_indices]
            if snv_msc_cache:
                snv_msc_cache = [snv_msc_cache[idx] for idx in selected_indices]
            kept = [kept[idx] for idx in selected_indices]
            wavs_kept = [wavs_kept[idx] for idx in selected_indices]
            Bk = len(kept)
    else:
        selection_summary = None
    if progress:
        progress.advance("steps")

    pseudo_summary: Optional[Dict[str, object]] = None
    pca_rgb_path: Optional[Path] = None
    if config.pseudo_rgb.enabled:
        pseudo_start = time.perf_counter()
        # Usa o cubo corrigido por SNV+MSC para gerar o pseudo-RGB, mantendo a consistência
        # com as imagens empregadas nas demais etapas (seleção, relatórios etc.).
        pseudo_bands = snv_msc_cache if snv_msc_cache else reflectance_cache
        pseudo_summary = run_pseudo_rgb_stage(
            config=config.pseudo_rgb,
            bands=pseudo_bands,
            kept=kept,
            wavs=wavs_kept,
            out_base=out_base,
        )
        pca_rgb_path = (
            Path(out_base)
            / config.pseudo_rgb.output_dir_name
            / "pca"
            / "pca_rgb.png"
        )
        timings["pseudo_rgb"] = time.perf_counter() - pseudo_start
    else:
        timings["pseudo_rgb"] = 0.0
    if progress:
        progress.advance("steps")

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
        pseudo_summary=pseudo_summary,
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
    pseudo_summary: Optional[Dict[str, object]] = None,
) -> None:
    config_dict = asdict(config)
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
        "pseudo_rgb": "Pseudo-RGB",
        "exportacao_relatorios": "Exportação de relatórios",
        "total": "Tempo total",
    }
    for key in (
        "descoberta_e_memmap",
        "geracao_variacoes",
        "pseudo_rgb",
        "exportacao_relatorios",
        "total",
    ):
        if key in timings:
            lines.append(f"- {stage_labels[key]}: {timings[key]:.2f} s")
    lines.append("")
    if pseudo_summary and config.pseudo_rgb.enabled:
        lines.append("Pseudo-RGB:")
        status = "habilitada" if pseudo_summary.get("enabled") else "desligada"
        lines.append(f"- Status: {status}")
        if pseudo_summary.get("active_method"):
            lines.append(f"- Método ativo: {pseudo_summary['active_method']}")
        if pseudo_summary.get("base_dir"):
            lines.append(f"- Pasta base: {pseudo_summary['base_dir']}")
        for method_info in pseudo_summary.get("methods", []):
            method = method_info.get("method")
            outputs = method_info.get("outputs", [])
            lines.append(f"  - {method}: {len(outputs)} imagem(ns)")
        lines.append("")

    active_variants = [v for v in VARIANT_SUBDIRS if getattr(config.toggles, v, False)]
    if active_variants:
        lines.append("Imagens geradas por variação:")
        for variant in active_variants:
            lines.append(f"- {variant}: {images_per_variant.get(variant, 0)}")
        lines.append("")

    with open(path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines))
