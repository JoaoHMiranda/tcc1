#!/usr/bin/env python3
"""Restaura labels básicas para as amostras de doentes (pca_rgb) e regenera augments."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[3]
DOENTES_ROOT = ROOT / "hsi_modificado" / "doentes"
# Label aproximada (classe 0) por amostra, em formato YOLO (xc, yc, w, h), normalizado.
LABELS: Dict[str, Tuple[float, float, float, float]] = {
    "ATCC13_240506-161053": (0.450000, 0.558621, 0.562500, 0.620690),
    "ATCC16_240506-161158": (0.412500, 0.553633, 0.612500, 0.678201),
    "ATCC27_240506-161129": (0.431250, 0.568493, 0.568750, 0.623288),
}


def write_label(path: Path, label: Tuple[float, float, float, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        fp.write(f"0 {label[0]:.6f} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f}\n")


def restore_doentes_labels(
    doentes_root: Path | str | None = None,
    *,
    overwrite: bool = False,
    active_samples: Sequence[str] | None = None,
) -> Tuple[List[Path], List[Path]]:
    """Cria labels YOLO padrão para as capturas de 'doentes'.

    Retorna:
        Tuple[List[Path], List[Path]]: caminhos criados e caminhos já existentes.
    """
    root_path = Path(doentes_root) if doentes_root is not None else DOENTES_ROOT
    created: List[Path] = []
    skipped: List[Path] = []
    target_samples = set(active_samples or LABELS.keys())
    for name, bbox in LABELS.items():
        if name not in target_samples:
            continue
        p_label = root_path / name / "pseudo_rgb" / "pca" / "pca_rgb.txt"
        if p_label.exists() and not overwrite:
            skipped.append(p_label)
            continue
        write_label(p_label, bbox)
        created.append(p_label)
    return created, skipped


def main() -> None:
    created, skipped = restore_doentes_labels()
    for path in created:
        print(f"[ok] Label criado: {path}")
    for path in skipped:
        print(f"[skip] Label já existe: {path}")


if __name__ == "__main__":
    main()
