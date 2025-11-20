#!/usr/bin/env python3
"""Gera augments offline das imagens PCA com labels transformados."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageEnhance

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = ROOT / "hsi_modificado" / "doentes"
DEFAULT_OUTPUT_METHOD = "pca_augmented"
N_PER_SAMPLE = 100  # número de imagens a gerar por amostra (inclui 90/180/270 + flips aleatórios)
DEFAULT_SOURCE_METHOD = "pca"
DEFAULT_BASE_NAME = "pca_rgb"


def load_labels(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    labels = []
    with label_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, x, y, w, h = parts
            labels.append((int(cls), float(x), float(y), float(w), float(h)))
    return labels


def save_labels(labels: List[Tuple[int, float, float, float, float]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fp:
        for cls, x, y, w, h in labels:
            fp.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def rotate_bbox(x: float, y: float, w: float, h: float, angle: int) -> Tuple[float, float, float, float]:
    # Coordenadas normalizadas; rotações em passos de 90.
    angle = angle % 360
    if angle == 0:
        return x, y, w, h
    if angle == 180:
        return 1 - x, 1 - y, w, h
    if angle == 90:
        return y, 1 - x, h, w
    if angle == 270:
        return 1 - y, x, h, w
    raise ValueError(f"Ângulo inválido: {angle}")


def flip_bbox(x: float, y: float, w: float, h: float, flip_h: bool, flip_v: bool) -> Tuple[float, float, float, float]:
    if flip_h:
        x = 1 - x
    if flip_v:
        y = 1 - y
    return x, y, w, h


def apply_transforms(img: Image.Image, labels: List[Tuple[int, float, float, float, float]], angle: int, flip_h: bool, flip_v: bool):
    # Aplica rotação (em 90°) e flips na imagem e nas boxes sem alterar tamanho final.
    out_img = img
    if angle == 90:
        out_img = out_img.transpose(Image.ROTATE_90)
    elif angle == 180:
        out_img = out_img.transpose(Image.ROTATE_180)
    elif angle == 270:
        out_img = out_img.transpose(Image.ROTATE_270)
    if flip_h:
        out_img = out_img.transpose(Image.FLIP_LEFT_RIGHT)
    if flip_v:
        out_img = out_img.transpose(Image.FLIP_TOP_BOTTOM)
    # Ajuste de bbox
    new_labels = []
    for cls, x, y, w, h in labels:
        nx, ny, nw, nh = rotate_bbox(x, y, w, h, angle)
        nx, ny, nw, nh = flip_bbox(nx, ny, nw, nh, flip_h, flip_v)
        new_labels.append((cls, nx, ny, nw, nh))
    return out_img, new_labels


def random_color_jitter(img: Image.Image) -> Image.Image:
    # Pequenas variações de brilho/contraste para diversificar.
    bright = ImageEnhance.Brightness(img).enhance(0.9 + random.random() * 0.2)
    contrast = ImageEnhance.Contrast(bright).enhance(0.9 + random.random() * 0.2)
    return contrast


def augment_sample(
    sample_dir: Path,
    output_method: str,
    n_per_sample: int,
    source_method: str = DEFAULT_SOURCE_METHOD,
    base_name: str = DEFAULT_BASE_NAME,
) -> int:
    pseudo_dir = sample_dir / "pseudo_rgb" / source_method
    pca_img = pseudo_dir / f"{base_name}.png"
    label_candidates = [
        pseudo_dir / f"{base_name}.txt",
        sample_dir / "_auto_labels" / source_method / f"{base_name}.txt",
    ]
    label_path = next((p for p in label_candidates if p.exists()), None)
    if not pca_img.exists() or label_path is None:
        return 0
    labels = load_labels(label_path)
    img = Image.open(pca_img).convert("RGB")
    out_dir = sample_dir / "pseudo_rgb" / output_method
    out_dir.mkdir(parents=True, exist_ok=True)
    generated = 0
    # Sempre gera versões padrão: 0, 90, 180, 270 e flips combinados.
    angles = [0, 90, 180, 270]
    flips = [(False, False), (True, False), (False, True), (True, True)]
    variants = angles * 2  # garante multiplicidade
    while generated < n_per_sample:
        angle = random.choice(variants)
        flip_h, flip_v = random.choice(flips)
        aug_img, aug_labels = apply_transforms(img, labels, angle, flip_h, flip_v)
        aug_img = random_color_jitter(aug_img)
        stem = f"{base_name}_aug_{generated:03d}"
        img_path = out_dir / f"{stem}.png"
        lbl_path = out_dir / f"{stem}.txt"
        aug_img.save(img_path)
        save_labels(aug_labels, lbl_path)
        generated += 1
    return generated


def main():
    parser = argparse.ArgumentParser(description="Gera augmentations offline das imagens PCA para treino YOLO.")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="Raiz das amostras pré-processadas (pseudo_rgb/pca).")
    parser.add_argument("--method", default=DEFAULT_OUTPUT_METHOD, help="Nome da subpasta onde salvar os augments (pseudo_rgb/<method>).")
    parser.add_argument("--per_sample", type=int, default=N_PER_SAMPLE, help="Quantidade de imagens geradas por amostra.")
    parser.add_argument("--source-method", default=DEFAULT_SOURCE_METHOD, help="Subpasta de origem dentro de pseudo_rgb (ex.: pca).")
    parser.add_argument("--base-name", default=DEFAULT_BASE_NAME, help="Prefixo dos arquivos de imagem/label usados como base.")
    args = parser.parse_args()

    source_root = Path(args.source).expanduser().resolve()
    total = 0
    count_samples = 0
    for sample_dir in sorted(source_root.glob("*")):
        if not sample_dir.is_dir():
            continue
        n = augment_sample(sample_dir, args.method, args.per_sample, args.source_method, args.base_name)
        if n > 0:
            count_samples += 1
            total += n
    print(f"[ok] Augmentações geradas: {total} imagens em {count_samples} amostras.")


if __name__ == "__main__":
    main()
