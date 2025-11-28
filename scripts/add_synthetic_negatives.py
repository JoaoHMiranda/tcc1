#!/usr/bin/env python3
"""Gera imagens sintéticas brancas e pretas marcadas como negativas (sem estafilococos)."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

# Dimensão original das PNGs do dataset (verificada com `file`).
WIDTH = 320
HEIGHT = 290
# Quantidade por cor.
COUNT_PER_COLOR = 100
# Pasta de treino (1ª amostra usada no split fixo).
TARGET_RELATIVE = Path("hsi_modificado/doentes/ATCC13_240506-161053/correcao_snv_msc_rgb_bands")


def _chunk(tag: bytes, data: bytes) -> bytes:
    """Monta um chunk PNG com CRC."""
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def write_solid_png(path: Path, width: int, height: int, value: int) -> None:
    """Escreve um PNG RGB sólido usando apenas bibliotecas padrão."""
    header = b"\x89PNG\r\n\x1a\n"
    ihdr = _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    row = bytes([0]) + bytes([value]) * (width * 3)  # filtro 0 + pixels
    raw = row * height
    idat = _chunk(b"IDAT", zlib.compress(raw))
    iend = _chunk(b"IEND", b"")
    path.write_bytes(header + ihdr + idat + iend)


def ensure_empty_label(path: Path) -> None:
    """Cria/zera o TXT para indicar ausência de estafilococos."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def generate(target: Path) -> int:
    target.mkdir(parents=True, exist_ok=True)
    created = 0
    for name, value in (("white", 255), ("black", 0)):
        for idx in range(1, COUNT_PER_COLOR + 1):
            stem = f"synthetic_{name}_{idx:03d}"
            png_path = target / f"{stem}.png"
            txt_path = target / f"{stem}.txt"

            if not png_path.exists():
                write_solid_png(png_path, WIDTH, HEIGHT, value)
                created += 1

            # Sempre sobrescreve o label para garantir negativo (sem classe).
            ensure_empty_label(txt_path)
    return created


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / TARGET_RELATIVE
    if not target.exists():
        raise SystemExit(f"Pasta alvo não encontrada: {target}")
    created = generate(target)
    print(f"[ok] {created} imagens sintéticas geradas/em uso em {target}")


if __name__ == "__main__":
    main()
