"""Helpers to parse and load ENVI hyperspectral data."""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def parse_envi_hdr(hdr_path: str) -> Dict[str, Any]:
    txt = open(hdr_path, "r", errors="ignore").read().replace("\r\n", "\n")
    meta: Dict[str, Any] = {}
    lines = [l.strip() for l in txt.split("\n") if l.strip() and not l.strip().startswith(";")]
    i = 0
    while i < len(lines):
        line = lines[i]
        if "=" not in line:
            i += 1
            continue
        key, val = [s.strip() for s in line.split("=", 1)]
        key_l = key.lower()
        block = val
        if "{" in val and "}" not in val:
            acc = [val]
            i += 1
            while i < len(lines) and "}" not in lines[i]:
                acc.append(lines[i])
                i += 1
            if i < len(lines):
                acc.append(lines[i])
            block = " ".join(acc)
        meta[key_l] = block
        i += 1

    def get_int(k: str, default=None):
        v = meta.get(k)
        if v is None:
            return default
        try:
            return int(re.findall(r"[-+]?\d+", v)[0])
        except Exception:
            return default

    def get_float_list_from_braces(k: str) -> Optional[List[float]]:
        block = meta.get(k)
        if not block:
            return None
        inside = re.findall(r"\{(.*)\}", block, re.S)
        if not inside:
            return None
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", inside[0])
        return [float(x) for x in nums]

    samples = get_int("samples")
    lines_ = get_int("lines")
    bands = get_int("bands")
    interleave = meta.get("interleave", "bsq").strip().lower()
    byte_order = get_int("byte order", 0)
    data_type = get_int("data type", 12)
    header_off = get_int("header offset", 0)

    wavelengths = get_float_list_from_braces("wavelength")
    if wavelengths is not None and bands is not None and len(wavelengths) != bands:
        if len(wavelengths) > bands:
            wavelengths = wavelengths[:bands]
        else:
            step = (wavelengths[-1] - wavelengths[-2]) if len(wavelengths) >= 2 else 1.0
            while len(wavelengths) < bands:
                last = wavelengths[-1] if wavelengths else 400.0
                wavelengths.append(last + step)

    return dict(
        samples=samples,
        lines=lines_,
        bands=bands,
        interleave=interleave,
        byte_order=byte_order,
        data_type=data_type,
        header_offset=header_off,
        wavelengths=wavelengths,
    )


def envi_dtype(data_type: int, byte_order: int) -> np.dtype:
    endian = "<" if byte_order == 0 else ">"
    mapping = {
        1: endian + "u1",
        2: endian + "i2",
        3: endian + "i4",
        4: endian + "f4",
        5: endian + "f8",
        12: endian + "u2",
        13: endian + "u4",
        14: endian + "i8",
        15: endian + "u8",
    }
    return np.dtype(mapping.get(data_type, endian + "f4"))


def open_envi_memmap(hdr_path: str, raw_path: str):
    meta = parse_envi_hdr(hdr_path)
    H, W, B = meta["lines"], meta["samples"], meta["bands"]
    interleave = meta["interleave"]
    dtype = envi_dtype(meta["data_type"], meta["byte_order"])
    offset = int(meta.get("header offset", 0) or 0)
    if interleave == "bsq":
        shape = (B, H, W)
    elif interleave == "bil":
        shape = (H, B, W)
    elif interleave == "bip":
        shape = (H, W, B)
    else:
        shape = (B, H, W)
    mm = np.memmap(raw_path, mode="r", dtype=dtype, shape=shape, offset=offset)
    return mm, meta


def get_band_view(mm: np.memmap, interleave: str, band_idx: int) -> np.ndarray:
    if interleave == "bsq":
        return mm[band_idx, :, :]
    if interleave == "bil":
        return mm[:, band_idx, :]
    if interleave == "bip":
        return mm[:, :, band_idx]
    return mm[band_idx, :, :]


def resize_to(img2d: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    src_h, src_w = img2d.shape
    if (src_h, src_w) == (target_h, target_w):
        return img2d
    y_src = np.linspace(0, src_h - 1, src_h)
    y_tgt = np.linspace(0, src_h - 1, target_h)
    tmp = np.empty((target_h, src_w), dtype=np.float64)
    for x in range(src_w):
        tmp[:, x] = np.interp(y_tgt, y_src, img2d[:, x])
    x_src = np.linspace(0, src_w - 1, src_w)
    x_tgt = np.linspace(0, src_w - 1, target_w)
    out = np.empty((target_h, target_w), dtype=np.float64)
    for y in range(target_h):
        out[y, :] = np.interp(x_tgt, x_src, tmp[y, :])
    return out


def is_dark_file(name_lower: str) -> bool:
    return ("dark" in name_lower) or ("darkref" in name_lower)


def is_white_file(name_lower: str) -> bool:
    return ("white" in name_lower) or ("whiteref" in name_lower)


def discover_set(folder: str) -> Tuple[str, str, str, str, str, str]:
    hdrs = sorted([f for f in os.listdir(folder) if f.lower().endswith(".hdr")])
    raws = set([f.lower() for f in os.listdir(folder) if f.lower().endswith(".raw")])
    if not hdrs:
        raise FileNotFoundError("Nenhum .hdr encontrado na pasta.")
    hdr_main = hdr_dark = hdr_white = None
    for hdr in hdrs:
        low = hdr.lower()
        if is_dark_file(low):
            hdr_dark = hdr
        elif is_white_file(low):
            hdr_white = hdr
        else:
            hdr_main = hdr
    if not (hdr_main and hdr_dark and hdr_white):
        raise RuntimeError("Inclua 'dark'/'darkref' e 'white'/'whiteref' nos nomes dos .hdr.")

    def pair_raw(hdr_name: str) -> str:
        stem = os.path.splitext(hdr_name)[0]
        candidate = stem + ".raw"
        if candidate.lower() in raws:
            return os.path.join(folder, stem + ".raw")
        raise FileNotFoundError(f"RAW de {hdr_name} não encontrado (esperado {candidate}).")

    return (
        os.path.join(folder, hdr_main),
        pair_raw(hdr_main),
        os.path.join(folder, hdr_dark),
        pair_raw(hdr_dark),
        os.path.join(folder, hdr_white),
        pair_raw(hdr_white),
    )
