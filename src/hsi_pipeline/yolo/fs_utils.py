"""Filesystem helpers for YOLO training outputs."""

from __future__ import annotations

from pathlib import Path


def next_available_path(base: Path) -> Path:
    """Return a unique path; appends _1, _2, ... if base exists."""
    if not base.exists():
        return base
    stem = base.name
    parent = base.parent
    idx = 1
    while True:
        candidate = parent / f"{stem}_{idx}"
        if not candidate.exists():
            return candidate
        idx += 1


__all__ = ["next_available_path"]
