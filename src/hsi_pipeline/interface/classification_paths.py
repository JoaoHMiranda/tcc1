"""Path resolution utilities for classification CLI."""

from __future__ import annotations

from pathlib import Path


def resolve_override(value: str | None) -> Path | None:
    if value is None:
        return None
    return Path(value).expanduser().resolve()


__all__ = ["resolve_override"]
