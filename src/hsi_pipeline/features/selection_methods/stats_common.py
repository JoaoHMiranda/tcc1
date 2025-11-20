"""Shared helpers for statistical tests."""

from __future__ import annotations

try:  # pragma: no cover - SciPy version dependent
    from scipy.stats import ConstantInputWarning as ConstantInputWarning
except ImportError:  # pragma: no cover
    class ConstantInputWarning(RuntimeWarning):
        pass


__all__ = ["ConstantInputWarning"]
