"""Helpers to check optional dependencies for YOLO export formats."""

from __future__ import annotations

from importlib import util
from typing import Dict, Iterable, Optional

# Minimal modules required for each export format supported in configs.
FORMAT_REQUIREMENTS: Dict[str, Iterable[str]] = {
    "onnx": ("onnx",),
    "torchscript": ("torch",),
    "openvino": ("openvino",),
    "coreml": ("coremltools",),
    "engine": ("tensorrt",),
}

# User-friendly hints describing how to install each dependency.
FORMAT_HINTS: Dict[str, str] = {
    "openvino": 'pip install "openvino>=2024.0.0"',
    "coreml": 'pip install "coremltools>=8.0"',
    "engine": 'pip install "tensorrt-cu12>7.0.0,!=10.1.0"',
}


def missing_dependency_message(format_name: str) -> Optional[str]:
    """Return a human-friendly message if export requirements are missing."""
    required = FORMAT_REQUIREMENTS.get(format_name.lower())
    if not required:
        return None
    missing = [pkg for pkg in required if util.find_spec(pkg) is None]
    if not missing:
        return None
    hint = FORMAT_HINTS.get(format_name.lower())
    packages = ", ".join(sorted(missing))
    if hint:
        return f"dependência(s) ausente(s): {packages}. Instale com: {hint}"
    return f"dependência(s) ausente(s): {packages}."


__all__ = ["missing_dependency_message"]
