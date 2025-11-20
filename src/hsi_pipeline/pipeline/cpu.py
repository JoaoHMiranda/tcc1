"""CPU threading utilities."""

from __future__ import annotations

import os
from typing import Optional

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore


def configure_cpu_workers(workers: Optional[int]) -> Optional[int]:
    """Set global thread limits for BLAS/OpenMP/OpenCV."""
    if workers is None or workers <= 0:
        return None
    value = str(int(workers))
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[var] = value
    if cv2 is not None and hasattr(cv2, "setNumThreads"):
        try:
            cv2.setNumThreads(int(workers))
        except Exception:
            pass
    return workers


__all__ = ["configure_cpu_workers"]
