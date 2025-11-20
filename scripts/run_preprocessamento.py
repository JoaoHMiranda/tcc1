#!/usr/bin/env python3
"""CLI wrapper for the preprocessing pipeline."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hsi_pipeline.interface.preprocess import main

if __name__ == "__main__":
    main()
