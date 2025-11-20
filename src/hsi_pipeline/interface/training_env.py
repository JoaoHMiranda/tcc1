"""Environment helpers for YOLO training CLI."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
TRAIN_DIR = ROOT / "train-yolo"


def configure_ultralytics_home(train_dir: Path | None = None) -> Path:
    directory = Path(train_dir) if train_dir else TRAIN_DIR
    os.environ.setdefault("ULTRALYTICS_HOME", str(directory))
    # Impede que o Ultralytics tente instalar dependências automaticamente via 'uv' (sem permissões).
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    return directory


def ensure_models_in_train_dir(root: Path, train_dir: Path | None = None) -> None:
    """Move/synchronize checkpoints locais para a pasta de treino."""
    directory = configure_ultralytics_home(train_dir)
    directory.mkdir(parents=True, exist_ok=True)
    for fname in ("yolo12x.pt", "yolo11n.pt"):
        src = root / fname
        dst = directory / fname
        if src.exists():
            if not dst.exists():
                shutil.move(str(src), dst)
            else:
                src.unlink()


__all__ = ["ROOT", "TRAIN_DIR", "configure_ultralytics_home", "ensure_models_in_train_dir"]
