"""Helpers for resolving dataset paths and shared path configurations."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union


@dataclass
class DatasetPathPair:
    name: Optional[str] = None
    input_root: Optional[str] = None
    output_root: Optional[str] = None


@dataclass
class GlobalPathConfig:
    input_root: Optional[str] = None
    output_root: Optional[str] = None
    dataset_pairs: List[DatasetPathPair] = field(default_factory=list)


def _extract_value(node):
    if isinstance(node, dict) and "value" in node:
        return node["value"]
    return node


def load_paths_config(path: Union[str, Path]) -> GlobalPathConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as fp:
        raw = json.load(fp)
    data = {key: _extract_value(value) for key, value in raw.items()}
    dataset_pairs_raw = data.get("dataset_pairs")
    dataset_pairs: List[DatasetPathPair] = []
    if dataset_pairs_raw is not None:
        entries = _extract_value(dataset_pairs_raw)
        if isinstance(entries, list):
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                dataset_pairs.append(
                    DatasetPathPair(
                        name=entry.get("name"),
                        input_root=entry.get("input_root"),
                        output_root=entry.get("output_root"),
                    )
                )
    return GlobalPathConfig(
        input_root=data.get("input_root"),
        output_root=data.get("output_root"),
        dataset_pairs=dataset_pairs,
    )


def resolve_out_base(folder: str, out_root: Optional[str]) -> Tuple[str, str, str]:
    folder_abs = os.path.abspath(folder)
    base = os.path.basename(os.path.normpath(folder_abs))
    if out_root:
        resolved_root = out_root
    else:
        parent_parent = os.path.dirname(os.path.dirname(folder_abs))
        resolved_root = os.path.join(parent_parent, "ATCC")
    os.makedirs(resolved_root, exist_ok=True)
    out_base = os.path.join(resolved_root, base)
    os.makedirs(out_base, exist_ok=True)
    return out_base, resolved_root, base


__all__ = [
    "DatasetPathPair",
    "GlobalPathConfig",
    "load_paths_config",
    "resolve_out_base",
]
