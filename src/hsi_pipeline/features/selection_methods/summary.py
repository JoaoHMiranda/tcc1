"""Summary writers for selection stage outputs."""

from __future__ import annotations

import json
import math
import os
from typing import Dict


def write_selection_summary(out_dir: str, summary: Dict[str, object]) -> None:
    json_path = os.path.join(out_dir, "selection_summary.json")
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)
    lines = [
        f"Status: {'habilitado' if summary.get('enabled') else 'desligado'}",
        f"Método ativo: {summary.get('active_method') or 'N/A'}",
    ]
    if summary.get("method_reason"):
        lines.append(f"Motivo do método padrão: {summary['method_reason']}")
    if summary.get("classification"):
        cls = summary["classification"]
        lines.append(f"Classificador ativo: {cls.get('active_classifier', 'N/A')}")
        if cls.get("reason"):
            lines.append(f"Motivo do classificador: {cls['reason']}")

        def format_metric(value: object) -> str:
            try:
                val = float(value)
            except (TypeError, ValueError):
                return "nan"
            if math.isnan(val):
                return "nan"
            return f"{val:.3f}"

        acc_str = format_metric(cls.get("accuracy"))
        f1_str = format_metric(cls.get("f1"))
        auc_str = format_metric(cls.get("roc_auc"))
        lines.append(f"Métricas → accuracy: {acc_str} | f1: {f1_str} | roc_auc: {auc_str}")
    txt_path = os.path.join(out_dir, "selection_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines))
