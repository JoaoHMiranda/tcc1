"""Reporting helpers for classification/inference outputs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


def relative_to_base(path: Path, base: Path | None) -> str:
    if base is None:
        return str(path)
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def collect_box_info(res) -> Tuple[int, float, float, List[dict]]:
    boxes = getattr(res, "boxes", None)
    if boxes is None or boxes.data is None:
        return 0, 0.0, 0.0, []
    confs = boxes.conf.cpu().numpy().tolist()
    xywh = boxes.xywhn.cpu().numpy().tolist()
    clss = boxes.cls.cpu().numpy().astype(int).tolist()
    names = res.names if hasattr(res, "names") else {}
    det_rows = []
    for (x, y, w, h), c, conf in zip(xywh, clss, confs):
        det_rows.append(
            {
                "class_id": int(c),
                "class_name": names.get(int(c), str(c)),
                "conf": float(conf),
                "x_center": float(x),
                "y_center": float(y),
                "width": float(w),
                "height": float(h),
            }
        )
    n_det = len(confs)
    mean_conf = float(sum(confs) / n_det) if n_det else 0.0
    max_conf = float(max(confs)) if n_det else 0.0
    return n_det, mean_conf, max_conf, det_rows


def write_csv_reports(
    prediction_results: Sequence[object],
    out_dir: Path,
    relative_to: Path | None = None,
) -> Tuple[Path, Path, List[dict]]:
    rows = []
    detail_rows = []
    for res in prediction_results:
        path = Path(res.path)
        n_det, mean_conf, max_conf, det_rows = collect_box_info(res)
        rel_path = relative_to_base(path, relative_to)
        rows.append(
            {
                "image": rel_path,
                "n_detections": n_det,
                "mean_conf": mean_conf,
                "max_conf": max_conf,
            }
        )
        for det in det_rows:
            det_row = det.copy()
            det_row["image"] = rel_path
            detail_rows.append(det_row)

    summary_path = out_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["image", "n_detections", "mean_conf", "max_conf"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    details_path = out_dir / "detections.csv"
    with details_path.open("w", newline="", encoding="utf-8") as fp:
        fieldnames = ["image", "class_id", "class_name", "conf", "x_center", "y_center", "width", "height"]
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in detail_rows:
            writer.writerow(row)
    return summary_path, details_path, rows


def write_text_report(
    summary_csv: Path,
    details_csv: Path,
    rows: List[dict],
    out_dir: Path,
    relative_to: Path | None = None,
) -> Path:
    total_imgs = len(rows)
    total_det = sum(r["n_detections"] for r in rows)
    conf_rows = [r for r in rows if r["n_detections"] > 0]
    mean_conf_all = sum(r["mean_conf"] for r in conf_rows) / len(conf_rows) if conf_rows else 0.0
    summary_rel = relative_to_base(summary_csv, relative_to)
    details_rel = relative_to_base(details_csv, relative_to)
    report_lines = [
        "Resumo da inferência (classificar / pseudo_rgb/pca)",
        f"- Imagens processadas: {total_imgs}",
        f"- Total de detecções: {total_det}",
        f"- Média de conf. (imagens com detecção): {mean_conf_all:.4f}",
        f"- CSV geral: {summary_rel}",
        f"- CSV de detecções: {details_rel}",
        "",
        "Por imagem (n_det, mean_conf, max_conf):",
    ]
    for r in rows:
        report_lines.append(
            f"* {r['image']}: n={r['n_detections']}, mean_conf={r['mean_conf']:.4f}, max_conf={r['max_conf']:.4f}"
        )
    report_path = out_dir / "summary.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    return report_path


__all__ = ["collect_box_info", "write_csv_reports", "write_text_report"]
