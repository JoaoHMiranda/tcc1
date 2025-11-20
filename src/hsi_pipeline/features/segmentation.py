"""Segmentation helpers for fast circular region proposals."""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


def largest_component_mask(gray: np.ndarray, min_area_frac: float = 0.01) -> Tuple[Optional[np.ndarray], Optional[Tuple[float, float, float]]]:
    H, W = gray.shape
    _, th1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, th2 = cv2.threshold(255 - gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    candidates = []
    kernel3 = np.ones((3, 3), np.uint8)
    kernel5 = np.ones((5, 5), np.uint8)
    for th in (th1, th2):
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel3, iterations=1)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel5, iterations=2)
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            candidates.append((0, None, None))
            continue
        cnt = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        if area < min_area_frac * H * W:
            candidates.append((0, None, None))
            continue
        (cx, cy), r = cv2.minEnclosingCircle(cnt)
        candidates.append((area, th, (float(cx), float(cy), float(r))))
    candidates.sort(key=lambda t: t[0], reverse=True)
    if candidates[0][0] <= 0:
        return None, None
    return candidates[0][1], candidates[0][2]
