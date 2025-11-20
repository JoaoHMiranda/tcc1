"""Reflectance computation utilities."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from ..data.envi_io import get_band_view, resize_to


class ReflectanceBandProvider:
    """Lazy band provider that caches reflectance computations."""

    def __init__(
        self,
        mm_main,
        meta_main,
        mm_dark,
        meta_dark,
        mm_white,
        meta_white,
        height: int,
        width: int,
        cache_size: int = 64,
    ):
        self.mm_main = mm_main
        self.meta_main = meta_main
        self.mm_dark = mm_dark
        self.meta_dark = meta_dark
        self.mm_white = mm_white
        self.meta_white = meta_white
        self.height = height
        self.width = width
        self.cache = OrderedDict()
        self.cache_size = max(3, int(cache_size))
        self.eps = 1e-8

    def compute_band(self, band: int) -> np.ndarray:
        I = get_band_view(self.mm_main, self.meta_main["interleave"], band).astype(np.float64)
        dark = get_band_view(self.mm_dark, self.meta_dark["interleave"], band).astype(np.float64)
        white = get_band_view(self.mm_white, self.meta_white["interleave"], band).astype(np.float64)
        if dark.shape != (self.height, self.width):
            dark = resize_to(dark, self.height, self.width)
        if white.shape != (self.height, self.width):
            white = resize_to(white, self.height, self.width)
        denom = white - dark
        denom[np.abs(denom) < self.eps] = self.eps
        reflectance = (I - dark) / denom
        return np.clip(reflectance, 0, 2.0)

    def get(self, band: int) -> np.ndarray:
        if band in self.cache:
            value = self.cache.pop(band)
            self.cache[band] = value
            return value
        reflectance = self.compute_band(band)
        self.cache[band] = reflectance
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return reflectance
