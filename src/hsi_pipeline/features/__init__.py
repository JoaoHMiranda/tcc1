"""Feature engineering: pseudo-RGB generation, band selection, segmentation."""

from .pseudo_rgb_generation import run_pseudo_rgb_stage
from .rgb import save_rgb_from_channels, scale_robust
from .selection import run_band_selection_pipeline
from .selection_runner import run_selection, run_selection_for_dataset
from .segmentation import largest_component_mask

__all__ = [
    "run_pseudo_rgb_stage",
    "save_rgb_from_channels",
    "scale_robust",
    "run_band_selection_pipeline",
    "run_selection",
    "run_selection_for_dataset",
    "largest_component_mask",
]
