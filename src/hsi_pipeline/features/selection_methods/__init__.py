"""Band-selection building blocks split per method."""

from .dataset import PixelSampleDataset, prepare_pixel_dataset
from .utils import (
    enabled_selection_methods,
    resolve_active_method,
    method_config,
    band_list_from_df,
    base_dataframe,
    save_dataframe,
)
from .models.anova import run_anova
from .models.t_test import run_t_test
from .models.kruskal import run_kruskal
from .models.random_forest import run_random_forest, DEFAULT_SELECTION_REASON
from .models.vip_pls_da import run_vip_scores
from .models.pca import run_pca
from .classification import (
    run_classification,
    enabled_classifiers,
    resolve_active_classifier,
    DEFAULT_CLASSIFICATION_REASON,
)
from .summary import write_selection_summary

__all__ = [
    "PixelSampleDataset",
    "prepare_pixel_dataset",
    "enabled_selection_methods",
    "resolve_active_method",
    "method_config",
    "band_list_from_df",
    "base_dataframe",
    "save_dataframe",
    "run_anova",
    "run_t_test",
    "run_kruskal",
    "run_random_forest",
    "run_vip_scores",
    "run_pca",
    "run_classification",
    "enabled_classifiers",
    "resolve_active_classifier",
    "write_selection_summary",
    "DEFAULT_SELECTION_REASON",
    "DEFAULT_CLASSIFICATION_REASON",
]
