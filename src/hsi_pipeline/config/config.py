"""Typed configuration for the HSI preprocessing pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union


@dataclass
class TrimConfig:
    left: int = 10
    right: int = 10


@dataclass
class MedianGuessSettings:
    sample_step: int = 4
    max_samples: int = 60
    blur_sigma: float = 1.2
    kernel_size: int = 5
    min_area_fraction: float = 0.02


@dataclass
class VariantOutputSettings:
    enabled: bool = True
    plot: bool = True


@dataclass
class OutputToggleConfig:
    correcao: VariantOutputSettings = field(default_factory=VariantOutputSettings)
    correcao_snv: VariantOutputSettings = field(default_factory=VariantOutputSettings)
    correcao_msc: VariantOutputSettings = field(default_factory=VariantOutputSettings)
    correcao_snv_msc: VariantOutputSettings = field(default_factory=VariantOutputSettings)
    export_metadata: bool = True


@dataclass
class PseudoRGBMethodToggles:
    manual: bool = False
    pca: bool = False
    linear: bool = False


@dataclass
class ManualPseudoRGBConfig:
    band_indices: Sequence[int] = (10, 40, 75)
    output_name: str = "manual_rgb.png"


@dataclass
class PCAPseudoRGBConfig:
    max_pixels: int = 25000
    standardize: bool = True
    whiten: bool = False
    random_state: int = 42


def default_linear_recipes():
    return [
        {
            "name": "yolo_default",
            "description": "Combinação linear inspirada em realces de contraste para YOLO.",
            "channels": {
                "r": [
                    {"orig_band_index": 30, "weight": 0.6},
                    {"orig_band_index": 45, "weight": 0.4},
                ],
                "g": [
                    {"orig_band_index": 60, "weight": 0.5},
                    {"orig_band_index": 72, "weight": 0.5},
                ],
                "b": [
                    {"orig_band_index": 90, "weight": 0.7},
                    {"orig_band_index": 110, "weight": 0.3},
                ],
            },
        }
    ]


@dataclass
class LinearComboConfig:
    recipes: Sequence[Dict[str, Any]] = field(default_factory=default_linear_recipes)


@dataclass
class PseudoRGBConfig:
    enabled: bool = False
    active_method: Optional[str] = None
    output_dir_name: str = "pseudo_rgb"
    toggles: PseudoRGBMethodToggles = field(default_factory=PseudoRGBMethodToggles)
    manual: ManualPseudoRGBConfig = field(default_factory=ManualPseudoRGBConfig)
    pca: PCAPseudoRGBConfig = field(default_factory=PCAPseudoRGBConfig)
    linear: LinearComboConfig = field(default_factory=LinearComboConfig)


@dataclass
class DatasetSplitConfig:
    train: float = 0.7
    val: float = 0.2
    test: float = 0.1


def default_export_formats():
    return ["onnx", "torchscript", "openvino", "coreml", "engine"]


@dataclass
class YoloAugmentationConfig:
    fliplr: float = 0.5
    flipud: float = 0.2
    degrees: float = 10.0
    crop: float = 0.0


@dataclass
class YoloTrainingConfig:
    enabled: bool = False
    out_root: Optional[str] = None
    models_root: Optional[str] = None
    pseudo_root: str = "pseudo_rgb"
    pseudo_method: str = "pca"
    dataset_output_dir: str = "yolo12_dataset"
    runs_dir: str = "yolo12_runs"
    annotations_root: Optional[str] = None
    labels_follow_images: bool = True
    label_extension: str = ".txt"
    missing_label_policy: str = "error"
    run_validation: bool = True
    amp: bool = False
    enhance_pseudo_rgb: bool = True
    training_root: Optional[str] = None
    model: str = "yolo12n.pt"
    imgsz: int = 640
    epochs: int = 100
    batch: int = 16
    patience: int = 50
    device: Optional[str] = None
    project_name: str = "yolo12_hsi"
    experiment_name: str = "exp"
    classes: Sequence[str] = field(default_factory=lambda: ["staphylococcus"])
    split: DatasetSplitConfig = field(default_factory=DatasetSplitConfig)
    random_state: int = 42
    datasets: Optional[Sequence[str]] = field(default_factory=lambda: ["doentes"])
    clean_output: bool = True
    train_extra_args: Dict[str, Any] = field(default_factory=dict)
    augmentations: YoloAugmentationConfig = field(default_factory=YoloAugmentationConfig)
    export_formats: Sequence[str] = field(default_factory=default_export_formats)


@dataclass
class AnovaSelectionSettings:
    enabled: bool = False
    alpha: float = 0.05


@dataclass
class TTestSelectionSettings:
    enabled: bool = False
    alpha: float = 0.05


@dataclass
class KruskalSelectionSettings:
    enabled: bool = False
    alpha: float = 0.05


@dataclass
class RandomForestSelectionSettings:
    enabled: bool = False
    n_estimators: int = 10000
    max_features: Union[str, float, int, None] = "sqrt"
    n_jobs: Optional[int] = 3
    class_weight: Optional[Union[str, Dict[str, float]]] = None
    bootstrap: bool = True


@dataclass
class VipPLSDASelectionSettings:
    enabled: bool = False
    n_components: Optional[int] = None


@dataclass
class PCASelectionSettings:
    enabled: bool = False
    n_components: Optional[int] = None
    whiten: bool = False


@dataclass
class BandSelectionMethodsConfig:
    anova: AnovaSelectionSettings = field(default_factory=AnovaSelectionSettings)
    t_test: TTestSelectionSettings = field(default_factory=TTestSelectionSettings)
    kruskal: KruskalSelectionSettings = field(default_factory=KruskalSelectionSettings)
    random_forest: RandomForestSelectionSettings = field(default_factory=RandomForestSelectionSettings)
    vip_pls_da: VipPLSDASelectionSettings = field(default_factory=VipPLSDASelectionSettings)
    pca: PCASelectionSettings = field(default_factory=PCASelectionSettings)


@dataclass
class ClassificationMethodToggles:
    svm_linear: bool = False
    random_forest: bool = False
    pls_da: bool = False


@dataclass
class ClassificationConfig:
    enabled: bool = False
    active_method: Optional[str] = None
    toggles: ClassificationMethodToggles = field(default_factory=ClassificationMethodToggles)
    test_size: float = 0.2
    random_state: int = 42


@dataclass
class BandSelectionConfig:
    enabled: bool = False
    active_method: Optional[str] = "random_forest"
    methods: BandSelectionMethodsConfig = field(default_factory=BandSelectionMethodsConfig)
    top_k_bands: int = 40
    sample_pixels_per_class: int = 1200
    min_pixels_per_class: int = 60
    roi_radius_scale: float = 0.9
    inner_background_scale: float = 1.05
    outer_background_scale: float = 1.35
    random_state: int = 42
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)


@dataclass
class PipelineConfig:
    """Master configuration of the pipeline."""

    folder: str = "/home/joaoh/programacao/TCC1/minha/ATCC27_240506-161129"
    out_root: Optional[str] = None
    enabled: bool = True
    delta_bands: int = 1
    cache_size_bands: int = 64
    trimming: TrimConfig = field(default_factory=TrimConfig)
    median_guess: MedianGuessSettings = field(default_factory=MedianGuessSettings)
    toggles: OutputToggleConfig = field(default_factory=OutputToggleConfig)
    median_source_variant: str = "correcao_msc"
    band_selection: BandSelectionConfig = field(default_factory=BandSelectionConfig)
    pseudo_rgb: PseudoRGBConfig = field(default_factory=PseudoRGBConfig)
    dataset_filters: Dict[str, bool] = field(default_factory=dict)


def strip_descriptions(node: Any) -> Any:
    if isinstance(node, dict):
        if set(node.keys()).issubset({"value", "description"}) and "value" in node:
            return node["value"]
        cleaned: Dict[str, Any] = {}
        for key, value in node.items():
            if key == "description":
                continue
            cleaned[key] = strip_descriptions(value)
        return cleaned
    if isinstance(node, list):
        return [strip_descriptions(item) for item in node]
    return node


def update_dataclass(instance, values: Dict[str, Any]):
    for key, value in values.items():
        if hasattr(instance, key):
            setattr(instance, key, value)
    return instance


def load_config_from_mapping(data: Dict[str, Any]) -> PipelineConfig:
    trimmed = strip_descriptions(data)
    base = PipelineConfig()
    base.folder = trimmed.get("folder", base.folder)
    base.out_root = trimmed.get("out_root", base.out_root)
    base.enabled = trimmed.get("enabled", base.enabled)
    base.delta_bands = trimmed.get("delta_bands", base.delta_bands)
    base.cache_size_bands = trimmed.get("cache_size_bands", base.cache_size_bands)

    trimming = trimmed.get("trimming", {})
    base.trimming = TrimConfig(
        left=trimming.get("left", base.trimming.left),
        right=trimming.get("right", base.trimming.right),
    )

    med = trimmed.get("median_guess", {})
    base.median_guess = update_dataclass(MedianGuessSettings(), med)

    toggles_cfg = trimmed.get("toggles", {})
    base.toggles = OutputToggleConfig()

    def _coerce_variant_settings(value) -> VariantOutputSettings:
        if isinstance(value, dict):
            return update_dataclass(VariantOutputSettings(), value)
        if isinstance(value, bool):
            return VariantOutputSettings(enabled=value, plot=value)
        return VariantOutputSettings()

    for field_name in ("correcao", "correcao_snv", "correcao_msc", "correcao_snv_msc"):
        val = toggles_cfg.get(field_name)
        if val is not None:
            setattr(base.toggles, field_name, _coerce_variant_settings(val))
    export_meta = toggles_cfg.get("export_metadata")
    if export_meta is not None:
        base.toggles.export_metadata = bool(export_meta)

    base.median_source_variant = trimmed.get(
        "median_source_variant", base.median_source_variant
    )
    dataset_filters = trimmed.get("dataset_filters")
    if isinstance(dataset_filters, dict):
        base.dataset_filters = dataset_filters

    selection_cfg_raw = trimmed.get("band_selection")
    selection_cfg = dict(selection_cfg_raw) if isinstance(selection_cfg_raw, dict) else {}
    extra_fields = (
        "enabled",
        "active_method",
        "top_k_bands",
        "sample_pixels_per_class",
        "min_pixels_per_class",
        "roi_radius_scale",
        "inner_background_scale",
        "outer_background_scale",
        "random_state",
    )
    inline_fields_present = any(field in trimmed for field in extra_fields if field != "enabled")
    has_selection_data = bool(selection_cfg) or inline_fields_present
    if has_selection_data:
        for field in extra_fields:
            if field in trimmed and field not in selection_cfg:
                selection_cfg[field] = trimmed[field]
        if "methods" not in selection_cfg and isinstance(trimmed.get("methods"), dict):
            selection_cfg["methods"] = trimmed.get("methods")
    else:
        selection_cfg = None
    if selection_cfg:
        selection_kwargs = {
            key: value
            for key, value in selection_cfg.items()
            if key not in {"methods", "classification"}
        }
        base.band_selection = update_dataclass(BandSelectionConfig(), selection_kwargs)
        methods_cfg = selection_cfg.get("methods")
        if isinstance(methods_cfg, dict):
            methods = BandSelectionMethodsConfig()
            method_type_map = {
                "anova": AnovaSelectionSettings,
                "t_test": TTestSelectionSettings,
                "kruskal": KruskalSelectionSettings,
                "random_forest": RandomForestSelectionSettings,
                "vip_pls_da": VipPLSDASelectionSettings,
                "pca": PCASelectionSettings,
            }
            for key, cls in method_type_map.items():
                values = methods_cfg.get(key)
                if isinstance(values, dict):
                    setattr(methods, key, update_dataclass(cls(), values))
            base.band_selection.methods = methods
        class_cfg = selection_cfg.get("classification")
        if isinstance(class_cfg, dict):
            class_kwargs = {
                key: value for key, value in class_cfg.items() if key != "toggles"
            }
            base.band_selection.classification = update_dataclass(
                ClassificationConfig(), class_kwargs
            )
            class_toggles = class_cfg.get("toggles")
            if isinstance(class_toggles, dict):
                base.band_selection.classification.toggles = update_dataclass(
                    ClassificationMethodToggles(), class_toggles
                )

    pseudo_cfg = trimmed.get("pseudo_rgb")
    if isinstance(pseudo_cfg, dict):
        pseudo_kwargs = {
            key: value
            for key, value in pseudo_cfg.items()
            if key not in {"toggles", "manual", "pca", "linear"}
        }
        base.pseudo_rgb = update_dataclass(PseudoRGBConfig(), pseudo_kwargs)
        pseudo_toggles = pseudo_cfg.get("toggles")
        if isinstance(pseudo_toggles, dict):
            base.pseudo_rgb.toggles = update_dataclass(
                PseudoRGBMethodToggles(), pseudo_toggles
            )
        manual_cfg = pseudo_cfg.get("manual")
        if isinstance(manual_cfg, dict):
            base.pseudo_rgb.manual = update_dataclass(ManualPseudoRGBConfig(), manual_cfg)
        pca_cfg = pseudo_cfg.get("pca")
        if isinstance(pca_cfg, dict):
            base.pseudo_rgb.pca = update_dataclass(PCAPseudoRGBConfig(), pca_cfg)
        linear_cfg = pseudo_cfg.get("linear")
        if isinstance(linear_cfg, dict):
            recipes = linear_cfg.get("recipes")
            linear_kwargs = {
                key: value for key, value in linear_cfg.items() if key != "recipes"
            }
            base.pseudo_rgb.linear = update_dataclass(LinearComboConfig(), linear_kwargs)
            if recipes is not None:
                base.pseudo_rgb.linear.recipes = recipes

    return base


def load_config_from_json(path: Union[str, Path]) -> PipelineConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return load_config_from_mapping(raw)


def load_yolo_training_config_from_json(path: Union[str, Path]) -> YoloTrainingConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    trimmed = strip_descriptions(raw)
    base = YoloTrainingConfig()
    split_cfg = trimmed.pop("split", None)
    if isinstance(split_cfg, dict):
        base.split = update_dataclass(DatasetSplitConfig(), split_cfg)
    train_args = trimmed.pop("train_extra_args", None)
    aug_cfg = trimmed.pop("augmentations", None)
    export_formats = trimmed.pop("export_formats", None)
    classes = trimmed.get("classes")
    datasets = trimmed.get("datasets")
    base = update_dataclass(base, trimmed)
    if classes is not None:
        base.classes = classes
    if datasets is not None:
        base.datasets = list(datasets)
    if train_args is not None:
        base.train_extra_args = train_args
    if isinstance(aug_cfg, dict):
        base.augmentations = update_dataclass(YoloAugmentationConfig(), aug_cfg)
    if export_formats is not None:
        base.export_formats = list(export_formats)
    return base


__all__ = [
    "PipelineConfig",
    "TrimConfig",
    "MedianGuessSettings",
    "VariantOutputSettings",
    "OutputToggleConfig",
    "PseudoRGBConfig",
    "PseudoRGBMethodToggles",
    "ManualPseudoRGBConfig",
    "PCAPseudoRGBConfig",
    "LinearComboConfig",
    "DatasetSplitConfig",
    "YoloAugmentationConfig",
    "YoloTrainingConfig",
    "AnovaSelectionSettings",
    "TTestSelectionSettings",
    "KruskalSelectionSettings",
    "RandomForestSelectionSettings",
    "VipPLSDASelectionSettings",
    "PCASelectionSettings",
    "BandSelectionMethodsConfig",
    "BandSelectionConfig",
    "ClassificationConfig",
    "ClassificationMethodToggles",
    "load_config_from_json",
    "load_config_from_mapping",
    "load_yolo_training_config_from_json",
]
