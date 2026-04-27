from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TargetTransformPolicy = Literal[
    "raw_level",
    "official_tcode_transformed",
    "custom_target_transform",
]

XTransformPolicy = Literal[
    "raw_level",
    "official_tcode_transformed",
    "custom_x_transform",
]

TcodePolicy = Literal[
    "raw_only",
    "official_tcode_only",
    "official_tcode_then_extra_preprocess",
    "extra_preprocess_only",
    "extra_preprocess_then_official_tcode",
    "custom_transform_sequence",
]

RepresentationPolicy = Literal[
    "raw_only",
    "official_tcode_only",
    "custom_transform_only",
]

PreprocessingAxisRole = Literal[
    "fixed_preprocessing",
    "swept_preprocessing",
    "ablation_preprocessing",
]

TcodeApplicationScope = Literal[
    "target_only",
    "predictors_only",
    "target_and_predictors",
    "none",
]

MissingPolicy = Literal[
    "none",
    "drop",
    "em_impute",
    "mean_impute",
    "median_impute",
    "ffill",
    "interpolate_linear",
    "drop_rows",
    "drop_columns",
    "drop_if_above_threshold",
    "missing_indicator",
    "custom",
]

OutlierPolicy = Literal[
    "none",
    "clip",
    "outlier_to_nan",
    "winsorize",
    "trim",
    "iqr_clip",
    "mad_clip",
    "zscore_clip",
    "outlier_to_missing",
    "custom",
]

ScalingPolicy = Literal[
    "none",
    "standard",
    "robust",
    "minmax",
    "demean_only",
    "unit_variance_only",
    "rank_scale",
    "custom",
]

DimensionalityReductionPolicy = Literal[
    "none",
    "pca",
    "static_factor",
    "custom",
]

FeatureSelectionPolicy = Literal[
    "none",
    "correlation_filter",
    "lasso_select",
    "mutual_information_screen",
    "custom",
]

PreprocessOrder = Literal[
    "none",
    "official_tcode_only",
    "extra_only",
    "official_tcode_then_extra",
    "extra_preprocess_then_official_tcode",
    "custom",
]

PreprocessFitScope = Literal[
    "not_applicable",
    "train_only",
    "expanding_train_only",
    "rolling_train_only",
]

InverseTransformPolicy = Literal[
    "none",
    "target_only",
    "forecast_scale_only",
    "custom",
]

EvaluationScale = Literal[
    "raw_level",
    "original_scale",
    "transformed_scale",
    "both",
]

TargetTransform = Literal[
    "level",
    "difference",
    "log",
    "log_difference",
    "growth_rate",
]

TargetNormalization = Literal[
    "none",
    "zscore_train_only",
    "robust_zscore",
    "minmax",
    "unit_variance",
]

TargetDomain = Literal[
    "unconstrained",
    "nonnegative",
    "bounded_0_1",
    "integer_count",
    "probability_target",
]

ScalingScope = Literal[
    "columnwise",
    "datewise_cross_sectional",
    "groupwise",
    "categorywise",
    "global_train_only",
]

AdditionalPreprocessing = Literal[
    "none",
    "smoothing_ma",
    "ema",
    "hp_filter",
    "bandpass_filter",
]

XLagCreation = Literal[
    "no_x_lags",
    "fixed_x_lags",
    "cv_selected_x_lags",
    "variable_specific_lags",
    "category_specific_lags",
]

FeatureGrouping = Literal[
    "none",
    "fred_category_group",
    "economic_theme_group",
    "lag_group",
    "factor_group",
]

FeatureSelectionSemantics = Literal[
    "select_before_factor",
    "select_after_factor",
    "select_after_custom_blocks",
]

RecipeMode = Literal[
    "fixed_recipe",
    "recipe_grid",
    "recipe_ablation",
    "paper_exact_recipe",
    "model_specific_recipe",
]


@dataclass(frozen=True)
class PreprocessContract:
    target_transform_policy: str
    x_transform_policy: str
    tcode_policy: str
    target_missing_policy: str
    x_missing_policy: str
    target_outlier_policy: str
    x_outlier_policy: str
    scaling_policy: str
    dimensionality_reduction_policy: str
    feature_selection_policy: str
    preprocess_order: str
    preprocess_fit_scope: str
    inverse_transform_policy: str
    evaluation_scale: str
    feature_selection_semantics: str = "select_before_factor"
    representation_policy: str = "raw_only"
    tcode_application_scope: str = "none"
    target_transform: str = "level"
    target_normalization: str = "none"
    target_domain: str = "unconstrained"
    scaling_scope: str = "columnwise"
    additional_preprocessing: str = "none"
    x_lag_creation: str = "no_x_lags"
    feature_grouping: str = "none"
