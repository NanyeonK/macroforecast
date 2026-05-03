from __future__ import annotations

from .registry import Rule, register_op
from ..types import L1RegimeMetadataArtifact, L3FeaturesArtifact, L4ForecastsArtifact, L4ModelArtifactsArtifact, L4TrainingMetadataArtifact


OPERATIONAL_MODEL_FAMILIES: tuple[str, ...] = (
    "ar_p",
    "ols",
    "ridge",
    "lasso",
    "elastic_net",
    "lasso_path",
    "bayesian_ridge",
    "glmboost",
    "huber",
    "var",
    "factor_augmented_ar",
    "factor_augmented_var",
    "principal_component_regression",
    "decision_tree",
    "random_forest",
    "extra_trees",
    "gradient_boosting",
    "xgboost",
    "lightgbm",
    "catboost",
    "svr_linear",
    "svr_rbf",
    "svr_poly",
    "mlp",
    "lstm",
    "gru",
    "transformer",
    "knn",
    "macroeconomic_random_forest",
    "bvar_minnesota",
    "bvar_normal_inverse_wishart",
    "dfm_mixed_mariano_murasawa",
)

FUTURE_MODEL_FAMILIES: tuple[str, ...] = (
    "midas_almon",
    "midas_beta",
    "midas_step",
    "dfm_unrestricted_midas",
)

MODEL_FAMILY_STATUS = {
    **{family: "operational" for family in OPERATIONAL_MODEL_FAMILIES},
    **{family: "future" for family in FUTURE_MODEL_FAMILIES},
}

SEARCH_ALGORITHMS = ("none", "grid_search", "random_search", "bayesian_optimization", "genetic_algorithm", "cv_path")
FORECAST_STRATEGIES = ("direct", "iterated", "path_average")
TRAINING_START_RULES = ("expanding", "rolling", "fixed")
REFIT_POLICIES = ("every_origin", "every_n_origins", "single_fit")
VALIDATION_METHODS = ("expanding_walk_forward", "rolling_walk_forward", "kfold", "time_series_split")


def get_family_status(family: str) -> str:
    return MODEL_FAMILY_STATUS[family]


def _family_operational(dag, nref) -> bool:
    family = dag.node(nref.node_id).params.get("family")
    return MODEL_FAMILY_STATUS.get(family) == "operational"


def _valid_strategy(dag, nref) -> bool:
    return dag.node(nref.node_id).params.get("forecast_strategy", "direct") in FORECAST_STRATEGIES


@register_op(
    name="fit_model",
    layer_scope=("l4",),
    input_types={"default": (L3FeaturesArtifact, L1RegimeMetadataArtifact)},
    output_type=L4ModelArtifactsArtifact,
    params_schema={
        "family": {"type": str, "default": "ridge", "sweepable": True, "options": OPERATIONAL_MODEL_FAMILIES + FUTURE_MODEL_FAMILIES},
        "forecast_strategy": {"type": str, "default": "direct", "sweepable": True, "options": FORECAST_STRATEGIES},
        "training_start_rule": {"type": str, "default": "expanding", "sweepable": True, "options": TRAINING_START_RULES},
        "refit_policy": {"type": str, "default": "every_origin", "sweepable": True, "options": REFIT_POLICIES},
        "search_algorithm": {"type": str, "default": "none", "sweepable": True, "options": SEARCH_ALGORITHMS},
        "tuning_objective": {"type": str, "default": "cv_mse", "sweepable": True},
        "validation_method": {"type": str, "default": "expanding_walk_forward", "sweepable": True, "options": VALIDATION_METHODS},
    },
    hard_rules=(
        Rule("hard", _family_operational, "model family is future or unknown"),
        Rule("hard", _valid_strategy, "forecast_strategy must be one of direct, iterated, path_average"),
    ),
)
def fit_model(inputs, params):
    """Estimator-builder used by the cache-driven DAG executor.

    Returns the fitted estimator instance directly. The full forecast loop
    (origins, refit policy, training-window slicing) is handled by
    :func:`macrocast.core.runtime.materialize_l4_minimal` which calls
    :func:`macrocast.core.runtime._build_l4_model` per origin. This op
    therefore returns a freshly built estimator so that
    :func:`predict` can fit-then-predict in a single DAG pass.
    """

    from ..runtime import _build_l4_model

    return _build_l4_model(params.get("family", "ridge"), dict(params))


@register_op(
    name="predict",
    layer_scope=("l4",),
    input_types={"default": (L4ModelArtifactsArtifact, L3FeaturesArtifact, L1RegimeMetadataArtifact)},
    output_type=L4ForecastsArtifact,
)
def predict(inputs, params):
    """Pass-through used by cache-driven DAG executors.

    The minimal-runtime forecast tensor materialization happens in
    :func:`macrocast.core.runtime.materialize_l4_minimal`. This op simply
    returns its first input so that explicit DAG edges remain valid.
    """

    return inputs[0] if isinstance(inputs, list) and inputs else inputs


@register_op(
    name="l4_model_artifacts_collect",
    layer_scope=("l4",),
    input_types={"default": L4ModelArtifactsArtifact},
    output_type=L4ModelArtifactsArtifact,
)
def l4_model_artifacts_collect(inputs, params):
    """Aggregate one or more :class:`L4ModelArtifactsArtifact` payloads."""

    if not inputs:
        return None
    if len(inputs) == 1:
        return inputs[0]
    artifacts = {}
    bench = {}
    for item in inputs:
        if hasattr(item, "artifacts"):
            artifacts.update(item.artifacts)
        if hasattr(item, "is_benchmark"):
            bench.update(item.is_benchmark)
    return L4ModelArtifactsArtifact(artifacts=artifacts, is_benchmark=bench)


@register_op(
    name="l4_training_metadata_build",
    layer_scope=("l4",),
    input_types={"default": L4ModelArtifactsArtifact},
    output_type=L4TrainingMetadataArtifact,
)
def l4_training_metadata_build(inputs, params):
    """Return the first incoming training-metadata artifact (no-op aggregator)."""

    return inputs[0] if isinstance(inputs, list) and inputs else inputs
