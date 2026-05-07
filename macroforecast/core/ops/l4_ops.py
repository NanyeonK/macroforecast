from __future__ import annotations

from .registry import Rule, register_op
from ..status import FUTURE, OPERATIONAL, ItemStatus, is_runnable
from ..types import L1RegimeMetadataArtifact, L3FeaturesArtifact, L4ForecastsArtifact, L4ModelArtifactsArtifact, L4TrainingMetadataArtifact


# Families whose runtime fully matches the published procedure named in
# the design (see ``plans/design/part2_l2_l3_l4.md`` § L4 Model family
# library). Schema-validates and runs end-to-end.
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
    # Promoted in v0.9.1 dev-stage v0.9.0F (audit-fix for paper 16,
    # Coulombe et al. 2022 JAE): closed-form non-linear ridge in the
    # dual via sklearn ``KernelRidge``.
    "kernel_ridge",
    "mlp",
    "lstm",
    "gru",
    "transformer",
    "knn",
    # Promoted in v0.2 (issue #185 / #186): closed-form Minnesota /
    # Normal-Inverse-Wishart posterior mean estimator.
    "bvar_minnesota",
    "bvar_normal_inverse_wishart",
    # Promoted in v0.2 (issue #184): two-stage Bernanke-Boivin-Eliasz
    # FAVAR (PCA factors + VAR(p) on (factors, target)).
    "factor_augmented_var",
    # Promoted in v0.2 (issue #187): Coulombe (2024) GTVP via per-leaf
    # local linear regressions on top of a time-aware random forest.
    "macroeconomic_random_forest",
    # Promoted in v0.2 (issue #188): Mariano-Murasawa-style dynamic factor
    # model via statsmodels' DynamicFactor (Kalman state-space MLE).
    "dfm_mixed_mariano_murasawa",
    # New in v0.3:
    # - quantile_regression_forest: Meinshausen (2006) -- record per-leaf
    #   training-target distributions and forecast arbitrary quantiles.
    # - bagging: bootstrap-aggregated wrapper around any base family for
    #   variance reduction; per-bag predictions surface as quantile
    #   intervals.
    "quantile_regression_forest",
    "bagging",
    # New in v0.9 (Phase 2 paper-coverage pass):
    # - mars: Multivariate Adaptive Regression Splines (Friedman 1991).
    #   Atomic non-linear basis-function regression with no sklearn
    #   analogue. Runtime wraps ``pyearth`` as an optional dep
    #   (``pip install macroforecast[mars]``); raises NotImplementedError
    #   with a clear hint when the extra is missing -- mirrors the
    #   xgboost / lightgbm / catboost / deep optional-dep pattern.
    "mars",
)

# Families whose v0.1 runtime did *not* faithfully implement the design's
# named procedure (see PR #163 codex review). Validator hard-rejects these
# so users do not silently receive numbers from the wrong algorithm.
# Each item has a tracking issue for v0.2 implementation; see
# ``plans/design/part2_l2_l3_l4.md`` § L4 for the gap description.
#
# - factor_augmented_var: no runtime wrapper at all (was a silent
#   ``NotImplementedError`` at fit time).
# - bvar_minnesota / bvar_normal_inverse_wishart: ``_BayesianVAR`` wrapper
#   delegates to plain ``_VARWrapper`` and does *not* apply Minnesota /
#   normal-inverse-Wishart prior shrinkage.
# - macroeconomic_random_forest: re-anchored in v0.8.9 to
#   ``_MRFExternalWrapper``, which delegates to Ryan Lucas's reference
#   implementation of Goulet Coulombe 2024 MRF, vendored under
#   ``macroforecast/_vendor/macro_random_forest/`` with surgical
#   numpy 2.x / pandas 2.x compatibility patches. The previous in-house
#   ``_MRFWrapper`` only implemented the per-leaf linear piece and was
#   missing both the random-walk regularisation and the Block Bayesian
#   Bootstrap forecast ensembles. Upstream:
#   https://github.com/RyanLucas3/MacroRandomForest.
# - dfm_mixed_mariano_murasawa: ``_DFMMixedFrequency`` is a PCA + AR(1)
#   approximation, not the Mariano-Murasawa Kalman state-space EM.
FUTURE_MODEL_FAMILIES: tuple[str, ...] = (
    # always future (Phase 1 design): MIDAS family awaits its own runtime.
    "midas_almon",
    "midas_beta",
    "midas_step",
    "dfm_unrestricted_midas",
)


# Back-compat shim: previous releases exposed ``PLANNED_MODEL_FAMILIES``
# as the bucket for "operational schema + approximation runtime". The
# v0.1 honesty pass collapses planned -> future. The empty tuple is kept
# so external imports do not crash; new code should use
# ``FUTURE_MODEL_FAMILIES`` and check :func:`get_family_status`.
PLANNED_MODEL_FAMILIES: tuple[str, ...] = ()


MODEL_FAMILY_STATUS: dict[str, ItemStatus] = {
    **{family: OPERATIONAL for family in OPERATIONAL_MODEL_FAMILIES},
    **{family: FUTURE for family in FUTURE_MODEL_FAMILIES},
}


SEARCH_ALGORITHMS = (
    "none",
    "grid_search",
    "random_search",
    "bayesian_optimization",
    "genetic_algorithm",
    "cv_path",
    # Coulombe-Surprenant-Leroux-Stevanovic (2022 JAE) Feature 3 schemes.
    # v0.9.0a0 audit-fix: previously these strings were silently dropped
    # because the validator's options enum did not list them, so the
    # paper's Feature-3 treatment effect was structurally identifiable
    # as zero by construction. ``_resolve_l4_tuning`` now branches on
    # each one for alpha-tunable linear families.
    "kfold",
    "poos",
    "aic",
    "bic",
    # Goulet Coulombe / Klieber / Barrette / Goebel (2024) Albacore §3.
    # Non-overlapping block CV: split T into K contiguous non-shuffled
    # blocks, hold each out in turn. Distinct from ``kfold`` (random
    # shuffle) and from ``TimeSeriesSplit`` (expanding sequential).
    "block_cv",
)
FORECAST_STRATEGIES = ("direct", "iterated", "path_average")
TRAINING_START_RULES = ("expanding", "rolling", "fixed")
REFIT_POLICIES = ("every_origin", "every_n_origins", "single_fit")
VALIDATION_METHODS = ("expanding_walk_forward", "rolling_walk_forward", "kfold", "time_series_split")


def get_family_status(family: str) -> ItemStatus:
    """Return the canonical 2-value :class:`ItemStatus` for ``family``.

    Compare against :data:`macroforecast.core.status.OPERATIONAL` /
    :data:`macroforecast.core.status.FUTURE` rather than string literals.
    """

    return MODEL_FAMILY_STATUS.get(family, OPERATIONAL)


def _family_operational(dag, nref) -> bool:
    """Validator predicate: only ``operational`` families pass.

    Future families (incl. the v0.1-honesty demotions: factor_augmented_var,
    BVAR Minnesota / NIW, macroeconomic_random_forest,
    dfm_mixed_mariano_murasawa) are hard-rejected at recipe validation time
    with a pointer at the v0.2 implementation issue.
    """

    family = dag.node(nref.node_id).params.get("family")
    return is_runnable(MODEL_FAMILY_STATUS.get(family))


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
        Rule("hard", _family_operational, "model family is future or unknown -- see v0.2 implementation tracker on GitHub"),
        Rule("hard", _valid_strategy, "forecast_strategy must be one of direct, iterated, path_average"),
    ),
)
def fit_model(inputs, params):
    """Estimator-builder used by the cache-driven DAG executor.

    Returns the fitted estimator instance directly. The full forecast loop
    (origins, refit policy, training-window slicing) is handled by
    :func:`macroforecast.core.runtime.materialize_l4_minimal` which calls
    :func:`macroforecast.core.runtime._build_l4_model` per origin. This op
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
    :func:`macroforecast.core.runtime.materialize_l4_minimal`. This op simply
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
