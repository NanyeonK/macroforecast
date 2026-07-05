"""Pure policy- and window-configuration derivation for the forecasting
runner (Phase 4 of the runner decomposition; bodies moved verbatim from
``macroforecast.forecasting.runner``): forecast/future-feature policy
normalization, the ``_*_for_policy`` FeatureSpec/target-transform derivation,
the recursive feature contract, and per-horizon window derivation.
"""
from __future__ import annotations

import warnings

from dataclasses import replace
from typing import Literal, cast

from macroforecast.feature_engineering import FeatureSpec, feature_spec
from macroforecast.feature_engineering.shared import TargetMode, TargetTransform
from macroforecast.window import ValWindow, WindowSpec

ForecastPolicy = Literal["direct", "direct_average", "path_average", "recursive"]
FutureFeaturePolicy = Literal["target_lags", "observed_future"]

# ``ModelSpec.input_kind == "supervised"`` models whose DOCUMENTED, intended
# construction is an explicit ``feature_spec(predictors=[], target_lags=...)``
# (see the "full study" AR arm in docs/guide/getting_started.md): the
# univariate/factor-augmented autoregression benchmarks. Cross-referenced with
# the same carve-out in ``macroforecast.pipeline.spec.DIRECT_POLICY_GUARD_MODELS``.
# See ``_feature_spec_for_policy`` for where this gates the
# default-feature-spec warning.
_TARGET_LAGS_BY_DESIGN_MODELS = frozenset({"ar", "far"})


def _feature_target_name(features: FeatureSpec) -> str | None:
    if features.target is not None:
        return features.target
    if len(features.targets) == 1:
        return features.targets[0]
    return None


def _normalize_forecast_policy(value: str) -> ForecastPolicy:
    aliases = {
        "direct": "direct",
        "single": "direct",
        "direct_average": "direct_average",
        "direct_avg": "direct_average",
        "average": "direct_average",
        "path_average": "path_average",
        "path_avg": "path_average",
        "path": "path_average",
        "recursive": "recursive",
        "iterated": "recursive",
    }
    if not isinstance(value, str):
        raise TypeError("forecast_policy must be a string")
    key = value.lower().replace("-", "_")
    if key not in aliases:
        raise ValueError(
            "forecast_policy must be one of: direct, direct_average, "
            "path_average, recursive"
        )
    return cast(ForecastPolicy, aliases[key])


def _normalize_future_feature_policy(
    value: str | None,
    *,
    forecast_policy: ForecastPolicy,
) -> FutureFeaturePolicy | None:
    if forecast_policy != "recursive":
        if value is not None:
            raise ValueError("future_feature_policy is only used with recursive forecasting")
        return None
    if value is None:
        return "target_lags"
    aliases = {
        "target_lags": "target_lags",
        "target_lag": "target_lags",
        "target_only": "target_lags",
        "ar": "target_lags",
        "observed_future": "observed_future",
        "oracle": "observed_future",
        "actual_future": "observed_future",
    }
    if not isinstance(value, str):
        raise TypeError("future_feature_policy must be a string")
    key = value.lower().replace("-", "_")
    if key not in aliases:
        raise ValueError(
            "future_feature_policy must be target_lags or observed_future"
        )
    return cast(FutureFeaturePolicy, aliases[key])


def _feature_spec_for_policy(
    features: FeatureSpec | None,
    *,
    target: str | None,
    horizon: int,
    forecast_policy: ForecastPolicy,
    future_feature_policy: FutureFeaturePolicy | None,
    target_transform: str | None,
    model_input_kind: str | None = None,
    model_name: str | None = None,
) -> FeatureSpec:
    # _target_transform_for_policy returns plain str but only ever yields valid
    # TargetTransform literals; likewise target_mode is a TargetMode literal.
    transform = cast(
        TargetTransform,
        _target_transform_for_policy(
            forecast_policy,
            feature_transform=None if features is None else features.target_transform,
            explicit=target_transform,
        ),
    )
    target_mode: TargetMode = "path" if forecast_policy == "path_average" else "direct"
    if features is None:
        if target is None:
            raise ValueError("target is required when data is not a FeatureSet")
        if forecast_policy == "recursive":
            return feature_spec(
                target=target,
                horizon=1,
                predictors=[],
                lags=None,
                target_lags=(0, 1),
                target_mode="direct",
                target_transform=transform,
                metadata={"future_feature_policy": future_feature_policy},
            )
        # Trap: a supervised model (random_forest, ridge, ...) with no explicit
        # FeatureSpec silently gets THIS default -- ``feature_spec(target=...,
        # horizon=..., target_mode=..., target_transform=...)`` with no
        # ``predictors``/``lags``/``target_lags`` -- which resolves at fit time
        # (``_resolve_predictors`` in feature_engineering/shared.py, documented
        # in docs/reference/feature_engineering.md's ``predictors`` row) to ALL
        # other panel columns at lags 0 and 1, with NO feature engineering
        # (no PCA/MARX/scaling) and the target's OWN lags NOT included (verified
        # empirically: ``fs.predictors is None`` -> ``base.predictors`` ("all")
        # -> every non-target column). So the surprise for a wide panel (e.g.
        # FRED-MD's 127 series) is not "too narrow" but "too wide and
        # uncurated, and missing the target's own dynamics" -- easy to misread
        # from ``predictors: ... = None`` alone without tracing the runtime
        # fallback. target/panel-input models (arima, var, dfm_*, ...) never
        # reach this branch at all: they are excluded upstream (input_kind ==
        # "panel" is routed to the panel runner before features are ever
        # resolved; input_kind == "target" models such as arima/ets never call
        # this function with model_input_kind == "supervised"). ``ar``/``far``
        # are the one carve-out WITHIN "supervised": their ``ModelSpec.input_kind``
        # is "supervised" (they fit an (X, y) regression under the hood), and
        # the documented pattern for them (see the "full study" arm in
        # docs/guide/getting_started.md) is an explicit ``target_lags=...``
        # FeatureSpec with ``predictors=[]`` -- the same ar/far carve-out
        # documented in ``macroforecast.pipeline.spec.DIRECT_POLICY_GUARD_MODELS``.
        if model_input_kind == "supervised" and model_name not in _TARGET_LAGS_BY_DESIGN_MODELS:
            _warn_default_feature_spec_used()
        return feature_spec(
            target=target,
            horizon=horizon,
            target_mode=target_mode,
            target_transform=transform,
        )
    if target is not None and features.target is not None and target != features.target:
        raise ValueError("target conflicts with the supplied FeatureSpec target")
    if len(features.targets) > 1:
        raise ValueError("forecasting.run currently supports one target per run")
    if features.horizons and len(features.horizons) > 1:
        raise ValueError(
            "FeatureSpec with multiple horizons should be passed through "
            "forecasting.run(..., horizons=...) so each horizon is fitted separately"
        )
    return replace(
        features,
        target=target or features.target,
        horizon=1 if forecast_policy == "recursive" else horizon,
        horizons=(),
        target_mode=target_mode,
        target_transform=transform,
    )


# Feature target-transforms that indicate the panel has ALREADY been
# differenced to stationarity. Only on such inputs does a change-based default
# target_transform double-difference; on raw/level panels it is correct.
_ALREADY_STATIONARY_TARGET_TRANSFORMS = frozenset(
    {
        "change",
        "growth",
        "log_growth",
        "log_change",
        "pct_change",
        "difference",
        "log_difference",
    }
)


def _warn_change_based_target_default(
    transform: str, feature_transform: str | None
) -> None:
    # Gate the warning on evidence that the panel is already stationary-
    # transformed. Firing on raw/level panels (where average_change/change is the
    # correct target) would be a false positive that trains users to ignore it.
    if feature_transform not in _ALREADY_STATIONARY_TARGET_TRANSFORMS:
        return
    warnings.warn(
        "forecast_policy yields a change-based target_transform "
        f"({transform!r}) by default while the features use an already-"
        f"stationary transform ({feature_transform!r}); this double-differences "
        "the target. Pass an explicit value-based target_transform "
        "('average_value' for direct_average, 'value' for path_average) to build "
        "averages from the one-period transformed series.",
        UserWarning,
        stacklevel=3,
    )


def _warn_default_feature_spec_used() -> None:
    # Identical message + call site each time -> Python's default warning
    # filter dedups repeats (same message/category/module/lineno), so a 60-cell
    # pipeline grid built from the same code path warns once, not 60 times.
    #
    # Message verified against the actual resolved FeatureSpec (not just this
    # function's own predictors=None default): ``_resolve_predictors`` falls
    # back to ``base.predictors``, whose own default is ``"all"``
    # (docs/reference/feature_engineering.md's ``predictors`` row), so this
    # branch's implicit FeatureSpec ends up using EVERY other panel column at
    # lags 0/1 with no feature engineering -- and, because ``target_lags`` is
    # never set here, none of the target's own history.
    warnings.warn(
        "arm used the implicit default feature spec (every other panel column "
        "at lags 0 and 1, no feature engineering -- and NOT the target's own "
        "lags) -- pass features=feature_spec(...) for explicit control over "
        "which predictors are used",
        UserWarning,
        stacklevel=3,
    )


def _target_transform_for_policy(
    forecast_policy: ForecastPolicy,
    *,
    feature_transform: str | None,
    explicit: str | None,
) -> str:
    if forecast_policy == "direct_average":
        if explicit is not None:
            return explicit if explicit.startswith("average_") else f"average_{explicit}"
        if feature_transform and str(feature_transform).startswith("average_"):
            return feature_transform
        _warn_change_based_target_default("average_change", feature_transform)
        return "average_change"
    if forecast_policy == "path_average":
        if explicit is not None:
            return explicit
        if feature_transform and feature_transform != "level":
            return feature_transform
        _warn_change_based_target_default("change", feature_transform)
        return "change"
    if forecast_policy == "recursive":
        if explicit is not None:
            return explicit
        if feature_transform and str(feature_transform).startswith("average_"):
            raise ValueError("recursive forecasting does not support average_* target transforms")
        return feature_transform or "level"
    if explicit is not None:
        return explicit
    return feature_transform or "level"


def _validate_recursive_feature_contract(
    features: FeatureSpec,
    *,
    future_feature_policy: FutureFeaturePolicy | None,
) -> None:
    target = _feature_target_name(features)
    if target is None:
        raise ValueError("recursive forecasting requires exactly one target")
    transform = str(features.target_transform)
    if transform.startswith("average_"):
        raise ValueError("recursive forecasting does not support average_* target transforms")
    if future_feature_policy == "observed_future":
        return
    if features.predictors != ():
        raise ValueError(
            "recursive forecasting with future_feature_policy='target_lags' "
            "requires FeatureSpec predictors to be empty and target_lags to "
            "declare the autoregressive inputs. Use future_feature_policy="
            "'observed_future' for an explicit oracle/scenario path with "
            "exogenous future predictors."
        )
    if not features.target_lags:
        raise ValueError(
            "recursive forecasting with future_feature_policy='target_lags' "
            "requires FeatureSpec target_lags"
        )
    if 0 not in features.target_lags:
        raise ValueError(
            "recursive forecasting with future_feature_policy='target_lags' "
            "requires target_lags to include 0 so predicted target values can "
            "feed the next step under macroforecast's row-date convention"
        )
    if features.feature_steps:
        raise ValueError(
            "recursive target_lags currently supports FeatureSpec shortcut lags "
            "only; feature_steps need a future-step registry before they can be "
            "updated recursively"
        )
    if features.rolling_windows or features.pca_components is not None:
        raise ValueError(
            "recursive target_lags currently supports target lag features and "
            "optional deterministic time features only"
        )


def _horizon_val_window(val: "ValWindow", horizon: int) -> "ValWindow":
    """Re-derive the train/validation embargo for an h-step target.

    ``window.from_cutoffs`` defaults ``val_embargo`` to ``horizon - 1`` (the
    standard h-step purge that keeps training labels from realising inside the
    validation block). When a multi-horizon run injects the per-horizon test
    horizon into a base window built at horizon 1 (the consolidated-spec path used
    to share the per-origin EM across horizons), the validation embargo must be
    re-derived for the new horizon too -- otherwise the validation splits for h>1
    keep the h=1 purge (embargo 0) and leak the h-step training labels into the
    validation fold (and, downstream, fail feature alignment). We therefore set the
    val embargo to ``max(current, horizon - 1)`` so the per-horizon window matches
    a window that was built directly with ``from_cutoffs(horizon=h)``.
    """
    current = val.embargo if val.embargo is not None else 0
    return replace(val, embargo=max(int(current), max(0, int(horizon) - 1)))


def _feature_window_for_policy(window_spec: WindowSpec, horizon: int) -> WindowSpec:
    """Use h for origin cutoff while fitting one feature row per origin."""

    return replace(
        window_spec,
        test=replace(window_spec.test, horizon=int(horizon)),
        val=_horizon_val_window(window_spec.val, int(horizon)),
        horizon=int(horizon),
    )


def _panel_window_for_horizon(window_spec: WindowSpec, horizon: int) -> WindowSpec:
    return replace(
        window_spec,
        test=replace(window_spec.test, horizon=int(horizon)),
        val=_horizon_val_window(window_spec.val, int(horizon)),
        horizon=int(horizon),
    )
