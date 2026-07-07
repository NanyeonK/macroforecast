from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
import difflib
import inspect
from typing import Any, Literal, cast

import pandas as pd
import numpy as np

from macroforecast.models.assemblage import (
    albacore_components,
    albacore_ranks,
    assemblage_regression,
    component_aggregation,
    rank_aggregation,
    supervised_aggregation,
)
from macroforecast.models.linear import (
    adaptive_elastic_net,
    adaptive_lasso,
    bayesian_ridge,
    elastic_net,
    fused_difference_ridge,
    glmboost,
    group_lasso,
    huber,
    lasso,
    nonneg_ridge,
    ols,
    pls,
    random_walk_ridge,
    ridge,
    scaled_pca,
    shrink_to_target_ridge,
    sparse_group_lasso,
    supervised_pca,
    supervised_scaled_pca,
)
from macroforecast.models.model_averaging import csr, jma
from macroforecast.models.neural import density_hnn, gru, hemisphere_nn, lstm, nn, transformer
from macroforecast.models.nonparametric import kernel_ridge, knn
from macroforecast.models.spline import mars
from macroforecast.models.svm import linear_svr, nu_svr, svr
from macroforecast.models.timeseries import (
    ar,
    naive,
    seasonal_naive,
    random_walk_drift,
    stlf,
    arima,
    auto_arima,
    bvar_minnesota,
    bvar_normal_inverse_wishart,
    dfm_mixed_mariano_murasawa,
    dfm_unrestricted_midas,
    ets,
    far,
    favar,
    holt_winters,
    midas_almon,
    midas_beta,
    midas_step,
    restricted_midas,
    theta_method,
    unrestricted_midas,
    var,
)
from macroforecast.models.tree import (
    catboost,
    decision_tree,
    extra_trees,
    gradient_boosting,
    lgba_plus,
    lgb_plus,
    lightgbm,
    macro_random_forest,
    quantile_regression_forest,
    random_forest,
    xgboost,
)
from macroforecast.models.tvp import tvp_ridge
from macroforecast.models.volatility import (
    egarch,
    garch11,
    gjr_garch,
    realized_garch,
    tgarch,
)

InputKind = Literal["supervised", "target", "panel", "volatility"]
SearchSpace = dict[str, tuple[Any, ...]]
SearchSpaces = dict[str, SearchSpace]
_ALLOWED_INPUT_KINDS = frozenset({"supervised", "target", "panel", "volatility"})


@dataclass(frozen=True)
class ModelParameter:
    """One model-owned parameter description."""

    name: str
    default: Any
    kind: str
    description: str
    tunable: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready parameter description."""

        return {
            "name": self.name,
            "default": _json_ready(self.default),
            "kind": self.kind,
            "description": self.description,
            "tunable": self.tunable,
        }


@dataclass(frozen=True)
class ModelSpec:
    """Callable model plus model-owned defaults and hyperparameter spaces."""

    name: str
    family: str
    fit_func: Callable[..., Any]
    default_params: dict[str, Any] = field(default_factory=dict)
    parameters: tuple[ModelParameter, ...] = ()
    search_spaces: SearchSpaces = field(default_factory=dict)
    default_search_method: str = "grid"
    default_preset: str = "standard"
    input_kind: InputKind = "supervised"
    preset: str = "standard"
    params: dict[str, Any] = field(default_factory=dict)
    backend: str = "internal"
    requires_extra: str | None = None
    requires_scaling: bool = False
    recommended_preprocessing: tuple[str, ...] = ()
    description: str = ""
    selection_method: str = "cv"

    def with_preset(self, preset: str) -> "ModelSpec":
        """Return the same model spec with a different hyperparameter preset."""

        if self.search_spaces and preset not in self.search_spaces:
            allowed = ", ".join(sorted(self.search_spaces))
            raise ValueError(
                f"{self.name!r} has no preset {preset!r}. Available presets: {allowed}."
            )
        return replace(self, preset=preset)

    def with_params(self, **params: Any) -> "ModelSpec":
        """Return the same model spec with fixed model parameters."""

        return replace(self, params={**self.params, **params})

    def search_space(self, preset: str | None = None) -> dict[str, tuple[Any, ...]]:
        """Return the model-owned hyperparameter space for one preset."""

        key = preset or self.preset or self.default_preset
        if not self.search_spaces:
            return {}
        if key not in self.search_spaces:
            allowed = ", ".join(sorted(self.search_spaces))
            raise ValueError(
                f"{self.name!r} has no preset {key!r}. Available presets: {allowed}."
            )
        return {name: tuple(values) for name, values in self.search_spaces[key].items()}

    def all_params(self, **params: Any) -> dict[str, Any]:
        """Merge default, fixed, and trial parameters."""

        return {**self.default_params, **self.params, **params}

    def fit(self, X: Any, y: Any | None = None, **params: Any) -> Any:
        """Fit the model according to the model's input convention."""

        merged = self.all_params(**params)
        if self.input_kind == "supervised":
            return self.fit_func(X, y, **merged)
        if self.input_kind == "target":
            target = X if y is None else y
            return self.fit_func(target, **merged)
        if self.input_kind == "panel":
            if y is not None:
                target = pd.Series(y).rename("__target__")
                panel = pd.concat([target, pd.DataFrame(X)], axis=1)
                if merged.get("target") is None:
                    merged["target"] = "__target__"
                return self.fit_func(panel, **merged)
            return self.fit_func(X, **merged)
        if self.input_kind == "volatility":
            target = X if y is None else y
            exog = None if y is None else X
            return self.fit_func(target, X=exog, **merged)
        raise ValueError(f"Unknown input_kind {self.input_kind!r}")

    def __call__(self, X: Any, y: Any | None = None, **params: Any) -> Any:
        return self.fit(X, y, **params)

    def describe(self) -> pd.DataFrame:
        """Return parameter documentation as a DataFrame."""

        rows = []
        for parameter in self.parameters:
            row: dict[str, Any] = {
                "model": self.name,
                "parameter": parameter.name,
                "default": parameter.default,
                "kind": parameter.kind,
                "tunable": parameter.tunable,
                "description": parameter.description,
            }
            for preset, space in self.search_spaces.items():
                row[f"{preset}_space"] = space.get(parameter.name)
            rows.append(row)
        return pd.DataFrame(rows)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready model specification."""

        return {
            "name": self.name,
            "family": self.family,
            "fit_func": _callable_name(self.fit_func),
            "default_params": _json_ready(self.default_params),
            "parameters": [parameter.to_dict() for parameter in self.parameters],
            "search_spaces": _json_ready(self.search_spaces),
            "default_search_method": self.default_search_method,
            "default_preset": self.default_preset,
            "input_kind": self.input_kind,
            "preset": self.preset,
            "params": _json_ready(self.params),
            "backend": self.backend,
            "requires_extra": self.requires_extra,
            "requires_scaling": self.requires_scaling,
            "recommended_preprocessing": _json_ready(self.recommended_preprocessing),
            "description": self.description,
        }

    def to_metadata(self) -> dict[str, Any]:
        """Return compact model metadata for selection and forecasting runners."""

        return {
            "model": self.name,
            "model_family": self.family,
            "model_preset": self.preset,
            "input_kind": self.input_kind,
            "backend": self.backend,
            "requires_extra": self.requires_extra,
            "requires_scaling": self.requires_scaling,
            "recommended_preprocessing": _json_ready(self.recommended_preprocessing),
            "default_search_method": self.default_search_method,
            "default_params": _json_ready(self.default_params),
            "params": _json_ready(self.params),
            "search_space": _json_ready(self.search_space()),
        }


def get_model(
    model: str | Callable[..., Any] | ModelSpec,
    *,
    preset: str | None = None,
    params: Mapping[str, Any] | None = None,
) -> ModelSpec:
    """Return a model spec by name, callable, or existing spec."""

    if isinstance(model, ModelSpec):
        spec = model
    elif isinstance(model, str):
        key = model.lower()
        if key not in MODEL_SPECS:
            allowed = ", ".join(sorted(MODEL_SPECS))
            suggestions = difflib.get_close_matches(key, sorted(MODEL_SPECS), n=3)
            hint = (
                " Did you mean "
                + ", ".join(repr(name) for name in suggestions)
                + "?"
                if suggestions
                else ""
            )
            raise ValueError(
                f"Unknown model {model!r}.{hint} Available models: {allowed}."
            )
        spec = MODEL_SPECS[key]
    elif callable(model):
        callable_spec = _MODEL_SPECS_BY_CALLABLE.get(model)
        if callable_spec is None:
            name = getattr(model, "__name__", repr(model))
            raise ValueError(
                f"No registered ModelSpec for callable {name!r}; wrap it: "
                f"mf.models.custom_model('my_model', {name})"
            )
        spec = callable_spec
    else:
        raise TypeError("model must be a model name, callable, or ModelSpec")
    if preset is not None:
        spec = spec.with_preset(preset)
    if params:
        spec = spec.with_params(**dict(params))
    return spec


def custom_model(
    name: str,
    fit_func: Callable[..., Any],
    *,
    family: str = "custom",
    default_params: Mapping[str, Any] | None = None,
    parameters: tuple[ModelParameter, ...] = (),
    search_spaces: SearchSpaces | None = None,
    default_search_method: str = "grid",
    default_preset: str | None = "standard",
    input_kind: InputKind = "supervised",
    backend: str = "custom",
    requires_extra: str | None = None,
    requires_scaling: bool = False,
    recommended_preprocessing: tuple[str, ...] = (),
    description: str | None = None,
    mf_digest: str | None = None,
) -> ModelSpec:
    """Build a user-owned ``ModelSpec`` without registering a package model.

    ``fit_func`` must be callable and accept the inputs required by
    ``input_kind`` (``X, y`` for supervised models; at least the target/panel
    object for target, panel, and volatility models). ``input_kind`` is validated
    at construction, and ``default_preset`` must name a key in ``search_spaces``
    when search spaces are supplied. Pass ``mf_digest=`` when this model should
    be reusable through ``pipeline_spec(result_store=...)``; the digest is stamped
    on ``fit_func.__mf_digest__``.
    """

    if not name:
        raise ValueError("custom model name must be non-empty")
    if not callable(fit_func):
        raise TypeError("custom model fit_func must be callable")
    if input_kind not in _ALLOWED_INPUT_KINDS:
        raise ValueError(
            f"custom model input_kind must be one of {sorted(_ALLOWED_INPUT_KINDS)}, "
            f"got {input_kind!r}"
        )
    try:
        signature = inspect.signature(fit_func)
    except (TypeError, ValueError) as exc:
        raise TypeError("custom model fit_func must have an inspectable signature") from exc
    positional = [
        param
        for param in signature.parameters.values()
        if param.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    minimum_positional = 2 if input_kind == "supervised" else 1
    has_varargs = any(
        param.kind is inspect.Parameter.VAR_POSITIONAL
        for param in signature.parameters.values()
    )
    if len(positional) < minimum_positional and not has_varargs:
        raise TypeError(
            f"custom model fit_func for input_kind={input_kind!r} must accept at "
            f"least {minimum_positional} positional argument(s)"
        )
    search_space_dict = dict(search_spaces or {})
    if default_preset is None:
        if search_space_dict:
            raise ValueError(
                "custom model default_preset=None is ambiguous when search_spaces "
                "is provided; pass one of the search_spaces keys"
            )
        default_preset_value = "standard"
    else:
        default_preset_value = str(default_preset)
    if search_space_dict and default_preset_value not in search_space_dict:
        raise ValueError(
            f"custom model default_preset {default_preset_value!r} is not in "
            f"search_spaces keys {sorted(search_space_dict)}"
        )
    if mf_digest is not None:
        setattr(fit_func, "__mf_digest__", str(mf_digest))
    return ModelSpec(
        name=str(name),
        family=str(family),
        fit_func=fit_func,
        default_params=dict(default_params or {}),
        parameters=parameters,
        search_spaces=search_space_dict,
        default_search_method=str(default_search_method),
        default_preset=default_preset_value,
        preset=default_preset_value,
        input_kind=cast("InputKind", input_kind),
        backend=str(backend),
        requires_extra=requires_extra,
        requires_scaling=bool(requires_scaling),
        recommended_preprocessing=tuple(recommended_preprocessing),
        description=description or f"User supplied model {_callable_name(fit_func)}.",
    )


def list_model_specs(*, family: str | None = None) -> pd.DataFrame:
    """List registered model specs."""

    rows = []
    for spec in MODEL_SPECS.values():
        if family is not None and spec.family != family:
            continue
        rows.append(
            {
                "name": spec.name,
                "family": spec.family,
                "input_kind": spec.input_kind,
                "backend": spec.backend,
                "requires_extra": spec.requires_extra,
                "requires_scaling": spec.requires_scaling,
                "recommended_preprocessing": spec.recommended_preprocessing,
                "default_search_method": spec.default_search_method,
                "default_preset": spec.default_preset,
                "presets": tuple(spec.search_spaces),
                "n_tunable": sum(parameter.tunable for parameter in spec.parameters),
                "description": spec.description,
            }
        )
    return pd.DataFrame(rows)


def describe_model(model: str | Callable[..., Any] | ModelSpec) -> pd.DataFrame:
    """Describe model-owned parameters and preset spaces."""

    return get_model(model).describe()


def model_search_space(
    model: str | Callable[..., Any] | ModelSpec,
    *,
    preset: str | None = None,
) -> dict[str, tuple[Any, ...]]:
    """Return a model-owned hyperparameter space."""

    return get_model(model, preset=preset).search_space()


def _p(
    name: str, default: Any, kind: str, description: str, tunable: bool = True
) -> ModelParameter:
    return ModelParameter(name, default, kind, description, tunable)


def _callable_name(func: Callable[..., Any]) -> str:
    module = getattr(func, "__module__", "")
    qualname = getattr(func, "__qualname__", getattr(func, "__name__", repr(func)))
    return f"{module}.{qualname}" if module else str(qualname)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def _spec(
    name: str,
    family: str,
    fit_func: Callable[..., Any],
    *,
    default_params: dict[str, Any] | None = None,
    parameters: tuple[ModelParameter, ...] = (),
    spaces: SearchSpaces | None = None,
    method: str = "grid",
    input_kind: InputKind = "supervised",
    backend: str = "internal",
    requires_extra: str | None = None,
    requires_scaling: bool = False,
    recommended_preprocessing: tuple[str, ...] = (),
    description: str = "",
    selection_method: str = "cv",
) -> ModelSpec:
    return ModelSpec(
        name=name,
        family=family,
        fit_func=fit_func,
        selection_method=selection_method,
        default_params=dict(default_params or {}),
        parameters=parameters,
        search_spaces=dict(spaces or {}),
        default_search_method=method,
        input_kind=input_kind,
        backend=backend,
        requires_extra=requires_extra,
        requires_scaling=requires_scaling,
        recommended_preprocessing=tuple(recommended_preprocessing),
        description=description,
    )


_ALPHA_SPACES: SearchSpaces = {
    "small": {"alpha": (0.01, 0.1, 1.0)},
    "standard": {"alpha": (0.001, 0.01, 0.1, 1.0, 10.0)},
    "wide": {"alpha": (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0)},
}

_TREE_SPACES: SearchSpaces = {
    "small": {"max_depth": (3, 5, None), "min_samples_leaf": (1, 3)},
    "standard": {"max_depth": (3, 5, 10, None), "min_samples_leaf": (1, 3, 5)},
    "wide": {
        "max_depth": (2, 3, 5, 10, 20, None),
        "min_samples_leaf": (1, 2, 3, 5, 10),
    },
}

_FOREST_SPACES: SearchSpaces = {
    "small": {
        "n_estimators": (50, 100),
        "max_depth": (3, 5, None),
        "min_samples_leaf": (1, 3),
    },
    "standard": {
        "n_estimators": (100, 200, 500),
        "max_depth": (3, 5, 10, None),
        "min_samples_leaf": (1, 3, 5),
    },
    "wide": {
        "n_estimators": (100, 200, 500, 1000),
        "max_depth": (3, 5, 10, 20, None),
        "min_samples_leaf": (1, 2, 3, 5, 10),
    },
}

_BOOSTING_SPACES: SearchSpaces = {
    "small": {
        "n_estimators": (50, 100),
        "learning_rate": (0.05, 0.1),
        "max_depth": (2, 3),
    },
    "standard": {
        "n_estimators": (100, 200, 500),
        "learning_rate": (0.03, 0.05, 0.1),
        "max_depth": (2, 3, 5),
    },
    "wide": {
        "n_estimators": (100, 200, 500, 1000),
        "learning_rate": (0.01, 0.03, 0.05, 0.1),
        "max_depth": (2, 3, 5, 8),
    },
}

_FACTOR_SPACES: SearchSpaces = {
    "small": {"n_factors": (1, 2, 3)},
    "standard": {"n_factors": (1, 2, 3, 5, 8)},
    "wide": {"n_factors": (1, 2, 3, 5, 8, 10, 12)},
}

_AR_SPACES: SearchSpaces = {
    "small": {"n_lag": (1, 2, 4)},
    "standard": {"n_lag": (1, 2, 4, 6, 12)},
    "wide": {"n_lag": (1, 2, 3, 4, 6, 9, 12, 18, 24)},
}

# favar's own (n_factors, n_lag) grid -- NOT the shared _FACTOR_SPACES/_AR_SPACES
# above. favar's Gibbs/Wishart posterior sampler cost per fit grows sharply with
# both n_factors and n_lag (the coefficient-draw covariance is a
# ((n_factors + 1) * n_lag)-square system rebuilt every MCMC iteration), unlike
# the OLS-based "far"/"ar"/"var" models that also use _FACTOR_SPACES/_AR_SPACES.
# The shared "standard" corner n_factors=8/n_lag=12 alone takes minutes per fit
# even with a handful of draws, which makes the default "standard"-preset grid
# search (25 combos, exhaustive) pathologically slow for favar specifically.
# Capping favar's own ranges keeps search dimensionality at 2 (n_factors, n_lag)
# unchanged while keeping every combo affordable; "wide" is deliberately not as
# rich as _FACTOR_SPACES/_AR_SPACES' "wide" for the same reason.
_FAVAR_SPACES: SearchSpaces = {
    "small": {"n_factors": (1, 2), "n_lag": (1, 2)},
    "standard": {"n_factors": (1, 2), "n_lag": (1, 2)},
    "wide": {"n_factors": (1, 2, 3, 5), "n_lag": (1, 2, 4, 6)},
}

_SVM_SPACES: SearchSpaces = {
    "small": {"C": (0.1, 1.0), "epsilon": (0.01, 0.1), "gamma": ("scale",)},
    "standard": {
        "C": (0.1, 1.0, 10.0),
        "epsilon": (0.01, 0.1, 0.2),
        "gamma": ("scale", "auto"),
    },
    "wide": {
        "C": (0.01, 0.1, 1.0, 10.0, 100.0),
        "epsilon": (0.001, 0.01, 0.1, 0.2),
        "gamma": ("scale", "auto"),
    },
}

_NN_SPACES: SearchSpaces = {
    "small": {
        "hidden_layer_sizes": ((32,), (64,)),
        "dropout": (0.0,),
        "learning_rate": (0.001,),
        "weight_decay": (0.0, 0.0001),
    },
    "standard": {
        "hidden_layer_sizes": ((64,), (100,), (64, 32)),
        "dropout": (0.0, 0.1),
        "learning_rate": (0.0005, 0.001),
        "weight_decay": (0.0, 0.0001, 0.001),
    },
    "wide": {
        "hidden_layer_sizes": ((32,), (64,), (100,), (128,), (100, 50), (128, 64)),
        "dropout": (0.0, 0.1, 0.25),
        "learning_rate": (0.0001, 0.0005, 0.001, 0.005),
        "weight_decay": (0.0, 0.00001, 0.0001, 0.001, 0.01),
    },
}

_RNN_SPACES: SearchSpaces = {
    "small": {
        "sequence_length": (2, 4),
        "hidden_size": (16, 32),
        "learning_rate": (0.001,),
    },
    "standard": {
        "sequence_length": (2, 4, 8),
        "hidden_size": (16, 32, 64),
        "learning_rate": (0.0005, 0.001),
    },
    "wide": {
        "sequence_length": (2, 4, 8, 12),
        "hidden_size": (16, 32, 64, 128),
        "learning_rate": (0.0001, 0.0005, 0.001, 0.005),
    },
}

_HNN_SPACES: SearchSpaces = {
    "small": {
        "neurons": (16, 32),
        "n_estimators": (3, 5),
        "learning_rate": (0.001,),
    },
    "standard": {
        # Trimmed 2026-07-05 (WP-B4, default-cost fix): (32, 64) -> (16, 32) /
        # (5, 10) -> (3, 5). `model_selection=None` still runs a 20-trial random
        # search over this preset even with no explicit tuning request (see
        # `_selection_for_model`/`_fit_one_model_at_origin` in
        # `forecasting/policies/base.py`), so the old preset multiplied a
        # 20-trial search across 4 origins x 4 policies into a scan that did
        # not finish inside a 900s timeout. See CHANGELOG for old->new values
        # and measured before/after scan costs. "wide" is unchanged and still
        # reaches the old (32, 64) / (5, 10) corner for deep/explicit use.
        "neurons": (16, 32),
        "n_estimators": (3, 5),
        "learning_rate": (0.0005, 0.001),
    },
    "wide": {
        "neurons": (32, 64, 128),
        "n_estimators": (5, 10, 25),
        "learning_rate": (0.0001, 0.0005, 0.001),
    },
}

_DENSITY_HNN_SPACES: SearchSpaces = {
    "small": {
        "neurons": (16, 32),
        "n_estimators": (3, 5),
        "prior_estimators": (2, 3),
        "learning_rate": (0.001,),
    },
    "standard": {
        # Trimmed 2026-07-05 (WP-B4, default-cost fix): (32, 64) -> (16, 32) /
        # (5, 10) -> (3, 5) / (3, 5) -> (2, 3). Same rationale as _HNN_SPACES
        # above -- `model_selection=None` still runs a 20-trial random search
        # over this preset. "wide" is unchanged and still reaches the old
        # (32, 64) / (5, 10) / (3, 5) corner for deep/explicit use.
        "neurons": (16, 32),
        "n_estimators": (3, 5),
        "prior_estimators": (2, 3),
        "learning_rate": (0.0005, 0.001),
    },
    "wide": {
        "neurons": (32, 64, 128),
        "n_estimators": (5, 10, 25),
        "prior_estimators": (3, 5, 10),
        "learning_rate": (0.0001, 0.0005, 0.001),
    },
}

MODEL_SPECS: dict[str, ModelSpec] = {
    "ols": _spec(
        "ols",
        "linear",
        ols,
        backend="sklearn.linear_model.LinearRegression",
        description="Ordinary least squares with no model-owned tuning space.",
    ),
    "ridge": _spec(
        "ridge",
        "linear",
        ridge,
        default_params={"alpha": 1.0},
        parameters=(_p("alpha", 1.0, "float", "L2 penalty strength."),),
        spaces=_ALPHA_SPACES,
        method="cv_path",
        backend="sklearn.linear_model.Ridge",
        description="Ridge regression.",
    ),
    "nonneg_ridge": _spec(
        "nonneg_ridge",
        "linear",
        nonneg_ridge,
        default_params={"alpha": 1.0, "fit_intercept": True},
        parameters=(
            _p("alpha", 1.0, "float", "L2 penalty strength."),
            _p(
                "fit_intercept",
                True,
                "bool",
                "Fit an intercept outside the constrained coefficients.",
                False,
            ),
        ),
        spaces=_ALPHA_SPACES,
        method="cv_path",
        backend="internal + scipy.optimize.nnls",
        description="Ridge regression with non-negative coefficients.",
    ),
    "shrink_to_target_ridge": _spec(
        "shrink_to_target_ridge",
        "linear",
        shrink_to_target_ridge,
        default_params={
            "alpha": 1.0,
            "prior_target": None,
            "simplex": False,
            "nonneg": False,
            "fit_intercept": True,
            "max_iter": 1000,
            "tol": 1e-9,
        },
        parameters=(
            _p("alpha", 1.0, "float", "Strength of the coefficient-target shrinkage."),
            _p(
                "prior_target",
                None,
                "float | sequence | mapping | None",
                "Coefficient target; None means zero, or uniform when simplex=True.",
                False,
            ),
            _p(
                "simplex", False, "bool", "Constrain coefficients to sum to one.", False
            ),
            _p(
                "nonneg",
                False,
                "bool",
                "Constrain coefficients to be non-negative.",
                False,
            ),
            _p(
                "fit_intercept",
                True,
                "bool",
                "Fit an intercept unless simplex=True.",
                False,
            ),
            _p("max_iter", 1000, "int", "SLSQP solver iteration cap.", False),
            _p("tol", 1e-9, "float", "SLSQP solver tolerance.", False),
        ),
        spaces=_ALPHA_SPACES,
        method="cv_path",
        backend="internal + scipy.optimize.minimize(SLSQP)",
        description="Ridge regression shrinking coefficients toward a target vector.",
    ),
    "fused_difference_ridge": _spec(
        "fused_difference_ridge",
        "linear",
        fused_difference_ridge,
        default_params={
            "alpha": 1.0,
            "difference_order": 1,
            "mean_equality": False,
            "nonneg": False,
            "fit_intercept": True,
            "max_iter": 1000,
            "tol": 1e-9,
        },
        parameters=(
            _p(
                "alpha",
                1.0,
                "float",
                "Strength of the adjacent-coefficient smoothness penalty.",
            ),
            _p(
                "difference_order",
                1,
                "int",
                "Finite-difference order applied to coefficients.",
                False,
            ),
            _p(
                "mean_equality",
                False,
                "bool",
                "Constrain fitted and observed sums to match.",
                False,
            ),
            _p(
                "nonneg",
                False,
                "bool",
                "Constrain coefficients to be non-negative.",
                False,
            ),
            _p(
                "fit_intercept",
                True,
                "bool",
                "Fit an intercept unless mean_equality=True.",
                False,
            ),
            _p("max_iter", 1000, "int", "SLSQP solver iteration cap.", False),
            _p("tol", 1e-9, "float", "SLSQP solver tolerance.", False),
        ),
        spaces=_ALPHA_SPACES,
        method="cv_path",
        backend="internal + scipy.optimize.minimize(SLSQP)",
        description="Ridge regression with a fused-difference coefficient prior.",
    ),
    "supervised_aggregation": _spec(
        "supervised_aggregation",
        "assemblage",
        supervised_aggregation,
        default_params={
            "space": "component",
            "penalty": "ridge",
            "alpha": 1.0,
            "reference_weights": None,
            "nonneg": True,
            "simplex": False,
            "mean_match": False,
            "difference_order": 1,
            "fit_intercept": False,
            "penalty_scale": "feature_std",
            "max_iter": 1000,
            "tol": 1e-9,
        },
        parameters=(
            _p("space", "component", "component | rank", "Aggregation space.", False),
            _p("penalty", "ridge", "ridge | target_shrinkage | fused_difference", "Coefficient penalty family.", False),
            _p("alpha", 1.0, "float", "Penalty strength."),
            _p("reference_weights", None, "mapping | sequence | None", "Reference basket weights for target shrinkage.", False),
            _p("nonneg", True, "bool", "Constrain weights to be non-negative.", False),
            _p("simplex", False, "bool", "Constrain weights to sum to one.", False),
            _p("mean_match", False, "bool", "Constrain fitted aggregate mean to match target mean.", False),
            _p("difference_order", 1, "int", "Finite-difference order for fused rank weights.", False),
            _p("fit_intercept", False, "bool", "Fit an intercept outside the aggregation weights.", False),
            _p("penalty_scale", "feature_std", "none | feature_std", "Scale penalties by component standard deviations.", False),
            _p("max_iter", 1000, "int", "SLSQP solver iteration cap.", False),
            _p("tol", 1e-9, "float", "SLSQP solver tolerance.", False),
        ),
        spaces=_ALPHA_SPACES,
        method="cv_path",
        backend="internal + scipy.optimize.minimize(SLSQP)",
        description="Generic constrained supervised aggregation derived from Albacore/assemblage primitives.",
    ),
    "component_aggregation": _spec(
        "component_aggregation",
        "assemblage",
        component_aggregation,
        default_params={
            "alpha": 1.0,
            "reference_weights": None,
            "penalty": None,
            "simplex": True,
            "nonneg": True,
            "penalty_scale": "feature_std",
            "max_iter": 1000,
            "tol": 1e-9,
        },
        parameters=(
            _p("alpha", 1.0, "float", "Penalty strength."),
            _p("reference_weights", None, "mapping | sequence | None", "Optional reference component weights.", False),
            _p("penalty", None, "ridge | target_shrinkage | None", "Penalty; None selects target_shrinkage when weights are supplied.", False),
            _p("simplex", True, "bool", "Constrain weights to sum to one.", False),
            _p("nonneg", True, "bool", "Constrain weights to be non-negative.", False),
            _p("penalty_scale", "feature_std", "none | feature_std", "Scale penalties by component standard deviations.", False),
            _p("max_iter", 1000, "int", "SLSQP solver iteration cap.", False),
            _p("tol", 1e-9, "float", "SLSQP solver tolerance.", False),
        ),
        spaces=_ALPHA_SPACES,
        method="cv_path",
        backend="internal + scipy.optimize.minimize(SLSQP)",
        description="Component-space supervised aggregation; generic Albacorecomps primitive.",
    ),
    "rank_aggregation": _spec(
        "rank_aggregation",
        "assemblage",
        rank_aggregation,
        default_params={
            "alpha": 1.0,
            "penalty": "fused_difference",
            "mean_match": True,
            "nonneg": True,
            "difference_order": 1,
            "penalty_scale": "feature_std",
            "max_iter": 1000,
            "tol": 1e-9,
        },
        parameters=(
            _p("alpha", 1.0, "float", "Penalty strength."),
            _p("penalty", "fused_difference", "ridge | fused_difference", "Rank-weight penalty family.", False),
            _p("mean_match", True, "bool", "Constrain fitted aggregate mean to match target mean.", False),
            _p("nonneg", True, "bool", "Constrain rank weights to be non-negative.", False),
            _p("difference_order", 1, "int", "Finite-difference order for rank weights.", False),
            _p("penalty_scale", "feature_std", "none | feature_std", "Scale penalties by rank standard deviations.", False),
            _p("max_iter", 1000, "int", "SLSQP solver iteration cap.", False),
            _p("tol", 1e-9, "float", "SLSQP solver tolerance.", False),
        ),
        spaces=_ALPHA_SPACES,
        method="cv_path",
        backend="internal + scipy.optimize.minimize(SLSQP)",
        description="Rank-space supervised aggregation; generic Albacoreranks primitive.",
    ),
    "assemblage_regression": _spec(
        "assemblage_regression",
        "assemblage",
        assemblage_regression,
        default_params={
            "space": "component",
            "alpha": 1.0,
            "reference_weights": None,
            "penalty": None,
            "max_iter": 1000,
            "tol": 1e-9,
        },
        parameters=(
            _p("space", "component", "component | rank", "Use component-space or rank-space aggregation.", False),
            _p("alpha", 1.0, "float", "Penalty strength."),
            _p("reference_weights", None, "mapping | sequence | None", "Optional reference weights for component space.", False),
            _p("penalty", None, "str | None", "Optional penalty override.", False),
            _p("max_iter", 1000, "int", "SLSQP solver iteration cap.", False),
            _p("tol", 1e-9, "float", "SLSQP solver tolerance.", False),
        ),
        spaces=_ALPHA_SPACES,
        method="cv_path",
        backend="internal + scipy.optimize.minimize(SLSQP)",
        description="Generic assemblage regression wrapper with component and rank variants.",
    ),
    "albacore_components": _spec(
        "albacore_components",
        "assemblage",
        albacore_components,
        default_params={
            "reference_weights": None,
            "alpha": 1.0,
            "max_iter": 1000,
            "tol": 1e-9,
        },
        parameters=(
            _p("reference_weights", None, "mapping | sequence | None", "Official or reference basket weights.", False),
            _p("alpha", 1.0, "float", "Target-shrinkage penalty strength."),
            _p("max_iter", 1000, "int", "SLSQP solver iteration cap.", False),
            _p("tol", 1e-9, "float", "SLSQP solver tolerance.", False),
        ),
        spaces=_ALPHA_SPACES,
        method="cv_path",
        backend="internal + scipy.optimize.minimize(SLSQP)",
        description="Inflation-specific component-space Albacore wrapper.",
    ),
    "albacore_ranks": _spec(
        "albacore_ranks",
        "assemblage",
        albacore_ranks,
        default_params={
            "alpha": 1.0,
            "difference_order": 1,
            "max_iter": 1000,
            "tol": 1e-9,
        },
        parameters=(
            _p("alpha", 1.0, "float", "Fused-difference penalty strength."),
            _p("difference_order", 1, "int", "Finite-difference order for rank weights.", False),
            _p("max_iter", 1000, "int", "SLSQP solver iteration cap.", False),
            _p("tol", 1e-9, "float", "SLSQP solver tolerance.", False),
        ),
        spaces=_ALPHA_SPACES,
        method="cv_path",
        backend="internal + scipy.optimize.minimize(SLSQP)",
        description="Inflation-specific rank-space Albacore wrapper.",
    ),
    "csr": _spec(
        "csr",
        "model_averaging",
        csr,
        default_params={"k": 4, "max_subsets": 5000, "random_state": 1071},
        parameters=(
            _p("k", 4, "int", "Subset size for each OLS member."),
            _p(
                "max_subsets",
                5000,
                "int",
                "Maximum distinct subsets to average before seeded subset sampling.",
                False,
            ),
            _p("random_state", 1071, "int | None", "Seed for subset sampling.", False),
        ),
        method="none",
        backend="internal numpy.linalg.lstsq",
        description="Complete Subset Regression; averages OLS forecasts over k-predictor subsets.",
    ),
    "jma": _spec(
        "jma",
        "model_averaging",
        jma,
        default_params={"candidates": "nested", "max_iter": 1000, "tol": 1e-9},
        parameters=(
            _p(
                "candidates",
                "nested",
                "nested",
                "Candidate model family; currently nested ordered OLS models.",
                False,
            ),
            _p("max_iter", 1000, "int", "SLSQP solver iteration cap.", False),
            _p("tol", 1e-9, "float", "SLSQP solver tolerance.", False),
        ),
        method="none",
        backend="internal numpy.linalg + scipy.optimize.minimize(SLSQP)",
        description="Jackknife Model Averaging with simplex weights chosen by OLS leave-one-out CV.",
    ),
    "random_walk_ridge": _spec(
        "random_walk_ridge",
        "linear",
        random_walk_ridge,
        default_params={"alpha": 1.0, "initial_alpha": 1.0, "fit_intercept": True},
        parameters=(
            _p(
                "alpha",
                1.0,
                "float",
                "Penalty on changes in adjacent coefficient vectors.",
            ),
            _p(
                "initial_alpha",
                1.0,
                "float",
                "Penalty on the first coefficient vector.",
                False,
            ),
            _p(
                "fit_intercept",
                True,
                "bool",
                "Fit an intercept outside the time-varying coefficients.",
                False,
            ),
        ),
        spaces=_ALPHA_SPACES,
        method="cv_path",
        backend="internal numpy.linalg.lstsq",
        description="Time-varying random-walk ridge fit, predicting with the final coefficient vector.",
    ),
    "tvp_ridge": _spec(
        "tvp_ridge",
        "linear",
        tvp_ridge,
        default_params={
            "lambda_candidates": None,
            "lambda2": 0.1,
            "kfold": 5,
            "cv_2srr": True,
            "sig_u_param": 0.75,
            "sig_eps_param": 0.75,
            "ols_prior": False,
            "random_state": 1071,
            "use_garch": True,
        },
        parameters=(
            _p(
                "lambda_candidates",
                None,
                "sequence[float] | None",
                "Candidate lambda values for the time-variation ridge penalty.",
            ),
            _p(
                "lambda2",
                0.1,
                "float",
                "Penalty on starting coefficient values beta_0.",
                False,
            ),
            _p("kfold", 5, "int", "Random k-fold count for lambda CV.", False),
            _p(
                "cv_2srr",
                True,
                "bool",
                "Run the second lambda CV after 2SRR variance reweighting.",
                False,
            ),
            _p(
                "sig_u_param",
                0.75,
                "float",
                "Shrinkage exponent for coefficient-innovation variance weights.",
                False,
            ),
            _p(
                "sig_eps_param",
                0.75,
                "float",
                "Shrinkage exponent for residual-volatility weights.",
                False,
            ),
            _p(
                "ols_prior",
                False,
                "bool",
                "Shrink starting coefficients toward OLS instead of zero.",
                False,
            ),
            _p("random_state", 1071, "int", "Random fold seed.", False),
            _p(
                "use_garch",
                True,
                "bool",
                "Use optional arch GARCH(1,1) residual volatility if installed.",
                False,
            ),
        ),
        spaces={
            "small": {
                "lambda_candidates": (
                    (0.01, 0.1, 1.0, 10.0),
                    (0.1, 1.0, 10.0, 100.0),
                )
            },
            "standard": {
                "lambda_candidates": (
                    tuple(float(v) for v in np.exp(np.linspace(-6.0, 20.0, num=15))),
                )
            },
            "wide": {
                "lambda_candidates": (
                    tuple(float(v) for v in np.exp(np.linspace(-8.0, 24.0, num=21))),
                )
            },
        },
        method="cv_path",
        backend="internal TVPRidge R/MV2SRR_v210407.R port + optional arch GARCH",
        description="Goulet Coulombe TVP ridge / 2SRR estimator.",
    ),
    "lasso": _spec(
        "lasso",
        "linear",
        lasso,
        default_params={"alpha": 1.0, "max_iter": 20000, "standardize": False},
        parameters=(
            _p("alpha", 1.0, "float", "L1 penalty strength."),
            _p("max_iter", 20000, "int", "Optimization iteration cap.", False),
            _p(
                "standardize",
                False,
                "bool",
                "Standardize predictors inside the fitted estimator.",
                False,
            ),
        ),
        spaces=_ALPHA_SPACES,
        method="cv_path",
        backend="sklearn.linear_model.Lasso",
        description="Lasso regression.",
    ),
    "elastic_net": _spec(
        "elastic_net",
        "linear",
        elastic_net,
        default_params={
            "alpha": 1.0,
            "l1_ratio": 0.5,
            "max_iter": 20000,
            "standardize": False,
        },
        parameters=(
            _p("alpha", 1.0, "float", "Overall penalty strength."),
            _p("l1_ratio", 0.5, "float", "L1 share of the elastic-net penalty."),
            _p("max_iter", 20000, "int", "Optimization iteration cap.", False),
            _p(
                "standardize",
                False,
                "bool",
                "Standardize predictors inside the fitted estimator.",
                False,
            ),
        ),
        spaces={
            "small": {"alpha": (0.01, 0.1, 1.0), "l1_ratio": (0.25, 0.5, 0.75)},
            "standard": {
                "alpha": (0.001, 0.01, 0.1, 1.0, 10.0),
                "l1_ratio": (0.1, 0.25, 0.5, 0.75, 0.9),
            },
            "wide": {
                "alpha": (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0),
                "l1_ratio": (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95),
            },
        },
        backend="sklearn.linear_model.ElasticNet",
        description="Elastic net regression.",
    ),
    "adaptive_lasso": _spec(
        "adaptive_lasso",
        "linear",
        adaptive_lasso,
        default_params={
            "alpha": 1.0,
            "gamma": 1.0,
            "initial": "ridge",
            "initial_alpha": 1.0,
            "eps": 1e-4,
            "normalize_weights": True,
            "max_iter": 20000,
            "tol": 1e-4,
            "random_state": None,
        },
        parameters=(
            _p("alpha", 1.0, "float", "Final adaptive lasso penalty strength."),
            _p(
                "gamma",
                1.0,
                "float",
                "Exponent applied to initial coefficient weights.",
            ),
            _p("initial", "ridge", "str", "Initial model: 'ridge' or 'ols'.", False),
            _p("initial_alpha", 1.0, "float", "Initial ridge penalty.", False),
            _p(
                "eps",
                1e-4,
                "float",
                "Small denominator floor for adaptive weights.",
                False,
            ),
            _p(
                "normalize_weights",
                True,
                "bool",
                "Rescale adaptive penalty weights to mean one, matching glmnet penalty.factor scaling.",
                False,
            ),
            _p("max_iter", 20000, "int", "Final solver iteration cap.", False),
            _p("tol", 1e-4, "float", "Final solver convergence tolerance.", False),
            _p("random_state", None, "int | None", "Final solver random seed.", False),
        ),
        spaces={
            "small": {"alpha": (0.01, 0.1, 1.0), "gamma": (1.0,)},
            "standard": {
                "alpha": (0.001, 0.01, 0.1, 1.0, 10.0),
                "gamma": (0.5, 1.0, 2.0),
            },
            "wide": {
                "alpha": (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0),
                "gamma": (0.5, 1.0, 1.5, 2.0),
            },
        },
        backend="internal adaptive weights + sklearn.linear_model.Lasso",
        description="Adaptive lasso using initial coefficient-based penalty weights.",
    ),
    "adaptive_elastic_net": _spec(
        "adaptive_elastic_net",
        "linear",
        adaptive_elastic_net,
        default_params={
            "alpha": 1.0,
            "l1_ratio": 0.5,
            "gamma": 1.0,
            "initial": "ridge",
            "initial_alpha": 1.0,
            "eps": 1e-4,
            "normalize_weights": True,
            "max_iter": 20000,
            "tol": 1e-4,
            "random_state": None,
        },
        parameters=(
            _p("alpha", 1.0, "float", "Final adaptive elastic-net penalty strength."),
            _p("l1_ratio", 0.5, "float", "L1 share of the final elastic-net penalty."),
            _p(
                "gamma",
                1.0,
                "float",
                "Exponent applied to initial coefficient weights.",
            ),
            _p("initial", "ridge", "str", "Initial model: 'ridge' or 'ols'.", False),
            _p("initial_alpha", 1.0, "float", "Initial ridge penalty.", False),
            _p(
                "eps",
                1e-4,
                "float",
                "Small denominator floor for adaptive weights.",
                False,
            ),
            _p(
                "normalize_weights",
                True,
                "bool",
                "Rescale adaptive penalty weights to mean one, matching glmnet penalty.factor scaling.",
                False,
            ),
            _p("max_iter", 20000, "int", "Final solver iteration cap.", False),
            _p("tol", 1e-4, "float", "Final solver convergence tolerance.", False),
            _p("random_state", None, "int | None", "Final solver random seed.", False),
        ),
        spaces={
            "small": {
                "alpha": (0.01, 0.1, 1.0),
                "l1_ratio": (0.25, 0.5, 0.75),
                "gamma": (1.0,),
            },
            "standard": {
                "alpha": (0.001, 0.01, 0.1, 1.0, 10.0),
                "l1_ratio": (0.1, 0.25, 0.5, 0.75, 0.9),
                "gamma": (0.5, 1.0, 2.0),
            },
            "wide": {
                "alpha": (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0),
                "l1_ratio": (0.05, 0.1, 0.25, 0.5, 0.75, 0.9),
                "gamma": (0.5, 1.0, 1.5, 2.0),
            },
        },
        backend="internal adaptive weights + sklearn.linear_model.ElasticNet",
        description="Adaptive elastic net using initial coefficient-based column weights.",
    ),
    "group_lasso": _spec(
        "group_lasso",
        "linear",
        group_lasso,
        default_params={
            "groups": None,
            "alpha": 1.0,
            "group_weights": None,
            "max_iter": 5000,
            "tol": 1e-5,
            "scale": True,
        },
        parameters=(
            _p(
                "groups",
                None,
                "sequence[str | int] | None",
                "One group label per predictor.",
                False,
            ),
            _p("alpha", 1.0, "float", "Group penalty strength."),
            _p(
                "group_weights",
                None,
                "dict[str, float] | None",
                "Optional group penalty weights.",
                False,
            ),
            _p("max_iter", 5000, "int", "Proximal-gradient iteration cap.", False),
            _p("tol", 1e-5, "float", "Proximal-gradient convergence tolerance.", False),
            _p(
                "scale",
                True,
                "bool",
                "Whether to standardize predictors inside the model.",
                False,
            ),
        ),
        spaces={
            "small": {"alpha": (0.01, 0.1, 1.0)},
            "standard": {"alpha": (0.001, 0.01, 0.1, 1.0, 10.0)},
            "wide": {"alpha": (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0)},
        },
        backend="internal proximal-gradient solver",
        description="Package-native group lasso with group-level sparsity.",
    ),
    "sparse_group_lasso": _spec(
        "sparse_group_lasso",
        "linear",
        sparse_group_lasso,
        default_params={
            "groups": None,
            "alpha": 1.0,
            "l1_ratio": 0.5,
            "group_weights": None,
            "max_iter": 5000,
            "tol": 1e-5,
            "scale": True,
        },
        parameters=(
            _p(
                "groups",
                None,
                "sequence[str | int] | None",
                "One group label per predictor.",
                False,
            ),
            _p("alpha", 1.0, "float", "Total sparse-group penalty strength."),
            _p(
                "l1_ratio",
                0.5,
                "float",
                "Feature-level L1 share; remaining share is group penalty.",
            ),
            _p(
                "group_weights",
                None,
                "dict[str, float] | None",
                "Optional group penalty weights.",
                False,
            ),
            _p("max_iter", 5000, "int", "Proximal-gradient iteration cap.", False),
            _p("tol", 1e-5, "float", "Proximal-gradient convergence tolerance.", False),
            _p(
                "scale",
                True,
                "bool",
                "Whether to standardize predictors inside the model.",
                False,
            ),
        ),
        spaces={
            "small": {"alpha": (0.01, 0.1, 1.0), "l1_ratio": (0.25, 0.5, 0.75)},
            "standard": {
                "alpha": (0.001, 0.01, 0.1, 1.0, 10.0),
                "l1_ratio": (0.1, 0.25, 0.5, 0.75, 0.9),
            },
            "wide": {
                "alpha": (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0),
                "l1_ratio": (0.05, 0.1, 0.25, 0.5, 0.75, 0.9),
            },
        },
        backend="internal proximal-gradient solver",
        description="Package-native sparse group lasso with group and feature-level sparsity.",
    ),
    "bayesian_ridge": _spec(
        "bayesian_ridge",
        "linear",
        bayesian_ridge,
        backend="sklearn.linear_model.BayesianRidge",
        description="Empirical-Bayes Bayesian ridge.",
    ),
    "huber": _spec(
        "huber",
        "linear",
        huber,
        default_params={"epsilon": 1.35, "max_iter": 1000},
        parameters=(
            _p("epsilon", 1.35, "float", "Huber loss transition threshold."),
            _p("max_iter", 1000, "int", "Optimization iteration cap.", False),
        ),
        spaces={
            "small": {"epsilon": (1.1, 1.35, 1.75)},
            "standard": {"epsilon": (1.1, 1.35, 1.5, 1.75, 2.0)},
            "wide": {"epsilon": (1.01, 1.1, 1.35, 1.5, 1.75, 2.0, 2.5)},
        },
        backend="sklearn.linear_model.HuberRegressor",
        description="Robust Huber regression.",
    ),
    "kernel_ridge": _spec(
        "kernel_ridge",
        "nonparametric",
        kernel_ridge,
        default_params={
            "alpha": 1.0,
            "kernel": "linear",
            "gamma": None,
            "degree": 3,
            "coef0": 1.0,
        },
        parameters=(
            _p("alpha", 1.0, "float", "Ridge penalty strength."),
            _p(
                "kernel",
                "linear",
                "str",
                "Kernel name: linear, rbf, polynomial, sigmoid, etc.",
            ),
            _p("gamma", None, "float | None", "Kernel coefficient.", False),
            _p("degree", 3, "int", "Polynomial kernel degree.", False),
            _p(
                "coef0",
                1.0,
                "float",
                "Independent term for polynomial/sigmoid kernels.",
                False,
            ),
        ),
        spaces={
            "small": {"alpha": (0.1, 1.0, 10.0), "kernel": ("linear", "rbf")},
            "standard": {
                "alpha": (0.01, 0.1, 1.0, 10.0),
                "kernel": ("linear", "rbf", "poly"),
                "gamma": (None, 0.01, 0.1),
            },
            "wide": {
                "alpha": (0.001, 0.01, 0.1, 1.0, 10.0, 100.0),
                "kernel": ("linear", "rbf", "poly", "sigmoid"),
                "gamma": (None, 0.001, 0.01, 0.1, 1.0),
                "degree": (2, 3, 4),
            },
        },
        method="random",
        backend="sklearn.kernel_ridge.KernelRidge",
        requires_scaling=True,
        recommended_preprocessing=("standardize predictors before nonlinear kernels",),
        description="Kernel ridge regression.",
    ),
    "knn": _spec(
        "knn",
        "nonparametric",
        knn,
        default_params={
            "n_neighbors": 5,
            "weights": "uniform",
            "metric": "minkowski",
            "p": 2,
        },
        parameters=(
            _p("n_neighbors", 5, "int", "Number of nearest neighbors."),
            _p("weights", "uniform", "str", "Neighbor weighting: uniform or distance."),
            _p("metric", "minkowski", "str", "Distance metric.", False),
            _p("p", 2, "int", "Minkowski distance order.", False),
        ),
        spaces={
            "small": {"n_neighbors": (3, 5, 10), "weights": ("uniform", "distance")},
            "standard": {
                "n_neighbors": (3, 5, 10, 20),
                "weights": ("uniform", "distance"),
                "p": (1, 2),
            },
            "wide": {
                "n_neighbors": (1, 3, 5, 10, 20, 40),
                "weights": ("uniform", "distance"),
                "p": (1, 2),
            },
        },
        method="random",
        backend="sklearn.neighbors.KNeighborsRegressor",
        requires_scaling=True,
        recommended_preprocessing=(
            "standardize predictors before distance-based fitting",
        ),
        description="K-nearest-neighbor regression.",
    ),
    "glmboost": _spec(
        "glmboost",
        "linear",
        glmboost,
        default_params={
            "n_iter": 100,
            "learning_rate": 0.1,
            "center": True,
            "candidate_sampling": "all",
            "candidate_count": None,
            "candidate_fraction": None,
            "candidate_cap": None,
            "candidate_min": 1,
            "candidate_rounding": "floor",
            "random_state": None,
        },
        parameters=(
            _p("n_iter", 100, "int", "Number of boosting iterations."),
            _p(
                "learning_rate",
                0.1,
                "float",
                "Shrinkage applied to each componentwise update.",
            ),
            _p(
                "center",
                True,
                "bool",
                "Center predictors before componentwise updates, matching mboost's default.",
                False,
            ),
            _p(
                "candidate_sampling",
                "all",
                "str",
                "Candidate-subset policy per boosting step: all or random.",
                False,
            ),
            _p(
                "candidate_count",
                None,
                "int | None",
                "Fixed candidate count when candidate_sampling='random'.",
                False,
            ),
            _p(
                "candidate_fraction",
                None,
                "float | None",
                "Candidate fraction when candidate_sampling='random'.",
                False,
            ),
            _p(
                "candidate_cap",
                None,
                "int | None",
                "Maximum sampled candidate count after resolving count/fraction.",
                False,
            ),
            _p(
                "candidate_min",
                1,
                "int",
                "Minimum sampled candidate count.",
                False,
            ),
            _p(
                "candidate_rounding",
                "floor",
                "str",
                "Rounding rule for candidate_fraction: floor, ceil, or round.",
                False,
            ),
            _p(
                "random_state",
                None,
                "int | None",
                "Seed for per-step candidate feature sampling.",
                False,
            ),
        ),
        spaces={
            "small": {"n_iter": (50, 100), "learning_rate": (0.05, 0.1)},
            "standard": {
                "n_iter": (50, 100, 200, 500),
                "learning_rate": (0.01, 0.05, 0.1),
            },
            "wide": {
                "n_iter": (50, 100, 200, 500, 1000),
                "learning_rate": (0.005, 0.01, 0.05, 0.1, 0.2),
            },
        },
        backend="internal componentwise L2 boosting",
        description="Componentwise linear boosting.",
    ),
    "svr": _spec(
        "svr",
        "support_vector",
        svr,
        default_params={
            "kernel": "rbf",
            "C": 1.0,
            "epsilon": 0.1,
            "gamma": "scale",
            "degree": 3,
            "coef0": 0.0,
            "shrinking": True,
            "tol": 1e-3,
            "cache_size": 200.0,
            "max_iter": -1,
        },
        parameters=(
            _p(
                "kernel",
                "rbf",
                "str",
                "SVR kernel: linear, poly, rbf, or sigmoid.",
                False,
            ),
            _p("C", 1.0, "float", "Regularization strength inverse."),
            _p("epsilon", 0.1, "float", "Epsilon-insensitive tube width."),
            _p(
                "gamma",
                "scale",
                "str | float",
                "Kernel coefficient for rbf/poly/sigmoid.",
            ),
            _p("degree", 3, "int", "Polynomial kernel degree.", False),
            _p(
                "coef0",
                0.0,
                "float",
                "Independent term for poly/sigmoid kernels.",
                False,
            ),
            _p(
                "shrinking",
                True,
                "bool",
                "Whether to use the shrinking heuristic.",
                False,
            ),
            _p("tol", 1e-3, "float", "Optimization tolerance.", False),
            _p("cache_size", 200.0, "float", "Kernel cache size in MB.", False),
            _p("max_iter", -1, "int", "Solver iteration cap; -1 means no cap.", False),
        ),
        spaces=_SVM_SPACES,
        method="random",
        backend="sklearn.svm.SVR",
        requires_scaling=True,
        recommended_preprocessing=("standardize predictors before fitting",),
        description="Kernel support-vector regression.",
    ),
    "linear_svr": _spec(
        "linear_svr",
        "support_vector",
        linear_svr,
        default_params={
            "C": 1.0,
            "epsilon": 0.0,
            "loss": "epsilon_insensitive",
            "tol": 1e-4,
            "max_iter": 10000,
            "random_state": 0,
        },
        parameters=(
            _p("C", 1.0, "float", "Regularization strength inverse."),
            _p("epsilon", 0.0, "float", "Epsilon-insensitive tube width."),
            _p("loss", "epsilon_insensitive", "str", "LinearSVR loss function.", False),
            _p("tol", 1e-4, "float", "Optimization tolerance.", False),
            _p("max_iter", 10000, "int", "Solver iteration cap.", False),
            _p("random_state", 0, "int | None", "Random seed.", False),
        ),
        spaces={
            "small": {"C": (0.1, 1.0), "epsilon": (0.0, 0.1)},
            "standard": {"C": (0.01, 0.1, 1.0, 10.0), "epsilon": (0.0, 0.01, 0.1)},
            "wide": {
                "C": (0.001, 0.01, 0.1, 1.0, 10.0, 100.0),
                "epsilon": (0.0, 0.001, 0.01, 0.1, 0.2),
            },
        },
        method="random",
        backend="sklearn.svm.LinearSVR",
        requires_scaling=True,
        recommended_preprocessing=("standardize predictors before fitting",),
        description="Linear support-vector regression.",
    ),
    "nu_svr": _spec(
        "nu_svr",
        "support_vector",
        nu_svr,
        default_params={
            "kernel": "rbf",
            "C": 1.0,
            "nu": 0.5,
            "gamma": "scale",
            "degree": 3,
            "coef0": 0.0,
            "shrinking": True,
            "tol": 1e-3,
            "cache_size": 200.0,
            "max_iter": -1,
        },
        parameters=(
            _p(
                "kernel",
                "rbf",
                "str",
                "NuSVR kernel: linear, poly, rbf, or sigmoid.",
                False,
            ),
            _p("C", 1.0, "float", "Regularization strength inverse."),
            _p(
                "nu",
                0.5,
                "float",
                "Upper/lower training-error and support-vector fraction control.",
            ),
            _p(
                "gamma",
                "scale",
                "str | float",
                "Kernel coefficient for rbf/poly/sigmoid.",
            ),
            _p("degree", 3, "int", "Polynomial kernel degree.", False),
            _p(
                "coef0",
                0.0,
                "float",
                "Independent term for poly/sigmoid kernels.",
                False,
            ),
            _p(
                "shrinking",
                True,
                "bool",
                "Whether to use the shrinking heuristic.",
                False,
            ),
            _p("tol", 1e-3, "float", "Optimization tolerance.", False),
            _p("cache_size", 200.0, "float", "Kernel cache size in MB.", False),
            _p("max_iter", -1, "int", "Solver iteration cap; -1 means no cap.", False),
        ),
        spaces={
            "small": {"C": (0.1, 1.0), "nu": (0.25, 0.5), "gamma": ("scale",)},
            "standard": {
                "C": (0.1, 1.0, 10.0),
                "nu": (0.25, 0.5, 0.75),
                "gamma": ("scale", "auto"),
            },
            "wide": {
                "C": (0.01, 0.1, 1.0, 10.0, 100.0),
                "nu": (0.1, 0.25, 0.5, 0.75, 0.9),
                "gamma": ("scale", "auto"),
            },
        },
        method="random",
        backend="sklearn.svm.NuSVR",
        requires_scaling=True,
        recommended_preprocessing=("standardize predictors before fitting",),
        description="Nu support-vector regression.",
    ),
    "nn": _spec(
        "nn",
        "neural",
        nn,
        default_params={
            "hidden_layer_sizes": (100,),
            "activation": "relu",
            "dropout": 0.0,
            "learning_rate": 0.001,
            "max_epochs": 100,
            "batch_size": 32,
            "weight_decay": 0.0,
            "optimizer": "adam",
            "loss": "mse",
            "random_state": 0,
            "device": "auto",
        },
        parameters=(
            _p(
                "hidden_layer_sizes",
                (100,),
                "tuple[int, ...]",
                "Feed-forward hidden layer widths.",
            ),
            _p(
                "activation",
                "relu",
                "str",
                "Activation: identity, logistic, sigmoid, tanh, relu, or gelu.",
                False,
            ),
            _p("dropout", 0.0, "float", "Dropout rate between hidden layers."),
            _p("learning_rate", 0.001, "float", "Optimizer learning rate."),
            _p("max_epochs", 100, "int", "Training epoch cap.", False),
            _p("batch_size", 32, "int", "Mini-batch size.", False),
            _p("weight_decay", 0.0, "float", "L2 weight decay."),
            _p(
                "optimizer",
                "adam",
                "str",
                "Torch optimizer: adam, sgd, or rmsprop.",
                False,
            ),
            _p("loss", "mse", "str", "Torch loss: mse or huber.", False),
            _p("random_state", 0, "int", "Random seed.", False),
            _p("device", "auto", "str", "Torch device: auto, cpu, or cuda.", False),
        ),
        spaces=_NN_SPACES,
        method="random",
        backend="torch.nn.Sequential",
        requires_extra="deep",
        recommended_preprocessing=(
            "handled internally: X and y are standardized inside each fit",
        ),
        description="Torch-backed feed-forward multilayer perceptron regressor.",
    ),
    "lstm": _spec(
        "lstm",
        "neural",
        lstm,
        default_params={
            "sequence_length": 4,
            "hidden_size": 32,
            "num_layers": 1,
            "dropout": 0.0,
            "learning_rate": 0.001,
            "max_epochs": 100,
            "batch_size": 32,
            "random_state": 0,
            "device": "auto",
        },
        parameters=(
            _p("sequence_length", 4, "int", "Trailing rows per recurrent sequence."),
            _p("hidden_size", 32, "int", "Recurrent hidden-state width."),
            _p("num_layers", 1, "int", "Number of recurrent layers.", False),
            _p("dropout", 0.0, "float", "Dropout between recurrent layers.", False),
            _p("learning_rate", 0.001, "float", "Adam learning rate."),
            _p("max_epochs", 100, "int", "Training epoch cap.", False),
            _p("batch_size", 32, "int", "Mini-batch size.", False),
            _p("random_state", 0, "int", "Random seed.", False),
            _p("device", "auto", "str", "Torch device: auto, cpu, or cuda.", False),
        ),
        spaces=_RNN_SPACES,
        method="random",
        backend="torch.nn.LSTM",
        requires_extra="deep",
        recommended_preprocessing=(
            "handled internally: X and y are standardized inside each fit",
        ),
        description="Torch-backed LSTM regressor.",
    ),
    "gru": _spec(
        "gru",
        "neural",
        gru,
        default_params={
            "sequence_length": 4,
            "hidden_size": 32,
            "num_layers": 1,
            "dropout": 0.0,
            "learning_rate": 0.001,
            "max_epochs": 100,
            "batch_size": 32,
            "random_state": 0,
            "device": "auto",
        },
        parameters=(
            _p("sequence_length", 4, "int", "Trailing rows per recurrent sequence."),
            _p("hidden_size", 32, "int", "Recurrent hidden-state width."),
            _p("num_layers", 1, "int", "Number of recurrent layers.", False),
            _p("dropout", 0.0, "float", "Dropout between recurrent layers.", False),
            _p("learning_rate", 0.001, "float", "Adam learning rate."),
            _p("max_epochs", 100, "int", "Training epoch cap.", False),
            _p("batch_size", 32, "int", "Mini-batch size.", False),
            _p("random_state", 0, "int", "Random seed.", False),
            _p("device", "auto", "str", "Torch device: auto, cpu, or cuda.", False),
        ),
        spaces=_RNN_SPACES,
        method="random",
        backend="torch.nn.GRU",
        requires_extra="deep",
        recommended_preprocessing=(
            "handled internally: X and y are standardized inside each fit",
        ),
        description="Torch-backed GRU regressor.",
    ),
    "transformer": _spec(
        "transformer",
        "neural",
        transformer,
        default_params={
            "sequence_length": 4,
            "hidden_size": 32,
            "num_layers": 1,
            "dropout": 0.0,
            "learning_rate": 0.001,
            "max_epochs": 100,
            "batch_size": 32,
            "random_state": 0,
            "device": "auto",
        },
        parameters=(
            _p("sequence_length", 4, "int", "Trailing rows per Transformer sequence."),
            _p("hidden_size", 32, "int", "Transformer feed-forward width."),
            _p("num_layers", 1, "int", "Number of encoder layers.", False),
            _p("dropout", 0.0, "float", "Transformer dropout rate.", False),
            _p("learning_rate", 0.001, "float", "Adam learning rate."),
            _p("max_epochs", 100, "int", "Training epoch cap.", False),
            _p("batch_size", 32, "int", "Mini-batch size.", False),
            _p("random_state", 0, "int", "Random seed.", False),
            _p("device", "auto", "str", "Torch device: auto, cpu, or cuda.", False),
        ),
        spaces=_RNN_SPACES,
        method="random",
        backend="torch.nn.TransformerEncoder",
        requires_extra="deep",
        recommended_preprocessing=(
            "handled internally: X and y are standardized inside each fit",
        ),
        description="Torch-backed Transformer encoder regressor.",
    ),
    "hemisphere_nn": _spec(
        "hemisphere_nn",
        "neural",
        hemisphere_nn,
        # Defaults trimmed 2026-07-05 (WP-B4, default-cost fix): n_estimators
        # 100->20, max_epochs 100->40, patience 15->8 (old values: see
        # CHANGELOG). Old values remain reachable by passing them explicitly,
        # e.g. `hemisphere_nn(X, y, n_estimators=100, max_epochs=100,
        # patience=15)`.
        default_params={
            "lc": 2,
            "lm": 2,
            "lv": 2,
            "neurons": 64,
            "dropout": 0.2,
            "learning_rate": 0.001,
            "max_epochs": 40,
            "n_estimators": 20,
            "subsample": 0.8,
            "nu": None,
            "variance_penalty": 1.0,
            "patience": 8,
            "validation_fraction": 0.2,
            "random_state": 0,
            "device": "auto",
            "quantile_levels": (0.05, 0.5, 0.95),
            "lr": None,
            "n_epochs": None,
            "B": None,
            "sub_rate": None,
            "lambda_emphasis": None,
            "val_frac": None,
        },
        parameters=(
            _p("lc", 2, "int", "Shared common-core depth.", False),
            _p("lm", 2, "int", "Mean-head depth after the common core.", False),
            _p("lv", 2, "int", "Variance-head depth after the common core.", False),
            _p("neurons", 64, "int", "Hidden width for all dense layers."),
            _p("dropout", 0.2, "float", "Dropout rate.", False),
            _p("learning_rate", 0.001, "float", "Adam learning rate."),
            _p("max_epochs", 40, "int", "Training epoch cap.", False),
            _p("n_estimators", 20, "int", "Number of blocked-subsample bags."),
            _p("subsample", 0.8, "float", "Blocked-subsample fraction.", False),
            _p("nu", None, "float | None", "Variance-emphasis target ratio.", False),
            _p(
                "variance_penalty",
                1.0,
                "float",
                "Soft penalty on the variance-emphasis target.",
                False,
            ),
            _p("patience", 8, "int", "Early-stopping patience.", False),
            _p(
                "validation_fraction",
                0.2,
                "float",
                "Chronological validation fraction.",
                False,
            ),
            _p("random_state", 0, "int", "Random seed.", False),
            _p("device", "auto", "str", "Torch device: auto, cpu, or cuda.", False),
            _p(
                "quantile_levels",
                (0.05, 0.5, 0.95),
                "tuple[float, ...]",
                "Default normal-approximation density quantiles.",
                False,
            ),
            _p("lr", None, "float | None", "Legacy alias for learning_rate.", False),
            _p("n_epochs", None, "int | None", "Legacy alias for max_epochs.", False),
            _p("B", None, "int | None", "Legacy alias for n_estimators.", False),
            _p("sub_rate", None, "float | None", "Legacy alias for subsample.", False),
            _p(
                "lambda_emphasis",
                None,
                "float | None",
                "Legacy alias for variance_penalty.",
                False,
            ),
            _p(
                "val_frac",
                None,
                "float | None",
                "Legacy alias for validation_fraction.",
                False,
            ),
        ),
        spaces=_HNN_SPACES,
        method="random",
        backend="torch dual-head dense network",
        requires_extra="deep",
        recommended_preprocessing=(
            "handled internally: X is standardized inside each fit",
        ),
        description="Bagged Hemisphere neural network with mean and variance heads.",
    ),
    "density_hnn": _spec(
        "density_hnn",
        "neural",
        density_hnn,
        # Defaults trimmed 2026-07-05 (WP-B4, default-cost fix): n_estimators
        # 100->20, prior_estimators 50->10, max_epochs 100->40, patience
        # 15->8 (old values: see CHANGELOG). `neurons` (400, the paper-faithful
        # Aionx DensityHNN width) is UNCHANGED -- `model_selection=None` always
        # overrides `neurons` via the "standard"/"wide" search preset anyway
        # (see `_DENSITY_HNN_SPACES`), so it was never the cost driver in the
        # policy-matrix scan; trimming it would only cost paper-fidelity for
        # no scan-cost benefit. Old values remain reachable by passing them
        # explicitly, e.g. `density_hnn(X, y, n_estimators=100,
        # prior_estimators=50, max_epochs=100, patience=15)`.
        default_params={
            "common_layers": 2,
            "mean_layers": 2,
            "volatility_layers": 2,
            "prior_layers": 3,
            "neurons": 400,
            "dropout": 0.2,
            "learning_rate": 0.001,
            "max_epochs": 40,
            "n_estimators": 20,
            "prior_estimators": 10,
            "subsample": 0.8,
            "block_size": 8,
            "volatility_emphasis": None,
            "rescale_volatility": True,
            "patience": 8,
            "random_state": 0,
            "device": "auto",
            "quantile_levels": (0.05, 0.5, 0.95),
            "volatility_clip": 0.05,
        },
        parameters=(
            _p("common_layers", 2, "int", "Shared common-core depth.", False),
            _p("mean_layers", 2, "int", "Conditional-mean hemisphere depth.", False),
            _p(
                "volatility_layers",
                2,
                "int",
                "Conditional-volatility hemisphere depth.",
                False,
            ),
            _p("prior_layers", 3, "int", "Plain prior-DNN depth.", False),
            _p("neurons", 400, "int", "Hidden width used by all dense blocks."),
            _p("dropout", 0.2, "float", "Dropout rate.", False),
            _p("learning_rate", 0.001, "float", "Adam learning rate."),
            _p("max_epochs", 40, "int", "Training epoch cap.", False),
            _p("n_estimators", 20, "int", "Density-HNN bootstrap ensemble size."),
            _p(
                "prior_estimators",
                10,
                "int",
                "Prior-DNN bootstrap ensemble size used to estimate volatility emphasis.",
            ),
            _p("subsample", 0.8, "float", "Blocked bootstrap sampling rate.", False),
            _p("block_size", 8, "int", "Time-series bootstrap block size.", False),
            _p(
                "volatility_emphasis",
                None,
                "float | None",
                "Override for Aionx volatility-emphasis parameter; None estimates it from prior-DNN OOB MSE.",
                False,
            ),
            _p(
                "rescale_volatility",
                True,
                "bool",
                "Apply Aionx blocked-OOB log residual-square volatility recalibration.",
                False,
            ),
            _p("patience", 8, "int", "Early-stopping patience.", False),
            _p("random_state", 0, "int", "Random seed.", False),
            _p("device", "auto", "str", "Torch device: auto, cpu, or cuda.", False),
            _p(
                "quantile_levels",
                (0.05, 0.5, 0.95),
                "tuple[float, ...]",
                "Default normal-approximation density quantiles.",
                False,
            ),
            _p(
                "volatility_clip",
                0.05,
                "float",
                "Minimum volatility in Gaussian negative log likelihood.",
                False,
            ),
        ),
        spaces=_DENSITY_HNN_SPACES,
        method="random",
        backend="torch-native Aionx DensityHNN port",
        requires_extra="deep",
        recommended_preprocessing=(
            "feature lags/trends are built before fitting; X and y are standardized inside each fit",
        ),
        description="Paper-faithful Density Hemisphere neural network with prior-DNN OOB volatility emphasis and OOB volatility rescaling.",
    ),
    "pls": _spec(
        "pls",
        "composite",
        pls,
        default_params={
            "n_components": 3,
            "scale": True,
            "max_iter": 500,
            "tol": 1e-6,
            "control_columns": None,
            "include_constant": True,
            "drop_control_columns": True,
            "quadratic_factors": False,
        },
        parameters=(
            _p("n_components", 3, "int", "Number of latent PLS components."),
            _p(
                "scale",
                True,
                "bool",
                "Whether to standardize predictors before PLS.",
                False,
            ),
            _p("max_iter", 500, "int", "NIPALS iteration cap.", False),
            _p("tol", 1e-6, "float", "NIPALS convergence tolerance.", False),
            _p(
                "control_columns",
                None,
                "Sequence[str] | None",
                "Optional X columns used as forecasting controls.",
                False,
            ),
            _p(
                "include_constant",
                True,
                "bool",
                "Whether to include a constant in the control block.",
                False,
            ),
            _p(
                "drop_control_columns",
                True,
                "bool",
                "Whether controls are excluded from the PLS block.",
                False,
            ),
            _p(
                "quadratic_factors",
                False,
                "bool",
                "Whether to add the Hounyo-Li PC2 squared-factor forecast head.",
                False,
            ),
        ),
        spaces={
            "small": {"n_components": (1, 2, 3)},
            "standard": {"n_components": (1, 2, 3, 5, 8)},
            "wide": {"n_components": (1, 2, 3, 5, 8, 10, 12, 20)},
        },
        backend="sklearn.cross_decomposition.PLSRegression",
        description="Partial least squares regression with optional Hounyo-Li-style control residualization.",
    ),
    "scaled_pca": _spec(
        "scaled_pca",
        "composite",
        scaled_pca,
        default_params={
            "n_components": 3,
            "scale": True,
            "control_columns": None,
            "include_constant": True,
            "drop_control_columns": True,
            "winsorize_slopes": None,
            "quadratic_factors": False,
        },
        parameters=(
            _p("n_components", 3, "int", "Number of Huang scaled-PCA factors."),
            _p(
                "scale",
                True,
                "bool",
                "Whether to standardize predictors inside the model.",
                False,
            ),
            _p(
                "control_columns",
                None,
                "Sequence[str] | None",
                "Optional X columns used as forecasting controls.",
                False,
            ),
            _p(
                "include_constant",
                True,
                "bool",
                "Whether to include a constant in the control block.",
                False,
            ),
            _p(
                "drop_control_columns",
                True,
                "bool",
                "Whether controls are excluded from the PCA block.",
                False,
            ),
            _p(
                "winsorize_slopes",
                None,
                "tuple[float, float] | None",
                "Optional percentile winsorization for scaling slopes.",
                False,
            ),
            _p(
                "quadratic_factors",
                False,
                "bool",
                "Whether to add the Hounyo-Li PC2 squared-factor forecast head.",
                False,
            ),
        ),
        spaces={
            "small": {"n_components": (1, 2, 3)},
            "standard": {"n_components": (1, 2, 3, 5, 8)},
            "wide": {"n_components": (1, 2, 3, 5, 8, 10, 12, 20)},
        },
        description="Huang et al. scaled PCA: marginal predictive-slope scaling followed by PCA.",
    ),
    "supervised_pca": _spec(
        "supervised_pca",
        "composite",
        supervised_pca,
        default_params={
            "n_components": 3,
            "n_selected": 50,
            "min_abs_corr": 0.0,
            "scale": True,
            "control_columns": None,
            "include_constant": True,
            "drop_control_columns": True,
            "preselect": "none",
            "t_threshold": 1.28,
            "elastic_net_alpha": 0.0002,
            "elastic_net_l1_ratio": 0.5,
            "quadratic_factors": False,
            "random_state": 0,
        },
        parameters=(
            _p("n_components", 3, "int", "Number of sequential supervised components."),
            _p(
                "n_selected", 50, "int | None", "Predictors selected at each SPCA step."
            ),
            _p(
                "min_abs_corr",
                0.0,
                "float",
                "Minimum absolute residual correlation retained before PCA.",
            ),
            _p(
                "scale",
                True,
                "bool",
                "Whether to standardize predictors and target inside the model.",
                False,
            ),
            _p(
                "control_columns",
                None,
                "Sequence[str] | None",
                "Optional X columns used as forecasting controls.",
                False,
            ),
            _p(
                "include_constant",
                True,
                "bool",
                "Whether to include a constant in the control block.",
                False,
            ),
            _p(
                "drop_control_columns",
                True,
                "bool",
                "Whether controls are excluded from the PCA block.",
                False,
            ),
            _p(
                "preselect",
                "none",
                "str",
                "Optional pre-selection: none, hard_tstat, or elastic_net.",
                False,
            ),
            _p(
                "t_threshold",
                1.28,
                "float",
                "Hard t-stat pre-selection threshold.",
                False,
            ),
            _p(
                "elastic_net_alpha",
                0.0002,
                "float",
                "Elastic-net pre-selection penalty.",
                False,
            ),
            _p(
                "elastic_net_l1_ratio",
                0.5,
                "float",
                "Elastic-net pre-selection L1 ratio.",
                False,
            ),
            _p(
                "quadratic_factors",
                False,
                "bool",
                "Whether to add the Hounyo-Li PC2 squared-factor forecast head.",
                False,
            ),
            _p(
                "random_state",
                0,
                "int",
                "Elastic-net pre-selection random seed.",
                False,
            ),
        ),
        spaces={
            "small": {
                "n_components": (1, 2, 3),
                "n_selected": (10, 25, 50),
                "min_abs_corr": (0.0,),
            },
            "standard": {
                "n_components": (1, 2, 3, 5),
                "n_selected": (10, 25, 50, 100),
                "min_abs_corr": (0.0, 0.05, 0.1),
            },
            "wide": {
                "n_components": (1, 2, 3, 5, 8),
                "n_selected": (10, 25, 50, 100, 200),
                "min_abs_corr": (0.0, 0.03, 0.05, 0.1, 0.2),
            },
        },
        description="Original-style iterative supervised PCA with residual correlation screening and projection.",
    ),
    "supervised_scaled_pca": _spec(
        "supervised_scaled_pca",
        "composite",
        supervised_scaled_pca,
        default_params={
            "n_components": 3,
            "n_selected": 50,
            "min_abs_corr": 0.0,
            "scale": True,
            "control_columns": None,
            "include_constant": True,
            "drop_control_columns": True,
            "preselect": "none",
            "t_threshold": 1.28,
            "elastic_net_alpha": 0.0002,
            "elastic_net_l1_ratio": 0.5,
            "quadratic_factors": False,
            "random_state": 0,
        },
        parameters=(
            _p("n_components", 3, "int", "Number of sequential SsPCA components."),
            _p(
                "n_selected",
                50,
                "int | None",
                "Predictors selected at each SPCA step after slope scaling.",
            ),
            _p(
                "min_abs_corr",
                0.0,
                "float",
                "Minimum absolute residual correlation retained before PCA.",
            ),
            _p(
                "scale",
                True,
                "bool",
                "Whether to standardize predictors and target inside the model.",
                False,
            ),
            _p(
                "control_columns",
                None,
                "Sequence[str] | None",
                "Optional X columns used as forecasting controls.",
                False,
            ),
            _p(
                "include_constant",
                True,
                "bool",
                "Whether to include a constant in the control block.",
                False,
            ),
            _p(
                "drop_control_columns",
                True,
                "bool",
                "Whether controls are excluded from the PCA block.",
                False,
            ),
            _p(
                "preselect",
                "none",
                "str",
                "Optional pre-selection: none, hard_tstat, or elastic_net.",
                False,
            ),
            _p(
                "t_threshold",
                1.28,
                "float",
                "Hard t-stat pre-selection threshold.",
                False,
            ),
            _p(
                "elastic_net_alpha",
                0.0002,
                "float",
                "Elastic-net pre-selection penalty.",
                False,
            ),
            _p(
                "elastic_net_l1_ratio",
                0.5,
                "float",
                "Elastic-net pre-selection L1 ratio.",
                False,
            ),
            _p(
                "quadratic_factors",
                False,
                "bool",
                "Whether to add the Hounyo-Li PC2 squared-factor forecast head.",
                False,
            ),
            _p(
                "random_state",
                0,
                "int",
                "Elastic-net pre-selection random seed.",
                False,
            ),
        ),
        spaces={
            "small": {
                "n_components": (1, 2, 3),
                "n_selected": (10, 25, 50),
                "min_abs_corr": (0.0,),
            },
            "standard": {
                "n_components": (1, 2, 3, 5),
                "n_selected": (10, 25, 50, 100),
                "min_abs_corr": (0.0, 0.05, 0.1),
            },
            "wide": {
                "n_components": (1, 2, 3, 5, 8),
                "n_selected": (10, 25, 50, 100, 200),
                "min_abs_corr": (0.0, 0.03, 0.05, 0.1, 0.2),
            },
        },
        description="Hounyo-Li supervised scaled PCA: marginal predictive-slope scaling followed by SPCA.",
    ),
    "ar": _spec(
        "ar",
        "timeseries",
        ar,
        default_params={"n_lag": 1, "direct": False},
        parameters=(
            _p("n_lag", 1, "int", "Autoregressive lag order."),
            _p("direct", False, "bool",
               "Direct multi-step projection onto fresh lags (set by the forecast policy).", False),
        ),
        spaces=_AR_SPACES,
        input_kind="supervised",
        description="Univariate autoregression.",
        selection_method="bic",
    ),
    "arima": _spec(
        "arima",
        "timeseries",
        arima,
        default_params={"order": (1, 0, 0), "seasonal_order": (0, 0, 0, 0), "trend": None},
        parameters=(
            _p("order", (1, 0, 0), "tuple[int, int, int]", "ARIMA (p, d, q) order."),
            _p("seasonal_order", (0, 0, 0, 0), "tuple[int, int, int, int]", "Seasonal (P, D, Q, m) order.", False),
            _p("trend", None, "str | None", "Deterministic trend ('n','c','t','ct').", False),
        ),
        input_kind="target",
        backend="statsmodels.tsa.arima.model.ARIMA",
        description="(Seasonal) ARIMA model.",
    ),
    "auto_arima": _spec(
        "auto_arima",
        "timeseries",
        auto_arima,
        default_params={"max_p": 5, "max_q": 5, "max_d": 2, "seasonal": False, "m": 1, "ic": "aicc"},
        parameters=(
            _p("max_p", 5, "int", "Maximum non-seasonal AR order.", False),
            _p("max_q", 5, "int", "Maximum non-seasonal MA order.", False),
            _p("max_d", 2, "int", "Maximum differencing order.", False),
            _p("seasonal", False, "bool", "Search seasonal orders.", False),
            _p("m", 1, "int", "Seasonal period.", False),
            _p("ic", "aicc", "str", "Selection criterion ('aicc','aic','bic').", False),
        ),
        input_kind="target",
        backend="statsmodels.tsa.arima.model.ARIMA",
        description="Automatic (seasonal) ARIMA order selection (forecast::auto.arima).",
    ),
    "var": _spec(
        "var",
        "timeseries",
        var,
        default_params={
            "target": None,
            "n_lag": 1,
            "type": "const",
            "season": None,
            "direct": False,
            "direct_horizon": 1,
        },
        parameters=(
            _p("target", None, "str | None", "Target column in the panel.", False),
            _p("n_lag", 1, "int", "VAR lag order."),
            _p(
                "type",
                "const",
                "str",
                "R vars::VAR deterministic terms: const, trend, both, or none.",
                False,
            ),
            _p(
                "season",
                None,
                "int | None",
                "Optional centered seasonal dummies, matching vars::VAR(season=...).",
                False,
            ),
            _p(
                "direct",
                False,
                "bool",
                "Internal: fit a horizon-specific point direct projection.",
                False,
            ),
            _p(
                "direct_horizon",
                1,
                "int",
                "Internal: horizon used when direct=True.",
                False,
            ),
        ),
        spaces=_AR_SPACES,
        input_kind="panel",
        backend="internal vars::VAR-aligned OLS",
        description="R vars::VAR-aligned vector autoregression point forecast.",
    ),
    "bvar_minnesota": _spec(
        "bvar_minnesota",
        "timeseries",
        bvar_minnesota,
        default_params={
            "target": None,
            "n_lag": 1,
            "kappa0": 2.0,
            "kappa1": 0.5,
            "nu0": 0.0,
            "s0": None,
            "iter": 300,
            "burnin": 100,
            "random_state": 0,
        },
        parameters=(
            _p("target", None, "str | None", "Target column in the panel.", False),
            _p("n_lag", 1, "int", "VAR lag order."),
            _p(
                "kappa0", 2.0, "float", "FAVAR/bvartools Minnesota own-lag prior scale."
            ),
            _p("kappa1", 0.5, "float", "FAVAR/bvartools Minnesota lag-decay exponent."),
            _p(
                "nu0",
                0.0,
                "float",
                "Inverse-Wishart degrees-of-freedom prior parameter.",
                False,
            ),
            _p(
                "s0",
                None,
                "float | matrix | None",
                "Inverse-Wishart scale prior parameter. None (default): "
                "data-dependent diag(AR/OLS residual variance) scale.",
                False,
            ),
            _p(
                "iter",
                300,
                "int",
                "Total Gibbs iterations (deep/paper-faithful default is 10000; "
                "pass explicitly to restore it).",
                False,
            ),
            _p(
                "burnin",
                100,
                "int",
                "Burn-in iterations discarded from posterior summaries "
                "(deep/paper-faithful default is 5000; pass explicitly to restore it).",
                False,
            ),
            _p("random_state", 0, "int", "Random seed for posterior draws.", False),
        ),
        spaces={
            "small": {"n_lag": (1, 2), "kappa0": (1.0, 2.0), "kappa1": (0.5,)},
            "standard": {
                "n_lag": (1, 2),
                "kappa0": (0.5, 1.0, 2.0),
                "kappa1": (0.5, 1.0),
            },
            "wide": {
                "n_lag": (1, 2, 4, 6, 12),
                "kappa0": (0.25, 0.5, 1.0, 2.0),
                "kappa1": (0.5, 1.0, 2.0),
            },
        },
        input_kind="panel",
        backend="internal FAVAR::BVAR-aligned Gibbs sampler",
        description="FAVAR::BVAR / bvartools Minnesota-prior Bayesian VAR posterior sampler.",
    ),
    "bvar_normal_inverse_wishart": _spec(
        "bvar_normal_inverse_wishart",
        "timeseries",
        bvar_normal_inverse_wishart,
        default_params={
            "target": None,
            "n_lag": 1,
            "b0": 0.0,
            "vb0": 0.0,
            "nu0": 0.0,
            "s0": None,
            "iter": 300,
            "burnin": 100,
            "random_state": 0,
        },
        parameters=(
            _p("target", None, "str | None", "Target column in the panel.", False),
            _p("n_lag", 1, "int", "VAR lag order."),
            _p("b0", 0.0, "float", "Normal prior mean for VAR coefficients.", False),
            _p(
                "vb0",
                0.0,
                "float",
                "Normal prior variance scale for VAR coefficients.",
                False,
            ),
            _p(
                "nu0",
                0.0,
                "float",
                "Inverse-Wishart degrees-of-freedom prior parameter.",
                False,
            ),
            _p(
                "s0",
                None,
                "float | matrix | None",
                "Inverse-Wishart scale prior parameter. None (default): "
                "data-dependent diag(AR/OLS residual variance) scale.",
                False,
            ),
            _p(
                "iter",
                300,
                "int",
                "Total Gibbs iterations (deep/paper-faithful default is 10000; "
                "pass explicitly to restore it).",
                False,
            ),
            _p(
                "burnin",
                100,
                "int",
                "Burn-in iterations discarded from posterior summaries "
                "(deep/paper-faithful default is 5000; pass explicitly to restore it).",
                False,
            ),
            _p("random_state", 0, "int", "Random seed for posterior draws.", False),
        ),
        spaces={
            "small": {"n_lag": (1, 2)},
            "standard": {"n_lag": (1, 2)},
            "wide": {"n_lag": (1, 2, 4, 6, 12)},
        },
        input_kind="panel",
        backend="internal FAVAR::BVAR-aligned Gibbs sampler",
        description="FAVAR::BVAR-aligned Bayesian VAR with normal/inverse-Wishart prior controls.",
    ),
    "ets": _spec(
        "ets",
        "timeseries",
        ets,
        default_params={
            "error": "add",
            "trend": None,
            "seasonal": None,
            "seasonal_periods": None,
            "damped_trend": False,
        },
        parameters=(
            _p("error", "add", "str", "ETS error form.", False),
            _p("trend", None, "str | None", "ETS trend form.", False),
            _p("seasonal", None, "str | None", "ETS seasonal form.", False),
            _p("seasonal_periods", None, "int | None", "Seasonal period.", False),
            _p("damped_trend", False, "bool", "Whether to damp the trend.", False),
        ),
        input_kind="target",
        backend="statsmodels.tsa.exponential_smoothing.ets.ETSModel",
        description="Statsmodels ETS target-only forecasting model.",
    ),
    "holt_winters": _spec(
        "holt_winters",
        "timeseries",
        holt_winters,
        default_params={
            "trend": "add",
            "seasonal": None,
            "seasonal_periods": None,
            "damped_trend": False,
        },
        parameters=(
            _p("trend", "add", "str | None", "Trend component.", False),
            _p("seasonal", None, "str | None", "Seasonal component.", False),
            _p("seasonal_periods", None, "int | None", "Seasonal period.", False),
            _p("damped_trend", False, "bool", "Whether to damp the trend.", False),
        ),
        input_kind="target",
        backend="statsmodels.tsa.holtwinters.ExponentialSmoothing",
        description="Holt-Winters exponential smoothing target-only forecasting model.",
    ),
    "naive": _spec(
        "naive",
        "timeseries",
        naive,
        default_params={},
        parameters=(),
        input_kind="target",
        backend="macroforecast",
        description="Random-walk (naive) baseline: carry the last value forward (forecast::naive).",
    ),
    "seasonal_naive": _spec(
        "seasonal_naive",
        "timeseries",
        seasonal_naive,
        default_params={"period": None},
        parameters=(
            _p("period", None, "int | None", "Seasonal period m; repeats the last m values.", False),
        ),
        input_kind="target",
        backend="macroforecast",
        description="Seasonal-naive baseline: repeat the last seasonal cycle (forecast::snaive).",
    ),
    "random_walk_drift": _spec(
        "random_walk_drift",
        "timeseries",
        random_walk_drift,
        default_params={},
        parameters=(),
        input_kind="target",
        backend="macroforecast",
        description="Random-walk-with-drift baseline (forecast::rwf(drift=TRUE)).",
    ),
    "stlf": _spec(
        "stlf",
        "timeseries",
        stlf,
        default_params={"period": None, "sa_method": "ets"},
        parameters=(
            _p("period", None, "int | None", "Seasonal period; inferred from the index if omitted.", False),
            _p("sa_method", "ets", "str", "Forecaster for the seasonally-adjusted series ('ets').", False),
        ),
        input_kind="target",
        backend="statsmodels.tsa.seasonal.STL",
        description="STL decomposition + forecast of the seasonally-adjusted series (forecast::stlf).",
    ),
    "theta_method": _spec(
        "theta_method",
        "timeseries",
        theta_method,
        default_params={"period": None, "deseasonalize": True, "use_test": True},
        parameters=(
            _p("period", None, "int | None", "Seasonal period.", False),
            _p("deseasonalize", True, "bool", "Whether to deseasonalize.", False),
            _p("use_test", True, "bool", "Statsmodels seasonality test flag.", False),
        ),
        input_kind="target",
        backend="statsmodels.tsa.forecasting.theta.ThetaModel",
        description="Theta method target-only forecasting model.",
    ),
    "dfm_mixed_mariano_murasawa": _spec(
        "dfm_mixed_mariano_murasawa",
        "mixed_frequency",
        dfm_mixed_mariano_murasawa,
        default_params={
            "target": None,
            "metadata": None,
            "monthly_columns": None,
            "quarterly_columns": None,
            "unsupported": "raise",
            "n_factors": 1,
            "factor_order": 1,
            "idiosyncratic_ar1": True,
            "standardize": True,
            "maxiter": 500,
            "tolerance": 1e-6,
        },
        parameters=(
            _p(
                "target",
                None,
                "str | None",
                "Target column; defaults to first quarterly column.",
                False,
            ),
            _p(
                "metadata",
                None,
                "Mapping[str, Any] | None",
                "Data metadata with native frequencies.",
                False,
            ),
            _p(
                "monthly_columns",
                None,
                "Iterable[str] | None",
                "Explicit monthly columns.",
                False,
            ),
            _p(
                "quarterly_columns",
                None,
                "Iterable[str] | None",
                "Explicit quarterly columns.",
                False,
            ),
            _p(
                "unsupported",
                "raise",
                "str",
                "Unsupported-frequency policy: 'raise' or 'drop'.",
                False,
            ),
            _p("n_factors", 1, "int", "Number of dynamic factors."),
            _p("factor_order", 1, "int", "VAR order for factor dynamics."),
            _p(
                "idiosyncratic_ar1",
                True,
                "bool",
                "Whether idiosyncratic disturbances are AR(1).",
                False,
            ),
            _p(
                "standardize",
                True,
                "bool",
                "Whether statsmodels standardizes observed series.",
                False,
            ),
            _p("maxiter", 500, "int", "EM iteration cap.", False),
            _p("tolerance", 1e-6, "float", "EM convergence tolerance.", False),
        ),
        spaces={
            "small": {"n_factors": (1,), "factor_order": (1,)},
            "standard": {"n_factors": (1, 2), "factor_order": (1, 2)},
            "wide": {"n_factors": (1, 2, 3), "factor_order": (1, 2, 3)},
        },
        input_kind="panel",
        backend="statsmodels.tsa.statespace.dynamic_factor_mq.DynamicFactorMQ",
        recommended_preprocessing=(
            "pass a native mixed monthly/quarterly panel from macroforecast.data.combine(..., frequency='native')",
            "keep quarterly flow variables on their observed quarterly dates; the model applies Mariano-Murasawa aggregation",
        ),
        description="Mixed-frequency dynamic factor model using Mariano-Murasawa quarterly aggregation.",
    ),
    "midas_almon": _spec(
        "midas_almon",
        "mixed_frequency",
        midas_almon,
        default_params={
            "polynomial_order": 2,
            "theta": None,
            "alpha": 0.0,
            "fit_intercept": True,
        },
        parameters=(
            _p("polynomial_order", 2, "int", "Almon polynomial order.", False),
            _p(
                "theta",
                None,
                "tuple[float, ...] | None",
                "midasr::nealmon shape coefficients; length must equal polynomial_order.",
                False,
            ),
            _p(
                "alpha",
                0.0,
                "float",
                "Optional ridge penalty on aggregated MIDAS regressors.",
            ),
            _p("fit_intercept", True, "bool", "Whether to fit an intercept.", False),
        ),
        spaces={
            "small": {"alpha": (0.0, 0.1, 1.0)},
            "standard": {"alpha": (0.0, 0.01, 0.1, 1.0), "polynomial_order": (1, 2, 3)},
            "wide": {
                "alpha": (0.0, 0.001, 0.01, 0.1, 1.0, 10.0),
                "polynomial_order": (1, 2, 3, 4),
            },
        },
        description="Fixed-shape MIDAS over lag groups using midasr::nealmon-style normalized exponential Almon weights.",
    ),
    "midas_beta": _spec(
        "midas_beta",
        "mixed_frequency",
        midas_beta,
        default_params={"beta_params": (1.0, 1.0), "alpha": 0.0, "fit_intercept": True},
        parameters=(
            _p(
                "beta_params",
                (1.0, 1.0),
                "tuple[float, float]",
                "midasr::nbetaMT beta lag-weight shape parameters.",
            ),
            _p(
                "alpha",
                0.0,
                "float",
                "Optional ridge penalty on aggregated MIDAS regressors.",
            ),
            _p("fit_intercept", True, "bool", "Whether to fit an intercept.", False),
        ),
        spaces={
            "small": {
                "alpha": (0.0, 0.1, 1.0),
                "beta_params": ((1.0, 1.0), (1.0, 2.0), (2.0, 1.0)),
            },
            "standard": {
                "alpha": (0.0, 0.01, 0.1, 1.0),
                "beta_params": ((1.0, 1.0), (1.0, 2.0), (2.0, 1.0), (2.0, 2.0)),
            },
            "wide": {
                "alpha": (0.0, 0.001, 0.01, 0.1, 1.0, 10.0),
                "beta_params": (
                    (0.5, 0.5),
                    (1.0, 1.0),
                    (1.0, 2.0),
                    (2.0, 1.0),
                    (2.0, 2.0),
                    (3.0, 1.0),
                ),
            },
        },
        description="Fixed-shape MIDAS over lag groups using midasr::nbetaMT-style beta weights.",
    ),
    "midas_step": _spec(
        "midas_step",
        "mixed_frequency",
        midas_step,
        default_params={
            "n_steps": 3,
            "step_bounds": None,
            "step_weights": None,
            "alpha": 0.0,
            "fit_intercept": True,
        },
        parameters=(
            _p(
                "n_steps",
                3,
                "int",
                "Number of lag buckets when step_bounds is not supplied.",
            ),
            _p(
                "step_bounds",
                None,
                "tuple[int, ...] | None",
                "Optional midasr::polystep-style interior cut points.",
                False,
            ),
            _p(
                "step_weights",
                None,
                "tuple[float, ...] | None",
                "Optional raw height for each step bucket.",
                False,
            ),
            _p(
                "alpha",
                0.0,
                "float",
                "Optional ridge penalty on aggregated MIDAS regressors.",
            ),
            _p("fit_intercept", True, "bool", "Whether to fit an intercept.", False),
        ),
        spaces={
            "small": {"n_steps": (2, 3), "alpha": (0.0, 0.1, 1.0)},
            "standard": {"n_steps": (2, 3, 4), "alpha": (0.0, 0.01, 0.1, 1.0)},
            "wide": {
                "n_steps": (2, 3, 4, 6),
                "alpha": (0.0, 0.001, 0.01, 0.1, 1.0, 10.0),
            },
        },
        description="Fixed-shape MIDAS over lag groups using normalized midasr::polystep-style step weights.",
    ),
    "restricted_midas": _spec(
        "restricted_midas",
        "mixed_frequency",
        restricted_midas,
        default_params={
            "weighting": "almon",
            "polynomial_order": 2,
            "start_params": None,
            "n_steps": 3,
            "step_bounds": None,
            "fit_intercept": True,
            "maxiter": 200,
            "tolerance": 1e-6,
        },
        parameters=(
            _p(
                "weighting",
                "almon",
                "str",
                "Restriction map: 'almon'/'nealmon', 'beta'/'nbetaMT', or 'step'/'polystep'.",
                False,
            ),
            _p(
                "polynomial_order",
                2,
                "int",
                "Almon polynomial order after the aggregate scale parameter.",
                False,
            ),
            _p(
                "start_params",
                None,
                "Mapping[str, Sequence[float]] | Sequence[float] | None",
                "midasr::midas_r-style starting values for each lag group.",
                False,
            ),
            _p(
                "n_steps",
                3,
                "int",
                "Number of step buckets when weighting='step' and step_bounds is not supplied.",
                False,
            ),
            _p(
                "step_bounds",
                None,
                "tuple[int, ...] | None",
                "Optional midasr::polystep-style interior cut points.",
                False,
            ),
            _p(
                "fit_intercept",
                True,
                "bool",
                "Whether to estimate an intercept outside the restricted lag maps.",
                False,
            ),
            _p(
                "maxiter",
                200,
                "int",
                "Maximum SciPy least_squares function evaluations "
                "(deep/paper-faithful default is 1000; pass explicitly to restore it).",
                False,
            ),
            _p(
                "tolerance",
                1e-6,
                "float",
                "least_squares xtol/ftol/gtol "
                "(deep/paper-faithful default is 1e-8; pass explicitly to restore it).",
                False,
            ),
        ),
        spaces={
            "small": {"weighting": ("almon",), "polynomial_order": (1, 2)},
            "standard": {"weighting": ("almon", "beta"), "polynomial_order": (1, 2, 3)},
            "wide": {
                "weighting": ("almon", "beta", "step"),
                "polynomial_order": (1, 2, 3, 4),
                "n_steps": (2, 3, 4),
            },
        },
        description="midasr::midas_r-style nonlinear restricted MIDAS over explicit lag columns.",
    ),
    "unrestricted_midas": _spec(
        "unrestricted_midas",
        "mixed_frequency",
        unrestricted_midas,
        default_params={"alpha": 0.0, "fit_intercept": True},
        parameters=(
            _p(
                "alpha",
                0.0,
                "float",
                "Optional ridge penalty on unrestricted lag coefficients.",
            ),
            _p("fit_intercept", True, "bool", "Whether to fit an intercept.", False),
        ),
        spaces={
            "small": {"alpha": (0.0, 0.1, 1.0)},
            "standard": {"alpha": (0.0, 0.01, 0.1, 1.0)},
            "wide": {"alpha": (0.0, 0.001, 0.01, 0.1, 1.0, 10.0)},
        },
        description="Unrestricted MIDAS over explicit lag columns.",
    ),
    "dfm_unrestricted_midas": _spec(
        "dfm_unrestricted_midas",
        "mixed_frequency",
        dfm_unrestricted_midas,
        default_params={
            "target": None,
            "metadata": None,
            "lag_columns": None,
            "lags": (0, 1, 2),
            "factor_lags": (0,),
            "target_frequency": "quarterly",
            "anchor_position": "period_end",
            "n_factors": 1,
            "factor_order": 1,
            "idiosyncratic_ar1": True,
            "standardize": True,
            "maxiter": 500,
            "tolerance": 1e-6,
            "alpha": 0.0,
            "fit_intercept": True,
            "drop_missing": True,
        },
        parameters=(
            _p("target", None, "str", "Target column.", False),
            _p(
                "metadata",
                None,
                "Mapping[str, Any] | None",
                "Data metadata with native frequencies.",
                False,
            ),
            _p(
                "lag_columns",
                None,
                "Iterable[str] | None",
                "Observed columns to add as unrestricted MIDAS lags.",
                False,
            ),
            _p(
                "lags",
                (0, 1, 2),
                "Iterable[int] | int",
                "Observed-column native-frequency lags.",
            ),
            _p("factor_lags", (0,), "Iterable[int] | int", "DFM factor monthly lags."),
            _p(
                "target_frequency",
                "quarterly",
                "str | None",
                "Frequency used to position target anchors.",
                False,
            ),
            _p(
                "anchor_position",
                "period_end",
                "str",
                "Anchor date positioning.",
                False,
            ),
            _p("n_factors", 1, "int", "Number of DFM factors."),
            _p("factor_order", 1, "int", "VAR order for DFM factor dynamics."),
            _p(
                "idiosyncratic_ar1",
                True,
                "bool",
                "Whether DFM idiosyncratic disturbances are AR(1).",
                False,
            ),
            _p(
                "standardize",
                True,
                "bool",
                "Whether DynamicFactorMQ standardizes observed variables.",
                False,
            ),
            _p("maxiter", 500, "int", "DFM EM iteration cap.", False),
            _p("tolerance", 1e-6, "float", "DFM EM convergence tolerance.", False),
            _p(
                "alpha",
                0.0,
                "float",
                "Optional ridge penalty on unrestricted MIDAS head.",
            ),
            _p("fit_intercept", True, "bool", "Whether to fit an intercept.", False),
            _p(
                "drop_missing",
                True,
                "bool",
                "Drop incomplete composite-design rows.",
                False,
            ),
        ),
        spaces={
            "small": {
                "lags": ((0, 1, 2),),
                "factor_lags": ((0,),),
                "n_factors": (1,),
                "factor_order": (1,),
                "alpha": (0.0, 0.1, 1.0),
            },
            "standard": {
                "lags": ((0, 1, 2), (0, 1, 2, 3, 4, 5)),
                "factor_lags": ((0,), (0, 1)),
                "n_factors": (1, 2),
                "factor_order": (1, 2),
                "alpha": (0.0, 0.01, 0.1, 1.0),
            },
            "wide": {
                "lags": ((0, 1, 2), (0, 1, 2, 3, 4, 5), tuple(range(12))),
                "factor_lags": ((0,), (0, 1), (0, 1, 2)),
                "n_factors": (1, 2, 3),
                "factor_order": (1, 2, 3),
                "alpha": (0.0, 0.001, 0.01, 0.1, 1.0, 10.0),
            },
        },
        input_kind="panel",
        backend="statsmodels DynamicFactorMQ + sklearn linear/ridge",
        recommended_preprocessing=(
            "pass a native mixed monthly/quarterly panel with column-level frequency metadata",
            "use feature_engineering.mixed_frequency_lags directly when you need full manual control",
        ),
        description="Composite DynamicFactorMQ factors plus unrestricted MIDAS forecast head.",
    ),
    "far": _spec(
        "far",
        "factor",
        far,
        default_params={"n_factors": 3, "n_lag": 1, "random_state": 0, "direct": False},
        parameters=(
            _p("n_factors", 3, "int", "Number of PCA factors."),
            _p("n_lag", 1, "int", "Autoregressive lag order."),
            _p("random_state", 0, "int", "PCA random seed.", False),
            _p("direct", False, "bool",
               "Direct multi-step projection onto fresh lags (set by the forecast policy).", False),
        ),
        spaces={
            key: {**space, **_AR_SPACES[key]} for key, space in _FACTOR_SPACES.items()
        },
        description="Factor-augmented autoregression.",
        selection_method="bic",
    ),
    "favar": _spec(
        "favar",
        "factor",
        favar,
        default_params={
            "n_factors": 2,
            "n_lag": 2,
            "fctmethod": "BGM",
            "slowcode": None,
            "factorprior": None,
            "varprior": None,
            "nburn": 100,
            "nrep": 200,
            "standardize": True,
            "random_state": 0,
        },
        parameters=(
            _p("n_factors", 2, "int", "Number of latent factors."),
            _p("n_lag", 2, "int", "VAR lag order on target plus factors."),
            _p(
                "fctmethod",
                "BGM",
                "str",
                "FAVAR factor identification method: BBE or BGM (default BGM; BBE requires slowcode).",
                False,
            ),
            _p(
                "slowcode",
                None,
                "Sequence[bool] | None",
                "Slow-variable mask required by BBE.",
                False,
            ),
            _p(
                "factorprior",
                None,
                "Mapping[str, Any] | None",
                "Factor loading prior controls.",
                False,
            ),
            _p(
                "varprior",
                None,
                "Mapping[str, Any] | None",
                "BVAR prior controls for the factor VAR block.",
                False,
            ),
            _p(
                "nburn",
                100,
                "int",
                "Burn-in iterations for loading/BVAR posterior draws "
                "(deep/paper-faithful default is 5000; pass explicitly to restore it).",
                False,
            ),
            _p(
                "nrep",
                200,
                "int",
                "Saved loading draws and post-burn BVAR draw count "
                "(deep/paper-faithful default is 15000; pass explicitly to restore it).",
                False,
            ),
            _p(
                "standardize",
                True,
                "bool",
                "Use R scale() semantics for X and y before factor extraction.",
                False,
            ),
            _p("random_state", 0, "int", "Random seed for posterior draws.", False),
        ),
        spaces=_FAVAR_SPACES,
        backend="internal FAVAR::FAVAR-aligned Bayesian sampler",
        description="FAVAR::FAVAR-aligned Bayesian factor-augmented VAR sampler.",
    ),
    "decision_tree": _spec(
        "decision_tree",
        "tree",
        decision_tree,
        default_params={"max_depth": None, "min_samples_leaf": 1, "random_state": 0},
        parameters=(
            _p("max_depth", None, "int | None", "Maximum tree depth."),
            _p("min_samples_leaf", 1, "int", "Minimum samples per terminal leaf."),
            _p("random_state", 0, "int", "Tree random seed.", False),
        ),
        spaces=_TREE_SPACES,
        backend="sklearn.tree.DecisionTreeRegressor",
        description="CART regression tree.",
    ),
    "random_forest": _spec(
        "random_forest",
        "tree",
        random_forest,
        default_params={
            "n_estimators": 200,
            "max_depth": None,
            "min_samples_leaf": 1,
            "random_state": 0,
            "n_jobs": None,
        },
        parameters=(
            _p("n_estimators", 200, "int", "Number of trees."),
            _p("max_depth", None, "int | None", "Maximum depth per tree."),
            _p("min_samples_leaf", 1, "int", "Minimum samples per terminal leaf."),
            _p("random_state", 0, "int", "Forest random seed.", False),
            _p("n_jobs", None, "int | None", "Parallel worker count (None resolves to meta.configure(n_jobs)).", False),
        ),
        spaces=_FOREST_SPACES,
        method="random",
        backend="sklearn.ensemble.RandomForestRegressor",
        description="Random forest regression.",
    ),
    "extra_trees": _spec(
        "extra_trees",
        "tree",
        extra_trees,
        default_params={
            "n_estimators": 200,
            "max_depth": None,
            "min_samples_leaf": 1,
            "random_state": 0,
            "n_jobs": None,
        },
        parameters=(
            _p("n_estimators", 200, "int", "Number of trees."),
            _p("max_depth", None, "int | None", "Maximum depth per tree."),
            _p("min_samples_leaf", 1, "int", "Minimum samples per terminal leaf."),
            _p("random_state", 0, "int", "Forest random seed.", False),
            _p("n_jobs", None, "int | None", "Parallel worker count (None resolves to meta.configure(n_jobs)).", False),
        ),
        spaces=_FOREST_SPACES,
        method="random",
        backend="sklearn.ensemble.ExtraTreesRegressor",
        description="Extremely randomized trees.",
    ),
    "gradient_boosting": _spec(
        "gradient_boosting",
        "tree",
        gradient_boosting,
        default_params={
            "n_estimators": 200,
            "learning_rate": 0.1,
            "max_depth": 3,
            "random_state": 0,
        },
        parameters=(
            _p("n_estimators", 200, "int", "Number of boosting stages."),
            _p("learning_rate", 0.1, "float", "Shrinkage per stage."),
            _p("max_depth", 3, "int", "Maximum tree depth."),
            _p("random_state", 0, "int", "Boosting random seed.", False),
        ),
        spaces=_BOOSTING_SPACES,
        method="random",
        backend="sklearn.ensemble.GradientBoostingRegressor",
        description="Gradient-boosted regression trees.",
    ),
    "mars": _spec(
        "mars",
        "spline",
        mars,
        default_params={
            "max_terms": 20,
            "max_degree": 1,
            "n_knots": 10,
            "min_improvement": 1e-6,
            "penalty": 2.0,
            "prune": True,
        },
        parameters=(
            _p(
                "max_terms",
                20,
                "int",
                "Maximum number of hinge basis terms including intercept.",
            ),
            _p(
                "max_degree",
                1,
                "int",
                "Maximum interaction degree among hinge factors.",
            ),
            _p("n_knots", 10, "int", "Candidate quantile knots per predictor."),
            _p(
                "min_improvement",
                1e-6,
                "float",
                "Forward-step relative RSS improvement floor.",
                False,
            ),
            _p("penalty", 2.0, "float", "GCV pruning complexity penalty.", False),
            _p("prune", True, "bool", "Whether to prune terms by GCV.", False),
        ),
        spaces={
            "small": {"max_terms": (8, 12), "max_degree": (1,), "n_knots": (5, 10)},
            "standard": {
                # Drops the max_degree=2 x max_terms=30 corner: that combo is the
                # single most expensive cell in the preset (each interaction-degree
                # candidate reruns the forward pass over all pairwise hinge
                # products, and max_terms=30 forces many more forward steps before
                # GCV pruning), which was enough on its own to blow the per-model
                # scan-worker time budget. max_degree=2 stays reachable at
                # max_terms<=20; the richer combo remains available in "wide".
                "max_terms": (10, 20),
                "max_degree": (1, 2),
                "n_knots": (5, 10),
            },
            "wide": {
                "max_terms": (10, 20, 30, 50),
                "max_degree": (1, 2),
                "n_knots": (5, 10, 20),
            },
        },
        method="random",
        backend="internal",
        description="Package-native MARS-style hinge-basis regression.",
    ),
    "xgboost": _spec(
        "xgboost",
        "tree",
        xgboost,
        default_params={
            "n_estimators": 300,
            "learning_rate": 0.1,
            "max_depth": 6,
            "subsample": 1.0,
            "random_state": 0,
        },
        parameters=(
            _p("n_estimators", 300, "int", "Number of boosting stages."),
            _p("learning_rate", 0.1, "float", "Shrinkage per stage."),
            _p("max_depth", 6, "int", "Maximum tree depth."),
            _p("subsample", 1.0, "float", "Row subsample share."),
            _p("random_state", 0, "int", "Boosting random seed.", False),
        ),
        spaces={
            key: {**space, "subsample": (0.6, 0.8, 1.0)}
            for key, space in _BOOSTING_SPACES.items()
        },
        method="random",
        backend="xgboost.XGBRegressor",
        requires_extra="xgboost",
        description="XGBoost regressor.",
    ),
    "lightgbm": _spec(
        "lightgbm",
        "tree",
        lightgbm,
        default_params={
            "n_estimators": 300,
            "learning_rate": 0.1,
            "max_depth": -1,
            "num_leaves": 31,
            "random_state": 0,
        },
        parameters=(
            _p("n_estimators", 300, "int", "Number of boosting stages."),
            _p("learning_rate", 0.1, "float", "Shrinkage per stage."),
            _p("max_depth", -1, "int", "Maximum tree depth; -1 means no limit."),
            _p("num_leaves", 31, "int", "Maximum leaves per tree."),
            _p("random_state", 0, "int", "Boosting random seed.", False),
        ),
        spaces={
            "small": {
                "n_estimators": (50, 100),
                "learning_rate": (0.05, 0.1),
                "max_depth": (-1, 3, 5),
                "num_leaves": (15, 31),
            },
            "standard": {
                "n_estimators": (100, 200, 500),
                "learning_rate": (0.03, 0.05, 0.1),
                "max_depth": (-1, 3, 5, 10),
                "num_leaves": (15, 31, 63),
            },
            "wide": {
                "n_estimators": (100, 200, 500, 1000),
                "learning_rate": (0.01, 0.03, 0.05, 0.1),
                "max_depth": (-1, 3, 5, 10, 20),
                "num_leaves": (15, 31, 63, 127),
            },
        },
        method="random",
        backend="lightgbm.LGBMRegressor",
        requires_extra="lightgbm",
        description="LightGBM regressor.",
    ),
    "lgb_plus": _spec(
        "lgb_plus",
        "tree",
        lgb_plus,
        default_params={
            "n_ensemble": 3,
            "n_steps": 30,
            "learning_rate": 0.05,
            "subsample": 0.7,
            "num_leaves": 5,
            "min_data_in_leaf": 20,
            "lambda_l2": 0.1,
            "linear_candidate_fraction": 0.5,
            "selection_method": "oob",
            "val_fraction": 0.2,
            "early_stop_patience": 50,
            "aggregation": "mean",
            "random_state": 0,
        },
        parameters=(
            _p(
                "n_ensemble",
                3,
                "int",
                "Independent LGB+ ensemble members "
                "(deep/paper-faithful default is 10; pass explicitly to restore it).",
            ),
            _p(
                "n_steps",
                30,
                "int",
                "Maximum tree/linear competition steps per member "
                "(deep/paper-faithful default is 200; pass explicitly to restore it).",
            ),
            _p(
                "learning_rate",
                0.05,
                "float",
                "Shared shrinkage for tree and linear updates.",
            ),
            _p("subsample", 0.7, "float", "Row subsample share per competition step."),
            _p(
                "num_leaves",
                5,
                "int",
                "Maximum leaves for each one-step LightGBM tree.",
            ),
            _p("min_data_in_leaf", 20, "int", "Minimum rows in a LightGBM leaf."),
            _p("lambda_l2", 0.1, "float", "LightGBM tree L2 regularization.", False),
            _p(
                "linear_candidate_fraction",
                0.5,
                "float",
                "Fraction of features sampled before greedy linear residual selection.",
            ),
            _p(
                "selection_method",
                "oob",
                "str",
                "Candidate judge: 'oob', 'validation', or 'training'.",
                False,
            ),
            _p(
                "val_fraction",
                0.2,
                "float",
                "Fixed validation share when selection_method='validation'.",
                False,
            ),
            _p(
                "early_stop_patience",
                50,
                "int | None",
                "Stop after no selection-loss improvement.",
                False,
            ),
            _p(
                "aggregation",
                "mean",
                "str",
                "Ensemble aggregation: 'mean' or 'median'.",
                False,
            ),
            _p("random_state", 0, "int | None", "Base random seed.", False),
        ),
        spaces={
            "small": {
                "n_ensemble": (3, 5),
                "n_steps": (50, 100),
                "learning_rate": (0.03, 0.05),
                "subsample": (0.6, 0.7),
                "num_leaves": (5, 7),
                "min_data_in_leaf": (10, 20),
                "linear_candidate_fraction": (0.33, 0.5),
            },
            # n_ensemble/n_steps here (not just default_params) drive the search-time
            # cost: model_selection=None still runs the "standard" preset's random
            # search, and its worst combo alone (n_ensemble=10 x n_steps=400 = 4000
            # sequential lgb.train calls per fit) was enough to blow the per-model
            # scan-worker time budget even after cheapening default_params. Capped to
            # keep any sampled combo affordable; "wide" keeps the richer, paper-style
            # range for deep/explicit use.
            "standard": {
                "n_ensemble": (3, 5),
                "n_steps": (30, 50, 100),
                "learning_rate": (0.02, 0.05, 0.1),
                "subsample": (0.6, 0.7, 0.8),
                "num_leaves": (5, 7, 10),
                "min_data_in_leaf": (10, 20, 30),
                "linear_candidate_fraction": (0.33, 0.5, 1.0),
            },
            "wide": {
                "n_ensemble": (5, 10, 20),
                "n_steps": (100, 200, 400, 600),
                "learning_rate": (0.01, 0.02, 0.05, 0.1),
                "subsample": (0.5, 0.6, 0.7, 0.8),
                "num_leaves": (5, 7, 10, 15),
                "min_data_in_leaf": (5, 10, 20, 30),
                "linear_candidate_fraction": (0.25, 0.33, 0.5, 1.0),
            },
        },
        method="random",
        backend="internal philgoucou/lgbplus-aligned + lightgbm.train",
        requires_extra="lightgbm",
        description="LGB+ competition hybrid boosting with tree/linear channel diagnostics.",
    ),
    "lgba_plus": _spec(
        "lgba_plus",
        "tree",
        lgba_plus,
        default_params={
            "n_runs": 1,
            "n_cycles": 25,
            "trees_per_cycle": 10,
            "lr_tree": 0.02,
            "lr_linear": 0.1,
            "num_leaves": 15,
            "min_data_in_leaf": 20,
            "subsample": 1.0,
            "random_state": 0,
        },
        parameters=(
            _p("n_runs", 1, "int", "Independent alternating-run ensemble members."),
            _p(
                "n_cycles",
                25,
                "int",
                "Alternating tree-block plus linear-update cycles.",
            ),
            _p("trees_per_cycle", 10, "int", "LightGBM residual trees per cycle."),
            _p("lr_tree", 0.02, "float", "Shrinkage for tree-block updates."),
            _p("lr_linear", 0.1, "float", "Shrinkage for univariate linear updates."),
            _p("num_leaves", 15, "int", "Maximum leaves for each residual tree."),
            _p("min_data_in_leaf", 20, "int", "Minimum rows in a LightGBM leaf."),
            _p(
                "subsample",
                1.0,
                "float",
                "LightGBM bagging fraction for tree blocks.",
                False,
            ),
            _p("random_state", 0, "int | None", "Base random seed.", False),
        ),
        spaces={
            "small": {
                "n_runs": (1, 3),
                "n_cycles": (5, 10),
                "trees_per_cycle": (3, 5),
                "lr_tree": (0.02, 0.05),
                "lr_linear": (0.05, 0.1),
                "num_leaves": (5, 10),
                "min_data_in_leaf": (10, 20),
            },
            "standard": {
                "n_runs": (1, 5),
                "n_cycles": (10, 25),
                "trees_per_cycle": (5, 10),
                "lr_tree": (0.01, 0.02, 0.05),
                "lr_linear": (0.05, 0.1, 0.2),
                "num_leaves": (5, 10, 15),
                "min_data_in_leaf": (10, 20, 30),
                "subsample": (0.7, 1.0),
            },
            "wide": {
                "n_runs": (1, 5, 10),
                "n_cycles": (10, 25, 50),
                "trees_per_cycle": (5, 10, 20),
                "lr_tree": (0.01, 0.02, 0.05),
                "lr_linear": (0.03, 0.05, 0.1, 0.2),
                "num_leaves": (5, 10, 15, 31),
                "min_data_in_leaf": (5, 10, 20, 30),
                "subsample": (0.6, 0.7, 1.0),
            },
        },
        method="random",
        backend="internal philgoucou/lgbplus-aligned + lightgbm.train",
        requires_extra="lightgbm",
        description="LGB^A+ alternating tree-block and greedy linear boosting.",
    ),
    "catboost": _spec(
        "catboost",
        "tree",
        catboost,
        default_params={
            "n_estimators": 300,
            "learning_rate": 0.1,
            "max_depth": 6,
            "random_state": 0,
            "verbose": False,
        },
        parameters=(
            _p("n_estimators", 300, "int", "Number of boosting stages."),
            _p("learning_rate", 0.1, "float", "Shrinkage per stage."),
            _p("max_depth", 6, "int", "Tree depth."),
            _p("random_state", 0, "int", "Boosting random seed.", False),
            _p("verbose", False, "bool", "CatBoost console output flag.", False),
        ),
        spaces=_BOOSTING_SPACES,
        method="random",
        backend="catboost.CatBoostRegressor",
        requires_extra="catboost",
        description="CatBoost regressor.",
    ),
    "quantile_regression_forest": _spec(
        "quantile_regression_forest",
        "tree",
        quantile_regression_forest,
        default_params={
            "n_estimators": 200,
            "max_depth": None,
            "min_samples_leaf": 1,
            "random_state": 0,
            "quantile_levels": (0.05, 0.5, 0.95),
        },
        parameters=(
            _p("n_estimators", 200, "int", "Number of trees."),
            _p("max_depth", None, "int | None", "Maximum depth per tree."),
            _p("min_samples_leaf", 1, "int", "Minimum samples per terminal leaf."),
            _p("random_state", 0, "int", "Forest random seed.", False),
            _p(
                "quantile_levels",
                (0.05, 0.5, 0.95),
                "tuple[float, ...]",
                "Default quantile levels returned by predict_quantiles().",
                False,
            ),
        ),
        spaces=_FOREST_SPACES,
        method="random",
        backend="sklearn.ensemble.RandomForestRegressor + internal leaf quantiles",
        description="Quantile regression forest.",
    ),
    "macro_random_forest": _spec(
        "macro_random_forest",
        "tree",
        macro_random_forest,
        default_params={
            "x_columns": None,
            "S_columns": None,
            "x_pos": None,
            "S_pos": None,
            "y_pos": 0,
            "B": 25,
            "minsize": 10,
            "mtry_frac": 1.0 / 3.0,
            "min_leaf_frac_of_x": 1.0,
            "VI": False,
            "ERT": False,
            "quantile_rate": None,
            "S_priority_vec": None,
            "random_x": False,
            "trend_push": 1,
            "howmany_random_x": 1,
            "howmany_keep_best_VI": 20,
            "cheap_look_at_GTVPs": True,
            "prior_var": None,
            "prior_mean": None,
            "subsampling_rate": 0.75,
            "rw_regul": 0.75,
            "keep_forest": False,
            "block_size": 12,
            "fast_rw": True,
            "ridge_lambda": 0.1,
            "HRW": 0,
            "resampling_opt": 2,
            "parallelise": False,
            "n_cores": 1,
            "print_b": False,
        },
        parameters=(
            _p(
                "x_columns",
                None,
                "list[str] | None",
                "Predictors in the time-varying linear equation.",
                False,
            ),
            _p(
                "S_columns",
                None,
                "list[str] | None",
                "State variables entering the forest split function.",
                False,
            ),
            _p(
                "x_pos",
                None,
                "list[int] | None",
                "Reference-package predictor positions after the target column.",
                False,
            ),
            _p(
                "S_pos",
                None,
                "list[int] | None",
                "Reference-package state positions after the target column.",
                False,
            ),
            _p(
                "y_pos",
                0,
                "int",
                "Fixed target position for the X/y callable adapter; must remain 0.",
                False,
            ),
            _p(
                "B",
                25,
                "int",
                "Number of MRF trees "
                "(deep/paper-faithful default is 50; pass explicitly to restore it).",
            ),
            _p("minsize", 10, "int", "Minimum node size before split attempts."),
            _p(
                "mtry_frac",
                1.0 / 3.0,
                "float",
                "Fraction of state variables considered at each split.",
            ),
            _p(
                "min_leaf_frac_of_x",
                1.0,
                "float",
                "Minimum leaf-size multiplier relative to local x dimension.",
            ),
            _p(
                "VI",
                False,
                "bool",
                "Enable variable-importance split search mode.",
                False,
            ),
            _p(
                "ERT",
                False,
                "bool",
                "Enable extremely randomized tree split mode.",
                False,
            ),
            _p(
                "quantile_rate",
                None,
                "float | None",
                "Optional quantile rate for quantile-oriented output.",
                False,
            ),
            _p(
                "S_priority_vec",
                None,
                "list[float] | None",
                "Optional priority weights over state variables.",
                False,
            ),
            _p(
                "random_x",
                False,
                "bool",
                "Use random subsets of local-linear predictors.",
                False,
            ),
            _p("trend_push", 1, "int", "Reference-package trend-push option.", False),
            _p(
                "howmany_random_x",
                1,
                "int",
                "Number of random local-linear predictor draws.",
                False,
            ),
            _p(
                "howmany_keep_best_VI",
                20,
                "int",
                "Number of best VI candidates retained.",
                False,
            ),
            _p(
                "cheap_look_at_GTVPs",
                True,
                "bool",
                "Use the reference package's cheaper GTVP inspection.",
                False,
            ),
            _p(
                "prior_var",
                None,
                "list[float] | None",
                "Optional prior variances for local coefficients.",
                False,
            ),
            _p(
                "prior_mean",
                None,
                "list[float] | None",
                "Optional prior means for local coefficients.",
                False,
            ),
            _p("subsampling_rate", 0.75, "float", "Subsample share used by each tree."),
            _p("rw_regul", 0.75, "float", "Random-walk shrinkage strength."),
            _p(
                "keep_forest",
                False,
                "bool",
                "Keep full reference forest object in memory.",
                False,
            ),
            _p(
                "block_size",
                12,
                "int",
                "Reference-package block size for time-series resampling.",
                False,
            ),
            _p(
                "fast_rw",
                True,
                "bool",
                "Use fast random-walk regularization path.",
                False,
            ),
            _p("ridge_lambda", 0.1, "float", "Ridge penalty for local linear fits."),
            _p(
                "HRW",
                0,
                "int",
                "Reference-package hierarchical random-walk option.",
                False,
            ),
            _p("resampling_opt", 2, "int", "Reference MRF resampling option."),
            _p(
                "parallelise",
                False,
                "bool",
                "Run the reference implementation in parallel.",
                False,
            ),
            _p("n_cores", 1, "int", "Reference implementation worker count.", False),
            _p("print_b", False, "bool", "Print tree progress.", False),
        ),
        spaces={
            "small": {
                "B": (10, 25),
                "minsize": (5, 10),
                "mtry_frac": (0.5, 1.0),
                "min_leaf_frac_of_x": (1.0,),
                "subsampling_rate": (0.75,),
                "rw_regul": (0.5, 0.75),
                "ridge_lambda": (0.1, 0.5),
                "resampling_opt": (2,),
            },
            # B (forest tree count) dominates per-fit cost, and model_selection=None
            # still runs this preset's random search regardless of default_params --
            # capped so no sampled combo reruns the expensive B=100 forest; "wide"
            # keeps the richer range for deep/explicit use.
            "standard": {
                "B": (10, 25),
                "minsize": (5, 10, 20),
                "mtry_frac": (1.0 / 3.0, 0.5, 1.0),
                "min_leaf_frac_of_x": (0.5, 1.0),
                "subsampling_rate": (0.5, 0.75),
                "rw_regul": (0.5, 0.75, 0.9),
                "ridge_lambda": (0.1, 0.5, 1.0),
                "resampling_opt": (1, 2),
            },
            "wide": {
                "B": (50, 100, 250),
                "minsize": (5, 10, 20, 40),
                "mtry_frac": (0.25, 1.0 / 3.0, 0.5, 0.75, 1.0),
                "min_leaf_frac_of_x": (0.25, 0.5, 1.0, 2.0),
                "subsampling_rate": (0.5, 0.63, 0.75, 0.9),
                "rw_regul": (0.25, 0.5, 0.75, 0.9),
                "ridge_lambda": (0.01, 0.1, 0.5, 1.0),
                "resampling_opt": (1, 2),
            },
        },
        method="random",
        backend="macroforecast.models._mrf_reference.MacroRandomForest",
        requires_extra="macro_random_forest",
        description="Adapter for the external MacroRandomForest package.",
    ),
    "garch11": _spec(
        "garch11",
        "volatility",
        garch11,
        default_params={
            "p": 1,
            "q": 1,
            "mean_model": "constant",
            "dist": "normal",
            "rescale": False,
        },
        parameters=(
            _p("p", 1, "int", "GARCH innovation lag order."),
            _p("q", 1, "int", "GARCH variance lag order."),
            _p("mean_model", "constant", "str", "Conditional mean model.", False),
            _p("dist", "normal", "str", "Innovation distribution."),
            _p("rescale", False, "bool", "arch package rescale option.", False),
        ),
        spaces={
            "small": {"p": (1,), "q": (1,), "dist": ("normal", "t")},
            "standard": {"p": (1, 2), "q": (1, 2), "dist": ("normal", "t")},
            "wide": {"p": (1, 2, 3), "q": (1, 2, 3), "dist": ("normal", "t", "skewt")},
        },
        input_kind="volatility",
        backend="arch.arch_model",
        requires_extra="arch",
        description="GARCH volatility model.",
    ),
    "egarch": _spec(
        "egarch",
        "volatility",
        egarch,
        default_params={
            "p": 1,
            "o": 1,
            "q": 1,
            "mean_model": "constant",
            "dist": "normal",
            "rescale": False,
        },
        parameters=(
            _p("p", 1, "int", "EGARCH innovation lag order."),
            _p("o", 1, "int", "Asymmetric innovation lag order (leverage term)."),
            _p("q", 1, "int", "EGARCH variance lag order."),
            _p("mean_model", "constant", "str", "Conditional mean model.", False),
            _p("dist", "normal", "str", "Innovation distribution."),
            _p("rescale", False, "bool", "arch package rescale option.", False),
        ),
        spaces={
            "small": {"p": (1,), "o": (0, 1), "q": (1,), "dist": ("normal", "t")},
            "standard": {
                "p": (1, 2),
                "o": (0, 1),
                "q": (1, 2),
                "dist": ("normal", "t"),
            },
            "wide": {
                "p": (1, 2, 3),
                "o": (0, 1, 2),
                "q": (1, 2, 3),
                "dist": ("normal", "t", "skewt"),
            },
        },
        input_kind="volatility",
        backend="arch.arch_model",
        requires_extra="arch",
        description="EGARCH volatility model.",
    ),
    "gjr_garch": _spec(
        "gjr_garch",
        "volatility",
        gjr_garch,
        default_params={
            "p": 1,
            "o": 1,
            "q": 1,
            "mean_model": "constant",
            "dist": "normal",
            "rescale": False,
        },
        parameters=(
            _p("p", 1, "int", "GARCH innovation lag order."),
            _p("o", 1, "int", "Asymmetric (leverage) lag order."),
            _p("q", 1, "int", "GARCH variance lag order."),
            _p("mean_model", "constant", "str", "Conditional mean model.", False),
            _p("dist", "normal", "str", "Innovation distribution."),
            _p("rescale", False, "bool", "arch package rescale option.", False),
        ),
        spaces={
            "small": {"p": (1,), "o": (1,), "q": (1,), "dist": ("normal", "t")},
            "standard": {"p": (1, 2), "o": (1, 2), "q": (1, 2), "dist": ("normal", "t")},
            "wide": {"p": (1, 2, 3), "o": (1, 2), "q": (1, 2, 3), "dist": ("normal", "t", "skewt")},
        },
        input_kind="volatility",
        backend="arch.arch_model",
        requires_extra="arch",
        description="GJR-GARCH asymmetric volatility model.",
    ),
    "tgarch": _spec(
        "tgarch",
        "volatility",
        tgarch,
        default_params={
            "p": 1,
            "o": 1,
            "q": 1,
            "mean_model": "constant",
            "dist": "normal",
            "rescale": False,
        },
        parameters=(
            _p("p", 1, "int", "GARCH innovation lag order."),
            _p("o", 1, "int", "Asymmetric (leverage) lag order."),
            _p("q", 1, "int", "GARCH variance lag order."),
            _p("mean_model", "constant", "str", "Conditional mean model.", False),
            _p("dist", "normal", "str", "Innovation distribution."),
            _p("rescale", False, "bool", "arch package rescale option.", False),
        ),
        spaces={
            "small": {"p": (1,), "o": (1,), "q": (1,), "dist": ("normal", "t")},
            "standard": {"p": (1, 2), "o": (1, 2), "q": (1, 2), "dist": ("normal", "t")},
            "wide": {"p": (1, 2, 3), "o": (1, 2), "q": (1, 2, 3), "dist": ("normal", "t", "skewt")},
        },
        input_kind="volatility",
        backend="arch.arch_model",
        requires_extra="arch",
        description="Threshold GARCH (TGARCH/Zakoian) volatility model.",
    ),
    "realized_garch": _spec(
        "realized_garch",
        "volatility",
        realized_garch,
        default_params={
            "realized_variance": None,
            "max_iter": 2000,
            "n_starts": 5,
            "random_state": 0,
        },
        parameters=(
            _p(
                "realized_variance",
                None,
                "str | None",
                "Column name for realized variance; if omitted, an r_t^2 proxy is used.",
                False,
            ),
            _p("max_iter", 2000, "int", "Optimizer iteration cap.", False),
            _p("n_starts", 5, "int", "Number of optimizer starting points."),
            _p("random_state", 0, "int", "Optimizer random seed.", False),
        ),
        spaces={
            "small": {"n_starts": (3, 5)},
            "standard": {"n_starts": (3, 5, 10)},
            "wide": {"n_starts": (3, 5, 10, 20)},
        },
        input_kind="volatility",
        backend="internal rugarch-realGARCH-style p=q=1 log-linear MLE",
        description="Compact realized GARCH volatility model.",
    ),
}

_MODEL_SPECS_BY_CALLABLE = {spec.fit_func: spec for spec in MODEL_SPECS.values()}


__all__ = [
    "MODEL_SPECS",
    "InputKind",
    "ModelParameter",
    "ModelSpec",
    "custom_model",
    "describe_model",
    "get_model",
    "list_model_specs",
    "model_search_space",
]
