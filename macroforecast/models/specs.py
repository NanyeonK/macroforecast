from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import pandas as pd

from macroforecast.models.linear import (
    adaptive_elastic_net,
    adaptive_lasso,
    bayesian_ridge,
    elastic_net,
    fused_difference_ridge,
    glmboost,
    group_lasso,
    huber,
    kernel_ridge,
    knn,
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
from macroforecast.models.neural import gru, hemisphere_nn, lstm, nn, transformer
from macroforecast.models.spline import mars
from macroforecast.models.svm import linear_svr, nu_svr, svr
from macroforecast.models.timeseries import (
    ar,
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
    theta_method,
    unrestricted_midas,
    var,
)
from macroforecast.models.tree import (
    bagging,
    booging,
    catboost,
    decision_tree,
    extra_trees,
    gradient_boosting,
    lightgbm,
    macro_random_forest,
    quantile_regression_forest,
    random_forest,
    slow_growing_tree,
    xgboost,
)
from macroforecast.models.volatility import egarch, garch11, realized_garch

InputKind = Literal["supervised", "target", "panel", "volatility"]
SearchSpace = dict[str, tuple[Any, ...]]
SearchSpaces = dict[str, SearchSpace]


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
            raise ValueError(f"Unknown model {model!r}. Available models: {allowed}.")
        spec = MODEL_SPECS[key]
    elif callable(model):
        callable_spec = _MODEL_SPECS_BY_CALLABLE.get(model)
        if callable_spec is None:
            name = getattr(model, "__name__", repr(model))
            raise ValueError(f"No registered ModelSpec for callable {name!r}")
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
    default_preset: str = "standard",
    input_kind: InputKind = "supervised",
    backend: str = "custom",
    requires_extra: str | None = None,
    requires_scaling: bool = False,
    recommended_preprocessing: tuple[str, ...] = (),
    description: str | None = None,
) -> ModelSpec:
    """Build a user-owned ``ModelSpec`` without registering a package model."""

    if not name:
        raise ValueError("custom model name must be non-empty")
    if not callable(fit_func):
        raise TypeError("custom model fit_func must be callable")
    return ModelSpec(
        name=str(name),
        family=str(family),
        fit_func=fit_func,
        default_params=dict(default_params or {}),
        parameters=parameters,
        search_spaces=dict(search_spaces or {}),
        default_search_method=str(default_search_method),
        default_preset=str(default_preset),
        preset=str(default_preset),
        input_kind=input_kind,
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
) -> ModelSpec:
    return ModelSpec(
        name=name,
        family=family,
        fit_func=fit_func,
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
        "neurons": (32, 64),
        "n_estimators": (5, 10),
        "learning_rate": (0.0005, 0.001),
    },
    "wide": {
        "neurons": (32, 64, 128),
        "n_estimators": (5, 10, 25),
        "learning_rate": (0.0001, 0.0005, 0.001),
    },
}

MODEL_SPECS: dict[str, ModelSpec] = {
    "ols": _spec(
        "ols",
        "linear",
        ols,
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
        description="Ridge regression.",
    ),
    "nonneg_ridge": _spec(
        "nonneg_ridge",
        "linear",
        nonneg_ridge,
        default_params={"alpha": 1.0, "fit_intercept": True},
        parameters=(
            _p("alpha", 1.0, "float", "L2 penalty strength."),
            _p("fit_intercept", True, "bool", "Fit an intercept outside the constrained coefficients.", False),
        ),
        spaces=_ALPHA_SPACES,
        method="cv_path",
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
            _p("simplex", False, "bool", "Constrain coefficients to sum to one.", False),
            _p("nonneg", False, "bool", "Constrain coefficients to be non-negative.", False),
            _p("fit_intercept", True, "bool", "Fit an intercept unless simplex=True.", False),
            _p("max_iter", 1000, "int", "SLSQP solver iteration cap.", False),
            _p("tol", 1e-9, "float", "SLSQP solver tolerance.", False),
        ),
        spaces=_ALPHA_SPACES,
        method="cv_path",
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
            _p("alpha", 1.0, "float", "Strength of the adjacent-coefficient smoothness penalty."),
            _p("difference_order", 1, "int", "Finite-difference order applied to coefficients.", False),
            _p("mean_equality", False, "bool", "Constrain fitted and observed sums to match.", False),
            _p("nonneg", False, "bool", "Constrain coefficients to be non-negative.", False),
            _p("fit_intercept", True, "bool", "Fit an intercept unless mean_equality=True.", False),
            _p("max_iter", 1000, "int", "SLSQP solver iteration cap.", False),
            _p("tol", 1e-9, "float", "SLSQP solver tolerance.", False),
        ),
        spaces=_ALPHA_SPACES,
        method="cv_path",
        description="Ridge regression with a fused-difference coefficient prior.",
    ),
    "random_walk_ridge": _spec(
        "random_walk_ridge",
        "linear",
        random_walk_ridge,
        default_params={"alpha": 1.0, "initial_alpha": 1.0, "fit_intercept": True},
        parameters=(
            _p("alpha", 1.0, "float", "Penalty on changes in adjacent coefficient vectors."),
            _p("initial_alpha", 1.0, "float", "Penalty on the first coefficient vector.", False),
            _p("fit_intercept", True, "bool", "Fit an intercept outside the time-varying coefficients.", False),
        ),
        spaces=_ALPHA_SPACES,
        method="cv_path",
        description="Time-varying random-walk ridge fit, predicting with the final coefficient vector.",
    ),
    "lasso": _spec(
        "lasso",
        "linear",
        lasso,
        default_params={"alpha": 1.0, "max_iter": 20000},
        parameters=(
            _p("alpha", 1.0, "float", "L1 penalty strength."),
            _p("max_iter", 20000, "int", "Optimization iteration cap.", False),
        ),
        spaces=_ALPHA_SPACES,
        method="cv_path",
        description="Lasso regression.",
    ),
    "elastic_net": _spec(
        "elastic_net",
        "linear",
        elastic_net,
        default_params={"alpha": 1.0, "l1_ratio": 0.5, "max_iter": 20000},
        parameters=(
            _p("alpha", 1.0, "float", "Overall penalty strength."),
            _p("l1_ratio", 0.5, "float", "L1 share of the elastic-net penalty."),
            _p("max_iter", 20000, "int", "Optimization iteration cap.", False),
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
            "max_iter": 20000,
            "tol": 1e-4,
            "random_state": None,
        },
        parameters=(
            _p("alpha", 1.0, "float", "Final adaptive lasso penalty strength."),
            _p("gamma", 1.0, "float", "Exponent applied to initial coefficient weights."),
            _p("initial", "ridge", "str", "Initial model: 'ridge' or 'ols'.", False),
            _p("initial_alpha", 1.0, "float", "Initial ridge penalty.", False),
            _p("eps", 1e-4, "float", "Small denominator floor for adaptive weights.", False),
            _p("max_iter", 20000, "int", "Final solver iteration cap.", False),
            _p("tol", 1e-4, "float", "Final solver convergence tolerance.", False),
            _p("random_state", 0, "int | None", "Final solver random seed.", False),
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
            "max_iter": 20000,
            "tol": 1e-4,
            "random_state": None,
        },
        parameters=(
            _p("alpha", 1.0, "float", "Final adaptive elastic-net penalty strength."),
            _p("l1_ratio", 0.5, "float", "L1 share of the final elastic-net penalty."),
            _p("gamma", 1.0, "float", "Exponent applied to initial coefficient weights."),
            _p("initial", "ridge", "str", "Initial model: 'ridge' or 'ols'.", False),
            _p("initial_alpha", 1.0, "float", "Initial ridge penalty.", False),
            _p("eps", 1e-4, "float", "Small denominator floor for adaptive weights.", False),
            _p("max_iter", 20000, "int", "Final solver iteration cap.", False),
            _p("tol", 1e-4, "float", "Final solver convergence tolerance.", False),
            _p("random_state", 0, "int | None", "Final solver random seed.", False),
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
            _p("groups", None, "sequence[str | int] | None", "One group label per predictor.", False),
            _p("alpha", 1.0, "float", "Group penalty strength."),
            _p("group_weights", None, "dict[str, float] | None", "Optional group penalty weights.", False),
            _p("max_iter", 5000, "int", "Proximal-gradient iteration cap.", False),
            _p("tol", 1e-5, "float", "Proximal-gradient convergence tolerance.", False),
            _p("scale", True, "bool", "Whether to standardize predictors inside the model.", False),
        ),
        spaces={
            "small": {"alpha": (0.01, 0.1, 1.0)},
            "standard": {"alpha": (0.001, 0.01, 0.1, 1.0, 10.0)},
            "wide": {"alpha": (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0)},
        },
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
            _p("groups", None, "sequence[str | int] | None", "One group label per predictor.", False),
            _p("alpha", 1.0, "float", "Total sparse-group penalty strength."),
            _p("l1_ratio", 0.5, "float", "Feature-level L1 share; remaining share is group penalty."),
            _p("group_weights", None, "dict[str, float] | None", "Optional group penalty weights.", False),
            _p("max_iter", 5000, "int", "Proximal-gradient iteration cap.", False),
            _p("tol", 1e-5, "float", "Proximal-gradient convergence tolerance.", False),
            _p("scale", True, "bool", "Whether to standardize predictors inside the model.", False),
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
        description="Package-native sparse group lasso with group and feature-level sparsity.",
    ),
    "bayesian_ridge": _spec(
        "bayesian_ridge",
        "linear",
        bayesian_ridge,
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
        description="Robust Huber regression.",
    ),
    "kernel_ridge": _spec(
        "kernel_ridge",
        "kernel",
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
            _p("kernel", "linear", "str", "Kernel name: linear, rbf, polynomial, sigmoid, etc."),
            _p("gamma", None, "float | None", "Kernel coefficient.", False),
            _p("degree", 3, "int", "Polynomial kernel degree.", False),
            _p("coef0", 1.0, "float", "Independent term for polynomial/sigmoid kernels.", False),
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
        default_params={"n_neighbors": 5, "weights": "uniform", "metric": "minkowski", "p": 2},
        parameters=(
            _p("n_neighbors", 5, "int", "Number of nearest neighbors."),
            _p("weights", "uniform", "str", "Neighbor weighting: uniform or distance."),
            _p("metric", "minkowski", "str", "Distance metric.", False),
            _p("p", 2, "int", "Minkowski distance order.", False),
        ),
        spaces={
            "small": {"n_neighbors": (3, 5, 10), "weights": ("uniform", "distance")},
            "standard": {"n_neighbors": (3, 5, 10, 20), "weights": ("uniform", "distance"), "p": (1, 2)},
            "wide": {"n_neighbors": (1, 3, 5, 10, 20, 40), "weights": ("uniform", "distance"), "p": (1, 2)},
        },
        method="random",
        backend="sklearn.neighbors.KNeighborsRegressor",
        requires_scaling=True,
        recommended_preprocessing=("standardize predictors before distance-based fitting",),
        description="K-nearest-neighbor regression.",
    ),
    "glmboost": _spec(
        "glmboost",
        "linear",
        glmboost,
        default_params={"n_iter": 100, "learning_rate": 0.1},
        parameters=(
            _p("n_iter", 100, "int", "Number of boosting iterations."),
            _p(
                "learning_rate",
                0.1,
                "float",
                "Shrinkage applied to each componentwise update.",
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
                "SVR kernel: linear, poly, rbf, sigmoid, or precomputed.",
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
            _p("random_state", 0, "int", "Random seed.", False),
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
                "NuSVR kernel: linear, poly, rbf, sigmoid, or precomputed.",
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
                "Activation: identity, logistic, tanh, or relu.",
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
        default_params={
            "lc": 2,
            "lm": 2,
            "lv": 2,
            "neurons": 64,
            "dropout": 0.2,
            "learning_rate": 0.001,
            "max_epochs": 100,
            "n_estimators": 100,
            "subsample": 0.8,
            "nu": None,
            "variance_penalty": 1.0,
            "patience": 15,
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
            _p("max_epochs", 100, "int", "Training epoch cap.", False),
            _p("n_estimators", 100, "int", "Number of blocked-subsample bags."),
            _p("subsample", 0.8, "float", "Blocked-subsample fraction.", False),
            _p("nu", None, "float | None", "Variance-emphasis target ratio.", False),
            _p("variance_penalty", 1.0, "float", "Soft penalty on the variance-emphasis target.", False),
            _p("patience", 15, "int", "Early-stopping patience.", False),
            _p("validation_fraction", 0.2, "float", "Chronological validation fraction.", False),
            _p("random_state", 0, "int", "Random seed.", False),
            _p("device", "auto", "str", "Torch device: auto, cpu, or cuda.", False),
            _p("quantile_levels", (0.05, 0.5, 0.95), "tuple[float, ...]", "Default normal-approximation density quantiles.", False),
            _p("lr", None, "float | None", "Legacy alias for learning_rate.", False),
            _p("n_epochs", None, "int | None", "Legacy alias for max_epochs.", False),
            _p("B", None, "int | None", "Legacy alias for n_estimators.", False),
            _p("sub_rate", None, "float | None", "Legacy alias for subsample.", False),
            _p("lambda_emphasis", None, "float | None", "Legacy alias for variance_penalty.", False),
            _p("val_frac", None, "float | None", "Legacy alias for validation_fraction.", False),
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
    "pls": _spec(
        "pls",
        "composite",
        pls,
        default_params={"n_components": 3, "scale": True, "max_iter": 500, "tol": 1e-6},
        parameters=(
            _p("n_components", 3, "int", "Number of latent PLS components."),
            _p("scale", True, "bool", "Whether to scale predictors inside PLS.", False),
            _p("max_iter", 500, "int", "NIPALS iteration cap.", False),
            _p("tol", 1e-6, "float", "NIPALS convergence tolerance.", False),
        ),
        spaces={
            "small": {"n_components": (1, 2, 3)},
            "standard": {"n_components": (1, 2, 3, 5, 8)},
            "wide": {"n_components": (1, 2, 3, 5, 8, 10, 12, 20)},
        },
        description="Partial least squares regression as a supervised dimension-reduction model.",
    ),
    "scaled_pca": _spec(
        "scaled_pca",
        "composite",
        scaled_pca,
        default_params={
            "n_components": 3,
            "scale": True,
            "control_columns": (),
            "include_constant": True,
            "drop_control_columns": True,
            "winsorize_slopes": None,
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
                (),
                "tuple[str, ...]",
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
            "control_columns": (),
            "include_constant": True,
            "drop_control_columns": True,
            "preselect": "none",
            "t_threshold": 1.28,
            "elastic_net_alpha": 0.0002,
            "elastic_net_l1_ratio": 0.5,
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
                (),
                "tuple[str, ...]",
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
            "control_columns": (),
            "include_constant": True,
            "drop_control_columns": True,
            "preselect": "none",
            "t_threshold": 1.28,
            "elastic_net_alpha": 0.0002,
            "elastic_net_l1_ratio": 0.5,
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
                (),
                "tuple[str, ...]",
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
        default_params={"n_lag": 1},
        parameters=(_p("n_lag", 1, "int", "Autoregressive lag order."),),
        spaces=_AR_SPACES,
        input_kind="target",
        description="Univariate autoregression.",
    ),
    "var": _spec(
        "var",
        "timeseries",
        var,
        default_params={"target": None, "n_lag": 1},
        parameters=(
            _p("target", None, "str | None", "Target column in the panel.", False),
            _p("n_lag", 1, "int", "VAR lag order."),
        ),
        spaces=_AR_SPACES,
        input_kind="panel",
        description="Vector autoregression.",
    ),
    "bvar_minnesota": _spec(
        "bvar_minnesota",
        "timeseries",
        bvar_minnesota,
        default_params={
            "target": None,
            "n_lag": 1,
            "shrinkage": 0.2,
            "intercept": True,
            "random_walk_prior": True,
        },
        parameters=(
            _p("target", None, "str | None", "Target column in the panel.", False),
            _p("n_lag", 1, "int", "VAR lag order."),
            _p("shrinkage", 0.2, "float", "Minnesota prior shrinkage scale."),
            _p("intercept", True, "bool", "Whether to include an intercept.", False),
            _p("random_walk_prior", True, "bool", "Prior mean one on own first lag.", False),
        ),
        spaces={
            "small": {"n_lag": (1, 2), "shrinkage": (0.1, 0.2, 0.5)},
            "standard": {"n_lag": (1, 2, 4), "shrinkage": (0.05, 0.1, 0.2, 0.5, 1.0)},
            "wide": {"n_lag": (1, 2, 4, 6, 12), "shrinkage": (0.01, 0.05, 0.1, 0.2, 0.5, 1.0)},
        },
        input_kind="panel",
        description="Compact Minnesota-prior Bayesian VAR point-forecast model.",
    ),
    "bvar_normal_inverse_wishart": _spec(
        "bvar_normal_inverse_wishart",
        "timeseries",
        bvar_normal_inverse_wishart,
        default_params={"target": None, "n_lag": 1, "shrinkage": 1.0, "intercept": True},
        parameters=(
            _p("target", None, "str | None", "Target column in the panel.", False),
            _p("n_lag", 1, "int", "VAR lag order."),
            _p("shrinkage", 1.0, "float", "Normal prior shrinkage scale."),
            _p("intercept", True, "bool", "Whether to include an intercept.", False),
        ),
        spaces={
            "small": {"n_lag": (1, 2), "shrinkage": (0.5, 1.0, 2.0)},
            "standard": {"n_lag": (1, 2, 4), "shrinkage": (0.2, 0.5, 1.0, 2.0, 5.0)},
            "wide": {"n_lag": (1, 2, 4, 6, 12), "shrinkage": (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)},
        },
        input_kind="panel",
        description="Compact normal-inverse-Wishart-style Bayesian VAR point-forecast model.",
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
        default_params={"trend": "add", "seasonal": None, "seasonal_periods": None, "damped_trend": False},
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
            "n_factors": 1,
            "factor_order": 1,
            "idiosyncratic_ar1": True,
            "standardize": True,
            "maxiter": 500,
            "tolerance": 1e-6,
        },
        parameters=(
            _p("target", None, "str | None", "Target column; defaults to first quarterly column.", False),
            _p("metadata", None, "Mapping[str, Any] | None", "Data metadata with native frequencies.", False),
            _p("monthly_columns", None, "Iterable[str] | None", "Explicit monthly columns.", False),
            _p("quarterly_columns", None, "Iterable[str] | None", "Explicit quarterly columns.", False),
            _p("unsupported", "raise", "str", "Unsupported-frequency policy: 'raise' or 'drop'.", False),
            _p("n_factors", 1, "int", "Number of dynamic factors."),
            _p("factor_order", 1, "int", "VAR order for factor dynamics."),
            _p("idiosyncratic_ar1", True, "bool", "Whether idiosyncratic disturbances are AR(1).", False),
            _p("standardize", True, "bool", "Whether statsmodels standardizes observed series.", False),
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
        default_params={"polynomial_order": 2, "theta": None, "alpha": 0.0, "fit_intercept": True},
        parameters=(
            _p("polynomial_order", 2, "int", "Almon polynomial order.", False),
            _p("theta", None, "tuple[float, ...] | None", "Almon polynomial coefficients; None gives equal weights.", False),
            _p("alpha", 0.0, "float", "Optional ridge penalty on aggregated MIDAS regressors."),
            _p("fit_intercept", True, "bool", "Whether to fit an intercept.", False),
        ),
        spaces={
            "small": {"alpha": (0.0, 0.1, 1.0)},
            "standard": {"alpha": (0.0, 0.01, 0.1, 1.0), "polynomial_order": (1, 2, 3)},
            "wide": {"alpha": (0.0, 0.001, 0.01, 0.1, 1.0, 10.0), "polynomial_order": (1, 2, 3, 4)},
        },
        description="MIDAS-style regression over lag groups using Almon weights.",
    ),
    "midas_beta": _spec(
        "midas_beta",
        "mixed_frequency",
        midas_beta,
        default_params={"beta_params": (1.0, 1.0), "alpha": 0.0, "fit_intercept": True},
        parameters=(
            _p("beta_params", (1.0, 1.0), "tuple[float, float]", "Beta lag-weight shape parameters."),
            _p("alpha", 0.0, "float", "Optional ridge penalty on aggregated MIDAS regressors."),
            _p("fit_intercept", True, "bool", "Whether to fit an intercept.", False),
        ),
        spaces={
            "small": {"alpha": (0.0, 0.1, 1.0), "beta_params": ((1.0, 1.0), (1.0, 2.0), (2.0, 1.0))},
            "standard": {
                "alpha": (0.0, 0.01, 0.1, 1.0),
                "beta_params": ((1.0, 1.0), (1.0, 2.0), (2.0, 1.0), (2.0, 2.0)),
            },
            "wide": {
                "alpha": (0.0, 0.001, 0.01, 0.1, 1.0, 10.0),
                "beta_params": ((0.5, 0.5), (1.0, 1.0), (1.0, 2.0), (2.0, 1.0), (2.0, 2.0), (3.0, 1.0)),
            },
        },
        description="MIDAS-style regression over lag groups using beta weights.",
    ),
    "midas_step": _spec(
        "midas_step",
        "mixed_frequency",
        midas_step,
        default_params={"n_steps": 3, "alpha": 0.0, "fit_intercept": True},
        parameters=(
            _p("n_steps", 3, "int", "Number of equal lag buckets."),
            _p("alpha", 0.0, "float", "Optional ridge penalty on aggregated MIDAS regressors."),
            _p("fit_intercept", True, "bool", "Whether to fit an intercept.", False),
        ),
        spaces={
            "small": {"n_steps": (2, 3), "alpha": (0.0, 0.1, 1.0)},
            "standard": {"n_steps": (2, 3, 4), "alpha": (0.0, 0.01, 0.1, 1.0)},
            "wide": {"n_steps": (2, 3, 4, 6), "alpha": (0.0, 0.001, 0.01, 0.1, 1.0, 10.0)},
        },
        description="MIDAS-style regression over lag groups using step-function weights.",
    ),
    "unrestricted_midas": _spec(
        "unrestricted_midas",
        "mixed_frequency",
        unrestricted_midas,
        default_params={"alpha": 0.0, "fit_intercept": True},
        parameters=(
            _p("alpha", 0.0, "float", "Optional ridge penalty on unrestricted lag coefficients."),
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
            "lag_columns": None,
            "lags": (0, 1, 2),
            "factor_lags": (0,),
            "target_frequency": "quarterly",
            "anchor_position": "period_end",
            "n_factors": 1,
            "factor_order": 1,
            "alpha": 0.0,
            "fit_intercept": True,
        },
        parameters=(
            _p("target", None, "str", "Target column.", False),
            _p("metadata", None, "Mapping[str, Any] | None", "Data metadata with native frequencies.", False),
            _p("lag_columns", None, "Iterable[str] | None", "Observed columns to add as unrestricted MIDAS lags.", False),
            _p("lags", (0, 1, 2), "Iterable[int] | int", "Observed-column native-frequency lags."),
            _p("factor_lags", (0,), "Iterable[int] | int", "DFM factor monthly lags."),
            _p("target_frequency", "quarterly", "str | None", "Frequency used to position target anchors.", False),
            _p("anchor_position", "period_end", "str", "Anchor date positioning.", False),
            _p("n_factors", 1, "int", "Number of DFM factors."),
            _p("factor_order", 1, "int", "VAR order for DFM factor dynamics."),
            _p("idiosyncratic_ar1", True, "bool", "Whether DFM idiosyncratic disturbances are AR(1).", False),
            _p("standardize", True, "bool", "Whether DynamicFactorMQ standardizes observed variables.", False),
            _p("maxiter", 500, "int", "DFM EM iteration cap.", False),
            _p("tolerance", 1e-6, "float", "DFM EM convergence tolerance.", False),
            _p("alpha", 0.0, "float", "Optional ridge penalty on unrestricted MIDAS head."),
            _p("fit_intercept", True, "bool", "Whether to fit an intercept.", False),
            _p("drop_missing", True, "bool", "Drop incomplete composite-design rows.", False),
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
        default_params={"n_factors": 3, "n_lag": 1, "random_state": 0},
        parameters=(
            _p("n_factors", 3, "int", "Number of PCA factors."),
            _p("n_lag", 1, "int", "Autoregressive lag order."),
            _p("random_state", 0, "int", "PCA random seed.", False),
        ),
        spaces={
            key: {**space, **_AR_SPACES[key]} for key, space in _FACTOR_SPACES.items()
        },
        description="Factor-augmented autoregression.",
    ),
    "favar": _spec(
        "favar",
        "factor",
        favar,
        default_params={"n_factors": 3, "n_lag": 1, "random_state": 0},
        parameters=(
            _p("n_factors", 3, "int", "Number of PCA factors."),
            _p("n_lag", 1, "int", "VAR lag order on target plus factors."),
            _p("random_state", 0, "int", "PCA random seed.", False),
        ),
        spaces={
            key: {**space, **_AR_SPACES[key]} for key, space in _FACTOR_SPACES.items()
        },
        description="Factor-augmented VAR.",
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
            "n_jobs": 1,
        },
        parameters=(
            _p("n_estimators", 200, "int", "Number of trees."),
            _p("max_depth", None, "int | None", "Maximum depth per tree."),
            _p("min_samples_leaf", 1, "int", "Minimum samples per terminal leaf."),
            _p("random_state", 0, "int", "Forest random seed.", False),
            _p("n_jobs", 1, "int | None", "Parallel worker count.", False),
        ),
        spaces=_FOREST_SPACES,
        method="random",
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
            "n_jobs": 1,
        },
        parameters=(
            _p("n_estimators", 200, "int", "Number of trees."),
            _p("max_depth", None, "int | None", "Maximum depth per tree."),
            _p("min_samples_leaf", 1, "int", "Minimum samples per terminal leaf."),
            _p("random_state", 0, "int", "Forest random seed.", False),
            _p("n_jobs", 1, "int | None", "Parallel worker count.", False),
        ),
        spaces=_FOREST_SPACES,
        method="random",
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
            _p("max_terms", 20, "int", "Maximum number of hinge basis terms including intercept."),
            _p("max_degree", 1, "int", "Maximum interaction degree among hinge factors."),
            _p("n_knots", 10, "int", "Candidate quantile knots per predictor."),
            _p("min_improvement", 1e-6, "float", "Forward-step relative RSS improvement floor.", False),
            _p("penalty", 2.0, "float", "GCV pruning complexity penalty.", False),
            _p("prune", True, "bool", "Whether to prune terms by GCV.", False),
        ),
        spaces={
            "small": {"max_terms": (8, 12), "max_degree": (1,), "n_knots": (5, 10)},
            "standard": {"max_terms": (10, 20, 30), "max_degree": (1, 2), "n_knots": (5, 10)},
            "wide": {"max_terms": (10, 20, 30, 50), "max_degree": (1, 2), "n_knots": (5, 10, 20)},
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
    "slow_growing_tree": _spec(
        "slow_growing_tree",
        "tree",
        slow_growing_tree,
        default_params={
            "eta": 0.1,
            "herfindahl_threshold": 0.25,
            "eta_depth_step": 0.01,
            "eta_max_plateau": 0.5,
            "mtry_frac": 1.0,
            "max_depth": 10,
            "random_state": 0,
            "min_leaf_size": 5,
        },
        parameters=(
            _p("eta", 0.1, "float", "Soft split leakage parameter."),
            _p(
                "herfindahl_threshold",
                0.25,
                "float",
                "Node concentration threshold for stopping.",
            ),
            _p(
                "eta_depth_step",
                0.01,
                "float",
                "Per-depth increase in soft split leakage.",
                False,
            ),
            _p(
                "eta_max_plateau",
                0.5,
                "float",
                "Upper plateau for depth-adjusted leakage.",
                False,
            ),
            _p(
                "mtry_frac",
                1.0,
                "float",
                "Fraction of candidate features considered at each split.",
            ),
            _p("max_depth", 10, "int | None", "Maximum tree depth."),
            _p("min_leaf_size", 5, "int", "Minimum effective leaf size."),
            _p("random_state", 0, "int", "Tree random seed.", False),
        ),
        spaces={
            "small": {
                "eta": (0.05, 0.1),
                "herfindahl_threshold": (0.2, 0.3),
                "mtry_frac": (0.75, 1.0),
                "max_depth": (5, 10),
                "min_leaf_size": (3, 5),
            },
            "standard": {
                "eta": (0.03, 0.05, 0.1),
                "herfindahl_threshold": (0.15, 0.25, 0.35),
                "mtry_frac": (0.5, 0.75, 1.0),
                "max_depth": (5, 10, None),
                "min_leaf_size": (3, 5, 10),
            },
            "wide": {
                "eta": (0.01, 0.03, 0.05, 0.1, 0.2),
                "herfindahl_threshold": (0.1, 0.15, 0.25, 0.35, 0.5),
                "mtry_frac": (0.33, 0.5, 0.75, 1.0),
                "max_depth": (3, 5, 10, 20, None),
                "min_leaf_size": (2, 3, 5, 10),
            },
        },
        method="random",
        description="Slow-growing tree with soft split propagation.",
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
        description="Quantile regression forest.",
    ),
    "bagging": _spec(
        "bagging",
        "ensemble",
        bagging,
        default_params={
            "base": "ridge",
            "n_estimators": 50,
            "max_samples": 0.8,
            "random_state": 0,
            "base_params": {},
            "strategy": "standard",
            "block_length": 4,
        },
        parameters=(
            _p("base", "ridge", "str", "Base estimator name."),
            _p("n_estimators", 50, "int", "Number of bootstrap models."),
            _p("max_samples", 0.8, "float", "Bootstrap sample fraction."),
            _p("base_params", {}, "dict", "Parameters passed to each base estimator.", False),
            _p("strategy", "standard", "str", "Bootstrap strategy: standard or block."),
            _p("block_length", 4, "int", "Block length when strategy='block'."),
            _p("random_state", 0, "int", "Ensemble random seed.", False),
        ),
        spaces={
            "small": {
                "base": ("ridge", "lasso"),
                "n_estimators": (10, 25),
                "max_samples": (0.6, 0.8),
                "strategy": ("standard",),
                "block_length": (4,),
            },
            "standard": {
                "base": ("ridge", "lasso", "decision_tree"),
                "n_estimators": (25, 50, 100),
                "max_samples": (0.5, 0.7, 0.9),
                "strategy": ("standard", "block"),
                "block_length": (4, 8),
            },
            "wide": {
                "base": (
                    "ridge",
                    "lasso",
                    "elastic_net",
                    "decision_tree",
                    "random_forest",
                ),
                "n_estimators": (25, 50, 100, 200),
                "max_samples": (0.4, 0.6, 0.8, 1.0),
                "strategy": ("standard", "block"),
                "block_length": (2, 4, 8, 12),
            },
        },
        method="random",
        description="Bootstrap aggregation ensemble.",
    ),
    "booging": _spec(
        "booging",
        "ensemble",
        booging,
        default_params={
            "B": 100,
            "sample_frac": 0.75,
            "inner_n_estimators": 1500,
            "inner_learning_rate": 0.1,
            "inner_max_depth": 3,
            "inner_subsample": 0.5,
            "da_noise_frac": 1.0 / 3.0,
            "da_drop_rate": 0.2,
            "random_state": 0,
        },
        parameters=(
            _p("B", 100, "int", "Number of overfit boosting models."),
            _p("sample_frac", 0.75, "float", "Row sample fraction per model."),
            _p("inner_n_estimators", 1500, "int", "Boosting stages inside each model."),
            _p("inner_learning_rate", 0.1, "float", "Inner boosting learning rate."),
            _p("inner_max_depth", 3, "int", "Inner boosting tree depth."),
            _p("inner_subsample", 0.5, "float", "Inner boosting subsample share."),
            _p(
                "da_noise_frac",
                1.0 / 3.0,
                "float",
                "Scale of feature-noise augmentation.",
            ),
            _p(
                "da_drop_rate",
                0.2,
                "float",
                "Share of augmented columns dropped per model.",
            ),
            _p("random_state", 0, "int", "Ensemble random seed.", False),
        ),
        spaces={
            "small": {
                "B": (5, 10),
                "sample_frac": (0.6, 0.8),
                "inner_n_estimators": (100, 300),
                "inner_learning_rate": (0.05, 0.1),
                "inner_max_depth": (2, 3),
                "inner_subsample": (0.5, 0.75),
                "da_noise_frac": (0.25, 1.0 / 3.0),
                "da_drop_rate": (0.1, 0.2),
            },
            "standard": {
                "B": (10, 25, 50),
                "sample_frac": (0.5, 0.75, 0.9),
                "inner_n_estimators": (300, 750, 1500),
                "inner_learning_rate": (0.03, 0.05, 0.1),
                "inner_max_depth": (2, 3, 5),
                "inner_subsample": (0.5, 0.75, 1.0),
                "da_noise_frac": (0.1, 0.25, 1.0 / 3.0),
                "da_drop_rate": (0.1, 0.2, 0.4),
            },
            "wide": {
                "B": (10, 25, 50, 100),
                "sample_frac": (0.4, 0.6, 0.75, 0.9),
                "inner_n_estimators": (300, 750, 1500, 2500),
                "inner_learning_rate": (0.01, 0.03, 0.05, 0.1),
                "inner_max_depth": (2, 3, 5, 8),
                "inner_subsample": (0.4, 0.5, 0.75, 1.0),
                "da_noise_frac": (0.05, 0.1, 0.25, 1.0 / 3.0, 0.5),
                "da_drop_rate": (0.0, 0.1, 0.2, 0.4, 0.6),
            },
        },
        method="random",
        description="Bagged overfit stochastic gradient boosting with augmentation.",
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
            "B": 50,
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
            _p("y_pos", 0, "int", "Reference-package target column position.", False),
            _p("B", 50, "int", "Number of MRF trees."),
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
            "standard": {
                "B": (25, 50, 100),
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
            "o": 0,
            "q": 1,
            "mean_model": "constant",
            "dist": "normal",
            "rescale": False,
        },
        parameters=(
            _p("p", 1, "int", "EGARCH innovation lag order."),
            _p("o", 0, "int", "Asymmetric innovation lag order."),
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
                "Column name for realized variance.",
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
        description="Realized GARCH volatility model.",
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
