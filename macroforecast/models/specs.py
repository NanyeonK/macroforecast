from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import pandas as pd

from macroforecast.models.linear import (
    bayesian_ridge,
    elastic_net,
    glmboost,
    huber,
    lasso,
    ols,
    pcr,
    ridge,
)
from macroforecast.models.timeseries import ar, far, favar, var
from macroforecast.models.tree import (
    bagging,
    booging,
    catboost,
    decision_tree,
    extra_trees,
    gradient_boosting,
    lightgbm,
    macro_random_forest,
    mars,
    quantile_regression_forest,
    random_forest,
    slow_growing_tree,
    xgboost,
)
from macroforecast.models.volatility import egarch, garch11, realized_garch

InputKind = Literal["supervised", "target", "panel", "volatility"]


@dataclass(frozen=True)
class ModelParameter:
    """One model-owned parameter description."""

    name: str
    default: Any
    kind: str
    description: str
    tunable: bool = True


@dataclass(frozen=True)
class ModelSpec:
    """Callable model plus model-owned defaults and hyperparameter spaces."""

    name: str
    family: str
    fit_func: Callable[..., Any]
    default_params: dict[str, Any] = field(default_factory=dict)
    parameters: tuple[ModelParameter, ...] = ()
    search_spaces: dict[str, dict[str, tuple[Any, ...]]] = field(default_factory=dict)
    default_search_method: str = "grid"
    default_preset: str = "standard"
    input_kind: InputKind = "supervised"
    preset: str = "standard"
    params: dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def with_preset(self, preset: str) -> "ModelSpec":
        """Return the same model spec with a different hyperparameter preset."""

        if self.search_spaces and preset not in self.search_spaces:
            allowed = ", ".join(sorted(self.search_spaces))
            raise ValueError(f"{self.name!r} has no preset {preset!r}. Available presets: {allowed}.")
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
            raise ValueError(f"{self.name!r} has no preset {key!r}. Available presets: {allowed}.")
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
        spec = _MODEL_SPECS_BY_CALLABLE.get(model)
        if spec is None:
            name = getattr(model, "__name__", repr(model))
            raise ValueError(f"No registered ModelSpec for callable {name!r}")
    else:
        raise TypeError("model must be a model name, callable, or ModelSpec")
    if preset is not None:
        spec = spec.with_preset(preset)
    if params:
        spec = spec.with_params(**dict(params))
    return spec


def list_model_specs(*, family: str | None = None) -> pd.DataFrame:
    """List registered model specs."""

    rows = []
    for spec in MODEL_SPECS.values():
        if family is not None and spec.family != family:
            continue
        rows.append({
            "name": spec.name,
            "family": spec.family,
            "input_kind": spec.input_kind,
            "default_search_method": spec.default_search_method,
            "default_preset": spec.default_preset,
            "presets": tuple(spec.search_spaces),
            "n_tunable": sum(parameter.tunable for parameter in spec.parameters),
            "description": spec.description,
        })
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


def _p(name: str, default: Any, kind: str, description: str, tunable: bool = True) -> ModelParameter:
    return ModelParameter(name, default, kind, description, tunable)


def _spec(
    name: str,
    family: str,
    fit_func: Callable[..., Any],
    *,
    default_params: dict[str, Any] | None = None,
    parameters: tuple[ModelParameter, ...] = (),
    spaces: dict[str, dict[str, tuple[Any, ...]]] | None = None,
    method: str = "grid",
    input_kind: InputKind = "supervised",
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
        description=description,
    )


_ALPHA_SPACES = {
    "small": {"alpha": (0.01, 0.1, 1.0)},
    "standard": {"alpha": (0.001, 0.01, 0.1, 1.0, 10.0)},
    "wide": {"alpha": (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0)},
}

_TREE_SPACES = {
    "small": {"max_depth": (3, 5, None), "min_samples_leaf": (1, 3)},
    "standard": {"max_depth": (3, 5, 10, None), "min_samples_leaf": (1, 3, 5)},
    "wide": {"max_depth": (2, 3, 5, 10, 20, None), "min_samples_leaf": (1, 2, 3, 5, 10)},
}

_FOREST_SPACES = {
    "small": {"n_estimators": (50, 100), "max_depth": (3, 5, None), "min_samples_leaf": (1, 3)},
    "standard": {"n_estimators": (100, 200, 500), "max_depth": (3, 5, 10, None), "min_samples_leaf": (1, 3, 5)},
    "wide": {"n_estimators": (100, 200, 500, 1000), "max_depth": (3, 5, 10, 20, None), "min_samples_leaf": (1, 2, 3, 5, 10)},
}

_BOOSTING_SPACES = {
    "small": {"n_estimators": (50, 100), "learning_rate": (0.05, 0.1), "max_depth": (2, 3)},
    "standard": {"n_estimators": (100, 200, 500), "learning_rate": (0.03, 0.05, 0.1), "max_depth": (2, 3, 5)},
    "wide": {"n_estimators": (100, 200, 500, 1000), "learning_rate": (0.01, 0.03, 0.05, 0.1), "max_depth": (2, 3, 5, 8)},
}

_FACTOR_SPACES = {
    "small": {"n_factors": (1, 2, 3)},
    "standard": {"n_factors": (1, 2, 3, 5, 8)},
    "wide": {"n_factors": (1, 2, 3, 5, 8, 10, 12)},
}

_AR_SPACES = {
    "small": {"n_lag": (1, 2, 4)},
    "standard": {"n_lag": (1, 2, 4, 6, 12)},
    "wide": {"n_lag": (1, 2, 3, 4, 6, 9, 12, 18, 24)},
}

MODEL_SPECS: dict[str, ModelSpec] = {
    "ols": _spec("ols", "linear", ols, description="Ordinary least squares with no model-owned tuning space."),
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
            "standard": {"alpha": (0.001, 0.01, 0.1, 1.0, 10.0), "l1_ratio": (0.1, 0.25, 0.5, 0.75, 0.9)},
            "wide": {"alpha": (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0), "l1_ratio": (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)},
        },
        description="Elastic net regression.",
    ),
    "bayesian_ridge": _spec("bayesian_ridge", "linear", bayesian_ridge, description="Empirical-Bayes Bayesian ridge."),
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
    "glmboost": _spec(
        "glmboost",
        "linear",
        glmboost,
        default_params={"n_iter": 100, "learning_rate": 0.1},
        parameters=(
            _p("n_iter", 100, "int", "Number of boosting iterations."),
            _p("learning_rate", 0.1, "float", "Shrinkage applied to each componentwise update."),
        ),
        spaces={
            "small": {"n_iter": (50, 100), "learning_rate": (0.05, 0.1)},
            "standard": {"n_iter": (50, 100, 200, 500), "learning_rate": (0.01, 0.05, 0.1)},
            "wide": {"n_iter": (50, 100, 200, 500, 1000), "learning_rate": (0.005, 0.01, 0.05, 0.1, 0.2)},
        },
        description="Componentwise linear boosting.",
    ),
    "pcr": _spec(
        "pcr",
        "factor",
        pcr,
        default_params={"n_components": 3, "random_state": 0},
        parameters=(
            _p("n_components", 3, "int", "Number of principal components."),
            _p("random_state", 0, "int", "PCA random seed.", False),
        ),
        spaces={
            "small": {"n_components": (1, 2, 3)},
            "standard": {"n_components": (1, 2, 3, 5, 8)},
            "wide": {"n_components": (1, 2, 3, 5, 8, 10, 12, 20)},
        },
        description="Principal component regression.",
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
        spaces={key: {**space, **_AR_SPACES[key]} for key, space in _FACTOR_SPACES.items()},
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
        spaces={key: {**space, **_AR_SPACES[key]} for key, space in _FACTOR_SPACES.items()},
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
        default_params={"n_estimators": 200, "max_depth": None, "min_samples_leaf": 1, "random_state": 0, "n_jobs": 1},
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
        default_params={"n_estimators": 200, "max_depth": None, "min_samples_leaf": 1, "random_state": 0, "n_jobs": 1},
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
        default_params={"n_estimators": 200, "learning_rate": 0.1, "max_depth": 3, "random_state": 0},
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
    "xgboost": _spec(
        "xgboost",
        "tree",
        xgboost,
        default_params={"n_estimators": 300, "learning_rate": 0.1, "max_depth": 6, "subsample": 1.0, "random_state": 0},
        parameters=(
            _p("n_estimators", 300, "int", "Number of boosting stages."),
            _p("learning_rate", 0.1, "float", "Shrinkage per stage."),
            _p("max_depth", 6, "int", "Maximum tree depth."),
            _p("subsample", 1.0, "float", "Row subsample share."),
            _p("random_state", 0, "int", "Boosting random seed.", False),
        ),
        spaces={key: {**space, "subsample": (0.6, 0.8, 1.0)} for key, space in _BOOSTING_SPACES.items()},
        method="random",
        description="XGBoost regressor.",
    ),
    "lightgbm": _spec(
        "lightgbm",
        "tree",
        lightgbm,
        default_params={"n_estimators": 300, "learning_rate": 0.1, "max_depth": -1, "num_leaves": 31, "random_state": 0},
        parameters=(
            _p("n_estimators", 300, "int", "Number of boosting stages."),
            _p("learning_rate", 0.1, "float", "Shrinkage per stage."),
            _p("max_depth", -1, "int", "Maximum tree depth; -1 means no limit."),
            _p("num_leaves", 31, "int", "Maximum leaves per tree."),
            _p("random_state", 0, "int", "Boosting random seed.", False),
        ),
        spaces={
            "small": {"n_estimators": (50, 100), "learning_rate": (0.05, 0.1), "max_depth": (-1, 3, 5), "num_leaves": (15, 31)},
            "standard": {"n_estimators": (100, 200, 500), "learning_rate": (0.03, 0.05, 0.1), "max_depth": (-1, 3, 5, 10), "num_leaves": (15, 31, 63)},
            "wide": {"n_estimators": (100, 200, 500, 1000), "learning_rate": (0.01, 0.03, 0.05, 0.1), "max_depth": (-1, 3, 5, 10, 20), "num_leaves": (15, 31, 63, 127)},
        },
        method="random",
        description="LightGBM regressor.",
    ),
    "catboost": _spec(
        "catboost",
        "tree",
        catboost,
        default_params={"n_estimators": 300, "learning_rate": 0.1, "max_depth": 6, "random_state": 0, "verbose": False},
        parameters=(
            _p("n_estimators", 300, "int", "Number of boosting stages."),
            _p("learning_rate", 0.1, "float", "Shrinkage per stage."),
            _p("max_depth", 6, "int", "Tree depth."),
            _p("random_state", 0, "int", "Boosting random seed.", False),
            _p("verbose", False, "bool", "CatBoost console output flag.", False),
        ),
        spaces=_BOOSTING_SPACES,
        method="random",
        description="CatBoost regressor.",
    ),
    "mars": _spec("mars", "spline", mars, description="Multivariate adaptive regression splines."),
    "slow_growing_tree": _spec(
        "slow_growing_tree",
        "tree",
        slow_growing_tree,
        default_params={"eta": 0.1, "herfindahl_threshold": 0.25, "eta_depth_step": 0.01, "eta_max_plateau": 0.5, "mtry_frac": 1.0, "max_depth": 10, "random_state": 0, "min_leaf_size": 5},
        parameters=(
            _p("eta", 0.1, "float", "Soft split leakage parameter."),
            _p("herfindahl_threshold", 0.25, "float", "Node concentration threshold for stopping."),
            _p("max_depth", 10, "int | None", "Maximum tree depth."),
            _p("min_leaf_size", 5, "int", "Minimum effective leaf size."),
            _p("random_state", 0, "int", "Tree random seed.", False),
        ),
        spaces={
            "small": {"eta": (0.05, 0.1), "herfindahl_threshold": (0.2, 0.3), "max_depth": (5, 10), "min_leaf_size": (3, 5)},
            "standard": {"eta": (0.03, 0.05, 0.1), "herfindahl_threshold": (0.15, 0.25, 0.35), "max_depth": (5, 10, None), "min_leaf_size": (3, 5, 10)},
            "wide": {"eta": (0.01, 0.03, 0.05, 0.1, 0.2), "herfindahl_threshold": (0.1, 0.15, 0.25, 0.35, 0.5), "max_depth": (3, 5, 10, 20, None), "min_leaf_size": (2, 3, 5, 10)},
        },
        method="random",
        description="Slow-growing tree with soft split propagation.",
    ),
    "quantile_regression_forest": _spec(
        "quantile_regression_forest",
        "tree",
        quantile_regression_forest,
        default_params={"n_estimators": 200, "max_depth": None, "min_samples_leaf": 1, "random_state": 0},
        parameters=(
            _p("n_estimators", 200, "int", "Number of trees."),
            _p("max_depth", None, "int | None", "Maximum depth per tree."),
            _p("min_samples_leaf", 1, "int", "Minimum samples per terminal leaf."),
            _p("random_state", 0, "int", "Forest random seed.", False),
        ),
        spaces=_FOREST_SPACES,
        method="random",
        description="Quantile regression forest.",
    ),
    "bagging": _spec(
        "bagging",
        "ensemble",
        bagging,
        default_params={"base": "ridge", "n_estimators": 50, "max_samples": 0.8, "random_state": 0, "strategy": "standard", "block_length": 4},
        parameters=(
            _p("base", "ridge", "str", "Base estimator name."),
            _p("n_estimators", 50, "int", "Number of bootstrap models."),
            _p("max_samples", 0.8, "float", "Bootstrap sample fraction."),
            _p("strategy", "standard", "str", "Bootstrap strategy: standard or block."),
            _p("block_length", 4, "int", "Block length when strategy='block'."),
            _p("random_state", 0, "int", "Ensemble random seed.", False),
        ),
        spaces={
            "small": {"base": ("ridge", "lasso"), "n_estimators": (10, 25), "max_samples": (0.6, 0.8)},
            "standard": {"base": ("ridge", "lasso", "decision_tree"), "n_estimators": (25, 50, 100), "max_samples": (0.5, 0.7, 0.9)},
            "wide": {"base": ("ridge", "lasso", "elastic_net", "decision_tree", "random_forest"), "n_estimators": (25, 50, 100, 200), "max_samples": (0.4, 0.6, 0.8, 1.0)},
        },
        method="random",
        description="Bootstrap aggregation ensemble.",
    ),
    "booging": _spec(
        "booging",
        "ensemble",
        booging,
        default_params={"B": 100, "sample_frac": 0.75, "inner_n_estimators": 1500, "inner_learning_rate": 0.1, "inner_max_depth": 3, "inner_subsample": 0.5, "da_noise_frac": 1.0 / 3.0, "da_drop_rate": 0.2, "random_state": 0},
        parameters=(
            _p("B", 100, "int", "Number of overfit boosting models."),
            _p("sample_frac", 0.75, "float", "Row sample fraction per model."),
            _p("inner_n_estimators", 1500, "int", "Boosting stages inside each model."),
            _p("inner_learning_rate", 0.1, "float", "Inner boosting learning rate."),
            _p("inner_max_depth", 3, "int", "Inner boosting tree depth."),
            _p("inner_subsample", 0.5, "float", "Inner boosting subsample share."),
            _p("random_state", 0, "int", "Ensemble random seed.", False),
        ),
        spaces={
            "small": {"B": (5, 10), "sample_frac": (0.6, 0.8), "inner_n_estimators": (100, 300), "inner_max_depth": (2, 3)},
            "standard": {"B": (10, 25, 50), "sample_frac": (0.5, 0.75, 0.9), "inner_n_estimators": (300, 750, 1500), "inner_max_depth": (2, 3, 5)},
            "wide": {"B": (10, 25, 50, 100), "sample_frac": (0.4, 0.6, 0.75, 0.9), "inner_n_estimators": (300, 750, 1500, 2500), "inner_max_depth": (2, 3, 5, 8)},
        },
        method="random",
        description="Bagged overfit stochastic gradient boosting with augmentation.",
    ),
    "macro_random_forest": _spec("macro_random_forest", "tree", macro_random_forest, description="Reserved adapter for the external Macroeconomic Random Forest backend."),
    "garch11": _spec(
        "garch11",
        "volatility",
        garch11,
        default_params={"p": 1, "q": 1, "mean_model": "constant", "dist": "normal", "rescale": False},
        parameters=(
            _p("p", 1, "int", "GARCH innovation lag order."),
            _p("q", 1, "int", "GARCH variance lag order."),
            _p("mean_model", "constant", "str", "Conditional mean model."),
            _p("dist", "normal", "str", "Innovation distribution."),
            _p("rescale", False, "bool", "arch package rescale option.", False),
        ),
        spaces={
            "small": {"p": (1,), "q": (1,), "dist": ("normal", "t")},
            "standard": {"p": (1, 2), "q": (1, 2), "dist": ("normal", "t")},
            "wide": {"p": (1, 2, 3), "q": (1, 2, 3), "dist": ("normal", "t", "skewt")},
        },
        input_kind="volatility",
        description="GARCH volatility model.",
    ),
    "egarch": _spec(
        "egarch",
        "volatility",
        egarch,
        default_params={"p": 1, "o": 0, "q": 1, "mean_model": "constant", "dist": "normal", "rescale": False},
        parameters=(
            _p("p", 1, "int", "EGARCH innovation lag order."),
            _p("o", 0, "int", "Asymmetric innovation lag order."),
            _p("q", 1, "int", "EGARCH variance lag order."),
            _p("mean_model", "constant", "str", "Conditional mean model."),
            _p("dist", "normal", "str", "Innovation distribution."),
            _p("rescale", False, "bool", "arch package rescale option.", False),
        ),
        spaces={
            "small": {"p": (1,), "o": (0, 1), "q": (1,), "dist": ("normal", "t")},
            "standard": {"p": (1, 2), "o": (0, 1), "q": (1, 2), "dist": ("normal", "t")},
            "wide": {"p": (1, 2, 3), "o": (0, 1, 2), "q": (1, 2, 3), "dist": ("normal", "t", "skewt")},
        },
        input_kind="volatility",
        description="EGARCH volatility model.",
    ),
    "realized_garch": _spec(
        "realized_garch",
        "volatility",
        realized_garch,
        default_params={"realized_variance": None, "max_iter": 2000, "n_starts": 5, "random_state": 0},
        parameters=(
            _p("realized_variance", None, "str | None", "Column name for realized variance.", False),
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
    "describe_model",
    "get_model",
    "list_model_specs",
    "model_search_space",
]
