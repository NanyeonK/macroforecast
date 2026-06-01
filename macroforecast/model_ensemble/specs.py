from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import pandas as pd

from macroforecast.model_ensemble.core import (
    bagging,
    booging,
    random_subspace,
    stacking,
    subagging,
    super_learner,
)
from macroforecast.models.specs import ModelParameter, ModelSpec

SearchSpace = dict[str, tuple[Any, ...]]
SearchSpaces = dict[str, SearchSpace]


def _p(
    name: str, default: Any, kind: str, description: str, tunable: bool = True
) -> ModelParameter:
    return ModelParameter(name, default, kind, description, tunable)


def _spec(
    name: str,
    fit_func: Callable[..., Any],
    *,
    default_params: dict[str, Any],
    parameters: tuple[ModelParameter, ...],
    spaces: SearchSpaces,
    method: str = "random",
    backend: str = "internal",
    description: str,
) -> ModelSpec:
    return ModelSpec(
        name=name,
        family="model_ensemble",
        fit_func=fit_func,
        default_params=default_params,
        parameters=parameters,
        search_spaces=spaces,
        default_search_method=method,
        input_kind="supervised",
        backend=backend,
        description=description,
    )


_BASE_SPACES = {
    "small": ("ridge", "lasso"),
    "standard": ("ridge", "lasso", "decision_tree"),
    "wide": ("ridge", "lasso", "elastic_net", "decision_tree", "random_forest"),
}

_BAGGING_SPACES: SearchSpaces = {
    "small": {
        "base": _BASE_SPACES["small"],
        "n_estimators": (10, 25),
        "max_samples": (0.6, 0.8),
        "strategy": ("standard",),
        "block_length": (4,),
        "replace": (True,),
    },
    "standard": {
        "base": _BASE_SPACES["standard"],
        "n_estimators": (25, 50, 100),
        "max_samples": (0.5, 0.7, 0.9),
        "strategy": ("standard", "block"),
        "block_length": (4, 8),
        "replace": (True,),
    },
    "wide": {
        "base": _BASE_SPACES["wide"],
        "n_estimators": (25, 50, 100, 200),
        "max_samples": (0.4, 0.6, 0.8, 1.0),
        "strategy": ("standard", "block"),
        "block_length": (2, 4, 8, 12),
        "replace": (True, False),
    },
}

_STACKING_SPACES: SearchSpaces = {
    "small": {
        "models": (("ridge", "lasso"), ("ridge", "decision_tree")),
        "meta_model": ("ridge",),
        "n_splits": (3,),
        "splitter": ("forward",),
        "passthrough": (False,),
    },
    "standard": {
        "models": (
            ("ridge", "lasso", "random_forest"),
            ("ridge", "lasso", "gradient_boosting"),
        ),
        "meta_model": ("ridge", "lasso"),
        "n_splits": (3, 5),
        "splitter": ("forward", "blocked"),
        "passthrough": (False, True),
    },
    "wide": {
        "models": (
            ("ridge", "lasso", "random_forest"),
            ("ridge", "elastic_net", "random_forest", "gradient_boosting"),
            ("ridge", "lasso", "knn", "svr"),
        ),
        "meta_model": ("ridge", "lasso", "elastic_net"),
        "n_splits": (3, 5, 8),
        "splitter": ("forward", "blocked", "kfold"),
        "passthrough": (False, True),
    },
}


MODEL_ENSEMBLE_SPECS: dict[str, ModelSpec] = {
    "bagging": _spec(
        "bagging",
        bagging,
        default_params={
            "base": "ridge",
            "n_estimators": 50,
            "max_samples": 0.8,
            "random_state": 0,
            "base_params": None,
            "strategy": "standard",
            "block_length": 4,
            "replace": True,
        },
        parameters=(
            _p("base", "ridge", "str", "Base estimator name."),
            _p("n_estimators", 50, "int", "Number of resampled member fits."),
            _p("max_samples", 0.8, "float", "Sample fraction per member."),
            _p("base_params", None, "dict | None", "Parameters for each base estimator.", False),
            _p("strategy", "standard", "str", "Resampling strategy: standard or block."),
            _p("block_length", 4, "int", "Block length when strategy='block'."),
            _p("replace", True, "bool", "Whether row sampling uses replacement."),
            _p("random_state", 0, "int", "Member resampling seed.", False),
        ),
        spaces=_BAGGING_SPACES,
        backend="internal member resampling + sklearn-compatible base estimators",
        description="Bootstrap or block-bootstrap fit-time model ensemble.",
    ),
    "subagging": _spec(
        "subagging",
        subagging,
        default_params={
            "base": "ridge",
            "n_estimators": 50,
            "max_samples": 0.632,
            "random_state": 0,
            "base_params": None,
        },
        parameters=(
            _p("base", "ridge", "str", "Base estimator name."),
            _p("n_estimators", 50, "int", "Number of subsampled member fits."),
            _p("max_samples", 0.632, "float", "Subsample fraction per member."),
            _p("base_params", None, "dict | None", "Parameters for each base estimator.", False),
            _p("random_state", 0, "int", "Member resampling seed.", False),
        ),
        spaces={
            key: {
                "base": values["base"],
                "n_estimators": values["n_estimators"],
                "max_samples": tuple(v for v in values["max_samples"] if v < 1.0),
            }
            for key, values in _BAGGING_SPACES.items()
        },
        backend="internal subagging + sklearn-compatible base estimators",
        description="Sampling-without-replacement fit-time model ensemble.",
    ),
    "random_subspace": _spec(
        "random_subspace",
        random_subspace,
        default_params={
            "base": "ridge",
            "n_estimators": 100,
            "max_features": 0.5,
            "max_samples": 1.0,
            "random_state": 0,
            "base_params": None,
        },
        parameters=(
            _p("base", "ridge", "str", "Base estimator name."),
            _p("n_estimators", 100, "int", "Number of random feature-subspace fits."),
            _p("max_features", 0.5, "float | int", "Feature subset size."),
            _p("max_samples", 1.0, "float", "Row subsample fraction per member."),
            _p("base_params", None, "dict | None", "Parameters for each base estimator.", False),
            _p("random_state", 0, "int", "Feature-subspace seed.", False),
        ),
        spaces={
            "small": {
                "base": _BASE_SPACES["small"],
                "n_estimators": (25, 50),
                "max_features": (0.4, 0.6),
                "max_samples": (0.8, 1.0),
            },
            "standard": {
                "base": _BASE_SPACES["standard"],
                "n_estimators": (50, 100),
                "max_features": (0.33, 0.5, 0.67),
                "max_samples": (0.6, 0.8, 1.0),
            },
            "wide": {
                "base": _BASE_SPACES["wide"],
                "n_estimators": (50, 100, 200),
                "max_features": (0.25, 0.33, 0.5, 0.67, 0.8),
                "max_samples": (0.5, 0.7, 0.9, 1.0),
            },
        },
        backend="internal random subspace + sklearn-compatible base estimators",
        description="Random feature-subspace fit-time model ensemble.",
    ),
    "stacking": _spec(
        "stacking",
        stacking,
        default_params={
            "models": ("ridge", "lasso", "random_forest"),
            "meta_model": "ridge",
            "n_splits": 5,
            "splitter": "forward",
            "random_state": 0,
            "model_params": None,
            "meta_params": None,
            "passthrough": False,
        },
        parameters=(
            _p("models", ("ridge", "lasso", "random_forest"), "tuple[str, ...]", "Base model library."),
            _p("meta_model", "ridge", "str", "Meta learner fit on OOF predictions."),
            _p("n_splits", 5, "int", "Number of OOF validation folds."),
            _p("splitter", "forward", "str", "OOF splitter: forward, blocked, or kfold."),
            _p("model_params", None, "dict | None", "Per-base model parameters.", False),
            _p("meta_params", None, "dict | None", "Meta-model parameters.", False),
            _p("passthrough", False, "bool", "Whether meta learner also receives original X."),
            _p("random_state", 0, "int", "Base/meta seed.", False),
        ),
        spaces=_STACKING_SPACES,
        backend="internal OOF stacking + sklearn-compatible base/meta estimators",
        description="Out-of-fold stacked fit-time model ensemble.",
    ),
    "super_learner": _spec(
        "super_learner",
        super_learner,
        default_params={
            "models": ("ridge", "lasso", "random_forest"),
            "n_splits": 5,
            "splitter": "forward",
            "weight_method": "nnls",
            "random_state": 0,
            "model_params": None,
        },
        parameters=(
            _p("models", ("ridge", "lasso", "random_forest"), "tuple[str, ...]", "Base learner library."),
            _p("n_splits", 5, "int", "Number of OOF validation folds."),
            _p("splitter", "forward", "str", "OOF splitter: forward, blocked, or kfold."),
            _p("weight_method", "nnls", "str", "Weight method: nnls, equal, or best."),
            _p("model_params", None, "dict | None", "Per-base learner parameters.", False),
            _p("random_state", 0, "int", "Base learner seed.", False),
        ),
        spaces={
            key: {
                "models": values["models"],
                "n_splits": values["n_splits"],
                "splitter": values["splitter"],
                "weight_method": ("nnls", "best", "equal"),
            }
            for key, values in _STACKING_SPACES.items()
        },
        backend="internal SuperLearner-style OOF NNLS/equal/best weighting",
        description="SuperLearner-style convex weighted fit-time model ensemble.",
    ),
    "booging": _spec(
        "booging",
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
            _p("B", 100, "int", "Number of overfit boosting members."),
            _p("sample_frac", 0.75, "float", "Row sample fraction per member."),
            _p("inner_n_estimators", 1500, "int", "Boosting stages inside each member."),
            _p("inner_learning_rate", 0.1, "float", "Inner boosting learning rate."),
            _p("inner_max_depth", 3, "int", "Inner boosting tree depth."),
            _p("inner_subsample", 0.5, "float", "Inner boosting subsample share."),
            _p("da_noise_frac", 1.0 / 3.0, "float", "Feature-noise augmentation scale."),
            _p("da_drop_rate", 0.2, "float", "Share of augmented columns dropped."),
            _p("random_state", 0, "int", "Member seed.", False),
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
        backend="internal augmentation/bagging + sklearn.ensemble.GradientBoostingRegressor",
        description="Bagged overfit stochastic gradient boosting with augmentation.",
    ),
}

_MODEL_ENSEMBLE_SPECS_BY_CALLABLE = {
    spec.fit_func: spec for spec in MODEL_ENSEMBLE_SPECS.values()
}


def get_model_ensemble(
    ensemble: str | Callable[..., Any] | ModelSpec,
    *,
    preset: str | None = None,
    params: Mapping[str, Any] | None = None,
) -> ModelSpec:
    """Return a fit-time model-ensemble spec by name, callable, or spec."""

    if isinstance(ensemble, ModelSpec):
        spec = ensemble
    elif isinstance(ensemble, str):
        key = ensemble.lower()
        if key not in MODEL_ENSEMBLE_SPECS:
            allowed = ", ".join(sorted(MODEL_ENSEMBLE_SPECS))
            raise ValueError(
                f"Unknown model ensemble {ensemble!r}. Available ensembles: {allowed}."
            )
        spec = MODEL_ENSEMBLE_SPECS[key]
    elif callable(ensemble):
        callable_spec = _MODEL_ENSEMBLE_SPECS_BY_CALLABLE.get(ensemble)
        if callable_spec is None:
            name = getattr(ensemble, "__name__", repr(ensemble))
            raise ValueError(f"No registered model-ensemble spec for callable {name!r}")
        spec = callable_spec
    else:
        raise TypeError("ensemble must be a name, callable, or ModelSpec")
    if preset is not None:
        spec = spec.with_preset(preset)
    if params:
        spec = spec.with_params(**dict(params))
    return spec


def custom_model_ensemble(
    name: str,
    fit_func: Callable[..., Any],
    *,
    default_params: Mapping[str, Any] | None = None,
    parameters: tuple[ModelParameter, ...] = (),
    search_spaces: SearchSpaces | None = None,
    default_search_method: str = "grid",
    default_preset: str = "standard",
    backend: str = "custom",
    description: str | None = None,
) -> ModelSpec:
    """Build a user-owned fit-time model-ensemble spec."""

    if not name:
        raise ValueError("custom model ensemble name must be non-empty")
    if not callable(fit_func):
        raise TypeError("custom model ensemble fit_func must be callable")
    return ModelSpec(
        name=str(name),
        family="model_ensemble",
        fit_func=fit_func,
        default_params=dict(default_params or {}),
        parameters=parameters,
        search_spaces=dict(search_spaces or {}),
        default_search_method=str(default_search_method),
        default_preset=str(default_preset),
        preset=str(default_preset),
        input_kind="supervised",
        backend=str(backend),
        description=description or f"User supplied model ensemble {fit_func}.",
    )


def list_model_ensemble_specs(*, family: str | None = None) -> pd.DataFrame:
    """List registered fit-time model-ensemble specs."""

    rows = []
    for spec in MODEL_ENSEMBLE_SPECS.values():
        if family is not None and spec.family != family:
            continue
        rows.append(
            {
                "name": spec.name,
                "family": spec.family,
                "input_kind": spec.input_kind,
                "backend": spec.backend,
                "default_search_method": spec.default_search_method,
                "default_preset": spec.default_preset,
                "presets": tuple(spec.search_spaces),
                "n_tunable": sum(parameter.tunable for parameter in spec.parameters),
                "description": spec.description,
            }
        )
    return pd.DataFrame(rows)


def describe_model_ensemble(ensemble: str | Callable[..., Any] | ModelSpec) -> pd.DataFrame:
    """Describe fit-time model-ensemble parameters and preset spaces."""

    return get_model_ensemble(ensemble).describe()


def model_ensemble_search_space(
    ensemble: str | Callable[..., Any] | ModelSpec,
    *,
    preset: str | None = None,
) -> dict[str, tuple[Any, ...]]:
    """Return a model-ensemble-owned hyperparameter space."""

    return get_model_ensemble(ensemble, preset=preset).search_space()


__all__ = [
    "MODEL_ENSEMBLE_SPECS",
    "custom_model_ensemble",
    "describe_model_ensemble",
    "get_model_ensemble",
    "list_model_ensemble_specs",
    "model_ensemble_search_space",
]
