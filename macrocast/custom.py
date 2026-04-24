"""Runtime registries for user-defined macrocast extensions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .preprocessing.feature_blocks import (
    FeatureBlockCallable,
    FeatureBlockCallableResult,
    validate_feature_block_callable_result,
)

CustomModelFunction = Callable[[Any, Any, Any, dict[str, Any]], Any]
CustomPreprocessorFunction = Callable[[Any, Any, Any, dict[str, Any]], Any]
TargetTransformerFactory = Callable[[], Any]
CustomFeatureBlockFunction = FeatureBlockCallable


@dataclass(frozen=True)
class CustomModelSpec:
    name: str
    function: CustomModelFunction
    description: str | None = None


@dataclass(frozen=True)
class CustomPreprocessorSpec:
    name: str
    function: CustomPreprocessorFunction
    description: str | None = None


@dataclass(frozen=True)
class TargetTransformerSpec:
    name: str
    factory: TargetTransformerFactory
    description: str | None = None
    model_scale: str = "transformed"
    forecast_scale: str = "raw"
    evaluation_scale: str = "raw"


@dataclass(frozen=True)
class CustomFeatureBlockSpec:
    name: str
    function: CustomFeatureBlockFunction
    block_kind: str
    description: str | None = None


_CUSTOM_MODELS: dict[str, CustomModelSpec] = {}
_CUSTOM_PREPROCESSORS: dict[str, CustomPreprocessorSpec] = {}
_TARGET_TRANSFORMERS: dict[str, TargetTransformerSpec] = {}
_CUSTOM_FEATURE_BLOCKS: dict[tuple[str, str], CustomFeatureBlockSpec] = {}


def _validate_name(name: str, *, kind: str) -> str:
    normalized = str(name).strip()
    if not normalized:
        raise ValueError(f"custom {kind} name must be non-empty")
    if normalized.startswith("_"):
        raise ValueError(f"custom {kind} name must not start with '_'")
    return normalized


def register_model(
    name: str,
    function: CustomModelFunction | None = None,
    *,
    description: str | None = None,
):
    """Register a user-defined model function.

    The function contract is:

    ``fn(X_train, y_train, X_test, context) -> prediction``

    ``X_test`` is currently one row. The return value may be a scalar or a
    one-element sequence/array. Registered names are accepted as
    ``model_family`` values by the compiler in the current Python process.
    """

    model_name = _validate_name(name, kind="model")

    def _decorator(fn: CustomModelFunction) -> CustomModelFunction:
        if not callable(fn):
            raise TypeError("custom model function must be callable")
        _CUSTOM_MODELS[model_name] = CustomModelSpec(
            name=model_name,
            function=fn,
            description=description,
        )
        return fn

    if function is not None:
        return _decorator(function)
    return _decorator


def custom_model(
    name: str,
    function: CustomModelFunction | None = None,
    *,
    description: str | None = None,
):
    """Decorator alias for :func:`register_model`."""

    return register_model(name, function=function, description=description)


def register_preprocessor(
    name: str,
    function: CustomPreprocessorFunction | None = None,
    *,
    description: str | None = None,
):
    """Register a user-defined matrix preprocessor.

    The function contract is:

    ``fn(X_train, y_train, X_test, context) -> (X_train, X_test)``

    ``y_train`` is provided as context for target-aware feature preprocessing,
    but the training target itself is not transformed in the MVP API.
    """

    preprocessor_name = _validate_name(name, kind="preprocessor")

    def _decorator(fn: CustomPreprocessorFunction) -> CustomPreprocessorFunction:
        if not callable(fn):
            raise TypeError("custom preprocessor function must be callable")
        _CUSTOM_PREPROCESSORS[preprocessor_name] = CustomPreprocessorSpec(
            name=preprocessor_name,
            function=fn,
            description=description,
        )
        return fn

    if function is not None:
        return _decorator(function)
    return _decorator


def custom_preprocessor(name: str, function: CustomPreprocessorFunction | None = None, **kwargs):
    """Decorator alias for :func:`register_preprocessor`."""
    return register_preprocessor(name, function=function, **kwargs)


def _target_transformer_factory(transformer: Any) -> TargetTransformerFactory:
    if isinstance(transformer, type):
        return transformer
    if callable(transformer):
        try:
            sample = transformer()
        except TypeError:
            return lambda: transformer
        missing = [
            method
            for method in ("fit", "transform", "inverse_transform_prediction")
            if not hasattr(sample, method)
        ]
        if not missing:
            return transformer
        return lambda: transformer
    return lambda: transformer


def _validate_target_transformer_instance(instance: Any, *, name: str) -> None:
    missing = [
        method
        for method in ("fit", "transform", "inverse_transform_prediction")
        if not callable(getattr(instance, method, None))
    ]
    if missing:
        raise TypeError(
            f"target transformer {name!r} must provide callable methods: {missing}"
        )


def register_target_transformer(
    name: str,
    transformer: Any | None = None,
    *,
    description: str | None = None,
):
    """Register a user-defined target transformer protocol.

    MVP registration records the contract but execution is intentionally not
    wired yet. The transformer instance must provide:

    ``fit(target_train, context)``
    ``transform(target, context)``
    ``inverse_transform_prediction(target_pred, context)``

    Target transformer execution must return final forecasts and metrics on
    the raw target scale.
    """

    transformer_name = _validate_name(name, kind="target transformer")

    def _decorator(obj: Any) -> Any:
        factory = _target_transformer_factory(obj)
        _validate_target_transformer_instance(factory(), name=transformer_name)
        _TARGET_TRANSFORMERS[transformer_name] = TargetTransformerSpec(
            name=transformer_name,
            factory=factory,
            description=description,
        )
        return obj

    if transformer is not None:
        return _decorator(transformer)
    return _decorator


def target_transformer(name: str, transformer: Any | None = None, **kwargs):
    """Decorator alias for :func:`register_target_transformer`."""
    return register_target_transformer(name, transformer=transformer, **kwargs)


def _validate_feature_block_kind(block_kind: str) -> str:
    kind = str(block_kind).strip()
    if kind not in {"temporal", "rotation", "factor"}:
        raise ValueError("custom feature block kind must be one of: temporal, rotation, factor")
    return kind


def register_feature_block(
    name: str,
    function: CustomFeatureBlockFunction | None = None,
    *,
    block_kind: str = "temporal",
    description: str | None = None,
):
    """Register a user-defined Layer 2 feature-block callable.

    The callable receives a :class:`FeatureBlockCallableContext` and must return
    a :class:`FeatureBlockCallableResult`. Runtime validates stable names and
    leakage metadata before appending the returned train/pred feature frames.
    """

    block_name = _validate_name(name, kind="feature block")
    kind = _validate_feature_block_kind(block_kind)

    def _decorator(fn: CustomFeatureBlockFunction) -> CustomFeatureBlockFunction:
        if not callable(fn):
            raise TypeError("custom feature block function must be callable")
        _CUSTOM_FEATURE_BLOCKS[(kind, block_name)] = CustomFeatureBlockSpec(
            name=block_name,
            function=fn,
            block_kind=kind,
            description=description,
        )
        return fn

    if function is not None:
        return _decorator(function)
    return _decorator


def custom_feature_block(name: str, function: CustomFeatureBlockFunction | None = None, **kwargs):
    """Decorator alias for :func:`register_feature_block`."""
    return register_feature_block(name, function=function, **kwargs)


def is_custom_feature_block(name: str, *, block_kind: str | None = None) -> bool:
    block_name = str(name)
    if block_kind is not None:
        return (_validate_feature_block_kind(block_kind), block_name) in _CUSTOM_FEATURE_BLOCKS
    return any(registered_name == block_name for _, registered_name in _CUSTOM_FEATURE_BLOCKS)


def get_custom_feature_block(name: str, *, block_kind: str) -> CustomFeatureBlockSpec:
    kind = _validate_feature_block_kind(block_kind)
    try:
        return _CUSTOM_FEATURE_BLOCKS[(kind, str(name))]
    except KeyError as exc:
        raise KeyError(f"custom {kind} feature block {name!r} is not registered") from exc


def list_custom_feature_blocks(*, block_kind: str | None = None) -> tuple[str, ...]:
    if block_kind is None:
        return tuple(sorted(name for _, name in _CUSTOM_FEATURE_BLOCKS))
    kind = _validate_feature_block_kind(block_kind)
    return tuple(sorted(name for (registered_kind, name) in _CUSTOM_FEATURE_BLOCKS if registered_kind == kind))


def validate_custom_feature_block_result(result: FeatureBlockCallableResult) -> None:
    validate_feature_block_callable_result(result)


def is_custom_model(name: str) -> bool:
    return str(name) in _CUSTOM_MODELS


def get_custom_model(name: str) -> CustomModelSpec:
    try:
        return _CUSTOM_MODELS[str(name)]
    except KeyError as exc:
        raise KeyError(f"custom model {name!r} is not registered") from exc


def list_custom_models() -> tuple[str, ...]:
    return tuple(sorted(_CUSTOM_MODELS))


def is_custom_preprocessor(name: str) -> bool:
    return str(name) in _CUSTOM_PREPROCESSORS


def get_custom_preprocessor(name: str) -> CustomPreprocessorSpec:
    try:
        return _CUSTOM_PREPROCESSORS[str(name)]
    except KeyError as exc:
        raise KeyError(f"custom preprocessor {name!r} is not registered") from exc


def list_custom_preprocessors() -> tuple[str, ...]:
    return tuple(sorted(_CUSTOM_PREPROCESSORS))


def is_custom_target_transformer(name: str) -> bool:
    return str(name) in _TARGET_TRANSFORMERS


def get_custom_target_transformer(name: str) -> TargetTransformerSpec:
    try:
        return _TARGET_TRANSFORMERS[str(name)]
    except KeyError as exc:
        raise KeyError(f"target transformer {name!r} is not registered") from exc


def list_custom_target_transformers() -> tuple[str, ...]:
    return tuple(sorted(_TARGET_TRANSFORMERS))


def clear_custom_models() -> None:
    """Clear custom model registry.

    Intended for tests and interactive cleanup.
    """

    _CUSTOM_MODELS.clear()


def clear_custom_preprocessors() -> None:
    """Clear custom preprocessor registry."""
    _CUSTOM_PREPROCESSORS.clear()


def clear_custom_target_transformers() -> None:
    """Clear custom target transformer registry."""
    _TARGET_TRANSFORMERS.clear()


def clear_custom_feature_blocks() -> None:
    """Clear custom feature block registry."""
    _CUSTOM_FEATURE_BLOCKS.clear()


def clear_custom_extensions() -> None:
    """Clear all runtime extension registries."""
    clear_custom_models()
    clear_custom_preprocessors()
    clear_custom_target_transformers()
    clear_custom_feature_blocks()
