"""Model-run resolution for the forecasting runner (Phase 4 of the runner
decomposition; bodies moved verbatim from
``macroforecast.forecasting.runner``): the ``_ModelRun`` unit, single-model
resolution against the model/ensemble registries, and the preset/params
mapping helpers and validators.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from macroforecast.model_ensemble import get_model_ensemble
from macroforecast.models import ModelSpec, get_model


@dataclass(frozen=True)
class _ModelRun:
    alias: str
    spec: ModelSpec


def _resolve_model_runs(
    model: str | Callable[..., Any] | ModelSpec,
    *,
    preset: str | Mapping[str, str | None] | None,
    params: Mapping[str, Any] | None,
) -> list[_ModelRun]:
    # ``run`` is ATOMIC: exactly one model per call. A mapping or a (non-str)
    # sequence used to fit several models in one run; that is now rejected at the
    # public boundary. The internal ``model_runs`` list is kept as a one-element
    # list so the downstream per-model loops iterate exactly once without churn.
    _reject_multi_model(model)
    single_model = cast(str | Callable[..., Any] | ModelSpec, model)
    base = _get_model_or_ensemble(single_model)
    spec = _get_model_or_ensemble(
        single_model,
        preset=_preset_for_model(preset, None, base.name),
        params=_params_for_model(params, None, base.name, model_spec=base),
    )
    runs = [_ModelRun(alias=spec.name, spec=spec)]
    _validate_preset_mapping(preset, runs)
    _validate_params_mapping(params, runs)
    return runs


def _reject_multi_model(model: Any) -> None:
    """Raise ``TypeError`` if ``model`` is a multi-model sequence or mapping.

    A ``ModelSpec`` is a mapping-like dataclass but is a SINGLE model, and a
    string is a single model name, so both are allowed. Anything else that is a
    ``Mapping`` or a non-string ``Sequence`` is a multi-model request and is no
    longer supported by the atomic ``run``.
    """
    if isinstance(model, ModelSpec):
        return
    if isinstance(model, Mapping) or _is_model_sequence(model):
        raise TypeError(
            "forecasting.run fits exactly ONE model per call; got a "
            f"{type(model).__name__} of models. Run one model per call "
            "(loop over single-model run() calls), or use the pipeline with one "
            "Arm per model to compare models in a single managed run."
        )


def _get_model_or_ensemble(
    model: str | Callable[..., Any] | ModelSpec,
    *,
    preset: str | None = None,
    params: Mapping[str, Any] | None = None,
) -> ModelSpec:
    try:
        return get_model(model, preset=preset, params=params)
    except ValueError as model_error:
        try:
            return get_model_ensemble(model, preset=preset, params=params)
        except ValueError:
            raise model_error from None


def _is_model_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    )


def _preset_for_model(
    preset: str | Mapping[str, str | None] | None,
    alias: str | None,
    model_name: str | None,
) -> str | None:
    if preset is None or isinstance(preset, str):
        return preset
    if alias is not None and alias in preset:
        return preset[alias]
    if model_name is not None and model_name in preset:
        return preset[model_name]
    return None


def _params_for_model(
    params: Mapping[str, Any] | None,
    alias: str | None,
    model_name: str | None,
    *,
    model_spec: ModelSpec | None = None,
) -> Mapping[str, Any] | None:
    if params is None:
        return None
    if alias is not None and isinstance(params.get(alias), Mapping):
        return params[alias]
    if model_name is not None and isinstance(params.get(model_name), Mapping):
        return params[model_name]
    if all(isinstance(value, Mapping) for value in params.values()):
        if model_spec is not None and set(params).issubset(
            _known_model_param_names(model_spec)
        ):
            return params
        return None
    return params


def _actual_model_params(model_spec: ModelSpec, params: Mapping[str, Any]) -> dict[str, Any]:
    return {**dict(model_spec.params), **dict(params)}


def _known_model_param_names(model_spec: ModelSpec) -> set[str]:
    return {
        *model_spec.default_params,
        *model_spec.params,
        *(parameter.name for parameter in model_spec.parameters),
    }


def _run_keys(model_runs: Sequence[_ModelRun]) -> set[str]:
    return {
        key
        for model_run in model_runs
        for key in (model_run.alias, model_run.spec.name)
    }


def _validate_preset_mapping(
    preset: str | Mapping[str, str | None] | None,
    model_runs: Sequence[_ModelRun],
) -> None:
    if preset is None or isinstance(preset, str):
        return
    unknown = set(preset) - _run_keys(model_runs)
    if unknown:
        allowed = ", ".join(sorted(_run_keys(model_runs)))
        raise ValueError(
            f"preset contains keys that do not match a model alias or spec: "
            f"{sorted(unknown)}. Available keys: {allowed}."
        )


def _validate_params_mapping(
    params: Mapping[str, Any] | None,
    model_runs: Sequence[_ModelRun],
) -> None:
    if params is None or not all(isinstance(value, Mapping) for value in params.values()):
        return
    keys = set(params)
    run_keys = _run_keys(model_runs)
    if keys & run_keys:
        unknown = keys - run_keys
        if unknown:
            allowed = ", ".join(sorted(run_keys))
            raise ValueError(
                f"params contains keys that do not match a model alias or spec: "
                f"{sorted(unknown)}. Available keys: {allowed}."
            )
        return
    if all(keys.issubset(_known_model_param_names(model_run.spec)) for model_run in model_runs):
        return
    allowed = ", ".join(sorted(run_keys))
    raise ValueError(
        "params looks model-keyed but no key matches a model alias or spec. "
        f"Use one of: {allowed}; or pass direct parameter names accepted by every model."
    )
