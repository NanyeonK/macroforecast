from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from macroforecast.interpretation._anatomy_utils import anatomy_output_transform
from macroforecast.metrics import MetricLike
from macroforecast.models import ModelFit, ModelSpec, get_model
from macroforecast.window import WindowSpec, resolve_window


@dataclass(frozen=True)
class AnatomyPipelineResult:
    """End-to-end ``anatomy`` backend output."""

    anatomy: Any
    explanations: dict[str, pd.DataFrame] = field(default_factory=dict)
    variable_importance: pd.DataFrame | None = None
    performance_values: dict[str, pd.DataFrame] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    metadata_schema: dict[str, Any] = field(
        default_factory=lambda: {"kind": "anatomy_pipeline_result", "version": 1}
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata_schema": dict(self.metadata_schema),
            "metadata": self.metadata,
            "explanations": {
                key: value.to_dict(orient="records")
                for key, value in self.explanations.items()
            },
            "variable_importance": None
            if self.variable_importance is None
            else self.variable_importance.to_dict(orient="records"),
            "performance_values": {
                key: value.to_dict(orient="records")
                for key, value in self.performance_values.items()
            },
        }

    def to_tables(self, *, prefix: str = "anatomy") -> dict[str, pd.DataFrame]:
        """Return JSON/CSV-ready anatomy output tables."""

        safe_prefix = _safe_table_name(prefix)
        tables: dict[str, pd.DataFrame] = {}
        for name, table in self.explanations.items():
            key = f"{safe_prefix}_explanation_{_safe_table_name(name)}"
            tables[key] = _attach_anatomy_table_schema(
                table.copy(),
                kind="anatomy_explanation_table",
                metadata={"section": "explanations", "name": str(name)},
            )
        if self.variable_importance is not None:
            tables[f"{safe_prefix}_variable_importance"] = _attach_anatomy_table_schema(
                self.variable_importance.copy(),
                kind="anatomy_variable_importance_table",
                metadata={"section": "variable_importance"},
            )
        for name, table in self.performance_values.items():
            key = f"{safe_prefix}_performance_{_safe_table_name(name)}"
            tables[key] = _attach_anatomy_table_schema(
                table.copy(),
                kind="anatomy_performance_value_table",
                metadata={"section": "performance_values", "name": str(name)},
            )
        tables[f"{safe_prefix}_metadata"] = _attach_anatomy_table_schema(
            _metadata_frame(self.metadata),
            kind="anatomy_metadata_table",
            metadata={"section": "metadata"},
        )
        return tables


ForecastShapleyResult = AnatomyPipelineResult


def forecast_shapley_output(
    value: Any,
    *,
    output: str = "oshapley",
    loss: str = "rmse",
    sidecar_name: str | None = None,
    prefix: str = "oshapley",
    model_groups: Mapping[str, Sequence[str] | Mapping[str, float]] | None = None,
    explanation_subset: pd.Index | Sequence[Any] | None = None,
    table: str | None = None,
) -> Any:
    """Select one oShapley/PBSV output from a result, sidecar, or backend object."""

    from macroforecast.interpretation.core import anatomy_explain, oshapley_vi, pbsv

    selected = _resolve_forecast_shapley_source(value, sidecar_name=sidecar_name)
    key = _canonical_forecast_shapley_output(output)
    if isinstance(selected, AnatomyPipelineResult):
        if key == "forecast":
            return selected.explanations["forecast"].copy()
        if key == "oshapley":
            if selected.variable_importance is None:
                raise ValueError("selected ForecastShapleyResult has no oShapley-VI table")
            return selected.variable_importance.copy()
        if key == "pbsv":
            loss_key = str(loss)
            if loss_key not in selected.performance_values:
                available = ", ".join(sorted(selected.performance_values))
                raise KeyError(
                    f"PBSV loss {loss_key!r} not found. Available losses: {available}"
                )
            return selected.performance_values[loss_key].copy()
        if key == "metadata":
            return _attach_anatomy_table_schema(
                _metadata_frame(selected.metadata),
                kind="forecast_shapley_metadata_table",
                metadata={"section": "metadata"},
            )
        if key == "summary":
            return selected.to_dict()
        if key == "tables":
            tables = selected.to_tables(prefix=prefix)
            if table is not None:
                table_key = str(table)
                if table_key not in tables:
                    raise KeyError(
                        f"table {table_key!r} not found. Available tables: {sorted(tables)}"
                    )
                return tables[table_key].copy()
            return tables
    if key == "forecast":
        return anatomy_explain(
            selected,
            model_groups=model_groups,
            metric="forecast",
            explanation_subset=explanation_subset,
        )
    if key == "oshapley":
        return oshapley_vi(
            selected,
            model_groups=model_groups,
            explanation_subset=explanation_subset,
        )
    if key == "pbsv":
        return pbsv(
            selected,
            model_groups=model_groups,
            loss=loss,
            explanation_subset=explanation_subset,
        )
    if key in {"metadata", "summary", "tables"}:
        raise TypeError(
            f"output={output!r} requires a ForecastShapleyResult or ForecastResult sidecar"
        )
    raise ValueError(f"unsupported forecast Shapley output {output!r}")


def oshapley_output(value: Any, **kwargs: Any) -> Any:
    """Alias for :func:`forecast_shapley_output`."""

    return forecast_shapley_output(value, **kwargs)


def anatomy_model(model: Any, *, feature_names: Sequence[str] | None = None) -> Any:
    """Wrap a macroforecast fit or estimator as ``anatomy.AnatomyModel``."""

    anatomy_mod = _optional_anatomy()
    names = tuple(
        str(value) for value in (feature_names or _model_feature_names(model))
    )

    def _predict(xs: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame(np.asarray(xs, dtype=float), columns=list(names) or None)
        if isinstance(model, ModelFit):
            values = model.predict(frame)
        elif hasattr(model, "predict"):
            values = model.predict(frame)
        else:
            raise TypeError("model must be a ModelFit or expose predict(X)")
        return np.asarray(values, dtype=float).reshape(-1)

    return anatomy_mod.AnatomyModel(_predict)


def anatomy_output_transformer(
    output: MetricLike = "forecast",
) -> Any:
    """Return an ``anatomy`` output transformer for forecast/loss explanations."""

    anatomy_mod = _optional_anatomy()
    transform = anatomy_output_transform(output)
    return anatomy_mod.AnatomyModelOutputTransformer(transform=transform)


def window_to_anatomy_subsets(
    window: WindowSpec | str | None,
    index: pd.Index | Sequence[Any],
    *,
    train_source: str = "fit",
) -> Any:
    """Convert a macroforecast window into exact ``anatomy`` train/test subsets."""

    anatomy_mod = _optional_anatomy()
    spec = resolve_window(window)
    labels = _coerce_anatomy_index(index)
    source = str(train_source).lower().replace("-", "_")
    if source not in {"fit", "estimation"}:
        raise ValueError("train_source must be 'fit' or 'estimation'")
    subsets = []
    for item in spec.iter_origins(labels):
        train_idx = item["fit_idx"] if source == "fit" else item["estimation_idx"]
        test_idx = item["test_idx"]
        subsets.append(
            _anatomy_subset_class()(
                train_subset=_contiguous_slice(train_idx, name="train_idx"),
                test_subset=_contiguous_slice(test_idx, name="test_idx"),
            )
        )
    return anatomy_mod.AnatomySubsets(index=labels, subsets=subsets)


def window_to_oshapley_subsets(
    window: WindowSpec | str | None,
    index: pd.Index | Sequence[Any],
    *,
    train_source: str = "fit",
) -> Any:
    """Convert a macroforecast window for oShapley/PBSV backend precompute."""

    return window_to_anatomy_subsets(window, index, train_source=train_source)


def anatomy_provider(
    X: Any,
    y: Any,
    models: str
    | Callable[..., Any]
    | ModelSpec
    | Sequence[str | Callable[..., Any] | ModelSpec]
    | Mapping[str, str | Callable[..., Any] | ModelSpec],
    *,
    window: WindowSpec | str | None = None,
    params: Mapping[str, Any] | None = None,
    target_name: str | None = None,
    train_source: str = "fit",
) -> Any:
    """Build an ``AnatomyModelProvider`` from aligned macroforecast X/y data."""

    anatomy_mod = _optional_anatomy()
    X_aligned, y_aligned = _aligned_xy(X, y)
    feature_names = tuple(str(column) for column in X_aligned.columns)
    y_name = str(target_name or y_aligned.name or "target")
    xy = pd.concat([y_aligned.rename(y_name), X_aligned], axis=1)
    subsets = window_to_anatomy_subsets(
        window,
        xy.index,
        train_source=train_source,
    )
    specs = _resolve_model_mapping(models)
    params_by_model = _resolve_model_params(specs, params)

    def _mapper(key: Any) -> Any:
        model_name = str(key.model_name)
        spec = specs[model_name]
        train = xy.iloc[subsets.get_train_subset(int(key.period))].copy()
        test = xy.iloc[subsets.get_test_subset(int(key.period))].copy()
        fit = spec(
            train.loc[:, list(feature_names)],
            train[y_name],
            **params_by_model.get(model_name, {}),
        )
        return anatomy_mod.AnatomyModelProvider.PeriodValue(
            train,
            test,
            anatomy_model(fit, feature_names=feature_names),
        )

    return anatomy_mod.AnatomyModelProvider(
        n_periods=int(subsets.n_periods),
        n_features=len(feature_names),
        model_names=list(specs),
        y_name=y_name,
        provider_fn=_mapper,
    )


def oshapley_provider(
    X: Any,
    y: Any,
    models: str
    | Callable[..., Any]
    | ModelSpec
    | Sequence[str | Callable[..., Any] | ModelSpec]
    | Mapping[str, str | Callable[..., Any] | ModelSpec],
    **kwargs: Any,
) -> Any:
    """Build the provider used by oShapley-VI and PBSV precompute."""

    return anatomy_provider(X, y, models, **kwargs)


def precompute_anatomy(
    X: Any,
    y: Any,
    models: str
    | Callable[..., Any]
    | ModelSpec
    | Sequence[str | Callable[..., Any] | ModelSpec]
    | Mapping[str, str | Callable[..., Any] | ModelSpec],
    *,
    window: WindowSpec | str | None = None,
    params: Mapping[str, Any] | None = None,
    target_name: str | None = None,
    train_source: str = "fit",
    n_iterations: int = 32,
    n_jobs: int = 1,
    background_data_subsample: float = 1.0,
    save_path: str | Path | None = None,
) -> Any:
    """Build, precompute, and optionally save an ``anatomy.Anatomy`` object."""

    anatomy_mod = _optional_anatomy()
    provider = anatomy_provider(
        X,
        y,
        models,
        window=window,
        params=params,
        target_name=target_name,
        train_source=train_source,
    )
    return anatomy_mod.Anatomy(
        provider=provider, n_iterations=int(n_iterations)
    ).precompute(
        n_jobs=int(n_jobs),
        background_data_subsample=float(background_data_subsample),
        save_path=None if save_path is None else str(save_path),
    )


def precompute_oshapley(
    X: Any,
    y: Any,
    models: str
    | Callable[..., Any]
    | ModelSpec
    | Sequence[str | Callable[..., Any] | ModelSpec]
    | Mapping[str, str | Callable[..., Any] | ModelSpec],
    **kwargs: Any,
) -> Any:
    """Precompute the backend object used by oShapley-VI and PBSV."""

    return precompute_anatomy(X, y, models, **kwargs)


def anatomy_pipeline(
    X: Any,
    y: Any,
    models: str
    | Callable[..., Any]
    | ModelSpec
    | Sequence[str | Callable[..., Any] | ModelSpec]
    | Mapping[str, str | Callable[..., Any] | ModelSpec],
    *,
    window: WindowSpec | str | None = None,
    params: Mapping[str, Any] | None = None,
    model_groups: Mapping[str, Sequence[str] | Mapping[str, float]] | None = None,
    target_name: str | None = None,
    train_source: str = "fit",
    losses: Sequence[str] = ("rmse",),
    n_iterations: int = 32,
    n_jobs: int = 1,
    background_data_subsample: float = 1.0,
    save_path: str | Path | None = None,
) -> AnatomyPipelineResult:
    """Run the complete ``anatomy`` provider, precompute, and summary path."""

    from macroforecast.interpretation.core import (
        anatomy_explain,
        oshapley_vi,
        pbsv,
    )

    anatomy_obj = precompute_anatomy(
        X,
        y,
        models,
        window=window,
        params=params,
        target_name=target_name,
        train_source=train_source,
        n_iterations=n_iterations,
        n_jobs=n_jobs,
        background_data_subsample=background_data_subsample,
        save_path=save_path,
    )
    explanations = {
        "forecast": anatomy_explain(
            anatomy_obj,
            model_groups=model_groups,
            metric="forecast",
        )
    }
    variable_importance = oshapley_vi(anatomy_obj, model_groups=model_groups)
    performance_values = {
        str(loss): pbsv(anatomy_obj, model_groups=model_groups, loss=str(loss))
        for loss in losses
    }
    return AnatomyPipelineResult(
        anatomy=anatomy_obj,
        explanations=explanations,
        variable_importance=variable_importance,
        performance_values=performance_values,
        metadata={
            "kind": "anatomy_pipeline",
            "window": resolve_window(window).to_dict(),
            "models": list(_resolve_model_mapping(models)),
            "losses": [str(loss) for loss in losses],
            "train_source": str(train_source),
            "target_name": str(target_name or pd.Series(y).name or "target"),
            "n_iterations": int(n_iterations),
            "n_jobs": int(n_jobs),
            "save_path": None if save_path is None else str(save_path),
        },
    )


def oshapley_pipeline(
    X: Any,
    y: Any,
    models: str
    | Callable[..., Any]
    | ModelSpec
    | Sequence[str | Callable[..., Any] | ModelSpec]
    | Mapping[str, str | Callable[..., Any] | ModelSpec],
    **kwargs: Any,
) -> ForecastShapleyResult:
    """Run the oShapley-VI/PBSV forecast-accuracy interpretation pipeline."""

    return anatomy_pipeline(X, y, models, **kwargs)


def anatomy_from_forecast_result(
    result: Any,
    X: Any,
    y: Any,
    models: str
    | Callable[..., Any]
    | ModelSpec
    | Sequence[str | Callable[..., Any] | ModelSpec]
    | Mapping[str, str | Callable[..., Any] | ModelSpec],
    *,
    window: WindowSpec | str | None,
    attach: bool = True,
    sidecar_name: str = "anatomy",
    params: Mapping[str, Any] | None = None,
    model_groups: Mapping[str, Sequence[str] | Mapping[str, float]] | None = None,
    target_name: str | None = None,
    train_source: str = "fit",
    losses: Sequence[str] = ("rmse",),
    n_iterations: int = 32,
    n_jobs: int = 1,
    background_data_subsample: float = 1.0,
    save_path: str | Path | None = None,
) -> Any:
    """Build anatomy outputs from a forecast result plus explicit X/y inputs."""

    if window is None:
        raise ValueError(
            "window is required because ForecastResult tables do not store the "
            "feature matrix and origin-wise train/test subsets needed by anatomy"
        )
    pipeline = anatomy_pipeline(
        X,
        y,
        models,
        window=window,
        params=params,
        model_groups=model_groups,
        target_name=target_name,
        train_source=train_source,
        losses=losses,
        n_iterations=n_iterations,
        n_jobs=n_jobs,
        background_data_subsample=background_data_subsample,
        save_path=save_path,
    )
    pipeline.metadata["forecast_result"] = {
        "forecast_rows": int(len(result.to_frame())) if hasattr(result, "to_frame") else None,
        "metadata_schema": getattr(result, "metadata", {}).get("metadata_schema")
        if hasattr(result, "metadata")
        else None,
    }
    if not attach:
        return pipeline
    if not hasattr(result, "with_sidecar"):
        raise TypeError("result must be a ForecastResult when attach=True")
    return result.with_sidecar(sidecar_name, pipeline)


def oshapley_from_forecast_result(
    result: Any,
    X: Any,
    y: Any,
    models: str
    | Callable[..., Any]
    | ModelSpec
    | Sequence[str | Callable[..., Any] | ModelSpec]
    | Mapping[str, str | Callable[..., Any] | ModelSpec],
    *,
    sidecar_name: str = "oshapley",
    **kwargs: Any,
) -> Any:
    """Build an oShapley-VI/PBSV sidecar for a completed forecast result."""

    return anatomy_from_forecast_result(
        result,
        X,
        y,
        models,
        sidecar_name=sidecar_name,
        **kwargs,
    )


def _optional_anatomy() -> Any:
    try:
        return import_module("anatomy")
    except ImportError as exc:  # pragma: no cover - optional dependency message.
        raise ImportError(
            "forecast-accuracy anatomy requires the optional anatomy backend. "
            "Install with `pip install anatomy` or "
            "`pip install 'macroforecast[interpretation]'`."
        ) from exc


def _anatomy_subset_class() -> Any:
    anatomy_mod = _optional_anatomy()
    if hasattr(anatomy_mod, "AnatomySubset"):
        return anatomy_mod.AnatomySubset
    return import_module("anatomy._subsets").AnatomySubset


def _coerce_anatomy_index(index: pd.Index | Sequence[Any]) -> pd.Index:
    labels = pd.Index(index)
    if labels.empty:
        raise ValueError("index must not be empty")
    if not labels.is_unique:
        raise ValueError("index must be unique for anatomy subsets")
    if not labels.is_monotonic_increasing:
        raise ValueError("index must be monotonic increasing for anatomy subsets")
    return labels


def _aligned_xy(X: Any, y: Any) -> tuple[pd.DataFrame, pd.Series]:
    X_frame = pd.DataFrame(X).copy()
    y_series = pd.Series(y).copy()
    joined = pd.concat([y_series.rename("__target__"), X_frame], axis=1).dropna()
    if joined.empty:
        raise ValueError("X and y have no aligned non-missing rows")
    y_out = joined.pop("__target__")
    y_out.name = getattr(y_series, "name", None) or "target"
    return joined.astype(float), y_out.astype(float)


def _contiguous_slice(indices: Any, *, name: str) -> slice:
    positions = np.asarray(indices, dtype=int).reshape(-1)
    if positions.size == 0:
        raise ValueError(f"{name} must not be empty")
    expected = np.arange(int(positions[0]), int(positions[-1]) + 1, dtype=int)
    if not np.array_equal(positions, expected):
        raise ValueError(f"{name} must be contiguous for anatomy subsets")
    return slice(int(positions[0]), int(positions[-1]) + 1)


def _resolve_model_mapping(
    models: str
    | Callable[..., Any]
    | ModelSpec
    | Sequence[str | Callable[..., Any] | ModelSpec]
    | Mapping[str, str | Callable[..., Any] | ModelSpec],
) -> dict[str, ModelSpec]:
    if isinstance(models, Mapping):
        return {str(alias): get_model(spec) for alias, spec in models.items()}
    if isinstance(models, (str, ModelSpec)) or callable(models):
        spec = get_model(models)
        return {spec.name: spec}
    resolved = {}
    for item in models:
        spec = get_model(item)
        key = spec.name
        if key in resolved:
            suffix = 2
            while f"{key}_{suffix}" in resolved:
                suffix += 1
            key = f"{key}_{suffix}"
        resolved[key] = spec
    return resolved


def _resolve_model_params(
    specs: Mapping[str, ModelSpec],
    params: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    if not params:
        return {alias: {} for alias in specs}
    values = dict(params)
    if all(isinstance(value, Mapping) for value in values.values()):
        return {alias: dict(values.get(alias, {})) for alias in specs}
    return {alias: dict(values) for alias in specs}


def _model_feature_names(model: Any) -> tuple[str, ...]:
    if isinstance(model, ModelFit):
        return tuple(model.feature_names)
    names = getattr(model, "feature_names", None)
    if names is not None:
        return tuple(str(value) for value in names)
    names_in = getattr(model, "feature_names_in_", None)
    if names_in is not None:
        return tuple(str(value) for value in names_in)
    return ()


def _attach_anatomy_table_schema(
    table: pd.DataFrame,
    *,
    kind: str,
    metadata: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    table.attrs["macroforecast_metadata_schema"] = {
        "kind": kind,
        "version": 1,
        "columns": [str(column) for column in table.columns],
        "n_rows": int(len(table)),
    }
    table.attrs["macroforecast_metadata"] = dict(metadata or {})
    return table


def _metadata_frame(metadata: Mapping[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def walk(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            if not value and path:
                rows.append({"path": path, "value": {}, "type": "dict"})
            for key, item in value.items():
                walk(item, f"{path}.{key}" if path else str(key))
        elif isinstance(value, (list, tuple)):
            if not value and path:
                rows.append({"path": path, "value": [], "type": "list"})
            for pos, item in enumerate(value):
                walk(item, f"{path}[{pos}]")
        else:
            rows.append({"path": path or "value", "value": value, "type": type(value).__name__})

    walk(metadata, "")
    return pd.DataFrame(rows)


def _safe_table_name(value: Any) -> str:
    safe = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in str(value))
    return safe.strip("_") or "anatomy"


def _resolve_forecast_shapley_source(
    value: Any, *, sidecar_name: str | None
) -> Any:
    if isinstance(value, AnatomyPipelineResult):
        return value
    if hasattr(value, "get_sidecar"):
        if sidecar_name is not None:
            selected = value.get_sidecar(sidecar_name)
            if selected is None:
                raise KeyError(f"ForecastResult sidecar {sidecar_name!r} was not found")
            return selected
        names = tuple(value.sidecar_names()) if hasattr(value, "sidecar_names") else ()
        for name in ("oshapley", "forecast_shapley", "anatomy"):
            selected = value.get_sidecar(name)
            if selected is not None:
                return selected
        if len(names) == 1:
            return value.get_sidecar(names[0])
        if names:
            raise ValueError(
                "multiple ForecastResult sidecars are present; pass sidecar_name="
                f" explicitly. Available sidecars: {list(names)}"
            )
        raise ValueError("ForecastResult has no forecast-Shapley sidecar")
    return value


def _canonical_forecast_shapley_output(output: str) -> str:
    key = str(output).lower().replace("-", "_")
    aliases = {
        "raw": "forecast",
        "prediction": "forecast",
        "explain": "forecast",
        "explanation": "forecast",
        "forecast_explanation": "forecast",
        "vi": "oshapley",
        "variable_importance": "oshapley",
        "importance": "oshapley",
        "oshapley_vi": "oshapley",
        "o_shapley": "oshapley",
        "o_shapley_vi": "oshapley",
        "performance": "pbsv",
        "performance_value": "pbsv",
        "performance_values": "pbsv",
        "loss": "pbsv",
        "loss_decomposition": "pbsv",
        "all": "tables",
    }
    return aliases.get(key, key)


__all__ = [
    "AnatomyPipelineResult",
    "ForecastShapleyResult",
    "anatomy_from_forecast_result",
    "anatomy_model",
    "anatomy_output_transformer",
    "anatomy_pipeline",
    "anatomy_provider",
    "forecast_shapley_output",
    "oshapley_from_forecast_result",
    "oshapley_output",
    "oshapley_pipeline",
    "oshapley_provider",
    "precompute_anatomy",
    "precompute_oshapley",
    "window_to_anatomy_subsets",
    "window_to_oshapley_subsets",
]
