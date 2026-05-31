from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CombinationSpec:
    """Forecast-combination request consumed by ``forecasting.run``."""

    method: str
    name: str
    models: tuple[str, ...] | None = None
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "name": self.name,
            "models": list(self.models) if self.models is not None else None,
            "params": dict(self.params),
        }


def combination_spec(
    method: str,
    *,
    name: str | None = None,
    models: Sequence[str] | None = None,
    **params: Any,
) -> CombinationSpec:
    """Build a runner-compatible forecast-combination spec."""

    canonical = _normalize_method(method)
    return CombinationSpec(
        method=canonical,
        name=name or f"combined_{canonical}",
        models=tuple(str(model) for model in models) if models is not None else None,
        params=dict(params),
    )


def combine_mean(forecasts: Any) -> pd.Series:
    """Equal-weight average forecast."""

    frame = _forecast_frame(forecasts)
    return frame.mean(axis=1).rename("combined")


def combine_median(forecasts: Any) -> pd.Series:
    """Cross-model median forecast."""

    frame = _forecast_frame(forecasts)
    return frame.median(axis=1).rename("combined")


def combine_trimmed_mean(forecasts: Any, *, trim: float = 0.1) -> pd.Series:
    """Trim extreme model forecasts before averaging."""

    frame = _forecast_frame(forecasts)
    if not 0 <= trim < 0.5:
        raise ValueError("trim must satisfy 0 <= trim < 0.5")
    values = np.sort(frame.to_numpy(dtype=float), axis=1)
    n_models = values.shape[1]
    cut = int(np.floor(trim * n_models))
    if cut:
        values = values[:, cut:-cut]
    return pd.Series(np.nanmean(values, axis=1), index=frame.index, name="combined")


def combine_winsorized_mean(forecasts: Any, *, limits: tuple[float, float] = (0.1, 0.1)) -> pd.Series:
    """Winsorize cross-model forecasts before averaging."""

    frame = _forecast_frame(forecasts)
    lower, upper = limits
    if lower < 0 or upper < 0 or lower + upper >= 1:
        raise ValueError("limits must be non-negative and sum to less than 1")
    q_low = frame.quantile(lower, axis=1)
    q_high = frame.quantile(1 - upper, axis=1)
    clipped = frame.clip(lower=q_low, upper=q_high, axis=0)
    return clipped.mean(axis=1).rename("combined")


def combine_inverse_mspe(
    forecasts: Any,
    y_true: Any,
    *,
    discount: float = 1.0,
    min_weight: float = 1e-12,
) -> pd.Series:
    """Combine forecasts with inverse discounted MSPE weights."""

    frame = _forecast_frame(forecasts)
    target = pd.Series(y_true).reindex(frame.index).astype(float)
    if not 0 < discount <= 1:
        raise ValueError("discount must satisfy 0 < discount <= 1")
    errors = frame.sub(target, axis=0) ** 2
    weights = pd.DataFrame(index=frame.index, columns=frame.columns, dtype=float)
    running = pd.Series(0.0, index=frame.columns, dtype=float)
    for step, date in enumerate(frame.index):
        if step == 0 or float(running.sum()) <= 0:
            weights.loc[date, :] = 1.0 / len(frame.columns)
        else:
            inv = 1.0 / running.clip(lower=min_weight)
            weights.loc[date, :] = inv / inv.sum()
        current = errors.loc[date]
        running = discount * running + current.fillna(running.mean() if running.notna().any() else 0.0)
    combined = (frame * weights).sum(axis=1)
    return combined.rename("combined")


combine_dmspe = combine_inverse_mspe


def combine_best_n(forecasts: Any, y_true: Any, *, n: int = 3) -> pd.Series:
    """Average the historically best ``n`` models by MSPE."""

    frame = _forecast_frame(forecasts)
    target = pd.Series(y_true).reindex(frame.index).astype(float)
    n_value = int(n)
    if n_value < 1:
        raise ValueError("n must be at least 1")
    mspe = frame.sub(target, axis=0).pow(2).expanding(min_periods=1).mean().shift(1)
    output = pd.Series(index=frame.index, dtype=float, name="combined")
    for date in frame.index:
        historical = mspe.loc[date]
        if historical.isna().all():
            best = frame.columns[:n_value]
        else:
            best = historical.fillna(float("inf")).sort_values().index[:n_value]
        output.loc[date] = frame.loc[date, best].mean()
    return output


def resolve_combinations(value: Any) -> list[CombinationSpec]:
    """Normalize runner ``combination=...`` input into concrete specs."""

    if value is None or value is False:
        return []
    if isinstance(value, CombinationSpec):
        return [value]
    if isinstance(value, str):
        return [combination_spec(value)]
    if isinstance(value, Mapping):
        if "method" in value:
            return [_mapping_to_spec(value)]
        return [_mapping_value_to_spec(alias, spec) for alias, spec in value.items()]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        specs: list[CombinationSpec] = []
        for item in value:
            specs.extend(resolve_combinations(item))
        return specs
    raise TypeError("combination must be a string, spec, sequence, mapping, or None")


def apply_combinations(
    forecasts: pd.DataFrame,
    specs: Sequence[CombinationSpec],
) -> list[dict[str, Any]]:
    """Create combined forecast records from base model forecast rows."""

    if not specs or forecasts.empty:
        return []
    required = {"date", "origin", "origin_pos", "horizon", "model", "prediction"}
    missing = required - set(forecasts.columns)
    if missing:
        raise ValueError(f"forecast table missing required columns: {sorted(missing)}")
    if "combined" in forecasts.columns:
        base = forecasts.loc[~forecasts["combined"].fillna(False).astype(bool)].copy()
    else:
        base = forecasts.copy()
    out: list[dict[str, Any]] = []
    for spec in specs:
        selected = _selected_model_rows(base, spec)
        if selected.empty:
            raise ValueError(
                f"combination {spec.name!r} has no matching base model forecasts"
            )
        for _, group in selected.groupby(["horizon"], dropna=False, sort=True):
            out.extend(_combined_group_records(group, spec))
    return out


def _combined_group_records(
    group: pd.DataFrame,
    spec: CombinationSpec,
) -> list[dict[str, Any]]:
    index_columns = ["date", "origin", "origin_pos", "horizon"]
    wide = (
        group.pivot_table(
            index=index_columns,
            columns="model",
            values="prediction",
            aggfunc="first",
        )
        .sort_index()
        .astype(float)
    )
    if wide.shape[1] < 1:
        return []
    actual = (
        group.drop_duplicates(index_columns)
        .set_index(index_columns)["actual"]
        .reindex(wide.index)
    )
    combined = _apply_combination_method(wide, actual, spec)
    templates = (
        group.sort_values(["date", "origin_pos", "model"])
        .drop_duplicates(index_columns)
        .set_index(index_columns)
    )
    records: list[dict[str, Any]] = []
    for key, prediction in combined.items():
        template = templates.loc[key].to_dict()
        record = {
            **template,
            "date": key[0],
            "origin": key[1],
            "origin_pos": key[2],
            "horizon": key[3],
            "model": spec.name,
            "model_spec": "forecast_combination",
            "prediction": None if pd.isna(prediction) else float(prediction),
            "variance_prediction": None,
            "quantile_predictions": None,
            "actual": None if pd.isna(actual.loc[key]) else float(actual.loc[key]),
            "params": dict(spec.params),
            "selection": None,
            "stored_model": None,
            "combined": True,
            "combination": spec.to_dict(),
        }
        records.append(record)
    return records


def _apply_combination_method(
    forecasts: pd.DataFrame,
    actual: pd.Series,
    spec: CombinationSpec,
) -> pd.Series:
    params = dict(spec.params)
    method = spec.method
    if method == "mean":
        return combine_mean(forecasts)
    if method == "median":
        return combine_median(forecasts)
    if method == "trimmed_mean":
        return combine_trimmed_mean(forecasts, **params)
    if method == "winsorized_mean":
        return combine_winsorized_mean(forecasts, **params)
    if method in {"inverse_mspe", "dmspe"}:
        return combine_inverse_mspe(forecasts, actual, **params)
    if method == "best_n":
        return combine_best_n(forecasts, actual, **params)
    raise ValueError(f"unsupported combination method {method!r}")


def _mapping_to_spec(value: Mapping[str, Any]) -> CombinationSpec:
    method = str(value["method"])
    reserved = {"method", "name", "models", "params"}
    params = dict(value.get("params", {}))
    params.update({key: item for key, item in value.items() if key not in reserved})
    return combination_spec(
        method,
        name=None if value.get("name") is None else str(value["name"]),
        models=value.get("models"),
        **params,
    )


def _mapping_value_to_spec(alias: Any, value: Any) -> CombinationSpec:
    if value is None:
        return combination_spec(str(alias))
    if isinstance(value, str):
        return combination_spec(value, name=str(alias))
    if isinstance(value, CombinationSpec):
        if value.name == f"combined_{value.method}":
            return CombinationSpec(
                method=value.method,
                name=str(alias),
                models=value.models,
                params=dict(value.params),
            )
        return value
    if isinstance(value, Mapping):
        spec = _mapping_to_spec({"name": str(alias), **value})
        return spec
    raise TypeError("mapping values must be strings, specs, mappings, or None")


def _selected_model_rows(frame: pd.DataFrame, spec: CombinationSpec) -> pd.DataFrame:
    base = frame.loc[frame["model"] != spec.name]
    if spec.models is None:
        return base
    return base.loc[base["model"].isin(spec.models)]


def _normalize_method(method: str) -> str:
    key = method.lower().replace("-", "_")
    aliases = {
        "average": "mean",
        "equal_weight": "mean",
        "equal_weighted": "mean",
        "trimmed": "trimmed_mean",
        "winsorized": "winsorized_mean",
        "inverse_mse": "inverse_mspe",
        "inverse_msfe": "inverse_mspe",
        "discounted_mspe": "dmspe",
        "best": "best_n",
    }
    key = aliases.get(key, key)
    allowed = {
        "mean",
        "median",
        "trimmed_mean",
        "winsorized_mean",
        "inverse_mspe",
        "dmspe",
        "best_n",
    }
    if key not in allowed:
        raise ValueError(f"unknown combination method {method!r}")
    return key


def _forecast_frame(forecasts: Any) -> pd.DataFrame:
    if isinstance(forecasts, pd.Series):
        frame = forecasts.to_frame()
    else:
        frame = pd.DataFrame(forecasts).copy()
    if frame.empty:
        raise ValueError("forecasts must not be empty")
    return frame.astype(float)


__all__ = [
    "CombinationSpec",
    "apply_combinations",
    "combine_best_n",
    "combine_dmspe",
    "combine_inverse_mspe",
    "combine_mean",
    "combine_median",
    "combine_trimmed_mean",
    "combine_winsorized_mean",
    "combination_spec",
    "resolve_combinations",
]
