from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, TypeAlias

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
from pandas.tseries.offsets import DateOffset


StageScope: TypeAlias = str
StageUpdate: TypeAlias = str | int | DateOffset

_SCOPE_ALIASES = {
    "full": "full_panel",
    "full_panel": "full_panel",
    "global": "full_panel",
    "entire_panel": "full_panel",
    "origin": "origin_available",
    "origin_available": "origin_available",
    "available": "origin_available",
    "available_history": "origin_available",
    "observed_available": "origin_available",
    "fit": "fit_window",
    "fit_window": "fit_window",
    "train": "fit_window",
    "train_window": "fit_window",
    "estimation_window": "fit_window",
    "reference": "fixed_reference",
    "fixed": "fixed_reference",
    "fixed_reference": "fixed_reference",
    "custom": "custom",
}

_UPDATE_ALIASES = {
    "each": "every_origin",
    "every": "every_origin",
    "every_origin": "every_origin",
    "origin": "every_origin",
    "on_retrain": "on_retrain",
    "retrain": "on_retrain",
    "never": "never",
    "once": "never",
}


@dataclass(frozen=True)
class StagePolicy:
    """Fit/apply timing rule for one forecasting-run stage.

    ``scope`` decides what sample a stateful stage may use. ``update`` decides
    when a runner should refit or reuse that stage state across forecast
    origins.
    """

    scope: StageScope = "fit_window"
    update: StageUpdate = "every_origin"
    reference_start: Any | None = None
    reference_end: Any | None = None
    apply_to: tuple[str, ...] = ("fit", "test")
    metadata: dict[str, Any] = field(default_factory=dict)
    selector: Callable[..., Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope", _normalize_scope(self.scope))
        object.__setattr__(self, "update", _normalize_update(self.update))
        object.__setattr__(self, "apply_to", tuple(str(value) for value in self.apply_to))
        if self.scope == "fixed_reference" and self.reference_start is None and self.reference_end is None:
            raise ValueError("fixed_reference stage policy requires reference_start or reference_end")
        if self.scope == "custom" and self.selector is None:
            raise ValueError("custom stage policy requires selector")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready policy description."""

        return {
            "scope": self.scope,
            "update": _json_ready(self.update),
            "reference_start": _json_ready(self.reference_start),
            "reference_end": _json_ready(self.reference_end),
            "apply_to": list(self.apply_to),
            "metadata": _json_ready(self.metadata),
            "selector": _callable_name(self.selector),
        }


def stage_policy(
    scope: StageScope = "fit_window",
    *,
    update: StageUpdate = "every_origin",
    reference_start: Any | None = None,
    reference_end: Any | None = None,
    apply_to: tuple[str, ...] | list[str] = ("fit", "test"),
    metadata: Mapping[str, Any] | None = None,
    selector: Callable[..., Any] | None = None,
) -> StagePolicy:
    """Create a reusable stage timing policy."""

    return StagePolicy(
        scope=scope,
        update=update,
        reference_start=reference_start,
        reference_end=reference_end,
        apply_to=tuple(apply_to),
        metadata=dict(metadata or {}),
        selector=selector,
    )


def custom_stage_policy(
    selector: Callable[..., Any],
    *,
    update: StageUpdate = "every_origin",
    apply_to: tuple[str, ...] | list[str] = ("fit", "test"),
    metadata: Mapping[str, Any] | None = None,
) -> StagePolicy:
    """Create a stage policy whose sample labels are supplied by a callable."""

    if not callable(selector):
        raise TypeError("custom stage selector must be callable")
    return stage_policy(
        "custom",
        update=update,
        apply_to=apply_to,
        metadata=metadata,
        selector=selector,
    )


def resolve_stage_policy(
    policy: StagePolicy | str | None,
    *,
    default_scope: StageScope = "fit_window",
) -> StagePolicy:
    """Return a ``StagePolicy`` from a policy object, scope name, or default."""

    if policy is None:
        return stage_policy(default_scope)
    if isinstance(policy, StagePolicy):
        return policy
    return stage_policy(str(policy))


def stage_index(
    index: Any,
    item: Mapping[str, Any] | None,
    policy: StagePolicy | str | None,
) -> pd.Index:
    """Return labels allowed by one stage policy for one origin item."""

    labels = pd.Index(index)
    resolved = resolve_stage_policy(policy)
    if resolved.scope == "full_panel":
        return labels
    if resolved.scope == "fixed_reference":
        start, end = _reference_bounds(labels, resolved)
        return labels[start:end + 1]
    if item is None:
        raise ValueError(f"stage policy scope={resolved.scope!r} requires an origin item")
    if resolved.scope == "custom":
        output = resolved.selector(labels, item=item, policy=resolved)  # type: ignore[misc]
        return _index_from_selector_output(labels, output)
    if resolved.scope == "origin_available":
        return labels[_positions_from_item(item, "estimation")]
    if resolved.scope == "fit_window":
        return labels[_positions_from_item(item, "fit")]
    raise ValueError("custom stage policies require a caller-supplied hook")


def stage_panel(
    panel: pd.DataFrame,
    item: Mapping[str, Any] | None,
    policy: StagePolicy | str | None,
) -> pd.DataFrame:
    """Return panel rows allowed by one stage policy for one origin item."""

    return panel.reindex(stage_index(panel.index, item, policy))


def _normalize_scope(scope: StageScope) -> str:
    key = str(scope).lower().replace("-", "_")
    if key not in _SCOPE_ALIASES:
        allowed = "full_panel, origin_available, fit_window, fixed_reference, custom"
        raise ValueError(f"Unknown stage policy scope {scope!r}. Available scopes: {allowed}.")
    return _SCOPE_ALIASES[key]


def _positions_from_item(item: Mapping[str, Any], prefix: str) -> np.ndarray:
    idx_key = f"{prefix}_idx"
    if idx_key in item:
        return np.asarray(item[idx_key], dtype=int)
    start_key = f"{prefix}_start_pos"
    end_key = f"{prefix}_end_pos"
    if start_key in item and end_key in item:
        return np.arange(int(item[start_key]), int(item[end_key]) + 1, dtype=int)
    row = item.get("row")
    if isinstance(row, Mapping):
        return _positions_from_item(row, prefix)
    raise ValueError(f"origin item does not contain {prefix!r} positions")


def _index_from_selector_output(labels: pd.Index, output: Any) -> pd.Index:
    if isinstance(output, pd.Series):
        output = output.to_numpy()
    if isinstance(output, np.ndarray) and output.dtype == bool:
        if len(output) != len(labels):
            raise ValueError("custom stage selector boolean mask length does not match index")
        selected = labels[output]
    elif isinstance(output, slice):
        selected = labels[output]
    else:
        values = list(output) if not isinstance(output, pd.Index) else list(output)
        if not values:
            raise ValueError("custom stage selector returned no labels")
        if all(isinstance(value, (int, np.integer)) for value in values):
            selected = labels[np.asarray(values, dtype=int)]
        else:
            selected = labels.intersection(pd.Index(values), sort=False)
    if len(selected) == 0:
        raise ValueError("custom stage selector returned no labels")
    return pd.Index(selected)


def _reference_bounds(labels: pd.Index, policy: StagePolicy) -> tuple[int, int]:
    if not labels.is_monotonic_increasing:
        raise ValueError("fixed_reference stage policy requires a monotonic index")
    start = 0 if policy.reference_start is None else int(
        labels.searchsorted(_coerce_label(labels, policy.reference_start), side="left")
    )
    end = len(labels) - 1 if policy.reference_end is None else int(
        labels.searchsorted(_coerce_label(labels, policy.reference_end), side="right")
    ) - 1
    if start < 0 or end >= len(labels) or start > end:
        raise ValueError("fixed_reference stage policy selects an empty reference index")
    return start, end


def _coerce_label(labels: pd.Index, value: Any) -> Any:
    if isinstance(labels, pd.DatetimeIndex):
        return pd.Timestamp(value)
    return value


def _normalize_update(update: StageUpdate) -> StageUpdate:
    if isinstance(update, (int, np.integer)) and not isinstance(update, bool):
        value = int(update)
        if value < 1:
            raise ValueError("stage policy integer update cadence must be at least 1")
        return value
    if isinstance(update, DateOffset):
        _check_date_offset(update)
        return update
    key = str(update).lower().replace("-", "_")
    if key in _UPDATE_ALIASES:
        return _UPDATE_ALIASES[key]
    try:
        offset = to_offset(str(update))
    except ValueError as exc:
        allowed = "every_origin, on_retrain, never, positive integer, pandas offset"
        raise ValueError(f"Unknown stage policy update {update!r}. Available updates: {allowed}.") from exc
    _check_date_offset(offset)
    return offset


def _check_date_offset(offset: DateOffset) -> None:
    base = pd.Timestamp("2000-01-31")
    if base + offset <= base:
        raise ValueError("stage policy date-offset update must advance dates")


def _json_ready(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, DateOffset):
        return value.freqstr
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if callable(value):
        return _callable_name(value)
    return value


def _callable_name(func: Callable[..., Any] | None) -> str | None:
    if func is None:
        return None
    module = getattr(func, "__module__", "")
    qualname = getattr(func, "__qualname__", getattr(func, "__name__", repr(func)))
    return f"{module}.{qualname}" if module else str(qualname)


__all__ = [
    "StagePolicy",
    "custom_stage_policy",
    "resolve_stage_policy",
    "stage_index",
    "stage_panel",
    "stage_policy",
]
