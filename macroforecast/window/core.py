from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeAlias, cast

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
from pandas.tseries.offsets import DateOffset

Split = tuple[np.ndarray, np.ndarray]
TemporalCadence: TypeAlias = int | str | DateOffset
TestStep: TypeAlias = TemporalCadence

_VALIDATION_ALIASES = {
    "last": "last_block",
    "last_block": "last_block",
    "holdout": "last_block",
    "poos": "poos",
    "pseudo_out_of_sample": "poos",
    "expanding": "expanding",
    "expanding_walk_forward": "expanding",
    "time_series_split": "expanding",
    "rolling": "rolling_blocks",
    "rolling_blocks": "rolling_blocks",
    "rolling_walk_forward": "rolling_blocks",
    "blocked_kfold": "blocked_kfold",
    "block_cv": "blocked_kfold",
    "kfold": "blocked_kfold",
    "random_kfold": "random_kfold",
    "iid_kfold": "random_kfold",
    "random_cv": "random_kfold",
}


def _check_n_samples(n_samples: int) -> int:
    n = int(n_samples)
    if n < 2:
        raise ValueError("n_samples must be at least 2")
    return n


def _check_nonnegative_int(name: str, value: int) -> int:
    out = int(value)
    if out < 0:
        raise ValueError(f"{name} must be non-negative")
    return out


def _check_positive_int(name: str, value: int) -> int:
    out = int(value)
    if out < 1:
        raise ValueError(f"{name} must be at least 1")
    return out


def _check_positive_integer_like(name: str, value: Any) -> int:
    if not _is_position_step(value):
        raise ValueError(f"{name} must be a positive integer")
    out = int(value)
    if out < 1:
        raise ValueError(f"{name} must be a positive integer")
    return out


def _is_position_step(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _check_temporal_cadence(name: str, cadence: TemporalCadence) -> int | DateOffset:
    if _is_position_step(cadence):
        return _check_positive_int(name, int(cadence))
    if isinstance(cadence, str):
        if not cadence.strip():
            raise ValueError(f"{name} offset string must not be empty")
        try:
            offset = cast(DateOffset, to_offset(cadence))
        except ValueError as exc:
            raise ValueError(
                f"{name} must be a positive integer, pandas offset string, or DateOffset"
            ) from exc
        _check_date_offset_advances(name, offset)
        return offset
    if isinstance(cadence, DateOffset):
        _check_date_offset_advances(name, cadence)
        return cadence
    raise TypeError(
        f"{name} must be a positive integer, pandas offset string, or DateOffset"
    )


def _check_test_step(step: TestStep) -> int | DateOffset:
    return _check_temporal_cadence("step", step)


def _check_date_offset_advances(name: str, offset: DateOffset) -> None:
    base = pd.Timestamp("2000-01-31")
    shifted = base + offset
    if shifted <= base:
        raise ValueError(f"{name} calendar/date-offset cadence must advance dates")


def _json_ready(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, DateOffset):
        return value.freqstr
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def _coerce_index(index: int | Sequence[Any] | pd.Index) -> pd.Index:
    if isinstance(index, int):
        labels = pd.RangeIndex(index)
    else:
        labels = pd.Index(index)
    if len(labels) < 1:
        raise ValueError("index must not be empty")
    if not labels.is_unique:
        raise ValueError("index must be unique")
    if not labels.is_monotonic_increasing:
        raise ValueError("index must be monotonic increasing")
    return labels


def _resolve_position(
    index: pd.Index,
    value: Any | None,
    *,
    default: int,
    side: str,
    name: str,
) -> int:
    n = len(index)
    if value is None:
        return min(max(int(default), 0), n - 1)
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        pos = int(value)
    else:
        if value in index:
            location = index.get_loc(value)
            if isinstance(location, slice):
                pos = int(location.start if side == "left" else location.stop - 1)
            elif isinstance(location, np.ndarray):
                positions = np.flatnonzero(location)
                pos = int(positions[0] if side == "left" else positions[-1])
            else:
                pos = int(location)
        else:
            target = pd.Timestamp(value) if isinstance(index, pd.DatetimeIndex) else value
            insertion = int(index.searchsorted(target, side=side))
            pos = insertion if side == "left" else insertion - 1
    if pos < 0 or pos >= n:
        raise ValueError(f"{name} is outside the index range")
    return pos


def _callable_descriptor(func: Callable[[int, int], int]) -> str:
    name = getattr(func, "__name__", None)
    if isinstance(name, str) and name:
        return name
    return repr(func)


def _resolve_rolling_estimation_size(
    estimation: EstimationWindow,
    *,
    horizon: int,
) -> int:
    h = _check_positive_integer_like("horizon", horizon)
    if estimation.mode != "rolling":
        raise ValueError("horizon-dependent estimation size requires rolling mode")
    if estimation.size_by_horizon is not None:
        if h not in estimation.size_by_horizon:
            raise ValueError(f"size_by_horizon is missing horizon {h}")
        return _check_positive_integer_like(
            f"size_by_horizon[{h}]",
            estimation.size_by_horizon[h],
        )
    if estimation.size_rule is not None:
        if estimation.size is None:
            raise ValueError("size_rule requires a base rolling size")
        try:
            resolved = estimation.size_rule(int(estimation.size), h)
        except Exception as exc:  # noqa: BLE001 - preserve user-facing context.
            raise ValueError(
                f"size_rule failed to resolve rolling size for horizon {h}"
            ) from exc
        return _check_positive_integer_like(
            f"size_rule result for horizon {h}",
            resolved,
        )
    if estimation.size is None:
        raise ValueError("rolling estimation window requires size")
    return int(estimation.size)


def _default_first_origin_position(
    estimation: EstimationWindow,
    n_samples: int,
    *,
    horizon: int,
) -> int:
    minimum = estimation.min_size
    if minimum is None:
        if estimation.mode == "rolling":
            minimum = _resolve_rolling_estimation_size(estimation, horizon=horizon)
        else:
            minimum = estimation.size
    if minimum is None:
        minimum = max(1, n_samples // 2)
    return int(minimum) + int(estimation.embargo)


def _estimation_start_position(
    estimation: EstimationWindow,
    *,
    estimation_start_bound: int,
    estimation_end_pos: int,
    horizon: int,
) -> int:
    mode = estimation.mode.lower().replace("-", "_")
    if mode == "expanding":
        return int(estimation_start_bound)
    if mode == "rolling":
        size = _resolve_rolling_estimation_size(estimation, horizon=horizon)
        return max(int(estimation_start_bound), int(estimation_end_pos) - int(size) + 1)
    if mode == "fixed":
        return int(estimation_start_bound)
    raise ValueError("estimation mode must be one of: expanding, rolling, fixed")


def _iter_origin_positions(
    labels: pd.Index,
    *,
    first_origin: int,
    last_origin: int,
    step: int | DateOffset,
) -> Iterator[int]:
    if isinstance(step, int):
        yield from range(first_origin, last_origin + 1, step)
        return
    if not isinstance(labels, pd.DatetimeIndex):
        raise TypeError("calendar/date-offset step requires a DatetimeIndex")
    current_pos = int(first_origin)
    current_label = labels[current_pos]
    while current_pos <= last_origin:
        yield current_pos
        next_label = current_label + step
        if next_label <= current_label:
            raise ValueError("calendar/date-offset step must advance dates")
        next_pos = int(labels.searchsorted(next_label, side="left"))
        if next_pos >= len(labels) or next_pos > last_origin:
            break
        if next_pos <= current_pos:
            raise ValueError("calendar/date-offset step did not advance the origin")
        current_pos = next_pos
        current_label = labels[current_pos]


def _cadence_due(
    labels: pd.Index,
    *,
    origin_pos: int,
    origin_count: int,
    cadence: int | DateOffset,
    last_run_label: pd.Timestamp | None,
    name: str,
) -> bool:
    if isinstance(cadence, int):
        return origin_count % cadence == 0
    if not isinstance(labels, pd.DatetimeIndex):
        raise TypeError(f"calendar/date-offset {name} requires a DatetimeIndex")
    current_label = labels[origin_pos]
    if last_run_label is None:
        return True
    next_run_label = last_run_label + cadence
    if next_run_label <= last_run_label:
        raise ValueError(f"{name} calendar/date-offset cadence must advance dates")
    return bool(current_label >= next_run_label)


@dataclass(frozen=True)
class EstimationWindow:
    """Pre-test estimation-sample rule applied at each test origin."""

    mode: str = "expanding"
    start: Any | None = None
    end: Any | None = None
    min_size: int | None = None
    size: int | None = None
    size_rule: Callable[[int, int], int] | None = None
    size_by_horizon: Mapping[int, int] | None = None
    embargo: int = 0
    retrain_every: TemporalCadence = 1

    def __post_init__(self) -> None:
        mode = self.mode.lower().replace("-", "_")
        if mode not in {"expanding", "rolling", "fixed"}:
            raise ValueError("estimation mode must be one of: expanding, rolling, fixed")
        object.__setattr__(self, "mode", mode)
        if self.min_size is not None:
            object.__setattr__(
                self,
                "min_size",
                _check_positive_int("min_size", self.min_size),
            )
        if self.size is not None:
            object.__setattr__(
                self,
                "size",
                _check_positive_int("size", self.size),
            )
        if self.size_rule is not None and self.size_by_horizon is not None:
            raise ValueError("size_rule and size_by_horizon are mutually exclusive")
        if self.size_rule is not None and not callable(self.size_rule):
            raise ValueError("size_rule must be callable")
        if self.size_by_horizon is not None:
            if not isinstance(self.size_by_horizon, Mapping):
                raise ValueError("size_by_horizon must be a mapping")
            if not self.size_by_horizon:
                raise ValueError("size_by_horizon must not be empty")
            checked_sizes: dict[int, int] = {}
            for raw_horizon, raw_size in self.size_by_horizon.items():
                h = _check_positive_integer_like("size_by_horizon horizon", raw_horizon)
                checked_sizes[h] = _check_positive_integer_like(
                    f"size_by_horizon[{h}]",
                    raw_size,
                )
            object.__setattr__(self, "size_by_horizon", checked_sizes)
        object.__setattr__(self, "embargo", _check_nonnegative_int("embargo", self.embargo))
        has_horizon_size = (
            self.size_rule is not None or self.size_by_horizon is not None
        )
        if mode != "rolling" and has_horizon_size:
            raise ValueError(
                "size_rule and size_by_horizon are only valid for rolling estimation"
            )
        if mode == "rolling":
            if self.size_rule is not None and self.size is None:
                raise ValueError("size_rule requires a base rolling size")
            if not has_horizon_size and self.size is None:
                raise ValueError("rolling estimation window requires size")
        object.__setattr__(
            self,
            "retrain_every",
            _check_temporal_cadence("retrain_every", self.retrain_every),
        )

    def to_dict(self) -> dict[str, Any]:
        out = {
            "mode": self.mode,
            "start": _json_ready(self.start),
            "end": _json_ready(self.end),
            "min_size": self.min_size,
            "size": self.size,
            "embargo": self.embargo,
            "retrain_every": _json_ready(self.retrain_every),
        }
        if self.size_rule is not None:
            out["size_rule"] = _callable_descriptor(self.size_rule)
        if self.size_by_horizon is not None:
            out["size_by_horizon"] = {
                int(horizon): int(size)
                for horizon, size in sorted(self.size_by_horizon.items())
            }
        return out


@dataclass(frozen=True)
class ValWindow:
    """Validation rule used for model and hyperparameter selection."""

    method: str = "expanding"
    size: int | None = None
    ratio: float = 0.2
    min_train_size: int | None = None
    n_splits: int = 5
    horizon: int = 1
    step: int = 1
    embargo: int | None = None
    random_state: int | None = None
    retune_every: TemporalCadence = 1
    retune_on_retrain: bool = True
    reuse_params: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "method", normalize_window_name(self.method))
        if self.size is not None:
            object.__setattr__(self, "size", _check_positive_int("size", self.size))
        if not 0 < float(self.ratio) < 1:
            raise ValueError("ratio must be between 0 and 1")
        if self.min_train_size is not None:
            object.__setattr__(
                self,
                "min_train_size",
                _check_positive_int("min_train_size", self.min_train_size),
            )
        object.__setattr__(self, "n_splits", _check_positive_int("n_splits", self.n_splits))
        object.__setattr__(self, "horizon", _check_positive_int("horizon", self.horizon))
        object.__setattr__(self, "step", _check_positive_int("step", self.step))
        if self.embargo is not None:
            object.__setattr__(self, "embargo", _check_nonnegative_int("embargo", self.embargo))
        object.__setattr__(
            self,
            "retune_every",
            _check_temporal_cadence("retune_every", self.retune_every),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "size": self.size,
            "ratio": self.ratio,
            "min_train_size": self.min_train_size,
            "n_splits": self.n_splits,
            "horizon": self.horizon,
            "step": self.step,
            "embargo": self.embargo,
            "random_state": self.random_state,
            "retune_every": _json_ready(self.retune_every),
            "retune_on_retrain": self.retune_on_retrain,
            "reuse_params": self.reuse_params,
        }


@dataclass(frozen=True)
class TestWindow:
    """Final test-origin and horizon rule."""

    first_origin: Any | None = None
    last_origin: Any | None = None
    horizon: int = 1
    step: TestStep = 1
    drop_incomplete: bool = True
    exclude: tuple[tuple[Any | None, Any | None], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "horizon", _check_positive_int("horizon", self.horizon))
        object.__setattr__(self, "step", _check_test_step(self.step))
        object.__setattr__(
            self,
            "exclude",
            tuple((start, end) for start, end in self.exclude),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "first_origin": _json_ready(self.first_origin),
            "last_origin": _json_ready(self.last_origin),
            "horizon": self.horizon,
            "step": _json_ready(self.step),
            "drop_incomplete": self.drop_incomplete,
            "exclude": _json_ready(self.exclude),
        }


@dataclass(frozen=True)
class AlignmentWindow:
    """Feature/target alignment rule before model fitting."""

    join: str = "inner"
    drop_missing: bool = True
    require_full_horizon: bool = True

    def __post_init__(self) -> None:
        if self.join not in {"inner", "left", "right", "outer"}:
            raise ValueError("join must be one of: inner, left, right, outer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "join": self.join,
            "drop_missing": self.drop_missing,
            "require_full_horizon": self.require_full_horizon,
        }


@dataclass(frozen=True)
class WindowSpec:
    """Macro forecasting time frame passed across selection/model/evaluation."""

    method: str = "expanding"
    estimation: EstimationWindow = field(default_factory=EstimationWindow)
    val: ValWindow = field(default_factory=ValWindow)
    test: TestWindow = field(default_factory=TestWindow)
    alignment: AlignmentWindow = field(default_factory=AlignmentWindow)
    validation_size: int | None = None
    validation_ratio: float = 0.2
    min_train_size: int | None = None
    n_splits: int = 5
    step: int = 1
    horizon: int = 1
    embargo: int = 0
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        method = normalize_window_name(self.method)
        if self.val == ValWindow() and method != self.val.method:
            object.__setattr__(
                self,
                "val",
                ValWindow(
                    method=method,
                    size=self.validation_size,
                    ratio=self.validation_ratio,
                    min_train_size=self.min_train_size,
                    n_splits=self.n_splits,
                    horizon=self.horizon,
                    step=self.step,
                    embargo=self.embargo,
                ),
            )
        object.__setattr__(self, "method", self.val.method)
        if self.validation_size is not None:
            object.__setattr__(
                self,
                "validation_size",
                _check_positive_int("validation_size", self.validation_size),
            )
        if self.min_train_size is not None:
            object.__setattr__(
                self,
                "min_train_size",
                _check_positive_int("min_train_size", self.min_train_size),
            )
        object.__setattr__(self, "n_splits", _check_positive_int("n_splits", self.n_splits))
        object.__setattr__(self, "step", _check_positive_int("step", self.step))
        object.__setattr__(self, "horizon", _check_positive_int("horizon", self.horizon))
        object.__setattr__(self, "embargo", _check_nonnegative_int("embargo", self.embargo))

    def split(self, n_samples: int) -> list[Split]:
        """Return train/val index splits for ``n_samples``."""

        return make_splitter(
            self.val.method,
            n_samples,
            validation_size=self.val.size
            if self.val.size is not None
            else self.validation_size,
            validation_ratio=self.val.ratio,
            min_train_size=self.min_train_size
            if self.min_train_size is not None
            else self.val.min_train_size
            if self.val.min_train_size is not None
            else None,
            n_splits=self.val.n_splits,
            step=self.val.step,
            horizon=self.val.horizon,
            random_state=self.val.random_state,
            embargo=self.val.embargo
            if self.val.embargo is not None
            else self.embargo
            if self.embargo != 0
            else self.estimation.embargo,
        )

    def to_table(self, n_samples: int, *, index: pd.Index | None = None) -> pd.DataFrame:
        """Return this window's train/val splits as an inspectable table."""

        return split_table(
            self.val.method,
            n_samples,
            index=index,
            validation_size=self.val.size
            if self.val.size is not None
            else self.validation_size,
            validation_ratio=self.val.ratio,
            min_train_size=self.min_train_size
            if self.min_train_size is not None
            else self.val.min_train_size
            if self.val.min_train_size is not None
            else None,
            n_splits=self.val.n_splits,
            step=self.val.step,
            horizon=self.val.horizon,
            random_state=self.val.random_state,
            embargo=self.val.embargo
            if self.val.embargo is not None
            else self.embargo
            if self.embargo != 0
            else self.estimation.embargo,
        )

    def plan(
        self,
        index: int | Sequence[Any] | pd.Index,
        *,
        exclude_origin: bool = False,
    ) -> pd.DataFrame:
        """Return an execution plan with estimation, val, and test metadata."""

        labels = _coerce_index(index)
        rows = self.origins(labels, exclude_origin=exclude_origin).copy()
        retune_group = -1
        selection_start_pos: int | None = None
        selection_end_pos: int | None = None
        retune_cadence = _check_temporal_cadence("retune_every", self.val.retune_every)
        last_retune_label: pd.Timestamp | None = None
        val_columns = [
            "val_method",
            "retune",
            "retune_group",
            "retune_cadence",
            "retune_on_retrain",
            "reuse_params",
            "selection_start",
            "selection_end",
            "selection_start_pos",
            "selection_end_pos",
            "n_selection",
            "n_val_splits",
            "val_start",
            "val_end",
            "val_start_pos",
            "val_end_pos",
        ]
        for column in val_columns:
            rows[column] = pd.NA
        for loc, row in rows.iterrows():
            origin_count = int(loc)
            # Retuning can be scheduled more often than refitting. By default,
            # scheduled retunes only run at origins where a new model is fit;
            # skipped origins carry the previous selection window forward.
            origin_pos = int(row["origin_pos"])
            scheduled_retune = _cadence_due(
                labels,
                origin_pos=origin_pos,
                origin_count=origin_count,
                cadence=retune_cadence,
                last_run_label=last_retune_label,
                name="retune_every",
            )
            retrain = bool(row["retrain"])
            retune = scheduled_retune and (
                retrain or not self.val.retune_on_retrain
            )
            if retune:
                retune_group += 1
                if not isinstance(retune_cadence, int):
                    last_retune_label = cast(pd.Timestamp, labels[origin_pos])
                selection_start_pos = int(row["estimation_start_pos"])
                selection_end_pos = int(row["estimation_end_pos"])
                splits = self.val_splits_for_origin(
                    labels, origin_pos, exclude_origin=exclude_origin
                )
                val_positions = np.concatenate([split[1] for split in splits])
                rows.at[loc, "n_val_splits"] = len(splits)
                rows.at[loc, "val_start"] = labels[int(val_positions.min())]
                rows.at[loc, "val_end"] = labels[int(val_positions.max())]
                rows.at[loc, "val_start_pos"] = int(val_positions.min())
                rows.at[loc, "val_end_pos"] = int(val_positions.max())
            rows.at[loc, "val_method"] = self.val.method
            rows.at[loc, "retune"] = bool(retune)
            rows.at[loc, "retune_group"] = int(retune_group)
            rows.at[loc, "retune_cadence"] = _json_ready(retune_cadence)
            rows.at[loc, "retune_on_retrain"] = bool(self.val.retune_on_retrain)
            rows.at[loc, "reuse_params"] = bool(self.val.reuse_params)
            if selection_start_pos is not None and selection_end_pos is not None:
                rows.at[loc, "selection_start"] = labels[selection_start_pos]
                rows.at[loc, "selection_end"] = labels[selection_end_pos]
                rows.at[loc, "selection_start_pos"] = int(selection_start_pos)
                rows.at[loc, "selection_end_pos"] = int(selection_end_pos)
                rows.at[loc, "n_selection"] = int(
                    selection_end_pos - selection_start_pos + 1
                )
            if not retune:
                rows.at[loc, "n_val_splits"] = 0
        return rows

    def val_splits_for_origin(
        self,
        index: int | Sequence[Any] | pd.Index,
        origin: Any,
        *,
        exclude_origin: bool = False,
    ) -> list[Split]:
        """Return absolute-position inner train/val splits for one test origin."""

        labels = _coerce_index(index)
        row = self._origin_row(labels, origin, exclude_origin=exclude_origin)
        start = int(row["estimation_start_pos"])
        end = int(row["estimation_end_pos"])
        n_samples = end - start + 1
        relative = make_splitter(
            self.val.method,
            n_samples,
            validation_size=self.val.size,
            validation_ratio=self.val.ratio,
            min_train_size=self.min_train_size
            if self.min_train_size is not None
            else self.val.min_train_size
            if self.val.min_train_size is not None
            else None,
            n_splits=self.val.n_splits,
            step=self.val.step,
            horizon=self.val.horizon,
            random_state=self.val.random_state,
            # Same three-level embargo fallback as split()/to_table(): the
            # WindowSpec-level embargo must not be skipped, or per-origin
            # validation splits disagree with the planned/reported splits when
            # WindowSpec(embargo=X, estimation=EstimationWindow(embargo=0)).
            embargo=self.val.embargo
            if self.val.embargo is not None
            else self.embargo
            if self.embargo != 0
            else self.estimation.embargo,
        )
        return [(train_idx + start, val_idx + start) for train_idx, val_idx in relative]

    def iter_origins(
        self,
        index: int | Sequence[Any] | pd.Index,
        *,
        exclude_origin: bool = False,
    ) -> Iterator[dict[str, Any]]:
        """Yield origin metadata and absolute-position slices for model runners."""

        labels = _coerce_index(index)
        for _, row in self.plan(labels, exclude_origin=exclude_origin).iterrows():
            fit_idx: np.ndarray = np.arange(
                int(row["fit_start_pos"]),
                int(row["fit_end_pos"]) + 1,
                dtype=int,
            )
            estimation_idx: np.ndarray = np.arange(
                int(row["estimation_start_pos"]),
                int(row["estimation_end_pos"]) + 1,
                dtype=int,
            )
            test_idx: np.ndarray = np.arange(
                int(row["test_start_pos"]),
                int(row["test_end_pos"]) + 1,
                dtype=int,
            )
            yield {
                "row": row.to_dict(),
                "estimation_idx": estimation_idx,
                "fit_idx": fit_idx,
                "test_idx": test_idx,
                "val_splits": self.val_splits_for_origin(
                    labels,
                    int(row["origin_pos"]),
                    exclude_origin=exclude_origin,
                )
                if bool(row["retune"])
                else [],
            }

    def iter_slices(
        self,
        X: pd.DataFrame | pd.Series,
        y: pd.Series | pd.DataFrame | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield origin metadata with already sliced ``X`` and optional ``y``."""

        aligned = self.align(X, y)
        if y is None:
            aligned_X = cast(pd.DataFrame, aligned)
            aligned_y = None
        else:
            aligned_X, aligned_y = cast(
                tuple[pd.DataFrame, pd.Series | pd.DataFrame],
                aligned,
            )
        if not isinstance(aligned_X, pd.DataFrame):
            raise TypeError("aligned X must be a pandas DataFrame")
        for item in self.iter_origins(aligned_X.index):
            estimation_idx = item["estimation_idx"]
            fit_idx = item["fit_idx"]
            test_idx = item["test_idx"]
            output = {
                **item,
                "X_estimation": aligned_X.iloc[estimation_idx],
                "X_fit": aligned_X.iloc[fit_idx],
                "X_test": aligned_X.iloc[test_idx],
            }
            if aligned_y is None:
                output.update({
                    "y_estimation": None,
                    "y_fit": None,
                    "y_test": None,
                })
            else:
                output.update({
                    "y_estimation": aligned_y.iloc[estimation_idx],
                    "y_fit": aligned_y.iloc[fit_idx],
                    "y_test": aligned_y.iloc[test_idx],
                })
            yield output

    def validate(
        self,
        index: int | Sequence[Any] | pd.Index,
        *,
        exclude_origin: bool = False,
    ) -> dict[str, Any]:
        """Return a validation report for this time-frame specification."""

        errors: list[str] = []
        warnings: list[str] = []
        try:
            labels = _coerce_index(index)
            plan = self.plan(labels, exclude_origin=exclude_origin)
        except Exception as exc:  # noqa: BLE001 - validation reports user-facing errors.
            return {
                "ok": False,
                "n_obs": len(index) if not isinstance(index, int) else int(index),
                "n_origins": 0,
                "n_retrain": 0,
                "n_retune": 0,
                "errors": [str(exc)],
                "warnings": warnings,
            }

        if (plan["estimation_end_pos"] >= plan["test_start_pos"]).any():
            errors.append("estimation_end must be before test_start for every origin")
        if (plan["fit_end_pos"] >= plan["test_start_pos"]).any():
            errors.append("fit_end must be before test_start for every origin")
        if (plan["n_estimation"] < 1).any():
            errors.append("all origins must have at least one estimation observation")
        if (plan["n_test"] < 1).any():
            errors.append("all origins must have at least one test observation")
        retune_rows = plan.loc[plan["retune"].astype(bool)]
        non_retune_rows = plan.loc[~plan["retune"].astype(bool)]
        if not retune_rows.empty and (retune_rows["n_val_splits"].astype(int) < 1).any():
            errors.append("retune origins must have at least one validation split")
        if not bool(plan["retune"].iloc[0]):
            errors.append("the first emitted origin must retune before parameters can be reused")
        if not self.val.reuse_params and not non_retune_rows.empty:
            errors.append("reuse_params=False requires retuning at every emitted origin")
        if self.val.retune_on_retrain and not retune_rows.empty:
            retune_without_retrain = ~retune_rows["retrain"].astype(bool)
            if retune_without_retrain.any():
                errors.append("retune_on_retrain=True only allows retuning at retrain origins")
        if plan["selection_start_pos"].isna().any() or plan["selection_end_pos"].isna().any():
            errors.append("all origins must have an active selection window")
        if (plan["fit_end_pos"] < plan["estimation_end_pos"]).any():
            warnings.append("fit window lags estimation window when retrain_every > 1")
        if plan["test_start_pos"].duplicated().any():
            warnings.append("test origins contain duplicate test_start positions")
        test_horizon = int(self.test.horizon)
        if test_horizon > 1 and (
            plan["estimation_end_pos"] + test_horizon > plan["test_start_pos"]
        ).any():
            warnings.append(
                "estimation/test embargo is below horizon - 1: the production fit "
                "includes training rows whose h-step targets realise at or after the "
                "forecast origin (pseudo-out-of-sample convention). Pass "
                "embargo=horizon - 1 for a strict real-time protocol."
            )
        return {
            "ok": not errors,
            "n_obs": len(labels),
            "n_origins": len(plan),
            "n_retrain": int(plan["retrain"].astype(bool).sum()),
            "n_retune": int(plan["retune"].astype(bool).sum()),
            "first_origin": _json_ready(plan["origin"].iloc[0]),
            "last_origin": _json_ready(plan["origin"].iloc[-1]),
            "errors": errors,
            "warnings": warnings,
        }

    def origins(
        self,
        index: int | Sequence[Any] | pd.Index,
        *,
        exclude_origin: bool = False,
    ) -> pd.DataFrame:
        """Return test-origin rows with train and test ranges.

        ``exclude_origin`` controls whether the test slice starts at the origin
        position itself (the default -- every feature-matrix forecast policy
        relies on ``test_idx[0] == origin_pos`` as the "as-of" feature row, see
        the ``_feature_window_for_policy`` call site in ``forecasting/runner.py``)
        or one position after it. Panel-input models treat every row of the test
        slice as a genuine forecast target (``_fit_predict_panel_origin``), so
        their origin date must not double as a forecast row, and the slice must
        extend one position further to still reach the requested horizon.

        Under ``exclude_origin=True`` the estimation window also runs THROUGH
        the origin (``origin_pos - embargo``) instead of stopping one short of
        it: the origin is the last in-sample date, so the model fits on data
        through the origin and its positional forecast steps 1..h line up with
        the test dates ``origin+1 .. origin+h``. The embargo gap between
        estimation end and test start is identical in both modes. See issue
        #423.
        """

        labels = _coerce_index(index)
        n = len(labels)
        if n < 2:
            raise ValueError("index must contain at least 2 observations")
        horizon = _check_positive_int("horizon", self.test.horizon)
        step = _check_test_step(self.test.step)
        embargo = _check_nonnegative_int("embargo", self.estimation.embargo)
        estimation_start_bound = _resolve_position(
            labels,
            self.estimation.start,
            default=0,
            side="left",
            name="estimation_start",
        )
        estimation_end_bound = _resolve_position(
            labels,
            self.estimation.end,
            default=n - 1,
            side="right",
            name="estimation_end",
        )
        first_default = estimation_start_bound + _default_first_origin_position(
            self.estimation,
            n,
            horizon=horizon,
        )
        last_default = (
            n - horizon - (1 if exclude_origin else 0)
            if self.test.drop_incomplete
            else n - 1
        )
        first_origin = _resolve_position(
            labels,
            self.test.first_origin,
            default=first_default,
            side="left",
            name="first_origin",
        )
        last_origin = _resolve_position(
            labels,
            self.test.last_origin,
            default=last_default,
            side="right",
            name="last_origin",
        )
        if first_origin > last_origin:
            raise ValueError("first_origin must be earlier than or equal to last_origin")
        rows: list[dict[str, Any]] = []
        retrain_group = -1
        fit_start_pos: int | None = None
        fit_end_pos: int | None = None
        retrain_cadence = _check_temporal_cadence(
            "retrain_every",
            self.estimation.retrain_every,
        )
        last_retrain_label: pd.Timestamp | None = None
        origin_positions = _iter_origin_positions(
            labels,
            first_origin=first_origin,
            last_origin=last_origin,
            step=step,
        )
        for origin_pos in origin_positions:
            if exclude_origin:
                # Panel test slice: origin_pos+1 .. origin_pos+horizon. The
                # origin date itself is excluded from the test window (it is
                # not a genuine forecast target), and the window extends one
                # position further than the historical formula so the true
                # h-step-ahead date is still reachable. See issue #423.
                test_start_pos = origin_pos + 1
                test_end_pos = origin_pos + horizon
                if test_start_pos >= n:
                    # Nothing exists past the origin at all; there is no test
                    # row to produce regardless of drop_incomplete.
                    continue
            else:
                test_start_pos = origin_pos
                test_end_pos = origin_pos + horizon - 1
            if test_end_pos >= n:
                if self.test.drop_incomplete:
                    continue
                test_end_pos = n - 1
            if exclude_origin:
                # The origin is the LAST IN-SAMPLE date: estimation runs
                # through the origin itself (minus embargo), and the first
                # test row sits at origin_pos + 1. Without this, panel
                # models would fit through origin-1 while their forecast
                # steps are labeled from origin+1 -- relabeling every value
                # one step late (panel models forecast positionally from
                # the end of their fit data). The embargo gap between
                # estimation end and test start is the same in both modes.
                # See issue #423.
                estimation_end_pos = min(origin_pos - embargo, estimation_end_bound)
            else:
                estimation_end_pos = min(origin_pos - embargo - 1, estimation_end_bound)
            estimation_start_pos = _estimation_start_position(
                self.estimation,
                estimation_start_bound=estimation_start_bound,
                estimation_end_pos=estimation_end_pos,
                horizon=horizon,
            )
            if estimation_end_pos < estimation_start_pos:
                continue
            n_estimation = estimation_end_pos - estimation_start_pos + 1
            if (
                self.estimation.min_size is not None
                and n_estimation < int(self.estimation.min_size)
            ):
                continue
            origin_count = len(rows)
            retrain = _cadence_due(
                labels,
                origin_pos=origin_pos,
                origin_count=origin_count,
                cadence=retrain_cadence,
                last_run_label=last_retrain_label,
                name="retrain_every",
            )
            if retrain:
                retrain_group += 1
                if not isinstance(retrain_cadence, int):
                    last_retrain_label = cast(pd.Timestamp, labels[origin_pos])
                fit_start_pos = estimation_start_pos
                fit_end_pos = estimation_end_pos
            if fit_start_pos is None or fit_end_pos is None:
                fit_start_pos = estimation_start_pos
                fit_end_pos = estimation_end_pos
            rows.append({
                "origin": labels[origin_pos],
                "origin_pos": int(origin_pos),
                "estimation_start": labels[estimation_start_pos],
                "estimation_end": labels[estimation_end_pos],
                "fit_start": labels[fit_start_pos],
                "fit_end": labels[fit_end_pos],
                "test_start": labels[test_start_pos],
                "test_end": labels[test_end_pos],
                "estimation_start_pos": int(estimation_start_pos),
                "estimation_end_pos": int(estimation_end_pos),
                "fit_start_pos": int(fit_start_pos),
                "fit_end_pos": int(fit_end_pos),
                "test_start_pos": int(test_start_pos),
                "test_end_pos": int(test_end_pos),
                "horizon": int(horizon),
                "step": _json_ready(step),
                "test_step": _json_ready(step),
                "retrain": bool(retrain),
                "retrain_group": int(retrain_group),
                "retrain_cadence": _json_ready(retrain_cadence),
                "estimation_mode": self.estimation.mode,
                "n_estimation": int(n_estimation),
                "n_fit": int(fit_end_pos - fit_start_pos + 1),
                "n_test": int(test_end_pos - test_start_pos + 1),
            })
        if not rows:
            raise ValueError("window produced no test origins")
        return pd.DataFrame(rows)

    def _origin_row(
        self,
        labels: pd.Index,
        origin: Any,
        *,
        exclude_origin: bool = False,
    ) -> pd.Series:
        rows = self.origins(labels, exclude_origin=exclude_origin)
        if isinstance(origin, pd.Series):
            return origin
        if isinstance(origin, dict):
            return pd.Series(origin)
        if isinstance(origin, (int, np.integer)) and not isinstance(origin, bool):
            matches = rows.loc[rows["origin_pos"] == int(origin)]
        else:
            matches = rows.loc[rows["origin"] == origin]
        if matches.empty:
            raise ValueError(f"origin {origin!r} is not in the window origin table")
        return matches.iloc[0]

    def test_mask(self, index: int | Sequence[Any] | pd.Index) -> pd.Series:
        """Return a boolean final-test mask indexed by the supplied labels."""

        labels = _coerce_index(index)
        mask = pd.Series(False, index=labels, name="test")
        origins = self.origins(labels)
        for _, row in origins.iterrows():
            mask.iloc[int(row["test_start_pos"]): int(row["test_end_pos"]) + 1] = True
        for start, end in self.test.exclude:
            start_pos = _resolve_position(
                labels,
                start,
                default=0,
                side="left",
                name="test_exclude_start",
            )
            end_pos = _resolve_position(
                labels,
                end,
                default=len(labels) - 1,
                side="right",
                name="test_exclude_end",
            )
            if start_pos <= end_pos:
                mask.iloc[start_pos:end_pos + 1] = False
        return mask

    def align(
        self,
        X: pd.DataFrame | pd.Series,
        y: pd.Series | pd.DataFrame | None = None,
    ) -> pd.DataFrame | tuple[pd.DataFrame, pd.Series | pd.DataFrame]:
        """Align feature and target objects according to the alignment rule."""

        x_frame = X.to_frame() if isinstance(X, pd.Series) else pd.DataFrame(X).copy()
        if y is None:
            return x_frame.dropna() if self.alignment.drop_missing else x_frame
        y_obj = y.copy()
        joined = x_frame.join(y_obj, how=self.alignment.join, rsuffix="__target")
        x_cols = list(x_frame.columns)
        y_cols = [column for column in joined.columns if column not in x_cols]
        if self.alignment.drop_missing:
            joined = joined.dropna()
        elif self.alignment.require_full_horizon and y_cols:
            joined = joined.dropna(subset=y_cols)
        aligned_X = joined.loc[:, x_cols]
        aligned_y = joined.loc[:, y_cols]
        if isinstance(y, pd.Series):
            if len(y_cols) != 1:
                raise ValueError("aligned target could not be resolved to one Series")
            return aligned_X, aligned_y.iloc[:, 0].rename(y.name)
        return aligned_X, aligned_y

    def to_dict(self) -> dict[str, Any]:
        """Return a metadata representation of the window."""

        return {
            "method": normalize_window_name(self.method),
            "estimation": self.estimation.to_dict(),
            "val": self.val.to_dict(),
            "test": self.test.to_dict(),
            "alignment": self.alignment.to_dict(),
            "validation_size": self.validation_size,
            "validation_ratio": self.validation_ratio,
            "min_train_size": self.min_train_size,
            "n_splits": self.n_splits,
            "step": self.step,
            "horizon": self.horizon,
            "embargo": self.embargo,
            "metadata": dict(self.metadata or {}),
        }


def normalize_window_name(window: str) -> str:
    """Return the canonical window method name for a method or alias."""

    key = str(window).lower().replace("-", "_")
    try:
        return _VALIDATION_ALIASES[key]
    except KeyError as exc:
        allowed = (
            "last_block, poos, expanding, rolling_blocks, blocked_kfold, "
            "random_kfold"
        )
        raise ValueError(
            f"Unknown window method {window!r}. Available methods: {allowed}."
        ) from exc


def resolve_window(window: WindowSpec | str | None = None) -> WindowSpec:
    """Return a ``WindowSpec`` from a spec, method name, or default."""

    if window is None:
        return expanding()
    if isinstance(window, WindowSpec):
        return window
    return WindowSpec(method=normalize_window_name(window))


def spec(
    *,
    estimation: EstimationWindow | None = None,
    val: ValWindow | None = None,
    test: TestWindow | None = None,
    alignment: AlignmentWindow | None = None,
    method: str = "expanding",
    metadata: dict[str, Any] | None = None,
) -> WindowSpec:
    """Compose a full estimation/val/test macro window from component windows."""

    estimation_rule = estimation or EstimationWindow()
    val_rule = val or ValWindow(method=method)
    test_rule = test or TestWindow(horizon=val_rule.horizon, step=val_rule.step)
    return WindowSpec(
        method=val_rule.method,
        estimation=estimation_rule,
        val=val_rule,
        test=test_rule,
        alignment=alignment or AlignmentWindow(),
        validation_size=val_rule.size,
        validation_ratio=val_rule.ratio,
        min_train_size=val_rule.min_train_size,
        n_splits=val_rule.n_splits,
        step=val_rule.step,
        horizon=val_rule.horizon,
        embargo=estimation_rule.embargo,
        metadata=metadata,
    )


def from_cutoffs(
    *,
    test_start: Any,
    test_end: Any | None = None,
    estimation_start: Any | None = None,
    mode: str = "expanding",
    estimation_size: int | None = None,
    estimation_size_rule: Callable[[int, int], int] | None = None,
    estimation_size_by_horizon: Mapping[int, int] | None = None,
    estimation_min_size: int | None = None,
    embargo: int = 0,
    retrain_every: TemporalCadence = 1,
    val_method: str = "last_block",
    val_size: int | None = None,
    val_ratio: float = 0.2,
    val_min_train_size: int | None = None,
    val_n_splits: int = 5,
    val_horizon: int | None = None,
    val_step: int = 1,
    val_embargo: int | None = None,
    val_random_state: int | None = None,
    retune_every: TemporalCadence = 1,
    retune_on_retrain: bool = True,
    reuse_params: bool = True,
    horizon: int = 1,
    step: TestStep = 1,
    drop_incomplete: bool = True,
    exclude: Sequence[tuple[Any | None, Any | None]] = (),
    alignment: AlignmentWindow | None = None,
    metadata: dict[str, Any] | None = None,
) -> WindowSpec:
    """Build a window from common estimation/test cutoff dates.

    Embargo conventions for h-step targets (``horizon > 1``)
    --------------------------------------------------------
    A row at position ``p`` carries the direct h-step target realised at
    ``p + horizon``. Two boundaries are embargoed independently:

    * Estimation/test boundary (``embargo``, default ``0``). With ``embargo=0``
      the production model for origin ``t`` is fit on rows up to ``t - 1`` whose
      targets realise at ``t + horizon - 1`` -- i.e. after the forecast is
      issued. This is the standard *pseudo-out-of-sample* convention used on a
      fixed (final-vintage) dataset, and it maximises the training sample. For a
      *strict real-time* protocol, where every training label must be observable
      at the origin, pass ``embargo=horizon - 1`` (the last training label then
      realises at the origin). The default is deliberately the pseudo-OOS choice;
      it does not enforce real-time observability.

    * Train/validation boundary (``val_embargo``, default ``horizon - 1``). This
      purges the gap between the last training label and the first validation
      input. The default ``horizon - 1`` leaves the single boundary timestamp
      shared between the last training label and the first validation feature;
      pass ``val_embargo=horizon`` for a fully disjoint purge.

    Both defaults are conventions, not guarantees of real-time observability;
    set the embargoes explicitly when the forecasting protocol requires it.
    """

    mode_key = str(mode).lower().replace("-", "_")
    has_horizon_size = (
        estimation_size_rule is not None or estimation_size_by_horizon is not None
    )
    if mode_key != "rolling" and has_horizon_size:
        raise ValueError(
            "estimation_size_rule and estimation_size_by_horizon are only valid "
            "when mode='rolling'"
        )
    if mode_key == "expanding":
        estimation = estimation_expanding(
            start=estimation_start,
            min_size=estimation_min_size,
            embargo=embargo,
            retrain_every=retrain_every,
        )
    elif mode_key == "rolling":
        if (
            estimation_size is None
            and estimation_size_rule is None
            and estimation_size_by_horizon is None
        ):
            raise ValueError("estimation_size is required when mode='rolling'")
        if estimation_size_rule is not None and estimation_size is None:
            raise ValueError(
                "estimation_size is required when estimation_size_rule is supplied"
            )
        estimation = estimation_rolling(
            start=estimation_start,
            size=estimation_size,
            size_rule=estimation_size_rule,
            size_by_horizon=estimation_size_by_horizon,
            min_size=estimation_min_size,
            embargo=embargo,
            retrain_every=retrain_every,
        )
    elif mode_key == "fixed":
        estimation = estimation_fixed(
            start=estimation_start,
            min_size=estimation_min_size,
            embargo=embargo,
            retrain_every=retrain_every,
        )
    else:
        raise ValueError("mode must be one of: expanding, rolling, fixed")

    val_key = normalize_window_name(val_method)
    inner_horizon = horizon if val_horizon is None else val_horizon
    # For an h-step target, a training row at position p realises its label at
    # p + h, so the validation train/val boundary must be embargoed by at least
    # horizon - 1 to keep training labels from realising inside the validation
    # block. Default the validation embargo to horizon - 1 (the standard h-step
    # purge) for EVERY validation method when the caller does not set one.
    effective_val_embargo = (
        val_embargo
        if val_embargo is not None
        else (max(0, int(inner_horizon) - 1) if inner_horizon else 0)
    )
    if val_key == "last_block":
        val = val_last_block(
            size=val_size,
            ratio=val_ratio,
            embargo=effective_val_embargo,
            retune_every=retune_every,
            retune_on_retrain=retune_on_retrain,
            reuse_params=reuse_params,
        )
    elif val_key == "poos":
        val = val_poos(
            size=val_size,
            ratio=val_ratio,
            embargo=effective_val_embargo,
            retune_every=retune_every,
            retune_on_retrain=retune_on_retrain,
            reuse_params=reuse_params,
        )
    elif val_key == "expanding":
        val = val_expanding(
            min_train_size=val_min_train_size,
            step=val_step,
            horizon=inner_horizon,
            embargo=effective_val_embargo,
            retune_every=retune_every,
            retune_on_retrain=retune_on_retrain,
            reuse_params=reuse_params,
        )
    elif val_key == "rolling_blocks":
        val = val_rolling_blocks(
            n_blocks=val_n_splits,
            block_size=val_size,
            embargo=effective_val_embargo,
            retune_every=retune_every,
            retune_on_retrain=retune_on_retrain,
            reuse_params=reuse_params,
        )
    elif val_key == "blocked_kfold":
        val = val_blocked_kfold(
            n_splits=val_n_splits,
            embargo=effective_val_embargo,
            retune_every=retune_every,
            retune_on_retrain=retune_on_retrain,
            reuse_params=reuse_params,
        )
    else:
        val = val_random_kfold(
            n_splits=val_n_splits,
            random_state=val_random_state,
            retune_every=retune_every,
            retune_on_retrain=retune_on_retrain,
            reuse_params=reuse_params,
        )

    meta = {"from_cutoffs": True}
    meta.update(metadata or {})
    return spec(
        estimation=estimation,
        val=val,
        test=test_origins(
            first_origin=test_start,
            last_origin=test_end,
            horizon=horizon,
            step=step,
            drop_incomplete=drop_incomplete,
            exclude=exclude,
        ),
        alignment=alignment,
        metadata=meta,
    )


def estimation_expanding(
    *,
    start: Any | None = None,
    min_size: int | None = None,
    embargo: int = 0,
    retrain_every: TemporalCadence = 1,
) -> EstimationWindow:
    """Estimation rule that expands from ``start`` through each test origin."""

    return EstimationWindow(
        mode="expanding",
        start=start,
        min_size=min_size,
        embargo=embargo,
        retrain_every=retrain_every,
    )


def estimation_rolling(
    *,
    start: Any | None = None,
    size: int | None = None,
    size_rule: Callable[[int, int], int] | None = None,
    size_by_horizon: Mapping[int, int] | None = None,
    min_size: int | None = None,
    embargo: int = 0,
    retrain_every: TemporalCadence = 1,
) -> EstimationWindow:
    """Estimation rule with a trailing sample size at each test origin."""

    return EstimationWindow(
        mode="rolling",
        start=start,
        size=size,
        size_rule=size_rule,
        size_by_horizon=size_by_horizon,
        min_size=min_size,
        embargo=embargo,
        retrain_every=retrain_every,
    )


def estimation_fixed(
    *,
    start: Any | None = None,
    end: Any | None = None,
    min_size: int | None = None,
    embargo: int = 0,
    retrain_every: TemporalCadence = 1,
) -> EstimationWindow:
    """Estimation rule with a fixed start and optional fixed end bound."""

    return EstimationWindow(
        mode="fixed",
        start=start,
        end=end,
        min_size=min_size,
        embargo=embargo,
        retrain_every=retrain_every,
    )


def val_last_block(
    *,
    size: int | None = None,
    ratio: float = 0.2,
    embargo: int | None = None,
    retune_every: TemporalCadence = 1,
    retune_on_retrain: bool = True,
    reuse_params: bool = True,
) -> ValWindow:
    """Validation rule with one final holdout block."""

    return ValWindow(
        method="last_block",
        size=size,
        ratio=ratio,
        embargo=embargo,
        retune_every=retune_every,
        retune_on_retrain=retune_on_retrain,
        reuse_params=reuse_params,
    )


def val_poos(
    *,
    size: int | None = None,
    ratio: float = 0.25,
    embargo: int | None = None,
    retune_every: TemporalCadence = 1,
    retune_on_retrain: bool = True,
    reuse_params: bool = True,
) -> ValWindow:
    """Validation rule with one-step pseudo-out-of-sample tail splits."""

    return ValWindow(
        method="poos",
        size=size,
        ratio=ratio,
        embargo=embargo,
        retune_every=retune_every,
        retune_on_retrain=retune_on_retrain,
        reuse_params=reuse_params,
    )


def val_expanding(
    *,
    min_train_size: int | None = None,
    step: int = 1,
    horizon: int = 1,
    embargo: int | None = None,
    retune_every: TemporalCadence = 1,
    retune_on_retrain: bool = True,
    reuse_params: bool = True,
) -> ValWindow:
    """Validation rule with expanding train windows."""

    return ValWindow(
        method="expanding",
        min_train_size=min_train_size,
        step=step,
        horizon=horizon,
        embargo=embargo,
        retune_every=retune_every,
        retune_on_retrain=retune_on_retrain,
        reuse_params=reuse_params,
    )


def val_rolling_blocks(
    *,
    n_blocks: int = 3,
    block_size: int | None = None,
    embargo: int | None = None,
    retune_every: TemporalCadence = 1,
    retune_on_retrain: bool = True,
    reuse_params: bool = True,
) -> ValWindow:
    """Validation rule with consecutive validation blocks over the sample tail."""

    return ValWindow(
        method="rolling_blocks",
        size=block_size,
        n_splits=n_blocks,
        embargo=embargo,
        retune_every=retune_every,
        retune_on_retrain=retune_on_retrain,
        reuse_params=reuse_params,
    )


def val_blocked_kfold(
    *,
    n_splits: int = 5,
    embargo: int | None = None,
    retune_every: TemporalCadence = 1,
    retune_on_retrain: bool = True,
    reuse_params: bool = True,
) -> ValWindow:
    """Validation rule with chronological blocked folds."""

    return ValWindow(
        method="blocked_kfold",
        n_splits=n_splits,
        embargo=embargo,
        retune_every=retune_every,
        retune_on_retrain=retune_on_retrain,
        reuse_params=reuse_params,
    )


def val_random_kfold(
    *,
    n_splits: int = 5,
    random_state: int | None = 0,
    retune_every: TemporalCadence = 1,
    retune_on_retrain: bool = True,
    reuse_params: bool = True,
) -> ValWindow:
    """Validation rule with randomly assigned iid-style folds.

    This is useful for reproducing papers that explicitly used random K-fold
    CV. It is not the default macro-forecasting validation rule because train
    folds can contain observations later than their validation folds.
    """

    return ValWindow(
        method="random_kfold",
        n_splits=n_splits,
        random_state=random_state,
        retune_every=retune_every,
        retune_on_retrain=retune_on_retrain,
        reuse_params=reuse_params,
    )


def test_origins(
    *,
    first_origin: Any | None = None,
    last_origin: Any | None = None,
    horizon: int = 1,
    step: TestStep = 1,
    drop_incomplete: bool = True,
    exclude: Sequence[tuple[Any | None, Any | None]] = (),
) -> TestWindow:
    """Final test-origin rule for model-stage out-of-sample runs."""

    return TestWindow(
        first_origin=first_origin,
        last_origin=last_origin,
        horizon=horizon,
        step=step,
        drop_incomplete=drop_incomplete,
        exclude=tuple(exclude),
    )


def alignment_drop_incomplete(
    *,
    join: str = "inner",
    require_full_horizon: bool = True,
) -> AlignmentWindow:
    """Alignment rule that drops rows with missing feature or target values."""

    return AlignmentWindow(
        join=join,
        drop_missing=True,
        require_full_horizon=require_full_horizon,
    )


def alignment_keep_missing(
    *,
    join: str = "inner",
    require_full_horizon: bool = True,
) -> AlignmentWindow:
    """Alignment rule that preserves missing rows after index alignment."""

    return AlignmentWindow(
        join=join,
        drop_missing=False,
        require_full_horizon=require_full_horizon,
    )


def last_block(
    *,
    validation_size: int | None = None,
    validation_ratio: float = 0.2,
    embargo: int = 0,
) -> WindowSpec:
    """Configure one final val block."""

    return spec(
        estimation=estimation_expanding(embargo=embargo),
        val=val_last_block(size=validation_size, ratio=validation_ratio, embargo=embargo),
        test=test_origins(horizon=1),
    )


def poos(
    *,
    validation_size: int | None = None,
    validation_ratio: float = 0.25,
    embargo: int = 0,
) -> WindowSpec:
    """Configure pseudo-out-of-sample one-step tail splits."""

    return spec(
        estimation=estimation_expanding(embargo=embargo),
        val=val_poos(size=validation_size, ratio=validation_ratio, embargo=embargo),
        test=test_origins(horizon=1),
    )


def expanding(
    *,
    min_train_size: int | None = None,
    step: int = 1,
    horizon: int = 1,
    embargo: int = 0,
) -> WindowSpec:
    """Configure expanding-window train/val splits."""

    return spec(
        estimation=estimation_expanding(min_size=min_train_size, embargo=embargo),
        val=val_expanding(
            min_train_size=min_train_size,
            step=step,
            horizon=horizon,
            embargo=embargo,
        ),
        test=test_origins(horizon=horizon, step=step),
    )


def rolling_blocks(
    *,
    n_blocks: int = 3,
    block_size: int | None = None,
    embargo: int = 0,
) -> WindowSpec:
    """Configure consecutive validation blocks over the sample tail."""

    return spec(
        estimation=estimation_expanding(embargo=embargo),
        val=val_rolling_blocks(n_blocks=n_blocks, block_size=block_size, embargo=embargo),
        test=test_origins(horizon=1),
    )


def blocked_kfold(*, n_splits: int = 5, embargo: int = 0) -> WindowSpec:
    """Configure chronological blocked k-fold validation."""

    return spec(
        estimation=estimation_expanding(embargo=embargo),
        val=val_blocked_kfold(n_splits=n_splits, embargo=embargo),
        test=test_origins(horizon=1),
    )


def random_kfold(*, n_splits: int = 5, random_state: int | None = 0) -> WindowSpec:
    """Configure randomly assigned iid-style K-fold validation."""

    return spec(
        estimation=estimation_expanding(),
        val=val_random_kfold(n_splits=n_splits, random_state=random_state),
        test=test_origins(horizon=1),
    )


def _resolve_validation_size(
    n_samples: int,
    *,
    validation_size: int | None,
    validation_ratio: float,
) -> int:
    if validation_size is not None:
        size = int(validation_size)
    else:
        if not 0 < validation_ratio < 1:
            raise ValueError("validation_ratio must be between 0 and 1")
        size = int(np.ceil(n_samples * validation_ratio))
    if size < 1:
        raise ValueError("validation_size must be at least 1")
    if size >= n_samples:
        raise ValueError("validation_size must be smaller than n_samples")
    return size


def _train_val(train_end: int, val_start: int, val_end: int) -> Split:
    if train_end <= 0:
        raise ValueError("split has no training observations")
    if val_start >= val_end:
        raise ValueError("split has no validation observations")
    return np.arange(train_end, dtype=int), np.arange(val_start, val_end, dtype=int)


def last_block_split(
    n_samples: int,
    *,
    validation_size: int | None = None,
    validation_ratio: float = 0.2,
    embargo: int = 0,
) -> Iterator[Split]:
    """Yield one split with the last block held out for validation."""

    n = _check_n_samples(n_samples)
    gap = _check_nonnegative_int("embargo", embargo)
    holdout = _resolve_validation_size(
        n, validation_size=validation_size, validation_ratio=validation_ratio
    )
    val_start = n - holdout
    train_end = val_start - gap
    yield _train_val(train_end, val_start, n)


def poos_split(
    n_samples: int,
    *,
    validation_size: int | None = None,
    validation_ratio: float = 0.25,
    embargo: int = 0,
) -> Iterator[Split]:
    """Yield pseudo-out-of-sample one-step validation splits over the tail block."""

    n = _check_n_samples(n_samples)
    gap = _check_nonnegative_int("embargo", embargo)
    holdout = _resolve_validation_size(
        n, validation_size=validation_size, validation_ratio=validation_ratio
    )
    start = n - holdout
    for val_start in range(start, n):
        train_end = val_start - gap
        yield _train_val(train_end, val_start, val_start + 1)


def expanding_split(
    n_samples: int,
    *,
    min_train_size: int | None = None,
    step: int = 1,
    horizon: int = 1,
    embargo: int = 0,
) -> Iterator[Split]:
    """Yield expanding-window validation splits."""

    n = _check_n_samples(n_samples)
    h = int(horizon)
    if h < 1:
        raise ValueError("horizon must be at least 1")
    if step < 1:
        raise ValueError("step must be at least 1")
    gap = _check_nonnegative_int("embargo", embargo)
    minimum = int(min_train_size) if min_train_size is not None else max(1, n // 2)
    if minimum >= n:
        raise ValueError("min_train_size must be smaller than n_samples")
    for train_end in range(minimum, n - gap - h + 1, int(step)):
        val_start = train_end + gap
        yield _train_val(train_end, val_start, val_start + h)


def rolling_blocks_split(
    n_samples: int,
    *,
    n_blocks: int = 3,
    block_size: int | None = None,
    embargo: int = 0,
) -> Iterator[Split]:
    """Yield consecutive validation blocks with all prior observations as training data."""

    n = _check_n_samples(n_samples)
    blocks = int(n_blocks)
    if blocks < 1:
        raise ValueError("n_blocks must be at least 1")
    size = int(block_size) if block_size is not None else max(1, n // (blocks + 2))
    if size < 1:
        raise ValueError("block_size must be at least 1")
    if blocks * size >= n:
        raise ValueError("n_blocks * block_size must be smaller than n_samples")
    gap = _check_nonnegative_int("embargo", embargo)
    start = n - blocks * size
    for block in range(blocks):
        val_start = start + block * size
        val_end = min(val_start + size, n)
        train_end = val_start - gap
        yield _train_val(train_end, val_start, val_end)


def blocked_kfold_split(
    n_samples: int,
    *,
    n_splits: int = 5,
    embargo: int = 0,
) -> Iterator[Split]:
    """Yield chronological blocked-fold splits using only past data for training."""

    n = _check_n_samples(n_samples)
    splits = int(n_splits)
    if splits < 2:
        raise ValueError("n_splits must be at least 2")
    if splits > n:
        raise ValueError("n_splits must be smaller than or equal to n_samples")
    boundaries = np.linspace(0, n, splits + 1, dtype=int)
    gap = _check_nonnegative_int("embargo", embargo)
    emitted = 0
    for fold in range(splits):
        val_start = int(boundaries[fold])
        val_end = int(boundaries[fold + 1])
        train_end = val_start - gap
        if train_end <= 0 or val_start >= val_end:
            continue
        emitted += 1
        yield _train_val(train_end, val_start, val_end)
    if emitted == 0:
        raise ValueError("blocked_kfold_split produced no valid chronological folds")


def random_kfold_split(
    n_samples: int,
    *,
    n_splits: int = 5,
    random_state: int | None = 0,
) -> Iterator[Split]:
    """Yield randomly assigned K-fold splits.

    Each fold trains on all non-validation positions. This intentionally does
    not enforce temporal ordering, so use it only when reproducing methods that
    used random iid folds, not as the default macro forecast validation design.
    """

    n = _check_n_samples(n_samples)
    splits = int(n_splits)
    if splits < 2:
        raise ValueError("n_splits must be at least 2")
    if splits > n:
        raise ValueError("n_splits must be smaller than or equal to n_samples")
    rng = np.random.default_rng(random_state)
    shuffled: np.ndarray = np.arange(n, dtype=int)
    rng.shuffle(shuffled)
    folds = np.array_split(shuffled, splits)
    all_idx: np.ndarray = np.arange(n, dtype=int)
    for fold in folds:
        val_idx = np.sort(fold.astype(int, copy=False))
        if len(val_idx) == 0:
            continue
        train_idx = np.setdiff1d(all_idx, val_idx, assume_unique=True)
        if len(train_idx) == 0:
            continue
        yield (train_idx.astype(int, copy=False), val_idx.astype(int, copy=False))


def make_splitter(
    validation: str,
    n_samples: int,
    *,
    validation_size: int | None = None,
    validation_ratio: float = 0.2,
    min_train_size: int | None = None,
    n_splits: int = 5,
    step: int = 1,
    horizon: int = 1,
    random_state: int | None = None,
    embargo: int = 0,
) -> list[Split]:
    """Build validation splits from a validation method name."""

    key = normalize_window_name(validation)
    if key == "last_block":
        splits = list(
            last_block_split(
                n_samples,
                validation_size=validation_size,
                validation_ratio=validation_ratio,
                embargo=embargo,
            )
        )
    elif key == "poos":
        splits = list(
            poos_split(
                n_samples,
                validation_size=validation_size,
                validation_ratio=validation_ratio,
                embargo=embargo,
            )
        )
    elif key == "expanding":
        splits = list(
            expanding_split(
                n_samples,
                min_train_size=min_train_size,
                step=step,
                horizon=horizon,
                embargo=embargo,
            )
        )
    elif key == "rolling_blocks":
        splits = list(
            rolling_blocks_split(
                n_samples,
                n_blocks=n_splits,
                block_size=validation_size,
                embargo=embargo,
            )
        )
    elif key == "blocked_kfold":
        splits = list(blocked_kfold_split(n_samples, n_splits=n_splits, embargo=embargo))
    elif key == "random_kfold":
        splits = list(
            random_kfold_split(
                n_samples,
                n_splits=n_splits,
                random_state=random_state,
            )
        )
    if not splits:
        raise ValueError(f"Validation method {key!r} produced no splits")
    return splits


def split_table(
    validation: str,
    n_samples: int,
    *,
    index: pd.Index | None = None,
    validation_size: int | None = None,
    validation_ratio: float = 0.2,
    min_train_size: int | None = None,
    n_splits: int = 5,
    step: int = 1,
    horizon: int = 1,
    random_state: int | None = None,
    embargo: int = 0,
) -> pd.DataFrame:
    """Return validation splits as an inspectable table."""

    splits = make_splitter(
        validation,
        n_samples,
        validation_size=validation_size,
        validation_ratio=validation_ratio,
        min_train_size=min_train_size,
        n_splits=n_splits,
        step=step,
        horizon=horizon,
        random_state=random_state,
        embargo=embargo,
    )
    labels = index if index is not None else pd.RangeIndex(int(n_samples))
    if len(labels) != int(n_samples):
        raise ValueError("index length must equal n_samples")
    rows = []
    for i, (train_idx, val_idx) in enumerate(splits):
        rows.append({
            "split": i,
            "n_train": int(len(train_idx)),
            "n_validation": int(len(val_idx)),
            "train_start": labels[int(train_idx[0])],
            "train_end": labels[int(train_idx[-1])],
            "validation_start": labels[int(val_idx[0])],
            "validation_end": labels[int(val_idx[-1])],
            "train_start_pos": int(train_idx[0]),
            "train_end_pos": int(train_idx[-1]),
            "validation_start_pos": int(val_idx[0]),
            "validation_end_pos": int(val_idx[-1]),
        })
    return pd.DataFrame(rows)


__all__ = [
    "AlignmentWindow",
    "EstimationWindow",
    "Split",
    "TestWindow",
    "ValWindow",
    "WindowSpec",
    "alignment_drop_incomplete",
    "alignment_keep_missing",
    "blocked_kfold",
    "blocked_kfold_split",
    "estimation_expanding",
    "estimation_fixed",
    "estimation_rolling",
    "expanding",
    "expanding_split",
    "from_cutoffs",
    "last_block",
    "last_block_split",
    "make_splitter",
    "normalize_window_name",
    "poos",
    "poos_split",
    "random_kfold",
    "random_kfold_split",
    "resolve_window",
    "rolling_blocks",
    "rolling_blocks_split",
    "spec",
    "split_table",
    "test_origins",
    "val_blocked_kfold",
    "val_expanding",
    "val_last_block",
    "val_poos",
    "val_random_kfold",
    "val_rolling_blocks",
]
