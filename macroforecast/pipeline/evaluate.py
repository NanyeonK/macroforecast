"""Pipeline evaluation: accuracy, DM/CW significance, MCS, cross-arm combinations (Stage 2)."""
from __future__ import annotations

import hashlib
import json
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from macroforecast.pipeline.spec import (
    CALIBRATION_EVAL_TESTS,
    CombinationContender,
    PipelineSpec,
    SubsampleWindow,
    contender_names,
)

if TYPE_CHECKING:
    # Type-only, and deliberately so: ``evaluation_inputs`` is the module that
    # loads FRED series, and importing it at runtime would put a data loader back
    # in this module's import graph -- the exact coupling this file is kept free
    # of. ``from __future__ import annotations`` makes the annotation a string,
    # so nothing here needs the module to exist at run time.
    from macroforecast.pipeline.evaluation_inputs import ResolvedEvaluationInputs

# The (target, horizon) grouping keys every evaluation table requires. When the
# master forecast frame is EMPTY (zero rows, hence no columns) or otherwise lacks
# these columns, ``groupby(["target","horizon"])`` raises KeyError; the tables
# instead return an empty frame with the right columns.
_GROUP_KEYS = {"target", "horizon"}

# The three metrics ``accuracy_table`` has always computed, kept as the default
# ``EvalSpec.metrics`` value and given their benchmark-relative formulas below
# regardless of how they were resolved (name or explicit request), so that the
# default output is byte-identical to before ``EvalSpec.metrics`` was consumed.
_DEFAULT_ACCURACY_METRICS: tuple[str, ...] = ("rmse", "relative_mse", "r2_oos")
_BUILTIN_ACCURACY_METRICS = {"rmse", "relative_mse", "r2_oos"}

_SIGNIFICANCE_COLUMNS = ["target", "horizon", "contender"]
_MCS_COLUMNS = ["target", "horizon", "contender", "in_mcs"]
_DENSITY_GROUP_BY = ["target", "horizon", "contender"]
_DENSITY_COLUMNS = ["target", "horizon", "contender", "n"]
_CALIBRATION_COLUMNS = [
    "target", "horizon", "contender", "test", "statistic", "p_value", "reject", "n_obs",
    "coverage_rate",
]
_DEGRADATION_EXCEPTIONS = (
    ValueError,
    FloatingPointError,
    ZeroDivisionError,
    np.linalg.LinAlgError,
)


def _has_groups(master: pd.DataFrame) -> bool:
    """True when ``master`` is non-empty and carries the (target, horizon) keys."""
    return not master.empty and _GROUP_KEYS.issubset(master.columns)


def _warn_failed_cells(master: pd.DataFrame) -> None:
    failed = list(getattr(master, "attrs", {}).get("macroforecast_failed_cells", []))
    if not failed:
        return
    first = failed[0]
    warnings.warn(
        f"evaluate() received a forecast frame with failed_cells={len(failed)}; "
        "failed arms are absent from accuracy/significance/MCS outputs. First "
        f"failed cell: target={first.get('target')!r}, arm={first.get('arm')!r}, "
        f"horizons={first.get('horizons')!r}, error={first.get('error')!r}.",
        RuntimeWarning,
        stacklevel=2,
    )


def _degraded_mask(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(False, index=frame.index)
    mask = pd.Series(False, index=frame.index)
    if "status" in frame.columns:
        status = frame["status"].astype("string")
        mask |= status == "degraded"
    if "reason" in frame.columns and "status" not in frame.columns:
        mask |= frame["reason"].notna()
    return mask


def _warn_degraded_results(result: Mapping[str, pd.DataFrame]) -> None:
    degraded = {
        name: int(_degraded_mask(frame).sum())
        for name, frame in result.items()
        if name in {"significance", "mcs"} and isinstance(frame, pd.DataFrame)
    }
    degraded = {name: count for name, count in degraded.items() if count}
    if not degraded:
        return
    details = ", ".join(f"{name}={count}" for name, count in sorted(degraded.items()))
    warnings.warn(
        "evaluation produced degraded significance/MCS row(s) with status/reason "
        f"instead of dropping them: {details}.",
        RuntimeWarning,
        stacklevel=2,
    )


def _evaluation_result(**tables: pd.DataFrame) -> dict[str, pd.DataFrame]:
    result = dict(tables)
    _warn_degraded_results(result)
    return result


# ``evaluate``/``accuracy_table``/... are pure-frame and duck-type friendly: some
# callers (and tests) pass a minimal spec double carrying only
# ``evaluation.benchmark``. The newly-consumed EvalSpec fields are therefore read
# with ``getattr`` defaults matching :class:`~macroforecast.pipeline.spec.EvalSpec`'s
# own defaults, so such spec doubles keep the exact pre-threading behavior.
_DEFAULT_EVAL_TESTS: tuple[str, ...] = ("dm", "cw", "mcs")
_PAIRWISE_LONG_TESTS = frozenset({"gw", "enc_new", "enc_t", "pt", "hm", "ag", "gr", "mz"})
_NESTED_QUADRATIC_TESTS = frozenset({"cw", "enc_new", "enc_t"})
_SET_COMPARISON_TESTS = frozenset({"spa", "rc", "stepm"})
_MULTI_HORIZON_TESTS = frozenset({"uspa", "aspa"})
#: Minimum common origins a pairwise contender-vs-benchmark test is run on, and
#: minimum aligned origins (over at least two horizons) for a joint multi-horizon
#: test. These are unchanged -- a Diebold-Mariano statistic on five origins is not
#: a weaker result, it is a number with no sampling distribution behind it. What
#: changed is that falling short is now REPORTED rather than dropped: the cell
#: leaves a row saying what was asked, what sample was available, and why nothing
#: was computed, so a missing row in a results table can no longer be mistaken for
#: a test that ran and found nothing.
_MIN_PAIRWISE_ORIGINS = 8
_MIN_JOINT_ORIGINS = 4
_MIN_JOINT_HORIZONS = 2
_SUBSAMPLE_DATE_COLUMN = "date"
_SUBSAMPLE_PROVENANCE_ATTR = "macroforecast_subsample_provenance"


def _eval_metrics(spec: PipelineSpec) -> tuple[str | Callable[..., float], ...]:
    return tuple(getattr(spec.evaluation, "metrics", _DEFAULT_ACCURACY_METRICS))


def _eval_tests(spec: PipelineSpec) -> tuple[str, ...]:
    return tuple(getattr(spec.evaluation, "tests", _DEFAULT_EVAL_TESTS))


def _eval_loss(spec: PipelineSpec) -> Callable[[Any, Any], Any] | None:
    return getattr(spec.evaluation, "loss", None)


def _eval_test_options(spec: PipelineSpec, test_name: str) -> dict[str, Any]:
    options = getattr(spec.evaluation, "test_options", {}) or {}
    return dict(options.get(test_name, {}))


def _eval_calibration_alpha(spec: PipelineSpec) -> float:
    return float(getattr(spec.evaluation, "calibration_alpha", 0.05))


def _eval_subsamples(spec: PipelineSpec) -> Mapping[str, SubsampleWindow] | None:
    subsamples = getattr(spec.evaluation, "subsamples", None)
    return cast("Mapping[str, SubsampleWindow] | None", subsamples or None)


def _parse_window_date(value: str | None) -> pd.Timestamp | None:
    return None if value is None else pd.Timestamp(value).normalize()


def _date_anchor_description(index: pd.DatetimeIndex) -> str:
    if index.empty:
        return "empty"
    normalized = pd.DatetimeIndex(index).dropna()
    if normalized.empty:
        return "empty"
    inferred = pd.infer_freq(normalized) if len(normalized) >= 3 else None
    days = pd.Index(normalized.day)
    if bool((days == 1).all()):
        if inferred and inferred.startswith("QS"):
            return "quarter-start (QS) indexed"
        return "month-start (MS) indexed"
    if bool(pd.Index(normalized.is_month_end).all()):
        if inferred and inferred.startswith("QE"):
            return "quarter-end (QE) indexed"
        return "month-end (ME) indexed"
    if inferred:
        return f"{inferred} indexed"
    start = normalized.min().date()
    end = normalized.max().date()
    return f"dates from {start} to {end}"


def _mask_anchor_advice(mask_index: pd.DatetimeIndex, target_index: pd.DatetimeIndex) -> str:
    mask_anchor = _date_anchor_description(mask_index)
    target_anchor = _date_anchor_description(target_index)
    if "month-end" in mask_anchor and "month-start" in target_anchor:
        return " Reindex the mask to month-start (freq='MS') dates."
    if "month-start" in mask_anchor and "month-end" in target_anchor:
        return " Reindex the mask to month-end (freq='ME') dates."
    if "quarter-end" in mask_anchor and "quarter-start" in target_anchor:
        return " Reindex the mask to quarter-start (freq='QS') dates."
    if "quarter-start" in mask_anchor and "quarter-end" in target_anchor:
        return " Reindex the mask to quarter-end (freq='QE') dates."
    return ""


def _target_mask_frequency(target_dates: pd.DatetimeIndex) -> str:
    if target_dates.empty:
        return "monthly"
    normalized = pd.DatetimeIndex(target_dates).sort_values()
    inferred = pd.infer_freq(normalized) if len(normalized) >= 3 else None
    if inferred and (inferred.startswith("QS") or inferred.startswith("QE")):
        return "quarterly"
    if bool((pd.Index(normalized.day) == 1).all()):
        deltas = [
            int((right - left).days)
            for left, right in zip(normalized[:-1], normalized[1:], strict=False)
            if right > left
        ]
        months = set(int(month) for month in normalized.month)
        if deltas and 80 <= float(pd.Series(deltas).median()) <= 100 and months <= {1, 4, 7, 10}:
            return "quarterly"
    return "monthly"


def _unique_target_dates(dates: pd.Series, *, name: str) -> pd.DatetimeIndex:
    if dates.isna().any():
        count = int(dates.isna().sum())
        raise ValueError(
            f"EvalSpec.subsamples[{name!r}].mask cannot align because forecast "
            f"target dates contain {count} invalid or missing value(s)"
        )
    return pd.DatetimeIndex(pd.unique(dates)).sort_values()


def _series_summary(series: pd.Series) -> dict[str, Any]:
    observed = series.dropna()
    if observed.empty:
        return {"n_obs": 0, "n_true": 0, "first": None, "last": None}
    bool_values = observed.astype(bool)
    return {
        "n_obs": int(observed.shape[0]),
        "n_true": int(bool_values.sum()),
        "first": pd.Timestamp(observed.index[0]).strftime("%Y-%m-%d"),
        "last": pd.Timestamp(observed.index[-1]).strftime("%Y-%m-%d"),
    }


def _user_mask_summary(mask: tuple[tuple[str, bool], ...]) -> dict[str, Any]:
    payload = json.dumps(list(mask), separators=(",", ":"), ensure_ascii=True)
    series = pd.Series(
        [value for _date, value in mask],
        index=pd.DatetimeIndex([date for date, _value in mask]),
        dtype=bool,
    )
    return {
        **_series_summary(series),
        "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }


def _user_mask_series(mask: tuple[tuple[str, bool], ...]) -> pd.Series:
    return pd.Series(
        [value for _date, value in mask],
        index=pd.DatetimeIndex([date for date, _value in mask]),
        dtype=bool,
    )


def _unresolved_named_mask_error(
    name: str, mask_name: str, resolved_source: str | None
) -> ValueError:
    """The error a named mask gets when its data was not resolved beforehand."""
    problem = (
        "no resolved mask was passed for it"
        if resolved_source is None
        else f"the resolved mask passed for it is {resolved_source!r}"
    )
    return ValueError(
        f"EvalSpec.subsamples[{name!r}].mask={mask_name!r} names an indicator "
        f"whose data must be resolved before evaluation, and {problem}. "
        "evaluate() scores a forecast frame and does not load data: "
        "run_pipeline() and rescore() resolve named masks for you, and a direct "
        "evaluate() call needs inputs=resolve_evaluation_inputs(master, spec) "
        "from macroforecast.pipeline."
    )


def _resolve_subsample_state_mask(
    *,
    name: str,
    mask_spec: Any,
    target_dates: pd.DatetimeIndex,
    resolved_masks: dict[str, dict[str, Any]],
    inputs: ResolvedEvaluationInputs | None,
) -> pd.Series:
    if isinstance(mask_spec, str):
        resolved = None if inputs is None else inputs.subsample_masks.get(name)
        # Checking the source, not just presence: inputs resolved for a
        # different spec would otherwise apply one indicator's dates while the
        # report published the other one's name.
        if resolved is None or resolved.mask_source != mask_spec:
            raise _unresolved_named_mask_error(
                name, mask_spec, None if resolved is None else resolved.mask_source
            )
        state = resolved.state
        resolved_masks[name] = {
            "mask_source": resolved.mask_source,
            "mask_summary": dict(resolved.mask_summary),
        }
    else:
        user_mask = cast("tuple[tuple[str, bool], ...]", mask_spec)
        state = _user_mask_series(user_mask)
        resolved_masks[name] = {
            "mask_source": "user_series",
            "mask_summary": _user_mask_summary(user_mask),
        }
    return _align_subsample_state_mask(name=name, state=state, target_dates=target_dates)


def _align_subsample_state_mask(
    *,
    name: str,
    state: pd.Series,
    target_dates: pd.DatetimeIndex,
) -> pd.Series:
    mask_index = pd.DatetimeIndex(state.index).normalize()
    if mask_index.has_duplicates:
        duplicated = mask_index[mask_index.duplicated()].unique()
        sample = ", ".join(ts.strftime("%Y-%m-%d") for ts in duplicated[:3])
        raise ValueError(f"EvalSpec.subsamples[{name!r}].mask has duplicate dates after normalization: {sample}")
    mask_series = pd.Series(state.to_numpy(), index=mask_index).sort_index()
    overlap = pd.DatetimeIndex(mask_series.index).intersection(target_dates)
    if overlap.empty:
        mask_anchor = _date_anchor_description(pd.DatetimeIndex(mask_series.index))
        target_anchor = _date_anchor_description(target_dates)
        advice = _mask_anchor_advice(pd.DatetimeIndex(mask_series.index), target_dates)
        raise ValueError(
            f"EvalSpec.subsamples[{name!r}].mask has no dates in common with "
            f"the forecast target dates: mask is {mask_anchor}; target dates "
            f"are {target_anchor}.{advice}"
        )

    missing = target_dates.difference(pd.DatetimeIndex(mask_series.index))
    if not missing.empty:
        sample = ", ".join(ts.strftime("%Y-%m-%d") for ts in missing[:5])
        raise ValueError(
            f"EvalSpec.subsamples[{name!r}].mask is missing {len(missing)} "
            f"forecast target date(s); first missing dates: {sample}. Extend "
            "or reindex the mask to cover all forecast target dates."
        )

    aligned = mask_series.reindex(target_dates)
    undecidable = aligned.isna()
    if bool(undecidable.any()):
        missing_values = pd.DatetimeIndex(aligned.index[undecidable])
        sample = ", ".join(ts.strftime("%Y-%m-%d") for ts in missing_values[:5])
        raise ValueError(
            f"EvalSpec.subsamples[{name!r}].mask contains NaN for "
            f"{len(missing_values)} covered target date(s); first dates: {sample}. "
            "Fill or drop undecidable dates before evaluating."
        )
    return aligned.astype(bool)


def subsample_provenance(
    spec: PipelineSpec,
    resolved_masks: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any] | None:
    subsamples = _eval_subsamples(spec)
    if subsamples is None:
        return None
    resolved = dict(resolved_masks or {})
    payload: dict[str, Any] = {}
    for name, window in subsamples.items():
        mask_source: str | None
        mask_summary: Mapping[str, Any] | None
        if window.mask is None:
            mask_source = None
            mask_summary = None
        elif isinstance(window.mask, str):
            record = dict(resolved.get(name, {}))
            mask_source = str(record.get("mask_source") or window.mask)
            raw_summary = record.get("mask_summary")
            mask_summary = dict(raw_summary) if isinstance(raw_summary, Mapping) else None
        else:
            user_mask = cast("tuple[tuple[str, bool], ...]", window.mask)
            mask_source = "user_series"
            mask_summary = _user_mask_summary(user_mask)
        payload[name] = {
            "start": window.start,
            "end": window.end,
            "exclude": [list(bounds) for bounds in window.exclude],
            "mask_source": mask_source,
            "mask_summary": mask_summary,
        }
    return payload


def _subsample_mask(
    master: pd.DataFrame,
    name: str,
    window: SubsampleWindow,
    resolved_masks: dict[str, dict[str, Any]],
    inputs: ResolvedEvaluationInputs | None,
) -> pd.Series:
    if _SUBSAMPLE_DATE_COLUMN not in master.columns:
        raise ValueError(
            "EvalSpec.subsamples filters forecast target dates, but the master "
            f"forecast frame has no {_SUBSAMPLE_DATE_COLUMN!r} column"
        )
    dates = pd.to_datetime(master[_SUBSAMPLE_DATE_COLUMN], errors="coerce").dt.normalize()
    mask = pd.Series(True, index=master.index)
    start = _parse_window_date(window.start)
    end = _parse_window_date(window.end)
    if start is not None:
        mask &= dates >= start
    if end is not None:
        mask &= dates <= end
    for raw_ex_start, raw_ex_end in window.exclude:
        ex_start = _parse_window_date(raw_ex_start)
        ex_end = _parse_window_date(raw_ex_end)
        assert ex_start is not None and ex_end is not None
        mask &= ~((dates >= ex_start) & (dates <= ex_end))
    if window.mask is not None:
        target_dates = _unique_target_dates(dates, name=name)
        state = _resolve_subsample_state_mask(
            name=name,
            mask_spec=window.mask,
            target_dates=target_dates,
            resolved_masks=resolved_masks,
            inputs=inputs,
        )
        row_state = dates.map(state)
        if row_state.isna().any():
            count = int(row_state.isna().sum())
            raise ValueError(
                f"EvalSpec.subsamples[{name!r}].mask could not classify "
                f"{count} forecast row(s); check target-date coverage"
            )
        mask &= row_state.astype(bool)
    return mask


def _subsample_frames(
    master: pd.DataFrame,
    spec: PipelineSpec,
    inputs: ResolvedEvaluationInputs | None,
) -> list[tuple[str, pd.DataFrame]] | None:
    subsamples = _eval_subsamples(spec)
    if subsamples is None:
        return None
    resolved_masks: dict[str, dict[str, Any]] = {}
    frames = [
        (name, master.loc[_subsample_mask(master, name, window, resolved_masks, inputs)].copy())
        for name, window in subsamples.items()
    ]
    master.attrs[_SUBSAMPLE_PROVENANCE_ATTR] = subsample_provenance(spec, resolved_masks)
    return frames


def _with_subsample_column(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    out = frame.copy()
    if "subsample" in out.columns:
        out["subsample"] = name
        return out
    insert_at = 1 if "target" in out.columns else 0
    out.insert(insert_at, "subsample", name)
    return out


def _concat_tables(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    return pd.concat(list(frames), ignore_index=True, sort=False)


def _warn_short_subsample(name: str, frame: pd.DataFrame) -> None:
    if frame.empty:
        warnings.warn(
            f"EvalSpec.subsamples[{name!r}] leaves no forecast rows after "
            "target-date filtering.",
            RuntimeWarning,
            stacklevel=3,
        )
        return
    if not _has_groups(frame) or "origin" not in frame.columns:
        return
    cell_counts = (
        frame.drop_duplicates(["target", "horizon", "origin"])
        .groupby(["target", "horizon"], dropna=False)
        .size()
    )
    short = cell_counts[cell_counts < 30]
    if short.empty:
        return
    cells = ", ".join(f"{target} h{horizon}: {int(n)}" for (target, horizon), n in short.head(5).items())
    more = "" if len(short) <= 5 else f" (+{len(short) - 5} more)"
    warnings.warn(
        f"EvalSpec.subsamples[{name!r}] leaves fewer than 30 forecast "
        f"observations for {len(short)} (target, horizon) cell(s): {cells}{more}.",
        RuntimeWarning,
        stacklevel=3,
    )


def _result_row(
    *,
    target: Any,
    horizon: Any,
    contender: str,
    test: str,
    result: Any,
    n_obs: int | None = None,
) -> dict[str, Any]:
    """Long-form significance row for a TestResult-like object."""

    p_value = getattr(result, "p_value", None)
    reject = getattr(result, "decision", None)
    status = "computed"
    reason = None
    metadata = getattr(result, "metadata", {}) or {}
    if test in {"enc_new", "enc_t"} and p_value is None and metadata.get("critical_value") is None:
        reject = None
        status = "inconclusive"
        reason = "no p_value or critical_value available for this nonstandard nested test"
    return {
        "target": target,
        "horizon": horizon,
        "contender": contender,
        "test": test,
        "statistic": getattr(result, "statistic", None),
        "p_value": p_value,
        "reject": reject,
        "n_obs": getattr(result, "n_obs", n_obs),
        "status": status,
        "reason": reason,
    }


def _not_computed_row(
    *,
    target: Any,
    horizon: Any,
    contender: str,
    test: str,
    n_obs: int | None,
    reason: str,
) -> dict[str, Any]:
    """A significance row for a test that was requested and could not be run.

    Same long-form shape as a computed row, with the statistic and p-value left
    NaN: ``status``/``reason`` say what happened, and no number is invented. It
    carries a ``test`` column even where the computed output would have been wide
    (DM/CW), because the row's whole job is to name the test that did not run.
    """
    return {
        "target": target,
        "horizon": horizon,
        "contender": contender,
        "test": test,
        "statistic": np.nan,
        "p_value": np.nan,
        "reject": None,
        "n_obs": n_obs,
        "status": "degraded",
        "reason": reason,
    }


def _pairwise_tests_for_cell(
    contender: str,
    *,
    want_dm: bool,
    want_cw: bool,
    requested_long: Sequence[str],
    custom_loss_blocked: set[str],
    nested: set[str],
) -> list[str]:
    """The pairwise tests this contender's cell WOULD have run, in report order.

    Mirrors the eligibility the main loop applies test by test, so an
    insufficient-sample cell reports exactly the tests it owed the caller --
    no Clark-West row for a contender that never declared nesting, and no row
    for a test a custom loss already ruled out with its own warning.
    """
    tests: list[str] = []
    if want_dm:
        tests.append("dm")
    if want_cw and contender in nested:
        tests.append("cw")
    for name in requested_long:
        if name in custom_loss_blocked:
            continue
        if name in {"enc_new", "enc_t"} and contender not in nested:
            continue
        tests.append(name)
    return tests


def _loss_values(
    y: np.ndarray,
    pred: np.ndarray,
    loss_fn: Callable[[Any, Any], Any] | None,
) -> np.ndarray:
    return (
        np.asarray(loss_fn(y, pred), dtype=float)
        if loss_fn is not None
        else (pred - y) ** 2
    )


def _directional_degenerate_reason(
    y: np.ndarray,
    pred: np.ndarray,
    *,
    threshold: float,
) -> str | None:
    forecast_raw = np.asarray(pred, dtype=float) - float(threshold)
    if forecast_raw.size < 2:
        return "fewer_than_two_observations"
    if float(np.linalg.norm(np.diff(forecast_raw))) <= 1e-12:
        return "constant_forecast"
    forecast_sign = forecast_raw > 0.0
    if np.unique(forecast_sign).size < 2:
        return "constant_forecast_sign"
    return None


def _degenerate_directional_row(
    *,
    target: Any,
    horizon: Any,
    contender: str,
    test: str,
    reason: str,
    n_obs: int,
) -> dict[str, Any]:
    return {
        "target": target,
        "horizon": horizon,
        "contender": contender,
        "test": test,
        "statistic": None,
        "p_value": None,
        "reject": None,
        "n_obs": n_obs,
        "status": "degenerate",
        "reason": reason,
    }


# combination methods that need realised values (estimated weights), and their lag
_ESTIMATED = {
    "inverse_mspe", "dmspe", "best_n", "bates_granger",
    "granger_ramanathan", "constrained_ls", "eigenvector", "regularized",
}

# Pooling rules that are well defined for a single member: the pool of one series IS
# that series. Estimated-weight methods above genuinely need two or more, because the
# weights are fitted across contenders.
_SINGLE_MEMBER_OK = {"mean", "median", "trimmed_mean", "winsorized_mean", "linear_pool"}


def _combine(method: str, frame: pd.DataFrame, y_true: pd.Series, *, horizon: int, **params: Any) -> pd.Series:
    """Dispatch a combination method name to the forecasting combine_* primitives."""
    from macroforecast import forecasting as F

    key = str(method).lower()
    simple = {
        "mean": F.combine_mean, "median": F.combine_median,
        "trimmed_mean": F.combine_trimmed_mean, "winsorized_mean": F.combine_winsorized_mean,
    }
    estimated: dict[str, Callable[..., pd.Series]] = {
        "inverse_mspe": F.combine_inverse_mspe, "dmspe": F.combine_dmspe,
        "best_n": F.combine_best_n, "bates_granger": F.combine_bates_granger,
        "granger_ramanathan": F.combine_granger_ramanathan, "constrained_ls": F.combine_constrained_ls,
        "eigenvector": F.combine_eigenvector, "regularized": F.combine_regularized,
    }
    if key in simple:
        return simple[key](frame, **params)
    if key in estimated:
        return estimated[key](frame, y_true, horizon=int(horizon), **params)
    raise ValueError(f"unsupported combination method {method!r}")


def apply_combinations(master: pd.DataFrame, spec: PipelineSpec) -> pd.DataFrame:
    """Append cross-arm combination contenders to the master forecast frame."""
    # An empty / column-less master (a cell that produced zero rows) has nothing to
    # combine; return it unchanged rather than letting the groupby raise KeyError.
    if not spec.combinations or not _has_groups(master):
        return master
    rows: list[pd.DataFrame] = []
    if "combined" in master.columns:
        base = master.loc[~master["combined"].fillna(False).astype(bool)]
    else:
        base = master
    existing = set(master["contender"].unique()) if "contender" in master.columns else set()
    for combo in spec.combinations:
        if combo.name in existing:
            continue  # already present -> idempotent re-application
        for (target, horizon), group in base.groupby(["target", "horizon"], dropna=False):
            wide = group.pivot_table(index="origin", columns="contender", values="prediction", aggfunc="mean").sort_index()
            actual = group.groupby("origin")["actual"].first().reindex(wide.index)
            date = group.groupby("origin")["date"].first().reindex(wide.index)
            if isinstance(combo.over, (list, tuple)):
                keep = [c for c in combo.over if c in wide.columns]
                if not keep:
                    warnings.warn(
                        f"combination {combo.name!r}: none of its members are present in "
                        f"target={target!r} horizon={horizon!r}; the contender is skipped.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    continue
                wide = wide[keep]
            if wide.shape[1] < 2:
                # A one-member pool used to be dropped here without a word, so the
                # contender simply never appeared and every metric derived from it read
                # NaN with nothing to explain it. A simple pool of one series is that
                # series, so produce it; the estimated-weight rules do need two or more,
                # and those now say so instead of vanishing.
                if str(combo.method).lower() not in _SINGLE_MEMBER_OK:
                    warnings.warn(
                        f"combination {combo.name!r} uses method {combo.method!r}, which "
                        f"estimates weights across contenders and needs at least two "
                        f"members; it has {wide.shape[1]}. The contender is skipped.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    continue
                if wide.shape[1] == 0:
                    continue
            params = dict(combo.params or {})
            if combo.shrink_to_equal is not None and str(combo.method).lower() in _ESTIMATED:
                params.setdefault("shrink_to_equal", combo.shrink_to_equal)
            if combo.weight_window is not None and str(combo.method).lower() in _ESTIMATED:
                params.setdefault("window", combo.weight_window)
            combined = _combine(combo.method, wide, actual, horizon=int(horizon), **params)
            rows.append(pd.DataFrame({
                "arm": combo.name, "model": combo.name, "contender": combo.name,
                "target": target, "horizon": horizon, "origin": wide.index,
                "date": date.to_numpy(), "prediction": combined.to_numpy(),
                "actual": actual.to_numpy(), "combined": True,
            }))
    if not rows:
        return master
    return pd.concat([master, *rows], ignore_index=True)


def _metric_column_name(metric: str | Callable[..., float]) -> str:
    """The accuracy-table column label for a metric: its name, or ``__name__``."""
    if isinstance(metric, str):
        return metric
    name = getattr(metric, "__name__", None)
    return str(name) if name else str(metric)


def _resolve_accuracy_metrics(
    metrics: Sequence[str | Callable[..., float]],
) -> list[tuple[Callable[..., float] | None, str, bool]]:
    """Resolve ``metrics`` to ``(callable_or_None, column_name, is_builtin)``.

    The three benchmark-relative builtins (``rmse``/``relative_mse``/``r2_oos``)
    keep their existing inline formulas whenever they are requested BY NAME (an
    explicit callable named e.g. ``rmse`` would instead go through the generic
    pointwise path below) -- ``callable_or_None`` is ``None`` for those. Every
    other entry is resolved through :func:`macroforecast.metrics.get_metric` up
    front (so an unknown metric name raises immediately, not silently as NaN)
    and applied generically as ``metric(y_true, y_pred) -> float``.
    """
    from macroforecast.metrics import get_metric

    resolved: list[tuple[Callable[..., float] | None, str, bool]] = []
    for metric in metrics:
        name = _metric_column_name(metric)
        is_builtin = isinstance(metric, str) and name in _BUILTIN_ACCURACY_METRICS
        fn = None if is_builtin else get_metric(metric)
        resolved.append((fn, name, is_builtin))
    return resolved


def _accuracy_columns(metric_names: Sequence[str]) -> list[str]:
    return ["target", "horizon", "contender", *metric_names, "n_common", "is_benchmark", "benchmark_present"]


def _point_metrics(spec: PipelineSpec) -> tuple[str | Callable[..., float], ...]:
    """``spec.evaluation.metrics`` minus the density-classified names.

    Density/interval metrics (``crps``/``gaussian_nll``/``qlike``/
    ``pinball_loss``/... -- see :func:`macroforecast.metrics.metric_kind`) need
    ``variance_prediction``/``quantile_predictions`` columns rather than plain
    ``(y_true, y_pred)``, so they are routed to :func:`density_table` and MUST
    NOT be fed through ``accuracy_table``'s generic pointwise path (which would
    call them as ``metric(y_true, y_pred)`` and raise a confusing
    ``TypeError``). Callables always pass through -- the classification is
    name-based.
    """
    from macroforecast.metrics import metric_kind

    return tuple(
        metric
        for metric in _eval_metrics(spec)
        if not (
            isinstance(metric, str)
            and metric_kind(metric) in {"variance", "volatility", "quantile"}
        )
    )


def accuracy_table(master: pd.DataFrame, spec: PipelineSpec) -> pd.DataFrame:
    """Per-``spec.evaluation.metrics`` accuracy per (target, horizon, contender).

    Density-classified metric names in ``spec.evaluation.metrics`` are excluded
    here and scored by :func:`density_table` instead (see :func:`_point_metrics`);
    the point metrics keep their exact prior behavior.
    """
    return _accuracy_against(master, spec.evaluation.benchmark, metrics=_point_metrics(spec))


def _accuracy_against(
    master: pd.DataFrame,
    bench: str,
    *,
    metrics: Sequence[str | Callable[..., float]] = _DEFAULT_ACCURACY_METRICS,
) -> pd.DataFrame:
    """The accuracy table scored against the named benchmark contender.

    Split out of :func:`accuracy_table` so callers that score the same forecasts
    against a benchmark that is not ``spec.evaluation.benchmark`` (e.g. the
    cross-policy common denominator) do not have to fabricate a spec. ``metrics``
    defaults to the historical three-column set (see :data:`_DEFAULT_ACCURACY_METRICS`);
    :func:`evaluate_cross_policy` relies on that default and does not thread a
    ``PipelineSpec`` through, so it is unaffected by ``EvalSpec.metrics``.
    """
    resolved = _resolve_accuracy_metrics(metrics)
    metric_names = [name for _, name, _ in resolved]
    columns = _accuracy_columns(metric_names)
    if not _has_groups(master):
        return pd.DataFrame(columns=columns)
    out: list[dict[str, Any]] = []
    ragged: list[tuple[Any, Any]] = []
    for (target, horizon), group in master.groupby(["target", "horizon"], dropna=False):
        # Per-contender PAIRWISE sample: each contender is scored against the
        # benchmark on the origins where BOTH it and the benchmark (and the
        # realised target) are observed -- NOT the listwise intersection across
        # ALL contenders. A single short-coverage contender (e.g. an arm whose
        # feature block starts late) must not silently truncate every other
        # contender's relRMSE sample and shift the evaluation period. The joint
        # listwise sample is kept only where a joint sample is genuinely required
        # (the Model Confidence Set), so ``n_common`` here is per-contender.
        wide = group.pivot_table(index="origin", columns="contender", values="prediction", aggfunc="mean")
        actual = group.groupby("origin")["actual"].first().reindex(wide.index)
        bench_present = bench in wide.columns
        bench_obs = wide[bench].notna() if bench_present else pd.Series(False, index=wide.index)
        coverage = {c: int((actual.notna() & wide[c].notna()).sum()) for c in wide.columns}
        if bench_present and len(set(coverage.values())) > 1:
            ragged.append((target, horizon))
        for contender in wide.columns:
            mask = actual.notna() & wide[contender].notna()
            if bench_present:
                mask = mask & bench_obs
            n = int(mask.sum())
            if n:
                y = actual.loc[mask].to_numpy(dtype=float)
                pred = wide.loc[mask, contender].to_numpy(dtype=float)
                mse = float(np.mean((pred - y) ** 2))
                bench_mse = (
                    float(np.mean((wide.loc[mask, bench].to_numpy(dtype=float) - y) ** 2))
                    if bench_present else np.nan
                )
            else:
                y = pred = np.asarray([], dtype=float)
                mse = bench_mse = np.nan
            ok = np.isfinite(mse) and np.isfinite(bench_mse) and bench_mse > 0
            row: dict[str, Any] = {"target": target, "horizon": horizon, "contender": contender}
            for fn, name, is_builtin in resolved:
                value: float
                if is_builtin:
                    if name == "rmse":
                        value = float(np.sqrt(mse)) if np.isfinite(mse) else np.nan
                    elif name == "relative_mse":
                        value = (mse / bench_mse) if ok else np.nan
                    else:  # "r2_oos"
                        value = (1.0 - mse / bench_mse) if ok else np.nan
                elif n and fn is not None:
                    value = float(fn(y, pred))
                else:
                    value = np.nan
                row[name] = value
            row["n_common"] = n
            row["is_benchmark"] = contender == bench
            row["benchmark_present"] = bench_present
            out.append(row)
    if ragged:
        cells = ", ".join(f"{t} h{h}" for t, h in ragged[:5])
        more = "" if len(ragged) <= 5 else f" (+{len(ragged) - 5} more)"
        warnings.warn(
            f"ragged forecast coverage across contenders in {len(ragged)} (target, horizon) "
            f"cell(s): {cells}{more}. relRMSE/RMSE/R2 use each contender's PAIRWISE sample "
            f"with the benchmark, so n_common is per-contender; a short-coverage arm no "
            f"longer truncates the others.",
            RuntimeWarning, stacklevel=2,
        )
    return pd.DataFrame(out, columns=columns)


def _finite_float(value: Any) -> float:
    """Coerce a test statistic / p-value to float, returning NaN when it is ``None``
    or otherwise not a real number. A Diebold-Mariano / Clark-West statistic is ``None``
    for a DEGENERATE pair -- most commonly two IDENTICAL forecasts (zero loss
    differential, so the test is undefined) -- and must be recorded as NaN
    (not-significant) so the significance table is still produced with every other pair
    intact, rather than crashing on ``float(None)``.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(np.nan)


def significance_table(master: pd.DataFrame, spec: PipelineSpec) -> pd.DataFrame:
    """Forecast-comparison tests of each contender vs the benchmark.

    The historical ``"dm"``/``"cw"`` output remains wide for byte-identity of
    default reports. Newly wired pairwise tests append long-form rows with a
    ``test`` column: ``"gw"``, ``"enc_new"``, ``"enc_t"``, ``"gr"``, plus
    ``"pt"``, ``"hm"``, and ``"ag"``. The directional rows test the
    CONTENDER's own sign skill, not a pairwise loss differential, but they use
    the same benchmark-aligned sample as DM/CW so all significance rows rest on
    the same available origins.

    Loss-differential tests use ``spec.evaluation.loss(y_true, y_pred)`` when
    set, else squared error. Clark-West, ENC-NEW, and ENC-T are derived under
    quadratic-loss nested-model assumptions, so a custom loss makes those tests
    skip with a ``UserWarning`` instead of silently computing an invalid result.
    """
    from macroforecast.tests import (
        anatolyev_gerko_test,
        clark_west_test,
        conditional_predictive_ability_test,
        dm_test,
        enc_new_test,
        enc_t_test,
        giacomini_white_test,
        henriksson_merton_test,
        mincer_zarnowitz_test,
        multi_horizon_spa_test,
        pesaran_timmermann_test,
    )

    bench = spec.evaluation.benchmark
    if not _has_groups(master):
        return pd.DataFrame(columns=_SIGNIFICANCE_COLUMNS)

    requested = _eval_tests(spec)
    want_dm = "dm" in requested
    want_cw = "cw" in requested and bool(spec.evaluation.cw_for_nested)
    requested_long = tuple(name for name in requested if name in _PAIRWISE_LONG_TESTS)
    requested_joint = tuple(name for name in requested if name in _MULTI_HORIZON_TESTS)
    if not want_dm and not want_cw and not requested_long and not requested_joint:
        return pd.DataFrame(columns=_SIGNIFICANCE_COLUMNS)

    # Clark-West is only licensed where the benchmark is nested in the contender.
    # Build the set of contenders whose arm declares nesting; CW is emitted only
    # for those. Non-nested contenders get DM only (CW would be an invalid test).
    nested: set[str] = set()
    for a in spec.arms:
        if getattr(a, "nested_in_benchmark", False):
            nested |= set(contender_names(a))
    # A combination is a contender too, and in the forecast-combination
    # literature CW on the combination is the headline test. It must declare
    # nesting itself -- see CombinationContender.nested_in_benchmark.
    combinations = tuple(getattr(spec, "combinations", ()) or ())
    declared_combinations = {
        str(c.name) for c in combinations if getattr(c, "nested_in_benchmark", False)
    }
    nested |= declared_combinations
    if want_cw and nested and combinations and not declared_combinations:
        warnings.warn(
            "Clark-West is requested and some arms declare nested_in_benchmark, "
            f"but none of the {len(combinations)} forecast combination(s) do, so "
            "their cw_stat/cw_p will be NaN. Set "
            "CombinationContender(nested_in_benchmark=True) on a combination whose "
            "members nest the benchmark (a simple pool of nested arms does).",
            UserWarning, stacklevel=2,
        )

    loss_fn = _eval_loss(spec)
    custom_loss_blocked = {
        name
        for name in _NESTED_QUADRATIC_TESTS
        if name in requested
        and loss_fn is not None
        and nested
        and (name != "cw" or want_cw)
    }
    cw_blocked_by_custom_loss = "cw" in custom_loss_blocked
    if custom_loss_blocked:
        labels = {
            "cw": "Clark-West",
            "enc_new": "ENC-NEW",
            "enc_t": "ENC-T",
        }
        skipped = ", ".join(labels[name] for name in sorted(custom_loss_blocked))
        warnings.warn(
            "EvalSpec.loss is a custom per-observation loss, but "
            f"{skipped} is derived under quadratic loss and is not a valid "
            f"test for an arbitrary loss function; skipping {skipped} for this "
            "evaluation (loss-differential tests that accept arbitrary losses "
            "still run under the custom loss).",
            UserWarning, stacklevel=2,
        )

    out: list[dict[str, Any]] = []
    dm_options = _eval_test_options(spec, "dm")
    cw_options = _eval_test_options(spec, "cw")
    loss_diffs: dict[tuple[Any, Any], dict[str, pd.Series]] = {}
    for (target, horizon), group in master.groupby(["target", "horizon"], dropna=False):
        pivot_f = group.pivot_table(index="origin", columns="contender", values="prediction", aggfunc="mean")
        actual = group.groupby("origin")["actual"].first().reindex(pivot_f.index)
        if bench not in pivot_f.columns:
            continue
        fb = pivot_f[bench]
        for contender in pivot_f.columns:
            if contender == bench:
                continue
            fc = pivot_f[contender]
            common = actual.notna() & fb.notna() & fc.notna()
            n_common = int(common.sum())
            if n_common < _MIN_PAIRWISE_ORIGINS:
                pairwise_reason = (
                    f"insufficient paired sample: {n_common} origin(s) where the "
                    f"contender, the benchmark {bench!r} and the realised target are "
                    f"all observed, below the {_MIN_PAIRWISE_ORIGINS} required for a "
                    "pairwise forecast-comparison test; not computed."
                )
                out.extend(
                    _not_computed_row(
                        target=target, horizon=horizon, contender=str(contender),
                        test=test_name, n_obs=n_common, reason=pairwise_reason,
                    )
                    for test_name in _pairwise_tests_for_cell(
                        str(contender),
                        want_dm=want_dm,
                        want_cw=want_cw and not cw_blocked_by_custom_loss,
                        requested_long=requested_long,
                        custom_loss_blocked=custom_loss_blocked,
                        nested=nested,
                    )
                )
                continue
            y = actual[common].to_numpy(float)
            fb_vals = fb[common].to_numpy(float)
            fc_vals = fc[common].to_numpy(float)
            loss_b = _loss_values(y, fb_vals, loss_fn)
            loss_c = _loss_values(y, fc_vals, loss_fn)
            # Kept for the multiple-testing pass below, which needs the family's
            # differentials JOINTLY (Romano-Wolf resamples them together) and
            # cannot recover them from the p-values alone. Oriented benchmark
            # minus contender, so a positive mean means the contender is better.
            loss_diffs.setdefault((target, horizon), {})[str(contender)] = pd.Series(
                np.asarray(loss_b, dtype=float) - np.asarray(loss_c, dtype=float),
                index=pd.Index(actual[common].index, name="origin"),
            )
            row: dict[str, Any] = {"target": target, "horizon": horizon, "contender": contender}
            has_result = False
            degraded_reasons: list[str] = []
            if want_dm:
                try:
                    dm = dm_test(
                        loss_c,
                        loss_b,
                        **{**dm_options, "horizon": int(horizon), "input_type": "loss"},
                    )
                    row["dm_stat"] = _finite_float(dm.statistic); row["dm_p"] = _finite_float(dm.p_value)
                except _DEGRADATION_EXCEPTIONS as exc:
                    row["dm_stat"] = np.nan; row["dm_p"] = np.nan
                    degraded_reasons.append(f"dm failed: {type(exc).__name__}: {exc}")
                has_result = True
            if want_cw and not cw_blocked_by_custom_loss and contender in nested:
                try:
                    cw = clark_west_test(
                        loss_b,
                        loss_c,
                        fb_vals,
                        fc_vals,
                        **{**cw_options, "horizon": int(horizon)},
                    )
                    row["cw_stat"] = _finite_float(cw.statistic); row["cw_p"] = _finite_float(cw.p_value)
                except _DEGRADATION_EXCEPTIONS as exc:
                    row["cw_stat"] = np.nan; row["cw_p"] = np.nan
                    degraded_reasons.append(f"cw failed: {type(exc).__name__}: {exc}")
                has_result = True
            if has_result:
                if degraded_reasons:
                    row["status"] = "degraded"
                    row["reason"] = "; ".join(degraded_reasons)
                out.append(row)
            n_obs = int(common.sum())
            for test_name in requested_long:
                if test_name in custom_loss_blocked:
                    continue
                if test_name in {"enc_new", "enc_t"} and contender not in nested:
                    continue
                if test_name == "gw":
                    res = giacomini_white_test(
                        loss_c,
                        loss_b,
                        **{
                            **_eval_test_options(spec, "gw"),
                            "horizon": int(horizon),
                        },
                    )
                    out.append(_result_row(
                        target=target, horizon=horizon, contender=str(contender),
                        test="gw", result=res, n_obs=n_obs,
                    ))
                elif test_name == "enc_new":
                    res = enc_new_test(
                        fb_vals - y,
                        fc_vals - y,
                        **_eval_test_options(spec, "enc_new"),
                    )
                    out.append(_result_row(
                        target=target, horizon=horizon, contender=str(contender),
                        test="enc_new", result=res, n_obs=n_obs,
                    ))
                elif test_name == "enc_t":
                    res = enc_t_test(
                        fb_vals - y,
                        fc_vals - y,
                        **{
                            **_eval_test_options(spec, "enc_t"),
                            "horizon": int(horizon),
                        },
                    )
                    out.append(_result_row(
                        target=target, horizon=horizon, contender=str(contender),
                        test="enc_t", result=res, n_obs=n_obs,
                    ))
                elif test_name == "pt":
                    options = _eval_test_options(spec, "pt")
                    reason = _directional_degenerate_reason(
                        y, fc_vals, threshold=float(options.get("threshold", 0.0))
                    )
                    if reason is not None:
                        out.append(_degenerate_directional_row(
                            target=target, horizon=horizon, contender=str(contender),
                            test="pt", reason=reason, n_obs=n_obs,
                        ))
                        continue
                    res = pesaran_timmermann_test(
                        y,
                        fc_vals,
                        **{**options, "method": "pesaran_timmermann"},
                    )
                    out.append(_result_row(
                        target=target, horizon=horizon, contender=str(contender),
                        test="pt", result=res, n_obs=n_obs,
                    ))
                elif test_name == "hm":
                    options = _eval_test_options(spec, "hm")
                    reason = _directional_degenerate_reason(
                        y, fc_vals, threshold=float(options.get("threshold", 0.0))
                    )
                    if reason is not None:
                        out.append(_degenerate_directional_row(
                            target=target, horizon=horizon, contender=str(contender),
                            test="hm", reason=reason, n_obs=n_obs,
                        ))
                        continue
                    res = henriksson_merton_test(
                        y,
                        fc_vals,
                        **{**options, "method": "henriksson_merton"},
                    )
                    out.append(_result_row(
                        target=target, horizon=horizon, contender=str(contender),
                        test="hm", result=res, n_obs=n_obs,
                    ))
                elif test_name == "ag":
                    options = _eval_test_options(spec, "ag")
                    reason = _directional_degenerate_reason(
                        y, fc_vals, threshold=float(options.get("threshold", 0.0))
                    )
                    if reason is not None:
                        out.append(_degenerate_directional_row(
                            target=target, horizon=horizon, contender=str(contender),
                            test="ag", reason=reason, n_obs=n_obs,
                        ))
                        continue
                    res = anatolyev_gerko_test(
                        y,
                        fc_vals,
                        **{**options, "method": "anatolyev_gerko"},
                    )
                    out.append(_result_row(
                        target=target, horizon=horizon, contender=str(contender),
                        test="ag", result=res, n_obs=n_obs,
                    ))
                elif test_name == "gr":
                    options = _eval_test_options(spec, "gr")
                    default_lag_truncate = min(max(int(horizon) - 1, 0), 5)
                    has_hac_lags = "hac_lags" in options
                    has_lag_truncate = "lag_truncate" in options
                    lag_truncate_source = (
                        "user" if has_hac_lags or has_lag_truncate else "default_min_h_minus_1_5"
                    )
                    if has_hac_lags:
                        options.pop("lag_truncate", None)
                    else:
                        options.setdefault("lag_truncate", default_lag_truncate)
                    gr = conditional_predictive_ability_test(
                        loss_c,
                        loss_b,
                        **{
                            **options,
                            "method": "giacomini_rossi",
                        },
                    )
                    out.append({
                        "target": target,
                        "horizon": horizon,
                        "contender": str(contender),
                        "test": "gr",
                        "statistic": gr.get("statistic"),
                        "p_value": None,
                        "reject": gr.get("decision"),
                        "n_obs": gr.get("n_obs"),
                        "critical_value": gr.get("critical_value"),
                        "window_size": gr.get("window_size"),
                        "lag_truncate": gr.get("lag_truncate"),
                        "hac_lags": gr.get("hac_lags", gr.get("lag_truncate")),
                        "lag_truncate_source": lag_truncate_source,
                        "status": "computed",
                    })
                elif test_name == "mz":
                    options = _eval_test_options(spec, "mz")
                    default_hac_lags = max(int(horizon) - 1, 0)
                    hac_lag_source = "user" if "hac_lags" in options else "default_h_minus_1"
                    options.setdefault("hac_lags", default_hac_lags)
                    res = mincer_zarnowitz_test(
                        y,
                        fc_vals,
                        **options,
                    )
                    row = _result_row(
                        target=target, horizon=horizon, contender=str(contender),
                        test="mz", result=res, n_obs=n_obs,
                    )
                    row.update({
                        "intercept": res.metadata.get("intercept"),
                        "slope": res.metadata.get("slope"),
                        "hac_lags": res.metadata.get("hac_lags"),
                        "hac_lag_source": hac_lag_source,
                    })
                    out.append(row)
    if requested_joint:
        for target, target_group in master.groupby("target", dropna=False):
            contenders = [
                contender
                for contender in target_group["contender"].dropna().unique()
                if contender != bench
            ]
            for contender in contenders:
                horizon_diffs: dict[Any, pd.Series] = {}
                # What each horizon actually offered, kept whether or not it
                # cleared the bar: the reason below has to name the evidence, and
                # "3 origins at h=1, 3 at h=2" is what makes the failure fixable.
                observed: dict[Any, int] = {}
                for horizon, group in target_group.groupby("horizon", dropna=False):
                    pivot_f = group.pivot_table(
                        index="origin", columns="contender", values="prediction", aggfunc="mean"
                    )
                    if bench not in pivot_f.columns or contender not in pivot_f.columns:
                        observed[horizon] = 0
                        continue
                    actual = group.groupby("origin")["actual"].first().reindex(pivot_f.index)
                    fb = pivot_f[bench]
                    fc = pivot_f[contender]
                    common = actual.notna() & fb.notna() & fc.notna()
                    observed[horizon] = int(common.sum())
                    if int(common.sum()) < _MIN_JOINT_ORIGINS:
                        continue
                    y = actual[common].to_numpy(float)
                    loss_b = _loss_values(y, fb[common].to_numpy(float), loss_fn)
                    loss_c = _loss_values(y, fc[common].to_numpy(float), loss_fn)
                    horizon_diffs[horizon] = pd.Series(
                        loss_b - loss_c,
                        index=actual[common].index,
                        name=horizon,
                    )
                per_horizon = ", ".join(
                    f"h={key}: {count}"
                    for key, count in sorted(observed.items(), key=lambda kv: str(kv[0]))
                ) or "no horizons present"
                shortfall: str | None = None
                if len(horizon_diffs) < _MIN_JOINT_HORIZONS:
                    shortfall = (
                        f"only {len(horizon_diffs)} horizon(s) reached the "
                        f"{_MIN_JOINT_ORIGINS} common origins a horizon needs, and a "
                        f"joint test needs at least {_MIN_JOINT_HORIZONS} such "
                        f"horizons (observed common origins -- {per_horizon})"
                    )
                    n_aligned = None
                else:
                    diff_panel = pd.concat(
                        horizon_diffs.values(), axis=1
                    ).dropna(axis=0, how="any")
                    n_aligned = int(diff_panel.shape[0])
                    if diff_panel.shape[0] < _MIN_JOINT_ORIGINS or diff_panel.shape[1] < _MIN_JOINT_HORIZONS:
                        # Each horizon cleared the bar on its own, but the joint
                        # test needs ONE sample of origins shared by all of them.
                        shortfall = (
                            f"{n_aligned} origin(s) common to all "
                            f"{int(diff_panel.shape[1])} horizon(s) after alignment, "
                            f"below the {_MIN_JOINT_ORIGINS} required (per-horizon "
                            f"common origins -- {per_horizon})"
                        )
                if shortfall is not None:
                    out.extend(
                        _not_computed_row(
                            target=target, horizon="joint", contender=str(contender),
                            test=test_name, n_obs=n_aligned,
                            reason=f"insufficient joint sample: {shortfall}; not computed.",
                        )
                        for test_name in requested_joint
                    )
                    continue
                for test_name in requested_joint:
                    res = multi_horizon_spa_test(
                        diff_panel,
                        **{
                            **_eval_test_options(spec, test_name),
                            "statistic": test_name,
                        },
                    )
                    out.append({
                        "target": target,
                        "horizon": "joint",
                        "contender": str(contender),
                        "test": test_name,
                        "statistic": res.statistic,
                        "p_value": res.p_value,
                        "reject": res.decision,
                        "n_obs": res.n_obs,
                        "critical_value": res.metadata.get("critical_value"),
                        "n_horizons": res.metadata.get("n_horizons"),
                        "horizons": res.metadata.get("horizons"),
                    })
    frame = pd.DataFrame(out) if out else pd.DataFrame(columns=_SIGNIFICANCE_COLUMNS)
    return _apply_multiple_testing(frame, spec, loss_diffs)


def _apply_multiple_testing(
    frame: pd.DataFrame,
    spec: PipelineSpec,
    loss_diffs: Mapping[tuple[Any, Any], Mapping[str, pd.Series]],
) -> pd.DataFrame:
    """Append ``<test>_p_adj`` / ``p_value_adj`` for ``EvalSpec.multiple_testing``.

    The family is the set of CONTENDERS within one ``(target, horizon)`` cell:
    that is what a reader compares at once when they scan a column of a results
    table looking for the winner, so that is the family an adjustment has to
    control. Cells are adjusted independently of one another.

    Long-form rows carry their test name in a ``test`` column and share one
    ``p_value`` column, so their family is ``(target, horizon, test)`` -- one
    test's contenders, not a pool of Giacomini-White and Pesaran-Timmermann
    p-values that answer different questions. They land in ``p_value_adj``.
    Before this, ``multiple_testing`` looked only for wide ``<test>_p`` columns,
    so a request for long-form-only tests (``tests=("gw",)``) silently produced
    no adjustment at all -- the one case where a caller who asked for
    multiplicity control got none and was not told.

    ``romano_wolf`` is the exception for long-form rows, and the reason is what
    this function has in hand -- not a claim about which tests use loss
    differentials. ``loss_diffs`` retains one series per contender, benchmark
    loss minus contender loss, and that is the only resampling input available
    here. It cannot reconstruct a long-form test's OWN statistic and null:
    Giacomini-White is itself a loss-differential test, but its statistic is a
    conditional one built on instruments this function never sees, and
    Mincer-Zarnowitz and the directional tests are not loss-differential
    statistics at all. Producing a ``p_value_adj`` for any of them from the
    retained series would report a step-down p-value for a statistic that was
    never resampled, so those rows keep it NaN and one warning names them.

    The wide ``dm_p_adj``/``cw_p_adj`` path is PRESERVED here, not extended and
    not corrected, and nothing in this docstring should be read as asserting it
    is exact. Under ``romano_wolf`` both wide families are resampled from the
    same retained loss panel, so ``cw_p_adj`` is not derived from the Clark-West
    statistic itself. Whether that pre-existing path is statistically aligned
    with each wide family's own statistic -- Clark-West in particular -- is a
    separate, separately confirmed finding, deliberately out of scope here.

    Returns ``frame`` unchanged when no method is configured, so the default
    report stays byte-identical.
    """

    method = getattr(spec.evaluation, "multiple_testing", None)
    if method is None or frame.empty:
        return frame
    from macroforecast.tests import adjust_pvalues, romano_wolf_pvalues

    key = str(method).strip().lower()
    wide_tests = [c[: -len("_p")] for c in frame.columns if c.endswith("_p") and c != "p_value"]
    long_form = {"test", "p_value"} <= set(frame.columns)
    if not wide_tests and not long_form:
        return frame

    for test in wide_tests:
        frame[f"{test}_p_adj"] = np.nan

    grouped = frame.groupby(["target", "horizon"], dropna=False)
    for (target, horizon), block in grouped:
        idx = block.index
        for test in wide_tests:
            raw = block[f"{test}_p"].to_numpy(dtype=float)
            if key == "romano_wolf":
                # One panel per cell -- the benchmark-minus-contender loss
                # series -- reused for every wide family, so `cw_p_adj` comes
                # from the same resampling as `dm_p_adj` rather than from the
                # Clark-West statistic. Preserved as-is here; aligning it is a
                # separate finding (see this function's docstring).
                cell = loss_diffs.get((target, horizon), {})
                names = block["contender"].astype(str).to_numpy()
                # The family is exactly the contenders this test actually
                # produced a p-value for. Romano-Wolf could compute one from the
                # differentials alone, but a `<test>_p_adj` beside an empty
                # `<test>_p` reads as an adjustment of something that was never
                # tested -- and those columns would also inflate the max
                # statistic that everyone else is charged against. All four
                # methods therefore adjust the same set.
                members = [
                    name for name, p_raw in zip(names, raw, strict=True)
                    if np.isfinite(p_raw) and name in cell
                ]
                if not members:
                    continue
                panel = pd.DataFrame({name: cell[name] for name in members})
                rw = romano_wolf_pvalues(panel, **_eval_test_options(spec, "romano_wolf"))
                by_name = dict(zip(panel.columns, rw, strict=True))
                adjusted = np.array(
                    [by_name.get(name, np.nan) for name in names], dtype=float
                )
                adjusted[~np.isfinite(raw)] = np.nan
            else:
                adjusted = adjust_pvalues(raw, method=key)
            frame.loc[idx, f"{test}_p_adj"] = adjusted
    if long_form:
        _adjust_long_form(frame, key=key)
    return frame


def _adjust_long_form(frame: pd.DataFrame, *, key: str) -> None:
    """Fill ``p_value_adj`` for long-form rows, in place, or say why it is empty."""

    from macroforecast.tests import adjust_pvalues

    frame["p_value_adj"] = np.nan
    rows = frame.loc[frame["test"].notna()]
    if rows.empty:
        return
    if key == "romano_wolf":
        # Named only where there was something to adjust: a test whose p-value is
        # NaN everywhere (``gr`` reports a critical value instead) is not being
        # denied an adjustment, so warning about it would be noise.
        unadjusted = sorted(
            {
                str(name)
                for name, block in rows.groupby("test", dropna=True)
                if np.isfinite(pd.to_numeric(block["p_value"], errors="coerce")).any()
            }
        )
        if unadjusted:
            warnings.warn(
                "multiple_testing='romano_wolf' can only resample the "
                "benchmark-minus-contender loss series the pipeline retains, "
                "which cannot reconstruct the test-specific statistic and null "
                f"of long-form test(s) {unadjusted}, so their p_value_adj is NOT "
                "computed and is left NaN rather than reported for a statistic "
                "that was never resampled. The existing wide dm_p_adj/cw_p_adj "
                "path is unaffected. Use 'holm', 'bonferroni', or 'bh' to adjust "
                "these long-form tests.",
                RuntimeWarning,
                stacklevel=2,
            )
        return
    for _, block in rows.groupby(["target", "horizon", "test"], dropna=False):
        raw = pd.to_numeric(block["p_value"], errors="coerce").to_numpy(dtype=float)
        # ``adjust_pvalues`` carries NaN through and excludes it from N, so a
        # degraded or inconclusive row neither receives an adjustment nor
        # inflates the family every other contender is charged against.
        frame.loc[block.index, "p_value_adj"] = adjust_pvalues(raw, method=key)



def mcs_table(master: pd.DataFrame, spec: PipelineSpec, *, n_boot: int = 499) -> pd.DataFrame:
    """Set-comparison results per (target, horizon).

    The historical MCS membership rows run only when ``"mcs"`` is requested and
    keep their old four-column shape when no other set-comparison test is
    requested. ``"spa"``, ``"rc"``, and ``"stepm"`` reuse the same long loss
    panel and append rows to this report table with a ``test`` column.

    The loss matrix is ``spec.evaluation.loss(y_true, y_pred)`` when set, else
    squared error (the prior, only behavior). These set comparisons are
    loss-agnostic as long as the loss is a per-observation scalar series.
    """
    from macroforecast.tests import (
        model_confidence_set,
        reality_check_test,
        stepm_test,
        superior_predictive_ability_test,
    )

    requested = _eval_tests(spec)
    want_mcs = "mcs" in requested
    requested_set = tuple(name for name in requested if name in _SET_COMPARISON_TESTS)
    if not want_mcs and not requested_set:
        return pd.DataFrame(columns=_MCS_COLUMNS)
    # An empty / column-less master has no scorable cells; the required columns
    # (actual/prediction as well as the group keys) may be entirely absent.
    if not _has_groups(master) or not {"actual", "prediction", "contender"}.issubset(master.columns):
        return pd.DataFrame(columns=_MCS_COLUMNS)
    panel = master.copy()
    panel = panel[panel["actual"].notna() & panel["prediction"].notna()]
    if panel.empty:
        return pd.DataFrame(columns=_MCS_COLUMNS)
    loss_fn = _eval_loss(spec)
    actual_vals = panel["actual"].astype(float).to_numpy()
    pred_vals = panel["prediction"].astype(float).to_numpy()
    loss_values = (
        np.asarray(loss_fn(actual_vals, pred_vals), dtype=float)
        if loss_fn is not None
        else (pred_vals - actual_vals) ** 2
    )
    panel = panel.assign(squared_error=loss_values, model_id=panel["contender"])
    out: list[dict[str, Any]] = []
    panel_columns = ["target", "horizon", "origin", "model_id", "squared_error"]
    loss_panel = panel[panel_columns]
    if want_mcs:
        mcs_options = dict(_eval_test_options(spec, "mcs"))
        mcs_options.setdefault("alpha", spec.evaluation.mcs_alpha)
        mcs_options.setdefault("n_boot", n_boot)
        mcs_options.setdefault("random_state", int(spec.seed or 0))
        mcs_options.update(
            {
                "loss": "squared_error",
                "model": "model_id",
                "origin": "origin",
                "target": "target",
                "horizon": "horizon",
            }
        )
        for (target, horizon), group in panel.groupby(["target", "horizon"], dropna=False):
            if group["model_id"].nunique() < 2:
                continue
            try:
                res = model_confidence_set(
                    group[["origin", "model_id", "squared_error"]],
                    **mcs_options,
                )
            except _DEGRADATION_EXCEPTIONS as exc:
                reason = f"mcs failed: {type(exc).__name__}: {exc}"
                for contender in group["model_id"].unique():
                    out.append({
                        "target": target,
                        "horizon": horizon,
                        "contender": contender,
                        "in_mcs": np.nan,
                        "status": "degraded",
                        "reason": reason,
                    })
                continue
            included = set()
            for entry in (res.get("mcs_inclusion") or []):
                included |= set(entry.get("models", []))
            for contender in group["model_id"].unique():
                out.append({
                    "target": target, "horizon": horizon, "contender": contender,
                    "in_mcs": contender in included,
                })
    set_functions = {
        "spa": superior_predictive_ability_test,
        "rc": reality_check_test,
        "stepm": stepm_test,
    }
    for test_name in requested_set:
        options = dict(_eval_test_options(spec, test_name))
        options.setdefault("random_state", int(spec.seed or 0))
        options.update(
            {
                "benchmark": spec.evaluation.benchmark,
                "loss": "squared_error",
                "model": "model_id",
                "origin": "origin",
                "target": "target",
                "horizon": "horizon",
            }
        )
        res = set_functions[test_name](loss_panel, **options)
        records = res.get("records") or []
        for record in records:
            target = record.get("target")
            horizon = record.get("horizon")
            group = loss_panel[
                (loss_panel["target"] == target)
                & (loss_panel["horizon"] == horizon)
            ]
            contenders = [
                str(name)
                for name in group["model_id"].dropna().unique()
                if str(name) != spec.evaluation.benchmark
            ]
            superior = {str(name) for name in (record.get("superior_models") or [])}
            mean_diff = record.get("mean_loss_difference") or {}
            for contender in contenders:
                out.append({
                    "target": target,
                    "horizon": horizon,
                    "contender": contender,
                    "test": test_name,
                    "benchmark": record.get("benchmark", spec.evaluation.benchmark),
                    "superior": contender in superior,
                    "reject": record.get("decision"),
                    "p_value": record.get("p_value"),
                    "n_obs": record.get("n_obs"),
                    "n_models": record.get("n_models"),
                    "mean_loss_difference": mean_diff.get(contender),
                    "status": record.get("status"),
                })
    return pd.DataFrame(out) if out else pd.DataFrame(columns=_MCS_COLUMNS)


# --------------------------------------------------------------------------- #
# Density / interval accuracy (Phase 1 density pipeline)
# --------------------------------------------------------------------------- #
# Wiring note: forecasting.run() ALREADY emits ``variance_prediction`` (a plain
# float) and ``quantile_predictions`` (a ``{level: value}`` mapping) on every
# forecast row for the direct policy, when the fitted model exposes
# ``predict_variance``/``predict_quantiles`` (see
# ``forecasting/policies/base.py::_variance_series``/``_quantile_frame``, called
# from ``forecasting/policies/direct.py``; other policies emit explicit
# ``None`` for both -- Phase 1 scope). ``macroforecast.metrics`` already has a
# full, registry-driven table-level evaluator for exactly these columns
# (:func:`macroforecast.metrics.evaluate_forecasts`, with ``crps``/
# ``gaussian_nll``/``log_score``/``negative_log_score``/``qlike`` and the
# pinball/coverage/interval-width/interval-score quantile bundle already in its
# metric registry). ``density_table`` below is a thin, EvalSpec-gated wrapper
# around that existing evaluator -- not a reimplementation -- so a requested
# density metric with no matching column raises the SAME actionable
# ``ValueError`` :func:`macroforecast.metrics.evaluate_forecasts` already
# raises today.


def _density_metric_names(spec: PipelineSpec) -> list[str]:
    """The subset of ``spec.evaluation.metrics`` that needs a distributional
    column (``variance_prediction``/``quantile_predictions``) rather than plain
    ``(y_true, y_pred)`` -- see :func:`macroforecast.metrics.metric_kind`.
    Callables are never density metrics (classification is name-based).
    """
    from macroforecast.metrics import metric_kind

    return [
        name
        for name in _eval_metrics(spec)
        if isinstance(name, str) and metric_kind(name) in {"variance", "volatility", "quantile"}
    ]


def density_table(master: pd.DataFrame, spec: PipelineSpec) -> pd.DataFrame:
    """Density/interval accuracy per (target, horizon, contender).

    Gated entirely by ``spec.evaluation.metrics``: a default ``EvalSpec``
    (``("rmse", "relative_mse", "r2_oos")``) requests no density metric, so this
    returns an empty frame WITHOUT ever inspecting ``master``'s columns -- a
    point-only run and a run with an unrequested variance-emitting model both
    produce the identical empty frame, matching ``accuracy_table``'s existing
    gating convention (see ``mcs_table`` for the same "not requested -> empty"
    shape).

    When at least one density metric is requested, delegates to
    :func:`macroforecast.metrics.evaluate_forecasts` grouped by
    ``(target, horizon, contender)``. That function computes variance-density
    metrics (``gaussian_nll``/``crps``) as a bundle whenever
    ``variance_prediction`` is present, and the quantile bundle (per-level
    ``pinball_loss_<level>`` plus ``coverage_rate_*``/``interval_width_*``/
    ``interval_score_*`` for every matched symmetric quantile pair) whenever
    ``quantile_predictions`` is present -- regardless of exactly which density
    metric name triggered the call. Requesting ANY one member of a bundle
    computes the whole bundle; this mirrors ``evaluate_forecasts``'s own
    pre-existing behavior (not new here) and is documented rather than
    surgically narrowed, since the two are computed from the same columns in
    one pass.
    """
    requested = _density_metric_names(spec)
    if not requested or not _has_groups(master):
        return pd.DataFrame(columns=_DENSITY_COLUMNS)
    if "contender" not in master.columns:
        return pd.DataFrame(columns=_DENSITY_COLUMNS)

    from macroforecast.metrics import evaluate_forecasts

    return evaluate_forecasts(
        master,
        by=_DENSITY_GROUP_BY,
        metrics=requested,
        actual="actual",
        prediction="prediction",
    )


def _eval_calibration_tests(spec: PipelineSpec) -> list[str]:
    return [name for name in _eval_tests(spec) if name in CALIBRATION_EVAL_TESTS]


def _quantile_dict_levels(value: Any) -> dict[float, float] | None:
    """Parse one ``quantile_predictions`` cell into ``{level: prediction}``
    floats, or ``None`` when the value is missing/not a mapping."""
    if not isinstance(value, dict):
        return None
    try:
        return {float(level): float(pred) for level, pred in value.items()}
    except (TypeError, ValueError):
        return None


def _widest_symmetric_quantile_pair(
    group: pd.DataFrame, *, quantile_predictions: str = "quantile_predictions"
) -> tuple[float, float] | None:
    """The widest symmetric ``(lower, upper)`` quantile-level pair (``lower +
    upper == 1``) common to every non-null row of ``group[quantile_predictions]``,
    or ``None`` when the column is absent/empty or carries no symmetric pair.
    """
    if quantile_predictions not in group.columns:
        return None
    parsed = [
        levels
        for value in group[quantile_predictions].dropna()
        if (levels := _quantile_dict_levels(value)) is not None
    ]
    if not parsed:
        return None
    common = set(parsed[0])
    for levels in parsed[1:]:
        common &= set(levels)
    lowers = sorted(
        lower
        for lower in common
        if lower < 0.5 and any(abs((1.0 - lower) - upper) < 1e-9 for upper in common)
    )
    if not lowers:
        return None
    lower = lowers[0]
    upper = next(upper for upper in common if abs(upper - (1.0 - lower)) < 1e-9)
    return (lower, upper)


def _quantile_bound_series(
    column: pd.Series, level: float
) -> pd.Series:
    """Extract the prediction at ``level`` from a ``quantile_predictions`` column."""

    def _at(value: Any) -> float:
        levels = _quantile_dict_levels(value)
        if levels is None:
            return float("nan")
        for key, pred in levels.items():
            if abs(key - level) < 1e-9:
                return pred
        return float("nan")

    return column.map(_at)


def _gaussian_pit(group: pd.DataFrame) -> pd.Series | None:
    """Gaussian PIT values ``Phi((actual - prediction) / sqrt(variance))`` for a
    group carrying ``variance_prediction``, or ``None`` when the column is
    absent. Rows with a non-finite or non-positive variance are dropped."""
    if "variance_prediction" not in group.columns:
        return None
    from scipy import stats as _stats

    valid = group[["actual", "prediction", "variance_prediction"]].dropna()
    valid = valid[valid["variance_prediction"] > 0.0]
    if valid.empty:
        return pd.Series(dtype=float)
    z = (valid["actual"] - valid["prediction"]) / np.sqrt(valid["variance_prediction"])
    return pd.Series(_stats.norm.cdf(z.to_numpy(dtype=float)), index=valid.index)


def calibration_table(master: pd.DataFrame, spec: PipelineSpec) -> pd.DataFrame:
    """PIT-based calibration diagnostics per (target, horizon, contender).

    Gated entirely by ``spec.evaluation.tests``: none of ``"berkowitz"``/
    ``"pit_autocorr"``/``"coverage"`` are in the default ``EvalSpec.tests``
    (``("dm", "cw", "mcs")``), so a default run returns an empty frame without
    inspecting ``master``'s columns, mirroring ``density_table``'s gating.

    ``"berkowitz"``/``"pit_autocorr"`` need ``variance_prediction`` (the PIT is
    the Gaussian CDF of the standardized residual) and reuse
    :func:`macroforecast.tests.density_interval_tests`; a requested test raises
    ``ValueError`` if ``variance_prediction`` is absent from ``master``'s
    schema entirely (not merely NaN for one contender -- a contender that
    itself never emitted a variance is silently skipped, degrading gracefully,
    the same convention :func:`macroforecast.metrics.evaluate_forecasts` uses).
    ``"coverage"`` needs a symmetric quantile pair from ``quantile_predictions``
    and reuses :func:`macroforecast.tests.interval_coverage_test`; same
    schema-level ValueError contract.
    """
    requested = _eval_calibration_tests(spec)
    if not requested or not _has_groups(master):
        return pd.DataFrame(columns=_CALIBRATION_COLUMNS)
    if "contender" not in master.columns:
        return pd.DataFrame(columns=_CALIBRATION_COLUMNS)

    needs_variance = bool({"berkowitz", "pit_autocorr"} & set(requested))
    if needs_variance and "variance_prediction" not in master.columns:
        missing = sorted({"berkowitz", "pit_autocorr"} & set(requested))
        raise ValueError(
            f"variance_prediction column is required for requested calibration "
            f"test(s) {missing}; no arm in this forecast frame emits a predictive "
            "variance (see macroforecast.forecasting.policies.base._variance_series)."
        )
    if "coverage" in requested and "quantile_predictions" not in master.columns:
        raise ValueError(
            "quantile_predictions column is required for the requested 'coverage' "
            "calibration test; no arm in this forecast frame emits quantile "
            "predictions (see macroforecast.forecasting.policies.base._quantile_frame)."
        )

    from macroforecast.tests import density_interval_tests, interval_coverage_test

    alpha = _eval_calibration_alpha(spec)
    rows: list[dict[str, Any]] = []
    for (target, horizon), group in master.groupby(["target", "horizon"], dropna=False):
        for contender, cgroup in group.groupby("contender", dropna=False):
            if needs_variance:
                pit = _gaussian_pit(cgroup)
                if pit is not None and len(pit) > 0:
                    diagnostics = density_interval_tests(pit, alpha=alpha)
                    if "berkowitz" in requested:
                        berkowitz = diagnostics.get("berkowitz", {})
                        rows.append({
                            "target": target, "horizon": horizon, "contender": contender,
                            "test": "berkowitz",
                            "statistic": berkowitz.get("lr_statistic"),
                            "p_value": berkowitz.get("p_value"),
                            "reject": berkowitz.get("reject"),
                            "n_obs": diagnostics.get("n_obs"),
                            "coverage_rate": None,
                        })
                    if "pit_autocorr" in requested:
                        pit_auto = diagnostics.get("pit_autocorrelation", {})
                        rows.append({
                            "target": target, "horizon": horizon, "contender": contender,
                            "test": "pit_autocorr",
                            "statistic": pit_auto.get("statistic"),
                            "p_value": pit_auto.get("p_value"),
                            "reject": pit_auto.get("decision"),
                            "n_obs": diagnostics.get("n_obs"),
                            "coverage_rate": None,
                        })
            if "coverage" in requested:
                pair = _widest_symmetric_quantile_pair(cgroup)
                if pair is not None:
                    lower_level, upper_level = pair
                    lower = _quantile_bound_series(cgroup["quantile_predictions"], lower_level)
                    upper = _quantile_bound_series(cgroup["quantile_predictions"], upper_level)
                    valid = pd.DataFrame({
                        "actual": cgroup["actual"], "lower": lower, "upper": upper,
                    }).dropna()
                    if not valid.empty:
                        coverage = interval_coverage_test(
                            valid["actual"], valid["lower"], valid["upper"],
                            alpha=1.0 - (upper_level - lower_level),
                        )
                        kupiec = coverage.get("kupiec_pof", {})
                        rows.append({
                            "target": target, "horizon": horizon, "contender": contender,
                            "test": "coverage",
                            "statistic": kupiec.get("lr_statistic"),
                            "p_value": kupiec.get("p_value"),
                            "reject": kupiec.get("reject"),
                            "n_obs": coverage.get("n_obs"),
                            "coverage_rate": coverage.get("coverage_rate"),
                        })
    return pd.DataFrame(rows, columns=_CALIBRATION_COLUMNS) if rows else pd.DataFrame(columns=_CALIBRATION_COLUMNS)


def evaluate_cross_policy(
    forecasts: pd.DataFrame,
    *,
    benchmark: str,
    benchmark_policy: str,
    policy_column: str = "forecast_policy",
    separator: str = "::",
) -> pd.DataFrame:
    """Score every ``(arm, forecast_policy)`` contender against ONE benchmark fixed
    to a single policy -- the common-denominator convention.

    Use this when the benchmark you want lives under a different forecast policy
    than the contenders. The GCLS (2021) appendix, for instance, scores both its
    direct and its path-average tables against a single FM benchmark, the direct
    FM. Running several policies for one target in a single spec pools the
    policies' rows for the same arm, because :func:`accuracy_table` keys the
    relative metrics on contender name within a ``(target, horizon)`` cell and
    does not split on policy. This helper does the qualification for you: it makes
    each ``(arm, forecast_policy)`` a distinct contender, scores all of them
    against the single ``(benchmark, benchmark_policy)`` arm, and returns a tidy
    accuracy table that keeps ``arm`` and ``forecast_policy`` as their own columns.

    Parameters
    ----------
    forecasts:
        The master forecast frame (``report.forecasts``). Must carry the columns
        ``target, horizon, origin, prediction, actual, contender`` and
        ``policy_column``.
    benchmark:
        The arm name of the benchmark (e.g. ``"FM"``).
    benchmark_policy:
        The forecast policy whose copy of ``benchmark`` is THE denominator for
        every contender (e.g. ``"direct_average"`` for the direct FM).
    policy_column:
        Column holding the per-row forecast policy. Default ``"forecast_policy"``.
    separator:
        Joins arm and policy into the qualified contender key, then splits them
        back out. Must not appear in any arm name or policy value; the default
        ``"::"`` is safe for the underscore-bearing policy names
        (``direct_average``, ``path_average``).

    Returns
    -------
    The accuracy table with one row per ``(target, horizon, arm, forecast_policy)``
    -- ``relative_mse`` / ``r2_oos`` / ``rmse`` computed pairwise against the fixed
    benchmark -- plus ``arm`` and ``forecast_policy`` columns.
    """
    required = {"target", "horizon", "origin", "prediction", "actual",
                "contender", policy_column}
    missing = required - set(forecasts.columns)
    if missing:
        raise ValueError(
            f"evaluate_cross_policy: forecasts frame is missing column(s) "
            f"{sorted(missing)}; a master forecast frame from run_pipeline carries them."
        )
    arm = forecasts["contender"].astype(str)
    policy = forecasts[policy_column].astype(str)
    if arm.str.contains(separator, regex=False).any() or policy.str.contains(separator, regex=False).any():
        raise ValueError(
            f"evaluate_cross_policy: separator {separator!r} appears inside an arm "
            f"name or a {policy_column} value; pass a separator that does not."
        )
    qualified = forecasts.copy()
    qualified["contender"] = arm + separator + policy

    bench_name = f"{benchmark}{separator}{benchmark_policy}"
    present = set(qualified["contender"].unique())
    if bench_name not in present:
        raise ValueError(
            f"evaluate_cross_policy: benchmark arm {benchmark!r} under policy "
            f"{benchmark_policy!r} (contender {bench_name!r}) is not present in the "
            f"forecasts; available contenders are {sorted(present)}."
        )

    acc = _accuracy_against(qualified, bench_name)
    # split the synthetic contender back into tidy arm / policy columns
    split = acc["contender"].str.rsplit(separator, n=1, expand=True)
    acc = acc.assign(arm=split[0].to_numpy(), **{policy_column: split[1].to_numpy()})
    ordered = [
        "target", "horizon", "contender", "arm", policy_column,
        "rmse", "relative_mse", "r2_oos", "n_common", "is_benchmark", "benchmark_present",
    ]
    return acc[ordered]


def evaluate(
    master: pd.DataFrame,
    spec: PipelineSpec,
    *,
    inputs: ResolvedEvaluationInputs | None = None,
) -> dict[str, pd.DataFrame]:
    """Run the full evaluation: combinations -> accuracy + significance + MCS
    + density + calibration.

    ``density``/``calibration`` are opt-in via ``EvalSpec.metrics``/``tests``
    (see ``density_table``/``calibration_table``); a default ``EvalSpec`` never
    computes them (empty frames), so ``forecasts``/``accuracy``/``significance``/
    ``mcs`` stay byte-identical to before these two keys existed.

    ``inputs`` carries the evaluation inputs that have to be *loaded* rather than
    computed -- today, the state series behind a named subsample mask such as
    ``"nber_recession"``. This function never loads them itself, so scoring one
    fixed frame is deterministic and works offline: ``run_pipeline`` and
    ``rescore`` call
    :func:`~macroforecast.pipeline.resolve_evaluation_inputs` and pass the result
    here, and a direct call whose ``EvalSpec.subsamples`` names an indicator must
    do the same or it raises. Everything else -- user-supplied masks, plain date
    windows, no subsamples at all -- needs no ``inputs`` and is unaffected.
    """
    _warn_failed_cells(master)
    full = apply_combinations(master, spec)
    subsample_frames = _subsample_frames(full, spec, inputs)
    if subsample_frames is not None:
        accuracy_parts: list[pd.DataFrame] = []
        significance_parts: list[pd.DataFrame] = []
        mcs_parts: list[pd.DataFrame] = []
        density_parts: list[pd.DataFrame] = []
        calibration_parts: list[pd.DataFrame] = []
        for name, frame in subsample_frames:
            _warn_short_subsample(name, frame)
            accuracy_parts.append(_with_subsample_column(accuracy_table(frame, spec), name))
            significance_parts.append(_with_subsample_column(significance_table(frame, spec), name))
            mcs_parts.append(_with_subsample_column(mcs_table(frame, spec), name))
            density_parts.append(_with_subsample_column(density_table(frame, spec), name))
            calibration_parts.append(_with_subsample_column(calibration_table(frame, spec), name))
        return _evaluation_result(
            forecasts=full,
            accuracy=_concat_tables(accuracy_parts),
            significance=_concat_tables(significance_parts),
            mcs=_concat_tables(mcs_parts),
            density=_concat_tables(density_parts),
            calibration=_concat_tables(calibration_parts),
        )
    return _evaluation_result(
        forecasts=full,
        accuracy=accuracy_table(full, spec),
        significance=significance_table(full, spec),
        mcs=mcs_table(full, spec),
        density=density_table(full, spec),
        calibration=calibration_table(full, spec),
    )
