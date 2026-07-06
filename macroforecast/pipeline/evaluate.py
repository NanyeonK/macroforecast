"""Pipeline evaluation: accuracy, DM/CW significance, MCS, cross-arm combinations (Stage 2)."""
from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from typing import Any, cast

import numpy as np
import pandas as pd

from macroforecast.pipeline.spec import (
    CALIBRATION_EVAL_TESTS,
    CombinationContender,
    PipelineSpec,
    contender_names,
)

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


def _has_groups(master: pd.DataFrame) -> bool:
    """True when ``master`` is non-empty and carries the (target, horizon) keys."""
    return not master.empty and _GROUP_KEYS.issubset(master.columns)


# ``evaluate``/``accuracy_table``/... are pure-frame and duck-type friendly: some
# callers (and tests) pass a minimal spec double carrying only
# ``evaluation.benchmark``. The newly-consumed EvalSpec fields are therefore read
# with ``getattr`` defaults matching :class:`~macroforecast.pipeline.spec.EvalSpec`'s
# own defaults, so such spec doubles keep the exact pre-threading behavior.
_DEFAULT_EVAL_TESTS: tuple[str, ...] = ("dm", "cw", "mcs")
_PAIRWISE_LONG_TESTS = frozenset({"gw", "enc_new", "enc_t", "pt", "hm", "ag", "gr"})
_NESTED_QUADRATIC_TESTS = frozenset({"cw", "enc_new", "enc_t"})
_SET_COMPARISON_TESTS = frozenset({"spa", "rc", "stepm"})


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

    return {
        "target": target,
        "horizon": horizon,
        "contender": contender,
        "test": test,
        "statistic": getattr(result, "statistic", None),
        "p_value": getattr(result, "p_value", None),
        "reject": getattr(result, "decision", None),
        "n_obs": getattr(result, "n_obs", n_obs),
    }


def _failed_result_row(
    *,
    target: Any,
    horizon: Any,
    contender: str,
    test: str,
    n_obs: int,
) -> dict[str, Any]:
    return {
        "target": target,
        "horizon": horizon,
        "contender": contender,
        "test": test,
        "statistic": np.nan,
        "p_value": np.nan,
        "reject": False,
        "n_obs": int(n_obs),
    }


# combination methods that need realised values (estimated weights), and their lag
_ESTIMATED = {
    "inverse_mspe", "dmspe", "best_n", "bates_granger",
    "granger_ramanathan", "constrained_ls", "eigenvector", "regularized",
}


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
                if len(keep) < 2:
                    continue
                wide = wide[keep]
            if wide.shape[1] < 2:
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
        pesaran_timmermann_test,
    )

    bench = spec.evaluation.benchmark
    if not _has_groups(master):
        return pd.DataFrame(columns=_SIGNIFICANCE_COLUMNS)

    requested = _eval_tests(spec)
    want_dm = "dm" in requested
    want_cw = "cw" in requested and bool(spec.evaluation.cw_for_nested)
    requested_long = tuple(name for name in requested if name in _PAIRWISE_LONG_TESTS)
    if not want_dm and not want_cw and not requested_long:
        return pd.DataFrame(columns=_SIGNIFICANCE_COLUMNS)

    # Clark-West is only licensed where the benchmark is nested in the contender.
    # Build the set of contenders whose arm declares nesting; CW is emitted only
    # for those. Non-nested contenders get DM only (CW would be an invalid test).
    nested: set[str] = set()
    for a in spec.arms:
        if getattr(a, "nested_in_benchmark", False):
            nested |= set(contender_names(a))

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
            if int(common.sum()) < 8:
                continue
            y = actual[common].to_numpy(float)
            fb_vals = fb[common].to_numpy(float)
            fc_vals = fc[common].to_numpy(float)
            if loss_fn is not None:
                loss_b = np.asarray(loss_fn(y, fb_vals), dtype=float)
                loss_c = np.asarray(loss_fn(y, fc_vals), dtype=float)
            else:
                loss_b = (fb_vals - y) ** 2
                loss_c = (fc_vals - y) ** 2
            row: dict[str, Any] = {"target": target, "horizon": horizon, "contender": contender}
            has_result = False
            if want_dm:
                try:
                    dm = dm_test(
                        loss_c,
                        loss_b,
                        **{**dm_options, "horizon": int(horizon), "input_type": "loss"},
                    )
                    row["dm_stat"] = float(cast(float, dm.statistic)); row["dm_p"] = float(cast(float, dm.p_value))
                except Exception:
                    row["dm_stat"] = np.nan; row["dm_p"] = np.nan
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
                    row["cw_stat"] = float(cast(float, cw.statistic)); row["cw_p"] = float(cast(float, cw.p_value))
                except Exception:
                    row["cw_stat"] = np.nan; row["cw_p"] = np.nan
                has_result = True
            if has_result:
                out.append(row)
            n_obs = int(common.sum())
            for test_name in requested_long:
                if test_name in custom_loss_blocked:
                    continue
                if test_name in {"enc_new", "enc_t"} and contender not in nested:
                    continue
                try:
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
                        res = pesaran_timmermann_test(
                            y,
                            fc_vals,
                            **{**_eval_test_options(spec, "pt"), "method": "pesaran_timmermann"},
                        )
                        out.append(_result_row(
                            target=target, horizon=horizon, contender=str(contender),
                            test="pt", result=res, n_obs=n_obs,
                        ))
                    elif test_name == "hm":
                        res = henriksson_merton_test(
                            y,
                            fc_vals,
                            **{**_eval_test_options(spec, "hm"), "method": "henriksson_merton"},
                        )
                        out.append(_result_row(
                            target=target, horizon=horizon, contender=str(contender),
                            test="hm", result=res, n_obs=n_obs,
                        ))
                    elif test_name == "ag":
                        res = anatolyev_gerko_test(
                            y,
                            fc_vals,
                            **{**_eval_test_options(spec, "ag"), "method": "anatolyev_gerko"},
                        )
                        out.append(_result_row(
                            target=target, horizon=horizon, contender=str(contender),
                            test="ag", result=res, n_obs=n_obs,
                        ))
                    elif test_name == "gr":
                        gr = conditional_predictive_ability_test(
                            loss_c,
                            loss_b,
                            **{
                                **_eval_test_options(spec, "gr"),
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
                        })
                except Exception:
                    out.append(_failed_result_row(
                        target=target, horizon=horizon, contender=str(contender),
                        test=test_name, n_obs=n_obs,
                    ))
    return pd.DataFrame(out) if out else pd.DataFrame(columns=_SIGNIFICANCE_COLUMNS)


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
            except Exception:
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
        try:
            res = set_functions[test_name](loss_panel, **options)
        except Exception:
            continue
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


def evaluate(master: pd.DataFrame, spec: PipelineSpec) -> dict[str, pd.DataFrame]:
    """Run the full evaluation: combinations -> accuracy + significance + MCS
    + density + calibration.

    ``density``/``calibration`` are opt-in via ``EvalSpec.metrics``/``tests``
    (see ``density_table``/``calibration_table``); a default ``EvalSpec`` never
    computes them (empty frames), so ``forecasts``/``accuracy``/``significance``/
    ``mcs`` stay byte-identical to before these two keys existed.
    """
    full = apply_combinations(master, spec)
    return {
        "forecasts": full,
        "accuracy": accuracy_table(full, spec),
        "significance": significance_table(full, spec),
        "mcs": mcs_table(full, spec),
        "density": density_table(full, spec),
        "calibration": calibration_table(full, spec),
    }
