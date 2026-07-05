"""Pipeline evaluation: accuracy, DM/CW significance, MCS, cross-arm combinations (Stage 2)."""
from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from typing import Any, cast

import numpy as np
import pandas as pd

from macroforecast.pipeline.spec import CombinationContender, PipelineSpec, contender_names

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


def _has_groups(master: pd.DataFrame) -> bool:
    """True when ``master`` is non-empty and carries the (target, horizon) keys."""
    return not master.empty and _GROUP_KEYS.issubset(master.columns)


# ``evaluate``/``accuracy_table``/... are pure-frame and duck-type friendly: some
# callers (and tests) pass a minimal spec double carrying only
# ``evaluation.benchmark``. The newly-consumed EvalSpec fields are therefore read
# with ``getattr`` defaults matching :class:`~macroforecast.pipeline.spec.EvalSpec`'s
# own defaults, so such spec doubles keep the exact pre-threading behavior.
_DEFAULT_EVAL_TESTS: tuple[str, ...] = ("dm", "cw", "mcs")


def _eval_metrics(spec: PipelineSpec) -> tuple[str | Callable[..., float], ...]:
    return tuple(getattr(spec.evaluation, "metrics", _DEFAULT_ACCURACY_METRICS))


def _eval_tests(spec: PipelineSpec) -> tuple[str, ...]:
    return tuple(getattr(spec.evaluation, "tests", _DEFAULT_EVAL_TESTS))


def _eval_loss(spec: PipelineSpec) -> Callable[[Any, Any], Any] | None:
    return getattr(spec.evaluation, "loss", None)


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


def accuracy_table(master: pd.DataFrame, spec: PipelineSpec) -> pd.DataFrame:
    """Per-``spec.evaluation.metrics`` accuracy per (target, horizon, contender)."""
    return _accuracy_against(master, spec.evaluation.benchmark, metrics=_eval_metrics(spec))


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
    """Diebold-Mariano and Clark-West p-values of each contender vs the benchmark.

    Only the tests named in ``spec.evaluation.tests`` run (``"dm"``/``"cw"``,
    ``cw`` additionally gated by ``cw_for_nested`` as before). The DM/CW loss
    differentials use ``spec.evaluation.loss(y_true, y_pred)`` when set, else
    squared error. Clark-West's adjustment term is only valid under quadratic
    loss, so when a custom ``loss`` is set and CW would otherwise run, CW is
    skipped (never silently computed against the wrong loss) and a
    ``UserWarning`` explains why; DM and MCS are loss-agnostic and are
    unaffected.
    """
    from macroforecast.tests import clark_west_test, dm_test

    bench = spec.evaluation.benchmark
    if not _has_groups(master):
        return pd.DataFrame(columns=_SIGNIFICANCE_COLUMNS)

    requested = _eval_tests(spec)
    want_dm = "dm" in requested
    want_cw = "cw" in requested and bool(spec.evaluation.cw_for_nested)
    if not want_dm and not want_cw:
        return pd.DataFrame(columns=_SIGNIFICANCE_COLUMNS)

    # Clark-West is only licensed where the benchmark is nested in the contender.
    # Build the set of contenders whose arm declares nesting; CW is emitted only
    # for those. Non-nested contenders get DM only (CW would be an invalid test).
    nested: set[str] = set()
    for a in spec.arms:
        if getattr(a, "nested_in_benchmark", False):
            nested |= set(contender_names(a))

    loss_fn = _eval_loss(spec)
    cw_blocked_by_custom_loss = bool(want_cw and loss_fn is not None and nested)
    if cw_blocked_by_custom_loss:
        warnings.warn(
            "EvalSpec.loss is a custom per-observation loss, but the Clark-West "
            "adjustment term is derived under quadratic loss and is not a valid "
            "test for an arbitrary loss function; skipping CW for this evaluation "
            "(DM and MCS are loss-agnostic and still run under the custom loss).",
            UserWarning, stacklevel=2,
        )

    out: list[dict[str, Any]] = []
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
                    dm = dm_test(loss_c, loss_b, horizon=int(horizon), input_type="loss")
                    row["dm_stat"] = float(cast(float, dm.statistic)); row["dm_p"] = float(cast(float, dm.p_value))
                except Exception:
                    row["dm_stat"] = np.nan; row["dm_p"] = np.nan
                has_result = True
            if want_cw and not cw_blocked_by_custom_loss and contender in nested:
                try:
                    cw = clark_west_test(loss_b, loss_c, fb_vals, fc_vals, horizon=int(horizon))
                    row["cw_stat"] = float(cast(float, cw.statistic)); row["cw_p"] = float(cast(float, cw.p_value))
                except Exception:
                    row["cw_stat"] = np.nan; row["cw_p"] = np.nan
                has_result = True
            if has_result:
                out.append(row)
    return pd.DataFrame(out) if out else pd.DataFrame(columns=_SIGNIFICANCE_COLUMNS)


def mcs_table(master: pd.DataFrame, spec: PipelineSpec, *, n_boot: int = 499) -> pd.DataFrame:
    """Model Confidence Set membership per (target, horizon).

    Runs only when ``"mcs"`` is in ``spec.evaluation.tests``. The loss matrix is
    ``spec.evaluation.loss(y_true, y_pred)`` when set, else squared error (the
    prior, only behavior); MCS is loss-agnostic, so any per-observation loss is
    valid here (unlike Clark-West, see :func:`significance_table`).
    """
    from macroforecast.tests import model_confidence_set

    if "mcs" not in _eval_tests(spec):
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
    for (target, horizon), group in panel.groupby(["target", "horizon"], dropna=False):
        if group["model_id"].nunique() < 2:
            continue
        try:
            res = model_confidence_set(
                group[["origin", "model_id", "squared_error"]],
                loss="squared_error", model="model_id", origin="origin",
                target="target", horizon="horizon", alpha=spec.evaluation.mcs_alpha,
                n_boot=n_boot, random_state=int(spec.seed or 0),
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
    return pd.DataFrame(out) if out else pd.DataFrame(columns=_MCS_COLUMNS)


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
    """Run the full evaluation: combinations -> accuracy + significance + MCS."""
    full = apply_combinations(master, spec)
    return {
        "forecasts": full,
        "accuracy": accuracy_table(full, spec),
        "significance": significance_table(full, spec),
        "mcs": mcs_table(full, spec),
    }
