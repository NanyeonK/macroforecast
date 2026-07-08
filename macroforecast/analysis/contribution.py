"""Axis-contribution regressions over forecast-error rows."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

Outcome = Literal["r2", "squared_error", "absolute_error"]
Vcov = Literal["hc0", "hac", "driscoll_kraay", "cluster"]


def axis_contribution(
    master: pd.DataFrame,
    *,
    features: list[str],
    outcome: Outcome = "r2",
    fixed_effects: tuple[str, ...] = ("target", "horizon", "date"),
    interactions: Mapping[str, pd.Series] | None = None,
    hac_lags: int | None = None,
    vcov: Vcov = "driscoll_kraay",
    cluster_by: str = "date",
    weights: Any = None,
    reference: str | None = None,
) -> pd.DataFrame:
    """Estimate descriptive feature-attribution regressions from forecast rows.

    The helper consumes any master forecast frame with ``actual`` and
    ``prediction`` columns, such as a live ``PipelineReport.forecasts``, a
    result-store load, or a rescored frame. ``features`` names tag columns
    directly (``"tag_NL"``) or by bare tag key (``"NL"`` resolves to
    ``"tag_NL"``). Each feature is expanded into treatment regressors, then OLS
    regresses the chosen error outcome on those regressors plus one joint fixed
    effect over ``fixed_effects``. The default therefore absorbs
    target-horizon-date cells, matching the treatment-regression design of
    Goulet Coulombe, Leroux, Stevanovic & Surprenant (2022), "How is Machine
    Learning Useful for Macroeconomic Forecasting?", Journal of Applied
    Econometrics 37(5) ("GCLS 2022") — the GCLS-style target-time-horizon
    treatment-regression pattern.

    ``outcome="r2"`` implements the GCLS-style row pseudo-R2 transform
    ``1 - e^2 / MSE(reference)``. The denominator is the reference contender's
    mean squared error within each target-horizon group, so the reference's mean
    pseudo-R2 is zero in each group and candidate means recover benchmark-relative
    OOS R2. Pass ``reference=`` to make the base explicit; if omitted, the helper
    infers it deterministically and records that choice in ``attrs``.

    Standard errors default to ``vcov="driscoll_kraay"``. This aggregates OLS
    score contributions within each distinct ``date`` and applies a Bartlett HAC
    kernel over the ordered dates, making the reported inference robust to
    within-date cross-sectional dependence and date-serial correlation. When
    ``hac_lags=None``, the bandwidth is the Newey-West fixed rule
    ``floor(4 * (T / 100) ** (2 / 9))`` over the number of distinct dates,
    truncated at ``T - 1``. This is a behavior change from the original
    row-stacked HAC path; coefficients are unchanged, but standard errors now
    match the GCLS-style panel inference by default.

    Other covariance choices are explicit: ``vcov="cluster"`` uses one-way
    date-clustered CR0 scores with no serial kernel, ``vcov="hc0"`` uses White
    HC0 on stacked rows, and ``vcov="hac"`` preserves the legacy single-index
    row-stacked Newey-West calculation. The legacy ``"hac"`` option is not a
    panel estimator because adjacent rows may be different targets, horizons, or
    contenders at the same date. This is a descriptive attribution regression
    for forecast-error decomposition. It does not identify causal effects of
    model-design features.

    Returns
    -------
    pandas.DataFrame
        One row per treatment or interaction term with ``feature``, ``level``,
        ``interaction``, ``coef``, ``se``, ``t``, ``p``, ``n``, ``outcome``,
        ``fe_spec``, and covariance metadata. A plot-ready long frame with the
        fitted row-level outcome is available in
        ``result.attrs["macroforecast_axis_contribution_plot_frame"]``.
    """

    if not features:
        raise ValueError("axis_contribution requires at least one feature")
    if outcome not in ("r2", "squared_error", "absolute_error"):
        raise ValueError("outcome must be one of 'r2', 'squared_error', or 'absolute_error'")
    if hac_lags is not None and int(hac_lags) < 0:
        raise ValueError("hac_lags must be nonnegative or None")
    vcov_key = _normalize_vcov(vcov)

    _require_columns(master, ("actual", "prediction"))
    work = _sort_for_hac(master.copy())
    if vcov_key in {"driscoll_kraay", "cluster"}:
        if cluster_by != "date":
            raise ValueError("cluster_by currently only supports 'date'")
        _require_columns(work, (cluster_by,))
    y, outcome_meta = _outcome_values(work, outcome=outcome, reference=reference)
    work["_mf_axis_outcome"] = y

    treatment, term_info, feature_columns = _feature_design(work, features)
    fixed_effect_design, fe_valid = _fixed_effect_design(work, fixed_effects)
    interaction_design, interaction_info = _interaction_design(
        work,
        treatment,
        term_info,
        interactions,
    )
    term_info = {**term_info, **interaction_info}

    x = pd.concat(
        [
            pd.DataFrame({"const": 1.0}, index=work.index),
            treatment,
            interaction_design,
            fixed_effect_design,
        ],
        axis=1,
    )
    w = _resolve_weights(work, weights)
    finite = np.isfinite(work["_mf_axis_outcome"].to_numpy(dtype=float))
    finite &= np.all(np.isfinite(x.to_numpy(dtype=float)), axis=1)
    finite &= fe_valid.to_numpy(dtype=bool)
    if w is not None:
        finite &= np.isfinite(w.to_numpy(dtype=float)) & (w.to_numpy(dtype=float) > 0)

    x_fit = x.loc[finite]
    y_fit = work.loc[finite, "_mf_axis_outcome"].astype(float)
    if x_fit.empty:
        raise ValueError("axis_contribution has no complete rows after filtering")

    ranked_columns, dropped_columns = _rank_columns(x_fit)
    x_ranked = x_fit[ranked_columns]
    fit_y = y_fit
    fit_x = x_ranked
    weights_used = w is not None
    if w is not None:
        sqrt_w = np.sqrt(w.loc[finite].astype(float))
        fit_y = y_fit * sqrt_w
        fit_x = x_ranked.mul(sqrt_w, axis=0)

    fit = _axis_ols_inference(
        fit_x,
        fit_y,
        vcov=vcov_key,
        hac_lags=hac_lags,
        cluster_values=(
            work.loc[finite, cluster_by]
            if vcov_key in {"driscoll_kraay", "cluster"}
            else None
        ),
        cluster_by=cluster_by,
    )
    by_name = {
        str(row["name"]): row
        for row in fit["coefficients"]
        if isinstance(row, Mapping)
    }
    rows: list[dict[str, Any]] = []
    fe_spec = _fe_spec_label(fixed_effects)
    for term, info in term_info.items():
        if term not in by_name:
            continue
        row = by_name[term]
        rows.append(
            {
                "feature": info["feature"],
                "level": info.get("level"),
                "interaction": info.get("interaction"),
                "term": term,
                "coef": float(row["estimate"]),
                "se": float(row["std_error"]),
                "t": float(row["t_value"]),
                "p": float(row["p_value"]),
                "n": int(fit["n_obs"]),
                "outcome": outcome,
                "fe_spec": fe_spec,
                "hac_lags": int(fit["lags"]),
                "vcov": str(fit["vcov"]),
                "cluster_by": fit.get("cluster_by"),
                "covariance": str(fit["covariance"]),
                "reference": outcome_meta.get("reference"),
            }
        )
    result = pd.DataFrame(
        rows,
        columns=[
            "feature",
            "level",
            "interaction",
            "term",
            "coef",
            "se",
            "t",
            "p",
            "n",
            "outcome",
            "fe_spec",
            "hac_lags",
            "vcov",
            "cluster_by",
            "covariance",
            "reference",
        ],
    )
    plot_columns = [
        column
        for column in [
            "target",
            "horizon",
            "date",
            "origin",
            "arm",
            "contender",
            *feature_columns,
        ]
        if column in work.columns
    ]
    plot_frame = work.loc[finite, plot_columns].copy()
    plot_frame["outcome"] = outcome
    plot_frame["outcome_value"] = y_fit.to_numpy(dtype=float)
    plot_frame["reference"] = outcome_meta.get("reference")

    metadata = {
        "kind": "axis_contribution",
        "version": 1,
        "outcome": outcome,
        "outcome_metadata": outcome_meta,
        "features": list(features),
        "feature_columns": feature_columns,
        "fixed_effects": list(fixed_effects),
        "fe_spec": fe_spec,
        "hac_lags": int(fit["lags"]),
        "hac_lag_source": str(fit["lag_source"]),
        "vcov": str(fit["vcov"]),
        "covariance": str(fit["covariance"]),
        "cluster_by": fit.get("cluster_by"),
        "n_clusters": fit.get("n_clusters"),
        "single_cluster_fallback": bool(fit.get("single_cluster_fallback", False)),
        "weights": "provided" if weights_used else None,
        "n_obs": int(fit["n_obs"]),
        "n_coef": int(fit["n_coef"]),
        "dropped_collinear_columns": dropped_columns,
        "estimator": "OLS with dummy-absorbed fixed effects and explicit covariance choice",
        "causal_interpretation": False,
    }
    result.attrs["macroforecast_metadata_schema"] = {
        "kind": "axis_contribution",
        "version": 1,
    }
    result.attrs["macroforecast_axis_contribution"] = metadata
    result.attrs["macroforecast_axis_contribution_plot_frame"] = plot_frame.reset_index(drop=True)
    return result


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"axis_contribution requires column(s): {missing}")


def _normalize_vcov(vcov: str) -> Vcov:
    key = str(vcov).lower()
    allowed = {"hc0", "hac", "driscoll_kraay", "cluster"}
    if key not in allowed:
        raise ValueError(
            "vcov must be one of 'hc0', 'hac', 'driscoll_kraay', or 'cluster'"
        )
    return cast(Vcov, key)


def _axis_ols_inference(
    x: pd.DataFrame,
    y: pd.Series,
    *,
    vcov: Vcov,
    hac_lags: int | None,
    cluster_values: pd.Series | None,
    cluster_by: str,
) -> dict[str, Any]:
    if vcov == "hc0":
        return _stacked_newey_west(
            x,
            y,
            lags=0,
            vcov="hc0",
            covariance="white_hc0_stacked",
            lag_source="not_applicable",
        )
    if vcov == "hac":
        lags = 0 if hac_lags is None else int(hac_lags)
        return _stacked_newey_west(
            x,
            y,
            lags=lags,
            vcov="hac",
            covariance="newey_west_bartlett_stacked",
            lag_source="default_zero" if hac_lags is None else "user",
        )
    if cluster_values is None:
        raise ValueError(f"vcov={vcov!r} requires cluster values")
    groups = _date_cluster_values(cluster_values, cluster_by=cluster_by)
    group_count = int(groups.nunique(dropna=False))
    if vcov == "cluster":
        lags = 0
        lag_source = "not_applicable"
    else:
        lags = _panel_hac_lags(group_count, hac_lags)
        lag_source = "newey_west_rule_dates" if hac_lags is None else "user"
    return _panel_score_sandwich(
        x,
        y,
        groups,
        lags=lags,
        vcov=vcov,
        covariance=(
            "driscoll_kraay_bartlett_date"
            if vcov == "driscoll_kraay"
            else "cluster_date_cr0"
        ),
        lag_source=lag_source,
        cluster_by=cluster_by,
    )


def _stacked_newey_west(
    x: pd.DataFrame,
    y: pd.Series,
    *,
    lags: int,
    vcov: Vcov,
    covariance: str,
    lag_source: str,
) -> dict[str, Any]:
    names, beta, bread, scores, n_obs, k = _ols_score_components(x, y)
    meat = _bartlett_meat(scores, int(lags))
    return _sandwich_result(
        names,
        beta,
        bread,
        meat,
        n_obs=n_obs,
        k=k,
        lags=int(lags),
        vcov=vcov,
        covariance=covariance,
        lag_source=lag_source,
        cluster_by=None,
        n_clusters=None,
        single_cluster_fallback=False,
        kernel="bartlett" if vcov == "hac" else None,
    )


def _panel_hac_lags(n_groups: int, hac_lags: int | None) -> int:
    if n_groups <= 1:
        return 0
    if hac_lags is None:
        lags = int(np.floor(4.0 * (n_groups / 100.0) ** (2.0 / 9.0)))
    else:
        lags = int(hac_lags)
    return min(lags, n_groups - 1)


def _date_cluster_values(values: pd.Series, *, cluster_by: str) -> pd.Series:
    if values.isna().any():
        raise ValueError(f"vcov clustering requires non-missing {cluster_by!r} values")
    dates = pd.to_datetime(values)
    if pd.isna(dates).any():
        raise ValueError(f"vcov clustering requires valid {cluster_by!r} values")
    return pd.Series(dates, index=values.index, name=cluster_by)


def _panel_score_sandwich(
    x: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    *,
    lags: int,
    vcov: Vcov,
    covariance: str,
    lag_source: str,
    cluster_by: str,
) -> dict[str, Any]:
    names, beta, bread, scores, n_obs, k = _ols_score_components(x, y)
    group_scores = _aggregate_scores_by_group(scores, groups)
    n_groups = int(group_scores.shape[0])
    single_cluster_fallback = n_groups <= 1
    if single_cluster_fallback:
        meat = scores.T @ scores
        lags = 0
    else:
        meat = _bartlett_meat(group_scores, int(lags))

    return _sandwich_result(
        names,
        beta,
        bread,
        meat,
        n_obs=n_obs,
        k=k,
        lags=int(lags),
        vcov=vcov,
        covariance=covariance,
        lag_source=lag_source,
        cluster_by=cluster_by,
        n_clusters=n_groups,
        single_cluster_fallback=single_cluster_fallback,
        kernel="bartlett" if vcov == "driscoll_kraay" else None,
    )


def _ols_score_components(
    x: pd.DataFrame,
    y: pd.Series,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, int, int]:
    names = [str(column) for column in x.columns]
    x_mat = np.asarray(x, dtype=float)
    y_vec = np.asarray(y, dtype=float).ravel()
    n_obs, k = x_mat.shape
    if n_obs <= k:
        raise ValueError("axis_contribution needs more observations than coefficients")
    xtx = x_mat.T @ x_mat
    bread = np.linalg.inv(xtx)
    beta = bread @ (x_mat.T @ y_vec)
    resid = y_vec - x_mat @ beta
    scores = x_mat * resid[:, None]
    return names, beta, bread, scores, int(n_obs), int(k)


def _sandwich_result(
    names: list[str],
    beta: np.ndarray,
    bread: np.ndarray,
    meat: np.ndarray,
    *,
    n_obs: int,
    k: int,
    lags: int,
    vcov: Vcov,
    covariance: str,
    lag_source: str,
    cluster_by: str | None,
    n_clusters: int | None,
    single_cluster_fallback: bool,
    kernel: str | None,
) -> dict[str, Any]:
    from scipy import stats as _stats

    vcov_matrix = bread @ meat @ bread
    se = np.sqrt(np.maximum(np.diag(vcov_matrix), 0.0))
    tstat = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    pval = 2.0 * _stats.t.sf(np.abs(tstat), df=n_obs - k)
    coefficients = [
        {
            "name": names[i],
            "estimate": float(beta[i]),
            "std_error": float(se[i]),
            "t_value": float(tstat[i]),
            "p_value": float(pval[i]),
        }
        for i in range(k)
    ]
    return {
        "n_obs": int(n_obs),
        "n_coef": int(k),
        "names": names,
        "lags": int(lags),
        "kernel": kernel,
        "coefficients": coefficients,
        "estimate": beta.tolist(),
        "std_error": se.tolist(),
        "t_value": tstat.tolist(),
        "p_value": pval.tolist(),
        "vcov": vcov,
        "covariance": covariance,
        "lag_source": lag_source,
        "cluster_by": cluster_by,
        "n_clusters": n_clusters,
        "single_cluster_fallback": single_cluster_fallback,
        "vcov_matrix": vcov_matrix.tolist(),
    }


def _aggregate_scores_by_group(scores: np.ndarray, groups: pd.Series) -> np.ndarray:
    frame = pd.DataFrame(scores)
    frame["_mf_cluster"] = groups.to_numpy()
    aggregated = frame.groupby("_mf_cluster", sort=True).sum(numeric_only=True)
    return np.asarray(aggregated, dtype=float)


def _bartlett_meat(scores: np.ndarray, lags: int) -> np.ndarray:
    band = min(max(int(lags), 0), scores.shape[0] - 1)
    meat = scores.T @ scores
    for lag in range(1, band + 1):
        weight = 1.0 - lag / (band + 1.0)
        gamma = scores[lag:].T @ scores[:-lag]
        meat += weight * (gamma + gamma.T)
    return meat


def _sort_for_hac(frame: pd.DataFrame) -> pd.DataFrame:
    order = [
        column
        for column in ("date", "origin", "target", "horizon", "contender", "arm")
        if column in frame.columns
    ]
    if not order:
        return frame.reset_index(drop=True)
    return frame.sort_values(order, kind="mergesort").reset_index(drop=True)


def _model_column(frame: pd.DataFrame) -> str:
    if "contender" in frame.columns:
        return "contender"
    if "arm" in frame.columns:
        return "arm"
    raise ValueError("outcome='r2' requires a 'contender' or 'arm' column for reference rows")


def _outcome_values(
    work: pd.DataFrame,
    *,
    outcome: Outcome,
    reference: str | None,
) -> tuple[pd.Series, dict[str, Any]]:
    err = work["actual"].astype(float) - work["prediction"].astype(float)
    squared = err.pow(2)
    if outcome == "squared_error":
        return squared.rename("_mf_axis_outcome"), {"transform": "squared_error"}
    if outcome == "absolute_error":
        return err.abs().rename("_mf_axis_outcome"), {"transform": "absolute_error"}

    _require_columns(work, ("target", "horizon"))
    model_col = _model_column(work)
    resolved, source = _resolve_reference(work, model_col=model_col, reference=reference)
    ref_mask = work[model_col].astype(str) == str(resolved)
    if not bool(ref_mask.any()):
        raise ValueError(f"reference {resolved!r} is not present in {model_col!r}")
    ref = work.loc[ref_mask, ["target", "horizon"]].copy()
    ref["_mf_ref_squared_error"] = squared.loc[ref_mask].to_numpy(dtype=float)
    denom = ref.groupby(["target", "horizon"], dropna=False)["_mf_ref_squared_error"].mean()
    keys = pd.MultiIndex.from_frame(work[["target", "horizon"]])
    denom_values = pd.Series(denom.reindex(keys).to_numpy(dtype=float), index=work.index)
    denom_values = denom_values.where(denom_values > 0)
    transformed = 1.0 - squared / denom_values
    return transformed.rename("_mf_axis_outcome"), {
        "transform": "pseudo_r2",
        "formula": "1 - squared_error / mean_reference_squared_error_by_target_horizon",
        "reference": str(resolved),
        "reference_source": source,
        "reference_column": model_col,
    }


def _resolve_reference(
    work: pd.DataFrame,
    *,
    model_col: str,
    reference: str | None,
) -> tuple[str, str]:
    if reference is not None:
        return str(reference), "user"
    for attr_name in ("macroforecast_benchmark", "benchmark"):
        value = work.attrs.get(attr_name)
        if value is not None:
            return str(value), f"frame.attrs[{attr_name!r}]"
    values = [str(value) for value in pd.unique(work[model_col].dropna())]
    if "benchmark" in values:
        return "benchmark", "contender_named_benchmark"
    if not values:
        raise ValueError(f"cannot infer reference from empty {model_col!r} column")
    return values[0], "first_observed_contender"


def _resolve_feature_column(frame: pd.DataFrame, feature: str) -> tuple[str, str]:
    if feature in frame.columns:
        return feature, feature[4:] if feature.startswith("tag_") else feature
    tag_column = f"tag_{feature}"
    if tag_column in frame.columns:
        return tag_column, feature
    raise ValueError(
        f"feature {feature!r} is not present as a column or as tag column {tag_column!r}"
    )


def _feature_design(
    work: pd.DataFrame,
    features: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], list[str]]:
    design = pd.DataFrame(index=work.index)
    info: dict[str, dict[str, Any]] = {}
    resolved_columns: list[str] = []
    for feature in features:
        column, label = _resolve_feature_column(work, str(feature))
        resolved_columns.append(column)
        series = work[column]
        before = set(design.columns)
        _append_feature_terms(design, info, series, column=column, label=label)
        if set(design.columns) == before:
            raise ValueError(f"feature {feature!r} has no regressable variation")
    return design, info, resolved_columns


def _append_feature_terms(
    design: pd.DataFrame,
    info: dict[str, dict[str, Any]],
    series: pd.Series,
    *,
    column: str,
    label: str,
) -> None:
    missing = series.isna()
    if pd.api.types.is_bool_dtype(series):
        term = f"{column}=True"
        values = series.astype(float)
        values.loc[missing] = np.nan
        if values.dropna().nunique() >= 2:
            design[term] = values
            info[term] = {"feature": label, "level": "True", "interaction": None}
        return
    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        levels = sorted(float(value) for value in pd.unique(numeric.dropna()))
        if levels == [0.0, 1.0]:
            term = f"{column}=1"
            values = numeric.astype(float)
            values.loc[missing] = np.nan
            design[term] = values
            info[term] = {"feature": label, "level": "1", "interaction": None}
        elif len(levels) >= 2:
            term = column
            values = numeric.astype(float)
            values.loc[missing] = np.nan
            design[term] = values
            info[term] = {"feature": label, "level": None, "interaction": None}
        return

    categories = sorted(pd.unique(series.dropna()), key=lambda value: str(value))
    if len(categories) < 2:
        return
    for level in categories[1:]:
        term = f"{column}={level}"
        values = (series == level).astype(float)
        values.loc[missing] = np.nan
        design[term] = values
        info[term] = {"feature": label, "level": str(level), "interaction": None}


def _fixed_effect_design(
    work: pd.DataFrame,
    fixed_effects: Sequence[str],
) -> tuple[pd.DataFrame, pd.Series]:
    if not fixed_effects:
        return pd.DataFrame(index=work.index), pd.Series(True, index=work.index)
    _require_columns(work, fixed_effects)
    fe_frame = work[list(fixed_effects)]
    valid = ~fe_frame.isna().any(axis=1)
    key = fe_frame.astype("string").agg("\x1f".join, axis=1)
    categories = sorted(pd.unique(key.loc[valid]), key=lambda value: str(value))
    design = pd.DataFrame(index=work.index)
    for idx, level in enumerate(categories[1:], start=1):
        column = f"fe_{idx}"
        values = (key == level).astype(float)
        values.loc[~valid] = np.nan
        design[column] = values
    return design, valid


def _interaction_design(
    work: pd.DataFrame,
    treatment: pd.DataFrame,
    term_info: Mapping[str, Mapping[str, Any]],
    interactions: Mapping[str, pd.Series] | None,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    design = pd.DataFrame(index=work.index)
    info: dict[str, dict[str, Any]] = {}
    if not interactions:
        return design, info
    if treatment.empty:
        raise ValueError("interactions require at least one treatment regressor")
    for name, state in interactions.items():
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError(f"interaction names must be valid identifiers, got {name!r}")
        aligned = _align_state_series(work, state, name=name)
        for term in treatment.columns:
            out_term = f"{term}:state_{name}"
            design[out_term] = treatment[term].astype(float) * aligned
            base = term_info[term]
            info[out_term] = {
                "feature": base["feature"],
                "level": base.get("level"),
                "interaction": name,
            }
    return design, info


def _align_state_series(work: pd.DataFrame, state: pd.Series, *, name: str) -> pd.Series:
    if not isinstance(state, pd.Series):
        state = pd.Series(state)
    values = pd.to_numeric(state, errors="coerce")
    key_column = "date" if "date" in work.columns else "origin" if "origin" in work.columns else None
    if key_column is None:
        if len(values) != len(work):
            raise ValueError(
                f"interaction state {name!r} must have one value per row when "
                "the frame has neither 'date' nor 'origin'"
            )
        return pd.Series(values.to_numpy(dtype=float), index=work.index)
    row_index = pd.to_datetime(work[key_column])
    state_index = pd.to_datetime(values.index)
    state_by_date = pd.Series(values.to_numpy(dtype=float), index=state_index)
    return pd.Series(state_by_date.reindex(row_index).to_numpy(dtype=float), index=work.index)


def _resolve_weights(work: pd.DataFrame, weights: Any) -> pd.Series | None:
    if weights is None:
        return None
    if isinstance(weights, str):
        _require_columns(work, (weights,))
        return pd.to_numeric(work[weights], errors="coerce")
    if isinstance(weights, pd.Series):
        aligned = weights.reindex(work.index)
        if aligned.isna().all() and len(weights) == len(work):
            aligned = pd.Series(weights.to_numpy(), index=work.index)
        return pd.to_numeric(aligned, errors="coerce")
    series = pd.Series(weights)
    if len(series) != len(work):
        raise ValueError("weights must be a column name or have one value per row")
    return pd.to_numeric(pd.Series(series.to_numpy(), index=work.index), errors="coerce")


def _rank_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    keep: list[str] = []
    dropped: list[str] = []
    matrix = frame.to_numpy(dtype=float)
    rank = 0
    keep_indices: list[int] = []
    for idx, column in enumerate(frame.columns):
        candidate_indices = [*keep_indices, idx]
        candidate = matrix[:, candidate_indices]
        new_rank = int(np.linalg.matrix_rank(candidate))
        if new_rank > rank:
            keep.append(str(column))
            keep_indices.append(idx)
            rank = new_rank
        else:
            dropped.append(str(column))
    return keep, dropped


def _fe_spec_label(fixed_effects: Sequence[str]) -> str:
    if not fixed_effects:
        return "none"
    return "joint(" + ",".join(str(item) for item in fixed_effects) + ")"


__all__ = ["axis_contribution"]
