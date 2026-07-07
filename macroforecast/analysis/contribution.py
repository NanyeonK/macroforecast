"""Axis-contribution regressions over forecast-error rows."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd

Outcome = Literal["r2", "squared_error", "absolute_error"]


def axis_contribution(
    master: pd.DataFrame,
    *,
    features: list[str],
    outcome: Outcome = "r2",
    fixed_effects: tuple[str, ...] = ("target", "horizon", "date"),
    interactions: Mapping[str, pd.Series] | None = None,
    hac_lags: int | None = None,
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

    Standard errors come from :func:`macroforecast.data_analysis.newey_west`, a
    Bartlett/Newey-West HAC covariance for OLS. Rows are sorted by ``date`` (or
    ``origin`` when no date column exists) before fitting; ``hac_lags=None`` uses
    zero lags. This is a descriptive attribution regression for forecast-error
    decomposition. It does not identify causal effects of model-design features.

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

    _require_columns(master, ("actual", "prediction"))
    work = _sort_for_hac(master.copy())
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

    from macroforecast.data_analysis import newey_west

    lags = 0 if hac_lags is None else int(hac_lags)
    fit = newey_west(fit_x, fit_y, lags=lags, add_intercept=False)
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
                "covariance": "newey_west_bartlett",
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
        "hac_lag_source": "default_zero" if hac_lags is None else "user",
        "weights": "provided" if weights_used else None,
        "n_obs": int(fit["n_obs"]),
        "n_coef": int(fit["n_coef"]),
        "dropped_collinear_columns": dropped_columns,
        "estimator": "OLS with dummy-absorbed fixed effects and Newey-West HAC covariance",
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
