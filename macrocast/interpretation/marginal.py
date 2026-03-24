"""OOS pseudo-R² and marginal contribution analysis (CLSS 2021 Eq. 11-12).

Implements the marginal contribution framework from Coulombe et al. (2021).
Each feature's average partial contribution to predictive accuracy is
estimated via OLS with fixed-effect absorption and Newey-West HAC SEs.

The result_df expected here is the output of ResultSet.to_dataframe(), which
has at minimum the columns:
    model_id, feature_set, horizon, forecast_date, y_hat, y_true,
    target_scheme

The column "target" is NOT in ResultSet.to_dataframe(). The variable being
forecast is identified implicitly by the experiment setup; when multiple
targets are pooled in one result_df the caller must add a "target" column
before passing it here. If "target" is absent we treat all rows as a single
target pool (equivalent to normalising by the grand OOS variance per horizon).

Reference: Coulombe, Leroux, Stevanovic, Surprenant (2021),
    "Macroeconomic Data Transformations Matter", IJF.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_Z_95: float = 1.959964  # 97.5th percentile of standard normal (95% CI)

# ---------------------------------------------------------------------------
# Feature-pair registry
# ---------------------------------------------------------------------------

# Maps each named feature to a list of (with_feature_set, without_feature_set)
# tuples.  The with-set contains the feature; the without-set is the ablated
# counterpart.  Auto-detection in marginal_contribution() filters this list to
# pairs whose both sides actually appear in the data.
_FEATURE_PAIRS: dict[str, list[tuple[str, str]]] = {
    "MARX": [
        ("F-MARX", "F"),
        ("F-X-MARX", "F-X"),
        ("X-MARX", "X"),
        ("F-X-MARX-Level", "F-X-Level"),
        ("X-MARX-Level", "X-Level"),
    ],
    "MAF": [
        ("F-MAF", "F"),
        ("F-X-MAF", "F-X"),
        ("X-MAF", "X"),
    ],
    "F": [
        ("F", "X"),
        ("F-X", "X"),
        ("F-MARX", "X-MARX"),
        ("F-MAF", "X-MAF"),
        ("F-Level", "X-Level"),
    ],
    # path_avg is not identified by feature_set but by target_scheme;
    # pairs are constructed dynamically in marginal_contribution().
    "path_avg": [],
}


# ---------------------------------------------------------------------------
# Eq. 11 — OOS pseudo-R² panel
# ---------------------------------------------------------------------------


def oos_r2_panel(
    result_df: pd.DataFrame,
    date_col: str = "forecast_date",
    horizon_col: str = "horizon",
    target_col: str | None = "target",
    y_hat_col: str = "y_hat",
    y_true_col: str = "y_true",
    benchmark_col: str | None = None,
) -> pd.DataFrame:
    """Add an ``oos_r2`` column to result_df.

    Two modes depending on ``benchmark_col``:

    **Variance-based** (``benchmark_col=None``, default — CLSS 2021 Eq. 11):
        For each (target, horizon) group the denominator is the OOS variance of
        the realised series:

            sigma²_{v,h} = (1/T_OOS) * sum_t (y_true_t - ybar_h^v)²

        OOS-R²_{t} = 1 - (y_true_t - y_hat_t)² / sigma²_{v,h}

    **Benchmark model** (``benchmark_col`` provided — Campbell-Thompson style):
        Denominator is the squared error of a named benchmark forecast column:

            OOS-R²_{t} = 1 - (y_true_t - y_hat_t)² / (y_true_t - y_bench_t)²

        The per-row result is then averaged by group to yield a pseudo-R².
        This allows comparing against AR, ARDI, or any other model in the table.

    Parameters
    ----------
    result_df : pd.DataFrame
        Forecast result table.  Required columns: ``y_hat``, ``y_true``,
        ``horizon`` (and optionally ``target``).
    date_col : str
        Column identifying the OOS forecast date. Default ``"forecast_date"``.
    horizon_col : str
        Column identifying the forecast horizon. Default ``"horizon"``.
    target_col : str or None
        Column identifying the target variable.  If None or absent, all rows
        are treated as a single target pool.
    y_hat_col : str
        Column with model point forecasts. Default ``"y_hat"``.
    y_true_col : str
        Column with realised values. Default ``"y_true"``.
    benchmark_col : str or None
        Column with benchmark model forecasts.  If provided, OOS-R² is computed
        as ``1 - MSE_model / MSE_benchmark`` using the benchmark's squared
        errors as denominator (Campbell-Thompson 2008).  If None, the OOS
        variance of ``y_true`` is used as denominator (CLSS 2021 Eq. 11).

    Returns
    -------
    pd.DataFrame
        Copy of ``result_df`` with an additional ``oos_r2`` column (float).
        Values can be negative; NaN is returned for groups with zero denominator.
    """
    df = result_df.copy()

    # Determine grouping keys
    group_keys: list[str] = [horizon_col]
    if target_col is not None and target_col in df.columns:
        group_keys = [target_col, horizon_col]

    # Squared errors of the model
    sq_err = (df[y_true_col] - df[y_hat_col]) ** 2

    if benchmark_col is not None:
        # Campbell-Thompson: denominator = benchmark squared errors
        if benchmark_col not in df.columns:
            raise ValueError(
                f"benchmark_col '{benchmark_col}' not found in result_df. "
                f"Available columns: {list(df.columns)}"
            )
        sq_err_bench = (df[y_true_col] - df[benchmark_col]) ** 2
        with np.errstate(invalid="ignore", divide="ignore"):
            oos_r2_vals = np.where(
                sq_err_bench == 0,
                np.nan,
                1.0 - sq_err.values / sq_err_bench.values,
            )
    else:
        # CLSS 2021 Eq. 11: denominator = group-level variance of y_true
        group_var = df.groupby(group_keys)[y_true_col].transform("var", ddof=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            oos_r2_vals = np.where(
                group_var == 0,
                np.nan,
                1.0 - sq_err.values / group_var.values,
            )

    df["oos_r2"] = oos_r2_vals
    return df


# ---------------------------------------------------------------------------
# MarginalEffect dataclass
# ---------------------------------------------------------------------------


@dataclass
class MarginalEffect:
    """Estimated marginal contribution of one feature, per model and horizon.

    Attributes
    ----------
    feature : str
        Feature name ("MARX", "MAF", "F", or "path_avg").
    model : str
        model_id string.
    horizon : int
        Forecast horizon h.
    alpha : float
        OLS intercept of delta ~ 1, i.e. mean(delta).
    se : float
        Newey-West HAC standard error.
    ci_low : float
        alpha - 1.96 * se.
    ci_high : float
        alpha + 1.96 * se.
    n_obs : int
        Number of delta observations used in estimation.
    """

    feature: str
    model: str
    horizon: int
    alpha: float
    se: float
    ci_low: float
    ci_high: float
    n_obs: int

    def to_dict(self) -> dict[str, object]:
        """Flatten to a plain dict for DataFrame construction."""
        return {
            "feature": self.feature,
            "model": self.model,
            "horizon": self.horizon,
            "alpha": self.alpha,
            "se": self.se,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "n_obs": self.n_obs,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_deltas(
    r2_df: pd.DataFrame,
    pair_with: str,
    pair_without: str,
    date_col: str,
    target_col: str | None,
    horizon_col: str,
    model_col: str,
) -> np.ndarray:
    """Compute delta = R²_with - R²_without for one matched pair.

    Merges the with and without subsets on (date, target, model_id) —
    the fixed-effect absorption is achieved by differencing within each
    (date, target) cell.

    Parameters
    ----------
    r2_df : pd.DataFrame
        Full result_df that already has the ``oos_r2`` column.
    pair_with, pair_without : str
        feature_set labels for the matched pair.

    Returns
    -------
    np.ndarray of shape (N_matched,)
        May be empty if the merge yields no rows.
    """
    mask_with = r2_df["feature_set"] == pair_with
    mask_without = r2_df["feature_set"] == pair_without

    sub_with = r2_df.loc[mask_with].copy()
    sub_without = r2_df.loc[mask_without].copy()

    if sub_with.empty or sub_without.empty:
        return np.array([], dtype=float)

    # Merge keys: (date, target, model, horizon)
    merge_keys = [date_col, horizon_col, model_col]
    if target_col is not None and target_col in r2_df.columns:
        merge_keys = [date_col, target_col, horizon_col, model_col]

    merged = sub_with[merge_keys + ["oos_r2"]].merge(
        sub_without[merge_keys + ["oos_r2"]],
        on=merge_keys,
        suffixes=("_with", "_without"),
    )

    if merged.empty:
        return np.array([], dtype=float)

    # Drop rows where either R² is NaN
    valid = merged["oos_r2_with"].notna() & merged["oos_r2_without"].notna()
    delta = (merged.loc[valid, "oos_r2_with"] - merged.loc[valid, "oos_r2_without"]).values
    return delta.astype(float)


def _newey_west_se(deltas: np.ndarray, bandwidth: int) -> tuple[float, float]:
    """OLS intercept-only regression with Newey-West HAC SE.

    Returns (alpha, se) where alpha = mean(deltas).
    Uses statsmodels OLS with cov_type='HAC'.
    """
    n = len(deltas)
    X = np.ones((n, 1))
    result = OLS(deltas, X).fit()
    nw_result = result.get_robustcov_results(cov_type="HAC", maxlags=bandwidth)
    alpha = float(nw_result.params[0])
    se = float(nw_result.bse[0])
    return alpha, se


def _default_bandwidth(n_dates: int) -> int:
    """Default Newey-West bandwidth: floor(T^{1/3})."""
    return max(1, math.floor(n_dates ** (1.0 / 3.0)))


# ---------------------------------------------------------------------------
# Eq. 12 — Marginal contribution
# ---------------------------------------------------------------------------


def marginal_contribution(
    result_df: pd.DataFrame,
    feature: str,
    feature_pairs: list[tuple[str, str]] | None = None,
    hac_bandwidth: int | None = None,
    date_col: str = "forecast_date",
    horizon_col: str = "horizon",
    target_col: str | None = "target",
    model_col: str = "model_id",
) -> pd.DataFrame:
    """Estimate the marginal contribution of ``feature`` via Eq. 12 of CLSS 2021.

    The estimator differences R² across matched (with, without) feature-set
    pairs, absorbing (target × horizon × date) fixed effects, then runs
    OLS on delta ~ 1 per (model, horizon) cell with Newey-West HAC SEs.

    Parameters
    ----------
    result_df : pd.DataFrame
        Result table from ResultSet.to_dataframe().  Must contain
        ``feature_set``, ``oos_r2`` (added by :func:`oos_r2_panel`),
        ``model_id``, ``horizon``, and the date column.
    feature : str
        One of "MARX", "MAF", "F", "path_avg".
    feature_pairs : list of (str, str) or None
        Explicit (with_feature_set, without_feature_set) pairs.
        Auto-detected from ``_FEATURE_PAIRS`` if None, filtered to pairs
        where both sides appear in ``result_df["feature_set"]``.
    hac_bandwidth : int or None
        Newey-West lag truncation.  Defaults to floor(T^{1/3}) where T is
        the number of unique OOS dates in result_df.
    date_col : str
        Column identifying forecast date. Default ``"forecast_date"``.
    horizon_col : str
        Column identifying forecast horizon. Default ``"horizon"``.
    target_col : str or None
        Column identifying target variable.  If None or absent, a single
        target pool is assumed.
    model_col : str
        Column with model identifier. Default ``"model_id"``.

    Returns
    -------
    pd.DataFrame
        One row per (model, horizon) cell that had enough observations.
        Columns: feature, model, horizon, alpha, se, ci_low, ci_high, n_obs.
    """
    if "oos_r2" not in result_df.columns:
        raise ValueError(
            "result_df is missing the 'oos_r2' column.  "
            "Call oos_r2_panel() first."
        )

    available_feature_sets = set(result_df["feature_set"].dropna().unique())

    # Resolve feature pairs
    if feature_pairs is not None:
        pairs = list(feature_pairs)
    elif feature == "path_avg":
        # path_avg is identified by target_scheme: "path_average" vs "direct"
        # Pairs are all (model, horizon) cells that have both schemes.
        # We handle this separately below.
        pairs = []
    else:
        candidate_pairs = _FEATURE_PAIRS.get(feature, [])
        pairs = [
            (w, wo)
            for (w, wo) in candidate_pairs
            if w in available_feature_sets and wo in available_feature_sets
        ]
        if not candidate_pairs and feature not in _FEATURE_PAIRS:
            warnings.warn(
                f"Feature '{feature}' is not in the default pair registry. "
                "Pass explicit feature_pairs or use one of: "
                f"{list(_FEATURE_PAIRS)}.",
                stacklevel=2,
            )

    # Default bandwidth based on full OOS date range
    if hac_bandwidth is None:
        n_dates = result_df[date_col].nunique()
        hac_bandwidth = _default_bandwidth(n_dates)

    # Effective target_col
    eff_target_col: str | None = (
        target_col if (target_col is not None and target_col in result_df.columns) else None
    )

    records: list[MarginalEffect] = []

    if feature == "path_avg":
        # Special handling: pairs are (path_average, direct) within same
        # model_id and feature_set.
        _records_path_avg(
            result_df=result_df,
            records=records,
            hac_bandwidth=hac_bandwidth,
            date_col=date_col,
            horizon_col=horizon_col,
            target_col=eff_target_col,
            model_col=model_col,
        )
    else:
        _records_feature_pairs(
            result_df=result_df,
            feature=feature,
            pairs=pairs,
            records=records,
            hac_bandwidth=hac_bandwidth,
            date_col=date_col,
            horizon_col=horizon_col,
            target_col=eff_target_col,
            model_col=model_col,
        )

    if not records:
        return pd.DataFrame(
            columns=["feature", "model", "horizon", "alpha", "se", "ci_low", "ci_high", "n_obs"]
        )

    return pd.DataFrame([r.to_dict() for r in records])


def _records_feature_pairs(
    result_df: pd.DataFrame,
    feature: str,
    pairs: list[tuple[str, str]],
    records: list[MarginalEffect],
    hac_bandwidth: int,
    date_col: str,
    horizon_col: str,
    target_col: str | None,
    model_col: str,
) -> None:
    """Populate records for standard feature-set pairs."""
    if not pairs:
        return

    # Iterate over (model, horizon) cells
    for (model_id, horizon), grp in result_df.groupby([model_col, horizon_col]):
        all_deltas: list[np.ndarray] = []
        for pair_with, pair_without in pairs:
            delta = _compute_deltas(
                r2_df=grp,
                pair_with=pair_with,
                pair_without=pair_without,
                date_col=date_col,
                target_col=target_col,
                horizon_col=horizon_col,
                model_col=model_col,
            )
            if len(delta) > 0:
                all_deltas.append(delta)

        if not all_deltas:
            continue

        deltas = np.concatenate(all_deltas)
        if len(deltas) < 2:
            continue

        alpha, se = _newey_west_se(deltas, hac_bandwidth)
        records.append(
            MarginalEffect(
                feature=feature,
                model=str(model_id),
                horizon=int(horizon),
                alpha=alpha,
                se=se,
                ci_low=alpha - _Z_95 * se,
                ci_high=alpha + _Z_95 * se,
                n_obs=len(deltas),
            )
        )


def _records_path_avg(
    result_df: pd.DataFrame,
    records: list[MarginalEffect],
    hac_bandwidth: int,
    date_col: str,
    horizon_col: str,
    target_col: str | None,
    model_col: str,
) -> None:
    """Populate records for the path_avg feature.

    Pairs are (target_scheme == "path_average") vs (target_scheme == "direct")
    within the same (model_id, feature_set, horizon, date, target) cell.
    """
    if "target_scheme" not in result_df.columns:
        warnings.warn(
            "marginal_contribution(..., feature='path_avg') requires a "
            "'target_scheme' column in result_df. Returning empty result.",
            stacklevel=3,
        )
        return

    for (model_id, horizon), grp in result_df.groupby([model_col, horizon_col]):
        sub_path = grp[grp["target_scheme"] == "path_average"]
        sub_direct = grp[grp["target_scheme"] == "direct"]

        if sub_path.empty or sub_direct.empty:
            continue

        # Merge on (date, target, feature_set) to get matched pairs
        merge_keys = [date_col, "feature_set"]
        if target_col is not None:
            merge_keys = [date_col, target_col, "feature_set"]

        merged = sub_path[merge_keys + ["oos_r2"]].merge(
            sub_direct[merge_keys + ["oos_r2"]],
            on=merge_keys,
            suffixes=("_path", "_direct"),
        )

        if merged.empty:
            continue

        valid = merged["oos_r2_path"].notna() & merged["oos_r2_direct"].notna()
        deltas = (merged.loc[valid, "oos_r2_path"] - merged.loc[valid, "oos_r2_direct"]).values

        if len(deltas) < 2:
            continue

        alpha, se = _newey_west_se(deltas.astype(float), hac_bandwidth)
        records.append(
            MarginalEffect(
                feature="path_avg",
                model=str(model_id),
                horizon=int(horizon),
                alpha=alpha,
                se=se,
                ci_low=alpha - _Z_95 * se,
                ci_high=alpha + _Z_95 * se,
                n_obs=int(valid.sum()),
            )
        )


# ---------------------------------------------------------------------------
# Convenience: all features at once
# ---------------------------------------------------------------------------


def marginal_contribution_all(
    result_df: pd.DataFrame,
    features: tuple[str, ...] | list[str] = ("MARX", "MAF", "F"),
    hac_bandwidth: int | None = None,
    date_col: str = "forecast_date",
    horizon_col: str = "horizon",
    target_col: str | None = "target",
    model_col: str = "model_id",
) -> pd.DataFrame:
    """Run :func:`marginal_contribution` for each feature and concatenate.

    Parameters
    ----------
    result_df : pd.DataFrame
        Result table with ``oos_r2`` column (from :func:`oos_r2_panel`).
    features : sequence of str
        Features to evaluate.  Defaults to ("MARX", "MAF", "F").
    hac_bandwidth : int or None
        Passed to each :func:`marginal_contribution` call.
    date_col, horizon_col, target_col, model_col : str
        Column name overrides (same as :func:`marginal_contribution`).

    Returns
    -------
    pd.DataFrame
        Stacked results with one row per (feature, model, horizon) cell.
    """
    frames: list[pd.DataFrame] = []
    for feat in features:
        df_feat = marginal_contribution(
            result_df=result_df,
            feature=feat,
            hac_bandwidth=hac_bandwidth,
            date_col=date_col,
            horizon_col=horizon_col,
            target_col=target_col,
            model_col=model_col,
        )
        if not df_feat.empty:
            frames.append(df_feat)

    if not frames:
        return pd.DataFrame(
            columns=["feature", "model", "horizon", "alpha", "se", "ci_low", "ci_high", "n_obs"]
        )

    return pd.concat(frames, ignore_index=True)
