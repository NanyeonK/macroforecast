"""Regime-conditional forecast evaluation.

Splits the OOS evaluation period into regimes defined by quantiles of a
regime indicator series (e.g. VXO for uncertainty, USREC for NBER recessions)
and computes MSFE and Relative MSFE within each regime.

Typical use: compare model accuracy in low vs. high uncertainty periods,
or recession vs. expansion, following CLSS 2022 Table 5.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RegimeResult:
    """Regime-conditional MSFE results.

    Attributes
    ----------
    regime_labels : list of str
        Names of each regime bin.
    msfe_by_regime : dict {model_id: {regime_label: msfe}}
    relative_msfe_by_regime : dict {model_id: {regime_label: rel_msfe}}
        Relative to the benchmark model.
    n_obs_by_regime : dict {regime_label: int}
        Number of OOS observations in each regime.
    summary_df : pd.DataFrame
        Tidy table.
    """

    regime_labels: list[str]
    msfe_by_regime: dict[str, dict[str, float]]
    relative_msfe_by_regime: dict[str, dict[str, float]]
    n_obs_by_regime: dict[str, int]
    summary_df: pd.DataFrame


def regime_conditional_msfe(
    result_df: pd.DataFrame,
    regime_series: pd.Series,
    n_quantiles: int = 3,
    regime_labels: list[str] | None = None,
    benchmark_model_id: str = "linear__none__bic__l2",
    horizon: int | None = None,
    date_col: str = "forecast_date",
    model_col: str = "model_id",
) -> RegimeResult:
    """Compute regime-conditional MSFE.

    Parameters
    ----------
    result_df : pd.DataFrame
        Forecast result table (from ResultSet.to_dataframe()).
    regime_series : pd.Series
        Indicator series indexed by date (same frequency as result_df).
        Quantiles of this series define the regimes.
        E.g., VXO uncertainty index or USREC binary recession indicator.
    n_quantiles : int
        Number of quantile bins.  Default 3 (low / medium / high).
        Ignored if the indicator is binary (only 0/1 values).
    regime_labels : list of str or None
        Custom regime names.  Length must match n_quantiles.
    benchmark_model_id : str
        model_id of the AR benchmark for relative MSFE.
    horizon : int or None
        Restrict to a single forecast horizon.

    Returns
    -------
    RegimeResult
    """
    df = result_df.copy()
    if horizon is not None:
        df = df[df["horizon"] == horizon]

    df[date_col] = pd.to_datetime(df[date_col])
    regime_series.index = pd.to_datetime(regime_series.index)

    # Merge regime indicator into result table
    df = df.merge(
        regime_series.rename("__regime__"),
        left_on=date_col,
        right_index=True,
        how="left",
    )
    df = df.dropna(subset=["__regime__"])

    # Determine regime bins
    unique_vals = df["__regime__"].nunique()
    if unique_vals <= 2:
        # Binary indicator (e.g., USREC)
        bins_labels = (
            ["expansion", "recession"] if regime_labels is None else regime_labels
        )
        df["__regime_bin__"] = (
            df["__regime__"]
            .map(
                {
                    df["__regime__"].unique()[0]: bins_labels[0],
                    df["__regime__"].unique()[-1]: bins_labels[-1],
                }
            )
            .astype(str)
        )
    else:
        if regime_labels is None:
            regime_labels = [f"Q{i + 1}" for i in range(n_quantiles)]
        quantile_edges = np.linspace(0, 1, n_quantiles + 1)
        quantile_values = df["__regime__"].quantile(quantile_edges).values
        # pd.cut with computed quantile edges
        df["__regime_bin__"] = pd.cut(
            df["__regime__"],
            bins=quantile_values,
            labels=regime_labels,
            include_lowest=True,
        ).astype(str)
        bins_labels = regime_labels

    df["__se__"] = (df["y_true"] - df["y_hat"]) ** 2

    # MSFE per model per regime
    grp = df.groupby([model_col, "__regime_bin__"])["__se__"].mean()

    # Benchmark MSFE per regime
    if benchmark_model_id not in df[model_col].unique():
        raise ValueError(
            f"Benchmark '{benchmark_model_id}' not in result_df. "
            f"Available: {list(df[model_col].unique())}"
        )
    bench_msfe = (
        df[df[model_col] == benchmark_model_id]
        .groupby("__regime_bin__")["__se__"]
        .mean()
    )

    model_ids = df[model_col].unique().tolist()
    msfe_by_regime: dict[str, dict[str, float]] = {m: {} for m in model_ids}
    rel_msfe_by_regime: dict[str, dict[str, float]] = {m: {} for m in model_ids}

    for model_id in model_ids:
        for regime in bins_labels:
            try:
                val = float(grp.loc[(model_id, regime)])
            except KeyError:
                val = float("nan")
            msfe_by_regime[model_id][regime] = val
            bm = float(bench_msfe.get(regime, float("nan")))
            rel_msfe_by_regime[model_id][regime] = val / bm if bm > 0 else float("nan")

    n_obs = df.groupby("__regime_bin__")["__se__"].count().to_dict()
    n_obs_by_regime = {r: int(n_obs.get(r, 0)) for r in bins_labels}

    # Build summary DataFrame
    rows = []
    for model_id in model_ids:
        for regime in bins_labels:
            rows.append(
                {
                    "model_id": model_id,
                    "regime": regime,
                    "msfe": msfe_by_regime[model_id][regime],
                    "rel_msfe": rel_msfe_by_regime[model_id][regime],
                    "n_obs": n_obs_by_regime[regime],
                }
            )
    summary = pd.DataFrame(rows)

    return RegimeResult(
        regime_labels=bins_labels,
        msfe_by_regime=msfe_by_regime,
        relative_msfe_by_regime=rel_msfe_by_regime,
        n_obs_by_regime=n_obs_by_regime,
        summary_df=summary,
    )
