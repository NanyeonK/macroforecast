"""Horse race evaluation for CLSS 2021 replication.

Produces relative RMSFE tables, best-spec identification, MCS membership, and
Diebold-Mariano tests vs a benchmark -- the four evaluation components required
for the CLSS 2021 replication.

The primary entry point is `horserace_summary`, which assembles all four
components into a `HorseRaceResult`. Individual table functions can also be
called independently.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from macrocast.evaluation.dm import dm_test
from macrocast.evaluation.mcs import mcs
from macrocast.evaluation.metrics import msfe


@dataclass
class HorseRaceResult:
    """Structured output of a full horse race evaluation.

    Attributes
    ----------
    rmsfe_table : pd.DataFrame
        Relative MSFE indexed by (model_id, feature_set), columns are horizons.
    best_specs : pd.DataFrame
        Per-horizon best (model_id, feature_set) with minimum RMSFE.
        Columns: horizon, model_id, feature_set, rmsfe.
    mcs_table : pd.DataFrame
        Boolean MCS membership indexed by (model_id, feature_set), columns horizons.
    dm_table : pd.DataFrame
        DM p-values vs benchmark indexed by (model_id, feature_set), columns horizons.
    """

    rmsfe_table: pd.DataFrame
    best_specs: pd.DataFrame
    mcs_table: pd.DataFrame
    dm_table: pd.DataFrame


def _detect_benchmark(result_df: pd.DataFrame) -> str:
    """Auto-detect the benchmark model_id from nonlinearity/regularization columns.

    Parameters
    ----------
    result_df : pd.DataFrame
        Output of ResultSet.to_dataframe().

    Returns
    -------
    str
        The unique model_id of the benchmark.

    Raises
    ------
    ValueError
        If zero or more than one model_id qualifies as the benchmark.
    """
    mask = (result_df["nonlinearity"] == "linear") & (
        result_df["regularization"] == "none"
    )
    candidate_ids = result_df.loc[mask, "model_id"].unique()
    if len(candidate_ids) == 0:
        raise ValueError(
            "No benchmark detected. Provide benchmark_id explicitly, or ensure a "
            "model with nonlinearity=='linear' and regularization=='none' exists."
        )
    if len(candidate_ids) > 1:
        raise ValueError(
            f"Multiple benchmark candidates detected: {list(candidate_ids)}. "
            "Provide benchmark_id explicitly."
        )
    return str(candidate_ids[0])


def relative_msfe_table(
    result_df: pd.DataFrame,
    benchmark_id: str | None = None,
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """Compute relative MSFE table indexed by (model_id, feature_set) x horizons.

    Parameters
    ----------
    result_df : pd.DataFrame
        Output of ResultSet.to_dataframe().
    benchmark_id : str or None
        model_id of the benchmark.  If None, auto-detected as the model with
        nonlinearity == "linear" and regularization == "none".
    horizons : list of int or None
        Subset of horizons.  Defaults to all unique horizons in result_df.

    Returns
    -------
    pd.DataFrame
        Relative MSFE indexed by (model_id, feature_set), columns are horizons.
        Values < 1 indicate improvement over benchmark.
    """
    if benchmark_id is None:
        benchmark_id = _detect_benchmark(result_df)

    if horizons is None:
        horizons = sorted(result_df["horizon"].unique().tolist())

    # Identify all unique (model_id, feature_set) pairs across the full result_df
    all_specs = (
        result_df[["model_id", "feature_set"]]
        .drop_duplicates()
        .sort_values(["model_id", "feature_set"])
    )
    index = pd.MultiIndex.from_frame(all_specs)

    rmsfe_records: dict[int, dict[tuple[str, str], float]] = {}

    for h in horizons:
        h_df = result_df.loc[result_df["horizon"] == h]

        # Compute benchmark MSFE for this horizon.
        # Use the benchmark model's own rows; if multiple feature_sets exist for
        # the benchmark, prefer feature_set == "" and fall back to the first found.
        bm_h = h_df.loc[h_df["model_id"] == benchmark_id]
        if bm_h.empty:
            raise ValueError(
                f"Benchmark model '{benchmark_id}' has no rows for horizon {h}."
            )

        if "" in bm_h["feature_set"].values:
            bm_fset = ""
        else:
            bm_fset = bm_h["feature_set"].iloc[0]
        bm_rows = bm_h.loc[bm_h["feature_set"] == bm_fset]

        bm_msfe = msfe(bm_rows["y_true"].to_numpy(), bm_rows["y_hat"].to_numpy())

        col_vals: dict[tuple[str, str], float] = {}
        for _, spec_row in all_specs.iterrows():
            mid = spec_row["model_id"]
            fset = spec_row["feature_set"]

            # Canonical benchmark row is 1.0 by definition (avoids floating-point
            # rounding artefacts from computing the same MSFE twice).
            if mid == benchmark_id and fset == bm_fset:
                col_vals[(mid, fset)] = 1.0
                continue

            rows = h_df.loc[(h_df["model_id"] == mid) & (h_df["feature_set"] == fset)]

            if rows.empty:
                col_vals[(mid, fset)] = float("nan")
                continue

            model_msfe = msfe(rows["y_true"].to_numpy(), rows["y_hat"].to_numpy())

            if bm_msfe == 0:
                col_vals[(mid, fset)] = float("nan")
            else:
                col_vals[(mid, fset)] = model_msfe / bm_msfe

        rmsfe_records[h] = col_vals

    # Assemble into a DataFrame with MultiIndex rows and horizon columns
    data = {
        h: [rmsfe_records[h].get((r["model_id"], r["feature_set"]), float("nan"))
            for _, r in all_specs.iterrows()]
        for h in horizons
    }
    table = pd.DataFrame(data, index=index)
    table.index.names = ["model_id", "feature_set"]
    table.columns.name = "horizon"
    return table


def best_spec_table(rmsfe_table: pd.DataFrame) -> pd.DataFrame:
    """Identify the best (model_id, feature_set) per horizon.

    Parameters
    ----------
    rmsfe_table : pd.DataFrame
        Output of relative_msfe_table().

    Returns
    -------
    pd.DataFrame with columns: [horizon, model_id, feature_set, rmsfe].
        One row per horizon, identifying the spec with the minimum relative MSFE.
    """
    rows = []
    for h in rmsfe_table.columns:
        col = rmsfe_table[h].dropna()
        if col.empty:
            rows.append({"horizon": h, "model_id": None, "feature_set": None, "rmsfe": float("nan")})
            continue
        best_idx = col.idxmin()
        rows.append({
            "horizon": h,
            "model_id": best_idx[0],
            "feature_set": best_idx[1],
            "rmsfe": col[best_idx],
        })
    return pd.DataFrame(rows, columns=["horizon", "model_id", "feature_set", "rmsfe"])


def mcs_membership_table(
    result_df: pd.DataFrame,
    horizons: list[int] | None = None,
    alpha: float = 0.10,
    block_size: int = 12,
    n_bootstrap: int = 1000,
) -> pd.DataFrame:
    """MCS membership table indexed by (model_id, feature_set) x horizons.

    Parameters
    ----------
    result_df : pd.DataFrame
    horizons : list of int or None
    alpha : float
    block_size : int
    n_bootstrap : int

    Returns
    -------
    pd.DataFrame of bool, indexed by (model_id, feature_set), columns horizons.
        True means the model is in the MCS at level alpha.
    """
    if horizons is None:
        horizons = sorted(result_df["horizon"].unique().tolist())

    all_specs = (
        result_df[["model_id", "feature_set"]]
        .drop_duplicates()
        .sort_values(["model_id", "feature_set"])
    )
    index = pd.MultiIndex.from_frame(all_specs)

    # Composite key for mcs() which expects a flat model identifier
    all_specs = all_specs.copy()
    all_specs["composite_id"] = all_specs["model_id"] + "|" + all_specs["feature_set"]
    composite_to_spec: dict[str, tuple[str, str]] = {
        row["composite_id"]: (row["model_id"], row["feature_set"])
        for _, row in all_specs.iterrows()
    }

    membership_data: dict[int, dict[tuple[str, str], bool | None]] = {}

    for h in horizons:
        h_df = result_df.loc[result_df["horizon"] == h].copy()

        # Build loss DataFrame in the format expected by mcs()
        h_df = h_df.copy()
        h_df["composite_id"] = h_df["model_id"] + "|" + h_df["feature_set"]
        h_df["squared_error"] = (h_df["y_true"] - h_df["y_hat"]) ** 2

        loss_df = h_df[["composite_id", "squared_error", "forecast_date"]].rename(
            columns={"composite_id": "model_id"}
        )

        mcs_result = mcs(
            loss_df=loss_df,
            alpha=alpha,
            block_size=block_size,
            n_bootstrap=n_bootstrap,
            loss_col="squared_error",
            model_col="model_id",
            date_col="forecast_date",
        )

        included_set = set(mcs_result.included)
        h_composite_ids = set(h_df["composite_id"].unique())
        col_vals: dict[tuple[str, str], bool | None] = {}
        for cid, spec in composite_to_spec.items():
            if cid in h_composite_ids:
                col_vals[spec] = cid in included_set
            else:
                # Spec absent for this horizon — NaN rather than False to avoid
                # conflating "not in MCS" with "did not forecast".
                col_vals[spec] = None

        membership_data[h] = col_vals

    data = {
        h: [membership_data[h].get((r["model_id"], r["feature_set"]), None)
            for _, r in all_specs.iterrows()]
        for h in horizons
    }
    table = pd.DataFrame(data, index=index)
    table.index.names = ["model_id", "feature_set"]
    table.columns.name = "horizon"
    return table


def dm_vs_benchmark_table(
    result_df: pd.DataFrame,
    benchmark_id: str | None = None,
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """DM test p-values vs benchmark for each (model_id, feature_set) x horizon.

    Parameters
    ----------
    result_df : pd.DataFrame
    benchmark_id : str or None
    horizons : list of int or None

    Returns
    -------
    pd.DataFrame of float (p-values), indexed by (model_id, feature_set), columns horizons.
        NaN for the benchmark row (cannot test against itself).
    """
    if benchmark_id is None:
        benchmark_id = _detect_benchmark(result_df)

    if horizons is None:
        horizons = sorted(result_df["horizon"].unique().tolist())

    all_specs = (
        result_df[["model_id", "feature_set"]]
        .drop_duplicates()
        .sort_values(["model_id", "feature_set"])
    )
    index = pd.MultiIndex.from_frame(all_specs)

    dm_data: dict[int, dict[tuple[str, str], float]] = {}

    for h in horizons:
        h_df = result_df.loc[result_df["horizon"] == h]

        # Extract benchmark forecasts for this horizon, selecting the single feature_set.
        # Prefer feature_set == "" if available, otherwise take the first one found.
        bm_h = h_df.loc[h_df["model_id"] == benchmark_id]
        if bm_h.empty:
            # No benchmark data for this horizon; fill column with NaN
            dm_data[h] = {
                (r["model_id"], r["feature_set"]): float("nan")
                for _, r in all_specs.iterrows()
            }
            continue

        if "" in bm_h["feature_set"].values:
            bm_rows = bm_h.loc[bm_h["feature_set"] == ""].sort_values("forecast_date")
        else:
            first_fs = bm_h["feature_set"].iloc[0]
            bm_rows = bm_h.loc[bm_h["feature_set"] == first_fs].sort_values("forecast_date")

        bm_indexed = bm_rows.set_index("forecast_date")[["y_true", "y_hat"]]

        col_vals: dict[tuple[str, str], float] = {}
        for _, spec_row in all_specs.iterrows():
            mid = spec_row["model_id"]
            fset = spec_row["feature_set"]

            # Benchmark vs itself: NaN by convention
            if mid == benchmark_id:
                col_vals[(mid, fset)] = float("nan")
                continue

            model_rows = (
                h_df.loc[(h_df["model_id"] == mid) & (h_df["feature_set"] == fset)]
                .sort_values("forecast_date")
                .set_index("forecast_date")[["y_true", "y_hat"]]
            )

            if model_rows.empty:
                col_vals[(mid, fset)] = float("nan")
                continue

            # Inner join on forecast_date to align arrays
            aligned = model_rows.join(
                bm_indexed.rename(columns={"y_true": "y_true_bm", "y_hat": "y_hat_bm"}),
                how="inner",
            )

            if len(aligned) < 2:
                col_vals[(mid, fset)] = float("nan")
                continue

            dm_result = dm_test(
                y_true=aligned["y_true"].to_numpy(),
                y_hat_1=aligned["y_hat"].to_numpy(),
                y_hat_2=aligned["y_hat_bm"].to_numpy(),
                h=h,
            )
            col_vals[(mid, fset)] = dm_result.p_value

        dm_data[h] = col_vals

    data = {
        h: [dm_data[h].get((r["model_id"], r["feature_set"]), float("nan"))
            for _, r in all_specs.iterrows()]
        for h in horizons
    }
    table = pd.DataFrame(data, index=index)
    table.index.names = ["model_id", "feature_set"]
    table.columns.name = "horizon"
    return table


def horserace_summary(
    result_df: pd.DataFrame,
    benchmark_id: str | None = None,
    horizons: list[int] | None = None,
    mcs_alpha: float = 0.10,
) -> HorseRaceResult:
    """Run all four evaluation components and return a HorseRaceResult.

    Parameters
    ----------
    result_df : pd.DataFrame
        Output of ResultSet.to_dataframe().
    benchmark_id : str or None
        Benchmark model ID. Auto-detected if None.
    horizons : list of int or None
        Subset of horizons. Defaults to all.
    mcs_alpha : float
        MCS significance level.

    Returns
    -------
    HorseRaceResult
    """
    # Auto-detect once so downstream functions use the same benchmark
    if benchmark_id is None:
        benchmark_id = _detect_benchmark(result_df)

    if horizons is None:
        horizons = sorted(result_df["horizon"].unique().tolist())

    rmsfe_tbl = relative_msfe_table(result_df, benchmark_id=benchmark_id, horizons=horizons)
    best = best_spec_table(rmsfe_tbl)
    mcs_tbl = mcs_membership_table(result_df, horizons=horizons, alpha=mcs_alpha)
    dm_tbl = dm_vs_benchmark_table(result_df, benchmark_id=benchmark_id, horizons=horizons)

    return HorseRaceResult(
        rmsfe_table=rmsfe_tbl,
        best_specs=best,
        mcs_table=mcs_tbl,
        dm_table=dm_tbl,
    )
