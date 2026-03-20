"""CLSS 2021 focused overnight replication.

Runs RF horse race across 6 information sets, 4 targets, 4 horizons.
Validates that MARX transformation improves RF forecasting.

Specifically replicates:
  - Fig 1/2 direction: Does MARX transformation help Random Forest?
  - Relative RMSFE: Does F-X-MARX beat F?

Usage: uv run python scripts/clss2021_overnight_run.py
Results saved to: ~/.macrocast/results/clss2021_overnight/
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from macrocast.data import load_fred_md
from macrocast.pipeline import (
    CVScheme,
    FeatureSpec,
    HorseRaceGrid,
    LossFunction,
    ModelSpec,
    Regularization,
    RFModel,
    GBModel,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RESULTS_DIR = Path.home() / ".macrocast" / "results" / "clss2021_overnight"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------

TARGETS: list[str] = ["INDPRO", "PAYEMS", "UNRATE", "CPIAUCSL"]
HORIZONS: list[int] = [1, 3, 6, 12]
OOS_START: str = "1999-01-01"
OOS_END: str = "2017-12-01"
N_JOBS: int = 4

# FeatureSpec shared hyper-parameters (CLSS 2021 defaults)
_FEAT_KWARGS: dict = dict(n_factors=8, n_lags=4, p_marx=4)

# ---------------------------------------------------------------------------
# 6 information sets (Table 1 / Fig 1 in CLSS 2021)
# ---------------------------------------------------------------------------
# label convention matches _auto_label() in experiment.py, but we set
# explicit labels to guarantee exact CLSS naming regardless of logic changes.

FEATURE_SPECS: list[FeatureSpec] = [
    # F: factors only, no raw X, no MARX
    FeatureSpec(
        use_factors=True,
        include_raw_x=False,
        use_marx=False,
        label="F",
        **_FEAT_KWARGS,
    ),
    # F-X: factors + raw predictors, no MARX
    FeatureSpec(
        use_factors=True,
        include_raw_x=True,
        use_marx=False,
        label="F-X",
        **_FEAT_KWARGS,
    ),
    # X-MARX: MARX features only (no PCA, no raw X)
    FeatureSpec(
        use_factors=False,
        include_raw_x=False,
        use_marx=True,
        marx_for_pca=False,
        label="X-MARX",
        **_FEAT_KWARGS,
    ),
    # F-MARX: factors + MARX features (MARX computed from raw X, not used for PCA)
    FeatureSpec(
        use_factors=True,
        include_raw_x=False,
        use_marx=True,
        marx_for_pca=False,
        label="F-MARX",
        **_FEAT_KWARGS,
    ),
    # F-X-MARX: factors + raw X + MARX (the dominant CLSS 2021 information set)
    FeatureSpec(
        use_factors=True,
        include_raw_x=True,
        use_marx=True,
        marx_for_pca=False,
        label="F-X-MARX",
        **_FEAT_KWARGS,
    ),
    # MAF: factors from MARX-transformed X (MARX used for PCA + MAF flag)
    FeatureSpec(
        use_factors=True,
        include_raw_x=False,
        use_marx=True,
        marx_for_pca=True,
        use_maf=True,
        label="MAF",
        **_FEAT_KWARGS,
    ),
]

# ---------------------------------------------------------------------------
# Model spec: RF with fast CV settings
# ---------------------------------------------------------------------------

RF_SPEC = ModelSpec(
    model_cls=RFModel,
    model_kwargs=dict(
        n_estimators=100,
        max_depth_grid=[3, 5],
        min_samples_leaf_grid=[5, 10],
        cv_folds=3,
    ),
    regularization=Regularization.NONE,
    cv_scheme=CVScheme.KFOLD(k=3),
    loss_function=LossFunction.L2,
    model_id="rf",
)

GB_SPEC = ModelSpec(
    model_cls=GBModel,
    model_kwargs=dict(
        n_estimators=100,
        max_depth_grid=[3, 5],
        learning_rate_grid=[0.05, 0.1],
        cv_folds=3,
    ),
    regularization=Regularization.NONE,
    cv_scheme=CVScheme.KFOLD(k=3),
    loss_function=LossFunction.L2,
    model_id="gb",
)

MODEL_SPECS: list[ModelSpec] = [RF_SPEC, GB_SPEC]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data(vintage: str = "2018-02") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load FRED-MD and return (panel_stationary, panel_levels).

    Attempts the requested vintage first. Falls back to the current release
    if the vintage is unavailable.

    Parameters
    ----------
    vintage : str
        FRED-MD vintage in YYYY-MM format.

    Returns
    -------
    panel_stationary : pd.DataFrame
        Stationary-transformed predictor panel, DatetimeIndex (monthly).
    panel_levels : pd.DataFrame
        Untransformed levels panel (same shape), for FeatureSpecs that need
        include_levels=True.  Not used in this script's 6 specs but wired
        in for completeness.
    """
    log.info("Loading FRED-MD vintage=%s ...", vintage)
    try:
        mf_levels = load_fred_md(vintage=vintage)
        log.info("Loaded vintage %s: %d obs x %d vars", vintage, *mf_levels.data.shape)
    except Exception as exc:
        log.warning(
            "Could not load vintage %s (%s). Falling back to current release.",
            vintage,
            exc,
        )
        mf_levels = load_fred_md(vintage=None)
        log.info("Loaded current release: %d obs x %d vars", *mf_levels.data.shape)

    # Levels panel (before stationarity transforms)
    panel_levels: pd.DataFrame = mf_levels.data.copy()

    # Stationary panel via McCracken-Ng tcodes
    mf_stat = mf_levels.transform()
    panel_stationary: pd.DataFrame = mf_stat.data.copy()

    log.info(
        "Stationary panel: %d obs x %d vars, date range %s to %s",
        *panel_stationary.shape,
        panel_stationary.index.min(),
        panel_stationary.index.max(),
    )

    return panel_stationary, panel_levels


# ---------------------------------------------------------------------------
# Relative RMSFE table
# ---------------------------------------------------------------------------


def compute_relative_rmsfe(df: pd.DataFrame) -> pd.DataFrame:
    """Compute relative RMSFE for each feature set vs benchmark F.

    Relative RMSFE = RMSFE(model) / RMSFE(F). Values < 1 indicate
    improvement over the F benchmark.

    Parameters
    ----------
    df : pd.DataFrame
        Combined results DataFrame from ResultSet.to_dataframe().

    Returns
    -------
    pd.DataFrame
        Pivot table: rows = feature_set, columns = horizon, values = rel_rmsfe.
        Index is sorted by the order in FEATURE_SPECS.
    """
    if df.empty:
        log.warning("Results DataFrame is empty; cannot compute RMSFE table.")
        return pd.DataFrame()

    # MSFE per feature_set x target x horizon
    df = df.copy()
    df["sq_err"] = (df["forecast"] - df["realization"]) ** 2

    msfe = (
        df.groupby(["feature_set", "target", "horizon"])["sq_err"]
        .mean()
        .reset_index()
        .rename(columns={"sq_err": "msfe"})
    )

    # Benchmark = feature_set "F"
    bench = msfe[msfe["feature_set"] == "F"][["target", "horizon", "msfe"]].rename(
        columns={"msfe": "msfe_bench"}
    )
    msfe = msfe.merge(bench, on=["target", "horizon"], how="left")
    msfe["rmsfe"] = msfe["msfe"] ** 0.5
    msfe["rmsfe_bench"] = msfe["msfe_bench"] ** 0.5
    msfe["rel_rmsfe"] = msfe["rmsfe"] / msfe["rmsfe_bench"]

    # Average across targets then pivot: feature_set x horizon
    summary = (
        msfe.groupby(["feature_set", "horizon"])["rel_rmsfe"]
        .mean()
        .reset_index()
    )
    pivot = summary.pivot(index="feature_set", columns="horizon", values="rel_rmsfe")

    # Sort rows in CLSS presentation order
    ordered_labels = [s.label for s in FEATURE_SPECS]
    pivot = pivot.reindex([l for l in ordered_labels if l in pivot.index])

    return pivot


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    run_start = time.time()
    log.info("=== CLSS 2021 overnight replication started ===")
    log.info("Targets: %s", TARGETS)
    log.info("Horizons: %s", HORIZONS)
    log.info("Feature sets: %s", [s.label for s in FEATURE_SPECS])
    log.info("OOS window: %s — %s", OOS_START, OOS_END)
    log.info("Results dir: %s", RESULTS_DIR)

    # Use vintage=None (current FRED-MD release); OOS evaluation ends 2017-12
    # to match the CLSS 2021 paper period.
    panel_stat, panel_levels = load_data(vintage=None)

    all_result_dfs: list[pd.DataFrame] = []

    for tgt in TARGETS:
        tgt_start = time.time()
        log.info("--- Target: %s  [%s] ---", tgt, datetime.now().strftime("%H:%M:%S"))

        # Guard: check target variable is present in the panel
        if tgt not in panel_stat.columns:
            log.error(
                "Target '%s' not found in panel columns. Skipping.", tgt
            )
            continue

        target_series: pd.Series = panel_stat[tgt].dropna()
        # Predictor panel excludes the target variable (avoid information
        # leakage; CLSS 2021 follows this convention)
        predictor_panel: pd.DataFrame = panel_stat.drop(columns=[tgt])
        predictor_levels: pd.DataFrame = panel_levels.drop(columns=[tgt], errors="ignore")

        try:
            grid = HorseRaceGrid(
                panel=predictor_panel,
                target=target_series,
                horizons=HORIZONS,
                model_specs=MODEL_SPECS,
                feature_specs=FEATURE_SPECS,
                panel_levels=predictor_levels,
                oos_start=OOS_START,
                oos_end=OOS_END,
                n_jobs=N_JOBS,
            )

            result_set = grid.run()
            result_df = result_set.to_dataframe()

            if result_df.empty:
                log.warning("HorseRaceGrid returned empty results for target %s.", tgt)
            else:
                # Tag with target name so we can identify rows after concatenation
                result_df["target"] = tgt

                # Save intermediate result
                out_path = RESULTS_DIR / f"{tgt}_results.parquet"
                result_df.to_parquet(out_path, index=False)
                log.info("Saved intermediate results: %s", out_path)

                all_result_dfs.append(result_df)

        except Exception as exc:
            log.exception("ERROR processing target %s: %s", tgt, exc)
            log.info("Continuing with remaining targets...")
            continue

        elapsed = time.time() - tgt_start
        log.info(
            "Target %s done in %.1f s (%.1f min)", tgt, elapsed, elapsed / 60
        )

    # ------------------------------------------------------------------
    # Combine and summarise
    # ------------------------------------------------------------------

    if not all_result_dfs:
        log.error("No results collected. Exiting.")
        return

    log.info("Combining results from %d targets ...", len(all_result_dfs))
    combined = pd.concat(all_result_dfs, ignore_index=True)

    combined_path = RESULTS_DIR / "combined_results.parquet"
    combined.to_parquet(combined_path, index=False)
    log.info("Saved combined results: %s  (%d rows)", combined_path, len(combined))

    # ------------------------------------------------------------------
    # Relative RMSFE table
    # ------------------------------------------------------------------

    log.info("\n=== Relative RMSFE (vs F benchmark, averaged over targets) ===")
    rmsfe_table = compute_relative_rmsfe(combined)

    if not rmsfe_table.empty:
        pd.set_option("display.float_format", "{:.4f}".format)
        pd.set_option("display.max_columns", None)
        print("\n" + rmsfe_table.to_string())
        print()

        # Also save the summary table
        rmsfe_path = RESULTS_DIR / "rmsfe_summary.parquet"
        rmsfe_table.to_parquet(rmsfe_path)
        log.info("Saved RMSFE summary: %s", rmsfe_path)
    else:
        log.warning("RMSFE table is empty; check results.")

    total = time.time() - run_start
    log.info(
        "=== Run complete in %.1f s (%.1f min) ===", total, total / 60
    )


if __name__ == "__main__":
    main()
