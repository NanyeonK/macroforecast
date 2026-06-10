"""GCLS (2021) "Macroeconomic data transformations matter" -- leak-free PIPELINE run.

One process per target (``--target``) so the 10 targets parallelise trivially. For
the given target and each requested policy we build ONE ``pipeline_spec`` per horizon
(the pipeline takes a single window, so -- exactly like ml_useful_2022/run_full.py --
we loop horizons and inject the per-horizon window) with all arms:

    FM (benchmark, factor model) + AR + five random-forest feature cases.

Preprocessing is SPEC-LEVEL and refit per origin on the retrain cadence
(``origin_available`` / ``on_retrain``), shared across arms via the per-target cache.
This removes the full-sample EM look-ahead present in the standalone single script.

LEVEL columns
-------------
The "Level" feature cases need raw (untransformed) FRED-MD level columns. With
spec-level preprocessing the pipeline applies the official t-code transform to every
column, so we cannot keep raw levels by simply joining them post-hoc as the single
script does. Instead we PRE-BUILD an augmented bundle: every raw FRED-MD column is
duplicated with a ``LEVEL__`` prefix and assigned t-code 1 (identity) in the bundle
metadata. The official transform then passes the LEVEL__ columns through unchanged,
and the per-origin EM imputation handles any missing values leak-free. This is
actually MORE leak-free than the single script, which selected "complete" level
columns using full-sample missingness. We still restrict the level FEATURE block to
columns complete over the processed index (a structural which-series choice, not a
data-value leak), matching the single script's predictor set.

Horizon consolidation (performance)
-----------------------------------
Each (target, policy) builds ONE ``pipeline_spec`` carrying ALL horizons
(``horizons=[1, 3, 6, 9, 12, 24]``) and calls ``run_pipeline`` ONCE. The per-origin
EM imputation is HORIZON-INDEPENDENT (it imputes the panel only up to each origin),
so the runner computes it once per origin and shares the FittedPreprocessor across
all horizons (and arms) via the cross-arm preprocessing cache keyed on
``origin_pos``. The consolidated report is then split by horizon into per-cell lean
parquet + accuracy CSV so resume granularity stays per (target, policy, horizon).

Smoke (``--smoke``): horizons {1, 3}, arms {FM, AR, RF_F-Level, RF_MARX},
test_end 1981-12.
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import warnings

warnings.simplefilter("ignore")

import pandas as pd  # noqa: E402

# Make the repo root importable so ``scripts.replication...`` resolves.
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import macroforecast as mf  # noqa: E402
from macroforecast.pipeline import EvalSpec, pipeline_spec, run_pipeline  # noqa: E402

from scripts.replication.gcls_2021_pipeline.registry import (  # noqa: E402
    CORE_ARMS,
    TARGET_MAP,
    YGROWTH_PREFIX,
    gcls_arms,
    gcls_targets,
    ygrowth_column,
)

import numpy as np  # noqa: E402

HORIZONS = (1, 3, 6, 9, 12, 24)

# Per-target one-period growth object for the forecast TARGET (the paper's "average
# growth rate" is the horizon-average of these). For UNRATE the paper's one-period
# object is the first difference of the level (FRED-MD t-code 2); for every other
# target it is the log-difference (Delta log). This is decoupled from the predictor
# t-codes on purpose: the paper transforms predictors X with the McCracken-Ng
# t-codes but forecasts the target Y as a growth/inflation rate.
_TARGET_GROWTH_KIND: dict[str, str] = {
    "UNRATE": "diff",  # Y_t - Y_{t-1}
}
# everything else defaults to "log_diff" (log Y_t - log Y_{t-1})


def _one_period_growth(level: pd.Series, kind: str) -> pd.Series:
    """One-period stationary growth object built from the RAW level series.

    ``"diff"``     -> first difference  Y_t - Y_{t-1}            (UNRATE).
    ``"log_diff"`` -> log difference    log Y_t - log Y_{t-1}   (all others).

    This is exactly the FRED-MD t-code-2 / t-code-5 transform, applied here directly
    to the raw level so the TARGET object is correct regardless of the column's own
    FRED-MD t-code (HOUST is t-code 4 = log-level, the I(2) price series are t-code 6
    = second log-difference; neither yields the paper's growth target if averaged).
    """
    values = level.astype(float)
    if kind == "diff":
        return values.diff()
    if kind == "log_diff":
        # log is only defined for strictly positive levels; non-positive entries
        # (none in these 9 series, but guard anyway) yield NaN, which the per-origin
        # EM imputation never touches for a complete YGROWTH__ column.
        positive = values.where(values > 0)
        return np.log(positive) - np.log(positive.shift(1))
    raise ValueError(f"unknown growth kind {kind!r}")



# --- parquet serialisation guard ---------------------------------------------
# The pipeline forecast frame carries object columns holding dicts/structs
# (params, model_selection, fixed_model_params, stored_model, window, ...). When
# such a column is empty pyarrow raises "Cannot write struct type ... with no
# child field to Parquet". Drop the non-scalar metadata columns and stringify any
# remaining object cells before writing; all columns needed for accuracy and the
# treatment regression (prediction, actual, contender, arm, target, horizon,
# origin, date, forecast_policy, target_transform, combined) are scalar and kept.
_PARQUET_DROP_COLS = {
    "params", "model_selection", "fixed_model_params", "stored_model", "window",
    "preprocessed", "combination", "quantile_predictions", "model_spec",
    "trained_model", "variance_prediction",
}


def _sanitize_for_parquet(df):
    keep = [c for c in df.columns if c not in _PARQUET_DROP_COLS]
    out = df[keep].copy()
    for c in list(out.columns):
        if out[c].dtype == object:
            out[c] = out[c].map(
                lambda v: v if (v is None or isinstance(v, (str, int, float, bool))) else str(v)
            )
    return out
# -----------------------------------------------------------------------------

def _augmented_bundle(raw):
    """Return a bundle whose panel also carries ``LEVEL__`` raw-level columns.

    Each LEVEL__ column is a copy of the corresponding raw FRED-MD column and is
    assigned t-code 1 (identity) so the official transform leaves it untouched. The
    transformed (official t-code) columns keep their original names.

    Performance: we duplicate ONLY the columns that are complete over the raw index
    (i.e. exactly the level-predictor set returned by ``_level_predictors``). The
    incomplete columns are never used as LEVEL__ features, so duplicating them would
    only enlarge the EM/factor SVD matrix at every origin for no benefit. The
    complete LEVEL__ copies carry no missing values, so the per-origin EM imputation
    leaves them untouched (it only widens the SVD; it never alters their values),
    which keeps the construction leak-free.
    """
    panel = raw.panel.copy()
    complete_cols = [c for c in raw.panel.columns if raw.panel[c].notna().all()]
    levels = raw.panel[complete_cols].add_prefix("LEVEL__")

    # TARGET growth columns (the paper's "average growth rate" object). For EVERY
    # target alias we build a dedicated ``YGROWTH__<col>`` column = the one-period
    # growth computed from the RAW level (Delta log, or Delta for UNRATE), decoupled
    # from the predictor t-codes. These columns are the stationary one-period object
    # already, so they get t-code 1 (identity) and the official transform passes them
    # through unchanged; the forecast policy's ``average_value`` then builds
    # (1/h) sum_{h'} growth_{t+h'} = the paper's direct h-step target. They are EXCLUDED
    # from the predictor set by ``transformed_predictors`` (which filters the
    # YGROWTH__ prefix) so they cannot leak as features.
    growth_frames: list[pd.Series] = []
    for col in TARGET_MAP.values():
        kind = _TARGET_GROWTH_KIND.get(col, "log_diff")
        growth = _one_period_growth(raw.panel[col], kind)
        growth.name = ygrowth_column(col)
        growth_frames.append(growth)
    growths = pd.concat(growth_frames, axis=1)

    new_panel = pd.concat([panel, levels, growths], axis=1)
    codes = dict(raw.metadata.get("transform_codes", {}))
    for column in complete_cols:
        codes[f"LEVEL__{column}"] = 1
    for col in TARGET_MAP.values():
        codes[ygrowth_column(col)] = 1
    # IMPORTANT (performance): the panel ``.attrs`` carries a large per-panel report
    # (input/output column lists, nested metadata copies). pandas deep-copies attrs
    # on virtually every operation via ``__finalize__``; doubling the columns and
    # the t-code map would blow up that deepcopy cost O(origins x arms x columns)
    # and dominate runtime. We therefore keep ONLY the t-code map the preprocessing
    # stage needs and a minimal metadata dict (``_resolve_transform_codes`` reads
    # ``metadata['transform_codes']`` first, so this is sufficient and leak-free).
    metadata = {
        k: raw.metadata.get(k)
        for k in ("dataset", "frequency", "version_mode", "vintage", "data_through")
        if k in raw.metadata
    }
    metadata["transform_codes"] = codes
    new_panel.attrs = {"macroforecast_transform_codes": dict(codes)}
    return dataclasses.replace(raw, panel=new_panel, metadata=metadata)


def _level_predictors(raw) -> list[str]:
    """LEVEL__ columns complete over the raw panel index (paper predictor set).

    The single script selected raw level columns with no missing values over the
    processed index. We reproduce that here on the raw index (which-series choice).
    """
    complete = [c for c in raw.panel.columns if raw.panel[c].notna().all()]
    return [f"LEVEL__{c}" for c in complete]


def _slice_by_horizon(frame, horizon: int):
    """Return the rows of ``frame`` for one horizon (or None if absent/empty)."""
    if frame is None or getattr(frame, "empty", True):
        return None
    if "horizon" not in frame.columns:
        return frame
    out = frame[frame["horizon"] == horizon]
    return out if not out.empty else None


def _write_cell_outputs(
    out_dir: Path, stem: str, forecasts, accuracy, significance, horizon: int
) -> dict[str, float]:
    """Persist forecast parquet + accuracy/significance CSVs for one (target, policy, horizon) cell.

    ``forecasts``/``accuracy``/``significance`` are the SINGLE-HORIZON slices of the
    consolidated multi-horizon report (see ``run_target``), so each horizon's lean
    parquet and accuracy CSV are written separately to keep resume granularity
    per-cell while the EM imputation was computed once per origin for all horizons.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    _sanitize_for_parquet(forecasts).to_parquet(out_dir / f"{stem}.parquet")

    accuracy = accuracy.assign(horizon=horizon)
    accuracy.to_csv(out_dir / f"{stem}_accuracy.csv", index=False)

    if significance is not None and not significance.empty:
        significance.assign(horizon=horizon).to_csv(
            out_dir / f"{stem}_significance.csv", index=False
        )

    # best contender by relative_mse among non-benchmark, benchmark-present rows
    contenders = accuracy[(~accuracy["is_benchmark"]) & accuracy["benchmark_present"]]
    best_name = ""
    best_rel = float("nan")
    if not contenders.empty:
        best = contenders.sort_values("relative_mse").iloc[0]
        best_name = str(best["contender"])
        best_rel = float(best["relative_mse"])
    return {"best": best_name, "relative_mse": best_rel}


def run_target(
    target_alias: str,
    *,
    policies: list[str],
    vintage: str,
    data_csv: str | None,
    out_root: Path,
    skip_existing: bool,
    smoke: bool,
    core_arms_only: bool = False,
) -> None:
    target = TARGET_MAP[target_alias]
    print(f"macroforecast {mf.__version__}", flush=True)

    raw = (
        mf.data.load_fred_md(local_source=data_csv)
        if data_csv
        else mf.data.load_fred_md(vintage=vintage)
    )
    bundle = _augmented_bundle(raw)

    # The forecast TARGET column is the dedicated one-period growth column built from
    # the raw level (see _augmented_bundle). It is also the feature target so that the
    # target lags and the target-derived MARX_y/MAF_y blocks operate on the growth
    # object (consistent with the forecast object); the PREDICTOR blocks still use the
    # official t-coded ``transformed_predictors``.
    target_growth = ygrowth_column(target)

    # Transformed predictors = all official-transform columns except the target and
    # the engineered YGROWTH__/LEVEL__ columns. The YGROWTH__ columns MUST be excluded
    # so the realized target object never leaks into the predictor set.
    transformed_predictors = [
        c
        for c in raw.panel.columns
        if c != target and not str(c).startswith(YGROWTH_PREFIX)
    ]
    level_predictors = _level_predictors(raw)

    # Shared spec-level preprocessing (leak-free, refit on the retrain cadence).
    pp = mf.preprocessing.preprocess_spec(
        transform="official",
        impute="em_factor",
        outliers="iqr",
        outlier_action="flag_as_nan",
        standardize="none",
    )
    pp_policy = mf.window.stage_policy("origin_available", update="on_retrain")
    feat_policy = mf.window.stage_policy("fit_window", update="on_retrain")

    # CONSOLIDATED HORIZONS (performance): build ONE pipeline_spec carrying ALL
    # horizons and call run_pipeline ONCE per (target, policy). The per-origin EM
    # imputation is horizon-independent (it only imputes the panel up to each
    # origin), so the runner computes it once per origin and shares it across all
    # horizons via the cross-arm/cross-horizon preprocessing cache (keyed on
    # origin_pos). The old code looped horizons and rebuilt a separate spec per
    # horizon, recomputing the (dominant) EM 6x per origin.
    horizons = (1, 3) if smoke else HORIZONS
    test_end = "1981-12" if smoke else "2017-12"

    n_estimators = 50 if smoke else 200

    base_arms = gcls_arms(
        target_growth,
        transformed_predictors,
        level_predictors,
        n_estimators=n_estimators,
        smoke=smoke,
        core_arms_only=core_arms_only,
    )
    # Attach the feature cadence policy to every arm (mirrors run_full.py).
    arms = [dataclasses.replace(a, feature_policy=feat_policy) for a in base_arms]

    # The runner injects each horizon into the window's test spec at execution
    # time, so the base window's ``horizon`` is overridden per horizon. We still
    # supply horizon=1 so the base window validates; test_end/retrain cadence are
    # the load-bearing fields.
    window = mf.window.from_cutoffs(
        estimation_start="1960-01",
        test_start="1980-01",
        test_end=test_end,
        mode="expanding",
        horizon=1,
        embargo=0,
        retrain_every=24,
        val_method="last_block",
        val_size=24,
    )

    for policy in policies:
        targets = [t for t in gcls_targets(policy) if t.name == target_growth]
        out_dir = out_root / target_alias / policy

        # Resume granularity stays per-cell: skip the whole spec only if EVERY
        # horizon parquet already exists (it is one EM pass anyway), else run it.
        stems = {h: f"{target_alias}_{policy}_h{h}" for h in horizons}
        if skip_existing and all((out_dir / f"{stems[h]}.parquet").exists() for h in horizons):
            print(
                f"[skip] {target_alias} {policy} all horizons {list(horizons)} (exist)",
                flush=True,
            )
            continue

        spec = pipeline_spec(
            data=bundle,
            targets=targets,
            horizons=list(horizons),
            window=window,
            arms=arms,
            evaluation=EvalSpec(benchmark="FM", tests=["dm", "cw", "mcs"]),
            preprocessing=pp,
            preprocessing_policy=pp_policy,
            save_models=False,
            checkpoint_dir=str(out_dir / "_checkpoints"),
        )
        t0 = time.time()
        try:
            rep = run_pipeline(spec)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[fail] {target_alias} {policy} (consolidated): "
                f"{type(exc).__name__}: {str(exc)[:160]}",
                flush=True,
            )
            if smoke:
                raise
            continue
        dt_spec = time.time() - t0

        # Split the consolidated report into per-(target, policy, horizon) cells.
        for h in horizons:
            stem = stems[h]
            forecasts_h = _slice_by_horizon(rep.forecasts, h)
            accuracy_h = _slice_by_horizon(rep.accuracy, h)
            if forecasts_h is None or accuracy_h is None:
                print(f"[warn] {target_alias} {policy} h{h}: no rows in report", flush=True)
                continue
            significance_h = _slice_by_horizon(rep.significance, h)
            info = _write_cell_outputs(
                out_dir, stem, forecasts_h, accuracy_h, significance_h, h
            )
            print(
                f"[done] {target_alias} {policy} h{h} "
                f"best={info['best']} relRMSE={info['relative_mse']:.3f}",
                flush=True,
            )
        print(
            f"[spec] {target_alias} {policy} horizons={list(horizons)} "
            f"arms={len(arms)} secs={dt_spec:.1f}",
            flush=True,
        )
        if smoke:
            _report_smoke(rep)


def _report_smoke(rep) -> None:
    """Print the accuracy table and leakage audit for the smoke run."""
    print("\n=== SMOKE accuracy table ===", flush=True)
    cols = [
        c
        for c in ["target", "horizon", "contender", "rmse", "relative_mse", "n_common",
                  "is_benchmark", "benchmark_present"]
        if c in rep.accuracy.columns
    ]
    print(rep.accuracy[cols].to_string(index=False), flush=True)
    print("\n=== SMOKE leakage_audit ===", flush=True)
    for key, value in rep.leakage_audit.items():
        print(f"{key}: {value}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, choices=list(TARGET_MAP.keys()))
    parser.add_argument("--policies", default="direct_average,path_average")
    parser.add_argument("--vintage", default="2018-01")
    parser.add_argument("--data-csv", default=None)
    parser.add_argument(
        "--out-root",
        default=str(HERE / "results"),
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="horizons={1,3} (tests consolidation), arms {FM, RF_F-Level}, test_end=1981-12.",
    )
    parser.add_argument(
        "--core-arms-only",
        action="store_true",
        help=(
            "Restrict the arm grid to the three core arms (FM, RF_F-Level, RF_MARX) "
            "needed for the appendix growth-target comparison."
        ),
    )
    args = parser.parse_args()

    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    for p in policies:
        if p not in {"direct_average", "path_average"}:
            raise SystemExit(f"unknown policy {p!r}")

    run_target(
        args.target,
        policies=policies,
        vintage=args.vintage,
        data_csv=args.data_csv,
        out_root=Path(args.out_root),
        skip_existing=args.skip_existing,
        smoke=args.smoke,
        core_arms_only=args.core_arms_only,
    )


if __name__ == "__main__":
    main()
