"""GCLS (2022, JAE) replication runner + G1 smoke gate.

Mirrors the ZWW / GCLS-2021 runner CLI. STAGE-1 exposes the G1 smoke gate; --full is
wired but intentionally NOT launched here (520k fits).

Window (design S3): expanding, estimation_start=1960-01, test 1980-01..2017-12 (456
origins), retrain_every=1 (monthly refit), retune_every=24, retune_on_retrain=False.
EvalSpec(benchmark="AR,BIC", tests=("dm","mcs")).

G1 smoke gate: 2 arms (AR,BIC benchmark + one ARDI) x INDPRO x h=1 over the full 456
origins. Asserts: completes with checkpoint + result_store; produces a FINITE
relative-RMSPE; and the refit/retune trap -- the benchmark REFITS all 456 origins
(retrain_every=1) while RETUNING only ~19 times (retune_every=24). The retune proof is
that the IC-selected n_lag is PIECEWISE-CONSTANT over 24-origin blocks (the search runs
only at block boundaries), while an applied-param / forecast row exists at EVERY origin.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

# Cap BLAS/OpenMP threads: the per-origin EM SVD otherwise oversubscribes all cores
# (BLAS thread-thrashing), which dominates wall-clock. n_jobs handles cell parallelism.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import macroforecast as mf
from macroforecast.pipeline import (
    EvalSpec,
    pipeline_spec,
    run_pipeline,
    selection_history,
)

from scripts.replication.gcls_2022_pipeline.data import (
    augmented_bundle,
    gcls_targets,
    target_aliases,
    yobj_column,
)
from scripts.replication.gcls_2022_pipeline.registry import build_gcls2022_arms

RETUNE_EVERY = 24
OUT_DIR = REPO / "runs" / "gcls_b4_stage1"


# --------------------------------------------------------------------------- #
def gcls_window():
    return mf.window.from_cutoffs(
        estimation_start="1960-01", test_start="1980-01", test_end="2017-12",
        mode="expanding", retrain_every=1, retune_every=RETUNE_EVERY,
        retune_on_retrain=False,
    )


def gcls_preprocessing(pp_update: int = 24):
    pp = mf.preprocessing.preprocess_spec(
        transform="official", impute="em_factor", outliers="iqr",
        outlier_action="flag_as_nan", standardize="none",
    )
    # EM/factor basis refit cadence (leak-free: fit on origin_available data, reused
    # between refits). update=24 mirrors the GCLS-2021 pipeline's documented performance
    # choice (~24x fewer EM SVDs than per-origin) and does NOT affect the model-level
    # refit/retune trap, which is governed by the WINDOW (retrain_every=1/retune_every=24).
    pp_policy = mf.window.stage_policy("origin_available", update=pp_update)
    return pp, pp_policy


# --------------------------------------------------------------------------- #
def build_spec(args, *, gate: str):
    bundle, predictors = augmented_bundle()
    all_arms = build_gcls2022_arms(yobj_column("INDPRO"), predictors)
    by_name = {a.name: a for a in all_arms}

    if gate == "g1":
        arm_names = ["AR,BIC", "ARDI,BIC"]
        target_cols = ["INDPRO"]
        horizons = [1]
    else:  # full / custom
        # Arm names contain commas ("ARDI,BIC"), so a comma-separated --arms
        # cannot express them. Semicolons take precedence when present.
        if args.arms:
            sep = ";" if ";" in args.arms else ","
            arm_names = [a.strip() for a in args.arms.split(sep) if a.strip()]
        else:
            arm_names = [a.name for a in all_arms]
        target_cols = args.targets.split(",") if args.targets else \
            [c for c in target_aliases().values()]  # alias names
        horizons = [int(h) for h in args.horizons.split(",")] if args.horizons \
            else [1, 3, 9, 12, 24]

    # resolve arms
    arms = []
    for name in arm_names:
        if name not in by_name:
            raise SystemExit(f"unknown arm {name!r}; available: {list(by_name)[:5]}...")
        arms.append(by_name[name])

    # resolve targets (accept column, alias, or YOBJ name)
    alias_to_col = {a: c for c, a in target_aliases().items()}  # YOBJ alias->YOBJ col
    all_targets = {t.name: t for t in gcls_targets()}
    col_by_plain = {c.replace("YOBJ__", ""): c for c in all_targets}
    tgts = []
    for tc in target_cols:
        yc = None
        if tc in all_targets:                       # full YOBJ__<col> name
            yc = tc
        elif tc in col_by_plain:                    # plain column: INDPRO/CPIAUCSL/HOUST/UNRATE/T10YFFM
            yc = col_by_plain[tc]
        elif tc in alias_to_col:                    # paper alias: INF/SPREAD/INDPRO/HOUST/UNRATE
            yc = alias_to_col[tc]
        elif yobj_column(tc) in all_targets:
            yc = yobj_column(tc)
        if yc is None:
            raise SystemExit(
                f"unknown target {tc!r}; use a column ({list(col_by_plain)}) or alias "
                f"({list(alias_to_col)})"
            )
        tgts.append(all_targets[yc])

    window = gcls_window()
    pp, pp_policy = gcls_preprocessing()
    rs = getattr(args, "result_store", None)
    if rs:
        store = str(rs)
        ckpt = str(rs) + "_ckpt"
    else:
        store = str(OUT_DIR / f"_result_store_{gate}")
        ckpt = str(OUT_DIR / f"_ckpt_{gate}")

    spec = pipeline_spec(
        data=bundle, targets=tgts, horizons=horizons, window=window, arms=arms,
        evaluation=EvalSpec(benchmark="AR,BIC", tests=("dm", "mcs")),
        preprocessing=pp, preprocessing_policy=pp_policy,
        result_store=store, checkpoint_dir=ckpt,
        # G1 records it by default; MF_GCLS_SELHIST=1 turns it on for any gate,
        # which is how the per-origin hyperparameter choices are audited without
        # instrumenting the model registry.
        selection_history=(gate == "g1" or os.environ.get("MF_GCLS_SELHIST") == "1"),
        n_jobs=args.n_jobs, seed=42,
    )
    return spec, store, ckpt


# --------------------------------------------------------------------------- #
def _relative_rmspe(accuracy, contender: str, benchmark: str = "AR,BIC"):
    """Return the ARDI-vs-benchmark relative RMSPE (= sqrt of relative_mse) if present."""
    import pandas as pd

    if accuracy is None or len(accuracy) == 0:
        return None
    df = accuracy.copy()
    arm_col = "contender" if "contender" in df.columns else ("arm" if "arm" in df.columns else None)
    if arm_col is None:
        return None
    row = df[df[arm_col] == contender]
    for col in ("relative_mse", "relative_rmse", "rel_mse"):
        if col in df.columns and len(row):
            val = float(pd.to_numeric(row[col], errors="coerce").iloc[0])
            return math.sqrt(val) if col == "relative_mse" and val >= 0 else val
    return None


def g1_gate(spec, store, ckpt) -> dict:
    import pandas as pd

    report = run_pipeline(spec)
    result: dict = {"gate": "G1", "completed": True}

    frame = report.forecasts
    bench = "AR,BIC"
    bench_rows = frame[frame["arm"] == bench]
    ardi_rows = frame[frame["arm"] == "ARDI,BIC"]
    refit_count = int(bench_rows["origin"].nunique())
    n_origins = int(frame["origin"].nunique())

    # relative RMSPE (ARDI vs AR,BIC)
    rel = _relative_rmspe(report.accuracy, "ARDI,BIC", bench)
    finite = rel is not None and math.isfinite(rel)

    # refit vs retune from selection_history. IMPORTANT: read from the checkpoint PATH
    # (selection_history(report) is empty -- the history is persisted on disk); the
    # benchmark arm is keyed "<target>__AR_BIC".
    hist = selection_history(str(ckpt))
    ar_nlag = hist[(hist["arm"].astype(str).str.endswith("AR_BIC")) & (hist["name"] == "n_lag")]
    ar_nlag = ar_nlag.sort_values("origin_pos")
    origins = sorted(ar_nlag["origin_pos"].unique().tolist())
    applied_count = int(ar_nlag["origin"].nunique())
    ranks = {op: i for i, op in enumerate(origins)}
    ar_nlag = ar_nlag.assign(block=ar_nlag["origin_pos"].map(lambda op: ranks[op] // RETUNE_EVERY))
    within_block_unique = ar_nlag.groupby("block")["value"].nunique()
    piecewise_constant = bool((within_block_unique <= 1).all()) and len(within_block_unique) > 0
    retune_count = int(within_block_unique.shape[0])  # number of 24-origin blocks
    n_lag_by_block = ar_nlag.groupby("block")["value"].first().to_dict()

    store_exists = Path(store).exists()
    ckpt_exists = Path(ckpt).exists()

    result.update({
        "n_test_origins": n_origins,
        "note_origin_count": "usable h=1 origins (nominal 456-month window minus the "
                             "final month whose h=1 target is outside the panel)",
        "benchmark_forecast_rows": int(len(bench_rows)),
        "ardi_forecast_rows": int(len(ardi_rows)),
        "refit_count": refit_count,
        "applied_param_origins": applied_count,
        "retune_count_blocks": retune_count,
        "retune_piecewise_constant_over_24blocks": piecewise_constant,
        "n_lag_selected_by_block": {int(k): (int(v) if v is not None else None)
                                    for k, v in n_lag_by_block.items()},
        "relative_rmspe_ardi_vs_arbic": rel,
        "relative_rmspe_finite": finite,
        "result_store_exists": store_exists,
        "checkpoint_exists": ckpt_exists,
    })

    # ---- assertions (report, don't crash, so the JSON always prints) ----
    checks = {
        "completed_with_store_and_ckpt": store_exists and ckpt_exists,
        # benchmark refits every USABLE origin (self-consistent; 455 for h=1 over the
        # 456-month window). retrain_every=1 trap: refit rows == n_origins, not ~19.
        "benchmark_refits_all_usable_origins": refit_count == n_origins == int(len(bench_rows)),
        "n_origins_expected_455_or_456": n_origins in (455, 456),
        "ardi_produced_rows": len(ardi_rows) > 0,
        "relative_rmspe_finite": finite,
        "retune_only_at_block_boundaries": piecewise_constant,
        "retune_count_approx_19": 18 <= retune_count <= 20,
        "refit_far_exceeds_retune (trap)": refit_count > 5 * retune_count,
    }
    result["checks"] = checks
    result["G1_PASS"] = all(checks.values())
    return result


# --------------------------------------------------------------------------- #
def g2_report(spec, store, ckpt, out_prefix="g2") -> dict:
    """Run the full arm grid for the requested targets/horizons; save the NATIVE
    per-(horizon) accuracy / DM-significance / MCS tables (the Table A1 material, now
    unblocked by the DM None-guard) plus a pooled AR,BIC-relative-RMSPE summary."""
    import time
    import pandas as pd

    t0 = time.time()
    report = run_pipeline(spec)
    wall_s = time.time() - t0

    # --- persist the NATIVE evaluation tables (per target, horizon, contender) ---
    native = {}
    for label, tbl in (("accuracy", report.accuracy),
                       ("significance", report.significance),
                       ("mcs", report.mcs)):
        path = OUT_DIR / f"{out_prefix}_{label}.csv"
        if tbl is not None and len(tbl):
            try:
                tbl.to_csv(path, index=False)
                native[label] = {"rows": int(len(tbl)), "path": str(path)}
            except Exception as exc:  # noqa: BLE001
                native[label] = {"rows": int(len(tbl)), "error": f"{type(exc).__name__}: {exc}"}
        else:
            native[label] = {"rows": 0, "path": None}

    frame = report.forecasts
    arms = list(spec.arms)
    arm_names = [a.name for a in arms]
    tags = {a.name: {k: v for k, v in dict(a.tags).items()} for a in arms}
    rows_per_arm = frame.groupby("arm").size().to_dict() if "arm" in frame.columns else {}
    produced = [n for n in arm_names if rows_per_arm.get(n, 0) > 0]
    missing = [n for n in arm_names if rows_per_arm.get(n, 0) == 0]

    # rel-RMSPE recomputed from the forecast frame (robust: the native evaluation's
    # significance/DM step can crash on identical forecast pairs and empty report.accuracy).
    def _mse(arm):
        s = frame[frame["arm"] == arm].dropna(subset=["prediction", "actual"])
        e = (pd.to_numeric(s["prediction"], errors="coerce")
             - pd.to_numeric(s["actual"], errors="coerce")).dropna()
        return (float((e.values ** 2).mean()) if len(e) else None), len(e)

    bench = "AR,BIC"
    mse_bench, _ = _mse(bench)
    bench_present_all = mse_bench is not None and mse_bench > 0
    table = []
    for n in arm_names:
        m, cnt = _mse(n)
        rel = (math.sqrt(m / mse_bench) if (m is not None and mse_bench) else None)
        table.append({"arm": n, "rel_rmspe": rel,
                      "rmse": (math.sqrt(m) if m is not None else None),
                      "n_common": cnt, "rows": int(rows_per_arm.get(n, 0)),
                      "tags": tags[n]})

    table.sort(key=lambda r: (r["rel_rmspe"] is None, r["rel_rmspe"] or 9e9))
    nonfinite = [r["arm"] for r in table
                 if r["rel_rmspe"] is None or not math.isfinite(r["rel_rmspe"])]
    store_bytes = (sum(f.stat().st_size for f in Path(store).rglob("*") if f.is_file())
                   if store and Path(store).exists() else 0)

    horizons = sorted(int(h) for h in frame["horizon"].unique()) if "horizon" in frame.columns else []
    n_cells_expected = len(arm_names) * len(horizons) if horizons else len(arm_names)
    checks = {
        "all_46_produced": len(produced) == len(arm_names) == 46,
        "zero_failed_cells": len(report.failed_cells) == 0,
        "zero_empty_cells": len(report.empty_cells) == 0,
        "benchmark_present_all": bench_present_all,
        "native_accuracy_populated": native["accuracy"]["rows"] > 0,
        "native_dm_populated": native["significance"]["rows"] > 0,
        "native_mcs_populated": native["mcs"]["rows"] > 0,
    }
    return {
        "run": f"G2 ({len(arm_names)} arms x targets={[t.name for t in spec.targets] if hasattr(spec,'targets') else '?'} x horizons={horizons})",
        "n_arms": len(arm_names),
        "horizons": horizons,
        "n_cells_expected": n_cells_expected,
        "arms_produced": len(produced),
        "arms_missing": missing,
        "failed_cells": len(report.failed_cells),
        "empty_cells": len(report.empty_cells),
        "evaluation_error": getattr(report, "evaluation_error", None),
        "benchmark_present_all": bench_present_all,
        "native_tables": native,
        "nonfinite_or_missing_pooled_rel_rmspe": nonfinite,
        "wall_seconds": round(wall_s, 1),
        "wall_minutes": round(wall_s / 60, 2),
        "wall_hours": round(wall_s / 3600, 2),
        "result_store_bytes": store_bytes,
        "result_store_mb": round(store_bytes / 1e6, 3),
        "pooled_rel_rmspe_table_note": "rel_rmspe here is POOLED across horizons; the "
                                       "per-horizon Table A1 numbers are in the native "
                                       "accuracy CSV (relative_mse -> sqrt = rel-RMSPE)",
        "pooled_rel_rmspe_table": table,
        "checks": checks,
        "G2_PASS": all(checks.values()),
    }


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="GCLS-2022 replication runner")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="run the G1 smoke gate")
    mode.add_argument("--full", action="store_true", help="run the full arm grid")
    ap.add_argument("--arms", default=None, help="comma-separated arm names")
    ap.add_argument("--targets", default=None, help="comma-separated targets (col/alias)")
    ap.add_argument("--horizons", default=None, help="comma-separated horizons")
    ap.add_argument("--n-jobs", default="1", dest="n_jobs", help="int or 'auto'")
    ap.add_argument("--result-store", default=None, dest="result_store")
    ap.add_argument("--out-prefix", default=None, dest="out_prefix", help="output file prefix")
    args = ap.parse_args()
    args.n_jobs = args.n_jobs if args.n_jobs == "auto" else int(args.n_jobs)

    if args.smoke:
        spec, store, ckpt = build_spec(args, gate="g1")
        res = g1_gate(spec, store, ckpt)
        (OUT_DIR / "g1_gate_result.json").write_text(json.dumps(res, indent=2))
        print(json.dumps(res, indent=2))
        return

    # --full (or general): all-arms (unless --arms) x targets x horizons. Saves the
    # NATIVE per-horizon accuracy / DM / MCS tables + a summary. result_store is
    # incremental (resumable across restarts); n_jobs parallelises cells.
    prefix = args.out_prefix or (
        "g2_" + "_".join((args.targets or "all").lower().split(",")) if args.targets else "g2_all"
    )
    spec, store, ckpt = build_spec(args, gate="full")
    print(f"[g2] arms={len(spec.arms)} targets={[t.name for t in spec.targets]} "
          f"horizons={args.horizons} store={store} n_jobs={args.n_jobs}", flush=True)
    res = g2_report(spec, store, ckpt, out_prefix=prefix)
    (OUT_DIR / f"{prefix}_result.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
