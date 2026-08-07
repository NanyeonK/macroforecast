"""Recompute the G2-smoke AR,BIC-relative RMSPE table directly from the on-disk
result-store forecasts (46 cells). No pipeline re-run."""
from __future__ import annotations
import sys, json, math
from pathlib import Path
REPO = Path("/home/nanyeon99/project/mf-b4-gcls2022")
sys.path.insert(0, str(REPO))
import numpy as np, pandas as pd
from scripts.replication.gcls_2022_pipeline.registry import build_gcls2022_arms
from scripts.replication.gcls_2022_pipeline.data import yobj_column

OUT = REPO / "runs" / "gcls_b4_stage1"
STORE = OUT / "_result_store_full"

frames = [pd.read_parquet(p) for p in sorted((STORE / "cells").glob("*.parquet"))]
fr = pd.concat(frames, ignore_index=True)
tags = {a.name: dict(a.tags) for a in build_gcls2022_arms(yobj_column("INDPRO"))}

BENCH = "AR,BIC"

def mse(arm):
    s = fr[fr["arm"] == arm].dropna(subset=["prediction", "actual"])
    e = pd.to_numeric(s["prediction"], errors="coerce") - pd.to_numeric(s["actual"], errors="coerce")
    e = e.dropna()
    return (float(np.mean(e.values ** 2)) if len(e) else None), len(e)

mse_bench, n_bench = mse(BENCH)
rows = []
for arm in tags:
    m, n = mse(arm)
    rel = (math.sqrt(m / mse_bench) if (m is not None and mse_bench and mse_bench > 0) else None)
    rows.append({
        "arm": arm, "rel_rmspe": rel, "rmse": (math.sqrt(m) if m is not None else None),
        "n": n, "tags": tags[arm],
    })
rows.sort(key=lambda r: (r["rel_rmspe"] is None, r["rel_rmspe"] if r["rel_rmspe"] is not None else 9e9))

nonfinite = [r["arm"] for r in rows if r["rel_rmspe"] is None or not math.isfinite(r["rel_rmspe"])]
store_bytes = sum(f.stat().st_size for f in STORE.rglob("*") if f.is_file())

# merge with the run's wall time
prev = json.loads((OUT / "g2_smoke_result.json").read_text())
out = {
    "run": "G2_SMOKE (46 arms x INDPRO x h=1, 455 origins)",
    "n_arms": len(rows), "n_origins": int(fr["origin"].nunique()),
    "arms_produced": int(fr["arm"].nunique()),
    "benchmark": BENCH, "benchmark_mse": mse_bench, "benchmark_n": n_bench,
    "failed_cells": prev["failed_cells"], "empty_cells": prev["empty_cells"],
    "benchmark_present_all": prev["benchmark_present_all"],
    "wall_minutes": prev["wall_minutes"],
    "result_store_mb": round(store_bytes / 1e6, 3),
    "nonfinite_or_missing_rel_rmspe": nonfinite,
    "rel_rmspe_table": rows,
    "checks": {
        "all_46_produced": int(fr["arm"].nunique()) == 46,
        "zero_failed_cells": prev["failed_cells"] == 0,
        "zero_empty_cells": prev["empty_cells"] == 0,
        "benchmark_present_all": bool(prev["benchmark_present_all"]),
        "all_rel_rmspe_finite": len(nonfinite) == 0,
    },
}
out["G2_SMOKE_PASS"] = all(out["checks"].values())
(OUT / "g2_smoke_result.json").write_text(json.dumps(out, indent=2))

# pretty print the table
print(f"PASS={out['G2_SMOKE_PASS']} arms={out['arms_produced']}/46 failed={out['failed_cells']} "
      f"empty={out['empty_cells']} bench_present_all={out['benchmark_present_all']} "
      f"wall={out['wall_minutes']}min store={out['result_store_mb']}MB nonfinite={len(nonfinite)}")
print(f"{'arm':20s} {'relRMSPE':>9s} {'rmse':>10s}  X NL SH        CV     LF")
for r in rows:
    t = r["tags"]
    rr = f"{r['rel_rmspe']:.4f}" if r["rel_rmspe"] is not None else "  NA"
    print(f"{r['arm']:20s} {rr:>9s} {r['rmse']:.6e}  {t['X']}  {t['NL']} {str(t['SH']):9s} {str(t['CV']):6s} {t['LF']}")
