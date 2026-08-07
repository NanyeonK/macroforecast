"""Re-derive the corrected G1 gate result from the ON-DISK artifacts of the completed
run (result_store forecasts + checkpoint selection_history). No pipeline re-run."""
from __future__ import annotations
import sys, json, math
from pathlib import Path
REPO = Path(__file__).resolve().parents[3] if (Path(__file__).resolve().parents[3] / "macroforecast").exists() else Path("/home/nanyeon99/project/mf-b4-gcls2022")
sys.path.insert(0, str(REPO))
import numpy as np, pandas as pd
from macroforecast import pipeline as P
from macroforecast.pipeline.result_store import ResultStore

OUT = REPO / "runs" / "gcls_b4_stage1"
STORE = OUT / "_result_store_g1"
CKPT = OUT / "_ckpt_g1"
RE = 24

# ---- forecasts from the result store (read cell parquets directly) ----
frames = [pd.read_parquet(p) for p in sorted((STORE / "cells").glob("*.parquet"))]
fr = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
n_origins = int(fr["origin"].nunique())
rows = fr.groupby("arm").size().to_dict()

# ---- relative RMSPE (ARDI vs AR,BIC) recomputed from stored forecasts ----
def _mse(arm):
    s = fr[fr["arm"] == arm].dropna(subset=["prediction", "actual"])
    e = pd.to_numeric(s["prediction"], errors="coerce") - pd.to_numeric(s["actual"], errors="coerce")
    e = e.dropna()
    return float(np.mean(e.values ** 2)), len(e)
mse_ar, n_ar = _mse("AR,BIC")
mse_ardi, n_ardi = _mse("ARDI,BIC")
rel_rmspe = math.sqrt(mse_ardi / mse_ar) if mse_ar > 0 else None

# ---- refit / retune trap from selection_history (checkpoint) ----
h = P.selection_history(str(CKPT))
nlag = h[(h["arm"].astype(str).str.endswith("AR_BIC")) & (h["name"] == "n_lag")].copy()
nlag = nlag.sort_values("origin_pos")
refit_count = int(nlag["origin"].nunique())
uniq = sorted(nlag["origin_pos"].unique().tolist())
rank = {op: i for i, op in enumerate(uniq)}
nlag["block"] = nlag["origin_pos"].map(lambda op: rank[op] // RE)
wb = nlag.groupby("block")["value"].nunique()
piecewise = bool((wb <= 1).all())
retune_blocks = int(wb.shape[0])
byblock = {int(k): int(v) for k, v in nlag.groupby("block")["value"].first().to_dict().items()}

res = {
    "gate": "G1",
    "completed": True,
    "window": "expanding, est 1960-01, test 1980-01..2017-12, retrain_every=1, retune_every=24",
    "n_usable_origins_h1": n_origins,
    "note_origin_count": "455 usable h=1 origins (design's nominal 456-month test window minus the final month, whose h=1 target 2018-01 is outside the panel)",
    "benchmark_forecast_rows": int(rows.get("AR,BIC", 0)),
    "ardi_forecast_rows": int(rows.get("ARDI,BIC", 0)),
    "refit_count": refit_count,
    "retune_count_blocks": retune_blocks,
    "retune_piecewise_constant_over_24blocks": piecewise,
    "n_lag_selected_by_block": byblock,
    "relative_rmspe_ardi_vs_arbic": rel_rmspe,
    "relative_rmspe_finite": (rel_rmspe is not None and math.isfinite(rel_rmspe)),
    "mse_ar_bic": mse_ar, "mse_ardi": mse_ardi,
    "result_store_exists": STORE.exists(),
    "checkpoint_exists": CKPT.exists(),
    "checks": {},
}
res["checks"] = {
    "completed_with_store_and_ckpt": STORE.exists() and CKPT.exists(),
    "benchmark_refits_all_usable_origins": refit_count == n_origins == res["benchmark_forecast_rows"],
    "ardi_produced_rows": res["ardi_forecast_rows"] > 0,
    "relative_rmspe_finite": res["relative_rmspe_finite"],
    "retune_only_at_block_boundaries": piecewise,
    "retune_count_approx_19": 18 <= retune_blocks <= 20,
    "refit_far_exceeds_retune (trap)": refit_count >= 20 * retune_blocks // 20 and refit_count > 5 * retune_blocks,
}
res["G1_PASS"] = all(res["checks"].values())
(OUT / "g1_gate_result.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
