"""Zhang, Wahab & Wang (2023, IJF) crude-oil volatility -- Stage-2 replication runner.

Builds the 13-arm ZWW pipeline (registry.zww_arms) over an expanding window and evaluates
R^2_OS (Campbell-Thompson, vs the AR benchmark) + Clark-West, per horizon. Mirrors the GCLS
2021 runner's CLI. One pipeline_spec PER HORIZON because ZWW's AR lag order L(h) differs by
horizon (a single multi-horizon spec cannot carry a per-horizon arm set).

Panels (built by build_panels): runs/zww_b3_stage2/zww_panel_{futures,spot}.csv
  columns: RV, LV(=ln RV), + 126 FRED-MD 2019:06 predictors (tcode+lag already applied).

Usage
-----
  # SMOKE (fast: 3 arms, h=1, short OOS 2015-01..2018-12):
  python -m scripts.replication.zww_2023_pipeline.replicate_zww2023 --market futures --smoke

  # FULL (all 13 arms x 4 horizons x full OOS 1998-01..2018-12), one market:
  python -m scripts.replication.zww_2023_pipeline.replicate_zww2023 --market futures --full
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
warnings.simplefilter("ignore")

import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import macroforecast as mf
from macroforecast.pipeline import EvalSpec, pipeline_spec, run_pipeline

from scripts.replication.zww_2023_pipeline.registry import (
    AR_LAG_PRESETS,
    ARM_ORDER,
    TARGET_COL,
    TARGET_TRANSFORM,
    LV_COL,
    LV_CTRL_COLS,
    zww_arms,
    zww_target,
)

HORIZONS = (1, 3, 6, 12)
STAGE2 = REPO_ROOT / "runs" / "zww_b3_stage2"
ESTIMATION_START = "1985-01-31"
OOS_START_FULL = "1998-01-31"
OOS_START_SMOKE = "2015-01-31"
SAMPLE_END = "2018-12-31"


def _panel_path(market: str) -> Path:
    return STAGE2 / f"zww_panel_{market}.csv"


def _load_bundle(market: str):
    path = _panel_path(market)
    if not path.exists():
        raise SystemExit(f"panel not found: {path} (run build_panels first)")
    header = pd.read_csv(path, nrows=0)
    cols = [c for c in header.columns if c != "date"]
    codes = {c: 1 for c in cols}  # identity t-code: tcode already applied in stage-1
    bundle = mf.data.load_custom_csv(
        str(path), date_col="date", frequency="monthly",
        transform_codes=codes, dataset=f"zww_{market}",
    )
    predictors = [c for c in cols if c not in (TARGET_COL, LV_COL) and c not in set(LV_CTRL_COLS)]
    return bundle, predictors


def _window(horizon: int, oos_start: str):
    return mf.window.from_cutoffs(
        estimation_start=ESTIMATION_START,
        test_start=oos_start,
        test_end=SAMPLE_END,
        mode="expanding",
        horizon=int(horizon),
        embargo=0,
        retrain_every=1,
        val_method="last_block",
        val_size=24,
    )


def run_market(
    market: str,
    *,
    horizons,
    arm_filter,
    ar_lags: str,
    smoke: bool,
    out_root: Path,
) -> None:
    print(f"macroforecast {mf.__version__}  market={market}  ar_lags={ar_lags}", flush=True)
    bundle, predictors = _load_bundle(market)
    print(f"panel: {len(predictors)} predictors + {TARGET_COL}/{LV_COL}", flush=True)
    oos_start = OOS_START_SMOKE if smoke else OOS_START_FULL
    target = zww_target()
    out_dir = out_root / market
    out_dir.mkdir(parents=True, exist_ok=True)

    for h in horizons:
        arms = zww_arms(h, predictors, ar_lags=ar_lags, smoke=smoke)
        if arm_filter is not None:
            keep = set(arm_filter) | {"AR"}  # benchmark always present
            avail = {a.name for a in arms}
            missing = [n for n in arm_filter if n not in avail]
            if missing:
                raise SystemExit(f"--arms unknown {missing}; available: {sorted(avail)}")
            arms = [a for a in arms if a.name in keep]
        spec = pipeline_spec(
            data=bundle,
            targets=[target],
            horizons=[int(h)],
            window=_window(h, oos_start),
            arms=arms,
            evaluation=EvalSpec(benchmark="AR", metrics=("r2_oos",), tests=("cw",)),
            preprocessing=None,       # passthrough: tcode already applied in stage-1
            seed=42,
            n_jobs="auto",                      # parallel cells (arm x origin) — statistically identical, faster
        )
        t0 = time.time()
        rep = run_pipeline(spec)
        dt = time.time() - t0
        _write_and_report(rep, market, h, arms, out_dir, dt, smoke, bundle)


def _write_and_report(rep, market, h, arms, out_dir, dt, smoke, bundle):
    acc = rep.accuracy
    sig = rep.significance
    acc.to_csv(out_dir / f"accuracy_h{h}.csv", index=False)
    if sig is not None and not getattr(sig, "empty", True):
        sig.to_csv(out_dir / f"significance_h{h}.csv", index=False)

    # merge R2_OS + CW into one view
    name_col = "contender" if "contender" in acc.columns else "arm"
    keep_cols = [c for c in (name_col, "r2_oos", "relative_mse", "rmse", "n_common",
                             "is_benchmark", "benchmark_present") if c in acc.columns]
    view = acc[keep_cols].copy()
    if sig is not None and not getattr(sig, "empty", True):
        scol = "contender" if "contender" in sig.columns else name_col
        cw_rows = sig[sig.get("test", "cw") == "cw"] if "test" in sig.columns else sig
        stat_col = next((c for c in ("statistic", "stat", "cw_stat") if c in cw_rows.columns), None)
        p_col = next((c for c in ("p_value", "pvalue", "pval") if c in cw_rows.columns), None)
        cols = [scol] + [c for c in (stat_col, p_col) if c]
        if stat_col:
            merged = view.merge(
                cw_rows[cols].rename(columns={scol: name_col, stat_col: "cw_stat", p_col: "cw_p"}),
                on=name_col, how="left",
            )
            view = merged
    order = {n: i for i, n in enumerate(ARM_ORDER)}
    view["_o"] = view[name_col].map(lambda n: order.get(n, 99))
    view = view.sort_values("_o").drop(columns="_o")
    print(f"\n=== {market} h={h}  ({dt:.1f}s, {len(arms)} arms) ===", flush=True)
    print(view.to_string(index=False), flush=True)

    if smoke:
        _smoke_sanity(rep, market, h, bundle)


def _smoke_sanity(rep, market, h, bundle):
    fc = rep.forecasts
    finite = True
    if "r2_oos" in rep.accuracy.columns:
        v = rep.accuracy["r2_oos"].to_numpy(dtype=float)
        finite = bool(np.isfinite(v[~np.isnan(v)]).all())
    print(f"[smoke] r2_oos finite: {finite}", flush=True)
    # Verify the pipeline's realized target equals the market's own log_average_value target.
    # The forecast rows carry an ORIGIN column (t); the realized actual at origin t is
    # ln(mean RV_{t+1..t+h}) = direct_target(log_average_value)[t]. Align on origin.
    if fc is None or getattr(fc, "empty", True) or "origin" not in fc.columns:
        return
    from macroforecast.feature_engineering.targets import direct_target
    panel = pd.read_csv(_panel_path(market), parse_dates=["date"]).set_index("date")
    tgt = direct_target(panel[[TARGET_COL]], target=TARGET_COL, horizons=[int(h)],
                        transform=TARGET_TRANSFORM)[f"{TARGET_COL}_{TARGET_TRANSFORM}_h{h}"]
    sub = fc.dropna(subset=["actual"]).copy()
    sub["origin"] = pd.to_datetime(sub["origin"])
    sub = sub.drop_duplicates(subset=["origin"]).set_index("origin")
    al = tgt.reindex(sub.index)
    diff = float((sub["actual"].astype(float) - al).abs().max())
    print(f"[smoke] realized-target vs {market} log_average_value h{h} max_abs_diff: {diff:.3e}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--market", required=True, choices=["futures", "spot"])
    p.add_argument("--smoke", action="store_true",
                   help="3 arms {AR,PCA-t_stat,Lasso}, h=1, short OOS 2015-01..2018-12")
    p.add_argument("--full", action="store_true", help="all 13 arms x 4 horizons x full OOS")
    p.add_argument("--arms", default=None, help="comma-separated arm names (AR always kept)")
    p.add_argument("--horizons", default=None, help="comma-separated horizons (subset of 1,3,6,12)")
    p.add_argument("--ar-lags", default="aic", choices=sorted(AR_LAG_PRESETS.keys()))
    p.add_argument("--out-root", default=str(STAGE2 / "results"))
    args = p.parse_args()

    if not args.smoke and not args.full:
        cmd = ("python -m scripts.replication.zww_2023_pipeline.replicate_zww2023 "
               f"--market {args.market} --full")
        print("[guard] neither --smoke nor --full given; refusing to run.", flush=True)
        print(f"[guard] to launch the FULL run: {cmd}", flush=True)
        return

    horizons = HORIZONS
    if args.smoke:
        horizons = (1,)
    if args.horizons is not None:
        req = {int(x) for x in args.horizons.split(",") if x.strip()}
        horizons = tuple(h for h in HORIZONS if h in req)
        if not horizons:
            raise SystemExit(f"--horizons selected none of {list(HORIZONS)}")
    arm_filter = ([a.strip() for a in args.arms.split(",") if a.strip()]
                  if args.arms is not None else None)

    run_market(
        args.market, horizons=horizons, arm_filter=arm_filter,
        ar_lags=args.ar_lags, smoke=args.smoke, out_root=Path(args.out_root),
    )


if __name__ == "__main__":
    main()
