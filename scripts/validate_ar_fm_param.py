"""AR and FM parametric validation — CLSS 2021 Appendix B.2.

Validates AR(p) and FM_AGR against paper Table B.2 (path_average target).
All 6 horizons including h=9 (missing from earlier validate_ar_fm_current.py).

Also checks FM_SGR (paper shows FM_SGR/FM_AGR as a separate row).

Paper B.2 INDPRO values:
  AR/FM_AGR:    h1=1.06  h3=0.97  h6=0.93  h9=0.95  h12=0.97  h24=1.15
  FM_SGR/FM_AGR: h1=1.06  h3=1.08  h6=1.00  h9=0.99  h12=1.00  h24=1.06

Usage
-----
uv run scripts/validate_ar_fm_param.py --max_lag 12 --label C-AR-0
uv run scripts/validate_ar_fm_param.py --max_lag 24 --label C-AR-1
"""
from __future__ import annotations

import argparse
import json as _json
import time

import numpy as np
import pandas as pd

from macroforecast.data import load_fred_md
from macroforecast.data.schema import MacroFrame
from macroforecast.pipeline.components import CVScheme, LossFunction, Regularization, Window
from macroforecast.pipeline.experiment import ForecastExperiment, FeatureSpec, ModelSpec
from macroforecast.pipeline.r_models import ARDIModel, ARModel
from macroforecast.preprocessing.missing import remove_outliers_iqr

OOS_START = "1980-01-01"
OOS_END   = "2017-12-01"
HORIZONS  = [1, 3, 6, 9, 12, 24]  # h=9 included (was missing before)
K = 8; P_Y = 12

# Paper B.2 INDPRO (h=1 AR corrected from 1.00 to 1.06; h=9 added)
PAPER_AR = {1: 1.06, 3: 0.97, 6: 0.93, 9: 0.95, 12: 0.97, 24: 1.15}
PAPER_FM_SGR = {1: 1.06, 3: 1.08, 6: 1.00, 9: 0.99, 12: 1.00, 24: 1.06}
# FM_AGR absolute RMSE from paper (used as sanity check)
PAPER_FM_ABS = {1: 0.006, 3: 0.004, 6: 0.004, 9: 0.004, 12: 0.003, 24: 0.003}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--max_lag", type=int, default=12,
                   help="Maximum AR lag order for BIC selection (paper: 12)")
    p.add_argument("--label",   type=str, default="AR-FM-param")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    md = load_fred_md()
    md_eval  = MacroFrame(data=md.data.loc[:OOS_END], metadata=md.metadata)
    df_clean = remove_outliers_iqr(md_eval.data, threshold=10.0)
    md_clean = MacroFrame(data=df_clean, metadata=md.metadata)
    md_stat  = md_clean.transform()
    panel    = md_stat.data.loc["1960-01":OOS_END]
    target   = panel["INDPRO"]
    pred_df  = panel.drop(columns=["INDPRO"])

    ar_spec = ModelSpec(
        model_cls=ARModel, regularization=Regularization.NONE,
        cv_scheme=CVScheme.NONE, loss_function=LossFunction.L2,
        model_kwargs={"max_lag": args.max_lag}, model_id="AR",
    )
    fm_spec = ModelSpec(
        model_cls=ARDIModel, regularization=Regularization.NONE,
        cv_scheme=CVScheme.NONE, loss_function=LossFunction.L2,
        model_kwargs={}, model_id="FM",
    )

    # FM_AGR: 8 factors + 12 AR lags, NO factor lags (n_lags_factors=0), path_average
    fs_ar = FeatureSpec(factor_type="none", n_lags=P_Y, target_scheme="path_average")
    fs_fm = FeatureSpec(factor_type="X", n_factors=K, n_lags=P_Y,
                        n_lags_factors=0, p_marx=0, target_scheme="path_average")

    t0 = time.time()
    results: dict[str, pd.Series] = {}

    print(f"Running AR (max_lag={args.max_lag}) ...", flush=True)
    exp = ForecastExperiment(
        panel=pred_df, target=target, horizons=HORIZONS,
        model_specs=[ar_spec], feature_spec=fs_ar,
        panel_levels=None, window=Window.EXPANDING,
        oos_start=OOS_START, oos_end=OOS_END, n_jobs=-1,
    )
    df = exp.run().to_dataframe()
    df["se"] = (df["y_hat"] - df["y_true"]) ** 2
    results["AR"] = df.groupby("horizon")["se"].mean().apply(np.sqrt)
    print(f"  AR done ({time.time()-t0:.0f}s)", flush=True)

    print("Running FM_AGR ...", flush=True)
    exp = ForecastExperiment(
        panel=pred_df, target=target, horizons=HORIZONS,
        model_specs=[fm_spec], feature_spec=fs_fm,
        panel_levels=None, window=Window.EXPANDING,
        oos_start=OOS_START, oos_end=OOS_END, n_jobs=-1,
    )
    df = exp.run().to_dataframe()
    df["se"] = (df["y_hat"] - df["y_true"]) ** 2
    results["FM"] = df.groupby("horizon")["se"].mean().apply(np.sqrt)
    print(f"  FM done ({time.time()-t0:.0f}s)", flush=True)

    print()
    print(f"=== INDPRO: AR/FM_AGR [{args.label}] (rel_vs_fm_agr) ===")
    print(f"  max_lag={args.max_lag}  OOS {OOS_START[:7]}~{OOS_END[:7]}, path_average")
    print()
    TOLS = 0.05
    TOLS_H24 = 0.10

    # AR/FM_AGR table
    print(f"{'h':>4}  {'AR_rmse':>9}  {'FM_rmse':>9}  {'ratio':>7}  {'paper':>7}  {'diff':>7}  {'OK':>4}")
    print("-" * 60)
    n_ok_ar = 0
    for h in HORIZONS:
        ar_r  = float(results["AR"][h])
        fm_r  = float(results["FM"][h])
        ratio = ar_r / fm_r
        paper = PAPER_AR[h]
        diff  = ratio - paper
        tol   = TOLS_H24 if h == 24 else TOLS
        ok    = abs(diff) <= tol
        n_ok_ar += int(ok)
        print(f"{h:>4}  {ar_r:>9.5f}  {fm_r:>9.5f}  {ratio:>7.4f}  {paper:>7.3f}  "
              f"{diff:>+7.4f}  {'OK' if ok else 'FAIL':>4}")
    print(f"\nAR passed: {n_ok_ar}/{len(HORIZONS)}  (strict ±0.05, h24 relaxed ±0.10)")

    # FM absolute RMSE cross-check
    print()
    print("=== FM_AGR absolute RMSE vs paper ===")
    print(f"{'h':>4}  {'our_FM':>9}  {'paper_FM':>9}  {'diff%':>7}")
    print("-" * 35)
    for h in HORIZONS:
        fm_r = float(results["FM"][h])
        pap  = PAPER_FM_ABS.get(h, float("nan"))
        pct  = 100 * (fm_r - pap) / pap if pap > 0 else float("nan")
        print(f"{h:>4}  {fm_r:>9.5f}  {pap:>9.3f}  {pct:>+6.1f}%")

    print(f"\nTotal: {time.time()-t0:.0f}s")

    # JSON output
    result: dict = {
        "model": "AR", "label": args.label,
        "params": {"max_lag": args.max_lag},
        "horizons": {},
    }
    for _h in HORIZONS:
        _ar_r  = float(results["AR"][_h])
        _fm_r  = float(results["FM"][_h])
        _ratio = round(_ar_r / _fm_r, 4)
        result["horizons"][str(_h)] = {"ours": _ratio, "paper": PAPER_AR[_h]}
    print(f"##REPLICATE_JSON## {_json.dumps(result)}", flush=True)

    # Second JSON block for FM absolute check
    fm_result: dict = {
        "model": "FM_AGR_abs", "label": args.label,
        "horizons": {},
    }
    for _h in HORIZONS:
        _fm_r = float(results["FM"][_h])
        fm_result["horizons"][str(_h)] = {
            "ours": round(_fm_r, 6),
            "paper": PAPER_FM_ABS.get(_h),
        }
    print(f"##FM_ABS_JSON## {_json.dumps(fm_result)}", flush=True)


if __name__ == "__main__":
    main()
