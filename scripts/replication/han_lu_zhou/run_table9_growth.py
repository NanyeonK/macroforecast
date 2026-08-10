"""HLZ Table 9 -- R2_OS during high- and low-growth periods.

The forecast paths are the ones already produced for Table 3 and verified against
the authors' archived columns to 1e-16. Table 9 changes nothing about the
forecasts: it scores the SAME paths on two subsamples. So this runs through
`evaluate(master, spec)` with `EvalSpec(subsamples=...)` and refits nothing --
which is also the first production exercise of `SubsampleWindow(mask=)`.

Regime definition. The paper says only that regimes are "based on real net cash
flow growth", split at the median; it never defines net cash flow, and no
construction appears in either the article or the Internet Appendix. [GAP]
It is resolved from the authors' archive instead: `WGpredictors2022.mat` ships
`ncf_full`, already coded 1 = low / 2 = high, and `Additional_results.m` line 194
uses exactly that coding. We take the variable, not a reconstruction of it.
"""
import sys, warnings, json
sys.path.insert(0, ".")
import numpy as np, pandas as pd
import scipy.io as sio
import macroforecast as mf
from macroforecast.pipeline import (Arm, EvalSpec, CombinationContender, SubsampleWindow,
                                    pipeline_spec, evaluate, TargetSpec)

ARCH = "/home/nanyeon99/data/han_lu_zhou/extracted/Codes/Data"
panel = pd.read_parquet("/tmp/hlz_panel.parquet")
NAMES = ["DP", "DY", "EP", "DE", "BM", "NTIS", "TBL", "LTY", "LTR", "TMS",
         "DFY", "DFR", "INFL", "SVAR"]
econ, y, idx = panel[NAMES], panel["mkt_excess"], panel.index

ncf = sio.loadmat(f"{ARCH}/WGpredictors2022.mat")["ncf_full"].ravel()
assert len(ncf) == len(idx), (len(ncf), len(idx))
regime = pd.Series(ncf, index=idx)
HIGH = (regime == 2)
LOW = (regime == 1)
print(f"regime over the full index: high={int(HIGH.sum())} low={int(LOW.sum())}")

def lag_block(k):
    return {f"{v}_L{k-1}": econ[v].shift(k - 1) for v in NAMES}
def ma_block(L):
    return {f"{v}_MA{L}": econ[v].rolling(L, min_periods=L).mean() for v in NAMES}
SIGNALS = {
    "current": lambda: dict(lag_block(1)),
    "dense6":  lambda: {k: v for L in range(1, 7) for k, v in ma_block(L).items()},
    "dense12": lambda: {k: v for L in range(1, 13) for k, v in ma_block(L).items()},
}
LABEL = {"current": "Xt", "dense6": "Xt + {MA2..MA6}", "dense12": "Xt + {MA2..MA12}"}
# printed Table 9: (overall, high, low) for panels A/B
PRINTED = {"current": (0.60, 0.41, 0.78), "dense6": (0.70, 0.38, 1.02),
           "dense12": (0.81, 0.35, 1.26)}
# printed panel C: incremental vs panel A
PRINTED_C = {"dense6": (0.10, -0.04, 0.23), "dense12": (0.21, -0.07, 0.47)}

out = {}
for design in ["current", "dense6", "dense12"]:
    sig = SIGNALS[design]()
    cols = list(sig)
    frame = pd.concat([y.rename("y"), pd.DataFrame(sig, index=idx)], axis=1)
    bundle = mf.data.custom_dataset(frame, transform_codes={c: 1 for c in frame.columns})
    w = mf.window.from_cutoffs(test_start=idx[457], horizon=1, embargo=0,
                               val_method="expanding", val_min_train_size=24)
    arms = [Arm("HA", model="hist_mean",
                features=mf.feature_engineering.feature_spec(target="y", target_lags=(1,)),
                is_benchmark=True)]
    arms += [Arm(c, model="ols",
                 features=mf.feature_engineering.feature_spec(
                     target="y", predictors=[c], lags=0, target_lags=None),
                 nested_in_benchmark=True) for c in cols]
    spec = pipeline_spec(
        data=bundle, targets=[TargetSpec("y", transform="level")], horizons=[1], window=w,
        arms=arms,
        evaluation=EvalSpec(
            benchmark="HA", metrics=("relative_mse", "r2_oos"),
            tests=("dm", "cw"), test_options={"dm": {"hac_lags": 4}},
            subsamples={"overall": SubsampleWindow(),
                        "high_growth": SubsampleWindow(mask=HIGH),
                        "low_growth": SubsampleWindow(mask=LOW)}),
        # a simple mean of arms that each nest HA nests HA too, so CW is licensed;
        # the flag exists as of the fix this replication produced (PR: cw-for-combinations)
        combinations=[CombinationContender(name=f"COMB_{design}", method="mean",
                                           over=tuple(cols), nested_in_benchmark=True)],
        save_models=False,
    )
    master = pd.read_parquet(f"/tmp/hlz_fc_{design}.parquet")
    master = master[~master["contender"].astype(str).str.startswith("COMB_")].copy()
    master["window"] = None          # JSON-ified on save; evaluate does not read it
    print(f"\n### {design}: master {master.shape}, arms {master['contender'].nunique()}", flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = evaluate(master, spec)
    acc = res["accuracy"]
    if design == "current":
        print("accuracy cols:", list(acc.columns))
        sub_col = [c for c in acc.columns if "subsample" in c.lower() or c == "sample"]
        print("subsample-ish cols:", sub_col,
              "values:", {c: sorted(map(str, acc[c].dropna().unique()))[:6] for c in sub_col})
    out[design] = (acc, res.get("significance"))
    acc.to_parquet(f"/tmp/hlz_t9_acc_{design}.parquet")
    sg = res.get("significance")
    if isinstance(sg, pd.DataFrame) and len(sg):
        sg.to_parquet(f"/tmp/hlz_t9_sig_{design}.parquet")
print("\nEVAL_DONE")
