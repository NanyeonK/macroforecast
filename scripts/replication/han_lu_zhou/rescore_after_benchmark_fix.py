"""Run the HA benchmark on all three HLZ bundles with the FIXED package.

Only the benchmark changes: the COMB paths are ols arms (supervised, untouched),
so they are reused from the saved runs. What is recomputed here is the
denominator, and then every number the note reports.
"""
import sys, warnings, json; sys.path.insert(0, ".")
import numpy as np, pandas as pd, scipy.io as sio
import macroforecast as mf
from macroforecast.pipeline import Arm, EvalSpec, TargetSpec, pipeline_spec, run_pipeline
warnings.simplefilter("ignore")

ARCH = "/home/nanyeon99/data/han_lu_zhou/extracted/Codes/Data"
t = sio.loadmat(f"{ARCH}/Data_trend.mat")
ha_auth = t["FC_HA"].ravel()
panel = pd.read_parquet("/tmp/hlz_panel.parquet")
NAMES = ["DP","DY","EP","DE","BM","NTIS","TBL","LTY","LTR","TMS","DFY","DFR","INFL","SVAR"]
y, idx = panel["mkt_excess"], panel.index
yv = y.to_numpy(float)
regime = pd.Series(sio.loadmat(f"{ARCH}/WGpredictors2022.mat")["ncf_full"].ravel(), index=idx)

def ma_block(L):
    return {f"{v}_MA{L}": panel[v].rolling(L, min_periods=L).mean() for v in NAMES}
SIG = {
    "current": lambda: {f"{v}_L0": panel[v] for v in NAMES},
    "dense6":  lambda: {k: v for L in range(1, 7) for k, v in ma_block(L).items()},
    "dense12": lambda: {k: v for L in range(1, 13) for k, v in ma_block(L).items()},
}
PRINTED = {"current": 0.599, "dense6": 0.699, "dense12": 0.805}
REF = {"current": (0.599, 0.411, 0.782), "dense6": (0.699, 0.375, 1.016),
       "dense12": (0.805, 0.345, 1.255)}
LAB = {"current": "Xt", "dense6": "Xt+{MA2..MA6}", "dense12": "Xt+{MA2..MA12}"}

out = {}
for d in ("current", "dense6", "dense12"):
    frame = pd.concat([y.rename("y"), pd.DataFrame(SIG[d](), index=idx)], axis=1)
    b = mf.data.custom_dataset(frame, transform_codes={c: 1 for c in frame.columns})
    w = mf.window.from_cutoffs(test_start=idx[456], horizon=1, embargo=0,
                               val_method="expanding", val_min_train_size=24)
    spec = pipeline_spec(data=b, targets=[TargetSpec("y", transform="level")],
                         horizons=[1], window=w,
                         arms=[Arm("HA", model="hist_mean",
                                   features=mf.feature_engineering.feature_spec(
                                       target="y", target_lags=(1,)), is_benchmark=True)],
                         evaluation=EvalSpec(benchmark="HA", metrics=("r2_oos",), tests=()),
                         save_models=False)
    rep = run_pipeline(spec)
    hrows = rep.forecasts.sort_values("date")
    ha_new = hrows["prediction"].to_numpy(float)
    hdates = pd.DatetimeIndex(hrows["date"])

    f = pd.read_parquet(f"/tmp/hlz_fc_{d}.parquet")
    c = f[f["contender"] == f"COMB_{d}"].sort_values("date")
    cd = pd.DatetimeIndex(c["date"])
    fc = c["prediction"].to_numpy(float); a = c["actual"].to_numpy(float)
    sel = hdates.isin(cd)
    ha_al = ha_new[sel]
    assert len(ha_al) == len(fc), (len(ha_al), len(fc))
    g = regime.reindex(cd).to_numpy()
    masks = [("overall", np.ones(len(fc), bool)), ("high", g == 2), ("low", g == 1)]
    vals = [100.0 * (1 - np.sum((a[m] - fc[m]) ** 2) / np.sum((a[m] - ha_al[m]) ** 2))
            for _, m in masks]
    n = min(len(ha_new), len(ha_auth))
    out[d] = dict(n=int(len(fc)), r2=[round(v, 3) for v in vals],
                  vs_auth=float(np.max(np.abs(ha_new[-n:] - ha_auth[-n:]))),
                  vs_y1t=float(np.max(np.abs(ha_al - np.array(
                      [np.mean(yv[1:i]) for i in panel.index.get_indexer(cd)])))))
    print(f"{LAB[d]:16s} R2 overall/high/low = {vals[0]:.3f} / {vals[1]:.3f} / {vals[2]:.3f}"
          f"   printed {PRINTED[d]:.3f} (Δ {vals[0]-PRINTED[d]:+.3f})")
    print(f"{'':16s} HA vs authors {out[d]['vs_auth']:.3e}   vs mean(y[1:t]) {out[d]['vs_y1t']:.3e}")
json.dump(out, open("/tmp/hlz_post_fix.json", "w"), indent=1)
print("\nPOST_FIX_DONE")
