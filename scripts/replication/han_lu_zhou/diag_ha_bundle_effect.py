"""Does the hist_mean window depend on OTHER arms' columns in the same bundle?

Same HA arm, same window spec, two bundles: one carrying only {y, DP}, one
carrying the full dense6 MA block (whose leading rows are NaN).
"""
import sys, warnings; sys.path.insert(0, ".")
import numpy as np, pandas as pd, scipy.io as sio
import macroforecast as mf
from macroforecast.models import timeseries as ts
_orig = ts._HistoricalMeanForecaster.fit
CAP = []
def fit(self, X, y, *a, **k):
    arr = np.asarray(y, float).ravel()
    CAP.append((len(arr), float(np.mean(arr)), float(arr[0])))
    return _orig(self, X, y, *a, **k)
ts._HistoricalMeanForecaster.fit = fit
from macroforecast.pipeline import Arm, EvalSpec, pipeline_spec, run_pipeline, TargetSpec

ARCH = "/home/nanyeon99/data/han_lu_zhou/extracted/Codes/Data"
ha_auth = sio.loadmat(f"{ARCH}/Data_trend.mat")["FC_HA"].ravel()
panel = pd.read_parquet("/tmp/hlz_panel.parquet")
NAMES = ["DP","DY","EP","DE","BM","NTIS","TBL","LTY","LTR","TMS","DFY","DFR","INFL","SVAR"]
econ, y, idx = panel[NAMES], panel["mkt_excess"], panel.index

def ma_block(L):
    return {f"{v}_MA{L}": econ[v].rolling(L, min_periods=L).mean() for v in NAMES}

FRAMES = {
    "minimal {y, DP}": pd.DataFrame({"y": y, "x": econ["DP"]}, index=idx),
    "dense6 bundle (84 MA cols)": pd.concat(
        [y.rename("y"), pd.DataFrame({k: v for L in range(1, 7) for k, v in ma_block(L).items()},
                                     index=idx)], axis=1),
}
print("| bundle | n rows HA saw @origin0 | HA[0] | max|Δ| vs authors' FC_HA |")
print("|---|---|---|---|")
for label, frame in FRAMES.items():
    CAP.clear()
    bundle = mf.data.custom_dataset(frame, transform_codes={c: 1 for c in frame.columns})
    w = mf.window.from_cutoffs(test_start=idx[456], horizon=1, embargo=0,
                               val_method="expanding", val_min_train_size=24)
    spec = pipeline_spec(data=bundle, targets=[TargetSpec("y", transform="level")],
                         horizons=[1], window=w,
                         arms=[Arm("HA", model="hist_mean",
                                   features=mf.feature_engineering.feature_spec(
                                       target="y", target_lags=(1,)), is_benchmark=True)],
                         evaluation=EvalSpec(benchmark="HA", metrics=("r2_oos",), tests=()),
                         save_models=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rep = run_pipeline(spec)
    p = rep.forecasts.sort_values("date")["prediction"].to_numpy(float)
    n = min(len(p), len(ha_auth))
    print(f"| {label} | {CAP[0][0] if CAP else '?'} | {p[0]:.6f} | "
          f"{np.max(np.abs(p[-n:] - ha_auth[-n:])):.3e} |")
yv = y.to_numpy(float)
print(f"\nauthors' mean(y[0:457]) = {np.mean(yv[:457]):.6f}   "
      f"mean(y[5:457]) = {np.mean(yv[5:457]):.6f}   mean(y[11:457]) = {np.mean(yv[11:457]):.6f}")
