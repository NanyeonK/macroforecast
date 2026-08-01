"""What target window does the hist_mean arm actually average at each origin?"""
import sys, warnings, json; sys.path.insert(0, ".")
import numpy as np, pandas as pd, scipy.io as sio
import macroforecast as mf
from macroforecast.models import timeseries as ts
_orig = ts._HistoricalMeanForecaster.fit
CAP = []
def fit(self, X, y, *a, **k):
    arr = np.asarray(y, float).ravel()
    CAP.append((len(arr), float(np.mean(arr)), float(arr[0]), float(arr[-1])))
    return _orig(self, X, y, *a, **k)
ts._HistoricalMeanForecaster.fit = fit

from macroforecast.pipeline import Arm, EvalSpec, pipeline_spec, run_pipeline, TargetSpec
panel = pd.read_parquet("/tmp/hlz_panel.parquet")
y = panel["mkt_excess"]; idx = panel.index
frame = pd.DataFrame({"y": y, "x": panel["DP"]}, index=idx)
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
f = rep.forecasts.sort_values("date")
yv = y.to_numpy(float)
print(f"first 3 origins -> (n_target_rows, mean, first, last):")
for c in CAP[:3]:
    print("  ", c)
print(f"\npipeline HA[0..2] = {np.round(f['prediction'].to_numpy(float)[:3], 6)}")
print(f"authors' rule mean(y[0:457]) = {np.mean(yv[:457]):.6f}   "
      f"mean(y[0:456]) = {np.mean(yv[:456]):.6f}")
print(f"y[0] = {yv[0]:.6f}  y[455] = {yv[455]:.6f}  y[456] = {yv[456]:.6f}")
n0 = CAP[0][0]
print(f"\nHA arm saw {n0} target rows; y[0:{n0}] mean = {np.mean(yv[:n0]):.6f}  "
      f"y[1:{n0+1}] mean = {np.mean(yv[1:n0+1]):.6f}")
print(f"  arm's first value {CAP[0][2]:.6f} == y[0]? {abs(CAP[0][2]-yv[0])<1e-12}"
      f"  == y[1]? {abs(CAP[0][2]-yv[1])<1e-12}")
print(f"  arm's last value  {CAP[0][3]:.6f} == y[{n0-1}]? {abs(CAP[0][3]-yv[n0-1])<1e-12}")
