"""Which HA feature spec reproduces the authors' prevailing mean exactly?"""
import sys, warnings; sys.path.insert(0, ".")
import numpy as np, pandas as pd, scipy.io as sio
import macroforecast as mf
from macroforecast.pipeline import Arm, EvalSpec, pipeline_spec, run_pipeline, TargetSpec

ARCH = "/home/nanyeon99/data/han_lu_zhou/extracted/Codes/Data"
ha_auth = sio.loadmat(f"{ARCH}/Data_trend.mat")["FC_HA"].ravel()
panel = pd.read_parquet("/tmp/hlz_panel.parquet")
y = panel["mkt_excess"]; idx = panel.index
frame = pd.DataFrame({"y": y, "x": panel["DP"]}, index=idx)
bundle = mf.data.custom_dataset(frame, transform_codes={c: 1 for c in frame.columns})
w = mf.window.from_cutoffs(test_start=idx[456], horizon=1, embargo=0,
                           val_method="expanding", val_min_train_size=24)

VARIANTS = {
    "target_lags=(1,)  [current]": dict(target="y", target_lags=(1,)),
    "target_lags=(0,)":            dict(target="y", target_lags=(0,)),
    "target_lags=None":            dict(target="y", target_lags=None),
    "target_lags=None, lags=None": dict(target="y", target_lags=None, lags=None),
}
print("| HA feature spec | n | max|Δ| vs authors' FC_HA |")
print("|---|---|---|")
for label, kw in VARIANTS.items():
    try:
        feats = mf.feature_engineering.feature_spec(**kw)
        spec = pipeline_spec(data=bundle, targets=[TargetSpec("y", transform="level")],
                             horizons=[1], window=w,
                             arms=[Arm("HA", model="hist_mean", features=feats, is_benchmark=True)],
                             evaluation=EvalSpec(benchmark="HA", metrics=("r2_oos",), tests=()),
                             save_models=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rep = run_pipeline(spec)
        p = rep.forecasts.sort_values("date")["prediction"].to_numpy(float)
        n = min(len(p), len(ha_auth))
        print(f"| {label} | {len(p)} | {np.max(np.abs(p[-n:] - ha_auth[-n:])):.3e} |")
    except Exception as e:
        print(f"| {label} | -- | {type(e).__name__}: {str(e)[:60]} |")
