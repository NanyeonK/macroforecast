"""Solve for the estimation start the dense6-bundle HA actually used."""
import numpy as np, pandas as pd, scipy.io as sio
ARCH = "/home/nanyeon99/data/han_lu_zhou/extracted/Codes/Data"
ha_auth = sio.loadmat(f"{ARCH}/Data_trend.mat")["FC_HA"].ravel()
panel = pd.read_parquet("/tmp/hlz_panel.parquet")
yv = panel["mkt_excess"].to_numpy(float)
R, P = 457, 696
for design in ("current", "dense6", "dense12"):
    try:
        f = pd.read_parquet(f"/tmp/hlz_fc_{design}.parquet")
    except Exception:
        continue
    ha = f[f["contender"] == "HA"].sort_values("date")["prediction"].to_numpy(float)
    if len(ha) != P:
        print(f"{design}: n={len(ha)} (not 696) -- skipping"); continue
    best = min(((np.max(np.abs(np.array([np.mean(yv[a : R + t - 1]) for t in range(1, P + 1)]) - ha)), a)
                for a in range(0, 40)))
    d, a = best
    print(f"{design:9s} HA[0]={ha[0]:.6f}  best start index a={a:2d} (max|Δ| {d:.2e})  "
          f"n@origin0={R-a}  vs authors' a=0 (n=457), max|Δ| vs authors {np.max(np.abs(ha - ha_auth)):.3e}")
print(f"\nMA window needed by design: current=1, dense6=6 (5 leading NaN), dense12=12 (11 leading NaN)")
