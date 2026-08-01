"""The whole residual is the HA benchmark. Which expanding window is it?"""
import numpy as np, pandas as pd, scipy.io as sio
ARCH = "/home/nanyeon99/data/han_lu_zhou/extracted/Codes/Data"
t = sio.loadmat(f"{ARCH}/Data_trend.mat")
ha_auth = t["FC_HA"].ravel(); y = t["y"].ravel(); R = int(t["R"].ravel()[0]); P = int(t["P"].ravel()[0])
print(f"R={R} P={P} len(y)={len(y)}")
f = pd.read_parquet("/tmp/hlz_fc_dense6.parquet")
ha_mine = f[f["contender"] == "HA"].sort_values("date")["prediction"].to_numpy(float)

# authors: FC_HA(t) = mean(y(1 : R+t-1))  (1-based) -> y[0 : R+t-1] 0-based
cand = {}
for drop in (0, 1, 2):
    for shift in (0, -1):
        v = np.array([np.mean(y[drop : R + tt - 1 + shift]) for tt in range(1, P + 1)])
        cand[f"y[{drop}:R+t-1{'' if shift==0 else f'{shift}'}]"] = v
print("\n| window | max|Δ| vs authors | max|Δ| vs mine |")
print("|---|---|---|")
for k, v in cand.items():
    print(f"| {k} | {np.max(np.abs(v - ha_auth)):.3e} | {np.max(np.abs(v - ha_mine)):.3e} |")
print(f"\ny[0] = {y[0]:.6f}   mean(y[:573]) = {np.mean(y[:573]):.6f}   "
      f"mean(y[1:573]) = {np.mean(y[1:573]):.6f}")
print(f"authors HA[117] = {ha_auth[117]:.6f}   mine = {ha_mine[117]:.6f}")
