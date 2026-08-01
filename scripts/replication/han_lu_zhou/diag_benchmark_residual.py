"""The dense6 forecast path is bit-identical to the authors'. So the residual
R2_OS gap must live in the denominator (HA) or in the realized returns. Swap one
piece at a time and see which one moves the number."""
import numpy as np, pandas as pd, scipy.io as sio
ARCH = "/home/nanyeon99/data/han_lu_zhou/extracted/Codes/Data"
t = sio.loadmat(f"{ARCH}/Data_trend.mat")
a_auth = t["actual"].ravel(); ha_auth = t["FC_HA"].ravel(); fc_auth = t["FC_Trend_Linear"][:, 1]

f = pd.read_parquet("/tmp/hlz_fc_dense6.parquet")
comb = f[f["contender"] == "COMB_dense6"].sort_values("date")
ha = f[f["contender"] == "HA"].sort_values("date")
fc_mine = comb["prediction"].to_numpy(float)
a_mine = comb["actual"].to_numpy(float)
ha_mine = ha["prediction"].to_numpy(float)
n = len(fc_mine)
print(f"n mine = {n}, n authors = {len(a_auth)}")
assert n == len(a_auth), "length mismatch"

def r2(a, h, f_):
    return 100.0 * (1.0 - np.sum((a - f_) ** 2) / np.sum((a - h) ** 2))

print(f"\nforecast path : max|Δ| = {np.max(np.abs(fc_mine - fc_auth)):.3e}")
print(f"HA path       : max|Δ| = {np.max(np.abs(ha_mine - ha_auth)):.3e}")
print(f"actuals       : max|Δ| = {np.max(np.abs(a_mine - a_auth)):.3e}")
print("\n| actual | HA | forecast | R2_OS % |")
print("|---|---|---|---|")
for la, aa in (("mine", a_mine), ("authors", a_auth)):
    for lh, hh in (("mine", ha_mine), ("authors", ha_auth)):
        for lf, ff in (("mine", fc_mine), ("authors", fc_auth)):
            print(f"| {la} | {lh} | {lf} | {r2(aa, hh, ff):.4f} |")
i = np.argsort(-np.abs(ha_mine - ha_auth))[:5]
print("\nlargest HA differences:")
for k in i:
    print(f"  idx {k:3d} ({comb['date'].iloc[k].date()}): mine {ha_mine[k]:.6f}  "
          f"authors {ha_auth[k]:.6f}  Δ {ha_mine[k]-ha_auth[k]:+.6f}")
