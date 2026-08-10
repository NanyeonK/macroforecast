"""B5: derive the Table-2 target values from the AUTHORS' archived forecast paths
(Dataverse doi:10.7910/DVN/7APSCU, CC0), using their own R2_OS and DM definitions.

These are the numbers their published scripts turn into Table 2, so they are the
comparison target for the macroforecast replication -- and because the archive
stores the full 696-month OOS path, parity can later be checked forecast-by-forecast
rather than only on the summary statistic.
"""
import scipy.io as sio
import numpy as np
from scipy.stats import norm

D = "/home/nanyeon99/data/han_lu_zhou/extracted/Codes/Data"
d = sio.loadmat(f"{D}/Data_trend.mat")
actual = d["actual"].ravel()
ha = d["FC_HA"].ravel()
fc = np.asarray(d["FC_Trend_Linear"], dtype=float)   # (696, 5)
print("OOS months:", len(actual), "| R =", int(d["R"].ravel()[0]), "| P =", int(d["P"].ravel()[0]))

def r2_os(a, b, f):
    return 100.0 * (1.0 - np.nanmean((a - f) ** 2) / np.nanmean((a - b) ** 2))

def dm_nw(y, f1, f2, L=4):
    y, f1, f2 = y.ravel(), f1.ravel(), f2.ravel()
    T = len(y)
    dd = (y - f1) ** 2 - (y - f2) ** 2
    dbar = dd.mean()
    e = dd - dbar
    g0 = (e ** 2).sum() / T
    g = [(e[:-l] * e[l:]).sum() / T for l in range(1, L + 1)]
    var = g0 + 2.0 * sum((1 - l / (L + 1)) * g[l - 1] for l in range(1, L + 1))
    t = dbar / np.sqrt(var / T)
    return t, 1 - norm.cdf(t)

rows = ["Current value", "3 lags", "MA6", "6 lags", "MA12"]
print("\n=== authors' Table 2 (from their archived forecast paths) ===")
print("| row | R2_OS (%) | dR2 vs current | DM t (NW4) | DM p |")
print("|---|---|---|---|---|")
base = fc[:, 0]
r2s = []
for i, name in enumerate(rows):
    r = r2_os(actual, ha, fc[:, i])
    r2s.append(r)
    if i == 0:
        print(f"| {name} | {r:7.3f} | -- | -- | -- |")
    else:
        t, p = dm_nw(actual, base, fc[:, i], 4)
        print(f"| {name} | {r:7.3f} | {r - r2s[0]:+7.3f} | {t:6.3f} | {p:.4f} |")

np.savez("/tmp/hlz_target_table2.npz", actual=actual, ha=ha, fc=fc,
         r2=np.array(r2s), rows=np.array(rows, dtype=object))
print("\nsaved /tmp/hlz_target_table2.npz")

# what other families are archived (targets for Tables 4-6)
print("\n=== other archived forecast families (Tables 4-6 targets) ===")
for k in sorted(d):
    if k.startswith("FC_") and k not in ("FC_HA", "FC_Trend_Linear"):
        v = np.asarray(d[k], dtype=float)
        r = [r2_os(actual, ha, v[:, j]) for j in range(v.shape[1])]
        print(f"  {k:16s} {v.shape}  R2_OS = " + ", ".join(f"{x:6.3f}" for x in r))
