"""Isolate the regime mask from our forecasts.

Score the AUTHORS' OWN archived combination paths (Data_trend.mat FC_Trend_Linear,
five designs x 696 months) with our regime coding. If the printed Table 9 comes
back, the mask and the scoring are right, and any residual in our own numbers is
attributable to our forecasts rather than to how we split or score them.
"""
import numpy as np, scipy.io as sio

ARCH = "/home/nanyeon99/data/han_lu_zhou/extracted/Codes/Data"
t = sio.loadmat(f"{ARCH}/Data_trend.mat")
w = sio.loadmat(f"{ARCH}/WGpredictors2022.mat")
actual = t["actual"].ravel()
ha = t["FC_HA"].ravel()
fc = t["FC_Trend_Linear"]           # (696, 5)
P = len(actual)
g = w["ncf_full"].ravel()[-P:]
hi, lo = (g == 2), (g == 1)
print(f"P={P}  high={hi.sum()}  low={lo.sum()}")

def r2(sel, f):
    a = actual[sel]
    return 100.0 * (1.0 - np.sum((a - f[sel]) ** 2) / np.sum((a - ha[sel]) ** 2))

DESIGNS = ["Xt", "Xt+{MA2..MA6}", "Xt+{MA2..MA12}", "Xt+{MA2..MA24}", "Xt+{MA2..MA36}"]
PRINTED = [(0.60, 0.41, 0.78), (0.70, 0.38, 1.02), (0.81, 0.35, 1.26),
           (0.80, 0.30, 1.28), (0.73, 0.28, 1.17)]
full = np.ones(P, bool)
print("\n| design | overall | high | low |  (archive-scored vs printed)")
print("|---|---|---|---|")
d = []
for j, name in enumerate(DESIGNS):
    got = (r2(full, fc[:, j]), r2(hi, fc[:, j]), r2(lo, fc[:, j]))
    p = PRINTED[j]
    print(f"| {name} | {got[0]:.3f} / {p[0]:.2f} | {got[1]:.3f} / {p[1]:.2f} | {got[2]:.3f} / {p[2]:.2f} |")
    d += [abs(a - b) for a, b in zip(got, p)]
d = np.array(d)
print(f"\n15 cells: max|Δ| = {d.max():.4f}pp   mean|Δ| = {d.mean():.4f}pp   "
      f"within 0.005 = {(d <= 0.005).sum()}/15")
