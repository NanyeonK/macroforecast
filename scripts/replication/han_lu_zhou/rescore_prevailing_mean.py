"""Score our own forecast paths against a prevailing mean computed from OUR panel.

Not the authors' FC_HA array -- the textbook benchmark, mean of the target over
everything observed up to each origin, built here from our own data. This is the
variant that issue #488 would make the package produce.
"""
import numpy as np, pandas as pd, scipy.io as sio
ARCH = "/home/nanyeon99/data/han_lu_zhou/extracted/Codes/Data"
ha_auth = sio.loadmat(f"{ARCH}/Data_trend.mat")["FC_HA"].ravel()
panel = pd.read_parquet("/tmp/hlz_panel.parquet")
yv = panel["mkt_excess"].to_numpy(float)
PRINTED = {"current": 0.599, "dense6": 0.699, "dense12": 0.805}
LABEL = {"current": "Xt", "dense6": "Xt + {MA2..MA6}", "dense12": "Xt + {MA2..MA12}"}

print("| design | n | package HA | prevailing mean (ours) | printed | Δ (package) | Δ (prevailing) |")
print("|---|---|---|---|---|---|---|")
for d in ("current", "dense6", "dense12"):
    try:
        f = pd.read_parquet(f"/tmp/hlz_fc_{d}.parquet")
    except Exception:
        continue
    comb = f[f["contender"] == f"COMB_{d}"].sort_values("date")
    ha = f[f["contender"] == "HA"].sort_values("date")["prediction"].to_numpy(float)
    fc = comb["prediction"].to_numpy(float)
    a = comb["actual"].to_numpy(float)
    # combination rows carry no origin_pos; recover the target index from the date
    loc = panel.index.get_indexer(pd.DatetimeIndex(comb["date"]))
    assert (loc >= 0).all(), "dates not found in panel"
    prev = np.array([np.mean(yv[:i]) for i in loc])   # everything observed before the target
    r2 = lambda h: 100.0 * (1.0 - np.sum((a - fc) ** 2) / np.sum((a - h) ** 2))
    pk, pv = r2(ha), r2(prev)
    p = PRINTED[d]
    print(f"| {LABEL[d]} | {len(fc)} | {pk:.3f} | **{pv:.3f}** | {p:.3f} | "
          f"{pk - p:+.3f} | **{pv - p:+.3f}** |")
    if len(prev) == len(ha_auth):
        print(f"|   |   | (our prevailing vs authors' FC_HA: max|Δ| = "
              f"{np.max(np.abs(prev - ha_auth)):.3e}) | | | | |")
