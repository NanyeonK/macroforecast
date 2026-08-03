"""Table 9 on the corrected 696-month paths, under both benchmarks."""
import numpy as np, pandas as pd, scipy.io as sio
ARCH = "/home/nanyeon99/data/han_lu_zhou/extracted/Codes/Data"
panel = pd.read_parquet("/tmp/hlz_panel.parquet")
yv = panel["mkt_excess"].to_numpy(float)
regime = pd.Series(sio.loadmat(f"{ARCH}/WGpredictors2022.mat")["ncf_full"].ravel(), index=panel.index)
REF = {  # archive-scored targets (unrounded), from verify_table9_mask.py
    "current": (0.599, 0.411, 0.782), "dense6": (0.699, 0.375, 1.016),
    "dense12": (0.805, 0.345, 1.255)}
PRINTED = {"current": (0.60, 0.41, 0.78), "dense6": (0.70, 0.38, 1.02),
           "dense12": (0.81, 0.35, 1.26)}
LAB = {"current": "Xt", "dense6": "Xt + {MA2..MA6}", "dense12": "Xt + {MA2..MA12}"}
print("| design | benchmark | overall | high growth | low growth | n |")
print("|---|---|---|---|---|---|")
store = {}
for d in ("current", "dense6", "dense12"):
    f = pd.read_parquet(f"/tmp/hlz_fc_{d}.parquet")
    comb = f[f["contender"] == f"COMB_{d}"].sort_values("date")
    ha = f[f["contender"] == "HA"].sort_values("date")["prediction"].to_numpy(float)
    fc = comb["prediction"].to_numpy(float); a = comb["actual"].to_numpy(float)
    dates = pd.DatetimeIndex(comb["date"])
    loc = panel.index.get_indexer(dates)
    prev = np.array([np.mean(yv[:i]) for i in loc])
    g = regime.reindex(dates).to_numpy()
    masks = [("overall", np.ones(len(fc), bool)), ("high", g == 2), ("low", g == 1)]
    for lab, h in (("package HA", ha), ("prevailing", prev)):
        vals = [100.0 * (1 - np.sum((a[m] - fc[m]) ** 2) / np.sum((a[m] - h[m]) ** 2))
                for _, m in masks]
        store[(d, lab)] = vals
        print(f"| {LAB[d]} | {lab} | {vals[0]:.3f} | {vals[1]:.3f} | {vals[2]:.3f} | {len(fc)} |")
    r = REF[d]
    print(f"| | *archive target* | {r[0]:.3f} | {r[1]:.3f} | {r[2]:.3f} | 696 |")
print("\n| design | max|Δ| vs archive target (package HA) | (prevailing) |")
print("|---|---|---|")
for d in ("current", "dense6", "dense12"):
    r = np.array(REF[d])
    dp = np.max(np.abs(np.array(store[(d, "package HA")]) - r))
    dv = np.max(np.abs(np.array(store[(d, "prevailing")]) - r))
    print(f"| {LAB[d]} | {dp:.4f} | **{dv:.4f}** |")
