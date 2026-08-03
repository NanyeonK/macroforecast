"""Collect the final Table 3 numbers (package HA vs prevailing mean vs printed)."""
import json, numpy as np, pandas as pd, scipy.io as sio
ARCH = "/home/nanyeon99/data/han_lu_zhou/extracted/Codes/Data"
ref = sio.loadmat(f"{ARCH}/Data_trend.mat")["FC_Trend_Linear"]
COL = {"current": 0, "dense6": 1, "dense12": 2}
PRINTED = {"current": 0.599, "dense6": 0.699, "dense12": 0.805}
panel = pd.read_parquet("/tmp/hlz_panel.parquet")
yv = panel["mkt_excess"].to_numpy(float)
out = {}
for d in ("current", "dense6", "dense12"):
    f = pd.read_parquet(f"/tmp/hlz_fc_{d}.parquet")
    comb = f[f["contender"] == f"COMB_{d}"].sort_values("date")
    ha = f[f["contender"] == "HA"].sort_values("date")["prediction"].to_numpy(float)
    fc = comb["prediction"].to_numpy(float); a = comb["actual"].to_numpy(float)
    loc = panel.index.get_indexer(pd.DatetimeIndex(comb["date"]))
    prev = np.array([np.mean(yv[:i]) for i in loc])
    r2 = lambda h: 100.0 * (1 - np.sum((a - fc) ** 2) / np.sum((a - h) ** 2))
    n = len(fc)
    par = np.max(np.abs(fc - ref[-n:, COL[d]]))
    out[d] = {"n": int(n), "pkg": round(r2(ha), 3), "prev": round(r2(prev), 3),
              "printed": PRINTED[d], "parity": f"{par:.2e}"}
    print(d, out[d])
json.dump(out, open("/tmp/hlz_t3_final.json", "w"), indent=1)
