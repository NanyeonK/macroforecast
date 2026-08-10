"""How much of Table 6 moves if the benchmark is the prevailing mean instead of
the package hist_mean (issue #488)?"""
import numpy as np, pandas as pd
panel = pd.read_parquet("/tmp/hlz_panel.parquet")
yv = panel["mkt_excess"].to_numpy(float)
LINEAR = ["PCR", "PLS", "S-PCR", "LASSO", "ENet"]
for design, label in (("L1", "Xt"), ("L6", "Xt+{MA2..MA6}"), ("L12", "Xt+{MA2..MA12}")):
    t4 = pd.read_parquet(f"/tmp/hlz_t4_{design}_J1.parquet")
    t4s = pd.read_parquet(f"/tmp/hlz_t4s_{design}.parquet")
    ha = t4[t4["contender"] == "HA"].sort_values("date")
    dates = pd.DatetimeIndex(ha["date"])
    loc = panel.index.get_indexer(dates)
    a = ha["actual"].to_numpy(float)
    h_pkg = ha["prediction"].to_numpy(float)
    h_prev = np.array([np.mean(yv[:i]) for i in loc])
    comps = []
    for m in ("PCR", "PLS", "S-PCR"):
        comps.append(t4[t4["contender"] == f"COMB_{m}"].sort_values("date")["prediction"].to_numpy(float))
    for cand in (["COMB_LASSO", "LASSO_lag1"], ["COMB_ENet", "ENet_lag1"]):
        for c in cand:
            r = t4s[t4s["contender"] == c]
            if len(r):
                comps.append(r.sort_values("date")["prediction"].to_numpy(float)); break
    ens = np.mean(np.vstack(comps), axis=0)
    r2 = lambda h: 100.0 * (1 - np.sum((a - ens) ** 2) / np.sum((a - h) ** 2))
    print(f"| {label:16s} Linear | package HA {r2(h_pkg):7.3f} | prevailing {r2(h_prev):7.3f} | "
          f"Δ {r2(h_prev) - r2(h_pkg):+.3f}pp | HA max|Δ| {np.max(np.abs(h_pkg - h_prev)):.2e} |")
