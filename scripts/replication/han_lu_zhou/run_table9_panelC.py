"""Panel C of Table 9: incremental R2_OS of the trend designs over Xt, with the
DM test the paper uses -- trend forecast against CURRENT-VALUE forecast (not
against HA), Newey-West with four lags.

The benchmark for panel C is a different contender from a different pipeline, so
this calls `mf.tests.dm_test` on the two aligned paths directly rather than
re-running an evaluation with a swapped benchmark.
"""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd, scipy.io as sio
import macroforecast as mf

ARCH = "/home/nanyeon99/data/han_lu_zhou/extracted/Codes/Data"
panel = pd.read_parquet("/tmp/hlz_panel.parquet")
regime = pd.Series(sio.loadmat(f"{ARCH}/WGpredictors2022.mat")["ncf_full"].ravel(), index=panel.index)

def path(design):
    f = pd.read_parquet(f"/tmp/hlz_fc_{design}.parquet")
    c = f[f["contender"] == f"COMB_{design}"].sort_values("date")
    return c.set_index("date")[["prediction", "actual"]]

base = path("current")
PRINTED_C = {"dense6": (0.10, -0.04, 0.23), "dense12": (0.21, -0.07, 0.47)}
LABEL = {"dense6": "Xt + {MA2..MA6}", "dense12": "Xt + {MA2..MA12}"}
MASKS = [("overall", None), ("high growth", 2), ("low growth", 1)]

print("| design | subsample | n | ΔR2_OS pp (mine) | printed | Δ | DM stat | DM p |")
print("|---|---|---|---|---|---|---|---|")
for d in ["dense6", "dense12"]:
    trend = path(d)
    common = base.index.intersection(trend.index)
    a = base.loc[common, "actual"].to_numpy(float)
    fb = base.loc[common, "prediction"].to_numpy(float)
    ft = trend.loc[common, "prediction"].to_numpy(float)
    hist = pd.Series(a, index=common)          # HA denominator is rebuilt per subsample
    for j, (name, code) in enumerate(MASKS):
        sel = np.ones(len(common), bool) if code is None else (regime.reindex(common) == code).to_numpy()
        aa, bb, tt = a[sel], fb[sel], ft[sel]
        ha = np.full_like(aa, np.nan)
        # HA path for the same dates, taken from the evaluated frame's benchmark
        hb = pd.read_parquet("/tmp/hlz_fc_current.parquet")
        hb = hb[hb["contender"] == "HA"].set_index("date")["prediction"].reindex(common).to_numpy(float)[sel]
        r2 = lambda f: 1.0 - np.sum((aa - f) ** 2) / np.sum((aa - hb) ** 2)
        inc = (r2(tt) - r2(bb)) * 100.0
        dm = mf.tests.dm_test((aa - bb) ** 2, (aa - tt) ** 2, horizon=1, hac_lags=4)
        p = PRINTED_C[d][j]
        print(f"| {LABEL[d]} | {name} | {int(sel.sum())} | {inc:+.3f} | {p:+.2f} | "
              f"{inc - p:+.3f} | {dm.statistic:.3f} | {dm.p_value:.3f} |")
