"""G3 headline drivers: confirm the CV-convergence finding, then run the NL
(nonlinearity) and X (big-data) treatment regressions -- the paper's actual answer --
via within-transform + Driscoll-Kraay SE on the target-variance-normalised pseudo-R2."""
import sys, glob, math
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from scipy import stats
from scripts.replication.gcls_2022_pipeline.registry import build_gcls2022_arms
from scripts.replication.gcls_2022_pipeline.data import yobj_column

cells = (glob.glob("runs/gcls_b4_stage1/_result_store_indpro/cells/*.parquet")
         + glob.glob("runs/gcls_b4_stage1/_result_store_g2rest/cells/*.parquet"))
KEEP = ["date", "origin", "horizon", "target", "prediction", "actual", "arm"]
M = pd.concat([pd.read_parquet(f, columns=KEEP) for f in cells], ignore_index=True).dropna(
    subset=["prediction", "actual"])
tags = {a.name: dict(a.tags) for a in build_gcls2022_arms(yobj_column("INDPRO"))}
for k in ("X", "NL", "SH", "CV", "LF"):
    M[k] = M["arm"].map(lambda a, kk=k: tags[a][kk])
den = M.groupby(["target", "horizon"])["actual"].transform(lambda s: ((s - s.mean()) ** 2).mean())
M["R2"] = 100.0 * (1.0 - (M["actual"] - M["prediction"]) ** 2 / den)
M["_grp"] = M["target"].astype(str) + "|" + M["horizon"].astype(str) + "|" + M["date"].astype(str)

# ---- confirm CV convergence: AR,BIC vs AR,POOS vs AR,KF identical? ----
print("=== CV-convergence check (identical forecasts collapse the CV treatment) ===")
for base in ("AR", "ARDI"):
    piv = M[M["arm"].isin([f"{base},BIC", f"{base},POOS", f"{base},KF", f"{base},AIC"])]
    piv = piv.pivot_table(index=["target", "horizon", "origin"], columns="arm", values="prediction")
    d_poos = (piv[f"{base},POOS"] - piv[f"{base},BIC"]).abs().mean()
    d_kf = (piv[f"{base},KF"] - piv[f"{base},BIC"]).abs().mean()
    d_aic = (piv[f"{base},AIC"] - piv[f"{base},BIC"]).abs().mean()
    print(f"  {base}: mean|POOS-BIC|={d_poos:.2e}  mean|KF-BIC|={d_kf:.2e}  mean|AIC-BIC|={d_aic:.2e}")


def within_dk(df, cols):
    g = df.groupby("_grp")
    yd = (df["R2"] - g["R2"].transform("mean")).to_numpy(float)
    Xd = np.column_stack([(df[c] - g[c].transform("mean")).to_numpy(float) for c in cols])
    keep = np.all(np.abs(Xd) > 1e-12, axis=0) | True
    XtX = Xd.T @ Xd
    beta = np.linalg.lstsq(Xd, yd, rcond=None)[0]
    u = yd - Xd @ beta
    sc = Xd * u[:, None]
    tmp = pd.DataFrame(sc, columns=cols); tmp["_t"] = df["date"].to_numpy()
    H = tmp.groupby("_t").sum().sort_index().to_numpy(float)
    T = H.shape[0]; L = min(int(math.floor(4 * (T / 100) ** (2 / 9))), T - 1)
    S = H.T @ H
    for l in range(1, L + 1):
        Gl = H[l:].T @ H[:-l]; S += (1 - l / (L + 1)) * (Gl + Gl.T)
    XtX_inv = np.linalg.pinv(XtX)
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(V)); t = beta / se; p = 2 * (1 - stats.norm.cdf(np.abs(t)))
    return beta, se, t, p, len(df), T


print("\n=== NL (nonlinearity: KRR/RF) + X (big data) treatment regressions ===")
print("outcome = target-variance pseudo-R2 x100; FE phi_{t,v,h}; DK SE; over all 46 arms")
for label, sub, cols in [
    ("NL+X (all arms)", M, ["NL", "X"]),
    ("NL | data-rich", M[M["X"] == 1], ["NL"]),
    ("NL | data-poor", M[M["X"] == 0], ["NL"]),
    ("X (big data)", M, ["X"]),
]:
    beta, se, t, p, n, T = within_dk(sub, cols)
    print(f"\n  [{label}]  n={n}, T={T}")
    for i, c in enumerate(cols):
        star = "***" if p[i] < .001 else "**" if p[i] < .01 else "*" if p[i] < .05 else ""
        print(f"    {c:4s} coef={beta[i]:+.3f}  DKse={se[i]:.3f}  t={t[i]:+.2f}  p={p[i]:.4f} {star}")
print("\nOK g3_drivers")
