"""G3 Table 2 (CVreg_main) faithful reproduction via within-transform (absorbing the
phi_{t,v,h} FE) + Driscoll-Kraay SE (same convention as axis_contribution, but scalable
to the 91200-obs panel). Outcome = target-variance-normalised pseudo-R2 (Eq r2_eq),
regressors CV-KF/CV-POOS/AIC (BIC baseline), over the 8 AR+ARDI models.
"""
import sys, glob, math
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from scripts.replication.gcls_2022_pipeline.registry import build_gcls2022_arms
from scripts.replication.gcls_2022_pipeline.data import yobj_column

cells = (glob.glob("runs/gcls_b4_stage1/_result_store_indpro/cells/*.parquet")
         + glob.glob("runs/gcls_b4_stage1/_result_store_g2rest/cells/*.parquet"))
KEEP = ["date", "origin", "horizon", "target", "prediction", "actual", "arm"]
M = pd.concat([pd.read_parquet(f, columns=KEEP) for f in cells], ignore_index=True)
M = M.dropna(subset=["prediction", "actual"])
tags = {a.name: dict(a.tags) for a in build_gcls2022_arms(yobj_column("INDPRO"))}
M["CV"] = M["arm"].map(lambda a: tags[a]["CV"])
M["X"] = M["arm"].map(lambda a: tags[a]["X"])

CV_ARMS = ["AR,BIC", "AR,AIC", "AR,POOS", "AR,KF", "ARDI,BIC", "ARDI,AIC", "ARDI,POOS", "ARDI,KF"]
m = M[M["arm"].isin(CV_ARMS)].copy()
# target-variance-normalised pseudo-R2 (paper Eq r2_eq), x100 for published scale
den = m.groupby(["target", "horizon"])["actual"].transform(lambda s: ((s - s.mean()) ** 2).mean())
m["R2"] = 100.0 * (1.0 - (m["actual"] - m["prediction"]) ** 2 / den)
m["CV_KF"] = (m["CV"] == "kfold").astype(float)
m["CV_POOS"] = (m["CV"] == "poos").astype(float)
m["CV_AIC"] = (m["CV"] == "aic").astype(float)
m["_grp"] = m["target"].astype(str) + "|" + m["horizon"].astype(str) + "|" + m["date"].astype(str)


def within_dk(df, cols, ycol="R2", tcol="date"):
    """FE = _grp absorbed by within-demeaning; DK SE clustered on tcol (Bartlett)."""
    g = df.groupby("_grp")
    yd = (df[ycol] - g[ycol].transform("mean")).to_numpy(float)
    Xd = np.column_stack([(df[c] - g[c].transform("mean")).to_numpy(float) for c in cols])
    XtX = Xd.T @ Xd
    XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ (Xd.T @ yd)
    u = yd - Xd @ beta
    # Driscoll-Kraay: aggregate scores by cluster time, Bartlett HAC over ordered times
    sc = Xd * u[:, None]
    tmp = pd.DataFrame(sc, columns=cols)
    tmp["_t"] = df[tcol].to_numpy()
    H = tmp.groupby("_t").sum().sort_index().to_numpy(float)  # T x k
    T = H.shape[0]
    L = min(int(math.floor(4 * (T / 100) ** (2 / 9))), T - 1)
    S = H.T @ H
    for l in range(1, L + 1):
        w = 1 - l / (L + 1)
        Gl = H[l:].T @ H[:-l]
        S += w * (Gl + Gl.T)
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(V))
    from scipy import stats
    t = beta / se
    p = 2 * (1 - stats.norm.cdf(np.abs(t)))
    return beta, se, t, p, len(df), T, L


COLS = ["CV_KF", "CV_POOS", "CV_AIC"]
PUB = {  # published Table 2 (coef, se) for All / Data-rich / Data-poor
    "All": {"CV_KF": (-0.0380, 0.800), "CV_POOS": (-1.351, 0.800), "CV_AIC": (-0.509, 0.800)},
    "Data-rich(ARDI)": {"CV_KF": (-0.314, 0.711), "CV_POOS": (-1.440, 0.711), "CV_AIC": (-0.648, 0.711)},
    "Data-poor(AR)": {"CV_KF": (0.237, 0.411), "CV_POOS": (-1.262, 0.411), "CV_AIC": (-0.370, 0.411)},
}
for label, sub in [("All", m), ("Data-rich(ARDI)", m[m["X"] == 1]), ("Data-poor(AR)", m[m["X"] == 0])]:
    beta, se, t, p, n, T, L = within_dk(sub, COLS)
    print(f"\n=== {label}  (n={n}, distinct dates T={T}, DK bandwidth L={L}) ===")
    print(f"{'term':9s} {'mine_coef':>10s} {'mine_DKse':>10s} {'t':>7s} {'p':>7s}   {'pub_coef':>9s} {'pub_se':>7s}")
    for i, c in enumerate(COLS):
        pc, ps = PUB[label][c]
        print(f"{c:9s} {beta[i]:>10.3f} {se[i]:>10.3f} {t[i]:>7.2f} {p[i]:>7.3f}   {pc:>9.3f} {ps:>7.3f}")
print("\nOK g3_manual")
