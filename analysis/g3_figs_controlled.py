"""P2: CONTROLLED e_eq2 per-feature treatment regressions (not the crude pooled +24).
Each feature f is estimated within M_f (models differing ~only in f), outcome =
target-variance pseudo-R2 x100, FE phi_{t,v,h} (within-transform), Driscoll-Kraay SE."""
import sys, glob, math
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from scipy import stats
from scripts.replication.gcls_2022_pipeline.registry import build_gcls2022_arms
from scripts.replication.gcls_2022_pipeline.data import yobj_column

cells = (glob.glob("runs/gcls_b4_stage1/_result_store_indpro/cells/*.parquet")
         + glob.glob("runs/gcls_b4_stage1/_result_store_g2rest/cells/*.parquet"))
M = pd.concat([pd.read_parquet(f, columns=["date", "horizon", "target", "prediction", "actual", "arm"])
               for f in cells], ignore_index=True).dropna(subset=["prediction", "actual"])
tags = {a.name: dict(a.tags) for a in build_gcls2022_arms(yobj_column("INDPRO"))}
den = M.groupby(["target", "horizon"])["actual"].transform(lambda s: ((s - s.mean()) ** 2).mean())
M["R2"] = 100.0 * (1.0 - (M["actual"] - M["prediction"]) ** 2 / den)
M["_grp"] = M["target"].astype(str) + "|" + M["horizon"].astype(str) + "|" + M["date"].astype(str)


def within_dk(df, cols):
    df = df.dropna(subset=cols + ["R2"])
    g = df.groupby("_grp")
    yd = (df["R2"] - g["R2"].transform("mean")).to_numpy(float)
    Xd = np.column_stack([(df[c] - g[c].transform("mean")).to_numpy(float) for c in cols])
    beta = np.linalg.lstsq(Xd, yd, rcond=None)[0]
    u = yd - Xd @ beta
    sc = Xd * u[:, None]
    tmp = pd.DataFrame(sc, columns=cols); tmp["_t"] = df["date"].to_numpy()
    H = tmp.groupby("_t").sum().sort_index().to_numpy(float)
    T = H.shape[0]; L = min(int(math.floor(4 * (T / 100) ** (2 / 9))), T - 1)
    S = H.T @ H
    for l in range(1, L + 1):
        Gl = H[l:].T @ H[:-l]; S += (1 - l / (L + 1)) * (Gl + Gl.T)
    XtXi = np.linalg.pinv(Xd.T @ Xd); V = XtXi @ S @ XtXi
    se = np.sqrt(np.diag(V)); t = beta / se; p = 2 * (1 - stats.norm.cdf(np.abs(t)))
    return beta, se, t, p, len(df)


def sub(arms):
    m = M[M["arm"].isin(arms)].copy()
    for k in ("X", "NL", "SH", "CV", "LF"):
        m[k] = m["arm"].map(lambda a, kk=k: tags[a][kk])
    return m


RR = ["RRAR,POOS", "RRAR,KF", "RRARDI,POOS", "RRARDI,KF"]
RF = ["RFAR,POOS", "RFAR,KF", "RFARDI,POOS", "RFARDI,KF"]
KRR = ["KRRAR,POOS", "KRRAR,KF", "KRRARDI,POOS", "KRRARDI,KF"]
B = [f"B{i},{m},{cv}" for i in (1, 2, 3) for m in ("ridge", "EN", "lasso") for cv in ("POOS", "KF")]
ARDI = ["ARDI,BIC", "ARDI,AIC", "ARDI,POOS", "ARDI,KF"]
SVRARDI = ["SVR-ARDI,Lin,POOS", "SVR-ARDI,Lin,KF", "SVR-ARDI,RBF,POOS", "SVR-ARDI,RBF,KF"]
CV8 = ["AR,BIC", "AR,AIC", "AR,POOS", "AR,KF", "ARDI,BIC", "ARDI,AIC", "ARDI,POOS", "ARDI,KF"]

print("CONTROLLED per-feature treatment effects (pseudo-R2 x100, FE phi_{t,v,h}, DK SE)\n")


def report(name, m, cols):
    b, se, t, p, n = within_dk(m, cols)
    for i, c in enumerate(cols):
        star = "***" if p[i] < .001 else "**" if p[i] < .01 else "*" if p[i] < .05 else ""
        print(f"  {name:22s} {c:8s} coef={b[i]:+7.3f}  DKse={se[i]:6.3f}  t={t[i]:+6.2f}  p={p[i]:.3g} {star}   (n={n})")


# NL: linear-ridge base vs RF/KRR (same features), controlling for X
mnl = sub(RR + RF + KRR); report("NL [RR|RF|KRR]", mnl, ["NL", "X"])
# SH: ARDI (no linear shrinkage) vs B1/B2/B3 shrinkage
msh = sub(ARDI + B); msh["SH_d"] = (msh["SH"] != "none").astype(float); report("SH [ARDI|B1/2/3]", msh, ["SH_d"])
# CV: the 8 AR/ARDI arms (expected collapse due to the bug)
mcv = sub(CV8)
mcv["CV_KF"] = (mcv["CV"] == "kfold").astype(float); mcv["CV_POOS"] = (mcv["CV"] == "poos").astype(float); mcv["CV_AIC"] = (mcv["CV"] == "aic").astype(float)
report("CV [AR/ARDI x BIC..]", mcv, ["CV_KF", "CV_POOS", "CV_AIC"])
# LF: SVR-ARDI (eps loss) vs KRR-ARDI (quad loss, both nonlinear kernel) -> isolates loss
mlf = sub(SVRARDI + KRR[2:]); mlf["LF_d"] = (mlf["LF"] == "eps").astype(float); report("LF [SVR-ARDI|KRRARDI]", mlf, ["LF_d"])
# X: big data over all 46 arms
mx = M.copy(); mx["X"] = mx["arm"].map(lambda a: tags[a]["X"]); report("X [all 46]", mx, ["X"])
print("\nOK figs_controlled")
