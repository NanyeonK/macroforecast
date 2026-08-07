"""G3 Table 2 (GCLS CVreg_main): the CV-feature treatment regression Eq.(e_eq2).
Outcome R2_{t,h,v,m}=1-e^2/[(1/T)sum(y-ybar)^2] (target-variance normalised), regressors
CV-KF/CV-POOS/AIC (BIC baseline), FE phi_{t,v,h}, DK SE. Over the 8 AR+ARDI models.
"""
import sys, glob
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
import macroforecast as mf
from scripts.replication.gcls_2022_pipeline.registry import build_gcls2022_arms
from scripts.replication.gcls_2022_pipeline.data import yobj_column

# ---- 1. build master from both result stores ----
cells = (glob.glob("runs/gcls_b4_stage1/_result_store_indpro/cells/*.parquet")
         + glob.glob("runs/gcls_b4_stage1/_result_store_g2rest/cells/*.parquet"))
KEEP = ["date", "origin", "horizon", "target", "prediction", "actual", "arm", "contender"]
master = pd.concat([pd.read_parquet(f, columns=KEEP) for f in cells], ignore_index=True)
master = master.dropna(subset=["prediction", "actual"])
print("master rows:", len(master), "arms:", master["arm"].nunique(),
      "targets:", master["target"].nunique(), "horizons:", sorted(master["horizon"].unique()))

# ---- 2. attach A3 tags from the registry ----
tags = {a.name: dict(a.tags) for a in build_gcls2022_arms(yobj_column("INDPRO"))}
for k in ("X", "NL", "SH", "CV", "LF"):
    master[f"tag_{k}"] = master["arm"].map(lambda a: tags[a][k])

# ---- 3. restrict to M_f for the CV feature = the 8 AR/ARDI models (all 4 CV methods) ----
CV_ARMS = ["AR,BIC", "AR,AIC", "AR,POOS", "AR,KF", "ARDI,BIC", "ARDI,AIC", "ARDI,POOS", "ARDI,KF"]
m = master[master["arm"].isin(CV_ARMS)].copy()
print("CV-regression rows (expect ~8*5*5*~450):", len(m))

# ---- 4. target-variance-normalised pseudo-R2 (paper Eq r2_eq) ----
# denom per (target, horizon) = variance of the actual around its mean (mean-forecast MSE)
den = (m.groupby(["target", "horizon"])["actual"]
       .apply(lambda s: float(((s - s.mean()) ** 2).mean())))
m["_den"] = m.set_index(["target", "horizon"]).index.map(den)
m["R2"] = 1.0 - (m["actual"] - m["prediction"]) ** 2 / m["_den"]
m["R2x100"] = 100.0 * m["R2"]

# CV dummies (BIC baseline)
m["CV_KF"] = (m["tag_CV"] == "kfold").astype(float)
m["CV_POOS"] = (m["tag_CV"] == "poos").astype(float)
m["CV_AIC"] = (m["tag_CV"] == "aic").astype(float)

# inject a MEAN pseudo-contender so axis_contribution's r2 denominator == target variance,
# with NaN treatment tags so it is dropped from the regression itself.
def run_axis(sub, label):
    ref = sub[["date", "origin", "horizon", "target", "actual"]].drop_duplicates(
        subset=["origin", "horizon", "target"]).copy()
    ybar = sub.groupby(["target", "horizon"])["actual"].mean()
    ref["prediction"] = ref.set_index(["target", "horizon"]).index.map(ybar)
    ref["arm"] = ref["contender"] = "MEAN"
    for k in ("X", "NL", "SH", "CV", "LF"):
        ref[f"tag_{k}"] = np.nan
    ref["CV_KF"] = ref["CV_POOS"] = ref["CV_AIC"] = np.nan
    big = pd.concat([sub, ref], ignore_index=True)
    res = mf.analysis.axis_contribution(
        big, features=["CV_KF", "CV_POOS", "CV_AIC"], outcome="r2",
        fixed_effects=("target", "horizon", "date"),
        vcov="driscoll_kraay", cluster_by="date", reference="MEAN")
    res = res.copy()
    res["x100"] = res["coef"] * 100
    print(f"\n=== axis_contribution CV regression [{label}] (DK SE) ===")
    print(res[["feature", "level", "coef", "x100", "se", "t", "p", "n"]].to_string(index=False))

for label, sub in [("All", m),
                   ("Data-rich(ARDI)", m[m["tag_X"] == 1]),
                   ("Data-poor(AR)", m[m["tag_X"] == 0])]:
    try:
        run_axis(sub, label)
    except Exception as exc:
        import traceback; traceback.print_exc()
        print(f"[{label}] axis_contribution FAILED: {type(exc).__name__}: {exc}")
print("\nOK g3_table2")
