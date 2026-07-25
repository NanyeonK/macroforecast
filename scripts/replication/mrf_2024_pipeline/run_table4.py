"""(3) Full Table-4 replication for ONE target (arg: GDP|INF|IR|SPREAD|HOUST).
Reuses UR's engineered predictor/factor/MAF block (target-independent), swaps only the
8 target y-lags + the direct target. Runs all 14 Table-4 models vs AR(4), seed=42."""
import sys, time, math, json, traceback
sys.path.insert(0, ".")
import numpy as np, pandas as pd, macroforecast as mf
from macroforecast.models import get_model
fe = mf.feature_engineering
TGT = sys.argv[1] if len(sys.argv) > 1 else "INF"
t_all = time.time()

# target series builder + paper Table-4 rows (14 models x 5 horizons)
TARGETS = {   # (fred_series, transform) ; transform: dlog|diff|level|spread
    "GDP":    ("GDPC1", "dlog"),
    "UR":     ("UNRATE", "dlog"),
    "INF":    ("CPIAUCSL", "dlog"),     # [ASSUMPTION] inflation = dlog CPI
    "IR":     ("GS1", "diff"),          # first-diff of the 1yr rate
    "SPREAD": (("GS10", "FEDFUNDS"), "spread"),  # level of 10yr - FFR
    "HOUST":  ("HOUST", "dlog"),        # [ASSUMPTION]
}
PAPER = {
 "UR":{"FA-AR":[0.83,0.80,0.88,1.18,1.25],"LASSO-MAF":[0.99,0.98,0.96,0.98,0.98],"Ridge-MAF":[0.99,0.92,0.94,0.98,1.01],"RF":[1.00,0.98,0.96,1.01,1.01],"RF-MAF":[0.85,0.85,0.87,0.94,0.95],"AR+RF":[0.84,0.84,0.84,0.90,0.95],"Tiny RF":[1.24,1.15,1.37,1.60,1.57],"FA-ARRF":[0.72,0.76,0.79,0.89,1.01],"ARRF":[0.90,0.90,0.87,0.95,0.98],"Tiny ARRF":[1.00,0.96,0.92,0.97,0.98],"VARRF":[1.24,0.89,0.91,0.95,1.04],"SETAR":[1.18,1.03,1.02,1.07,1.09],"STAR":[1.10,0.97,1.01,1.04,1.06],"TV-AR":[1.00,0.99,1.34,1.14,1.11]},
 "GDP":{"FA-AR":[1.02,0.96,1.03,1.36,1.37],"LASSO-MAF":[0.96,0.98,0.98,0.98,1.00],"Ridge-MAF":[0.89,0.98,0.99,0.98,0.99],"RF":[0.94,0.99,1.00,0.98,0.99],"RF-MAF":[0.86,0.91,0.98,1.00,0.99],"AR+RF":[0.89,0.93,0.99,1.00,0.96],"Tiny RF":[1.03,1.01,1.03,1.08,1.15],"FA-ARRF":[0.86,0.97,0.97,1.01,1.06],"ARRF":[0.93,0.94,0.95,0.97,1.00],"Tiny ARRF":[1.04,1.03,0.98,0.98,1.01],"VARRF":[1.20,0.99,0.89,1.00,1.04],"SETAR":[1.01,0.97,0.97,0.98,1.00],"STAR":[1.03,0.98,0.96,0.95,0.97],"TV-AR":[0.99,1.03,0.96,0.98,1.00]},
 "INF":{"FA-AR":[1.01,1.01,1.08,1.32,1.21],"LASSO-MAF":[0.93,0.96,0.92,0.96,0.98],"Ridge-MAF":[0.95,0.92,0.87,0.90,1.27],"RF":[0.98,0.92,0.94,1.01,1.44],"RF-MAF":[0.88,0.82,0.85,0.88,0.88],"AR+RF":[1.23,1.00,0.96,1.00,0.94],"Tiny RF":[0.90,0.88,0.86,0.86,0.88],"FA-ARRF":[0.94,0.94,0.89,0.91,0.91],"ARRF":[0.89,0.86,0.91,0.85,0.92],"Tiny ARRF":[0.87,0.87,0.95,0.92,0.94],"VARRF":[0.96,0.91,0.87,0.87,0.91],"SETAR":[1.05,0.86,0.90,0.94,0.96],"STAR":[1.00,0.86,0.87,0.89,0.92],"TV-AR":[0.93,0.89,0.91,0.98,0.98]},
 "IR":{"FA-AR":[1.85,1.49,0.96,1.87,1.58],"LASSO-MAF":[1.02,0.96,1.00,0.95,0.98],"Ridge-MAF":[1.55,1.01,1.03,0.99,1.02],"RF":[1.17,1.00,1.03,1.00,1.03],"RF-MAF":[1.11,0.93,1.04,0.93,0.96],"AR+RF":[0.97,0.98,0.99,0.93,0.96],"Tiny RF":[0.99,1.29,1.39,1.23,1.20],"FA-ARRF":[1.29,1.22,0.99,0.98,1.04],"ARRF":[0.94,0.93,0.97,0.95,0.96],"Tiny ARRF":[0.92,0.92,1.12,1.07,1.10],"VARRF":[1.43,1.10,0.97,1.12,0.98],"SETAR":[1.39,1.15,1.08,1.19,1.25],"STAR":[1.20,1.11,1.07,1.14,1.20],"TV-AR":[0.97,1.04,1.09,1.06,1.06]},
 "SPREAD":{"FA-AR":[1.28,1.13,0.86,1.51,1.28],"LASSO-MAF":[2.16,1.20,0.95,0.80,0.76],"Ridge-MAF":[0.93,0.77,1.01,1.13,0.96],"RF":[0.91,0.66,0.81,0.98,0.92],"RF-MAF":[0.95,0.78,0.69,0.80,0.83],"AR+RF":[0.79,0.72,0.61,0.80,0.89],"Tiny RF":[0.96,0.93,1.48,1.43,1.36],"FA-ARRF":[1.08,0.80,0.66,0.72,0.82],"ARRF":[0.89,0.78,0.73,0.82,0.88],"Tiny ARRF":[1.06,1.11,1.07,1.05,0.99],"VARRF":[0.77,0.74,0.69,0.74,0.85],"SETAR":[1.51,1.19,1.04,1.03,1.11],"STAR":[1.53,1.20,1.06,1.06,1.14],"TV-AR":[0.98,1.04,1.30,1.19,0.99]},
 "HOUST":{"FA-AR":[1.13,1.13,1.11,1.40,1.04],"LASSO-MAF":[1.04,0.99,0.98,0.96,0.95],"Ridge-MAF":[0.94,0.94,0.97,0.96,0.95],"RF":[0.92,0.95,0.97,0.96,0.95],"RF-MAF":[1.00,1.01,1.01,0.96,0.99],"AR+RF":[1.01,1.02,1.03,1.01,1.02],"Tiny RF":[1.24,1.10,1.12,1.16,1.44],"FA-ARRF":[1.08,1.06,1.02,0.97,0.96],"ARRF":[0.94,1.00,1.00,0.99,0.99],"Tiny ARRF":[0.95,1.02,1.02,1.00,1.01],"VARRF":[1.09,0.99,1.02,0.98,1.00],"SETAR":[1.01,0.94,0.95,0.95,0.95],"STAR":[0.99,0.97,0.96,0.96,0.95],"TV-AR":[1.00,1.01,1.08,0.99,1.03]},
}
try:
    F0 = pd.read_parquet("/tmp/g2_F.parquet"); idx = F0.index
    # predictor block = everything except UR's own y-lags
    PRED_BLOCK = [c for c in map(str, F0.columns) if not c.startswith("URT_lag")]
    b = mf.data.load_fred_qd("2020-01"); P, TC = b.panel, b.metadata["transform_codes"]
    def series(tgt):
        fs, tr = TARGETS[tgt]
        if tr == "dlog": return np.log(pd.to_numeric(P[fs], errors="coerce")).diff()
        if tr == "diff": return pd.to_numeric(P[fs], errors="coerce").diff()
        if tr == "level": return pd.to_numeric(P[fs], errors="coerce")
        if tr == "spread": return (pd.to_numeric(P[fs[0]], errors="coerce") - pd.to_numeric(P[fs[1]], errors="coerce"))
    y = series(TGT).reindex(idx).rename("Y")
    ylags = pd.DataFrame({f"Y_lag{L}": y.shift(L) for L in range(1, 9)}, index=idx)
    F = pd.concat([F0[PRED_BLOCK], ylags], axis=1)
    S_full = [c for c in map(str, F.columns)]
    S_tiny = [f"Y_lag{k}" for k in range(1, 9)] + ["trend"]
    AR_LIT=["Y_lag1","Y_lag2","Y_lag3","Y_lag4"]; AR2=["Y_lag1","Y_lag2"]; LAGS=["Y_lag1","Y_lag2"]
    x_arrf=LAGS; x_fa=["Y_lag1","Y_lag2","pc1_lag1","pc2_lag1"]
    x_var=["Y_lag1","Y_lag2","GDPC1_lag1","GS1_lag1","CPIAUCSL_lag1"]
    FA_COLS=["Y_lag1","Y_lag2","Y_lag3","Y_lag4","pc1_lag1","pc1_lag2","pc2_lag1","pc2_lag2"]
    # RF 8-lag set (target-independent predictor lags)
    W=None
    Xp=pd.DataFrame({c: (np.log(pd.to_numeric(P[c],errors="coerce").where(pd.to_numeric(P[c],errors="coerce")>0)).diff().diff() if int(TC.get(c,1))==6 else (np.log(pd.to_numeric(P[c],errors="coerce").where(pd.to_numeric(P[c],errors="coerce")>0)).diff() if int(TC.get(c,1))==5 else (pd.to_numeric(P[c],errors="coerce").diff() if int(TC.get(c,1))==2 else pd.to_numeric(P[c],errors="coerce")))) for c in P.columns}, index=P.index)
    PREDS=[c for c in Xp.columns if not Xp.loc["1961-07-01":"2002-12-31"][c].isna().any()]
    RF8=pd.DataFrame({f"{c}_L{L}":Xp[c].shift(L) for c in PREDS for L in range(1,9)}, index=Xp.index).reindex(idx)

    mrf=get_model("macro_random_forest"); ols=get_model("ols"); lasso=get_model("lasso"); ridge=get_model("ridge"); rf=get_model("random_forest"); tvp=get_model("tvp_ridge"); setar=get_model("setar"); star=get_model("star")
    MK=dict(B=50,minsize=10,mtry_frac=1/3,min_leaf_frac_of_x=1.0,subsampling_rate=0.75,rw_regul=0.75,block_size=12,ridge_lambda=0.1,HRW=0,resampling_opt=2,trend_push=1,random_state=42,parallelise=True,n_cores=12)
    poos=idx[(idx>=pd.Timestamp("2003-01-01"))&(idx<=pd.Timestamp("2014-12-31"))]; est=list(poos[::8])
    def olsp(cols,tr,blk,frame,tgt):
        A=np.column_stack([np.ones(len(tr)),frame.loc[tr,cols].to_numpy(float)]); beta,*_=np.linalg.lstsq(A,tgt.loc[tr].to_numpy(float),rcond=None)
        return np.column_stack([np.ones(len(blk)),frame.loc[blk,cols].to_numpy(float)])@beta, beta
    def penal(model,tr,blk,tgt,grid):
        k=max(12,int(0.2*len(tr))); trn,val=tr[:-k],tr[-k:]; yv=tgt.loc[val].to_numpy(float); best=(np.inf,grid[len(grid)//2])
        for a in grid:
            try:
                pv=np.asarray(model(F.loc[trn,S_full],tgt.loc[trn],alpha=float(a)).predict(F.loc[val,S_full]),dtype=float).ravel()
                e=math.sqrt(np.mean((pv-yv)**2)); best=(e,a) if e<best[0] else best
            except Exception: pass
        return np.asarray(model(F.loc[tr,S_full],tgt.loc[tr],alpha=float(best[1])).predict(F.loc[blk,S_full]),dtype=float).ravel()
    MODELS=["FA-AR","LASSO-MAF","Ridge-MAF","RF","RF-MAF","AR+RF","Tiny RF","FA-ARRF","ARRF","Tiny ARRF","VARRF","SETAR","STAR","TV-AR"]
    LG=np.logspace(-5,-1,9); RG=np.logspace(-2,3,9)
    def predict_model(nm,tr,blk,tgt):
        yt=tgt.loc[tr].rename("Y")
        if nm=="FA-AR": return olsp(FA_COLS,tr,blk,F,tgt)[0]
        if nm=="LASSO-MAF": return penal(lasso,tr,blk,tgt,LG)
        if nm=="Ridge-MAF": return penal(ridge,tr,blk,tgt,RG)
        if nm=="RF":
            trR=tr[RF8.loc[tr].notna().all(axis=1).to_numpy()]
            return np.asarray(rf(RF8.loc[trR],tgt.loc[trR],random_state=42,n_jobs=8).predict(RF8.loc[blk].fillna(0.0)),dtype=float).ravel()
        if nm=="RF-MAF": return np.asarray(rf(F.loc[tr,S_full],yt,random_state=42,n_jobs=8).predict(F.loc[blk,S_full]),dtype=float).ravel()
        if nm=="AR+RF":
            arp,b2=olsp(AR2,tr,blk,F,tgt); arfit=np.column_stack([np.ones(len(tr)),F.loc[tr,AR2].to_numpy(float)])@b2
            resid=pd.Series(tgt.loc[tr].to_numpy(float)-arfit,index=tr,name="Y")
            rr=np.asarray(rf(F.loc[tr,S_full],resid,random_state=42,n_jobs=8).predict(F.loc[blk,S_full]),dtype=float).ravel()
            return arp+rr
        if nm=="Tiny RF": return np.asarray(rf(F.loc[tr,S_tiny],yt,random_state=42,n_jobs=8).predict(F.loc[blk,S_tiny]),dtype=float).ravel()
        if nm in ("FA-ARRF","ARRF","VARRF","Tiny ARRF"):
            xc={"ARRF":x_arrf,"FA-ARRF":x_fa,"VARRF":x_var,"Tiny ARRF":x_arrf}[nm]; sc=S_tiny if nm=="Tiny ARRF" else S_full
            return np.asarray(mrf(F.loc[tr],yt,x_columns=xc,S_columns=sc,**MK).predict(F.loc[blk]),dtype=float).ravel()
        if nm=="SETAR": return np.asarray(setar(F.loc[tr,LAGS],yt,n_lag=2).predict(F.loc[blk,LAGS]),dtype=float).ravel()
        if nm=="STAR": return np.asarray(star(F.loc[tr,LAGS],yt,n_lag=2).predict(F.loc[blk,LAGS]),dtype=float).ravel()
        if nm=="TV-AR": return np.asarray(tvp(F.loc[tr,LAGS],yt,random_state=42).predict(F.loc[blk,LAGS]),dtype=float).ravel()
    res={}
    for h in (1,2,4,6,8):
        tgt=y.shift(-h); arp={}; mp={nm:{} for nm in MODELS}
        for i,tau in enumerate(est):
            tr=idx[(idx>=pd.Timestamp("1961-07-01"))&(idx<tau)]; ok=tgt.reindex(tr).notna()&F.loc[tr].notna().all(axis=1); tr=tr[ok.to_numpy()]
            nxt=est[i+1] if i+1<len(est) else pd.Timestamp("2015-01-01"); blk=poos[(poos>=tau)&(poos<nxt)]; blk=blk[F.loc[blk].notna().all(axis=1)&tgt.reindex(blk).notna()]
            if len(tr)<40 or len(blk)==0: continue
            for d,v in zip(blk, olsp(AR_LIT,tr,blk,F,tgt)[0]): arp[d]=float(v)
            for nm in MODELS:
                try:
                    for d,v in zip(blk, predict_model(nm,tr,blk,tgt)): mp[nm][d]=float(v)
                except Exception as e:
                    if h==1 and i==0: print("  %s err: %s"%(nm,repr(e)[:90]),flush=True)
        for nm in MODELS:
            common=sorted(set(arp)&set(mp[nm]))
            if not common: continue
            act=tgt.reindex(common).to_numpy(float)
            r=math.sqrt(np.mean((np.array([mp[nm][d] for d in common])-act)**2))/math.sqrt(np.mean((np.array([arp[d] for d in common])-act)**2))
            res[(nm,h)]=r
        print("  h=%d done"%h,flush=True)
    print("RES_START %s"%TGT)
    print("| model | h1 | h2 | h4 | h6 | h8 | mean|Δ| |"); print("|---|---|---|---|---|---|---|")
    P4=PAPER[TGT]; allmd=[]
    for nm in MODELS:
        cells=[]; diffs=[]
        for j,h in enumerate((1,2,4,6,8)):
            r=res.get((nm,h)); p=P4[nm][j]
            if r is None: cells.append("--"); continue
            cells.append("%.3f/%.2f"%(r,p)); diffs.append(abs(r-p))
        md=float(np.mean(diffs)) if diffs else float("nan"); allmd.append(md)
        print("| %s | %s | %.3f |"%(nm," | ".join(cells),md))
    print("TARGET %s overall mean|Δ| = %.3f"%(TGT, float(np.nanmean(allmd))))
    json.dump({f"{k[0]}|h{k[1]}":v for k,v in res.items()}, open(f"/tmp/g2_{TGT}.json","w"), indent=1)
    print("RES_END total=%.1f min"%((time.time()-t_all)/60))
except Exception:
    traceback.print_exc()
