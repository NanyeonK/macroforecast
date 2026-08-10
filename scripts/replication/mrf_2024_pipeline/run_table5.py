"""(5) Full Table-5 (Monthly) replication for ONE target (arg: IP|UR|INF|SPREAD|HOUST).
Builds monthly S_t once (target-independent block, cached to /tmp) then swaps the 12 y-lags
+ h-period AVERAGE-growth target per run. Runs all 11 Table-5 models vs AR(12) [lead
benchmark per A.6 text] AND vs AR(4) [GAP variant per Table-5 caption], seed=42.
See docs/replication/mrf_2024_table5_monthly.md for the full write-up."""
import sys, os, time, math, json, traceback
sys.path.insert(0, ".")
import numpy as np, pandas as pd, macroforecast as mf
from macroforecast.models import get_model
fe = mf.feature_engineering
TGT = sys.argv[1] if len(sys.argv) > 1 else "UR"
HSET = tuple(int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "1,3,9,12,24").split(","))
CORES = int(sys.argv[3]) if len(sys.argv) > 3 else 8
t_all = time.time()

TARGETS = {   # (fred_series, transform); mirrors run_table4.py's TARGETS dict; IP replaces GDP
    "IP":     ("INDPRO", "dlog"),
    "UR":     ("UNRATE", "dlog"),
    "INF":    ("CPIAUCSL", "dlog"),          # [ASSUMPTION] same as quarterly INF (dlog CPI)
    "SPREAD": (("GS10", "FEDFUNDS"), "spread"),  # same definition as quarterly Table 4 SPREAD
    "HOUST":  ("HOUST", "dlog"),             # [ASSUMPTION] inherited from quarterly runner
}

# Paper Table 5 (arXiv 2006.12724 / JAE 2024), column order:
MCOLS = ["AR4", "AO-12", "AO-h", "FA-AR", "RF", "RF-MAF", "AR+RF", "ARRF", "FA-ARRF", "Tiny ARRF", "VARRF"]
PAPER = {
 "IP": {1:[1.00,1.11,1.14,0.96,1.03,0.94,0.97,0.99,0.96,1.02,1.02],
        3:[1.02,1.17,1.02,0.99,1.12,0.98,0.96,1.03,1.01,1.02,1.08],
        9:[1.01,1.04,1.03,1.06,1.02,1.06,1.02,1.04,1.10,1.09,1.03],
        12:[1.01,1.00,1.00,1.05,0.99,0.97,0.91,0.97,1.05,1.13,0.96],
        24:[1.00,0.84,0.84,1.17,0.92,0.86,0.86,0.88,0.95,1.11,0.89]},
 "UR": {1:[1.01,1.03,1.09,0.95,0.97,0.87,0.95,0.91,0.90,0.98,0.94],
        3:[1.00,1.10,1.05,0.86,1.05,0.81,0.92,0.89,0.82,1.03,0.89],
        9:[0.99,1.11,1.10,0.92,1.02,0.96,0.91,0.97,0.98,1.16,0.97],
        12:[0.99,1.07,1.07,0.96,0.97,0.96,0.91,0.99,0.94,1.17,0.96],
        24:[1.02,1.02,1.03,1.06,0.91,0.84,0.81,0.91,0.97,1.28,0.87]},
 "SPREAD": {1:[0.99,2.88,1.23,1.21,3.52,1.07,0.91,0.99,0.98,0.96,0.93],
        3:[1.01,1.68,1.07,1.25,1.69,0.82,0.81,1.06,0.85,1.00,0.88],
        9:[1.01,1.36,1.27,1.06,0.94,0.73,0.72,0.70,0.62,1.07,0.67],
        12:[1.02,1.28,1.28,1.05,0.80,0.66,0.60,0.68,0.65,1.07,0.64],
        24:[1.03,1.34,1.34,0.96,0.80,0.70,0.71,0.69,0.63,0.90,0.70]},
 "INF": {1:[1.02,1.11,1.18,0.99,1.07,1.06,1.01,0.95,0.96,0.95,0.93],
        3:[1.04,1.02,1.24,1.04,0.93,0.88,1.05,0.90,0.88,0.90,0.88],
        9:[1.07,0.92,1.01,1.16,0.86,0.78,1.15,0.72,0.82,0.73,0.76],
        12:[1.09,0.91,0.91,1.21,0.88,0.79,1.15,0.73,0.67,0.67,0.70],
        24:[1.04,0.90,0.86,1.35,1.00,1.12,1.12,0.71,0.69,0.55,0.73]},
 "HOUST": {1:[1.00,1.10,1.35,1.07,1.08,1.02,1.00,1.01,1.02,1.02,1.01],
        3:[0.96,1.06,1.34,1.15,1.03,1.07,1.03,1.04,1.03,1.01,1.04],
        9:[0.98,1.05,1.12,1.35,0.98,1.02,1.01,1.02,1.14,1.03,1.03],
        12:[0.98,1.05,1.05,1.32,0.95,1.00,1.01,1.00,1.12,1.11,1.03],
        24:[0.95,1.09,1.07,1.17,0.87,0.94,0.95,1.00,1.15,1.23,1.06]},
}

CACHE_PRED = "/tmp/table5_Fpred.parquet"
CACHE_RF12 = "/tmp/table5_RF12.parquet"

try:
    b = mf.data.load_fred_md("2020-01")
    P, TC = b.panel, b.metadata["transform_codes"]
    print("loaded FRED-MD: panel=%s range=%s..%s" % (P.shape, P.index.min(), P.index.max()), flush=True)

    def tcode_transform(col):
        s = pd.to_numeric(P[col], errors="coerce")
        tc = int(TC.get(col, 1))
        if tc == 6: return np.log(s.where(s > 0)).diff().diff()
        if tc == 5: return np.log(s.where(s > 0)).diff()
        if tc == 4: return np.log(s.where(s > 0))
        if tc == 2: return s.diff()
        return s  # tcode 1 (correct: level) and tcode 3/7 (inherited gap from run_table4.py; falls to level)

    Xp = pd.DataFrame({c: tcode_transform(c) for c in P.columns}, index=P.index)
    t0 = time.time()
    if os.path.exists(CACHE_PRED) and os.path.exists(CACHE_RF12):
        F_pred = pd.read_parquet(CACHE_PRED)
        RF12 = pd.read_parquet(CACHE_RF12)
        print("loaded cached predictor block: F_pred=%s RF12=%s (%.1fs)" % (F_pred.shape, RF12.shape, time.time() - t0), flush=True)
    else:
        burn_check = Xp.iloc[3:].loc[:"2002-12-31"]
        PREDS = [c for c in Xp.columns if not burn_check[c].isna().any()]
        print("PREDS complete-case through 2002-12: %d/%d series" % (len(PREDS), Xp.shape[1]), flush=True)
        RF12 = pd.DataFrame({f"{c}_L{L}": Xp[c].shift(L) for c in PREDS for L in range(1, 13)}, index=Xp.index)
        steps = [
            fe.lag_step(name="lag2", input="panel", lags=(1, 2), include=True),
            fe.pca_step(name="pc", input="panel", n_components=5, fit_policy="expanding", include=False, random_state=42),
            fe.lag_step(name="pc_lag", input="pc", lags=tuple(range(1, 13)), include=True),
            fe.maf_step(name="maf", input="panel", max_lag=12, n_components=2, fit_policy="expanding", include=True, random_state=42),
        ]
        F_pred = fe.compose_features(Xp[PREDS], steps, drop_missing=False)
        trend = pd.Series(np.arange(len(Xp.index), dtype=float), index=Xp.index, name="trend")
        F_pred = pd.concat([F_pred, trend], axis=1)
        F_pred.to_parquet(CACHE_PRED)
        RF12.to_parquet(CACHE_RF12)
        print("built predictor block: F_pred=%s RF12=%s (%.1fs)" % (F_pred.shape, RF12.shape, time.time() - t0), flush=True)
    idx = F_pred.index

    def series(tgt):
        fs, tr = TARGETS[tgt]
        if tr == "dlog": return np.log(pd.to_numeric(P[fs], errors="coerce")).diff()
        if tr == "spread": return pd.to_numeric(P[fs[0]], errors="coerce") - pd.to_numeric(P[fs[1]], errors="coerce")

    y = series(TGT).reindex(idx).rename("Y")
    ylags = pd.DataFrame({f"Y_lag{L}": y.shift(L) for L in range(1, 13)}, index=idx)
    F = pd.concat([F_pred, ylags], axis=1)
    S_full = [str(c) for c in F.columns]
    S_tiny = [f"Y_lag{k}" for k in range(1, 13)] + ["trend"]
    AR4_LAGS = ["Y_lag1", "Y_lag2", "Y_lag3", "Y_lag4"]
    AR12_LAGS = [f"Y_lag{k}" for k in range(1, 13)]
    LAGS = ["Y_lag1", "Y_lag2"]; AR2 = LAGS
    x_arrf = LAGS; x_fa = ["Y_lag1", "Y_lag2", "pc1_lag1", "pc2_lag1"]
    x_var = ["Y_lag1", "Y_lag2", "INDPRO_lag1", "GS1_lag1", "CPIAUCSL_lag1"]
    FA_COLS = ["Y_lag1", "Y_lag2", "Y_lag3", "Y_lag4", "pc1_lag1", "pc1_lag2", "pc2_lag1", "pc2_lag2"]
    for c in x_var:
        if c not in F.columns:
            print("WARNING x_var column missing: %s" % c, flush=True)

    mrf = get_model("macro_random_forest"); rf = get_model("random_forest")
    MK = dict(B=50, minsize=10, mtry_frac=1/3, min_leaf_frac_of_x=1.0, subsampling_rate=0.75, rw_regul=0.75,
              block_size=12, ridge_lambda=0.1, HRW=0, resampling_opt=2, trend_push=1, random_state=42,
              parallelise=True, n_cores=CORES)
    poos = idx[(idx >= pd.Timestamp("2003-01-01")) & (idx <= pd.Timestamp("2014-12-31"))]
    est = list(poos[::24])
    print("poos=%d months, est points=%s" % (len(poos), [d.date() for d in est]), flush=True)

    def olsp(cols, tr, blk, frame, tgt):
        A = np.column_stack([np.ones(len(tr)), frame.loc[tr, cols].to_numpy(float)])
        beta, *_ = np.linalg.lstsq(A, tgt.loc[tr].to_numpy(float), rcond=None)
        return np.column_stack([np.ones(len(blk)), frame.loc[blk, cols].to_numpy(float)]) @ beta, beta

    MODELS = MCOLS  # ["AR4","AO-12","AO-h","FA-AR","RF","RF-MAF","AR+RF","ARRF","FA-ARRF","Tiny ARRF","VARRF"]

    def predict_model(nm, tr, blk, tgt):
        yt = tgt.loc[tr].rename("Y")
        if nm == "FA-AR": return olsp(FA_COLS, tr, blk, F, tgt)[0]
        if nm == "RF":
            trR = tr[RF12.loc[tr].notna().all(axis=1).to_numpy()]
            return np.asarray(rf(RF12.loc[trR], tgt.loc[trR], random_state=42, n_jobs=CORES).predict(RF12.loc[blk].fillna(0.0)), dtype=float).ravel()
        if nm == "RF-MAF":
            return np.asarray(rf(F.loc[tr, S_full], yt, random_state=42, n_jobs=CORES).predict(F.loc[blk, S_full]), dtype=float).ravel()
        if nm == "AR+RF":
            arp, b2 = olsp(AR2, tr, blk, F, tgt)
            arfit = np.column_stack([np.ones(len(tr)), F.loc[tr, AR2].to_numpy(float)]) @ b2
            resid = pd.Series(tgt.loc[tr].to_numpy(float) - arfit, index=tr, name="Y")
            rr = np.asarray(rf(F.loc[tr, S_full], resid, random_state=42, n_jobs=CORES).predict(F.loc[blk, S_full]), dtype=float).ravel()
            return arp + rr
        if nm in ("FA-ARRF", "ARRF", "VARRF", "Tiny ARRF"):
            xc = {"ARRF": x_arrf, "FA-ARRF": x_fa, "VARRF": x_var, "Tiny ARRF": x_arrf}[nm]
            sc = S_tiny if nm == "Tiny ARRF" else S_full
            return np.asarray(mrf(F.loc[tr], yt, x_columns=xc, S_columns=sc, **MK).predict(F.loc[blk]), dtype=float).ravel()

    def h_avg_target(y, h):
        parts = [y.shift(-hh) for hh in range(1, h + 1)]
        M = pd.concat(parts, axis=1)
        return M.mean(axis=1, skipna=False)

    res = {}  # (nm,h,bench) -> ratio
    for h in HSET:
        t_h0 = time.time()
        tgt = h_avg_target(y, h)
        arp4 = {}; arp12 = {}; mp = {nm: {} for nm in MODELS}
        for i, tau in enumerate(est):
            tr = idx[idx < tau]
            ok = tgt.reindex(tr).notna() & F.loc[tr].notna().all(axis=1)
            tr = tr[ok.to_numpy()]
            nxt = est[i + 1] if i + 1 < len(est) else pd.Timestamp("2015-01-01")
            blk = poos[(poos >= tau) & (poos < nxt)]
            blk = blk[F.loc[blk].notna().all(axis=1).to_numpy() & tgt.reindex(blk).notna().to_numpy()]
            if len(tr) < 60 or len(blk) == 0: continue
            a4, _ = olsp(AR4_LAGS, tr, blk, F, tgt)
            a12, _ = olsp(AR12_LAGS, tr, blk, F, tgt)
            for d, v in zip(blk, a4): arp4[d] = float(v); mp["AR4"][d] = float(v)
            for d, v in zip(blk, a12): arp12[d] = float(v)
            for d in blk:
                w12 = y.loc[:d].iloc[-12:]
                if len(w12) == 12 and w12.notna().all(): mp["AO-12"][d] = float(w12.mean())
                wh = y.loc[:d].iloc[-h:]
                if len(wh) == h and wh.notna().all(): mp["AO-h"][d] = float(wh.mean())
            for nm in MODELS:
                if nm in ("AR4", "AO-12", "AO-h"): continue
                t_m0 = time.time()
                try:
                    for d, v in zip(blk, predict_model(nm, tr, blk, tgt)): mp[nm][d] = float(v)
                except Exception as e:
                    print("  %s h=%d tau=%s err: %s" % (nm, h, tau.date(), repr(e)[:150]), flush=True)
                if nm in ("ARRF", "FA-ARRF", "Tiny ARRF", "VARRF", "RF", "RF-MAF"):
                    print("    %-10s h=%2d tau=%s n_tr=%4d n_blk=%3d  %.1fs" % (nm, h, tau.date(), len(tr), len(blk), time.time() - t_m0), flush=True)
        for nm in MODELS:
            for bench_name, arp in (("AR4", arp4), ("AR12", arp12)):
                common = sorted(set(arp) & set(mp[nm]))
                if not common: continue
                act = tgt.reindex(common).to_numpy(float)
                num = math.sqrt(np.mean((np.array([mp[nm][d] for d in common]) - act) ** 2))
                den = math.sqrt(np.mean((np.array([arp[d] for d in common]) - act) ** 2))
                res[(nm, h, bench_name)] = num / den
        print("  h=%d done in %.1f min" % (h, (time.time() - t_h0) / 60), flush=True)

    print("RES_START %s h=%s" % (TGT, list(HSET)))
    for bench_name in ("AR12", "AR4"):
        print("--- benchmark=%s ---" % bench_name)
        print("| model | " + " | ".join("h%d" % h for h in HSET) + " | mean|Δ| |")
        print("|---|" + "---|" * (len(HSET) + 1))
        allmd = []
        for j, nm in enumerate(MODELS):
            cells = []; diffs = []
            for h in HSET:
                r = res.get((nm, h, bench_name)); p = PAPER[TGT][h][j]
                if r is None: cells.append("--"); continue
                cells.append("%.3f/%.2f" % (r, p)); diffs.append(abs(r - p))
            md = float(np.nanmean(diffs)) if diffs else float("nan"); allmd.append(md)
            print("| %s | %s | %.3f |" % (nm, " | ".join(cells), md))
        print("TARGET %s h=%s benchmark=%s partial mean|Δ| = %.3f" % (TGT, list(HSET), bench_name, float(np.nanmean(allmd))))
    hs_tag = "-".join(str(h) for h in HSET)
    json.dump({f"{k[0]}|h{k[1]}|{k[2]}": v for k, v in res.items()}, open(f"/tmp/table5_{TGT}_h{hs_tag}.json", "w"), indent=1)
    print("RES_END total=%.1f min" % ((time.time() - t_all) / 60))
except Exception:
    traceback.print_exc()
