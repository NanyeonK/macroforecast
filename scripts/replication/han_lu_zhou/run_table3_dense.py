"""B5 / Table 2 — all five forecast-combination designs through macroforecast.

Serial baseline already measured on the `current` design: 20.7 min, R2_OS 0.611%.
This run uses n_jobs to cut wall-clock (objective 3) and re-checks that the
parallel path returns the same number (objective 4).
"""
import sys, time, json, warnings
sys.path.insert(0, ".")
import numpy as np, pandas as pd
import macroforecast as mf
from macroforecast.pipeline import (Arm, EvalSpec, CombinationContender,
                                    pipeline_spec, run_pipeline, TargetSpec)

N_JOBS = int(sys.argv[1]) if len(sys.argv) > 1 else 4
DESIGNS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["dense6", "dense12", "dense24", "dense36"]

panel = pd.read_parquet("/tmp/hlz_panel.parquet")
NAMES = ["DP", "DY", "EP", "DE", "BM", "NTIS", "TBL", "LTY", "LTR", "TMS",
         "DFY", "DFR", "INFL", "SVAR"]
econ, y, idx = panel[NAMES], panel["mkt_excess"], panel.index

def lag_block(k):
    return {f"{v}_L{k-1}": econ[v].shift(k - 1) for v in NAMES}

def ma_block(L):
    return {f"{v}_MA{L}": econ[v].rolling(L, min_periods=L).mean() for v in NAMES}

# Table 3 (dense trend combinations): current values, then MA ladders 1..L.
# Established empirically -- each archived FC_Trend_Linear column was matched to its
# design by cross-checking a faithful re-implementation against the stored path.
SIGNALS = {
    "current": lambda: dict(lag_block(1)),
    "dense6":  lambda: {k: v for L in range(1, 7) for k, v in ma_block(L).items()},
    "dense12": lambda: {k: v for L in range(1, 13) for k, v in ma_block(L).items()},
    "dense24": lambda: {k: v for L in range(1, 25) for k, v in ma_block(L).items()},
    "dense36": lambda: {k: v for L in range(1, 37) for k, v in ma_block(L).items()},
}
# authors' Table 2 values, derived from their archived 696-month forecast paths
TARGET = {"current": 0.599, "dense6": 0.699, "dense12": 0.805, "dense24": 0.796, "dense36": 0.730}
LABEL = {"current": "Current value (14)", "dense6": "MA 1..6 (84)",
         "dense12": "MA 1..12 (168)", "dense24": "MA 1..24 (336)", "dense36": "MA 1..36 (504)"}
REF_COL = {"current": 0, "dense6": 1, "dense12": 2, "dense24": 3, "dense36": 4}

def parquet_safe(df):
    """dict/None-valued object columns break pyarrow (empty struct); JSON them."""
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object and out[c].map(lambda v: isinstance(v, (dict, list))).any():
            out[c] = out[c].map(lambda v: json.dumps(v, default=str) if isinstance(v, (dict, list)) else v)
    return out

results = {}
for design in DESIGNS:
    sig = SIGNALS[design]()
    cols = list(sig)
    frame = pd.concat([y.rename("y"), pd.DataFrame(sig, index=idx)], axis=1)
    bundle = mf.data.custom_dataset(frame, transform_codes={c: 1 for c in frame.columns})
    test_start = idx[457]
    w = mf.window.from_cutoffs(test_start=test_start, horizon=1, embargo=0,
                               val_method="expanding", val_min_train_size=24)
    arms = [Arm("HA", model="hist_mean",
                features=mf.feature_engineering.feature_spec(target="y", target_lags=(1,)),
                is_benchmark=True)]
    arms += [Arm(c, model="ols",
                 features=mf.feature_engineering.feature_spec(
                     target="y", predictors=[c], lags=0, target_lags=None),
                 nested_in_benchmark=True) for c in cols]
    spec = pipeline_spec(
        data=bundle, targets=[TargetSpec("y", transform="level")], horizons=[1], window=w,
        arms=arms,
        evaluation=EvalSpec(benchmark="HA", metrics=("relative_mse", "r2_oos"),
                            tests=("dm", "cw"), test_options={"dm": {"hac_lags": 4}}),
        combinations=[CombinationContender(name=f"COMB_{design}", method="mean", over=tuple(cols))],
        save_models=False, n_jobs=N_JOBS,
    )
    print(f"\n########## design={design}  signals={len(cols)}  n_jobs={N_JOBS} ##########", flush=True)
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rep = run_pipeline(spec)
    mins = (time.time() - t0) / 60
    acc = rep.accuracy
    row = acc[acc["contender"] == f"COMB_{design}"]
    mine = float(row["r2_oos"].iloc[0]) * 100.0 if len(row) else float("nan")
    results[design] = (mine, mins)
    print(f"R2_OS mine = {mine:.3f}%   authors = {TARGET[design]:.3f}%   "
          f"Δ = {mine - TARGET[design]:+.3f}pp   ({mins:.1f} min)", flush=True)
    sig_t = getattr(rep, "significance", None)
    if isinstance(sig_t, pd.DataFrame) and len(sig_t):
        sub = sig_t[sig_t["contender"] == f"COMB_{design}"]
        if len(sub):
            print("  tests:", sub.drop(columns=[c for c in ("target", "horizon") if c in sub]).to_string(index=False), flush=True)
    try:
        import numpy as _np
        _t = _np.load("/tmp/hlz_target_table2.npz", allow_pickle=True)
        _ref = _t["fc"][:, REF_COL[design]]
        _c = rep.forecasts[rep.forecasts["contender"] == f"COMB_{design}"].sort_values("date")
        _p = _c["prediction"].to_numpy(float)
        _n = min(len(_p), len(_ref))
        _d = _p[-_n:] - _ref[-_n:]
        print(f"  PATH PARITY vs authors' archived path: n={_n} max|Δ|={_np.nanmax(_np.abs(_d)):.3e} "
              f"corr={_np.corrcoef(_p[-_n:], _ref[-_n:])[0,1]:.8f}", flush=True)
    except Exception as _e:
        print("  (path parity unavailable)", _e, flush=True)
    parquet_safe(rep.forecasts).to_parquet(f"/tmp/hlz_fc_{design}.parquet")
    acc.to_parquet(f"/tmp/hlz_acc_{design}.parquet")

print("\n########## TABLE 2 — mine vs authors ##########")
print("| design | signals | mine R2_OS % | authors % | Δ pp | min |")
print("|---|---|---|---|---|---|")
for d in DESIGNS:
    if d in results:
        m, mins = results[d]
        print(f"| {LABEL[d]} | {len(SIGNALS[d]())} | {m:.3f} | {TARGET[d]:.3f} | {m - TARGET[d]:+.3f} | {mins:.1f} |")
print("ALL_DESIGNS_DONE")
