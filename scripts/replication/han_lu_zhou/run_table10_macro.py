"""HLZ Table 10 -- forecasting quarterly inflation and GDP growth with macro trends.

This is the paper's only genuinely MACRO exhibit, and the sharpest test of the
BYOD path: the data is a 304-quarter panel out of the authors' archive, not
FRED-MD/QD, so every series is a custom target and a custom predictor.

Specification, read off the authors' own code (Forecasts_in_and_out_of_sample.m
:795-925) rather than inferred from the printed row labels -- which matters,
because the labels are misleading. `y_t + {MA2,...,MA8}` is NOT the consecutive
ladder MA2..MA8: `L_max = [1, 2, 4, 8, 12]`, so the ladder DOUBLES, and each row
is the equal-weighted mean of the single-MA forecasts up to that level:

    row 1 (benchmark)  mean over k in {1}
    row 2  y + {MA2}   mean over k in {1, 2}
    row 3  y + {MA4}   mean over k in {1, 2, 4}
    row 4  y + {MA8}   mean over k in {1, 2, 4, 8}
    row 5  y + {MA12}  mean over k in {1, 2, 4, 8, 12}

Each member is a BIVARIATE regression y_{s+1} ~ 1 + MA_k(y_s) + z_s. R2_OS is
against row 1 -- the current-value model with the same control -- not against a
historical average. Columns 2-6 use one control at a time; column 7 averages over
all (k, control) pairs. DM with Newey-West, five lags.

One estimation, six evaluations: the arms are fit once and `evaluate(master, spec)`
is then called per column with a different `benchmark`, since only the benchmark
changes between columns.
"""
import sys, time, json, warnings
sys.path.insert(0, ".")
import numpy as np, pandas as pd, scipy.io as sio
import macroforecast as mf
from macroforecast.pipeline import (Arm, EvalSpec, CombinationContender,
                                    pipeline_spec, run_pipeline, evaluate, TargetSpec)

N_JOBS = int(sys.argv[1]) if len(sys.argv) > 1 else 6
ARCH = "/home/nanyeon99/data/han_lu_zhou/extracted/Codes/Data"
m = sio.loadmat(f"{ARCH}/Data_macro_indicator.mat")
T = m["ECON"].shape[0]
idx = pd.period_range("1947Q1", periods=T, freq="Q").to_timestamp(how="end").normalize()
CONTROLS = ["RF", "DP", "UNE", "INF", "GDP"]          # printed column order
TARGETS = {"INF": 0, "GDP": 1}                         # ECON columns
LADDER = [1, 2, 4, 8, 12]
ROWS = [("y + {MA2}", [1, 2]), ("y + {MA2,MA4}", [1, 2, 4]),
        ("y + {MA2..MA8}", [1, 2, 4, 8]), ("y + {MA2..MA12}", [1, 2, 4, 8, 12])]
R = (1964 - 1947 + 1) * 4                              # 72 -> first OOS target 1965Q1
PRINTED = {   # (RF, D/P, UNRATE, Inflation, GDP growth, Comb)
    ("INF", 0): (0.71, 1.68, 2.23, 1.04, 1.64, 1.45),
    ("INF", 1): (5.61, 8.85, 10.10, 7.94, 9.15, 8.11),
    ("INF", 2): (4.51, 8.12, 10.34, 7.34, 9.34, 8.32),
    ("INF", 3): (3.99, 8.62, 12.14, 8.39, 10.18, 9.12),
    ("GDP", 0): (6.39, 6.63, 7.46, 6.12, 3.14, 6.30),
    ("GDP", 1): (10.37, 10.93, 12.40, 10.37, 1.92, 10.10),
    ("GDP", 2): (11.47, 12.49, 15.02, 12.33, 1.49, 11.68),
    ("GDP", 3): (12.08, 13.43, 16.46, 13.73, 1.59, 12.82),
}

def parquet_safe(df):
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object and out[c].map(lambda v: isinstance(v, (dict, list))).any():
            out[c] = out[c].map(lambda v: json.dumps(v, default=str) if isinstance(v, (dict, list)) else v)
    return out

out_rows = {}
for tname, tcol in TARGETS.items():
    y = pd.Series(m["ECON"][:, tcol], index=idx, name="y")
    cols = {"y": y}
    for k in LADDER:
        cols[f"MA{k}"] = y.rolling(k, min_periods=k).mean()
    for c in CONTROLS:
        cols[f"z_{c}"] = pd.Series(m[c].ravel(), index=idx)
    frame = pd.DataFrame(cols, index=idx)
    bundle = mf.data.custom_dataset(frame, transform_codes={c: 1 for c in frame.columns})
    w = mf.window.from_cutoffs(test_start=idx[R - 1], horizon=1, embargo=0,
                               val_method="expanding", val_min_train_size=8)
    arms = []
    for k in LADDER:
        for c in CONTROLS:
            arms.append(Arm(f"k{k}_{c}", model="ols",
                            features=mf.feature_engineering.feature_spec(
                                target="y", predictors=[f"MA{k}", f"z_{c}"],
                                lags=0, target_lags=None)))
    combos = []
    for label, ks in ROWS:                              # per-control columns
        for c in CONTROLS:
            combos.append(CombinationContender(
                name=f"COL_{c}_{len(ks)}", method="mean",
                over=tuple(f"k{k}_{c}" for k in ks)))
    combos.append(CombinationContender(name="COMB_bench", method="mean",
                                       over=tuple(f"k1_{c}" for c in CONTROLS)))
    for label, ks in ROWS:                              # the Comb column
        combos.append(CombinationContender(
            name=f"COMB_{len(ks)}", method="mean",
            over=tuple(f"k{k}_{c}" for k in ks for c in CONTROLS)))
    spec = pipeline_spec(
        data=bundle, targets=[TargetSpec("y", transform="level")], horizons=[1], window=w,
        arms=arms,
        evaluation=EvalSpec(benchmark=f"k1_{CONTROLS[0]}", metrics=("r2_oos",),
                            tests=("dm",), test_options={"dm": {"hac_lags": 5}}),
        combinations=combos, save_models=False, n_jobs=N_JOBS,
    )
    print(f"\n########## target={tname}  arms={len(arms)}  n_jobs={N_JOBS} ##########", flush=True)
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rep = run_pipeline(spec)
    print(f"  fit: {(time.time()-t0)/60:.1f} min, {rep.forecasts['date'].nunique()} OOS quarters "
          f"({rep.forecasts['date'].min().date()} .. {rep.forecasts['date'].max().date()})", flush=True)
    parquet_safe(rep.forecasts).to_parquet(f"/tmp/hlz_t10_{tname}.parquet")

    master = rep.forecasts[~rep.forecasts["contender"].astype(str)
                           .str.startswith(("COL_", "COMB_"))].copy()
    master["window"] = None
    for c in CONTROLS + ["Comb"]:                       # one evaluation per column
        bench = f"k1_{c}" if c != "Comb" else "COMB_bench"
        ev = EvalSpec(benchmark=bench, metrics=("r2_oos",), tests=("dm",),
                      test_options={"dm": {"hac_lags": 5}})
        sp = pipeline_spec(data=bundle, targets=[TargetSpec("y", transform="level")],
                           horizons=[1], window=w, arms=arms, evaluation=ev,
                           combinations=combos, save_models=False)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = evaluate(master, sp)
        except Exception as exc:
            print(f"  !! column {c}: evaluate failed with benchmark={bench}: "
                  f"{type(exc).__name__}: {exc}", flush=True)
            continue
        acc, sg = res["accuracy"], res["significance"]
        for j, (label, ks) in enumerate(ROWS):
            name = f"COL_{c}_{len(ks)}" if c != "Comb" else f"COMB_{len(ks)}"
            r = acc[acc["contender"] == name]
            s = sg[sg["contender"] == name] if len(sg) else sg
            out_rows[(tname, j, c)] = (
                float(r["r2_oos"].iloc[0]) * 100 if len(r) else np.nan,
                float(s["dm_p"].iloc[0]) if len(s) and "dm_p" in s else np.nan,
            )

HDR = ["RF", "D/P", "UNRATE", "Inflation", "GDP growth", "Comb"]
KEY = CONTROLS + ["Comb"]
print("\n########## TABLE 10 -- mine vs printed (R2_OS %, vs the y_t + z_t benchmark) ##########")
print("| panel | design | " + " | ".join(f"{h} mine | printed | Δ" for h in HDR) + " |")
print("|" + "---|" * 20)
deltas = []
for tname, panel in (("INF", "A: Inflation"), ("GDP", "B: GDP growth")):
    for j, (label, _) in enumerate(ROWS):
        cells = []
        for i, c in enumerate(KEY):
            mine = out_rows.get((tname, j, c), (np.nan, np.nan))[0]
            p = PRINTED[(tname, j)][i]
            cells.append(f"{mine:.2f} | {p:.2f} | {mine - p:+.2f}")
            if np.isfinite(mine):
                deltas.append(abs(mine - p))
        print(f"| {panel} | {label} | " + " | ".join(cells) + " |")
d = np.array(deltas)
if len(d):
    print(f"\n{len(d)} cells: max|Δ| = {d.max():.3f}pp  mean|Δ| = {d.mean():.3f}pp  "
          f"within 0.05 = {(d<=0.05).sum()}  within 0.25 = {(d<=0.25).sum()}  within 1.0 = {(d<=1.0).sum()}")
json.dump({f"{k[0]}|{k[1]}|{k[2]}": v for k, v in out_rows.items()},
          open("/tmp/hlz_t10.json", "w"), indent=1)
print("TABLE10_DONE")
