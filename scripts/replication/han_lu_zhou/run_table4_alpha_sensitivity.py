"""B5 / Table 4 stage 2b — penalty sensitivity for the shrinkage columns.

Why this exists. Running the appendix's rule as written -- "the strength of penalties are
determined recursively through threefold cross-validations" (IA §IA1.1), expressed as
`mf.model_selection.grid({"alpha": ...}, validation_splitter=mf.recursive_threefold())` --
the cross-validation chose the LARGEST penalty in the grid at 696 of 696 origins. Measured
`alpha_max` (the smallest penalty that zeroes every coefficient) is 1.098 at the first
origin, so that selection is the null model: the forecast collapses to the training mean,
which is the benchmark, and `R²_OS` lands at ~0 against the paper's 0.86-1.18.

The appendix does not pin the selection down further, so the penalty is NOT tuned toward
the paper's numbers. Instead the sensitivity is reported: a few fixed penalties spanning
the useful range, side by side with the CV result, so a reader can see how much of the
published figure rides on how the penalty is chosen.
"""
import sys, time, json, warnings
sys.path.insert(0, ".")
import numpy as np, pandas as pd
import macroforecast as mf
from macroforecast.pipeline import (Arm, EvalSpec, CombinationContender,
                                    pipeline_spec, run_pipeline, TargetSpec)

N_JOBS = int(sys.argv[1]) if len(sys.argv) > 1 else 4
DESIGNS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["L1", "L6", "L12"]
ALPHAS = [float(a) for a in (sys.argv[3] if len(sys.argv) > 3 else "0.001,0.01,0.05").split(",")]

panel = pd.read_parquet("/tmp/hlz_panel.parquet")
NAMES = ["DP", "DY", "EP", "DE", "BM", "NTIS", "TBL", "LTY", "LTR", "TMS",
         "DFY", "DFR", "INFL", "SVAR"]
econ, y, idx = panel[NAMES], panel["mkt_excess"], panel.index
LMAP = {"L1": 1, "L6": 6, "L12": 12, "L24": 24, "L36": 36}
PAPER = {"L1": (-0.31, -0.35), "L6": (0.86, 0.59), "L12": (1.11, 1.18)}
ARCH_COL = {"L1": 0, "L6": 1, "L12": 2}
import scipy.io as sio
_d = sio.loadmat("/home/nanyeon99/data/han_lu_zhou/extracted/Codes/Data/Data_trend.mat")
ARCHIVE = {"LASSO": np.asarray(_d["FC_LASSO"], float),
           "ENet": np.asarray(_d["FC_ENET"], float)}


def parquet_safe(df):
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object and out[c].map(lambda v: isinstance(v, (dict, list))).any():
            out[c] = out[c].map(lambda v: json.dumps(v, default=str) if isinstance(v, (dict, list)) else v)
    return out


test_start = idx[456]
results = {}
for design in DESIGNS:
    L = LMAP[design]
    cols, lag_groups = {}, {}
    for l in range(1, L + 1):
        g = []
        for v in NAMES:
            name = f"{v}_MA{l}"
            cols[name] = econ[v].rolling(l, min_periods=l).mean()
            g.append(name)
        lag_groups[l] = g
    frame = pd.concat([y.rename("y"), pd.DataFrame(cols, index=idx)], axis=1)
    bundle = mf.data.custom_dataset(frame, transform_codes={c: 1 for c in frame.columns})
    w = mf.window.from_cutoffs(test_start=test_start, horizon=1, embargo=0,
                               val_method="expanding", val_min_train_size=24)

    for alpha in ALPHAS:
        arms = [Arm("HA", model="hist_mean",
                    features=mf.feature_engineering.feature_spec(target="y", target_lags=(1,)),
                    is_benchmark=True)]
        combos, tags = [], {}
        for label, model, extra in (("LASSO", "lasso", {}),
                                    ("ENet", "elastic_net", {"l1_ratio": 0.5})):
            names = []
            for l in range(1, L + 1):
                arm = f"{label}_lag{l}"
                names.append(arm)
                arms.append(Arm(
                    arm, model=model,
                    params={"alpha": alpha, "standardize": True, **extra},
                    features=mf.feature_engineering.feature_spec(
                        target="y", predictors=lag_groups[l], lags=0, target_lags=None),
                ))
            combos.append(CombinationContender(name=f"COMB_{label}", method="mean",
                                               over=tuple(names)))
            tags[label] = f"COMB_{label}"

        spec = pipeline_spec(
            data=bundle, targets=[TargetSpec("y", transform="level")], horizons=[1], window=w,
            arms=arms,
            evaluation=EvalSpec(benchmark="HA", metrics=("relative_mse", "r2_oos"),
                                tests=("dm", "cw"), test_options={"dm": {"hac_lags": 4}}),
            combinations=combos, save_models=False, n_jobs=N_JOBS,
        )
        print(f"\n########## design={design} (L={L})  alpha={alpha}  arms={len(arms)} ##########",
              flush=True)
        t0 = time.time()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rep = run_pipeline(spec)
        except Exception as exc:
            print(f"  RUN FAILED: {type(exc).__name__}: {str(exc)[:160]}", flush=True)
            continue
        mins = (time.time() - t0) / 60
        acc = rep.accuracy
        for label, cname in tags.items():
            row = acc[acc["contender"] == cname]
            mine = float(row["r2_oos"].iloc[0]) * 100.0 if len(row) else float("nan")
            pap = PAPER[design][["LASSO", "ENet"].index(label)]
            ref = ARCHIVE[label][:, ARCH_COL[design]]
            c = rep.forecasts[rep.forecasts["contender"] == cname].sort_values("date")
            p = c["prediction"].to_numpy(float)
            n = min(len(p), len(ref))
            md = float(np.nanmax(np.abs(p[-n:] - ref[-n:]))) if n else float("nan")
            results[(design, alpha, label)] = (mine, pap, md)
            print(f"  {label:6s} alpha={alpha:<7g} mine={mine:7.3f}%  paper={pap:6.2f}%  "
                  f"Δ={mine-pap:+7.3f}pp   path max|Δ|={md:.3e}  ({mins:.1f} min)", flush=True)

print("\n########## TABLE 4 stage 2b — penalty sensitivity ##########")
print("| design | alpha | model | mine % | paper % | Δ pp | path max|Δ| |")
print("|---|---|---|---|---|---|---|")
for (design, alpha, label), (mine, pap, md) in results.items():
    print(f"| {design} | {alpha:g} | {label} | {mine:.3f} | {pap:.2f} | {mine-pap:+.3f} | {md:.2e} |")
print("T4_ALPHA_SENS_DONE")
