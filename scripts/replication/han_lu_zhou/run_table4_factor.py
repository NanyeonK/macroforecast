"""B5 / Table 4, stage 1 — the factor columns (PCR, PLS, S-PCR).

Structure taken from the Internet Appendix, not guessed:
  IA1.2.1 PCR      -- split the 14xL trend predictors into 14 groups, one per variable;
                      each group holds that variable's MA_1..MA_L; extract the first J
                      principal components within the group and forecast from them; this
                      yields 14 forecasts, pooled by simple average (Eq. IA7).
  IA1.2.2 PLS      -- same 14 variable groups, one latent factor per group (Eq. IA9/IA10),
                      14 forecasts pooled by simple average.
  IA1.2.3 scaled   -- same groups; scale each MA by its slope on the target, then ordinary
          PCA        PCR on the scaled predictors; 14 forecasts pooled by average.

Internal consistency check on the reading: at L=1 each group holds a single column, so all
three collapse to the univariate predictive regression -- and the paper's current-value
cells for PCR/PLS/S-PCR are all 0.60, equal to Table 2 panel A. They should reproduce that.

`[GAP]` J is never given a numeric value in the paper or the appendix ("J << L"), and the
archive ships only the forecasts, not the estimation code. J is therefore NOT tuned to the
target: every J in J_GRID is run and reported side by side.
"""
import sys, time, json, warnings
sys.path.insert(0, ".")
import numpy as np, pandas as pd
import macroforecast as mf
from macroforecast.pipeline import (Arm, EvalSpec, CombinationContender,
                                    pipeline_spec, run_pipeline, TargetSpec)

N_JOBS = int(sys.argv[1]) if len(sys.argv) > 1 else 4
DESIGNS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["L1", "L6", "L12"]
J_GRID = tuple(int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "1,2,3").split(","))

panel = pd.read_parquet("/tmp/hlz_panel.parquet")
NAMES = ["DP", "DY", "EP", "DE", "BM", "NTIS", "TBL", "LTY", "LTR", "TMS",
         "DFY", "DFR", "INFL", "SVAR"]
econ, y, idx = panel[NAMES], panel["mkt_excess"], panel.index
LMAP = {"L1": 1, "L6": 6, "L12": 12, "L24": 24, "L36": 36}

# printed Table 4 of the published paper (RAPS 16(2), p. 257), verified against the PDF
PAPER = {  # design -> (PCR, PLS, S-PCR)
    "L1":  (0.60, 0.60, 0.60),
    "L6":  (0.74, 0.79, 0.80),
    "L12": (0.91, 0.94, 0.92),
    "L24": (0.93, 0.98, 0.99),
    "L36": (0.79, 0.88, 0.94),
}
ARCH_COL = {"L1": 0, "L6": 1, "L12": 2, "L24": 3, "L36": 4}
_t = np.load("/tmp/hlz_target_table2.npz", allow_pickle=True)  # actual/ha only used here
import scipy.io as sio
_d = sio.loadmat("/home/nanyeon99/data/han_lu_zhou/extracted/Codes/Data/Data_trend.mat")
ARCHIVE = {"PCR": np.asarray(_d["FC_PCA"], float),
           "PLS": np.asarray(_d["FC_PLS"], float),
           "S-PCR": np.asarray(_d["FC_SPCA"], float)}


def parquet_safe(df):
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object and out[c].map(lambda v: isinstance(v, (dict, list))).any():
            out[c] = out[c].map(lambda v: json.dumps(v, default=str) if isinstance(v, (dict, list)) else v)
    return out


test_start = idx[456]   # first ORIGIN -> first target 1965-01, 696 months
results = {}
for design in DESIGNS:
    L = LMAP[design]
    cols = {}
    groups = {}
    for v in NAMES:
        gcols = []
        for l in range(1, L + 1):
            name = f"{v}_MA{l}"
            cols[name] = econ[v].rolling(l, min_periods=l).mean()
            gcols.append(name)
        groups[v] = gcols
    frame = pd.concat([y.rename("y"), pd.DataFrame(cols, index=idx)], axis=1)
    bundle = mf.data.custom_dataset(frame, transform_codes={c: 1 for c in frame.columns})
    w = mf.window.from_cutoffs(test_start=test_start, horizon=1, embargo=0,
                               val_method="expanding", val_min_train_size=24)

    for J in J_GRID:
        if J > L:            # cannot take more components than the group has columns
            continue
        arms = [Arm("HA", model="hist_mean",
                    features=mf.feature_engineering.feature_spec(target="y", target_lags=(1,)),
                    is_benchmark=True)]
        combos, tags = [], {}
        for label, model, params in (("PCR", "pcr", dict(n_components=J)),
                                     ("PLS", "pls", dict(n_components=min(J, 1) if L == 1 else 1)),
                                     ("S-PCR", "scaled_pca", dict(n_components=J))):
            names = []
            for v in NAMES:
                arm = f"{label}_{v}"
                names.append(arm)
                arms.append(Arm(
                    arm, model=model, params=params,
                    features=mf.feature_engineering.feature_spec(
                        target="y", predictors=groups[v], lags=0, target_lags=None),
                    nested_in_benchmark=False))
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
        print(f"\n########## design={design} (L={L})  J={J}  arms={len(arms)}  n_jobs={N_JOBS} ##########",
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
        for k, (label, cname) in enumerate(tags.items()):
            row = acc[acc["contender"] == cname]
            mine = float(row["r2_oos"].iloc[0]) * 100.0 if len(row) else float("nan")
            pap = PAPER[design][["PCR", "PLS", "S-PCR"].index(label)]
            # path parity against the authors' archived forecast path
            ref = ARCHIVE[label][:, ARCH_COL[design]]
            c = rep.forecasts[rep.forecasts["contender"] == cname].sort_values("date")
            p = c["prediction"].to_numpy(float)
            n = min(len(p), len(ref))
            md = float(np.nanmax(np.abs(p[-n:] - ref[-n:]))) if n else float("nan")
            results[(design, J, label)] = (mine, pap, md)
            print(f"  {label:6s} J={J}  mine={mine:7.3f}%  paper={pap:5.2f}%  "
                  f"Δ={mine-pap:+7.3f}pp   path max|Δ|={md:.3e}  ({mins:.1f} min)", flush=True)
        parquet_safe(rep.forecasts).to_parquet(f"/tmp/hlz_t4_{design}_J{J}.parquet")

print("\n########## TABLE 4 stage 1 — factor columns ##########")
print("| design | J | model | mine % | paper % | Δ pp | path max|Δ| |")
print("|---|---|---|---|---|---|---|")
for (design, J, label), (mine, pap, md) in results.items():
    print(f"| {design} | {J} | {label} | {mine:.3f} | {pap:.2f} | {mine-pap:+.3f} | {md:.2e} |")
print("T4_STAGE1_DONE")
