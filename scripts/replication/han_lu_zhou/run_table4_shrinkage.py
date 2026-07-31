"""B5 / Table 4, stage 2 — the shrinkage columns (LASSO, ENet).

Structure from the Internet Appendix (Eq. IA4/IA5), not guessed:
  the 14 x L trend predictors are split into **L groups by MA lag** -- group l holds the
  same-lag moving average of all 14 variables -- LASSO (or elastic net) selects within each
  group and forecasts, giving L forecasts, which are pooled by a simple average. The penalty
  is determined "recursively through threefold cross-validations", which the package exposes
  directly as `mf.recursive_threefold()`.

Note this is the mirror image of the factor columns: PCR/PLS/S-PCR group **by variable**
(14 groups of L columns), the shrinkage columns group **by lag** (L groups of 14 columns).

`[ASSUMPTION]` The appendix does not say whether predictors are standardized before the
penalty. They must be here: within a group the 14 Welch-Goyal variables differ in scale by
orders of magnitude (SVAR ~1e-4 against TBL ~3), and macroforecast's own guard refuses an
unstandardized penalized fit on such a set. `standardize=True` throughout.
"""
import sys, time, json, warnings
sys.path.insert(0, ".")
import numpy as np, pandas as pd
import macroforecast as mf
from macroforecast.pipeline import (Arm, EvalSpec, CombinationContender,
                                    pipeline_spec, run_pipeline, TargetSpec)

N_JOBS = int(sys.argv[1]) if len(sys.argv) > 1 else 4
DESIGNS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["L1", "L6", "L12"]

panel = pd.read_parquet("/tmp/hlz_panel.parquet")
NAMES = ["DP", "DY", "EP", "DE", "BM", "NTIS", "TBL", "LTY", "LTR", "TMS",
         "DFY", "DFR", "INFL", "SVAR"]
econ, y, idx = panel[NAMES], panel["mkt_excess"], panel.index
LMAP = {"L1": 1, "L6": 6, "L12": 12, "L24": 24, "L36": 36}

# printed Table 4 of the published paper (RAPS 16(2), p. 257)
PAPER = {"L1": (-0.31, -0.35), "L6": (0.86, 0.59), "L12": (1.11, 1.18),
         "L24": (0.72, 0.86), "L36": (0.57, 0.68)}          # (LASSO, ENet)
ARCH_COL = {"L1": 0, "L6": 1, "L12": 2, "L24": 3, "L36": 4}
import scipy.io as sio
_d = sio.loadmat("/home/nanyeon99/data/han_lu_zhou/extracted/Codes/Data/Data_trend.mat")
ARCHIVE = {"LASSO": np.asarray(_d["FC_LASSO"], float),
           "ENet": np.asarray(_d["FC_ENET"], float)}

ALPHAS = [float(a) for a in np.logspace(-4, 0, 13)]


def parquet_safe(df):
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object and out[c].map(lambda v: isinstance(v, (dict, list))).any():
            out[c] = out[c].map(lambda v: json.dumps(v, default=str) if isinstance(v, (dict, list)) else v)
    return out


test_start = idx[456]      # first ORIGIN -> first target 1965-01, 696 months
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
        lag_groups[l] = g          # group l = the 14 variables at MA lag l
    frame = pd.concat([y.rename("y"), pd.DataFrame(cols, index=idx)], axis=1)
    bundle = mf.data.custom_dataset(frame, transform_codes={c: 1 for c in frame.columns})
    w = mf.window.from_cutoffs(test_start=test_start, horizon=1, embargo=0,
                               val_method="expanding", val_min_train_size=24)

    arms = [Arm("HA", model="hist_mean",
                features=mf.feature_engineering.feature_spec(target="y", target_lags=(1,)),
                is_benchmark=True)]
    combos, tags = [], {}
    for label, model, grid in (
        ("LASSO", "lasso", {"alpha": ALPHAS}),
        ("ENet", "elastic_net", {"alpha": ALPHAS, "l1_ratio": [0.5]}),
    ):
        names = []
        for l in range(1, L + 1):
            arm = f"{label}_lag{l}"
            names.append(arm)
            arms.append(Arm(
                arm, model=model, params={"standardize": True},
                features=mf.feature_engineering.feature_spec(
                    target="y", predictors=lag_groups[l], lags=0, target_lags=None),
                # the paper's recursive threefold CV, re-run at every origin
                model_selection=mf.model_selection.grid(
                    grid, validation_splitter=mf.recursive_threefold()),
            ))
        combos.append(CombinationContender(name=f"COMB_{label}", method="mean", over=tuple(names)))
        tags[label] = f"COMB_{label}"

    spec = pipeline_spec(
        data=bundle, targets=[TargetSpec("y", transform="level")], horizons=[1], window=w,
        arms=arms,
        evaluation=EvalSpec(benchmark="HA", metrics=("relative_mse", "r2_oos"),
                            tests=("dm", "cw"), test_options={"dm": {"hac_lags": 4}}),
        combinations=combos, save_models=False, n_jobs=N_JOBS,
    )
    print(f"\n########## design={design} (L={L})  arms={len(arms)}  n_jobs={N_JOBS} ##########",
          flush=True)
    t0 = time.time()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rep = run_pipeline(spec)
    except Exception as exc:
        print(f"  RUN FAILED: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
        continue
    mins = (time.time() - t0) / 60
    acc = rep.accuracy
    for k, (label, cname) in enumerate(tags.items()):
        row = acc[acc["contender"] == cname]
        mine = float(row["r2_oos"].iloc[0]) * 100.0 if len(row) else float("nan")
        pap = PAPER[design][["LASSO", "ENet"].index(label)]
        ref = ARCHIVE[label][:, ARCH_COL[design]]
        c = rep.forecasts[rep.forecasts["contender"] == cname].sort_values("date")
        p = c["prediction"].to_numpy(float)
        n = min(len(p), len(ref))
        md = float(np.nanmax(np.abs(p[-n:] - ref[-n:]))) if n else float("nan")
        results[(design, label)] = (mine, pap, md)
        print(f"  {label:6s}  mine={mine:7.3f}%  paper={pap:6.2f}%  Δ={mine-pap:+7.3f}pp   "
              f"path max|Δ|={md:.3e}  ({mins:.1f} min)", flush=True)
    parquet_safe(rep.forecasts).to_parquet(f"/tmp/hlz_t4s_{design}.parquet")

print("\n########## TABLE 4 stage 2 — shrinkage columns ##########")
print("| design | model | mine % | paper % | Δ pp | path max|Δ| |")
print("|---|---|---|---|---|---|")
for (design, label), (mine, pap, md) in results.items():
    print(f"| {design} | {label} | {mine:.3f} | {pap:.2f} | {mine-pap:+.3f} | {md:.2e} |")
print("T4_STAGE2_DONE")
