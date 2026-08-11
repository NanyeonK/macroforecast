"""CV-collapse evidence: dump the SELECTED hyperparameters per origin for the 8 CV arms
(AR/ARDI x BIC/AIC/POOS/KF), INDPRO h=1, via selection_history. Confirm whether POOS-CV
and K-fold literally pick BIC's model each origin, and whether the grid+validation ran."""
import sys, warnings
sys.path.insert(0, ".")
import pandas as pd
import macroforecast as mf
from macroforecast.pipeline import EvalSpec, pipeline_spec, run_pipeline, selection_history
from scripts.replication.gcls_2022_pipeline.data import augmented_bundle, gcls_targets, yobj_column
from scripts.replication.gcls_2022_pipeline.registry import build_gcls2022_arms

bundle, predictors = augmented_bundle()
T = yobj_column("INDPRO")
tgt = [t for t in gcls_targets() if t.name == T]
arms = [a for a in build_gcls2022_arms(T, predictors)
        if a.name in ["AR,BIC", "AR,AIC", "AR,POOS", "AR,KF",
                      "ARDI,BIC", "ARDI,AIC", "ARDI,POOS", "ARDI,KF"]]
# confirm each arm's SearchSpec wiring
for a in arms:
    ms = a.model_selection
    print(f"  {a.name:10s} model={a.model:4s} method={getattr(ms,'method',None)} "
          f"crit={getattr(ms,'criterion',None)} splitter={getattr(getattr(ms,'validation_splitter',None),'method',None)} "
          f"grid={getattr(ms,'param_grid',None)}", flush=True)

window = mf.window.from_cutoffs(estimation_start="1960-01", test_start="1980-01", test_end="1982-12",
                               mode="expanding", retrain_every=1, retune_every=24, retune_on_retrain=False)
pp = mf.preprocessing.preprocess_spec(transform="official", impute="em_factor", outliers="iqr",
                                      outlier_action="flag_as_nan", standardize="none")
ppp = mf.window.stage_policy("origin_available", update=24)
ckpt = "runs/gcls_b4_stage1/_cvdump_ckpt"
spec = pipeline_spec(data=bundle, targets=tgt, horizons=[1], window=window, arms=arms,
                     evaluation=EvalSpec(benchmark="AR,BIC", tests=()), preprocessing=pp,
                     preprocessing_policy=ppp, selection_history=True, checkpoint_dir=ckpt,
                     n_jobs=1, seed=42)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    run_pipeline(spec)

h = selection_history(ckpt)
print("\nselection_history cols:", list(h.columns))
# selected params per arm per origin (params only)
p = h[h["kind"] == "param"].copy()
p["arm_short"] = p["arm"].astype(str).str.replace("INDPRO__", "", regex=False)
# pivot: for each origin, the selected n_lag / n_factors per arm
for name in ("n_lag", "n_factors"):
    sub = p[p["name"] == name]
    if len(sub):
        piv = sub.pivot_table(index="origin_pos", columns="arm_short", values="value", aggfunc="first")
        print(f"\n=== selected {name} per origin (rows=origins) ===")
        print(piv.head(8).to_string())
        print("... distinct value-tuples across origins per arm:")
        for c in piv.columns:
            print(f"    {c:16s}: values={sorted(piv[c].dropna().unique().tolist())}")
# method / score evidence that grid+validation ran (score non-null for grid arms)
print("\n=== method + score presence per arm (did grid/validation run?) ===")
for arm in sorted(p["arm_short"].unique()):
    rows = h[h["arm"].astype(str).str.replace("INDPRO__", "", regex=False) == arm]
    meth = rows["method"].dropna().unique().tolist()
    nscore = int(rows["score"].notna().sum())
    print(f"    {arm:16s}: methods={meth} score_rows={nscore}")
print("\nOK cv_dump")
