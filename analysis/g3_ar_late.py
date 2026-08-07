"""Quick: do AR,POOS / AR,KF now select != AR,BIC at a LATE origin (2009-2011)?
AR arms only (fast, no far/EM-heavy), fixed package."""
import sys, warnings
sys.path.insert(0, ".")
import macroforecast as mf
from macroforecast.pipeline import EvalSpec, pipeline_spec, run_pipeline, selection_history
from scripts.replication.gcls_2022_pipeline.data import augmented_bundle, gcls_targets, yobj_column
from scripts.replication.gcls_2022_pipeline.registry import build_gcls2022_arms
warnings.simplefilter("ignore")

bundle, predictors = augmented_bundle()
T = yobj_column("INDPRO")
tgt = [t for t in gcls_targets() if t.name == T]
arms = [a for a in build_gcls2022_arms(T, predictors) if a.name in ("AR,BIC", "AR,AIC", "AR,POOS", "AR,KF")]
window = mf.window.from_cutoffs(estimation_start="1960-01", test_start="2009-06", test_end="2011-06",
                                mode="expanding", retrain_every=1, retune_every=24, retune_on_retrain=False)
pp = mf.preprocessing.preprocess_spec(transform="official", impute="em_factor", outliers="iqr",
                                      outlier_action="flag_as_nan", standardize="none")
ppp = mf.window.stage_policy("origin_available", update=24)
ckpt = "runs/gcls_b4_stage1/_arlate_ckpt"
spec = pipeline_spec(data=bundle, targets=tgt, horizons=[1], window=window, arms=arms,
                     evaluation=EvalSpec(benchmark="AR,BIC", tests=()), preprocessing=pp,
                     preprocessing_policy=ppp, selection_history=True, checkpoint_dir=ckpt, n_jobs=1, seed=42)
run_pipeline(spec)
h = selection_history(ckpt)
nlag = h[h["name"] == "n_lag"].copy()
nlag["a"] = nlag["arm"].astype(str).str.replace("INDPRO__", "", regex=False)
piv = nlag.pivot_table(index="origin_pos", columns="a", values="value", aggfunc="first")
print(piv.head(6).to_string())
for c in piv.columns:
    print(f"  {c:10s}: n_lag values = {sorted(piv[c].dropna().unique().tolist())}")
print("OK ar_late")
