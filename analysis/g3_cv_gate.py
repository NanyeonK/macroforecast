"""Confirm the gate: are the grid/CV arms hitting _resolve_degraded_selection (empty
selection_splits) instead of select_params? Log should_select, selection_splits size,
and which path each arm takes."""
import sys, warnings
sys.path.insert(0, ".")
import macroforecast as mf
import macroforecast.forecasting.policies.base as B
from macroforecast.pipeline import EvalSpec, pipeline_spec, run_pipeline
from scripts.replication.gcls_2022_pipeline.data import augmented_bundle, gcls_targets, yobj_column
from scripts.replication.gcls_2022_pipeline.registry import build_gcls2022_arms
warnings.simplefilter("ignore")

HITS = {"degraded": 0, "select_params": 0, "ic": 0}
_deg = B._resolve_degraded_selection
def deg(*a, **k):
    HITS["degraded"] += 1
    return _deg(*a, **k)
B._resolve_degraded_selection = deg
_sp = B.select_params
def sp(*a, **k):
    HITS["select_params"] += 1
    return _sp(*a, **k)
B.select_params = sp
_ic = B.select_by_information_criterion
def ic(*a, **k):
    HITS["ic"] += 1
    return _ic(*a, **k)
B.select_by_information_criterion = ic

bundle, predictors = augmented_bundle()
T = yobj_column("INDPRO")
tgt = [t for t in gcls_targets() if t.name == T]

# probe the window's validation split directly
for vm, kw in [("default(last_block)", {}), ("poos", {"val_method": "poos", "val_ratio": 0.25}),
               ("last_block+size", {"val_method": "last_block", "val_size": 60})]:
    w = mf.window.from_cutoffs(estimation_start="1960-01", test_start="2009-06", test_end="2009-07",
                               mode="expanding", retrain_every=1, retune_every=24,
                               retune_on_retrain=False, **kw)
    try:
        vspec = w.val
        print(f"  window val [{vm}]: method={vspec.method} size={getattr(vspec,'size',None)} "
              f"ratio={getattr(vspec,'ratio',None)} n_splits={getattr(vspec,'n_splits',None)}")
    except Exception as e:
        print(f"  window val [{vm}]: {e}")

for vm, kw in [("default", {}), ("poos", {"val_method": "poos", "val_ratio": 0.25})]:
    HITS.update(degraded=0, select_params=0, ic=0)
    arms = [a for a in build_gcls2022_arms(T, predictors) if a.name in ("AR,BIC", "AR,POOS", "AR,KF")]
    window = mf.window.from_cutoffs(estimation_start="1960-01", test_start="2009-06", test_end="2009-08",
                                    mode="expanding", retrain_every=1, retune_every=24,
                                    retune_on_retrain=False, **kw)
    pp = mf.preprocessing.preprocess_spec(transform="official", impute="em_factor", outliers="iqr",
                                          outlier_action="flag_as_nan", standardize="none")
    ppp = mf.window.stage_policy("origin_available", update=24)
    ckpt = f"runs/gcls_b4_stage1/_gate_{vm}_ckpt"
    spec = pipeline_spec(data=bundle, targets=tgt, horizons=[1], window=window, arms=arms,
                         evaluation=EvalSpec(benchmark="AR,BIC", tests=()), preprocessing=pp,
                         preprocessing_policy=ppp, selection_history=True, checkpoint_dir=ckpt, n_jobs=1, seed=42)
    rep = run_pipeline(spec)
    h = mf.pipeline.selection_history(ckpt)
    nlag = h[(h["name"] == "n_lag")]
    sel = {a: sorted(nlag[nlag["arm"].astype(str).str.endswith(a.replace(",", "_"))]["value"].unique().tolist())
           for a in ("AR_BIC", "AR_POOS", "AR_KF")}
    print(f"\n[val={vm}]  hits={HITS}  selected n_lag: {sel}")
print("\nOK gate")
