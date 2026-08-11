"""Instrument the pipeline's select_params call for AR,POOS to see WHY it collapses to
n_lag=1: log the search's validation_splitter, the splits passed, X shape, and result."""
import sys, warnings
sys.path.insert(0, ".")
import macroforecast as mf
import macroforecast.forecasting.policies.base as B
from macroforecast.pipeline import EvalSpec, pipeline_spec, run_pipeline
from scripts.replication.gcls_2022_pipeline.data import augmented_bundle, gcls_targets, yobj_column
from scripts.replication.gcls_2022_pipeline.registry import build_gcls2022_arms
warnings.simplefilter("ignore")

_orig = B.select_params
LOG = []
def traced(model, X, y=None, search=None, *, splits=None, **kw):
    r = _orig(model, X, y, search=search, splits=splits, **kw)
    sp = getattr(search, "validation_splitter", None)
    LOG.append({
        "method": getattr(search, "method", None),
        "splitter": getattr(sp, "method", None),
        "splits_passed": (None if splits is None else f"{len(splits)} splits"),
        "n_splits_kw": kw.get("allow_non_temporal_splits"),
        "X_shape": getattr(X, "shape", None),
        "grid": getattr(search, "param_grid", None),
        "best": dict(r.best_params),
    })
    return r
B.select_params = traced

bundle, predictors = augmented_bundle()
T = yobj_column("INDPRO")
tgt = [t for t in gcls_targets() if t.name == T]
arms = [a for a in build_gcls2022_arms(T, predictors) if a.name in ("AR,BIC", "AR,POOS", "AR,KF")]
# late window so POOS should pick a bigger model (isolation: n_lag~12 at 2010)
window = mf.window.from_cutoffs(estimation_start="1960-01", test_start="2009-06", test_end="2009-07",
                                mode="expanding", retrain_every=1, retune_every=24, retune_on_retrain=False)
pp = mf.preprocessing.preprocess_spec(transform="official", impute="em_factor", outliers="iqr",
                                      outlier_action="flag_as_nan", standardize="none")
ppp = mf.window.stage_policy("origin_available", update=24)
spec = pipeline_spec(data=bundle, targets=tgt, horizons=[1], window=window, arms=arms,
                     evaluation=EvalSpec(benchmark="AR,BIC", tests=()), preprocessing=pp,
                     preprocessing_policy=ppp, n_jobs=1, seed=42)
run_pipeline(spec)
print(f"select_params called {len(LOG)} times:")
for e in LOG:
    print(" ", e)
print("OK instrument")
