"""Match the pipeline exactly: ar(direct=True) + POOS/KF/BIC over the grid, dumping the
per-candidate validation score, at an early (1980) and late (2010) sample."""
import sys, warnings
sys.path.insert(0, ".")
import numpy as np, pandas as pd
import macroforecast as mf
from macroforecast.model_selection import SearchSpec, validation_splitter, select_params
from macroforecast.model_selection.search import select_by_information_criterion
from scripts.replication.gcls_2022_pipeline.data import augmented_bundle, yobj_column
warnings.simplefilter("ignore")

bundle, predictors = augmented_bundle()
res = mf.preprocessing.reprocess(bundle, transform="official", impute="em_factor",
                                 outliers="iqr", outlier_action="flag_as_nan", standardize="none")
T = yobj_column("INDPRO")


def design(cut):
    y = res.panel.loc[:cut, T].dropna()
    X = pd.concat({f"{T}_lag{k}": y.shift(k) for k in range(0, 13)}, axis=1)
    tgt = y.shift(-1)
    df = pd.concat([X, tgt.rename("__y__")], axis=1).dropna()
    return df.drop(columns="__y__"), df["__y__"]


grid = {"n_lag": (1, 3, 6, 12)}
for cut in ("1980-01-01", "2010-01-01"):
    Xm, ym = design(cut)
    print(f"\n########## sample up to {cut}  (n={len(Xm)}) ##########")
    for cv, sp in [("poos", validation_splitter("poos", validation_ratio=0.25)),
                   ("kfold", validation_splitter("random_kfold", n_splits=5))]:
        for direct in (True, False):
            spec = SearchSpec(method="grid", param_grid=grid, validation_splitter=sp)
            r = select_params("ar", Xm, ym, search=spec, metric="mse",
                              fixed_params={"direct": direct})
            md = r.to_metadata()
            inner = md.get("metadata", {})
            trials = inner.get("trials") or inner.get("candidate_scores") or inner.get("scores")
            print(f"  [{cv} direct={direct}] best={r.best_params} best_score={md.get('best_score'):.3e}"
                  + (f"  trials={trials}" if trials else ""))
    for direct in (True, False):
        rb = select_by_information_criterion("ar", Xm, ym, criterion="bic", fixed_params={"direct": direct},
            search=SearchSpec(method="information_criterion", param_grid=grid, criterion="bic"))
        print(f"  [bic direct={direct}] best={rb.best_params}")
print("\nOK isolate2")
