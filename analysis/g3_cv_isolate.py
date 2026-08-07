"""Isolate the POOS-CV / K-fold selection: score every grid candidate on a real AR and
ARDI feature matrix, to see whether the validation genuinely prefers the smallest model
(legitimate parsimony) or the scoring is degenerate (bug)."""
import sys, warnings
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
import macroforecast as mf
from macroforecast.model_selection import SearchSpec, validation_splitter, select_params
from macroforecast.model_selection.search import select_by_information_criterion

warnings.simplefilter("ignore")
# build a real preprocessed panel up to a late origin (1995) -> AR target-lag features
from scripts.replication.gcls_2022_pipeline.data import augmented_bundle, yobj_column
bundle, predictors = augmented_bundle()
res = mf.preprocessing.reprocess(bundle, transform="official", impute="em_factor",
                                 outliers="iqr", outlier_action="flag_as_nan", standardize="none")
panel = res.panel.loc[:"1995-01-01"]
T = yobj_column("INDPRO")
y = panel[T].dropna()
# AR design: y_{t} on its own lags (build lag matrix, direct h=1 target)
LAGS = 12
X = pd.concat({f"{T}_lag{k}": y.shift(k) for k in range(0, LAGS + 1)}, axis=1)
target = y.shift(-1)  # h=1
df = pd.concat([X, target.rename("__y__")], axis=1).dropna()
Xm, ym = df.drop(columns="__y__"), df["__y__"]
print("AR design:", Xm.shape, "sample:", list(Xm.columns)[:4])

grid = {"n_lag": (1, 3, 6, 12)}
for cv, splitter in [("poos", validation_splitter("poos", validation_ratio=0.25)),
                     ("kfold", validation_splitter("random_kfold", n_splits=5))]:
    spec = SearchSpec(method="grid", param_grid=grid, validation_splitter=splitter)
    r = select_params("ar", Xm, ym, search=spec, metric="mse")
    print(f"\n[AR {cv}] best={r.best_params}")
    # dump per-candidate scores if available
    md = r.to_metadata() if hasattr(r, "to_metadata") else {}
    cand = md.get("candidates") or md.get("scores") or getattr(r, "scores", None)
    print("  metadata keys:", list(md.keys())[:12])
    for k in ("candidates", "scores", "all_scores", "cv_results", "trials"):
        if k in md:
            print(f"  {k}:", md[k])
# IC(bic)
rb = select_by_information_criterion("ar", Xm, ym, criterion="bic",
                                     search=SearchSpec(method="information_criterion", param_grid=grid, criterion="bic"))
print("\n[AR bic] best:", rb.best_params)
print("\nOK isolate")
