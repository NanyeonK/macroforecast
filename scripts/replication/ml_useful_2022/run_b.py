"""ML-Useful (b): valid (leak-free) bounded comparison with the freeze/vary contract.

Preprocessing is SPEC-LEVEL (shared across arms) and refits on the retrain cadence
(update="on_retrain", retrain_every=24 -> the paper's 24-month refresh) rather than
every origin. t-codes are window-invariant; outliers/EM/standardise refit on cadence.
This removes both the per-origin and (largely) the per-arm redundancy that made the
naive leak-free run ~24x too slow.
"""
from __future__ import annotations

import os
import sys
import time
import warnings

warnings.simplefilter("ignore")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import pandas as pd  # noqa: E402
import macroforecast as mf  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from registry import ml_useful_arms  # noqa: E402
from treatment import treatment_effects  # noqa: E402
from macroforecast.pipeline import EvalSpec, TargetSpec, pipeline_spec, run_pipeline  # noqa: E402


def main(data_csv: str, *, horizons=(1, 12), n_origins=24) -> None:
    raw = mf.data.load_fred_md(local_source=data_csv)
    idx = raw.panel.index
    predictors = tuple(c for c in raw.panel.columns
                       if c not in {"INDPRO", "UNRATE", "CPIAUCSL", "HOUST", "T10YFFM"})
    pp = mf.preprocessing.preprocess_spec(transform="official", impute="em_factor",
                                          outliers="iqr", outlier_action="flag_as_nan", standardize="none")
    policy = mf.window.stage_policy("origin_available", update="on_retrain")  # cadence, not every origin
    keep = {"AR", "ARDI", "RFAR", "RFARDI"}
    arms = [a for a in ml_useful_arms("INDPRO", predictors, subset="core") if a.name in keep]
    targets = [TargetSpec(t, transform="average_value", policy="direct_average") for t in ("INDPRO", "UNRATE")]

    panels = []
    for h in horizons:
        window = mf.window.from_cutoffs(
            estimation_start=idx[300], test_start=idx[-(n_origins + h)], test_end=idx[-1],
            mode="expanding", horizon=h, embargo=0, retrain_every=24,
            val_method="last_block", val_size=24,
        )
        t0 = time.time()
        rep = run_pipeline(pipeline_spec(
            data=raw, targets=targets, horizons=[h], window=window, arms=arms,
            evaluation=EvalSpec(benchmark="AR", tests=["dm"]),
            preprocessing=pp, preprocessing_policy=policy, save_models=False,
        ))
        panels.append(rep.forecasts)
        print(f"=== h={h} relative RMSE (vs AR)  [{time.time()-t0:.0f}s] ===")
        print(rep.accuracy.sort_values(["target", "relative_mse"])
              [["target", "contender", "relative_mse", "n_common"]].to_string(index=False))

    te = treatment_effects(pd.concat(panels, ignore_index=True), arms, features=("nonlinear",))
    a = te.get("alpha", {}).get("nonlinear", {})
    print("\n=== pooled nonlinearity treatment effect (Eq.11, leak-free) ===")
    print(f"alpha[nonlinear] = {a.get('estimate'):.5f}  t = {a.get('t_value'):.3f}  "
          f"p = {a.get('p_value'):.4f}  n = {te.get('n_obs')}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/ml_useful_data/2018-01.csv")
