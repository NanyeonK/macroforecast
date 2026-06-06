"""ML-Useful (c): full-design leak-free run, incremental + resumable.

5 targets x horizons {1,3,9,12,24} x core arms {AR,ARDI,RFAR,RFARDI}, full POOS
(1980-2017), leak-free (spec-level preprocessing + on_retrain cadence + cross-arm
cache + feature cadence). Writes per-(target,horizon) accuracy and the forecast
panel as each cell finishes (resumable via skip-existing), then the pooled Eq.11
treatment effect. This is a multi-hour compute job (cf. the GCLS INDPRO run ~15h).
"""
from __future__ import annotations

import dataclasses
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import warnings
warnings.simplefilter("ignore")

import pandas as pd  # noqa: E402
import macroforecast as mf  # noqa: E402

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
from registry import ml_useful_arms, ml_useful_targets  # noqa: E402
from treatment import treatment_effects  # noqa: E402
from macroforecast.pipeline import EvalSpec, pipeline_spec, run_pipeline  # noqa: E402

RESULTS = os.path.join(HERE, "results")
HORIZONS = (1, 3, 9, 12, 24)
ARM_KEEP = {"AR", "ARDI", "RFAR", "RFARDI"}


def main(data_csv: str) -> None:
    os.makedirs(RESULTS, exist_ok=True)
    raw = mf.data.load_fred_md(local_source=data_csv)
    idx = raw.panel.index
    predictors = tuple(c for c in raw.panel.columns
                       if c not in {t.name for t in ml_useful_targets()})
    pp = mf.preprocessing.preprocess_spec(transform="official", impute="em_factor",
                                          outliers="iqr", outlier_action="flag_as_nan", standardize="none")
    pol = mf.window.stage_policy("origin_available", update="on_retrain")
    fpol = mf.window.stage_policy("fit_window", update="on_retrain")
    base_arms = [a for a in ml_useful_arms("INDPRO", predictors, subset="core") if a.name in ARM_KEEP]
    arms = [dataclasses.replace(a, feature_policy=fpol) for a in base_arms]

    acc_rows = []
    for tspec in ml_useful_targets():
        for h in HORIZONS:
            tag = f"{tspec.name}_h{h}"
            panel_path = os.path.join(RESULTS, f"forecast_{tag}.parquet")
            if os.path.exists(panel_path):
                print(f"[skip] {tag} (exists)", flush=True)
                continue
            try:
                window = mf.window.from_cutoffs(
                    estimation_start="1960-01", test_start="1980-01", test_end="2017-12",
                    mode="expanding", horizon=h, embargo=0, retrain_every=24,
                    val_method="last_block", val_size=24,
                )
                t0 = time.time()
                rep = run_pipeline(pipeline_spec(
                    data=raw, targets=[tspec], horizons=[h], window=window, arms=arms,
                    evaluation=EvalSpec(benchmark="AR", tests=["dm"]),
                    preprocessing=pp, preprocessing_policy=pol, save_models=False,
                ))
                rep.forecasts.to_parquet(panel_path)
                acc = rep.accuracy.assign(horizon=h)
                acc.to_csv(os.path.join(RESULTS, f"accuracy_{tag}.csv"), index=False)
                acc_rows.append(acc)
                dt = time.time() - t0
                best = acc.sort_values("relative_mse").iloc[0]
                print(f"[done] {tag} {dt:.0f}s  best={best['contender']} relRMSE={best['relative_mse']:.3f}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[fail] {tag}: {type(exc).__name__}: {str(exc)[:100]}", flush=True)

    # pooled treatment effect over all completed cells
    panels = [pd.read_parquet(os.path.join(RESULTS, f))
              for f in os.listdir(RESULTS) if f.startswith("forecast_") and f.endswith(".parquet")]
    if panels:
        te = treatment_effects(pd.concat(panels, ignore_index=True), arms, features=("nonlinear",))
        a = te.get("alpha", {}).get("nonlinear", {})
        with open(os.path.join(RESULTS, "treatment_effect.txt"), "w") as fh:
            fh.write(f"alpha[nonlinear]={a.get('estimate')}  t={a.get('t_value')}  p={a.get('p_value')}  n={te.get('n_obs')}\n")
        print(f"[treatment] alpha[nonlinear]={a.get('estimate'):.5f} t={a.get('t_value'):.3f} p={a.get('p_value'):.4f} n={te.get('n_obs')}", flush=True)
    if acc_rows:
        pd.concat(acc_rows, ignore_index=True).to_csv(os.path.join(RESULTS, "accuracy_all.csv"), index=False)
    print("[complete] ML-Useful full run finished", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/ml_useful_data/2018-01.csv")
