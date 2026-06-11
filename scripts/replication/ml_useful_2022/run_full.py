"""ML-Useful (c): full-design leak-free run, incremental + resumable.

5 targets x horizons {1,3,9,12,24} x core arms {AR,ARDI,RFAR,RFARDI}, full POOS
(1980-2017), leak-free (spec-level preprocessing + on_retrain cadence + cross-arm
cache + feature cadence). Writes per-(target,horizon) accuracy and the forecast
panel as each cell finishes (resumable via skip-existing), then the pooled Eq.11
treatment effect. This is a multi-hour compute job (cf. the GCLS INDPRO run ~15h).

Horizon consolidation (performance + correctness)
-------------------------------------------------
Each target builds ONE ``pipeline_spec`` carrying ALL horizons
(``horizons=[1, 3, 9, 12, 24]``) and calls ``run_pipeline`` ONCE, mirroring
``gcls_2021_pipeline/run_pipeline_full.py``. The per-origin EM imputation is
horizon-independent (it only imputes the panel up to each origin), so the runner
computes it once per origin and shares it across all horizons (and arms) via the
cross-arm/cross-horizon preprocessing cache keyed on ``origin_pos``. The
consolidated report is then split by horizon into per-(target, horizon) lean
parquet + accuracy CSV so resume granularity stays per cell.

The earlier version looped single horizons (``horizons=[h]``) sharing ONE
checkpoint directory per (target, arm). Origin positions are horizon-independent,
so that collided: the first horizon's checkpoint origins were reused for every
later horizon, silently forecasting horizon 1 for h>1. The multi-horizon path
namespaces each horizon's checkpoint under ``h<h>``, so it is both faster (one EM
pass) and free of that collision.
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



# --- parquet serialisation guard ---------------------------------------------
# The pipeline forecast frame carries object columns holding dicts/structs
# (params, model_selection, fixed_model_params, stored_model, window, ...). When
# such a column is empty pyarrow raises "Cannot write struct type ... with no
# child field to Parquet". Drop the non-scalar metadata columns and stringify any
# remaining object cells before writing; all columns needed for accuracy and the
# treatment regression (prediction, actual, contender, arm, target, horizon,
# origin, date, forecast_policy, target_transform, combined) are scalar and kept.
_PARQUET_DROP_COLS = {
    "params", "model_selection", "fixed_model_params", "stored_model", "window",
    "preprocessed", "combination", "quantile_predictions", "model_spec",
    "trained_model", "variance_prediction",
}


def _sanitize_for_parquet(df):
    keep = [c for c in df.columns if c not in _PARQUET_DROP_COLS]
    out = df[keep].copy()
    for c in list(out.columns):
        if out[c].dtype == object:
            out[c] = out[c].map(
                lambda v: v if (v is None or isinstance(v, (str, int, float, bool))) else str(v)
            )
    return out
# -----------------------------------------------------------------------------


def _slice_by_horizon(frame, horizon: int):
    """Return the rows of ``frame`` for one horizon (or None if absent/empty)."""
    if frame is None or getattr(frame, "empty", True):
        return None
    if "horizon" not in frame.columns:
        return frame
    out = frame[frame["horizon"] == horizon]
    return out if not out.empty else None


def main(data_csv: str) -> None:
    os.makedirs(RESULTS, exist_ok=True)
    raw = mf.data.load_fred_md(local_source=data_csv)
    predictors = tuple(c for c in raw.panel.columns
                       if c not in {t.name for t in ml_useful_targets()})
    pp = mf.preprocessing.preprocess_spec(transform="official", impute="em_factor",
                                          outliers="iqr", outlier_action="flag_as_nan", standardize="none")
    pol = mf.window.stage_policy("origin_available", update="on_retrain")
    fpol = mf.window.stage_policy("fit_window", update="on_retrain")
    base_arms = [a for a in ml_useful_arms("INDPRO", predictors, subset="core") if a.name in ARM_KEEP]
    arms = [dataclasses.replace(a, feature_policy=fpol) for a in base_arms]

    # The runner injects each horizon into the window's test spec at execution
    # time, so the base window's ``horizon`` is overridden per horizon. We supply
    # horizon=1 so the base window validates; test_start/test_end/retrain cadence
    # are the load-bearing fields.
    #
    # Hyperparameter tuning follows the paper's K-fold cross-validation (Goulet
    # Coulombe, Leroux, Stevanovic, Surprenant 2022, Sec. 3): a random K-fold CV
    # rather than a temporal hold-out. ``random_kfold`` is the methodologically
    # faithful choice and is robust at early origins where a single trailing
    # ``last_block`` of size 24 cannot be formed from the (often sparse) target-
    # available selection sample. The runner special-cases random_kfold via
    # ``_allow_non_temporal_selection_splits`` so the non-temporal folds are
    # accepted by ``select_params``.
    window = mf.window.from_cutoffs(
        estimation_start="1960-01", test_start="1980-01", test_end="2017-12",
        mode="expanding", horizon=1, embargo=0, retrain_every=24,
        val_method="random_kfold", val_n_splits=5, val_random_state=0,
    )

    acc_rows = []
    for tspec in ml_useful_targets():
        # Resume granularity stays per (target, horizon): skip the whole target
        # only if EVERY horizon parquet already exists (it is one EM pass anyway).
        stems = {h: f"{tspec.name}_h{h}" for h in HORIZONS}
        if all(os.path.exists(os.path.join(RESULTS, f"forecast_{stems[h]}.parquet")) for h in HORIZONS):
            print(f"[skip] {tspec.name} all horizons {list(HORIZONS)} (exist)", flush=True)
            continue

        # ONE consolidated multi-horizon spec per target: the per-origin EM is
        # computed once and shared across all horizons (and arms) via the
        # cross-horizon preprocessing cache. ``checkpoint_dir`` namespaces each
        # horizon's origins under ``<cell>/h<h>`` so horizons never collide.
        t0 = time.time()
        try:
            rep = run_pipeline(pipeline_spec(
                data=raw, targets=[tspec], horizons=list(HORIZONS), window=window, arms=arms,
                evaluation=EvalSpec(benchmark="AR", tests=["dm"]),
                preprocessing=pp, preprocessing_policy=pol, save_models=False,
                checkpoint_dir=os.path.join(RESULTS, "_checkpoints"),
            ))
        except Exception as exc:  # noqa: BLE001
            print(f"[fail] {tspec.name} (consolidated): {type(exc).__name__}: {str(exc)[:120]}", flush=True)
            continue
        dt_spec = time.time() - t0

        # Split the consolidated report into per-(target, horizon) cells.
        for h in HORIZONS:
            tag = stems[h]
            forecasts_h = _slice_by_horizon(rep.forecasts, h)
            accuracy_h = _slice_by_horizon(rep.accuracy, h)
            if forecasts_h is None or accuracy_h is None:
                print(f"[warn] {tag}: no rows in report", flush=True)
                continue
            _sanitize_for_parquet(forecasts_h).to_parquet(os.path.join(RESULTS, f"forecast_{tag}.parquet"))
            acc = accuracy_h.assign(horizon=h)
            acc.to_csv(os.path.join(RESULTS, f"accuracy_{tag}.csv"), index=False)
            acc_rows.append(acc)
            best = acc.sort_values("relative_mse").iloc[0]
            print(f"[done] {tag} best={best['contender']} relRMSE={best['relative_mse']:.3f}", flush=True)
        print(f"[spec] {tspec.name} horizons={list(HORIZONS)} arms={len(arms)} secs={dt_spec:.0f}", flush=True)

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
