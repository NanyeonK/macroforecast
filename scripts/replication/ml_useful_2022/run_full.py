"""ML-Useful (c): full-design leak-free run, incremental + resumable.

5 targets x horizons {1,3,9,12,24} x core arms {AR,ARDI,RFAR,RFARDI}, full POOS
(1980-2017), leak-free (spec-level preprocessing + on_retrain cadence + cross-arm
cache + feature cadence). Writes per-(target,horizon) accuracy and the forecast
panel as each cell finishes (resumable via skip-existing), then the pooled Eq.11
treatment effect. This is a multi-hour compute job (cf. the GCLS INDPRO run ~15h).

Combined multi-target consolidation (performance + correctness)
---------------------------------------------------------------
The run builds ONE ``pipeline_spec`` over ALL 5 targets x 4 arms x 5 horizons
(``targets=[5 ResolvedTargets]``, ``horizons=[1, 3, 9, 12, 24]``) and calls
``run_pipeline`` ONCE with ``n_jobs="auto"``. The auto allocator sees ~100 cells
and splits the 48 cores across cell workers, so the whole CPU saturates and the
expensive RFARDI-h24 cells of DIFFERENT targets run CONCURRENTLY instead of
one-target-at-a-time (the prior per-target loop left ~8 cores idle).

The arms carry a single feature spec; the pipeline re-targets each arm to the
active YTARGET__ column (``run._run_one_arm_target`` rewrites ``features.target``
and clears ``features.targets``). The ML-Useful feature steps are TARGET-AGNOSTIC
(PCA factors over the YTARGET-excluded predictors; the only target-derived block
is ``target_lags``, which reads ``features.target``), so one arm set is correct
for every target -- verified by ``_verify_combined.py`` (combined INDPRO forecasts
== per-target INDPRO forecasts, and INDPRO vs CPI differ at the right scale).

The per-origin EM imputation is horizon-independent (it only imputes the panel up
to each origin), so the runner shares it across all horizons (and arms) per target
via the preprocessing cache keyed on ``origin_pos``. The consolidated report is
then split by (target, horizon) into per-cell lean parquet + accuracy CSV so
resume granularity stays per cell.

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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import macroforecast as mf  # noqa: E402

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
from registry import (  # noqa: E402
    TARGET_KIND,
    YTARGET_PREFIX,
    ml_useful_arms,
    ml_useful_targets,
    ytarget_column,
)
from treatment import treatment_effects  # noqa: E402
from macroforecast.pipeline import EvalSpec, pipeline_spec, run_pipeline  # noqa: E402

RESULTS = os.path.join(HERE, "results")
HORIZONS = (1, 3, 9, 12, 24)
ARM_KEEP = {"AR", "ARDI", "RFAR", "RFARDI"}

# Raw FRED-MD columns underlying the 5 forecast targets (paper eqs fcst0/fcst1).
TARGET_COLUMNS = ("INDPRO", "CPIAUCSL", "HOUST", "UNRATE", "T10YFFM")


def _one_period_object(level: pd.Series, kind: str) -> pd.Series:
    """One-period forecast OBJECT built from the RAW level series (paper target).

    ``"log_diff"`` -> log Y_t - log Y_{t-1}   (INDPRO, CPIAUCSL, HOUST: I(1) growth).
    ``"diff"``     -> Y_t - Y_{t-1}            (UNRATE: I(1) change, no log).
    ``"level"``    -> Y_t                      (T10YFFM: I(0) spread level).

    This is DECOUPLED from the predictor t-codes on purpose: building the target on
    the already-t-coded panel column double-transforms (log of a signed Delta-log is
    NaN -> empty long-horizon cells). Computing it directly from the raw level is the
    paper's construction regardless of the column's own FRED-MD t-code (HOUST is
    t-code 4 = log level, CPIAUCSL is t-code 6 = second log-difference).
    """
    values = level.astype(float)
    if kind == "diff":
        return values.diff()
    if kind == "level":
        return values
    if kind == "log_diff":
        # log is only defined for strictly positive levels (all 4 growth targets are
        # strictly positive in this vintage); non-positive entries yield NaN.
        positive = values.where(values > 0)
        return np.log(positive) - np.log(positive.shift(1))
    raise ValueError(f"unknown target kind {kind!r}")


def _augmented_bundle(raw):
    """Return a bundle whose panel also carries the dedicated ``YTARGET__<col>`` columns.

    Each ``YTARGET__<col>`` column is the one-period forecast object (Delta log,
    Delta, or the raw level) computed from the RAW level (see ``_one_period_object``)
    and assigned t-code 1 (identity) so the official panel transform passes it through
    unchanged. The forecast policy's ``average_value`` (direct_average) then builds
    (1/h) sum_{h'} object_{t+h'} = the paper's average growth/change target; for
    T10YFFM the ``direct`` policy forecasts the level h-ahead. The YTARGET__ columns
    are EXCLUDED from the predictor set so the realized target never leaks as a feature.
    """
    panel = raw.panel.copy()
    target_frames: list[pd.Series] = []
    for col in TARGET_COLUMNS:
        obj = _one_period_object(raw.panel[col], TARGET_KIND[col])
        obj.name = ytarget_column(col)
        target_frames.append(obj)
    targets = pd.concat(target_frames, axis=1)

    new_panel = pd.concat([panel, targets], axis=1)
    codes = dict(raw.metadata.get("transform_codes", {}))
    for col in TARGET_COLUMNS:
        codes[ytarget_column(col)] = 1  # identity: official transform passes through
    metadata = {
        k: raw.metadata.get(k)
        for k in ("dataset", "frequency", "version_mode", "vintage", "data_through")
        if k in raw.metadata
    }
    metadata["transform_codes"] = codes
    new_panel.attrs = {"macroforecast_transform_codes": dict(codes)}
    return dataclasses.replace(raw, panel=new_panel, metadata=metadata)



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
    # Augment the panel with the dedicated YTARGET__<col> forecast-object columns
    # (built from the RAW level, identity t-code). The official panel transform then
    # passes them through unchanged; the forecast policy averages the one-period
    # object over the horizon (paper eqs fcst0/fcst1).
    bundle = _augmented_bundle(raw)
    # Predictors = the full McCracken-Ng t-coded panel EXCLUDING the engineered
    # YTARGET__ columns (so the realized target never leaks as a feature). The raw
    # target columns remain in the predictor panel as standard panel members.
    predictors = tuple(c for c in bundle.panel.columns
                       if not str(c).startswith(YTARGET_PREFIX))
    pp = mf.preprocessing.preprocess_spec(transform="official", impute="em_factor",
                                          outliers="iqr", outlier_action="flag_as_nan", standardize="none")
    # EM-factor imputation stays on a 24-origin cadence (its own cost driver);
    # leak-free (past data only) and ~unchanged values vs every-origin refit.
    pol = mf.window.stage_policy("origin_available", update=24)
    fpol = mf.window.stage_policy("fit_window", update=24)
    # Build the arms once against the INDPRO YTARGET column; the pipeline runtime
    # re-targets each arm's feature spec (target lags / target-derived blocks) to the
    # per-target YTARGET__ column, while the PCA-factor predictor block stays fixed.
    base_arms = [a for a in ml_useful_arms(ytarget_column("INDPRO"), predictors, subset="core")
                 if a.name in ARM_KEEP]
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
    # Coefficients RE-ESTIMATED every origin (retrain_every=1, standard expanding
    # POOS); expensive HP grid search only every 24 months (retune_every=24,
    # retune_on_retrain=False, reuse_params=True) per the paper's "re-optimize
    # hyperparameters every two years" (Sec. 4.3). retrain_every=24 previously froze
    # the whole fit for 24 months -- the AR benchmark forecasts recursively from its
    # training tail and ignores X_test, so its forecast went stale for 24 months,
    # inflating every model's relative RMSPE (acute for the volatile I(0) T10YFFM
    # level at h=1: AR_RMSE 1.34 frozen vs 0.56 refit each origin).
    window = mf.window.from_cutoffs(
        estimation_start="1960-01", test_start="1980-01", test_end="2017-12",
        mode="expanding", horizon=1, embargo=0,
        retrain_every=1, retune_every=24, retune_on_retrain=False, reuse_params=True,
        val_method="random_kfold", val_n_splits=5, val_random_state=0,
    )

    acc_rows = []
    # ONE combined multi-TARGET, multi-horizon spec: all 5 targets x 4 arms x 5
    # horizons = 100 cells in a single pipeline_spec with n_jobs="auto". The auto
    # allocator (pipeline.spec) sees ~100 cells and splits the 48 cores across cell
    # workers so the whole CPU saturates -- the expensive RFARDI-h24 cells of
    # DIFFERENT targets now run CONCURRENTLY instead of one-target-at-a-time. The
    # pipeline re-targets each arm's feature spec to the active YTARGET__ column
    # (run._run_one_arm_target rewrites features.target / clears features.targets);
    # the ML-Useful feature steps are target-AGNOSTIC (PCA factors over the
    # YTARGET-excluded predictors; the only target-derived block is target_lags,
    # which reads features.target), so one arm set is correct for every target.
    tspecs = list(ml_useful_targets())
    # Per-(target, horizon) output filenames key on the RAW column (strip the
    # YTARGET__ prefix) -> forecast_INDPRO_h1.parquet, matching the prior layout.
    def _raw_col(name: str) -> str:
        return name[len(YTARGET_PREFIX):] if name.startswith(YTARGET_PREFIX) else name

    stems = {(t.name, h): f"{_raw_col(t.name)}_h{h}" for t in tspecs for h in HORIZONS}
    # Resume: drop already-complete (target, horizon) cells from the spec. Skip a
    # whole TARGET only if every horizon parquet exists; otherwise the target stays
    # in the combined spec (one EM pass shared across its horizons/arms anyway).
    pending = [
        t for t in tspecs
        if not all(
            os.path.exists(os.path.join(RESULTS, f"forecast_{stems[(t.name, h)]}.parquet"))
            for h in HORIZONS
        )
    ]
    for t in tspecs:
        if t not in pending:
            print(f"[skip] {t.name} all horizons {list(HORIZONS)} (exist)", flush=True)

    if pending:
        t0 = time.time()
        rep = run_pipeline(pipeline_spec(
            data=bundle, targets=pending, horizons=list(HORIZONS), window=window, arms=arms,
            evaluation=EvalSpec(benchmark="AR", tests=["dm"]),
            preprocessing=pp, preprocessing_policy=pol, save_models=False,
            checkpoint_dir=os.path.join(RESULTS, "_checkpoints"),
            n_jobs="auto",  # auto: ~100 cells -> all 48 cores; deterministic == n_jobs=1
        ))
        dt_spec = time.time() - t0

        # Surface failed and zero-row cells across ALL targets so a silently-missing
        # (target, horizon) is visible in the run log (does not change numerics).
        for fc in (rep.failed_cells or ()):
            print(
                f"[failed-cell] target={fc.get('target')} arm={fc.get('arm')} "
                f"horizons={fc.get('horizons')} error={fc.get('error')}",
                flush=True,
            )
        for ec in (rep.empty_cells or ()):
            print(
                f"[empty-cell] target={ec.get('target')} horizon={ec.get('horizon')} "
                f"arms={ec.get('arms')} (ran OK, zero rows -> absent from evaluation)",
                flush=True,
            )

        # Split the consolidated report into per-(target, horizon) cells, keyed on
        # BOTH target and horizon (the report now spans every pending target).
        for t in pending:
            for h in HORIZONS:
                tag = stems[(t.name, h)]
                fch = rep.forecasts
                ach = rep.accuracy
                forecasts_h = fch[(fch["target"] == t.name) & (fch["horizon"] == h)] \
                    if fch is not None and not getattr(fch, "empty", True) else None
                accuracy_h = ach[(ach["target"] == t.name) & (ach["horizon"] == h)] \
                    if ach is not None and not getattr(ach, "empty", True) else None
                if forecasts_h is None or forecasts_h.empty or accuracy_h is None or accuracy_h.empty:
                    print(f"[warn] {tag}: no rows in report", flush=True)
                    continue
                _sanitize_for_parquet(forecasts_h).to_parquet(os.path.join(RESULTS, f"forecast_{tag}.parquet"))
                acc = accuracy_h.assign(horizon=h)
                acc.to_csv(os.path.join(RESULTS, f"accuracy_{tag}.csv"), index=False)
                acc_rows.append(acc)
                best = acc.sort_values("relative_mse").iloc[0]
                print(f"[done] {tag} best={best['contender']} relRMSE={best['relative_mse']:.3f}", flush=True)
        print(f"[spec] combined targets={len(pending)} horizons={list(HORIZONS)} arms={len(arms)} secs={dt_spec:.0f}", flush=True)

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
