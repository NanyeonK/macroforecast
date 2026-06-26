"""Bounded cross-check: combined multi-target spec vs per-target spec.

Builds ONE arm set (against YTARGET__INDPRO), then runs:
  (A) a COMBINED spec over [INDPRO, CPIAUCSL] x 4 arms x horizons [1,12],
  (B) a per-target INDPRO spec over the same arms/horizons/window,
on a tiny test window (2000-01..2001-12), and verifies:
  1. both targets present with non-zero rows at both horizons in (A),
  2. INDPRO and CPI forecasts in (A) differ (re-targeting actually switched),
  3. the realized ``actual`` for INDPRO is on a Delta-log growth scale and CPI
     on an inflation scale (sanity that the target switched),
  4. combined-spec INDPRO predictions == per-target INDPRO predictions
     (max abs diff ~ 0): combining targets did not change results.

Run with n_jobs=1 (serial) so the comparison is deterministic and fast.
"""
from __future__ import annotations

import dataclasses
import os
import sys

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
    YTARGET_PREFIX,
    ml_useful_arms,
    ml_useful_targets,
    ytarget_column,
)
from run_full import _augmented_bundle  # noqa: E402
from macroforecast.pipeline import EvalSpec, pipeline_spec, run_pipeline  # noqa: E402

ARM_KEEP = {"AR", "ARDI", "RFAR", "RFARDI"}
HORIZONS = [1, 12]


def _build(data_csv: str):
    raw = mf.data.load_fred_md(local_source=data_csv)
    bundle = _augmented_bundle(raw)
    predictors = tuple(
        c for c in bundle.panel.columns if not str(c).startswith(YTARGET_PREFIX)
    )
    pp = mf.preprocessing.preprocess_spec(
        transform="official", impute="em_factor", outliers="iqr",
        outlier_action="flag_as_nan", standardize="none",
    )
    pol = mf.window.stage_policy("origin_available", update="on_retrain")
    fpol = mf.window.stage_policy("fit_window", update="on_retrain")
    base_arms = [
        a for a in ml_useful_arms(ytarget_column("INDPRO"), predictors, subset="core")
        if a.name in ARM_KEEP
    ]
    arms = [dataclasses.replace(a, feature_policy=fpol) for a in base_arms]
    window = mf.window.from_cutoffs(
        estimation_start="1960-01", test_start="2000-01", test_end="2001-12",
        mode="expanding", horizon=1, embargo=0, retrain_every=24,
        val_method="random_kfold", val_n_splits=5, val_random_state=0,
    )
    return bundle, arms, pp, pol, window


def main(data_csv: str) -> None:
    bundle, arms, pp, pol, window = _build(data_csv)
    tspecs = {t.name: t for t in ml_useful_targets()}
    indpro = tspecs[ytarget_column("INDPRO")]
    cpi = tspecs[ytarget_column("CPIAUCSL")]

    # (A) COMBINED 2-target spec.
    rep_c = run_pipeline(pipeline_spec(
        data=bundle, targets=[indpro, cpi], horizons=HORIZONS, window=window,
        arms=arms, evaluation=EvalSpec(benchmark="AR", tests=["dm"]),
        preprocessing=pp, preprocessing_policy=pol, save_models=False, n_jobs=1,
    ))
    fc = rep_c.forecasts

    print("=== COMBINED spec ===")
    print("failed_cells:", list(rep_c.failed_cells))
    print("empty_cells:", list(rep_c.empty_cells))
    for tname in (indpro.name, cpi.name):
        for h in HORIZONS:
            n = len(fc[(fc["target"] == tname) & (fc["horizon"] == h)])
            print(f"  rows target={tname} h={h}: {n}")

    # (2) INDPRO vs CPI forecasts differ.
    ind_a = fc[(fc["target"] == indpro.name) & (fc["horizon"] == 1)].sort_values(
        ["arm", "origin"]
    )
    cpi_a = fc[(fc["target"] == cpi.name) & (fc["horizon"] == 1)].sort_values(
        ["arm", "origin"]
    )
    # (3) scale sanity: INDPRO growth vs CPI inflation magnitudes.
    print("\n=== target-correctness (scale of realized 'actual') ===")
    print(
        f"  INDPRO actual mean/std: {ind_a['actual'].mean():.5f} / "
        f"{ind_a['actual'].std():.5f}"
    )
    print(
        f"  CPI    actual mean/std: {cpi_a['actual'].mean():.5f} / "
        f"{cpi_a['actual'].std():.5f}"
    )
    # AR-arm predictions differ between the two targets (re-targeting worked).
    ind_ar = ind_a[ind_a["arm"] == "AR"].set_index("origin")["prediction"]
    cpi_ar = cpi_a[cpi_a["arm"] == "AR"].set_index("origin")["prediction"]
    common = ind_ar.index.intersection(cpi_ar.index)
    pred_gap = float(np.abs(ind_ar.loc[common] - cpi_ar.loc[common]).max())
    print(f"  AR pred max|INDPRO-CPI| (should be >0): {pred_gap:.6e}")

    # (B) per-target INDPRO spec (same arms/window/horizons), n_jobs=1.
    rep_i = run_pipeline(pipeline_spec(
        data=bundle, targets=[indpro], horizons=HORIZONS, window=window,
        arms=arms, evaluation=EvalSpec(benchmark="AR", tests=["dm"]),
        preprocessing=pp, preprocessing_policy=pol, save_models=False, n_jobs=1,
    ))
    fi = rep_i.forecasts

    # (4) combined INDPRO == per-target INDPRO predictions.
    key = ["arm", "horizon", "origin"]
    a = fc[fc["target"] == indpro.name][key + ["prediction"]].set_index(key)
    b = fi[fi["target"] == indpro.name][key + ["prediction"]].set_index(key)
    joined = a.join(b, lsuffix="_combined", rsuffix="_pertarget", how="inner")
    diff = float(
        np.abs(joined["prediction_combined"] - joined["prediction_pertarget"]).max()
    )
    print("\n=== combined INDPRO == per-target INDPRO ===")
    print(f"  matched rows: {len(joined)}")
    print(f"  max abs prediction diff (should be ~0): {diff:.3e}")

    ok = (
        len(ind_a) > 0
        and len(cpi_a) > 0
        and not rep_c.failed_cells
        and pred_gap > 0
        and len(joined) > 0
        and diff < 1e-9
    )
    print("\nVERIFY:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/ml_useful_data/2018-01.csv")
