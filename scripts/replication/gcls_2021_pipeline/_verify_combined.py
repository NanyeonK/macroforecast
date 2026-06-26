"""Bounded GCLS cross-check: combined multi-target spec vs per-target spec.

Exercises the MARX_y / MAF_y re-targeting fix (columns=None on the target_panel
input). Builds the arm set ONCE (against INDPRO's YGROWTH__ column), then runs:
  (A) a COMBINED spec over [INDPRO, CPI] with arms {FM, AR, RF_MARX} (RF_MARX uses
      the MARX_y target-derived block, so it is the load-bearing re-targeting test),
  (B) a per-target INDPRO spec over the same arms/horizons/window,
on a tiny window (test_start 1980-01, test_end 1981-12), horizons [1, 3], n_jobs=1.

Verifies:
  1. both targets present with non-zero rows at both horizons in (A),
  2. INDPRO and CPI RF_MARX forecasts differ (re-targeting switched the target),
  3. combined INDPRO predictions == per-target INDPRO predictions (max abs ~0).
If (3) failed it would mean MARX_y stayed pinned to the build-time target column.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import warnings

warnings.simplefilter("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import macroforecast as mf  # noqa: E402
from macroforecast.pipeline import EvalSpec, pipeline_spec, run_pipeline  # noqa: E402

from scripts.replication.gcls_2021_pipeline.registry import (  # noqa: E402
    TARGET_MAP,
    YGROWTH_PREFIX,
    gcls_arms,
    gcls_targets,
    ygrowth_column,
)
from scripts.replication.gcls_2021_pipeline.run_pipeline_full import (  # noqa: E402
    _augmented_bundle,
    _level_predictors,
)

ARMS_KEEP = {"FM", "AR", "RF_MARX"}
HORIZONS = [1, 3]


def main(data_csv: str | None) -> None:
    raw = (
        mf.data.load_fred_md(local_source=data_csv)
        if data_csv
        else mf.data.load_fred_md(vintage="2018-01")
    )
    bundle = _augmented_bundle(raw)
    raw_targets = set(TARGET_MAP.values())
    transformed_predictors = [
        c for c in raw.panel.columns
        if c not in raw_targets and not str(c).startswith(YGROWTH_PREFIX)
    ]
    level_predictors = _level_predictors(raw)
    pp = mf.preprocessing.preprocess_spec(
        transform="official", impute="em_factor", outliers="iqr",
        outlier_action="flag_as_nan", standardize="none",
    )
    pp_pol = mf.window.stage_policy("origin_available", update="on_retrain")
    fpol = mf.window.stage_policy("fit_window", update="on_retrain")

    indpro_g = ygrowth_column(TARGET_MAP["INDPRO"])
    cpi_g = ygrowth_column(TARGET_MAP["CPI"])

    import dataclasses
    base = [a for a in gcls_arms(indpro_g, transformed_predictors, level_predictors,
                                 n_estimators=50) if a.name in ARMS_KEEP]
    arms = [dataclasses.replace(a, feature_policy=fpol) for a in base]

    window = mf.window.from_cutoffs(
        estimation_start="1960-01", test_start="1980-01", test_end="1981-12",
        mode="expanding", horizon=1, embargo=0, retrain_every=24,
        val_method="last_block", val_size=24,
    )
    policy = "direct_average"
    tmap = {t.name: t for t in gcls_targets(policy)}
    indpro, cpi = tmap[indpro_g], tmap[cpi_g]

    # (A) combined.
    rep_c = run_pipeline(pipeline_spec(
        data=bundle, targets=[indpro, cpi], horizons=HORIZONS, window=window,
        arms=arms, evaluation=EvalSpec(benchmark="FM", tests=["dm"]),
        preprocessing=pp, preprocessing_policy=pp_pol, save_models=False, n_jobs=1,
    ))
    fc = rep_c.forecasts
    print("=== GCLS COMBINED spec ===")
    print("failed_cells:", list(rep_c.failed_cells))
    for tn in (indpro_g, cpi_g):
        for h in HORIZONS:
            print(f"  rows target={tn} h={h}: {len(fc[(fc['target']==tn)&(fc['horizon']==h)])}")

    ind_marx = fc[(fc["target"] == indpro_g) & (fc["arm"] == "RF_MARX") & (fc["horizon"] == 1)] \
        .set_index("origin")["prediction"]
    cpi_marx = fc[(fc["target"] == cpi_g) & (fc["arm"] == "RF_MARX") & (fc["horizon"] == 1)] \
        .set_index("origin")["prediction"]
    common = ind_marx.index.intersection(cpi_marx.index)
    gap = float(np.abs(ind_marx.loc[common] - cpi_marx.loc[common]).max()) if len(common) else float("nan")
    print(f"  RF_MARX pred max|INDPRO-CPI| (should be >0): {gap:.6e}")

    # (B) per-target INDPRO.
    rep_i = run_pipeline(pipeline_spec(
        data=bundle, targets=[indpro], horizons=HORIZONS, window=window,
        arms=arms, evaluation=EvalSpec(benchmark="FM", tests=["dm"]),
        preprocessing=pp, preprocessing_policy=pp_pol, save_models=False, n_jobs=1,
    ))
    fi = rep_i.forecasts
    key = ["arm", "horizon", "origin"]
    a = fc[fc["target"] == indpro_g][key + ["prediction"]].set_index(key)
    b = fi[fi["target"] == indpro_g][key + ["prediction"]].set_index(key)
    joined = a.join(b, lsuffix="_c", rsuffix="_p", how="inner")
    diff = float(np.abs(joined["prediction_c"] - joined["prediction_p"]).max())
    print("\n=== combined INDPRO == per-target INDPRO ===")
    print(f"  matched rows: {len(joined)}")
    print(f"  max abs prediction diff (should be ~0): {diff:.3e}")

    ok = (
        len(ind_marx) > 0 and len(cpi_marx) > 0 and not rep_c.failed_cells
        and gap > 0 and len(joined) > 0 and diff < 1e-9
    )
    print("\nVERIFY:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
