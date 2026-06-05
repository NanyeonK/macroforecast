"""Leak-free bounded ML-Useful run: preprocessing moved INSIDE the window (origin-available).

Unlike run_bounded.py (which preprocessed upfront on the full sample -> look-ahead), this
feeds the RAW FRED-MD bundle and attaches an origin-available preprocessing spec to every
arm, so the t-code transform, EM-factor imputation, and outlier thresholds are recomputed
within each expanding training window. This is the faithful POOS setup (step a).
Tiny scope here is a leak-free machinery proof; step b scales it to a meaningful comparison.
"""
from __future__ import annotations

import dataclasses
import os
import sys
import warnings

warnings.simplefilter("ignore")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import macroforecast as mf  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from registry import ml_useful_arms  # noqa: E402
from macroforecast.pipeline import EvalSpec, TargetSpec, pipeline_spec, run_pipeline  # noqa: E402


def main(data_csv: str, *, targets=("INDPRO",), horizon=1, n_origins=6,
         arm_names=("AR", "RFARDI"), subset="core") -> None:
    raw = mf.data.load_fred_md(local_source=data_csv)          # RAW levels + tcode metadata
    idx = raw.panel.index
    predictors = tuple(c for c in raw.panel.columns
                       if c not in {"INDPRO", "UNRATE", "CPIAUCSL", "HOUST", "T10YFFM"})

    # leak-free: official t-codes + EM imputation + outliers, refit every origin
    pp = mf.preprocessing.preprocess_spec(
        transform="official", impute="em_factor", outliers="iqr",
        outlier_action="flag_as_nan", standardize="none",
    )
    policy = mf.window.stage_policy("origin_available")

    tgt_specs = [TargetSpec(t, transform="average_value", policy="direct_average") for t in targets]
    window = mf.window.from_cutoffs(
        estimation_start=idx[300], test_start=idx[-n_origins], test_end=idx[-1],
        mode="expanding", horizon=horizon, embargo=0, val_method="last_block", val_size=24,
    )
    arms = []
    for a in ml_useful_arms("INDPRO", predictors, subset=subset):
        if a.name in set(arm_names):
            arms.append(dataclasses.replace(a, preprocessing=pp, preprocessing_policy=policy))

    spec = pipeline_spec(
        data=raw, targets=tgt_specs, horizons=[horizon], window=window,
        arms=arms, evaluation=EvalSpec(benchmark="AR", tests=["dm"]), save_models=False,
    )
    report = run_pipeline(spec)
    acc = report.accuracy.sort_values(["target", "relative_mse"])
    print("=== leak-free relative RMSE (vs AR), per-window preprocessing ===")
    print(acc[["target", "contender", "rmse", "relative_mse", "n_common"]].to_string(index=False))
    print("\nleakage_audit:", report.leakage_audit.get("window_warnings"))
    print("preprocessing policy on arms:", {a.name: str(a.preprocessing_policy.scope) for a in arms})


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/ml_useful_data/2018-01.csv")
