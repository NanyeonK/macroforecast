"""Bounded ML-Useful (JAE 2022) replication run on the official 2018-01 FRED-MD vintage.

Proves the reconstruction end-to-end: official data -> stationarise (McCracken-Ng
t-codes) -> Table-1 arms with ML-feature flags -> pipeline forecasts -> relative
RMSE + nonlinearity treatment effect. Bounded scope (few targets/horizons, short
POOS window) for a fast validation; scale up for the full Tables A1-A5 + Eq.11.
"""
from __future__ import annotations

import os
import sys
import warnings

warnings.simplefilter("ignore")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import macroforecast as mf  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from registry import ml_useful_arms  # noqa: E402
from treatment import treatment_effects  # noqa: E402
from macroforecast.pipeline import EvalSpec, TargetSpec, pipeline_spec, run_pipeline  # noqa: E402


def main(data_csv: str) -> None:
    raw = mf.data.load_fred_md(local_source=data_csv)
    pre = mf.preprocessing.reprocess(raw, transform="official")
    panel = pre.panel.iloc[20:]  # drop leading NaN rows from differencing; demo preprocesses upfront
    # bounded scope
    targets = [TargetSpec("INDPRO", transform="average_value", policy="direct_average")]
    horizon = 1
    idx = panel.index
    window = mf.window.from_cutoffs(
        estimation_start=idx[300], test_start=idx[-12], test_end=idx[-1],
        mode="expanding", horizon=horizon, embargo=0, val_method="last_block", val_size=24,
    )
    predictors = tuple(c for c in panel.columns if c not in {"INDPRO","UNRATE","CPIAUCSL","HOUST","T10YFFM"})
    arms = []
    seen = set()
    for tname in ("INDPRO",):  # arms are target-agnostic; build once, pipeline retargets
        for a in ml_useful_arms(tname, predictors, subset="core"):
            if a.name in {"AR","ARDI","RFAR","RFARDI"} and a.name not in seen:
                arms.append(a); seen.add(a.name)
    spec = pipeline_spec(
        data=panel, targets=targets, horizons=[horizon], window=window,
        arms=arms, evaluation=EvalSpec(benchmark="AR", tests=["dm"]), save_models=False,
    )
    report = run_pipeline(spec)
    acc = report.accuracy.sort_values(["target", "relative_mse"])
    print("=== relative RMSE (vs AR), bounded ===")
    print(acc[["target", "contender", "rmse", "relative_mse", "n_common"]].to_string(index=False))
    te = treatment_effects(report.forecasts, arms, features=("nonlinear",))
    print("\n=== nonlinearity treatment effect (alpha, Eq.11 bounded) ===")
    print("alpha[nonlinear] =", te["alpha"], " n_obs =", te["n_obs"])
    print("(paper Finding 1: nonlinearity alpha should be positive)")


if __name__ == "__main__":
    csv = sys.argv[1] if len(sys.argv) > 1 else "/tmp/ml_useful_data/2018-01.csv"
    main(csv)
