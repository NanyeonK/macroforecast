"""Goulet Coulombe, Leroux, Stevanovic, Surprenant (2021), "Macroeconomic data
transformations matter" (IJF), expressed as macroforecast PIPELINE arms.

This is the leak-free, pipeline-based replacement for the standalone
``scripts/replication/gcls_2021_table2_single.py``. The single script calls
``mf.preprocessing.reprocess(bundle)`` ONCE on the full sample (the EM imputation
therefore sees future data -- a look-ahead). Here the preprocessing is SPEC-LEVEL
and refit per origin on the retrain cadence (``origin_available`` / ``on_retrain``),
which is both leak-free and shared across arms via the per-target cache.

The feature-case construction (F / X / MARX / MAF / Level) and model parameters are
imported faithfully from the single script so the two stay in lockstep.
"""
from __future__ import annotations

from typing import Any

import macroforecast as mf
from macroforecast.pipeline import Arm, TargetSpec

# Reuse the exact feature/model construction from the single-cell replication so
# the pipeline and the standalone script cannot drift apart.
from scripts.replication.gcls_2021_table2_single import (  # noqa: E402
    TARGET_MAP,
    paper_feature_steps,
    paper_model_params,
)

# Default random-forest size (paper Table 2 uses 200 trees; the single script's CLI
# default is 200 as well). Exposed so the runner can override for smoke runs.
RF_N_ESTIMATORS = 200
RANDOM_STATE = 123

# The six feature cases that the random-forest arms span in Table 2. Names are kept
# exactly as requested so downstream parquet/CSV keys are stable.
RF_FEATURE_CASES: dict[str, str] = {
    "RF_F-Level": "F-Level",
    "RF_MARX": "MARX",
    "RF_F-X-MARX-Level": "F-X-MARX-Level",
    "RF_F-X-MAF": "F-X-MAF",
    "RF_X-Level": "X-Level",
}


def gcls_targets(policy: str) -> list[TargetSpec]:
    """The 10 paper targets resolved to the requested forecast policy.

    The forecast OBJECT is the average over the horizon of the one-period
    FRED-MD-transformed series (the paper's "average growth rate" construction).
    Because the pipeline applies the official t-code transform at the
    preprocessing stage, the target column reaching ``forecasting.run`` is already
    the one-period stationary series; we therefore set ``transform="value"`` and
    let the forecast policy build the horizon average:

      * ``direct_average`` -> target_transform resolves to "average_value"
        (direct multi-step average), matching the single script's
        ``forecast_policy="direct_average", target_transform="value"``.
      * ``path_average``   -> target_transform resolves to "value" with
        ``target_mode="path"``, matching the single script's path construction.

    ``transform``/``policy`` are set EXPLICITLY (not derived from the t-code) so the
    construction is identical for every series regardless of its integration order.
    """
    if policy not in {"direct_average", "path_average"}:
        raise ValueError(f"unknown policy {policy!r}")
    return [
        TargetSpec(name=col, transform="value", policy=policy)
        for col in TARGET_MAP.values()
    ]


def gcls_arms(
    target: str,
    transformed_predictors: list[str],
    level_predictors: list[str],
    *,
    n_estimators: int = RF_N_ESTIMATORS,
    random_state: int = RANDOM_STATE,
    smoke: bool = False,
) -> list[Arm]:
    """Build the GCLS arm grid for one (column) target.

    Arms:
      * ``FM``  -- factor model (model "far", F-only PCA factors n_factors=8,
        n_lag=12). This is the BENCHMARK.
      * ``AR``  -- autoregression on target lags only (no predictors).
      * five random-forest arms, one per feature case in ``RF_FEATURE_CASES``.

    Preprocessing is SPEC-LEVEL (shared cache), so every arm sets
    ``preprocessing=None`` to reuse the per-origin FittedPreprocessor.

    ``nested_in_benchmark`` is left False on all arms: the random forests do not
    nest the linear factor model, and the AR does not nest FM either, so Clark-West
    is not licensed against the FM benchmark. Diebold-Mariano is always reported.
    """

    def _features(case: str):
        # paper_feature_steps maps the case string to pca/lag/marx/maf/level steps.
        steps = paper_feature_steps(case, target, transformed_predictors, level_predictors)
        # Predictors must include the level block for any "Level" case so the lag_step
        # over level_predictors can resolve its columns.
        case_predictors = list(transformed_predictors)
        if "Level" in case.split("-"):
            case_predictors = case_predictors + list(level_predictors)
        return mf.feature_engineering.feature_spec(
            target=target,
            predictors=case_predictors,
            steps=steps,
            target_lags=range(0, 13),
            # target_transform/target_mode are overridden by forecasting.run from the
            # resolved forecast policy; set the value-based default here so a stray
            # direct run still builds averages from the one-period series.
            target_transform="value",
        )

    # FM benchmark: factors only, no target lags beyond what FAR uses internally.
    fm_features = mf.feature_engineering.feature_spec(
        target=target,
        predictors=list(transformed_predictors),
        steps=paper_feature_steps("F", target, transformed_predictors, level_predictors),
        target_lags=range(0, 13),
        target_transform="value",
    )
    arms: list[Arm] = [
        Arm(
            name="FM",
            model="far",
            features=fm_features,
            params=paper_model_params("far", random_state=random_state, n_estimators=n_estimators),
            is_benchmark=True,
            metadata={"feature_case": "F", "role": "benchmark"},
        ),
        Arm(
            name="AR",
            model="ar",
            features=mf.feature_engineering.feature_spec(
                target=target,
                predictors=[],
                target_lags=range(0, 13),
                target_transform="value",
            ),
            metadata={"feature_case": "AR", "role": "data-poor benchmark"},
        ),
    ]

    rf_cases = RF_FEATURE_CASES
    if smoke:
        # Smoke restricts to two representative RF arms: F-Level (uses the LEVEL__
        # raw-level block) and MARX (no level block) so both feature paths are
        # exercised against the FM benchmark.
        rf_cases = {"RF_F-Level": "F-Level", "RF_MARX": "MARX"}

    for arm_name, case in rf_cases.items():
        arms.append(
            Arm(
                name=arm_name,
                model="random_forest",
                features=_features(case),
                params=paper_model_params(
                    "random_forest", random_state=random_state, n_estimators=n_estimators
                ),
                nested_in_benchmark=False,
                metadata={"feature_case": case, "role": "contender"},
            )
        )
    return arms
