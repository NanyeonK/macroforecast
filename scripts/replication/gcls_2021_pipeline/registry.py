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

# The three "core" arms needed for the appendix growth-target comparison: the FM
# benchmark plus the two random-forest feature cases used in that table (RF_F-Level
# and RF_MARX). The ``--core-arms-only`` flag restricts the grid to these so the
# corrected re-run of HOUST/M2/CPI/PPI stays fast.
CORE_ARMS: tuple[str, ...] = ("FM", "RF_F-Level", "RF_MARX")

# Prefix for the dedicated per-target growth column (the paper's "average growth
# rate" object). The TARGET is the horizon-average of this one-period growth, which
# is built from the RAW level in run_pipeline_full._augmented_bundle and is therefore
# correct regardless of the column's own FRED-MD t-code.
YGROWTH_PREFIX = "YGROWTH__"


def ygrowth_column(column: str) -> str:
    """Name of the dedicated growth-target column for a raw FRED-MD ``column``."""
    return f"{YGROWTH_PREFIX}{column}"


def gcls_targets(policy: str) -> list[TargetSpec]:
    """The 10 paper targets resolved to the requested forecast policy.

    The forecast OBJECT is the average over the horizon of the one-period GROWTH of
    the target (the paper's "average growth rate" construction, p.1343 / appendix
    p.52). The target reaches ``forecasting.run`` as the dedicated ``YGROWTH__<col>``
    column, which is the one-period growth (Delta log, or Delta for UNRATE) built
    from the RAW level in ``run_pipeline_full._augmented_bundle`` and assigned an
    identity t-code so the official transform passes it through unchanged. We then
    set ``transform="value"`` and let the forecast policy build the horizon average:

      * ``direct_average`` -> target_transform resolves to "average_value", i.e.
        (1/h) sum_{h'=1..h} growth_{t+h'} (direct multi-step average).
      * ``path_average``   -> target_transform resolves to "value" with
        ``target_mode="path"`` (fit each one-period growth separately, then average).

    Building the target from ``YGROWTH__`` rather than the t-coded panel column is
    what fixes HOUST (t-code 4 = log level) and the I(2) price series M2/CPI/PPI
    (t-code 6 = second log-difference): averaging the panel column there gives the
    wrong scale, whereas averaging the YGROWTH__ growth gives the paper's object. For
    the t-code-5 (Delta log) and t-code-2 (Delta) targets this is numerically
    equivalent to the previous behaviour.

    ``transform``/``policy`` are set EXPLICITLY (not derived from the t-code) so the
    construction is identical for every series regardless of its integration order.
    """
    if policy not in {"direct_average", "path_average"}:
        raise ValueError(f"unknown policy {policy!r}")
    return [
        TargetSpec(name=ygrowth_column(col), transform="value", policy=policy)
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
    core_arms_only: bool = False,
) -> list[Arm]:
    """Build the GCLS arm grid for one target.

    ``target`` is the forecast TARGET column, i.e. the ``YGROWTH__<col>`` one-period
    growth column (the paper's "average growth rate" object). It is used both as the
    feature spec's target (so target lags are lags of the growth object) and as the
    column for the target-derived MARX_y/MAF_y blocks (so they read the growth object
    from ``target_panel``). The PREDICTOR blocks (F/X/MARX_X/MAF_X/Level) still use the
    McCracken-Ng t-coded ``transformed_predictors``/``level_predictors`` -- t-codes for
    predictors X, growth for the target Y, exactly as in the paper.

    Passing the YGROWTH__ column as the feature target also prevents the pipeline
    runtime from re-targeting the feature spec (``run._run_one_arm_target`` only
    rewrites ``features.target`` when it differs from the resolved target name), which
    would otherwise leave the MARX_y ``columns=[...]`` pointing at a column absent
    from ``target_panel``.

    Arms:
      * ``FM``  -- factor model (model "far", F-only PCA factors n_factors=8,
        n_lag=12). This is the BENCHMARK.
      * ``AR``  -- autoregression on target lags only (no predictors).
      * five random-forest arms, one per feature case in ``RF_FEATURE_CASES``.

    ``core_arms_only`` restricts the grid to ``CORE_ARMS`` (FM, RF_F-Level, RF_MARX),
    the arms needed for the appendix growth-target comparison.

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
    if smoke or core_arms_only:
        # Smoke / core-arms-only restrict to two representative RF arms: F-Level
        # (uses the LEVEL__ raw-level block) and MARX (no level block) so both
        # feature paths are exercised against the FM benchmark. These two plus FM
        # are exactly CORE_ARMS, the appendix growth-target comparison set.
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

    if core_arms_only:
        # Keep only the three core arms (FM benchmark + the two RF feature cases).
        # The benchmark FM must always be present for relative-MSE evaluation.
        arms = [a for a in arms if a.name in CORE_ARMS]
    return arms
