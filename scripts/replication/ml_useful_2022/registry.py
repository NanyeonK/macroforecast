"""ML-Useful (Goulet Coulombe, Leroux, Stevanovic, Surprenant 2022, JAE 10.1002/jae.2910).

Table-1 model grid expressed as macroforecast pipeline ARMS carrying the paper's four
ML-feature flags (nonlinearity, regularization, hyperparameter-CV, loss) plus the
data-poor/data-rich regime. The feature flags are deterministic per arm so the
treatment-effect regression (Eq. 11) can recover marginal pseudo-R2 by feature.

Official package is data-only (no runnable code); this is a paper-faithful
reconstruction. See the review's "Replication Reconstruction Notes" for the spec.
"""
from __future__ import annotations

from typing import Any

import macroforecast as mf
from macroforecast.pipeline import Arm, TargetSpec


# Prefix for the dedicated per-target forecast-OBJECT column (the paper's eqs
# fcst0/fcst1 target). The target is built FROM THE RAW LEVEL Y_t in
# run_full._augmented_bundle and assigned an identity t-code so the official panel
# transform passes it through unchanged. This DECOUPLES the forecast target from
# the predictor t-codes and avoids the double-transform that t-coding-then-log_growth
# produced (NaN for HOUST/CPI long horizons). Mirrors the GCLS ``YGROWTH__`` pattern.
YTARGET_PREFIX = "YTARGET__"


def ytarget_column(column: str) -> str:
    """Name of the dedicated forecast-target column for a raw FRED-MD ``column``."""
    return f"{YTARGET_PREFIX}{column}"


# Per-target one-period object built FROM THE RAW LEVEL (run_full._augmented_bundle).
# The h-step forecast TARGET is the horizon-average (direct_average) of this object,
# except T10YFFM which is the level forecast h-ahead (direct, no averaging).
#   "log_diff" -> log Y_t - log Y_{t-1}  (INDPRO, CPIAUCSL, HOUST: I(1) avg growth)
#   "diff"     -> Y_t - Y_{t-1}          (UNRATE: I(1) avg change, no log)
#   "level"    -> Y_t                    (T10YFFM: I(0) spread, forecast the level)
TARGET_KIND: dict[str, str] = {
    "INDPRO": "log_diff",
    "CPIAUCSL": "log_diff",   # inflation = Delta log (NOT the panel's Delta^2-log t-code 6)
    "HOUST": "log_diff",      # re-differenced from level (panel t-code 4 is log level)
    "UNRATE": "diff",
    "T10YFFM": "level",
}


# Section 4 / Appendix F target construction rules (paper eqs fcst0/fcst1).
# Each target forecasts the dedicated ``YTARGET__<col>`` column with transform="value":
#   * direct_average -> target_transform resolves to "average_value" =
#     (1/h) sum_{h'=1..h} object_{t+h'} (the paper's average growth/change rate).
#   * direct         -> target_transform resolves to "value" = object_{t+h} =
#     the level h-ahead (T10YFFM spread; no average, no log, can be negative).
def ml_useful_targets() -> list[TargetSpec]:
    return [
        TargetSpec(ytarget_column("INDPRO"), transform="value", policy="direct_average"),    # I(1) log avg growth
        TargetSpec(ytarget_column("CPIAUCSL"), transform="value", policy="direct_average"),  # I(1) inflation (Delta log)
        TargetSpec(ytarget_column("HOUST"), transform="value", policy="direct_average"),     # I(1) log avg growth (from level)
        TargetSpec(ytarget_column("UNRATE"), transform="value", policy="direct_average"),    # I(1) avg change, no log
        TargetSpec(ytarget_column("T10YFFM"), transform="value", policy="direct"),           # I(0) spread level h-ahead
    ]


def _ardi_features(target: str, predictors: tuple[str, ...], *, p_y: int = 6, n_factors: int = 6, p_f: int = 3):
    """Data-rich ARDI features: target lags + PCA factors of the transformed panel."""
    return mf.feature_engineering.feature_spec(
        target=target,
        predictors=list(predictors),
        lags=None,
        target_lags=tuple(range(0, p_y + 1)),
        feature_steps=[
            mf.feature_engineering.pca_step(name="F_raw", columns=list(predictors), n_components=n_factors, scale=True, include=False),
            mf.feature_engineering.lag_step(name="F", input="F_raw", lags=range(0, p_f + 1), include=True),
        ],
    )


def _ar_features(target: str, *, p_y: int = 6):
    """Data-poor AR features: target lags only."""
    return mf.feature_engineering.feature_spec(
        target=target, predictors=[], lags=None, target_lags=tuple(range(0, p_y + 1))
    )


def _flags(*, regime: str, nonlinear: bool, regularization: str, cv: str, loss: str, kernel: str | None) -> dict[str, Any]:
    return {
        "data_regime": regime, "nonlinear": nonlinear, "regularization": regularization,
        "cv": cv, "loss": loss, "kernel": kernel,
    }


def ml_useful_arms(target: str, predictors: tuple[str, ...], *, subset: str = "core") -> list[Arm]:
    """Build the Table-1 arms for one target. ``subset='core'`` is a representative
    grid spanning the four features; ``subset='full'`` adds the remaining EN/SVR cells."""
    ar = _ar_features(target)
    ardi = _ardi_features(target, predictors)
    arms: list[Arm] = [
        # data-poor benchmark spine
        Arm("AR", model="ar", features=ar,
            metadata=_flags(regime="data-poor", nonlinear=False, regularization="none", cv="ic", loss="L2", kernel=None)),
        Arm("RFAR", model="random_forest", features=ar,
            metadata=_flags(regime="data-poor", nonlinear=True, regularization="none", cv="kfold", loss="L2", kernel=None)),
        Arm("KRRAR", model="kernel_ridge", features=ar,
            metadata=_flags(regime="data-poor", nonlinear=True, regularization="ridge", cv="kfold", loss="L2", kernel="rbf")),
        # data-rich
        Arm("ARDI", model="far", features=ardi,
            metadata=_flags(regime="data-rich", nonlinear=False, regularization="ARDI", cv="ic", loss="L2", kernel=None)),
        Arm("RRARDI", model="ridge", features=ardi,
            metadata=_flags(regime="data-rich", nonlinear=False, regularization="ridge", cv="kfold", loss="L2", kernel=None)),
        Arm("ENARDI", model="elastic_net", features=ardi,
            metadata=_flags(regime="data-rich", nonlinear=False, regularization="elastic_net", cv="kfold", loss="L2", kernel=None)),
        Arm("RFARDI", model="random_forest", features=ardi,
            metadata=_flags(regime="data-rich", nonlinear=True, regularization="ARDI", cv="kfold", loss="L2", kernel=None)),
        Arm("KRRARDI", model="kernel_ridge", features=ardi,
            metadata=_flags(regime="data-rich", nonlinear=True, regularization="ridge", cv="kfold", loss="L2", kernel="rbf")),
    ]
    if subset == "full":
        arms += [
            Arm("SVRARDI_RBF", model="svr", features=ardi,
                metadata=_flags(regime="data-rich", nonlinear=True, regularization="ridge", cv="kfold", loss="eps_insensitive", kernel="rbf")),
            Arm("SVRARDI_Lin", model="linear_svr", features=ardi,
                metadata=_flags(regime="data-rich", nonlinear=False, regularization="ridge", cv="kfold", loss="eps_insensitive", kernel=None)),
        ]
    return arms
