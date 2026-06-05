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


# Section 4 / Appendix F target construction rules (review lines 212-221).
def ml_useful_targets() -> list[TargetSpec]:
    return [
        TargetSpec("INDPRO", transform="log_growth", policy="direct_average"),   # I(1) log avg growth
        TargetSpec("CPIAUCSL", transform="log_growth", policy="direct_average"), # I(1) price treatment (paper)
        TargetSpec("HOUST", transform="log_growth", policy="direct_average"),    # tcode-4 override per paper
        TargetSpec("UNRATE", transform="change", policy="direct_average"),       # I(1) avg change, no log
        TargetSpec("T10YFFM", transform="level", policy="direct"),               # stationary spread level
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
