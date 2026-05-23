"""Public model class namespace for macroforecast.

Exposes 22 promoted L4 model classes as stable public symbols. Each class is
a thin subclass of the corresponding private runtime class (``_<Name>``).
``isinstance`` checks pass for both the public and private name.

Sub-modules
-----------
- ``macroforecast.models.linear``     -- 14 linear/MIDAS/ridge-variant classes
- ``macroforecast.models.bayesian``   -- 3 Bayesian/DFM classes
- ``macroforecast.models.volatility`` -- 2 volatility classes
- ``macroforecast.models.timeseries`` -- 3 time-series classes

Flat re-export
--------------
All 22 classes are importable directly from ``macroforecast.models``::

    from macroforecast.models import (
        MidasAlmon, MidasBeta, MidasStep, UnrestrictedMidas,
        LinearAR, FactorAugmentedAR,
        NonNegRidge, TwoStageRandomWalkRidge, ShrinkToTargetRidge,
        FusedDifferenceRidge,
        PrincipalComponentRegression, FactorAugmentedVAR,
        VAR, GLMBoost,
        BVAR, BVARMinnesota, DFMMixedFrequency,
        GARCH, RealizedGARCH,
        ETS, Theta, HoltWinters,
    )

Cycle 63 -- 22 L4 classes promoted via thin subclassing.
"""
from __future__ import annotations

from .linear import (
    MidasAlmon,
    MidasBeta,
    MidasStep,
    UnrestrictedMidas,
    LinearAR,
    FactorAugmentedAR,
    NonNegRidge,
    TwoStageRandomWalkRidge,
    ShrinkToTargetRidge,
    FusedDifferenceRidge,
    PrincipalComponentRegression,
    FactorAugmentedVAR,
    VAR,
    GLMBoost,
)

from .bayesian import (
    BVAR,
    BVARMinnesota,
    DFMMixedFrequency,
)

from .volatility import (
    GARCH,
    RealizedGARCH,
)

from .timeseries import (
    ETS,
    Theta,
    HoltWinters,
)

__all__ = [
    # linear.py (14)
    "MidasAlmon",
    "MidasBeta",
    "MidasStep",
    "UnrestrictedMidas",
    "LinearAR",
    "FactorAugmentedAR",
    "NonNegRidge",
    "TwoStageRandomWalkRidge",
    "ShrinkToTargetRidge",
    "FusedDifferenceRidge",
    "PrincipalComponentRegression",
    "FactorAugmentedVAR",
    "VAR",
    "GLMBoost",
    # bayesian.py (3)
    "BVAR",
    "BVARMinnesota",
    "DFMMixedFrequency",
    # volatility.py (2)
    "GARCH",
    "RealizedGARCH",
    # timeseries.py (3)
    "ETS",
    "Theta",
    "HoltWinters",
]
