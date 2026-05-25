"""Public model class namespace for macroforecast.

Exposes 30 promoted L4 model classes as stable public symbols. Each class is
a thin subclass of the corresponding private runtime class (``_<Name>``).
``isinstance`` checks pass for both the public and private name.

Sub-modules
-----------
- ``macroforecast.models.linear``     -- 14 linear/MIDAS/ridge-variant classes
- ``macroforecast.models.bayesian``   -- 3 Bayesian/DFM classes
- ``macroforecast.models.volatility`` -- 2 volatility classes
- ``macroforecast.models.timeseries`` -- 3 time-series classes
- ``macroforecast.models.tree``       -- 6 tree/ensemble/KNN classes
- ``macroforecast.models.neural``     -- 2 neural network classes

Flat re-export
--------------
All 30 classes are importable directly from ``macroforecast.models``::

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
        SlowGrowingTree, QuantileRegressionForest, Bagging, Booging,
        MacroRandomForest, KNN,
        SequenceModel, HemisphereNN,
    )

Cycle 63 -- 22 L4 classes promoted via thin subclassing.
Cycle 64 -- 8 additional classes: tree (6) + neural (2).
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

from .tree import (
    SlowGrowingTree,
    QuantileRegressionForest,
    Bagging,
    Booging,
    MacroRandomForest,
    KNN,
)

from .neural import (
    SequenceModel,
    HemisphereNN,
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
    # tree.py (6) -- C64
    "SlowGrowingTree",
    "QuantileRegressionForest",
    "Bagging",
    "Booging",
    "MacroRandomForest",
    "KNN",
    # neural.py (2) -- C64
    "SequenceModel",
    "HemisphereNN",
]
