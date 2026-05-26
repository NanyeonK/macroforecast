"""Public model class namespace for macroforecast.

Exposes 30 promoted L4 model classes as stable public symbols. Each class is
a thin subclass of the corresponding private runtime class (``_<Name>``).
``isinstance`` checks pass for both the public and private name.

Sub-modules
-----------
- ``macroforecast.layers.l4_models.linear``     -- 14 linear/MIDAS/ridge-variant classes
- ``macroforecast.layers.l4_models.bayesian``   -- 3 Bayesian/DFM classes
- ``macroforecast.layers.l4_models.volatility`` -- 2 volatility classes
- ``macroforecast.layers.l4_models.timeseries`` -- 3 time-series classes
- ``macroforecast.layers.l4_models.tree``       -- 6 tree/ensemble/KNN classes
- ``macroforecast.layers.l4_models.neural``     -- 2 neural network classes

Flat re-export
--------------
All 30 classes are importable directly from ``macroforecast.layers.l4_models``::

    from macroforecast.layers.l4_models import (
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

v0.9.5 -- 22 L4 classes promoted via thin subclassing.
v0.9.5 -- 8 additional classes: tree (6) + neural (2).
"""
from __future__ import annotations

from importlib import import_module
from typing import Any

# Map each public symbol to the submodule that defines it.
# Imports are deferred to avoid circular imports with macroforecast.core.
_SYMBOL_MODULE: dict[str, str] = {
    # linear.py (14)
    "MidasAlmon": ".linear",
    "MidasBeta": ".linear",
    "MidasStep": ".linear",
    "UnrestrictedMidas": ".linear",
    "LinearAR": ".linear",
    "FactorAugmentedAR": ".linear",
    "NonNegRidge": ".linear",
    "TwoStageRandomWalkRidge": ".linear",
    "ShrinkToTargetRidge": ".linear",
    "FusedDifferenceRidge": ".linear",
    "PrincipalComponentRegression": ".linear",
    "FactorAugmentedVAR": ".linear",
    "VAR": ".linear",
    "GLMBoost": ".linear",
    # bayesian.py (3)
    "BVAR": ".bayesian",
    "BVARMinnesota": ".bayesian",
    "DFMMixedFrequency": ".bayesian",
    # volatility.py (2)
    "GARCH": ".volatility",
    "RealizedGARCH": ".volatility",
    # timeseries.py (3)
    "ETS": ".timeseries",
    "Theta": ".timeseries",
    "HoltWinters": ".timeseries",
    # tree.py (6) -- v0.9.5
    "SlowGrowingTree": ".tree",
    "QuantileRegressionForest": ".tree",
    "Bagging": ".tree",
    "Booging": ".tree",
    "MacroRandomForest": ".tree",
    "KNN": ".tree",
    # neural.py (2) -- v0.9.5
    "SequenceModel": ".neural",
    "HemisphereNN": ".neural",
}

__all__ = list(_SYMBOL_MODULE)


def __getattr__(name: str) -> Any:
    if name in _SYMBOL_MODULE:
        mod = import_module(_SYMBOL_MODULE[name], __name__)
        obj = getattr(mod, name)
        globals()[name] = obj
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_SYMBOL_MODULE))
