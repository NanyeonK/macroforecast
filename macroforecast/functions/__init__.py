"""Standalone function-op namespace.

Each export is a thin sklearn-style or pure-numeric wrapper around an
internal adapter, allowing partial use of macroforecast operations
without constructing a full recipe.

Cycle 22 POC: ``ridge_fit`` + ``theil_u1`` + ``theil_u2``.
Subsequent cycles will extend to L3 ops, L4 families, L5 metrics,
L6 tests, L7 importance.

Example usage::

    import macroforecast as mf
    import numpy as np
    import pandas as pd

    X = pd.DataFrame({"x1": [1, 2, 3, 4, 5], "x2": [2, 3, 4, 5, 6]})
    y = pd.Series([2.0, 3.0, 4.0, 5.0, 6.0])

    result = mf.functions.ridge_fit(X, y, alpha=0.5)
    print(result.coef_)          # array of coefficients
    print(result.predict(X))     # predictions

    u1 = mf.functions.theil_u1(np.array([1, 2, 3]), np.array([1.5, 2.5, 3.5]))
    print(u1)                    # 0.1044...
"""
from __future__ import annotations

from .ridge import RidgeFitResult, ridge_fit
from .theil_u import theil_u1, theil_u2

__all__ = ["RidgeFitResult", "ridge_fit", "theil_u1", "theil_u2"]
