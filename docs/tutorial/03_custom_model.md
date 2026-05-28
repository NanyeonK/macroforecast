# Build your own model

Build a custom forecasting model by subclassing `sklearn.base.BaseEstimator` and
`RegressorMixin`, use it directly in a `TimeSeriesSplit` loop, and optionally
register it with the recipe pipeline. No YAML is required for the main body of
this tutorial.

---

## The sklearn subclassing contract

macroforecast model classes in `macroforecast.models` all follow the sklearn
estimator convention. Any class that inherits from `sklearn.base.BaseEstimator`
and `sklearn.base.RegressorMixin` can be used in the same way.

`BaseEstimator` provides `get_params()` and `set_params()` automatically, derived
from the argument names in `__init__`. These methods are required for sklearn
pipeline compatibility, including `clone()` and `GridSearchCV`. No extra
implementation is needed beyond writing `__init__` in the standard way.

`RegressorMixin` provides a default `score(X, y)` method that computes the R^2
coefficient of determination, which makes the class usable directly with sklearn
scoring utilities.

The only contract that macroforecast requires to treat the class as a model is
the pair `fit(X, y) -> self` and `predict(X) -> np.ndarray`. If your class
satisfies those two methods and inherits from `BaseEstimator` and `RegressorMixin`,
it is compatible with the standalone `TimeSeriesSplit` loop, sklearn pipelines,
and the recipe registration API.

---

## A concrete example -- ConstantTrendPlusAR

The class below fits a linear time trend plus an AR(1) residual correction by OLS.
It accepts `X` in both `fit` and `predict` to satisfy the sklearn convention, but
does not use feature columns. The `X` argument serves only to convey the row count
at predict time.

```python
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


class ConstantTrendPlusAR(BaseEstimator, RegressorMixin):
    """Linear time trend plus AR(1) residual forecaster.

    Fits the model y_t = alpha + beta*t + phi*y_{t-1} + eps_t by OLS.
    At predict time, extrapolates the trend and adds the AR(1) correction.

    Parameters
    ----------
    fit_intercept : bool, default True
        Whether to include a constant term in the design matrix.

    Notes
    -----
    X is accepted in fit and predict to satisfy the sklearn estimator
    convention, but is not used in estimation or forecasting.
    """

    def __init__(self, fit_intercept: bool = True) -> None:
        self.fit_intercept = fit_intercept

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ConstantTrendPlusAR":
        n = len(y)
        t = np.arange(n, dtype=float)
        y_vals = y.values

        # Design matrix: [1, t, y_{t-1}] or [t, y_{t-1}] without intercept
        ar_lag = np.concatenate([[y_vals[0]], y_vals[:-1]])
        if self.fit_intercept:
            Z = np.column_stack([np.ones(n), t, ar_lag])
        else:
            Z = np.column_stack([t, ar_lag])

        # OLS via least-squares solver
        self.coef_, _, _, _ = np.linalg.lstsq(Z, y_vals, rcond=None)
        self._n_train = n
        self._last_y = float(y_vals[-1])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # X is used only to determine the number of forecast periods
        n_test = len(X)
        t_future = np.arange(self._n_train, self._n_train + n_test, dtype=float)
        preds = []
        last_y = self._last_y

        for i, t_i in enumerate(t_future):
            if self.fit_intercept:
                z = np.array([1.0, t_i, last_y])
            else:
                z = np.array([t_i, last_y])
            yhat = float(z @ self.coef_)
            preds.append(yhat)
            last_y = yhat   # AR(1) recursion: each period feeds the next

        return np.array(preds)
```

---

## Use the custom class directly

The class works immediately in a `TimeSeriesSplit` loop, without any registration
or YAML. The synthetic data from {doc}`02_full_study` is reused here.

```python
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import pandas as pd

# y and X are the same series from Tutorial 02
tscv = TimeSeriesSplit(n_splits=5)
mse_list = []

for train_idx, test_idx in tscv.split(y):
    X_tr = X.iloc[train_idx]   # ConstantTrendPlusAR does not use columns,
    X_te = X.iloc[test_idx]    # but len(X_te) sets the forecast horizon.
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

    m = ConstantTrendPlusAR(fit_intercept=True)
    m.fit(X_tr, y_tr)
    preds = m.predict(X_te)
    mse_list.append(np.mean((preds - y_te.values) ** 2))

print(f"ConstantTrendPlusAR mean CV MSE: {np.mean(mse_list):.4f}")
```

Because `ConstantTrendPlusAR` inherits from `BaseEstimator`, sklearn utilities
such as `clone()` and `check_estimator()` work out of the box. The `get_params()`
method returns `{"fit_intercept": True}` automatically, derived from the `__init__`
argument name.

---

## Inheriting macroforecast machinery (optional)

If your custom model is a variant of an existing macroforecast model, you
can subclass both the private implementation class and the sklearn mixins. The 30
public model classes in `mf.models` follow exactly this pattern. For example,
`LinearAR` is defined as `class LinearAR(_LinearARModel): pass`. A custom variant
that adds a constraint would look like:

```python
class MyLinearAR(_LinearARModel, BaseEstimator, RegressorMixin):
    pass
```

This preserves the private class behavior while gaining `get_params`, `set_params`,
and `score`. In practice, if you only need minor modifications, subclassing the
private class is the most direct path. If you need full control, start from
`BaseEstimator` and `RegressorMixin` directly, as in `ConstantTrendPlusAR` above.

---

## When to register for the recipe pipeline

If you want the recipe runtime to dispatch your class by name in a YAML recipe,
register it as a functional wrapper via `macroforecast.custom.register_model`. The
registration API accepts a callable with the signature
`fn(X_train, y_train, X_test, context) -> scalar`, where `X_test` is a single
prediction-period row.

```python
import macroforecast.custom as mf_custom

def constant_trend_plus_ar(X_train, y_train, X_test, context):
    """Functional wrapper for ConstantTrendPlusAR for recipe dispatch.

    The recipe runtime calls this once per forecast origin per horizon.
    X_test is one row; the return value is a single forecast scalar.
    """
    model = ConstantTrendPlusAR(fit_intercept=True)
    # Wrap inputs as DataFrames if the runtime passes numpy arrays
    X_tr = pd.DataFrame(X_train) if not isinstance(X_train, pd.DataFrame) else X_train
    y_tr = pd.Series(y_train) if not isinstance(y_train, pd.Series) else y_train
    X_te = pd.DataFrame(X_test) if not isinstance(X_test, pd.DataFrame) else X_test
    model.fit(X_tr, y_tr)
    return float(model.predict(X_te)[0])

mf_custom.register_model("constant_trend_plus_ar", constant_trend_plus_ar)
```

After registration, the name `"constant_trend_plus_ar"` is accepted as a
`model_family` value in any recipe running in the same Python process. For
full details on the registration contract and context fields, see
{doc}`../how_to/add_custom_model`.
