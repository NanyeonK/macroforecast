# How to select features with Boruta

The goal is to select a subset of features under proper false-positive
control using the Kursa-Rudnicki (2010) Boruta algorithm. We use the
public class `Boruta` from `macroforecast.feature_selection`, which
follows the sklearn `fit` and `transform` interface and inherits from
`BaseEstimator` and `TransformerMixin` since C64. The implementation is
pure numpy and sklearn, so no external `boruta` package is required.

## Setup

```python
import numpy as np
import pandas as pd
from macroforecast.feature_selection import Boruta

rng = np.random.RandomState(0)
n, p = 300, 20
X = pd.DataFrame(rng.randn(n, p), columns=[f"x{i}" for i in range(p)])

# Only x0, x1, x2 are truly relevant; the remaining 17 are pure noise.
y = X["x0"] + 0.7 * X["x1"] - 0.5 * X["x2"] + 0.3 * rng.randn(n)
y = pd.Series(y, name="y")
```

## Fit and transform

```python
selector = Boruta(
    n_estimators_rf=100,
    max_iter=100,
    alpha=0.05,
    include_tentative=False,
    n_shadow_copies=6,
    random_state=0,
)
selector.fit(X, y)

X_selected = selector.transform(X)
print(selector.selected_features_)
```

The Bonferroni significance level `alpha=0.05` and the multi-shadow
calibration `n_shadow_copies=6` jointly control the family-wise
false-positive rate. The C59 fix removed the always-accept-one fallback,
so an empty `selected_features_` list is a legitimate outcome under a
null DGP.

## Output

After `fit`, the estimator exposes `selected_features_`
(`list[str]`), `feature_names_in_` (a numpy object array), and
`n_features_in_` (int). The call `transform(X)` returns a
`pd.DataFrame` with only the selected columns, and `fit_transform(X, y)`
chains the two calls.

## Common pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| `selected_features_ == []` on real data | Genuine null DGP, or weak signal relative to noise | Increase `n_estimators_rf` or `max_iter`, and verify the target has measurable signal |
| `NotFittedError` on `transform` | Called before `fit` | Call `selector.fit(X, y)` first |
| Selection differs between runs | `random_state` not set, or upstream RNG state shifted | Pin `random_state=0` and serialize the upstream RNG state |
| High false-positive rate | `n_shadow_copies` set too low | Keep the default `n_shadow_copies=6` or higher, since C59 verified that values below 6 inflate the FP rate |

## See also

- {doc}`tune_hyperparameters`
- Paper: Kursa and Rudnicki (2010), Journal of Statistical Software 36(11).
