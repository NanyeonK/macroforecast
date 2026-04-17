# Preprocessing separation rule (Phase 3)

The `separation_rule` axis controls the leak discipline between train and test partitions during preprocessing. It is independent of `preprocess_fit_scope` (which decides *where* the fit is performed) — `separation_rule` decides *what data the fit is permitted to see*.

| Mode | Fit data | Use case |
|---|---|---|
| `strict_separation` (default) | `X_train` only | Operational forecasts; eliminates look-ahead bias |
| `shared_transform_then_split` | `X_train ∪ X_test` | Reproducing legacy code that fits on the whole panel |
| `joint_preprocessor` | User-supplied joint pipeline | Paper replication where authors describe a joint scheme |
| `target_only_transform` | `y_train` only, X passthrough | Robust target normalisation experiments |
| `X_only_transform` | `X_train` only, y passthrough | Feature normalisation without touching y |

## API

```python
from macrocast.preprocessing.separation import apply_separation_rule, LeakError
from sklearn.preprocessing import StandardScaler

X_train_t, X_test_t, y_train_t, y_test_t = apply_separation_rule(
    rule="strict_separation",
    X_train=X_train, X_test=X_test,
    y_train=y_train, y_test=y_test,
    preprocessor=StandardScaler(),
)
```

`strict_separation` enforces two invariants:
1. `preprocessor.fit` is deterministic on `X_train` alone — non-determinism raises `LeakError`.
2. `X_train` is not mutated during fit — mutation raises `LeakError`.

`joint_preprocessor` requires a non-`None` preprocessor; passing `None` raises `LeakError` with a clear message.

## YAML usage

```yaml
preprocessing:
  separation_rule: strict_separation
  preprocess_fit_scope: train_only
  scaling_policy: standard
```

To replicate a paper that fits PCA on the full panel:

```yaml
preprocessing:
  separation_rule: shared_transform_then_split
  preprocess_fit_scope: train_only  # ignored when rule shares data
  scaling_policy: standard
  dimensionality_reduction_policy: pca
```

## When to deviate from `strict_separation`

Only when explicitly replicating a paper or stress-testing leakage sensitivity. Sweep recipes that include both `strict_separation` and `shared_transform_then_split` quantify the bias from leakage on a per-recipe basis — this is one of the canonical macrocast horse races.
