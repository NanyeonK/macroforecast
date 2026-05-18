# Standalone functions — L7 feature importance

L7 provides model interpretation and feature importance. In the standalone
paradigm these are planned as:

```python
mf.functions.<op>(fit_result, X, y, **kwargs) -> ImportanceResult
```

An `ImportanceResult` carries `.importances_mean`, `.importances_std`,
`.feature_names_in_` (if available), and a `.plot()` method.

> **Cycle 22 note** — L7 standalone callables are planned for a future cycle.
> This page documents 8 grouped operational ops. The encyclopedia link at the
> bottom covers the full ~30-op L7 surface including SHAP family, IRF/FEVD,
> and attribution ops.

## Native importance (2 ops)

| Op | One-liner | Encyclopedia |
|---|---|---|
| `model_native_linear_coef` | Signed coefficients from linear models | [op axis](../encyclopedia/l7/axes/op.md#model-native-linear-coef) |
| `model_native_tree_importance` | Gini / split-count importance from tree models | [op axis](../encyclopedia/l7/axes/op.md#model-native-tree-importance) |

## Permutation importance (2 ops)

| Op | One-liner | Encyclopedia |
|---|---|---|
| `permutation_importance` | Model-agnostic permutation feature importance (Breiman 2001) | [op axis](../encyclopedia/l7/axes/op.md#permutation-importance) |
| `permutation_importance_strobl` | Strobl (2008) conditional permutation importance (bias-corrected) | [op axis](../encyclopedia/l7/axes/op.md#permutation-importance-strobl) |

## Dependence (2 ops)

| Op | One-liner | Encyclopedia |
|---|---|---|
| `partial_dependence` | Friedman (2001) marginal effect curves | [op axis](../encyclopedia/l7/axes/op.md#partial-dependence) |
| `accumulated_local_effect` | ALE plots (Apley & Zhu 2020), de-aliased | [op axis](../encyclopedia/l7/axes/op.md#accumulated-local-effect) |

## SHAP (2 ops)

| Op | One-liner | Encyclopedia |
|---|---|---|
| `shap_linear` | SHAP values via linear explainer (no extra deps) | [op axis](../encyclopedia/l7/axes/op.md#shap-linear) |
| `shap_tree` | SHAP values via tree explainer (requires `macroforecast[shap]`) | [op axis](../encyclopedia/l7/axes/op.md#shap-tree) |

## Quick example (recipe DSL)

```yaml
7_interpretation:
  enabled: true
  fixed_axes:
    op: permutation_importance
    top_k_features_to_show: 10
    figure_type: bar_global
    figure_format: png
```

## Quick example (standalone — planned)

```python
import macroforecast as mf
import numpy as np

rng = np.random.RandomState(42)
X = rng.randn(120, 6)
y = X[:, 0] * 2 + X[:, 3] * -1 + 0.3 * rng.randn(120)

fit = mf.functions.ridge_fit(X, y, alpha=0.5)

# Planned standalone call (future cycle):
# imp = mf.functions.permutation_importance(fit, X, y, n_repeats=30, random_state=42)
# print(imp.importances_mean)
```

## Related

- [L4 fit](l4_fit.md) — the fit result is the primary input to L7 ops.
- [L5 metrics](l5_metrics.md) — evaluation context for importance analysis.
- [Encyclopedia L7 op axis](../encyclopedia/l7/axes/op.md) — full per-op
  reference including SHAP family (kernel / deep / interaction), BVAR PIP,
  IRF / FEVD / generalized IRF, forecast decomposition, and lineage attribution.
