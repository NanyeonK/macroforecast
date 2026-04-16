# Feature Importance

macrocast supports 12 importance methods. This guide helps you choose the right method.

## Decision flowchart

1. **Tree model (RF, XGBoost, LightGBM)?** Use `tree_shap` (exact, fast)
2. **Linear model (Ridge, Lasso)?** Use `linear_shap` (exact, fast) or `linear_coefficients`
3. **Any model, global importance?** Use `permutation_importance`
4. **Want to explain individual predictions?** Use `lime` or `kernel_shap`
5. **Want partial effects of a feature?** Use `pdp`, `ice`, or `ale`
6. **Features have natural groups (FRED categories)?** Use `grouped_permutation`
7. **Want to check stability of rankings?** Use `importance_stability`

## SHAP family

### `tree_shap` — TreeSHAP
- **Compatible models:** `randomforest`, `extratrees`, `gbm`, `xgboost`, `lightgbm`, `catboost`
- **How it works:** Exact Shapley values using tree structure. O(TLD) complexity.
- **Advantage:** Exact, fast, theoretically grounded
- **Output:** Per-feature mean |SHAP value| across all predictions
- **Reference:** Lundberg & Lee (2017), Lundberg et al. (2020)

### `kernel_shap` — KernelSHAP
- **Compatible models:** Any (model-agnostic)
- **How it works:** Approximate Shapley values by sampling coalitions and fitting weighted linear model
- **Advantage:** Works with any black-box model
- **Disadvantage:** Slower than TreeSHAP, approximate
- **Use when:** Model is not tree-based (MLP, SVR_rbf)

### `linear_shap` — LinearSHAP
- **Compatible models:** `ridge`, `lasso`, `elasticnet`, `bayesianridge`, `ols`, `huber`, `adaptivelasso`, `svr_linear`
- **How it works:** Exact Shapley values for linear models using coefficient decomposition
- **Advantage:** Exact, very fast
- **Use when:** Model is linear and you want theoretically exact importance

## Model-agnostic

### `permutation_importance` — Permutation Feature Importance (PFI)
- **Compatible models:** Any
- **How it works:** Randomly permute each feature, measure increase in prediction error
- **Advantage:** Simple, intuitive, model-agnostic
- **Disadvantage:** Can be misleading with correlated features
- **Reference:** Breiman (2001)

### `feature_ablation` — Feature Ablation
- **Compatible models:** Any
- **How it works:** Replace each feature with its mean value, measure change in predictions
- **Similar to PFI** but deterministic (no randomness)

## Local surrogate

### `lime` — LIME (Local Interpretable Model-Agnostic Explanations)
- **Compatible models:** Any
- **How it works:** Fit a local linear model around a specific prediction point
- **Advantage:** Explains individual predictions, model-agnostic
- **Use when:** You want to understand WHY a specific forecast was high/low
- **Reference:** Ribeiro et al. (2016)

## Partial dependence

### `pdp` — Partial Dependence Plot
- **How it works:** Average model prediction as one feature varies, marginalizing over others
- **Shows:** Average effect of a feature on predictions
- **Limitation:** Can be misleading with correlated features

### `ice` — Individual Conditional Expectation
- **How it works:** Like PDP but shows individual curves for each observation
- **Shows:** Heterogeneity in feature effects across observations
- **Use when:** You suspect the effect of a feature varies across the sample

### `ale` — Accumulated Local Effects
- **How it works:** Measures local effect of a feature using conditional differences
- **Advantage over PDP:** Unbiased even with correlated features
- **Use when:** Features are correlated (common in macro panels)
- **Reference:** Apley & Zhu (2020)

## Grouped

### `grouped_permutation` — Grouped Permutation Importance
- **How it works:** Permute entire groups of features together (e.g., all "labor market" variables)
- **Use when:** Features have natural economic categories (FRED-MD categories: output, labor, housing, prices, ...)
- **Advantage:** Answers "how important is the labor market block?" not just individual variables

## Stability

### `importance_stability` — Bootstrap Importance Stability
- **How it works:** Repeat importance computation across bootstrap samples, measure rank stability
- **Reports:** Coefficient of variation of importance ranks, sign consistency
- **Use when:** You want to know if the top-5 variables are robustly important or just noise

## Compatibility matrix

| Method | Linear | Tree | SVR_rbf | MLP | Speed |
|--------|:------:|:----:|:-------:|:---:|:-----:|
| `tree_shap` | - | Yes | - | - | Fast |
| `kernel_shap` | Yes | Yes | Yes | Yes | Slow |
| `linear_shap` | Yes | - | - | - | Fast |
| `permutation_importance` | Yes | Yes | Yes | Yes | Medium |
| `feature_ablation` | Yes | Yes | Yes | Yes | Medium |
| `lime` | Yes | Yes | Yes | Yes | Medium |
| `pdp` | Yes | Yes | Yes | Yes | Slow |
| `ice` | Yes | Yes | Yes | Yes | Slow |
| `ale` | Yes | Yes | Yes | Yes | Medium |
| `grouped_permutation` | Yes | Yes | Yes | Yes | Medium |
| `importance_stability` | Yes | Yes | Yes | Yes | Slow |
| `minimal_importance` | Yes* | Yes* | - | - | Fast |

*`minimal_importance` = coefficient magnitude (linear) or feature_importances_ (tree)

**See also:**
- [Mathematical Background: Importance Methods](../math/importance_methods.md) — formal definitions
- [Example: Importance Gallery](../examples/importance_gallery.md) — runnable code
- [User Guide: Models](models.md) — model compatibility details
