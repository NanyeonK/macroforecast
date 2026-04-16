# Importance Methods — Mathematical Background

## SHAP (SHapley Additive exPlanations)

SHAP values decompose a prediction into per-feature contributions based on Shapley values from cooperative game theory.

For a model $f$ and feature values $x$, the SHAP value of feature $j$ is:

$$\phi_j(f, x) = \sum_{S \subseteq N \setminus \{j\}} \frac{|S|!(|N|-|S|-1)!}{|N|!} [f(x_S \cup x_j) - f(x_S)]$$

where $N$ is the set of all features, and $f(x_S)$ is the expected prediction when only features in $S$ are "present."

### TreeSHAP
Exact computation for tree-based models in O(TLD^2) time, where T = number of trees, L = max leaves, D = max depth.

Reference: Lundberg et al. (2020)

### LinearSHAP
For linear models $f(x) = \beta_0 + \sum \beta_j x_j$, SHAP values are exact:

$$\phi_j = \beta_j (x_j - E[x_j])$$

### KernelSHAP
Model-agnostic approximation using weighted linear regression on binary coalition vectors.

Reference: Lundberg & Lee (2017)

## Permutation Feature Importance (PFI)

$$\text{PFI}_j = \frac{1}{K} \sum_{k=1}^{K} [L(\tilde{y}^{(k)}_j) - L(\hat{y})]$$

where $\tilde{y}^{(k)}_j$ is the prediction when feature $j$ is randomly permuted in the $k$-th repetition.

Reference: Breiman (2001)

## Partial Dependence Plot (PDP)

$$\hat{f}_j(x_j) = \frac{1}{n} \sum_{i=1}^{n} f(x_j, x_{-j}^{(i)})$$

The PDP shows the average effect of feature $x_j$ on predictions, marginalizing over all other features.

## Accumulated Local Effects (ALE)

$$\hat{f}_{j,ALE}(x_j) = \int_{x_{j,min}}^{x_j} E\left[\frac{\partial f(X)}{\partial x_j} \Big| x_j = z\right] dz$$

In practice, computed via finite differences over quantile bins. Unlike PDP, ALE is unbiased with correlated features.

Reference: Apley & Zhu (2020)

## LIME

Fit a local linear model $g$ that approximates $f$ near a point $x^*$:

$$g = \arg\min_{g \in G} L(f, g, \pi_{x^*}) + \Omega(g)$$

where $\pi_{x^*}$ is a proximity kernel and $\Omega(g)$ penalizes complexity.

Reference: Ribeiro et al. (2016)

**See also:** [User Guide: Importance](../user_guide/importance.md) — when to use each method
