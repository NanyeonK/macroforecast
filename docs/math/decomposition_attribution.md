# Decomposition attribution

Mathematical definition of the Phase 7 attribution.

## Setup

Let a completed sweep have $V$ variants indexed $i = 1, \ldots, V$, each
evaluated on $H$ forecast horizons. Let $y_{i,h}$ be the variant's primary
metric (default `msfe`) on horizon $h$. Aggregate-level observations are
the pairs $(i, h)$; there are $N = V \cdot H$ of them.

The grand mean is

$$\bar{y} = \frac{1}{N}\sum_{i,h} y_{i,h}$$

Total sum of squares:

$$\mathrm{SS}_{\mathrm{total}} = \sum_{i,h}\bigl(y_{i,h} - \bar{y}\bigr)^2$$

## One-way ANOVA per axis

For an axis $a$ with $K$ distinct values across the sweep, partition
observations by their axis value:

$$\mathrm{SS}_{\mathrm{between}}^{(a)} = \sum_{k=1}^{K} n_k \bigl(\bar{y}_k - \bar{y}\bigr)^2$$

where $n_k$ is the number of observations with axis value $k$ and
$\bar{y}_k$ is the within-group mean.

The axis share is

$$\mathrm{share}^{(a)} = \frac{\mathrm{SS}_{\mathrm{between}}^{(a)}}{\mathrm{SS}_{\mathrm{total}}}$$

For significance, `macrocast` computes the standard F-statistic

$$F = \frac{\mathrm{SS}_{\mathrm{between}}/(K-1)}{\mathrm{SS}_{\mathrm{within}}/(N-K)}$$

and reports the survival-function p-value from scipy's F-distribution when
$K \geq 2$ and $\mathrm{SS}_{\mathrm{within}} > 0$. Degenerate cases yield
`significance_p = NaN`.

## Component aggregation

Each registry axis carries a `component` tag (`preprocessing`,
`nonlinearity`, …, `None`). For a component $c$:

$$\mathrm{share}_c = \frac{1}{\mathrm{SS}_{\mathrm{total}}} \sum_{a \in c} \mathrm{SS}_{\mathrm{between}}^{(a)}$$

Shares across components do **not** sum to 1 because residual variance
(within-group + axes tagged `None` + interactions) is not attributed.

## Zero-variance edge cases

- $\mathrm{SS}_{\mathrm{total}} = 0$: all shares are defined as $0$ and
  significance is `NaN`.
- A single axis value across the whole sweep ($K = 1$):
  $\mathrm{SS}_{\mathrm{between}} = 0$ and significance is `NaN`.

## Relationship to Shapley (v1.1)

One-way ANOVA attributes each axis independently; it ignores
*interactions* between axes. Shapley attribution averages each axis's
marginal contribution over all $2^A$ subsets of axes, correctly handling
interactions but requiring exponentially more configurations.

ADR-002 chooses ANOVA as the v1.0 baseline on the grounds that

1. it runs in $O(A \cdot N)$ time,
2. interaction-free attribution is a common reviewer ask, and
3. the Shapley extension plugs into the same engine by swapping
   `attribution_method="shapley"` — data layout and schema do not change.
