# Statistical Tests — Mathematical Background

## Diebold-Mariano (DM) test

Let $d_t = L(e_t^m) - L(e_t^b)$ be the loss differential, where $L(\cdot)$ is typically the squared loss.

**Test statistic:**

$$DM = \frac{\bar{d}}{\hat{\sigma}_{\bar{d}}} \xrightarrow{d} N(0,1)$$

where $\bar{d} = T^{-1}\sum d_t$ and $\hat{\sigma}_{\bar{d}}$ is the HAC standard error (Newey-West with bandwidth $h-1$ for $h$-step forecasts).

**H0:** $E[d_t] = 0$ (equal predictive ability)

Reference: Diebold & Mariano (1995)

## Harvey-Leybourne-Newbold (DM-HLN) correction

$$DM_{HLN} = DM \cdot \left[\frac{T + 1 - 2h + h(h-1)/T}{T}\right]^{1/2}$$

with a $t(T-1)$ critical value instead of normal.

Reference: Harvey, Leybourne & Newbold (1997)

## Clark-West (CW) test

For nested models, let $\hat{f}_t = (e_t^b)^2 - [(e_t^m)^2 - (\hat{y}_t^m - \hat{y}_t^b)^2]$.

**Test statistic:**

$$CW = \frac{\bar{f}}{\hat{\sigma}_{\bar{f}}} \xrightarrow{d} N(0,1)$$

**H0:** The larger model does not improve forecast accuracy.

One-sided test: reject H0 if CW > z_alpha (model improves).

Reference: Clark & West (2007)

## Model Confidence Set (MCS)

Given $M_0$ models, MCS iteratively eliminates the worst model until the surviving set is not significantly different from the best:

1. Test $H_0$: all models in current set have equal predictive ability (using max-$t$ or range statistic)
2. If rejected, remove the model with the worst relative performance
3. Repeat until $H_0$ is not rejected

The surviving models at significance level $\alpha$ form the $(1-\alpha)$ Model Confidence Set.

Reference: Hansen, Lunde & Nason (2011)

## Giacomini-White Conditional Predictive Ability (CPA)

Tests whether the loss differential $d_t$ is predictable by observable variables $h_t$:

$$d_t = h_t' \delta + u_t$$

**H0:** $\delta = 0$ (loss differential is unpredictable)

**Test:** Wald test on $\hat{\delta}$.

Reference: Giacomini & White (2006)

## Mincer-Zarnowitz regression

$$y_t = \alpha + \beta \hat{y}_t + \varepsilon_t$$

**H0:** $\alpha = 0, \beta = 1$ (forecast is optimal: unbiased and efficient)

Joint F-test.

Reference: Mincer & Zarnowitz (1969)

## Pesaran-Timmermann directional accuracy test

Tests whether the forecast correctly predicts the sign of changes more often than chance:

$$PT = \frac{\hat{P} - \hat{P}^*}{\sqrt{\text{Var}(\hat{P} - \hat{P}^*)}} \xrightarrow{d} N(0,1)$$

where $\hat{P}$ is the observed proportion of correct sign predictions and $\hat{P}^*$ is the expected proportion under independence.

Reference: Pesaran & Timmermann (1992)

**See also:** [User Guide: Statistical Tests](../user_guide/stat_tests.md) — when to use each test
