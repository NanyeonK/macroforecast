# `pi_correction`

[Back to L4](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``pi_correction`` on sub-layer ``L4_E_predict`` (layer ``l4``).

## Sub-layer

**L4_E_predict**

## Axis metadata

- Default: `'none'`
- Sweepable: True
- Status: operational

## Operational status summary

- Operational: 2 option(s)
- Future: 0 option(s)

## Options

### `none`  --  operational

No PI correction; standard Gaussian-residual sigma.

Default predict-op behaviour: prediction-interval bands derive from the fitted family's residual variance σ²_ε (Gaussian approximation around the point forecast). This treats factor regressors and parameter estimates as if they were observed exactly. Appropriate for non-factor-augmented families (OLS, ridge, AR_p, etc.) or when factor estimation noise is negligible relative to residual variance.

**When to use**

Default for any family that does not estimate latent factors as regressors -- the residual-variance band is honest in that case.

**When NOT to use**

Factor-augmented forecasts where estimated factors enter the regression -- use ``bai_ng`` to inflate the band for the factor-estimation noise.

**References**

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'
* Bai & Ng (2006) 'Confidence Intervals for Diffusion Index Forecasts and Inference for Factor-Augmented Regressions', Econometrica 74(4): 1133-1150.

**Related options**: [`bai_ng`](#bai-ng)

_Last reviewed 2026-05-04 by macroforecast author._

### `bai_ng`  --  operational

Bai-Ng (2006) generated-regressor PI correction.

Activates the Bai-Ng (2006) Theorem 3 + Corollary 1 correction to the prediction-interval sigma. The corrected sigma reflects (a) factor-estimation noise V₂/N where V₂ = β̂_F^T (Λ̂ diag(Σ̂_e) Λ̂^T / N) β̂_F, (b) parameter-estimation noise V₁/T from the OLS coefficient covariance evaluated at the last training factor row, and (c) the residual variance σ²_ε. Active only when the upstream fitted family is ``factor_augmented_ar``; for any other family the predict op falls through to the uncorrected Gaussian-residual sigma.

**When to use**

Factor-augmented forecasts (FAR / FAVAR-style) where the band should be honest about factor-estimation noise on top of the usual parameter and residual uncertainty.

**When NOT to use**

Non-factor families -- the correction is a no-op there. Use ``none`` to keep the predict op's default behaviour.

**References**

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'
* Bai & Ng (2006) 'Confidence Intervals for Diffusion Index Forecasts and Inference for Factor-Augmented Regressions', Econometrica 74(4): 1133-1150.

**Related options**: [`none`](#none)

_Last reviewed 2026-05-04 by macroforecast author._
