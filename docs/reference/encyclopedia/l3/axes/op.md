# `op`

[Back to L3](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``op`` on sub-layer ``L3_A_step_op`` (layer ``l3``).

## Sub-layer

**L3_A_step_op**

## Axis metadata

- Default: `'lag'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 52 option(s)
- Future: 0 option(s)

## Options

### `adaptive_ma_rf`  --  operational

AlbaMA -- RF-driven adaptive moving average smoother for a single time series.

See [adaptive_ma_rf function page](../op/adaptive_ma_rf.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.adaptive_ma_rf_transform``.

### `asymmetric_trim`  --  operational

Albacore-family rank-space transformation (Goulet Coulombe et al. 2024).

See [asymmetric_trim function page](../op/asymmetric_trim.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.asymmetric_trim_transform``.

### `boruta_selection`  --  operational

All-relevant feature selection via shadow-feature random forest (Kursa-Rudnicki 2010).

See [boruta_selection function page](../op/boruta_selection.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.boruta_selection``.

### `cumsum`  --  operational

Cumulative sum of a series.

See [cumsum function page](../op/cumsum.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.cumsum_transform``.

### `dfm`  --  operational

Dynamic factor model -- Kalman state-space factor extraction.

See [dfm function page](../op/dfm.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.dfm_transform``.

### `diff`  --  operational

First difference: ``y_t - y_{t-1}``.

See [diff function page](../op/diff.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.diff_transform``.

### `feature_selection`  --  operational

Filter columns by variance / correlation / lasso pre-screen.

See [feature_selection function page](../op/feature_selection.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.feature_selection_transform``.

### `fourier`  --  operational

Fourier basis features -- sin/cos at fixed harmonics.

See [fourier function page](../op/fourier.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.fourier_transform``.

### `genetic_algorithm_selection`  --  operational

Evolutionary feature subset search via genetic algorithm (Goldberg 1989).

See [genetic_algorithm_selection function page](../op/genetic_algorithm_selection.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.genetic_algorithm_selection``.

### `hamilton_filter`  --  operational

Hamilton (2018) regression-based detrend (HP-filter alternative).

See [hamilton_filter function page](../op/hamilton_filter.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.hamilton_filter_transform``.

### `holiday`  --  operational

Holiday / event dummy variables.

See [holiday function page](../op/holiday.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.holiday_transform``.

### `hp_filter`  --  operational

Hodrick-Prescott filter -- trend / cycle decomposition.

See [hp_filter function page](../op/hp_filter.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.hp_filter_transform``.

### `interaction`  --  operational

Pairwise interaction terms only (no pure powers).

See [interaction function page](../op/interaction.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.interaction_terms_transform``.

### `kernel`  --  operational

Kernel-feature pre-step (Random Fourier / Nyström handle).

**Alias** -- no dedicated function page. See canonical ``kernel_features`` (`op/kernel_features.md`) for full documentation + standalone usage.

Generic handle for an explicit kernel-feature embedding; concrete dispatch is determined by ``params.kernel`` (``rbf`` / ``poly`` / ``laplacian``). For named variants use ``kernel_features`` (RBF Random Fourier) or ``nystroem``.

**When to use**

Kernel-augmented linear / SVM pipelines.

**References**

* macroforecast design Part 2, L3: 'feature engineering is a DAG of typed transforms; cascade-depth bounds the longest chain at cascade_max_depth.'

**Related options**: [`kernel_features`](#kernel-features), [`nystroem`](#nystroem)

_Last reviewed 2026-05-05 by macroforecast author._

### `kernel_features`  --  operational

Random Fourier features -- approximate RBF kernel via random projection.

See [kernel_features function page](../op/kernel_features.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.kernel_features_transform``.

### `l3_feature_bundle`  --  operational

Internal sentinel: bundle column-producing children into the X matrix.

Synthetic node emitted by the runtime to collect the outputs of all leaf transforms into a single ``X`` artifact consumed by L4. Authors do not write this node by hand; the cascade builder inserts it.

Surfaced in ``operational_options`` so the schema completeness test stays consistent.

**When to use**

Never authored directly -- inserted by the cascade builder.

**References**

* macroforecast design Part 2, L3: 'feature engineering is a DAG of typed transforms; cascade-depth bounds the longest chain at cascade_max_depth.'

**Related options**: [`l3_metadata_build`](#l3-metadata-build)

_Last reviewed 2026-05-05 by macroforecast author._

### `l3_metadata_build`  --  operational

Internal sentinel: assemble L3 metadata sink (lineage / pipeline definitions).

Synthetic node that produces the ``column_lineage`` and ``pipeline_definitions`` sinks from the cascade graph. Powers L7 ``lineage_attribution`` and ``transformation_attribution`` downstream.

**When to use**

Never authored directly -- inserted by the cascade builder when L7 lineage hooks are active.

**References**

* macroforecast design Part 2, L3: 'feature engineering is a DAG of typed transforms; cascade-depth bounds the longest chain at cascade_max_depth.'

**Related options**: [`l3_feature_bundle`](#l3-feature-bundle)

_Last reviewed 2026-05-05 by macroforecast author._

### `lag`  --  operational

Lagged target/predictor block.

See [lag function page](../op/lag.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.lag_matrix``.

### `lasso_path_selection`  --  operational

Feature selection along the Lasso regularisation path (Efron et al. 2004).

See [lasso_path_selection function page](../op/lasso_path_selection.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.lasso_path_selection``.

### `level`  --  operational

Pass-through: keep the series at its original level.

No-op transform; the column flows through unchanged. Used as an explicit anchor in the DAG so downstream ops can reference the level form even when the L2 transform_policy converted to log-differences. Useful when you want both ``level`` and ``diff`` branches in the same recipe.

**When to use**

Authoring branches that need the level form alongside a transformed branch.

**References**

* macroforecast design Part 2, L3: 'feature engineering is a DAG of typed transforms; cascade-depth bounds the longest chain at cascade_max_depth.'

**Related options**: [`diff`](#diff), [`log`](#log), [`log_diff`](#log-diff), [`pct_change`](#pct-change)

_Last reviewed 2026-05-05 by macroforecast author._

### `log`  --  operational

Natural logarithm: ``ln(y)``.

See [log function page](../op/log.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.log_transform``.

### `log_diff`  --  operational

Log first difference: ``ln(y_t) - ln(y_{t-1})``.

See [log_diff function page](../op/log_diff.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.log_diff_transform``.

### `ma_increasing_order`  --  operational

MARX -- moving averages of increasing order (Coulombe 2024).

See [ma_increasing_order function page](../op/ma_increasing_order.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.ma_increasing_order_transform``.

### `ma_window`  --  operational

Trailing moving average over a fixed window.

See [ma_window function page](../op/ma_window.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.ma_window_transform``.

### `maf_per_variable_pca`  --  operational

Per-variable MAF via PCA on lag-panels -- Coulombe et al. (2021 IJF) Eq. (7).

See [maf_per_variable_pca function page](../op/maf_per_variable_pca.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.maf_per_variable_pca_transform``.

### `midas`  --  operational

MIDAS Almon / Exp-Almon / Beta weighted lag polynomial (Ghysels-Sinko-Valkanov 2007).

Parametric mixed-frequency aggregation: emits one column per HF predictor whose value is the weighted sum ``Σ_k ω_k(θ̂) · x_{t·m − k}``. The weight kernel ``ω_k(θ)`` is fitted by NLS against the LF ``target_signal`` input.

**Three weighting families**:
* ``almon`` -- polynomial basis ``ω_k = Σ_q θ_q · k^q`` of order ``polynomial_order`` (default 2). Optional sum-to-one normalisation.
* ``exp_almon`` (default) -- ``ω_k ∝ exp(θ_1 k + θ_2 k²)``; numerically stable and the GSV 2007 §3 default.
* ``beta`` -- ``ω_k ∝ k_norm^{θ_1−1} (1 − k_norm)^{θ_2−1}`` for ``k_norm = (k+1)/(K+1)``; flexible monotone / hump kernel.

Defaults: ``freq_ratio = 3``, ``n_lags_high = 12``, ``sum_to_one = True``, ``max_iter = 200``. Optimiser: ``scipy.optimize.minimize`` (Nelder-Mead). Per-predictor ``theta_hat`` / ``weights`` / ``converged`` stashed in ``result.attrs['midas_fit']`` for L7 inspection.

Requires a ``target_signal`` input port -- shares routing with ``scaled_pca``.

**When to use**

Mixed-frequency nowcasting where parametric lag weights are desired; reproducing GSV 2007 macro / asset-pricing applications.

**When NOT to use**

High-noise predictors where NLS optimiser diverges (use ``u_midas`` + downstream regularised regression instead).

**References**

* macroforecast design Part 2, L3: 'feature engineering is a DAG of typed transforms; cascade-depth bounds the longest chain at cascade_max_depth.'
* Ghysels, Sinko & Valkanov (2007) 'MIDAS Regressions: Further Results and New Directions', Econometric Reviews 26(1): 53-90.
* Ghysels, Santa-Clara & Valkanov (2004) 'The MIDAS Touch: Mixed Data Sampling Regression Models', UCLA / UNC working paper.

**Related options**: [`u_midas`](#u-midas), [`scaled_pca`](#scaled-pca), [`lag`](#lag)

_Last reviewed 2026-05-05 by macroforecast author._

### `nystroem`  --  operational

Nyström kernel approximation -- subset-based feature map.

See [nystroem function page](../op/nystroem.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.nystroem_transform``.

### `nystroem_features`  --  operational

Alias for ``nystroem`` -- explicit feature-stage name.

**Alias** -- no dedicated function page. See canonical ``nystroem`` (`op/nystroem.md`) for full documentation + standalone usage.

Identical to ``nystroem``; preferred when a multi-stage pipeline names its kernel approximation explicitly in the lineage graph.

**When to use**

Multi-stage pipelines that separate kernel approximation from downstream linear fits.

**References**

* macroforecast design Part 2, L3: 'feature engineering is a DAG of typed transforms; cascade-depth bounds the longest chain at cascade_max_depth.'

**Related options**: [`nystroem`](#nystroem)

_Last reviewed 2026-05-05 by macroforecast author._

### `partial_least_squares`  --  operational

Partial least squares regression -- supervised factor extraction.

See [partial_least_squares function page](../op/partial_least_squares.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.partial_least_squares_transform``.

### `pca`  --  operational

Principal component analysis -- linear factor extraction.

See [pca function page](../op/pca.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.pca_transform``.

### `pct_change`  --  operational

Period-over-period percentage change: ``(y_t / y_{t-1}) - 1``.

See [pct_change function page](../op/pct_change.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.pct_change_transform``.

### `polynomial`  --  operational

Polynomial basis expansion -- degree-d powers of input.

**Alias** -- no dedicated function page. See canonical ``polynomial_expansion`` (`op/polynomial_expansion.md`) for full documentation + standalone usage.

sklearn ``PolynomialFeatures`` of degree ``params.degree``. Includes interaction terms by default; set ``params.interaction_only=True`` for products without pure powers.

**When to use**

Capturing low-order non-linearity for linear / kernel models.

**When NOT to use**

High dimension (degree > 3 with many predictors) -- explodes the design matrix; use kernel methods instead.

**References**

* macroforecast design Part 2, L3: 'feature engineering is a DAG of typed transforms; cascade-depth bounds the longest chain at cascade_max_depth.'

**Related options**: [`interaction`](#interaction), [`kernel_features`](#kernel-features), [`polynomial_expansion`](#polynomial-expansion)

_Last reviewed 2026-05-05 by macroforecast author._

### `polynomial_expansion`  --  operational

Alias for ``polynomial`` -- explicit expansion node in cascade pipelines.

See [polynomial_expansion function page](../op/polynomial_expansion.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.polynomial_expansion_transform``.

### `random_projection`  --  operational

Johnson-Lindenstrauss random Gaussian projection.

See [random_projection function page](../op/random_projection.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.random_projection_transform``.

### `recursive_feature_elimination`  --  operational

Backward stepwise feature pruning via estimator importance (Guyon et al. 2002).

See [recursive_feature_elimination function page](../op/recursive_feature_elimination.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.recursive_feature_elimination``.

### `regime_indicator`  --  operational

Discrete regime / state indicator from L1.G.

Lifts the L1.G regime sink (``regime_indicator``) into the feature panel as a categorical column. Required when L4 is configured for ``regime_use = predictor`` or ``conditional_intercept``.

**When to use**

Regime-conditional forecasts where the model needs explicit access to the state.

**References**

* macroforecast design Part 2, L3: 'feature engineering is a DAG of typed transforms; cascade-depth bounds the longest chain at cascade_max_depth.'

**Related options**: [`season_dummy`](#season-dummy), [`time_trend`](#time-trend)

_Last reviewed 2026-05-05 by macroforecast author._

### `savitzky_golay_filter`  --  operational

Polynomial-fit smoothing filter (Savitzky & Golay 1964).

See [savitzky_golay_filter function page](../op/savitzky_golay_filter.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.savitzky_golay_transform``.

### `scale`  --  operational

Standardise to zero mean and unit variance.

See [scale function page](../op/scale.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.scale_transform``.

### `scaled_pca`  --  operational

Scaled / weighted PCA (target-aware factor extraction).

See [scaled_pca function page](../op/scaled_pca.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.scaled_pca_transform``.

### `season_dummy`  --  operational

Calendar dummy variables (month-of-year, quarter-of-year).

See [season_dummy function page](../op/season_dummy.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.season_dummy_transform``.

### `seasonal_lag`  --  operational

Lag at a seasonal period (e.g. y_{t-12} for monthly data).

See [seasonal_lag function page](../op/seasonal_lag.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.seasonal_lag_matrix``.

### `sliced_inverse_regression`  --  operational

sSUFF / Sliced inverse regression (scaled) -- supervised dimension reduction (Huang-Jiang-Li-Tong-Zhou 2022).

See [sliced_inverse_regression function page](../op/sliced_inverse_regression.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.sliced_inverse_regression_transform``.

### `sparse_pca`  --  operational

Sparse PCA -- L1-penalised factor loadings (sklearn / Zou-Hastie-Tibshirani 2006).

See [sparse_pca function page](../op/sparse_pca.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.sparse_pca_transform``.

### `sparse_pca_chen_rohe`  --  operational

Chen-Rohe (2023) Sparse Component Analysis -- non-diagonal D variant.

See [sparse_pca_chen_rohe function page](../op/sparse_pca_chen_rohe.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.sparse_pca_chen_rohe_transform``.

### `stability_selection`  --  operational

Feature selection by subsampling stability -- selection probability threshold (Meinshausen-Bühlmann 2010).

See [stability_selection function page](../op/stability_selection.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.stability_selection``.

### `supervised_pca`  --  operational

Supervised PCA (Giglio-Xiu-Zhang 2025) -- screen-then-PCA on a target panel.

See [supervised_pca function page](../op/supervised_pca.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.supervised_pca_transform``.

### `target_construction`  --  operational

Build the supervised target (``y``) from the panel.

Constructs the regression target according to ``forecast_strategy`` (direct / iterated / cumulative_average) and the L1.F horizon set. Outputs the ``y`` artifact that L4 fit_model nodes consume.

Required as the leaf of every L3 DAG; the runtime auto-injects it when the user does not.

**When to use**

Always required -- runtime auto-inserts when missing.

**References**

* macroforecast design Part 2, L3: 'feature engineering is a DAG of typed transforms; cascade-depth bounds the longest chain at cascade_max_depth.'

**Related options**: [`level`](#level), [`diff`](#diff), [`log_diff`](#log-diff)

_Last reviewed 2026-05-05 by macroforecast author._

### `time_trend`  --  operational

Deterministic linear time trend (``t = 1, 2, ...``).

See [time_trend function page](../op/time_trend.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.time_trend_transform``.

### `u_midas`  --  operational

Unrestricted MIDAS lag stack -- mixed-frequency aggregation primitive (Foroni-Marcellino-Schumacher 2015).

Atomic mixed-frequency primitive: stacks ``K = n_lags_high`` high-frequency lags of each predictor at low-frequency dates without imposing any weighting structure. For HF column ``col`` and LF index position ``t·m``, emits ``col_lag0, col_lag1, …, col_lag{K-1}`` where ``col_lagk[t] = frame[col].iloc[t·m − k]``.

The downstream L4 OLS estimator recovers data-driven lag coefficients -- the *unrestricted* MIDAS regression (paper §3.2 eq.(20)):

``y_{t×k} = μ₀ + μ₁ y_{t×k−k} + ψ₀ x_{t×k−1} + ψ₁ x_{t×k−2} + … + ψ_K x_{t×k−K} + ε_{t×k}``

where μ₀, μ₁, and ψ(L) are estimated by OLS (paper §3.2 p.11). Ridge regularisation is available as an explicit opt-in via ``regularization='ridge'`` in the ``paper_methods.u_midas(...)`` recipe; it deviates from the paper's estimator choice and is not the default.

**Lag-order selection**: ``n_lags_high='bic'`` (default) runs BIC over K ∈ {1, …, ceil(1.5 × freq_ratio)}, fitting OLS at each candidate and selecting K* = argmin BIC (paper §3.2 p.11 + §3.5). Pass an integer to fix K. ``'aic'`` is also accepted.

**AR(1) y-lag**: the ``paper_methods.u_midas(...)`` helper sets ``include_y_lag=True`` by default, prepending the lagged target ``y_lag1`` as the leftmost design-matrix column (μ₁ term of eq.(20)). Set ``include_y_lag=False`` to match the simplified §2.3 eq.(14) form with no AR component.

**Defaults**: ``freq_ratio = 3`` (quarterly target / monthly HF), ``n_lags_high = 'bic'``; ``target_freq = 'low'`` subsamples the LF anchor dates. ``temporal_rule`` is required and rejects ``full_sample_once`` so the aggregation respects walk-forward boundaries.

Surfaces the Borup-Rapach-Schütte (2023) mixed-frequency ML-nowcasting workflow as a 1-line recipe via ``paper_methods.u_midas(...)``.

**When to use**

Macro nowcasting with monthly predictors and quarterly targets; mixed-frequency feature engineering when no parametric weight kernel is desired.

**When NOT to use**

When ``n_lags_high · n_HF_columns`` is large relative to T even after BIC selects a small K -- BIC penalises over-parameterisation but cannot reduce the number of predictor columns. Use ``midas`` parametric weighting instead, or pair with downstream lasso / ridge (set ``regularization='ridge'``) to handle wide design matrices.

**References**

* macroforecast design Part 2, L3: 'feature engineering is a DAG of typed transforms; cascade-depth bounds the longest chain at cascade_max_depth.'
* Foroni, Marcellino & Schumacher (2011/2015) 'Unrestricted Mixed Data Sampling (MIDAS): MIDAS Regressions With Unrestricted Lag Polynomials'. Bundesbank Discussion Paper Series 1, No. 35/2011; published as JRSS-A 178(1): 57-82. DOI 10.1111/rssa.12043.
* Borup, Rapach & Schütte (2023) 'Mixed-frequency machine learning: Nowcasting and backcasting weekly initial claims with daily internet search-volume data', International Journal of Forecasting 39(3): 1122-1144.

**Related options**: [`midas`](#midas), [`lag`](#lag), [`ma_window`](#ma-window)

_Last reviewed 2026-05-05 by macroforecast author._

### `varimax`  --  operational

Varimax-rotated factors (orthogonal rotation for interpretability).

See [varimax function page](../op/varimax.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.varimax_transform``.

### `varimax_rotation`  --  operational

Alias for ``varimax`` -- rotation step in a multi-stage factor pipeline.

**Alias** -- no dedicated function page. See canonical ``varimax`` (`op/varimax.md`) for full documentation + standalone usage.

Identical operation to ``varimax`` but registered separately so a cascading L3 pipeline can declare ``pca → varimax_rotation`` as two visible nodes in its lineage.

**When to use**

Multi-stage pipelines that explicitly separate factor extraction from rotation.

**References**

* macroforecast design Part 2, L3: 'feature engineering is a DAG of typed transforms; cascade-depth bounds the longest chain at cascade_max_depth.'

**Related options**: [`varimax`](#varimax), [`pca`](#pca)

_Last reviewed 2026-05-05 by macroforecast author._

### `wavelet`  --  operational

Discrete wavelet transform -- multi-scale time-frequency features.

See [wavelet function page](../op/wavelet.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.wavelet_transform``.
