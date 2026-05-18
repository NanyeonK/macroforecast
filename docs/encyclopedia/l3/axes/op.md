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

- Operational: 47 option(s)
- Future: 5 option(s)

## Options

### `adaptive_ma_rf`  --  operational

AlbaMA -- RF-driven adaptive moving average smoother for a single time series.

See [adaptive_ma_rf function page](../op/adaptive_ma_rf.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.adaptive_ma_rf_transform``.

### `asymmetric_trim`  --  operational

Albacore-family rank-space transformation (Goulet Coulombe et al. 2024).

See [asymmetric_trim function page](../op/asymmetric_trim.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.asymmetric_trim_transform``.

### `boruta_selection`  --  future

_(no schema description for `boruta_selection`)_

> TBD: option doc not yet authored for this value. The encyclopedia falls back to the bare schema description above. PRs adding a full ``OptionDoc`` entry under ``macroforecast/scaffold/option_docs/l3.py`` are welcome.

### `cumsum`  --  operational

Cumulative sum of a series.

See [cumsum function page](../op/cumsum.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.cumsum_transform``.

### `dfm`  --  operational

Dynamic factor model -- Kalman state-space factor extraction.

statsmodels ``DynamicFactor`` MLE estimate of latent factors with idiosyncratic AR(1) errors. Differs from ``pca`` in that factors are smoothed via the Kalman filter and respect a factor-VAR transition.

When the panel is mixed-frequency (FRED-SD), the runtime auto-routes to ``DynamicFactorMQ`` (Mariano-Murasawa 2003).

**When to use**

Smoothed factors with an explicit dynamic; mixed-frequency panels (FRED-SD).

**References**

* macroforecast design Part 2, L3: 'feature engineering is a DAG of typed transforms; cascade-depth bounds the longest chain at cascade_max_depth.'
* Mariano & Murasawa (2003) 'A new coincident index of business cycles based on monthly and quarterly series', JAE 18(4): 427-443.

**Related options**: [`pca`](#pca), [`scaled_pca`](#scaled-pca)

_Last reviewed 2026-05-05 by macroforecast author._

### `diff`  --  operational

First difference: ``y_t - y_{t-1}``.

See [diff function page](../op/diff.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.diff_transform``.

### `feature_selection`  --  operational

Filter columns by variance / correlation / lasso pre-screen.

Drops columns failing one of three criteria configured via ``params.method``:

* ``variance`` -- drop columns with variance below ``params.threshold``.
* ``correlation`` -- drop columns with pairwise correlation above ``params.threshold`` (keeps the first).
* ``lasso`` -- fit a Lasso pre-screen and keep columns with non-zero coefficients.

**When to use**

Trimming the panel before expensive downstream estimators (NN, SVM, kernel) when high-dim noise dominates.

**When NOT to use**

Tree models -- they handle irrelevant features natively.

**References**

* macroforecast design Part 2, L3: 'feature engineering is a DAG of typed transforms; cascade-depth bounds the longest chain at cascade_max_depth.'

**Related options**: [`scale`](#scale), [`pca`](#pca)

_Last reviewed 2026-05-05 by macroforecast author._

### `fourier`  --  operational

Fourier basis features -- sin/cos at fixed harmonics.

See [fourier function page](../op/fourier.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.fourier_transform``.

### `genetic_algorithm_selection`  --  future

_(no schema description for `genetic_algorithm_selection`)_

> TBD: option doc not yet authored for this value. The encyclopedia falls back to the bare schema description above. PRs adding a full ``OptionDoc`` entry under ``macroforecast/scaffold/option_docs/l3.py`` are welcome.

### `hamilton_filter`  --  operational

Hamilton (2018) regression-based detrend (HP-filter alternative).

See [hamilton_filter function page](../op/hamilton_filter.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.hamilton_filter_transform``.

### `holiday`  --  operational

Holiday / event dummy variables.

0/1 indicators for calendar holidays (US federal by default; ``params.country`` selects locale via the ``holidays`` package). For business / financial macro series.

**When to use**

Daily / weekly business-cycle series where holidays create discrete level shifts.

**When NOT to use**

Pure macro series at monthly+ frequency where holidays are absorbed by ``season_dummy``.

**References**

* macroforecast design Part 2, L3: 'feature engineering is a DAG of typed transforms; cascade-depth bounds the longest chain at cascade_max_depth.'

**Related options**: [`season_dummy`](#season-dummy)

_Last reviewed 2026-05-05 by macroforecast author._

### `hp_filter`  --  operational

Hodrick-Prescott filter -- trend / cycle decomposition.

See [hp_filter function page](../op/hp_filter.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.hp_filter_transform``.

### `interaction`  --  operational

Pairwise interaction terms only (no pure powers).

See [interaction function page](../op/interaction.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.interaction_terms_transform``.

### `kernel`  --  operational

Kernel-feature pre-step (Random Fourier / Nyström handle).

Generic handle for an explicit kernel-feature embedding; concrete dispatch is determined by ``params.kernel`` (``rbf`` / ``poly`` / ``laplacian``). For named variants use ``kernel_features`` (RBF Random Fourier) or ``nystroem``.

**When to use**

Kernel-augmented linear / SVM pipelines.

**References**

* macroforecast design Part 2, L3: 'feature engineering is a DAG of typed transforms; cascade-depth bounds the longest chain at cascade_max_depth.'

**Related options**: [`kernel_features`](#kernel-features), [`nystroem`](#nystroem)

_Last reviewed 2026-05-05 by macroforecast author._

### `kernel_features`  --  operational

Random Fourier features -- approximate RBF kernel via random projection.

sklearn ``RBFSampler``: maps inputs to ``params.n_components`` random Fourier features whose dot product approximates the RBF kernel. Enables linear models to fit RBF-kernelised responses at training-set-size linear cost.

**When to use**

Kernel-augmented ridge / SVM at scale (n > 10k).

**When NOT to use**

Small-sample problems where exact kernel SVM is feasible.

**References**

* macroforecast design Part 2, L3: 'feature engineering is a DAG of typed transforms; cascade-depth bounds the longest chain at cascade_max_depth.'
* Rahimi & Recht (2007) 'Random Features for Large-Scale Kernel Machines', NeurIPS.

**Related options**: [`kernel`](#kernel), [`nystroem`](#nystroem)

_Last reviewed 2026-05-05 by macroforecast author._

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

### `lasso_path_selection`  --  future

_(no schema description for `lasso_path_selection`)_

> TBD: option doc not yet authored for this value. The encyclopedia falls back to the bare schema description above. PRs adding a full ``OptionDoc`` entry under ``macroforecast/scaffold/option_docs/l3.py`` are welcome.

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

sklearn ``Nystroem`` constructs a low-rank approximation of an arbitrary kernel matrix using a random subsample of training points. More accurate than Random Fourier features for non-RBF kernels but with a larger memory footprint.

**When to use**

Non-RBF kernel-augmented linear models (poly / sigmoid).

**References**

* macroforecast design Part 2, L3: 'feature engineering is a DAG of typed transforms; cascade-depth bounds the longest chain at cascade_max_depth.'

**Related options**: [`kernel_features`](#kernel-features), [`kernel`](#kernel)

_Last reviewed 2026-05-05 by macroforecast author._

### `nystroem_features`  --  operational

Alias for ``nystroem`` -- explicit feature-stage name.

Identical to ``nystroem``; preferred when a multi-stage pipeline names its kernel approximation explicitly in the lineage graph.

**When to use**

Multi-stage pipelines that separate kernel approximation from downstream linear fits.

**References**

* macroforecast design Part 2, L3: 'feature engineering is a DAG of typed transforms; cascade-depth bounds the longest chain at cascade_max_depth.'

**Related options**: [`nystroem`](#nystroem)

_Last reviewed 2026-05-05 by macroforecast author._

### `partial_least_squares`  --  operational

Partial least squares regression -- supervised factor extraction.

Computes orthogonal latent components that maximise the covariance with the target (not just predictor variance, as in PCA). sklearn's ``PLSRegression``; ``params.n_components``.

**When to use**

When a target-supervised reduction is preferable to PCA's unsupervised projection.

**References**

* macroforecast design Part 2, L3: 'feature engineering is a DAG of typed transforms; cascade-depth bounds the longest chain at cascade_max_depth.'
* Wold, Sjöström & Eriksson (2001) 'PLS-regression: a basic tool of chemometrics', Chemometrics and Intelligent Laboratory Systems 58(2): 109-130.

**Related options**: [`pca`](#pca), [`scaled_pca`](#scaled-pca)

_Last reviewed 2026-05-05 by macroforecast author._

### `pca`  --  operational

Principal component analysis -- linear factor extraction.

See [pca function page](../op/pca.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.pca_transform``.

### `pct_change`  --  operational

Period-over-period percentage change: ``(y_t / y_{t-1}) - 1``.

See [pct_change function page](../op/pct_change.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.pct_change_transform``.

### `polynomial`  --  operational

Polynomial basis expansion -- degree-d powers of input.

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

Reduces dimensionality by multiplying with a random Gaussian matrix scaled to (approximately) preserve pairwise distances. Cheap baseline for dimensionality reduction; sklearn's ``GaussianRandomProjection``.

**When to use**

Sweep baselines / sanity checks against PCA's structured reduction.

**References**

* macroforecast design Part 2, L3: 'feature engineering is a DAG of typed transforms; cascade-depth bounds the longest chain at cascade_max_depth.'

**Related options**: [`pca`](#pca), [`kernel_features`](#kernel-features)

_Last reviewed 2026-05-05 by macroforecast author._

### `recursive_feature_elimination`  --  future

_(no schema description for `recursive_feature_elimination`)_

> TBD: option doc not yet authored for this value. The encyclopedia falls back to the bare schema description above. PRs adding a full ``OptionDoc`` entry under ``macroforecast/scaffold/option_docs/l3.py`` are welcome.

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

Weights each column by its predictive correlation with the target before performing PCA. Implements the Huang-Jiang-Tu-Zhou (2022) scaled PCA for forecasting macro variables.

Reduces to plain PCA when all weights are equal.

**When to use**

When standard PCA's leading factor is dominated by predictively-irrelevant variance.

**References**

* macroforecast design Part 2, L3: 'feature engineering is a DAG of typed transforms; cascade-depth bounds the longest chain at cascade_max_depth.'
* Huang, Jiang, Tu & Zhou (2022) 'Scaled PCA: A New Approach to Dimension Reduction', Management Science 68(3): 1678-1695.

**Related options**: [`pca`](#pca), [`partial_least_squares`](#partial-least-squares)

_Last reviewed 2026-05-05 by macroforecast author._

### `season_dummy`  --  operational

Calendar dummy variables (month-of-year, quarter-of-year).

See [season_dummy function page](../op/season_dummy.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.season_dummy_transform``.

### `seasonal_lag`  --  operational

Lag at a seasonal period (e.g. y_{t-12} for monthly data).

See [seasonal_lag function page](../op/seasonal_lag.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.seasonal_lag_matrix``.

### `sliced_inverse_regression`  --  operational

sSUFF / Sliced inverse regression (scaled) -- supervised dimension reduction (Huang-Jiang-Li-Tong-Zhou 2022).

Supervised dimension reduction extending ``scaled_pca`` to non-linear y → X dependence. Pipeline: (1) standardise X; (2) optional column-wise predictive scaling (``scaling_method`` = ``scaled_pca`` reuses the Huang-Zhou OLS-slope; ``marginal_R2`` uses sign(β_j)·√R²_j; ``none`` skips); (3) sort rows by y and partition into ``n_slices`` H contiguous slices; (4) compute weighted between-slice covariance ``Σ_S = Σ_h (n_h/n) · m̄_h · m̄_h^⊤``; (5) take the top-``n_components`` eigenvectors as factor loadings; (6) project the full panel onto these directions. The sSUFF augmentation (Huang-Zhou-Tong 2022) recovers latent factors with higher correlation than plain SIR in the macro-panel regime where signals are sparse over predictors.

Defaults: ``n_components = 2``, ``n_slices = 10``, ``scaling_method = 'scaled_pca'``. Requires a ``target_signal`` input port; ``temporal_rule`` is required and rejects ``full_sample_once``.

**When to use**

Supervised factor extraction from macro panels with non-linear y → X structure; alternative to ``scaled_pca`` when the predictive direction is non-monotone.

**When NOT to use**

Very small T (need ≥ 5·n_slices observations after dropping NaN); strictly linear y → X relationship (``scaled_pca`` is sufficient).

**References**

* macroforecast design Part 2, L3: 'feature engineering is a DAG of typed transforms; cascade-depth bounds the longest chain at cascade_max_depth.'
* Huang, Jiang, Li, Tong & Zhou (2022) 'Scaled PCA: A New Approach to Dimension Reduction', Management Science 68(3): 1678-1695.
* Fan, Xue & Yao (2017) 'Sufficient forecasting using factor models', Journal of Econometrics 201(2): 292-306.
* Li (1991) 'Sliced Inverse Regression for Dimension Reduction', JASA 86(414): 316-327.

**Related options**: [`scaled_pca`](#scaled-pca), [`supervised_pca`](#supervised-pca), [`partial_least_squares`](#partial-least-squares)

_Last reviewed 2026-05-05 by macroforecast author._

### `sparse_pca`  --  operational

Sparse PCA -- L1-penalised factor loadings (sklearn / Zou-Hastie-Tibshirani 2006).

Variant of PCA where loadings are pushed toward zero by an L1 penalty (``params.alpha``). Yields more interpretable factors at the cost of a small reconstruction loss; uses sklearn's ``SparsePCA``.

**When to use**

When you want factor loadings to map cleanly onto a small subset of original predictors (interpretability).

**When NOT to use**

When pure variance maximisation is more important than interpretability -- use plain ``pca``. For the Chen-Rohe (2023) SCA variant used in Zhou-Rapach (2025) Sparse Macro-Finance Factors, use ``sparse_pca_chen_rohe`` instead.

**References**

* macroforecast design Part 2, L3: 'feature engineering is a DAG of typed transforms; cascade-depth bounds the longest chain at cascade_max_depth.'

**Related options**: [`pca`](#pca), [`scaled_pca`](#scaled-pca), [`sparse_pca_chen_rohe`](#sparse-pca-chen-rohe), [`supervised_pca`](#supervised-pca)

_Last reviewed 2026-05-05 by macroforecast author._

### `sparse_pca_chen_rohe`  --  operational

Chen-Rohe (2023) Sparse Component Analysis -- non-diagonal D variant.

Sparse component analysis solving ``min_{Z,D,Θ} ‖X − Z D Θ'‖_F`` s.t. ``Z ∈ S(T,J)``, ``Θ ∈ S(M,J)``, ``‖Θ‖_1 ≤ ζ`` (Chen-Rohe 2023; Rapach & Zhou 2025 eq. 3). Differs from ``sparse_pca`` (sklearn / Zou-Hastie-Tibshirani 2006) in two ways: (1) the central matrix D is *not* restricted to be diagonal, which lets SCA explain more total variation for a given sparsity budget; (2) the single hyperparameter ``ζ ∈ [J, J√M]`` enters as an ℓ_1 budget *constraint* rather than a Lagrangian penalty.

Implementation: alternating maximisation of the equivalent bilinear convex-hull form ``max_{Z,Θ} ‖Z' X Θ‖_F`` over ``H(T,J) × H(M,J)`` (Zhou-Rapach 2025 eq. 4), iterating SVD-projection of Z and L1-budget projection of Θ. Used as the macro-side stage in Rapach & Zhou (2025) Sparse Macro-Finance Factors. Operational v0.9.1 dev-stage v0.9.0C-3.

Hyperparams: ``n_components`` (= J; default 4), ``zeta`` (= L1 budget; ``0.0`` defaults to J = most-binding boundary the paper finds optimal in CV), ``max_iter`` (default 200), ``random_state``.

**When to use**

Sparse macro-finance factor extraction with non-diagonal D; the Rapach-Zhou (2025) macro-side procedure.

**When NOT to use**

When sklearn-style L1-penalised loadings are sufficient -- prefer the cheaper ``sparse_pca``.

**References**

* macroforecast design Part 2, L3: 'feature engineering is a DAG of typed transforms; cascade-depth bounds the longest chain at cascade_max_depth.'
* Chen & Rohe (2023) 'A New Basis for Sparse Principal Component Analysis', Journal of Computational and Graphical Statistics. arXiv:2007.00596.
* Rapach & Zhou (2025) 'Sparse Macro-Finance Factors' working paper -- §2.1 eqs. (3)-(4).

**Related options**: [`sparse_pca`](#sparse-pca), [`supervised_pca`](#supervised-pca), [`scaled_pca`](#scaled-pca), [`pca`](#pca)

_Last reviewed 2026-05-05 by macroforecast author._

### `stability_selection`  --  future

_(no schema description for `stability_selection`)_

> TBD: option doc not yet authored for this value. The encyclopedia falls back to the bare schema description above. PRs adding a full ``OptionDoc`` entry under ``macroforecast/scaffold/option_docs/l3.py`` are welcome.

### `supervised_pca`  --  operational

Supervised PCA (Giglio-Xiu-Zhang 2025) -- screen-then-PCA on a target panel.

Two-stage supervised reduction:
  1. For each target column ``g``, rank panel columns by univariate correlation with ``g`` and keep the top ``⌊q · M⌋`` (q ∈ (0, 1] hyperparameter; default 0.5);
  2. Run PCA on the screened sub-panel, returning P supervised components.

Refinement of Giglio-Xiu (2021) three-pass: screening makes the construction robust to weak factors and omitted-variable bias. Used as the asset-side stage of Rapach & Zhou (2025) Sparse Macro-Finance Factors for risk-premium estimation. Distinct from ``partial_least_squares`` (PLS uses covariance-maximising NIPALS over all columns; SPCA uses correlation-screened PCA on a sub-panel) and from ``scaled_pca`` (Huang-Jiang-Tu-Zhou 2022 weights every column; SPCA hard-screens).

Operational v0.9.1 dev-stage v0.9.0C-4. Hyperparams: ``n_components`` (= P; default 4), ``q`` (screening rate; default 0.5).

**When to use**

Cross-sectional asset-pricing factor extraction; weak-factor-robust supervised reduction; Rapach-Zhou (2025) replication.

**When NOT to use**

When the supervisory signal is dense (every panel column matters) -- prefer ``scaled_pca`` or ``partial_least_squares``.

**References**

* macroforecast design Part 2, L3: 'feature engineering is a DAG of typed transforms; cascade-depth bounds the longest chain at cascade_max_depth.'
* Giglio, Xiu & Zhang (2025) 'Test Assets and Weak Factors', Journal of Finance, forthcoming.
* Giglio & Xiu (2021) 'Asset Pricing with Omitted Factors', Journal of Political Economy 129(7): 1947-1990.
* Rapach & Zhou (2025) 'Sparse Macro-Finance Factors' working paper -- §2.2 eqs. (5)-(8).

**Related options**: [`partial_least_squares`](#partial-least-squares), [`scaled_pca`](#scaled-pca), [`sparse_pca_chen_rohe`](#sparse-pca-chen-rohe), [`pca`](#pca)

_Last reviewed 2026-05-05 by macroforecast author._

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

Adds a column ``t`` to the panel; with ``params.degree > 1`` appends polynomial trends. Deterministic complement to stochastic detrending (HP / Hamilton).

**When to use**

Trend-stationary linear models where a deterministic trend is part of the DGP.

**When NOT to use**

Series with structural breaks -- use ``regime_indicator`` or stochastic detrending instead.

**References**

* macroforecast design Part 2, L3: 'feature engineering is a DAG of typed transforms; cascade-depth bounds the longest chain at cascade_max_depth.'

**Related options**: [`hp_filter`](#hp-filter), [`hamilton_filter`](#hamilton-filter)

_Last reviewed 2026-05-05 by macroforecast author._

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

Applies a varimax rotation to PCA loadings, maximising the variance of squared loadings within each factor. Produces factors that load heavily on a small subset of original predictors -- useful for naming / labelling factors.

**When to use**

Factor analysis where downstream interpretation requires distinct, well-named factors.

**References**

* macroforecast design Part 2, L3: 'feature engineering is a DAG of typed transforms; cascade-depth bounds the longest chain at cascade_max_depth.'

**Related options**: [`pca`](#pca), [`sparse_pca`](#sparse-pca)

_Last reviewed 2026-05-05 by macroforecast author._

### `varimax_rotation`  --  operational

Alias for ``varimax`` -- rotation step in a multi-stage factor pipeline.

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
