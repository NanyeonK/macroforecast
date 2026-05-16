# Changelog

Notable changes since the v0.0.0 schema reset. See ``CLAUDE.md`` for the
full per-version honesty-pass history embedded in repo documentation.

## [Unreleased]

### Audit Phase fix bundle (Cycle 14) — 13 source fixes from Cycle 13 audit checklist

#### BREAKING
- **J-3** L1 sink hash (`l1_data_definition_v1`) no longer depends on `cache_root`. Same recipe run with different `output_directory` now produces identical L1 hash. `mf.replicate()` returns `sink_hashes_match=True` for standard usage. Pre-existing manifests will report mismatch on re-replicate; users must regenerate manifests once.

#### Added (P1)
- **J-1** FRED-QD loader is NaT-safe (`data_through` no longer crashes when last index is NaT).
- **J-4** Custom preprocessor errors now surface (was: silent `except Exception` swallow + false `applied=True` in manifest).
- **J-5** L3 `feature_selection` op now hard-rejects `temporal_rule: full_sample_once` matching the `scale`/`pca` pattern (runtime was already per-origin).
- **K-1** `macroforecast.defaults` accessible at top level (was: `AttributeError`).
- **K-2** `ManifestExecutionResult` now exposes `.forecasts`, `.metrics`, `.ranking`, `.manifest` as documented in `simple_api/quickstart.md`.
- **K-3** Manifest provenance auto-captures `data_revision_tag` for FRED `current_vintage` + records `sample_start_resolved` / `sample_end_resolved`.

#### Added (P2)
- **K-4** `vintage_policy: real_time_alfred` hard-rejected as future feature.
- **L1-1** RuntimeError messages now include layer name + recipe key path (L1, L4 coverage).
- **L1-2** Manifest now captures `warnings` (top-level + per-cell).
- **L1-3** Validator warns on unknown top-level recipe key + unknown `leaf_config` key (UserWarning, not raise).
- **L1-4** L3.5 `selection_view: none` no longer triggers false "requires feature_selection" error.
- **L1-5** DM/CW result dict now includes `decision`, `alternative`, `correction_method` fields (`decision_at_5pct` retained for backward compat).
- **L2-1** `markdown_report` export: tabulate available (added to extras / clean ImportError).
- **L2-2** `mf.run('/nonexistent.yaml')` now raises `FileNotFoundError` (was: confusing "YAML root must be a mapping").
- **L2-3** CLI now prints manifest path on success; invalid YAML shows clean 1-3 line error (was: raw 20-line traceback).
- **L2-4** `mf.run(output_directory=...)` kwarg now propagates to L8 `leaf_config.output_directory` (was: silently ignored unless recipe set it).
- **L2-5** SHAP op subsamples to 2000 rows with `UserWarning` for larger panels.

#### Deferred (NOT_REPRODUCIBLE / out of cycle scope)
- **J-2** sweep markers via `mf.run(Path)` — Cycle 13 finding NOT_REPRODUCIBLE in current `ec388d17`. Path and string branches use identical recipe canonicalization. Deferred unless reproducer surfaces.

#### Cross-model review patches (Cycle 15) — P1 corrections

These three fixes address Cycle 14 issues that a single-model reviewer (Claude) missed but cross-model review (MiniMax M2.5 + Codex GPT-5.4 xhigh) caught.

- **M-1** Manifest `sample_start_resolved` / `sample_end_resolved` now reflect the post-L2 sample window (was: pre-window raw panel index). Users with explicit `sample_start_rule=fixed_date` now see the resolved window in manifest provenance.
- **M-2** Custom preprocessor: signature validation pre-call via `inspect.signature`; body TypeErrors no longer misattributed to "wrong signature".
- **M-3** L6 DM/CW result: `decision_at_5pct` accessible for backward compat with `DeprecationWarning`, but excluded from `keys()` / `__iter__` / `len()` to avoid silent length change for code iterating keys.

#### Internal / regression fixes
- **L3** `_jsonable` now handles `np.generic` and `np.ndarray` (mirrors `_json_safe` pattern in `execution.py`). Fixes a regression introduced by L2-4 where panel data with `np.float64` cells caused `yaml.safe_dump` `RepresenterError` via the activated manifest write path.
- **C15.5** `test_r4_02_paper_table2_k60_midas_wins` marked `@pytest.mark.deep` to prevent hour-scale runs in non-deep CI/local pytest. Test logic unchanged; computationally infeasible at k=60 (135,000 lstsq calls). Use `pytest -m deep` to run explicitly.
- **C15.6** U-MIDAS BIC lag selection: emit `UserWarning` when `K_max > 30` (computed as `ceil(1.5 × freq_ratio)`). At very high frequency ratios (e.g., k=60), brute-force BIC enumeration is computationally intractable. Warning suggests setting `n_lags_high` manually to bypass search. Algorithm unchanged — paper-faithful behavior preserved when warning is suppressed or user overrides.

#### Notes
- Cycle 14 ships as v0.9.1 per user decision. 1 BREAKING item (J-3) enumerated above.
- 7 HANDOFF items deferred to v0.9.2 backlog: SHAP threshold customizability beyond 2000 default, cross-cell L2 memoization, `mf.replicate(override_recipe=...)`, P3 docs replication hash table refresh (scriber's domain in this cycle).

### Added

- `macroforecast.core.stages`: new module exposing `STAGE_BY_LAYER` (13-entry
  `dict[str, str]` bijection mapping every `LayerId` value to its stage label),
  `stage_of(*, layer_id, sink_name)` helper (resolves a layer ID or a sink
  contract name to its stage label; raises `ValueError` on unknown input), and
  `StageLabel` type alias (`str`). Three additive public exports added to
  `macroforecast.core.__all__` — no breaking changes. Test count: 1432 → 1505
  (+73 scenarios, T-01 through T-73; zero new regressions). Used by future Kedro
  adapter (P1) for `kedro viz` layer tag assignment and Wizard P2 for navigator
  rail color coding and Mosaic Cube grouping. Run: `2026-05-13-phase-stage-labels`.

  **Stage labels** (9 base + 4 diagnostic):

  | Layer | Stage label |
  |-------|-------------|
  | `l0` | `meta` |
  | `l1` | `data` |
  | `l2` | `clean` |
  | `l3` | `features` |
  | `l4` | `forecasts` |
  | `l5` | `evaluation` |
  | `l6` | `tests` |
  | `l7` | `interpretation` |
  | `l8` | `artifacts` |
  | `l1_5` | `data_diagnostic` |
  | `l2_5` | `clean_diagnostic` |
  | `l3_5` | `features_diagnostic` |
  | `l4_5` | `model_diagnostic` |

- `macroforecast/wizard/` — Solara-based web UI for YAML recipe authoring (Phase
  P2a MVP). Entry point: `macroforecast wizard [--port PORT] [--no-browser]
  [recipe.yaml]` (default port 8765). 14 new modules implementing a 3-pane
  Cursor-inspired layout: left layer rail (L0–L8, color-coded by
  `STAGE_BY_LAYER`), center workspace (form or DAG placeholder), right live YAML
  preview. The L0 form is fully wired via `option_docs` schema (`layer_form_schema`
  + `OptionInput`); L1/L2/L5/L6 form infrastructure is generated but
  gate-verified in P2b; L3/L4/L7 show a DAG placeholder until P3. Reactive state
  (`RecipeState`, `current_recipe`, `yaml_text`) guarantees bidirectional
  recipe ↔ YAML sync on every field edit. Optional install:
  `pip install 'macroforecast[wizard]'` (adds `solara>=1.30`). Test count:
  1505 → 1550 (+45 scenarios across 8 test files; 1 xfailed: LR-04 Solara click
  simulation limitation, deferred to P2c). Run:
  `2026-05-13-phase-wizard-p2-skeleton`.

### Changed

- **BREAKING (public API default)**: `mf.forecast()` and `mf.Experiment()` default `random_seed` changed from `0` → `42` to align with the `DEFAULT_PROFILE` source-of-truth in `macroforecast.defaults`. Users who relied on the old default seed for reproducibility must pass `random_seed=0` explicitly.
- `macroforecast.defaults.DEFAULT_PROFILE` "model_family" key changed from `"ar"` (dead reference — not in L4 family registry) to `"ar_p"` (valid AR(p) family). User-facing behavior unchanged because `Experiment` signature already used `"ar_p"`; this fixes a silent internal inconsistency.
- New public constants in `macroforecast.defaults`: `DEFAULT_MODEL_FAMILY`, `DEFAULT_RANDOM_SEED`, `DEFAULT_HORIZONS`, `DEFAULT_FORECAST_STRATEGY`, `DEFAULT_TRAINING_START_RULE`, `DEFAULT_REFIT_POLICY`. Use these instead of hardcoded literals when calling the Simple API programmatically.

### Deprecated

- `macroforecast/scaffold/wizard.py` (stdlib CLI wizard) — `run_wizard()` now
  emits `DeprecationWarning` on every call:
  `"macroforecast.scaffold.wizard (CLI wizard) is deprecated as of v0.9. Use macroforecast wizard for the new browser-based wizard. This module will be removed in v1.0."`
  Use `macroforecast wizard` (Solara web UI) instead. Removal scheduled for v1.0.

### Removed

- `docs/_html_extra/navigator_app/` (116K static HTML/JS app: `app.js` 91K +
  `styles.css` 16K + `index.html` 2K) — superseded by the new Solara-based
  `macroforecast wizard` (P2a). Last meaningful update was 2026-05-02
  (abandoned redesign revert). Future visualization in
  `macroforecast.adapters.kedro` (P1, planned). Run:
  `2026-05-13-phase-navigator-app-cleanup`.

### Internal

- Solara 1.57.3 + Starlette 0.41.3 tested and confirmed compatible. Starlette
  `<1.0` upper bound not yet pinned in `pyproject.toml` — deferred to P2b.
  Starlette `>=1.0` will break the `on_startup` interface used by
  `wizard/cli.py`. Note in `wizard/cli.py` documents this pending constraint.

---

### Audit Phase fix bundle (Cycle 12) -- 15 source fixes from Codex audit checklist

#### BREAKING
- **F-P1-1** L1 `release_lag_rule` now enforced at runtime -- predictor columns shift forward by their declared lag; forecasts respect data-availability latency. Results computed with `release_lag_rule != "ignore_release_lag"` in prior versions may differ.
- **F-P1-2** L2 imputation now computed per-origin (expanding window) when `imputation_temporal_rule = expanding_window_per_origin` (default). Removes lookahead from full-sample mean / EM / fill.
- **F-P1-3** L2 outlier policies now computed per-origin under same rule. Removes lookahead from full-sample IQR / z-score / winsorize.
- **F-P1-6** FRED-SD `sd_states` (state filter) is now applied -- previously validated but silently ignored.
- **F-P1-7** FRED-SD `fred_sd_variable_group` is now applied -- variable groups resolve to actual variable lists via `raw/fred_sd_groups.py`.
- **F-P1-9** L5 benchmark `benchmark_window`, `benchmark_scope`, `regime_metrics` axes now raise on non-default values (previously silently ignored). Future implementation tracked in TODO comments.
- **F-P1-10** `forecasts.csv` schema gains two columns appended at end: `forecast_date` and `actual`. New order: `model_id, target, horizon, origin, forecast, forecast_date, actual`. JSON export updated to match.
- **F-P1-11** L7 figure axes `figure_dpi`, `figure_format`, `top_k_features_to_show`, `precision_digits` are now honored. Previously hard-coded `.pdf` output is replaced by `figure_format` value.
- **F-P1-12** Recipe hash algorithm changed from Python `hash()` (process-salted) to `hashlib.sha256(canonical_json)[:16]`. Recipe hashes are now stable across processes; prior manifest values will mismatch on re-run.
- **F-P1-14** L2 `official_transform_scope` is now honored. Previously tcodes were applied to all series regardless of scope; results may differ for users who set non-default scope.

#### Added
- **F-P0-1** Simple-API `start=` / `end=` accept `YYYY` and `YYYY-MM` partial-ISO forms (normalized to first-of-month / last-of-month). Unblocks all `simple_api/*.md` documented examples.
- **F-P1-4** L1 panels with duplicate dates now raise `RuntimeError` listing offending dates (was: silent coalesce / undefined behavior).
- **F-P1-5** Custom CSV loader rejects FRED-official-format headers (`Transform:` first row) with a hint to use `dataset="fred_md"` instead (was: silent corruption of first data row).
- **F-P1-8** L5 metric registry adds `medae`, `theil_u1`, `theil_u2`, `success_ratio`.
- **F-P1-13** L8 manifest gains `cache_root` provenance field.

#### Notes
- Cycle 12 release stays at v0.9.1 (no v0.10 bump) per user decision; BREAKING items enumerated for visibility.
- Documentation drift (`v0.9.0a0` -> `v0.9.0`, `35+ families` -> `40+ families`, `v0.9.4` -> `v0.9.0` in CONVENTIONS) handled by scriber as separate docs commit.

---

## [0.9.0] -- 2026-05-13 -- "v0.9.0 stable cut (F-02 + DOCS-1 + MC-RECAL closure)"

After the 16-paper full-coverage alpha pre-release (`v0.9.0a0`, 2026-05-12), the
three remaining stable prereqs were closed: F-02 (phantom replaced with Marcellino-
Schumacher 2010 Factor MIDAS), DOCS-1 (option_docs/l3.py u_midas description synced
to F-07-R defaults + Sphinx csv fix), and MC-RECAL (paper-symmetric MC re-calibration;
mean_ratio 0.9173 anchors paper Table 2 0.91). All 17 corpus papers reach paper-
faithful or paper-anchored status. F-07-R closure preserved. Production code
untouched by MC-RECAL.

### Phase F-14-17 LOW batch (2026-05-12)

Run `2026-05-12-phase-f14-f17-low-batch`. Primary commit `c9285077`. Reviewer GO.
Test count: 1417 baseline → 1431 (+14 new tests). No new regressions. Closes
Round 7 PDF-direct audit LOW findings across papers 14, 15, 16, 17.

* **F-14** — `_ShrinkToTargetRidge._resolve_target` hard-error unification
  (`core/runtime.py:4451`). When `prior_target=None`, the method previously emitted a
  `UserWarning` and returned a uniform `1/K` fallback. It now raises `ValueError`
  immediately, matching the guard already present in the `maximally_forward_looking`
  helper. Paper Albacore (Goulet Coulombe et al. 2024) Eq. (1) is undefined without
  basket weights `w_headline`; silent fallback was misleading. **Breaking change** for
  hand-crafted YAML recipes that omitted `prior_target` under `prior: shrink_to_target`.
  Test `test_albacore_prior_target_none_emits_warning` renamed →
  `test_albacore_prior_target_none_raises_value_error` and updated to use
  `pytest.raises(ValueError, match=r"prior_target")`.

* **F-15** — New standalone helper `sparse_macro_factors_risk_premia()`
  (`recipes/paper_methods.py:1785`). Implements Rapach-Zhou (2025) §2.3 Strategy Step 3:
  Supervised PCA (SPCA) risk-premium estimation on N equity test asset excess returns.
  Six-step pipeline: boundary-row drop → 5-fold CV over `q_grid` to select screening
  proportion `q*` → per-factor asset screening by |corr(R_j, G_f)| → factor-mimicking
  portfolio (FMP) weights via OLS + normalization → time-series OLS for loading matrix
  β̂ (N × J) → cross-sectional SPCA risk premia γ̂ = (β̂'β̂)⁻¹β̂'μ̂_R. Returns a dict
  with keys `gamma_hat`, `beta_hat`, `fmp_returns`, `fmp_weights`, `screened_assets`,
  `q_selected`. This is a **standalone analysis function**, not a `macroforecast.run`-
  compatible recipe (risk-premium estimation is an asset-pricing post-processing step,
  not a macro-forecasting recipe). Exported via `__all__`.

* **F-16** — New L3 op `maf_per_variable_pca` (`core/ops/l3_ops.py:203`,
  `core/runtime.py:15047`). Implements Coulombe et al. (2021 IJF) Eq. (7) per-variable
  Moving Average Factors (MAF) via PCA on lag-panels. For each series k = 1..K: builds
  the T × (n_lags+1) lag-panel `[X_k, L X_k, ..., L^{n_lags} X_k]`, runs PCA
  retaining `n_components_per_var` components, and concatenates across K series to
  produce a (T, K × n_components_per_var) panel (paper default n_lags=12,
  n_components_per_var=2 → output T × 2K). **Backward compatibility preserved**: the
  existing stacked-PCA MAF cell (`ma_increasing_order → pca(n_components=4)`) in the
  16-cell horse-race helper (`paper_methods.py:1993–2016`) is unchanged. The new op is
  an additive extension. OptionDoc entry `_OP_MAF_PER_VARIABLE_PCA` added to
  `scaffold/option_docs/l3.py` for wizard/sphinx completeness check.

* **F-17** — `attach_eval_blocks=True` tutorial smoke tests added
  (`tests/core/test_phase_f17_paper17.py`, 2 tests). Closes the gap that no test
  exercised the full `ml_useful_macro_horse_race(attach_eval_blocks=True)` path
  end-to-end. Test 1 (`test_paper17_attach_eval_blocks_recipe_structure`) verifies
  that recipe dicts include `"5_evaluation"` and `"6_statistical_tests"` keys with
  correct sub-layers `L6_A_equal_predictive` and `L6_D_multiple_model`. Test 2
  (`test_paper17_attach_eval_blocks_true_runs_and_has_l6_artefacts`) verifies that
  `macroforecast.run()` executes without error on a minimal synthetic panel and that
  `result.cells` is non-empty — the observable proxy for paper Eq. (10) α_F
  treatment-effect regression execution (Coulombe et al. 2022 JAE §2.3).

### Phase F-02 slot 02 phantom replace (2026-05-13)

Run `2026-05-13-phase-f02-fmidas-replace`. Closes the F-02 MEDIUM prereq.

* **F-02** — Slot 02 of the 17-paper corpus remapped from phantom citation
  (`arctic_sea_ice_dfm`, Coulombe-Goebel 2021 VARCTIC paper which contains no DFM
  content) to Marcellino & Schumacher (2010) "Factor MIDAS for Nowcasting and
  Forecasting with Ragged-Edge Data." Oxford Bulletin of Economics and Statistics 72(4),
  518–550. DOI 10.1111/j.1468-0084.2010.00591.x. EUI preprint ECO-2008-16
  (cadmus.eui.eu handle 1814/8087).

  New helper `factor_midas_nowcast()` added to `macroforecast.recipes.paper_methods`.
  Implements the two-step Factor MIDAS pipeline as a pure recipe constructor wiring
  existing operational L3 ops: `op: "dfm"` (static PCA factor extraction) followed by
  `op: "u_midas"` (unrestricted MIDAS lag aggregation), then `family: "ols"` at L4.
  Exported via `__all__`.

  **Implementation assumptions** (PDF unavailable; reconstruction from abstract +
  established literature; PDF §-references pending post-acquisition verification):
  1. Factor extraction = static PCA (paper Method B), not Kalman smoother (Method D).
  2. MIDAS variant = unrestricted U-MIDAS, not parametric exp-Almon.
  3. Default `n_factors=1` (paper uses r=1–2 for German GDP).
  4. Default `n_lags_high="bic"` (BIC lag selection per Foroni-Marcellino-Schumacher 2015).

  Paper slot in `tests/core/test_paper_helpers_e2e.py` now live:
  `test_paper_02_factor_midas_nowcast` replaces the phantom placeholder comment.
  Test count: 15 → 16 e2e smoke tests.

  Module docstring updated: "16-paper" → "17-paper" corpus reference.

  **Note**: The DOI in the original dispatch prompt (`10.1016/j.ijforecast.2010.02.006`,
  IJF 26(4):581-587) was incorrect; that DOI resolves to Kuzin-Marcellino-Schumacher
  (2011) "MIDAS vs. Mixed-Frequency VAR." The correct DOI for Marcellino-Schumacher
  (2010) is 10.1111/j.1468-0084.2010.00591.x (OBES). Corrected in all code references.

### Phase DOCS-1 u_midas option_docs sync + Sphinx csv fix (2026-05-13)

Run `2026-05-13-phase-docs1-umidas-option-docs-sync`. Closes the DOCS-1 LOW prereq (follow-on to F-02).

* **DOCS-1** — `macroforecast/scaffold/option_docs/l3.py` `_OP_U_MIDAS` description
  updated to reflect the F-07-R paper-faithful implementation. Key changes:
  OLS is now documented as the default estimator (paper §3.2 p.11 "estimated by simple
  OLS"); ridge is noted as an explicit opt-in via `regularization='ridge'`. BIC
  lag-order selection (`n_lags_high='bic'` default, K_max = ceil(1.5 × freq_ratio))
  documented per paper §3.2 p.11 + §3.5. AR(1) y-lag term (`include_y_lag=True`
  helper default, μ₁ term of eq.(20)) documented. Stale wording "ridge / OLS / lasso"
  and `n_lags_high = 6` removed. Reference block updated to combine the Bundesbank
  Discussion Paper Series 1 No. 35/2011 and JRSS-A 178(1): 57-82 entries with
  DOI 10.1111/rssa.12043.

* **Sphinx warning** — `docs/for_researchers/user_data_workflow.md` line 31:
  fenced code block language identifier changed from `csv` to `text`. Suppresses
  `WARNING: Pygments lexer name 'csv' is not known [misc.highlighting_failure]`
  at Sphinx build time without altering the rendered output.

### Phase MC-RECAL paper-symmetric MC re-calibration (2026-05-13)

Run `2026-05-13-phase-mc-recal-paper-symmetric`. Closes the MC-RECAL LOW prereq.

* **MC-RECAL** — `tests/core/test_f07_umidas_tester.py` TEST-R4-01 (paper Table 2
  anchor) updated to paper-symmetric comparison. Both U-MIDAS (eq.20) and MIDAS
  baseline (eq.18) now include the AR(1) y-lag term. MIDAS baseline implements
  the full common-factor restriction `(1 - β_1 L^k) B(L,θ) x_{τk-1}`:
  `resid = y - β_0 - β_1·y_lag - β_2·agg_t + β_1·β_2·agg_tk` (5-parameter NLS).
  mean_ratio: 0.9928 (asymmetric, F-07-R4 baseline) → 0.9173 (paper Table 2
  anchor 0.91, 0.80% rel error). Tolerance [0.79, 1.03] UNCHANGED. Production
  U-MIDAS code (`runtime.py`, `paper_methods.py`) untouched — F-07-R closure
  preserved. (Foroni-Marcellino-Schumacher 2011/2015 OBES; Bundesbank DP 35/2011)

### Remaining v0.9.0 stable prereqs

The following three items must be resolved before the `v0.9.0` stable tag is cut:

| Prereq | Severity | Status | Notes |
|--------|----------|--------|-------|
| F-02: slot 02 phantom citation — Marcellino-Schumacher 2010 | MEDIUM | CLOSED | `factor_midas_nowcast()` added; slot 02 remapped to OBES 72(4) 518-550. |
| DOCS-1: `option_docs/l3.py` u_midas description sync + Sphinx csv warning | LOW | CLOSED | `_OP_U_MIDAS` description updated to F-07-R defaults (OLS default, BIC lag selection, AR(1) y-lag, DOI); csv→text fence fix. |
| MC-RECAL: paper-exact symmetric MC re-calibration | LOW | CLOSED | TEST-R4-01 updated to symmetric comparison (both models with AR term); MIDAS baseline upgraded to full common-factor eq.(18) 5-param NLS; mean_ratio 0.9928 → 0.9173 (paper anchor 0.91). |

---

## [0.9.0a0] -- 2026-05-07 -- "16-paper full-coverage cut (alpha pre-release)"

**Pre-release status**. The 2026-05-07 independent paper-vs-implementation
audit (3 subagents, 16 papers) found 12/16 ✅ Match and 4/16 ⚠️ Partial
after the v0.9.0F audit-fix dev stage. The remaining 4 gaps were
closed in v0.9.0F+ post-audit fixes:

* **Paper 4** VARCTIC Bayesian IRF posterior — ``_BayesianVAR`` now
  samples from the asymptotic multivariate-Normal posterior on VAR
  coefficients (``vec(β) ~ N(β̂, Σ_u ⊗ (Z'Z)⁻¹)``) when
  ``params['n_posterior_draws'] > 0``. Each draw produces a Cholesky-
  orthogonalised IRF; mean + 5/16/84/95 percentile bands cached on
  ``model._posterior_irf``. Helper
  ``_BayesianVAR._sample_posterior_irf`` + ``_compute_orth_irfs``.
  Pinv-based posterior covariance reconstruction tolerates collinear
  designs (common when X already contains lagged y).
* **Paper 9** HNN in-MLE constraint — replaced per-epoch post-batch
  bias rescaling with a Lagrangian penalty term
  ``λ_emphasis · (mean(h_v) − ν · var(y))²`` added to the NLL loss.
  Paper §3.2 Ingredient 2 (in-MLE constraint) now matched exactly.
  ``params['lambda_emphasis']`` defaults to 1.0; user may scale.
* **Paper 15** Macro Data Transforms 16-cell — new helper
  ``macroeconomic_data_transformations_horse_race`` returns
  ``{cell × family → recipe}`` enumerating Coulombe et al. (2021)
  Table 1's full grid (F / X / MARX / MAF / Level combinations × 6
  forecasting families). Users iterate and aggregate per-cell metrics.
* **Paper 16** ML for Macro 4-feature — new helper
  ``ml_useful_macro_horse_race`` returns ``{case × h × cv → recipe}``
  for the paper's 4-feature decomposition: nonlinearity (linear /
  KRR / RF), regularization (Lasso / EN / Ridge), CV (k-fold / POOS
  / AIC / BIC), loss (L2 / SVR ε-insensitive). Reference baselines
  (FM, OLS) included.

**Final audit verdict (post-paper-2 reclassification + 4-gap fix)**:
all 16 papers ✅ Match at the V verification standard.

* **Paper 2** Sea Ice DFM (Diebold / Göbel / Goulet Coulombe / Rudebusch
  / Zhang 2020 arXiv:2003.14276) — initial audit flagged ⚠️ on the
  assumption that the paper required Mariano-Murasawa mixed-frequency
  estimation. Re-audit on 2026-05-07 (post-PDF acquisition) confirms
  the paper is a *single-frequency* state-space DFM with Kalman filter
  + smoother on a 4-satellite SIE panel — exactly the procedure that
  ``_DFMMixedFrequency`` defaults to via statsmodels ``DynamicFactor``
  when ``mixed_frequency=False``. Verdict upgraded to ✅ Match.
* **Paper 4** Arctic VARCTIC — BVAR routing is now operational
  (v0.9.0F), but the Bayesian-Minnesota-with-statsmodels-VAR-IRF
  construction is a paper-implicit choice (Sims-style IRF over
  posterior-mean coefficients); no formal Bayesian IRF posterior.
* **Paper 9** HNN — Ingredient 4 reality check is implemented as a
  post-hoc log-linear rescaling (paper Eq. 9-10) but not as the
  in-MLE constraint paper §3.2 implicitly applies. The constraint
  enforcement is still per-epoch bias rescaling.
* **Paper 15** Macro Data Transformations Matter — pipeline supports
  the L3-rotation horse race; the 16-cell Z_t enumeration recipe
  from the paper's Table 1 is not shipped out of the box.
* **Paper 16** ML for Macro Forecasting — KRR is now a first-class L4
  family (v0.9.0F), but no out-of-the-box 4-feature × treatment
  horse-race recipe is shipped to reproduce the paper's main result.

The remaining 11/16 papers are paper-faithful at the V verification
standard. Users may treat the package as a near-faithful replication
substrate and adjust hyperparameters where conservative defaults
preserve back-compat (e.g. ``var_innovations=True`` for Sparse Macro
Factors, ``inner_n_estimators=1500`` for Booging).

Stable v0.9.0 will close the 5 remaining gaps in v0.9.0a1 / a2 / b0
release candidates.

### Post-2026-05-07 hardening (Phase D cycles, 2026-05-12)

After the 2026-05-07 initial alpha cut, four independent audit-fix dev stages
were applied on server1. None change the PyPI release version (still 0.9.0a0);
they harden paper-faithfulness and close LOW/MEDIUM audit findings.

**Round 5 anchor-free sweep** (2026-05-12, workflow 8, 16 papers):
- 4 ✅ strict Match / 10 ✅ PASS WITH NOTE (LOW) / 2 ⚠️ PASS WITH NOTE (MEDIUM) / 0 ❌ Mismatch
- MEDIUM gaps: Paper 13 (target-mode point_forecast vs cumulative_average) + Paper 15 (temporal_rule + F-branch lag missing in 12/16 Table 1 cells)
- All HIGH and CRITICAL findings from Round 1 closed.

**Phase D-1** (workflow 1): Paper 13 target-mode fix (`cumulative_average`) + Paper 15 `temporal_rule` + F-branch lag. Test count: 1311 → 1319 (+8 tests).

**Phase D-2a** (workflow 10 simplified): LOW housekeeping bundle — stale xfail decorators removed (papers 7, 9, 10, 12, 14), Almon positivity / n_slices / docstring / multi-seed cosmetic items. Test count: 1319 → 1327 (+8 tests).

**Phase D-2b** (workflow 10 simplified): Additional LOW items from Round 5. Test count: 1327 → 1327 (no net change; 0 new tests in this slice).

**Phase D-2c** (workflow 10 simplified): 9 paper-grounded LOW housekeeping items — docstring corrections (papers 2, 5, 10, 11), default value alignments (papers 3, 4, 9), error-quality improvement (paper 12 NN `ValueError`), structural test pin (paper 16 H⁺ `n_components`). HEAD `3e2d3b03`. Test count: 1327 → 1329 (+2 tests).

**Demote pattern**: 9 demotes total across the full audit-fix cycle (8 code-side, 1 report-side). Pattern stabilising — Round 5 surfaces no new code-side demotes.

**Final test count at HEAD `3e2d3b03`**: 1329 passing, 9 failed (pre-existing MRF `ndarray not contiguous` from vendored `np.matrix()` — not regressions), 25 skipped.



The single published cut after v0.8.9 that closes the 16-paper Phase-2
audit (2026-05-07): every paper in the user-curated reading list reaches
✅ Operational at the V verification standard. Five internal dev stages
build up to this release; none of v0.9.0A-E ship to PyPI separately.

### v0.9.0C — Tier 3 + Sparse Macro Factors (2026-05-07)

Four operational promotions, all paper-faithful:

* **AlbaMA** (``adaptive_ma_rf`` L3 op) — Goulet Coulombe & Klieber
  (2025) "An Adaptive Moving Average for Macroeconomic Monitoring". RF
  with K=1 (time index only); ``min_samples_leaf`` lower-bounds the
  realised window length. Modes: ``sided='two'`` (full-sample fit) /
  ``sided='one'`` (expanding-window per-t fit). Helper
  ``_adaptive_ma_rf``.
* **HNN** (``mlp.architecture=hemisphere`` + ``mlp.loss=volatility_emphasis``)
  — Goulet Coulombe / Frenette / Klieber (2025 JAE) Hemisphere Neural
  Networks. Common-core ReLU stack + dual head (mean + variance) +
  Gaussian NLL loss; volatility-emphasis ν enforced via per-epoch
  rescaling of the softplus head's bias. Helper ``_HemisphereNN``;
  requires ``[deep]`` extra (torch>=2.0).
* **Sparse Macro G3** — V2.5 follow-through. Two new L3 ops:
  - ``sparse_pca_chen_rohe`` (Chen-Rohe 2023 SCA, non-diagonal D, ℓ_1
    budget; alternating bilinear maximisation per Zhou-Rapach 2025
    eq. 4). Helper ``_sparse_pca_chen_rohe``.
  - ``supervised_pca`` (Giglio-Xiu-Zhang 2025; screen-then-PCA on top
    ``q · M`` columns by univariate correlation with target). Helper
    ``_supervised_pca``.
  Existing ``sparse_pca`` (sklearn / Zou-Hastie-Tibshirani 2006)
  unchanged; the three sparse-PCA-family ops are now distinct.

Nine new known-answer tests pin: AlbaMA piecewise-constant recovery
(MSE within noise floor) + one-sided NaN edge; HNN finite predictions
+ paper-coupled requirement (architecture + loss must both be set);
SCA factor recovery (>0.9 corr with truth) + SPCA target alignment
(>0.5 corr) + SCA distinct from sklearn SparsePCA.

### v0.9.0D — anatomy adapter Path B (2026-05-07)

Goulet Coulombe et al. forwards work via Borup, Goulet Coulombe,
Rapach, Montes Schütte & Schwenk-Nebbe (2022) "Anatomy of Out-of-Sample
Forecasting Accuracy". Operational L7 ops:

* ``oshapley_vi`` -- mean of |per-instance Shapley values| across OOS
  rows (paper Eq. 16). Default identity transformer.
* ``pbsv`` -- global squared-error transformer; signed coefficients
  surface loss-reducing predictors per paper Eq. 24.

Both derive from a single ``Anatomy.explain(transformer=...)`` call.
The 2026-05-07 audit (errata E3) corrected the v09 plan sketch which
referenced non-existent ``Anatomy.oshapley_vi(...)`` /
``Anatomy.pbsv(...)`` methods.

Path B uses the *final-window* fitted model for every period; status
column = ``"degraded"`` to signal the audit-flagged approximation.
Selected automatically when ``params["initial_window"]`` is absent.

### v0.9.0F — Audit-fix dev stage (2026-05-07)

Three independent paper-vs-implementation audits (3 subagents, 16 papers
each on a separate slice) found that v0.9.0A-E had 6/16 ✅ Match and
10/16 ⚠️ Partial — the previous "16/16 operational" claim was an
overclaim. v0.9.0F closes the eight P0/P1 audit findings. Aggregate
flips to **paper-faithful for the canonical paths**, with conservative
defaults that the user may still adjust.

**P0 — Critical paper-defining gaps:**

* **Paper 9 HNN** (Goulet Coulombe / Frenette / Klieber 2025 JAE) —
  Three paper Ingredients were missing or stubbed:
  - **Ingredient 2 (ν proxy)**: replaced default ν=0.5 with a paper-
    faithful plain-NN OOB residual proxy (paper p.11 footnote 2:
    ``ν = mean(ε̂²_NN) / var(y)``, capped at 0.99). Helper
    ``_compute_nu_proxy``.
  - **Ingredient 3 (blocked subsamples)**: replaced random row
    sampling with contiguous time-block draws per Eq. 8. Helper
    ``_blocked_subsample``.
  - **Ingredient 4 (reality check)**: implemented Eq. 9-10 log-linear
    regression of OOB squared residuals on log(h_v) and the bootstrap
    correction ``ς̂``. Predict-time variance can now be corrected via
    ``h_v ← exp(adj_intercept) · h_v_raw ** adj_slope``. Helper
    ``_apply_reality_check``.
  - Bumped default B from 50 → 100 (still below paper's 1000 for
    L4 cell-level cost).
* **Paper 14 Sparse Macro Factors** (Rapach & Zhou 2025) — added the
  **VAR(1) innovation step** (paper Strategy step 2). Opt-in via
  ``params['var_innovations']=True``; output columns rename to
  ``scaf_*`` (sparse macro-finance factors) instead of ``sca_*``
  (sparse PC scores). The default (False) preserves v0.9.0C-3 behaviour.
* **Paper 12 Dual Interpretation portfolio metrics** — fixed
  ``leverage`` from L1 norm to **signed sum** per paper Eq. p.21
  (``FL = Σ w_{ji}``); fixed ``short`` to signed ≤ 0 per paper. Legacy
  absolute-value variants surfaced as ``leverage_l1`` and ``short_abs``
  for backward-compatible plotting.
* **Paper 4 VARCTIC BVAR routing** — ``_BayesianVAR.fit`` now also
  fits a parallel statsmodels ``VAR(p)`` and exposes ``_results``
  alongside the closed-form posterior coefficients. The L7 IRF /
  FEVD / historical_decomposition ops can now route on a Bayesian
  Minnesota fit (Sims-style IRF construction over the posterior-mean
  coefficient matrix is what the paper actually does).

**P1 — Important default mismatches:**

* **Paper 11 Anatomy** — bumped ``n_iterations`` default 50 → 500 to
  match paper M=500 (Borup et al. 2022 p.16 footnote 16). Path A now
  also marks the period as ``"degraded"`` when sklearn-clone fails on
  a custom estimator (was: silently used final-window fit while still
  labelling result "operational").
* **Paper 3 SGT** — added paper p.87 rule-of-thumb defaults:
  ``eta_depth_step=0.01`` (was 0.0), ``eta_max_plateau=0.5`` (was 1.0
  via the clip). Effective plateau is ``max(self.eta, plateau)`` so
  η=1 (CART parity) still works as expected. Surfaced ``mtry_frac``
  sub-axis (paper p.88 §2.3 specifies mtry=0.75); default 1.0 = scan
  every column (paper-silent baseline).
* **Paper 6 Booging** — bumped ``inner_n_estimators`` default 500 →
  1500 to match paper Table 2 / Appendix B's "deliberately overfit"
  prescription. Below 1500, the bag-prune theorem's pruning effect is
  weakened.
* **Paper 16 KRR (Kernel Ridge Regression)** — registered as a
  first-class L4 family (``kernel_ridge``) backed by sklearn
  ``KernelRidge``. Paper headline non-linearity feature (Eq. 16,
  §3.1.1) was previously not exposable as a recipe family; the
  Nystroem-features approximation was the only available route.

The P0 fixes alter user-facing numerics: any pipeline previously
relying on the v0.9.0E ``leverage`` value, the v0.9.0E sparse_pca_chen_rohe
output (now opt-out of VAR(1) by default to preserve back-compat),
the v0.9.0E ``oshapley_vi`` / ``pbsv`` (M=50 vs new M=500), or the
v0.9.0E HNN (ν=0.5 default vs new NN-OOB proxy) will produce
*different* numbers in v0.9.1. The P1 changes adjust defaults but the
old behaviour is reachable via explicit param overrides.

After v0.9.0F + post-audit reclassifications + 4-gap fix
(2026-05-07), the audit verdict is:

| Verdict | Count | Papers |
|---|---|---|
| ✅ Match | **16/16** | 1, 5, 7, 8, 10, 13 (v0.9.0E baseline) + 3, 6, 11, 12, 14 (v0.9.0F audit-fixed) + 2 (post-audit reclassification post-PDF acquisition arXiv:2003.14276) + 4 (BVAR posterior IRF added), 9 (in-MLE Lagrangian penalty), 15 (16-cell horse-race helper), 16 (4-feature horse-race helper + KRR L4 family) |

### v0.9.0E — anatomy Path A (faithful per-origin refit; 2026-05-07)

When the recipe sets ``params["initial_window"]``, the adapter routes
through ``AnatomySubsets.generate(EXPANDING, initial_window=W)`` and
refits a fresh sklearn-cloned model at every walk-forward origin. Status
column flips to ``"operational"``. The L4 layer gains no new artifact
(``L4PerOriginModelsArtifact`` was the original spec but the
``sklearn.base.clone(fitted) + per-period .fit()`` path achieves the
same paper-faithful semantics without an L4 schema change).

When ``params["initial_window"]`` is absent, Path B (degraded) is
selected; the two paths share the same code path and differ only in
the AnatomySubsets schedule.

### v0.9.0B item 6 — SGT `decision_tree.split_shrinkage` operational (2026-05-07)

Goulet Coulombe (2024) "Slow-Growing Trees" (SLOTH). The 2026-05-07
audit (errata E1) confirmed that the previous "post-fit, multiply each
leaf value by ``(1-η)^depth``" sketch was wrong — paper Algorithm 1
applies the soft weight ``(1 − η)`` *in-fit* during weight propagation
at every split. sklearn ``DecisionTreeRegressor`` cannot reproduce this
because the *splits themselves* depend on soft weights (every row,
including rule-violators, contributes to the SSE objective).

* New helper class ``_SlowGrowingTree`` in ``core/runtime.py`` implements
  Algorithm 1 from scratch. Iterative BFS construction, weighted-SSE
  split search, Herfindahl ``H_l ≡ Σω² / (Σω)²`` stopping rule.
* ``decision_tree.split_shrinkage = η`` (formerly future-gated) is now
  operational. Sub-axis params ``herfindahl_threshold`` (default 0.25),
  ``eta_depth_step`` (paper rule-of-thumb default 0.0), ``max_depth``.
* Limit cases pinned by tests: η = 1 → CART-like high-R² fit; η < 1 →
  smoother fit (paper Figure 2 SLOTH vs CART distinction).
* Soft-weighted predict path traverses both branches with propagated
  test weights ``w_test_branch ← w · (1 - η · I[rule violated])``;
  prediction is the leaf-weighted-mean aggregate.
* ``_F_DECISION_TREE`` OPTION_DOCS rewritten to document the soft-weight
  algorithm + sub-axis params.
* Four new known-answer tests in ``tests/core/test_v09_paper_coverage.py``.

### v0.9.0B item 5 — `dual_decomposition` for tree-bagging ensembles (2026-05-07)

Goulet Coulombe / Goebel / Klieber (2024) §3.2: random-forest predictions
admit a clean dual representation as a weighted sum of training targets
via the **leaf-co-occurrence kernel**:

  ``wⱼ(xₜ) = (1/B) Σ_b 1[j ∈ B_b] · 1[leaf_b(xₜ) == leaf_b(xⱼ)] / leaf_size_b(xⱼ)``

with ``B_b`` the bootstrap subset for tree b (sklearn's
``estimators_samples_``). Reproduces ``forest.predict`` to machine
precision (~4e-16) on both ``RandomForestRegressor`` (bootstrap=True
with sampling-with-replacement multiplicity) and
``ExtraTreesRegressor`` (bootstrap=False default).

* New helper ``_rf_leaf_cooccurrence_weights`` in ``core/runtime.py``.
* ``_dual_decomposition_frame`` ladder extended:
  - ``hasattr(coef_)`` → existing linear closed-form;
  - tree-bagging (RF / ExtraTrees) → leaf-co-occurrence kernel;
  - everything else → NotImplementedError with redirect.
* ``GradientBoostingRegressor`` deliberately rejected: residual-stage
  bagging does not factor into a sum-of-training-targets representation.
* ``frame.attrs["method"]`` carries ``"linear_closed_form"`` or
  ``"rf_leaf_cooccurrence_kernel"`` for downstream renderers.
* ``_DUAL_DECOMPOSITION`` OPTION_DOCS rewritten: drops "non-linear
  future" language, adds the RF / ExtraTrees paragraph + paper §3.2
  reference.
* Three new known-answer tests (RF bit-exact with bootstrap, ExtraTrees
  bit-exact, GBM rejection); existing v0.8.9 reject-test updated to
  exercise the GBM family now that RF is operational.

### v0.9.0B item 4 — Booging `bagging.strategy=booging` operational (2026-05-07)

Goulet Coulombe (2024) "To Bag is to Prune" (arXiv:2008.07063). The
2026-05-07 audit (errata E2) confirmed that the previous "K rounds:
bag-on-residuals" sketch did not match the paper. The actual algorithm:

* **Outer bagging** of ``B = 100`` subsamples (sampling-without-
  replacement at fraction ``0.75``);
* **Inner Stochastic Gradient Boosted Trees** with ``n_estimators=500``
  set high (over-fit on purpose), ``learning_rate=0.1``, ``max_depth=4``,
  ``subsample=0.5`` for intra-boost row stochasticity;
* **Data Augmentation**: each predictor column ``X_k`` duplicated as
  ``X̃_k = X_k + N(0, (σ_k · da_noise_frac)²)``, ``da_noise_frac=1/3``;
* **Per-bag column dropping** at ``da_drop_rate=0.2``;
* **Bag-prune theorem** (paper §2): outer bagging replaces tuning the
  boosting depth ``S`` -- over-fit the inner SGB and let the bag
  average prune.

* New helper class ``_BoogingWrapper`` in ``core/runtime.py``.
* L4 dispatch routes ``bagging.strategy='booging'`` to the new wrapper.
* ``sequential_residual`` is retained as a legacy alias for back-compat;
  the option now routes to the same outer-bagging-of-inner-SGB
  construction (the alias preserves recipe-level back-compat for any
  user that adopted the schema-only name in v0.8.9).
* ``_F_BAGGING`` OPTION_DOCS rewritten to describe all three strategies
  (``standard`` / ``block`` / ``booging``) with paper-faithful prose.
* Encyclopedia regenerated (189 pages, no count drift).
* Four new known-answer tests in ``tests/core/test_v09_paper_coverage.py``:
  outer-bag fitting, alias dispatch parity, DA design-width verification,
  end-to-end recipe smoke.

### v0.9.0B items 2-3 — Maximally FL Albacore priors operational (2026-05-07)

Goulet Coulombe / Klieber / Barrette / Goebel (2024) "Maximally Forward-
Looking Core Inflation" introduces two assemblage-regression variants
that decompose into ridge sub-axis options. Both are constrained
generalised ridge fits solved via scipy SLSQP.

* **G1 ``ridge.prior=shrink_to_target`` (Albacore_comps Variant A)**.
  ``arg min ‖y − Xw‖² + α‖w − w_target‖²`` s.t. ``w ≥ 0``, ``w'1 = 1``
  (Eq. 1 of the paper). Closed-form unconstrained solution
  ``(X'X + αI)⁻¹(X'y + αw_target)`` projected onto the simplex via
  SLSQP. Helper class ``_ShrinkToTargetRidge``. Sub-axis params:
  ``prior_target`` (default uniform ``1/K``), ``prior_simplex`` (default
  True). Limit cases pinned by tests:
  - α = 0 → unconstrained / NNLS / OLS (recovers convex-combo truth);
  - α → ∞ → returns ``w_target`` exactly;
  - ``w_target=0``, simplex off, nonneg → equivalent to ``_NonNegRidge``.

* **G2 ``ridge.prior=fused_difference`` (Albacore_ranks Variant B)**.
  ``arg min ‖y − Xw‖² + α‖Dw‖²`` s.t. ``w ≥ 0``,
  ``mean(y) = mean(Xw)``, where D is the first-difference operator
  (Eq. 2). Pairs with the L3 ``asymmetric_trim`` op (B-6 v0.8.9) for
  rank-space transformation. Helper class ``_FusedDifferenceRidge``.
  Sub-axis params: ``prior_diff_order`` (default 1), ``prior_mean_
  equality`` (default True). Limit cases pinned by tests:
  - α = 0 → standard OLS (with mean-equality off);
  - α → ∞ → uniform weights (level pinned by mean-equality);
  - mean-equality holds to 1e-4 in finite samples.

Both variants compose with ``coefficient_constraint=nonneg`` at the L4
dispatch level. Seven new known-answer tests in
``tests/core/test_v09_paper_coverage.py`` pin the limit cases and the
end-to-end recipe-runs-to-completion path.

### v0.9.0B item 1 — 2SRR `ridge.prior=random_walk` operational (2026-05-07)

Coulombe (2025 IJF) "Time-Varying Parameters as Ridge Regressions" two-
step closed-form generalised ridge. Eq. 11 of the paper:

* Step 1: ``β̂ = C Z' (Z Z' + λ I_T)⁻¹ y`` where ``Z = WC``,
  ``W = [diag(X_1) | ... | diag(X_K)]``, ``C = I_K ⊗ C_RW`` and
  ``C_RW`` is lower-triangular ones (cumulative-sum operator).
* Step 2: recover per-time residual variance ``σ²_ε(t)`` via EWMA
  (default; RiskMetrics λ=0.94) or GARCH(1,1) (when ``arch>=5.0``);
  rescale per-coefficient ``σ²_u`` to mean ``1/λ``; solve
  ``θ̂ = Ω_θ Z' (Z Ω_θ Z' + Ω_ε)⁻¹ y``.

Two ``T × T`` matrix solves; **no iteration**. The 2026-05-07 audit
confirmed the closed-form interpretation of Eq. 11 and resolved the
risk-register entry that had hedged toward iterative ridge fallback.

* New helper class ``_TwoStageRandomWalkRidge`` in ``core/runtime.py``.
* ``ridge.prior=random_walk`` dispatch routes to the new helper instead
  of raising NotImplementedError.
* ``_F_RIDGE`` OPTION_DOCS marks the option operational; adds
  ``params.vol_model`` (``ewma`` default / ``garch11``) sub-axis.
* Predict semantics: one-step-ahead under the random-walk assumption
  ``β_{T+1} ≈ β_T``; the wrapper exposes the full per-time β̂ path
  via ``model._beta_path`` (T × K) for L7 GTVP-style consumption.
* Four known-answer tests in
  ``tests/core/test_v09_paper_coverage.py``: random-walk truth recovery,
  α→∞ static-OLS limit, NaN robustness, end-to-end recipe smoke.

### v0.9.0A — Errata patch + paper-10 doc cross-ref (2026-05-07, in progress)

Closes documentation gaps before any algorithmic work. Two changes:

* **v09_paper_coverage_plan.md errata**. Four corrections from the
  2026-05-07 deep-dive (each item separately enumerated in the plan
  document):
  - **E1 SGT** (``decision_tree.split_shrinkage``): the previous "post-
    fit, multiply each leaf value by ``(1-η)^depth``" sketch is
    incorrect. Goulet Coulombe (2024) Algorithm 1 applies the soft
    weight ``(1 − η)`` *in-fit* during weight propagation at every
    split. sklearn extension is insufficient → custom soft-weighted
    tree implementation required (~250-350 LOC, effort 1-2 d → 5-7 d).
  - **E2 Booging** (``bagging.strategy=sequential_residual``): the
    previous "outer K rounds: bag a base learner on residuals" sketch
    does not match the paper. The actual algorithm is *outer B = 100
    bags of (intentionally over-fitted) inner Stochastic Gradient
    Boosted Trees + Data Augmentation*. Schema option to be renamed
    ``booging`` in v0.9.0B with ``sequential_residual`` retained as
    alias for back-compat.
  - **E3 anatomy adapter API**: anatomy 0.1.6 has *no* dedicated
    ``oshapley_vi(...)`` / ``pbsv(...)`` methods. Both metrics are
    derived from a single ``Anatomy.explain(transformer=...)`` call
    with different ``AnatomyModelOutputTransformer`` instances. The
    plan's Phase 4 sketch must be replaced.
  - **E4 anatomy per-origin refit**: Borup paper Eq. 11 requires the
    per-origin fitted model. macroforecast's ``L4ModelArtifactsArtifact``
    keeps a single fitted_object per ``model_id``. Path B (degraded,
    final-window fit + warning) ships in v0.9.0D; Path A (faithful,
    new ``L4PerOriginModelsArtifact``) in v0.9.0E.
  - **2SRR closed-form RESOLVED**: the risk-register entry "fall back
    to iterative ridge" can be deleted — paper Eq. 11 is unambiguously
    closed-form, two ``T × T`` solves.
  - Three new schema rows added: G1 ``ridge.prior=shrink_to_target``
    (Maximally FL Albacore_comps), G2 ``ridge.prior=fused_difference``
    (Maximally FL Albacore_ranks), G3 ``sparse_pca_chen_rohe`` +
    ``supervised_pca`` (V2.5 follow-through for Sparse Macro Factors).

* **Paper 10 (OLS as Attention Mechanism) doc cross-ref**. Goulet
  Coulombe (2026, SSRN 5200864) shows OLS predictions coincide with a
  restricted attention module (paper eqs. 17-19). The compute is
  identical to the closed-form ridge representer already implemented
  in the ``dual_decomposition`` L7 op (operational since v0.8.9 B-3),
  so no new runtime is needed. The op's ``OPTION_DOCS`` entry now
  carries the Goulet Coulombe (2026) reference and a description
  paragraph documenting the equivalence; encyclopedia regenerated.

## [0.8.9] -- 2026-05-06 -- "Phase 1 paper-coverage promotions + groundwork"

Combined cut that lands the v0.9.x paper-coverage groundwork (schema +
recipe gallery + helper module) **plus the Phase 1 Tier 1 algorithmic
promotions** of five atomic primitives that decompose into closed-form
or scipy-based implementations.

The v0.9.x algorithmic-promotion plan lives at
``docs/architecture/v09_paper_coverage_plan.md``.

### Honesty-pass fix (V2.4: DFM mixed-frequency vs Mariano-Murasawa 2003)

The verification audit
(``docs/architecture/v089_verification_results.md`` § V2.4) found that
the ``dfm_mixed_mariano_murasawa`` family's mixed-frequency code path,
operational since v0.25 #245, **had never run successfully**. Two bugs
caused every mixed-frequency recipe to silently degrade to the single-
frequency ``DynamicFactor`` path (a generic DFM, not the M-M aggregator):

* The runtime passed ``endog_quarterly=...`` together with
  ``k_endog_monthly=len(monthly)`` to
  ``statsmodels DynamicFactorMQ``; statsmodels rejects this combination
  with ``ValueError`` ("``k_endog_monthly`` cannot be specified when
  ``endog_quarterly`` is given"). The silent ``try/except`` caught the
  exception and routed the user into the single-frequency fallback
  without warning.
* When users supplied quarterly variables NaN-padded at non-quarter-end
  months on a monthly DateTimeIndex (the natural shape coming out of a
  FRED-MD + FRED-QD panel), statsmodels rejected the input because its
  quarterly-endog contract requires a quarterly-frequency
  DateTimeIndex (``freqstr`` starting with 'Q'). The runtime had no
  index-normalisation step.

v0.8.9 lands two surgical patches in
``_DFMMixedFrequency._fit_mixed_frequency``:

* Drop ``k_endog_monthly`` from the kwargs when ``endog_quarterly`` is
  non-None (statsmodels infers it).
* Drop the all-NaN rows in the quarterly block and reindex to a
  quarterly DateTimeIndex with ``freq='QE'`` (pandas 3.0 spelling;
  statsmodels' frequency check inspects only ``freqstr[0]``).

Two diagnostic attributes added for future regression detection:

* ``_mq_failure_reason: str | None`` -- populated when MQ requested
  but did not run, replacing the previous "exception swallowed" path.
* ``_idiosyncratic_ar1: bool | None`` -- ``True`` when the
  Mariano-Murasawa (2010) Eq. 4 AR(1)-idiosyncratic spec was active,
  ``False`` when the runtime fell back to i.i.d. idiosyncratic
  errors, ``None`` when MQ did not run.

This is a behaviour change: recipes declaring ``mixed_frequency=True``
will now actually run the Mariano-Murasawa (2003) monthly-state
aggregator instead of silently using the single-frequency fallback.
Forecasts will differ from v0.25--v0.8.6 outputs for the same recipes.

Four known-answer tests pin the behaviour:
``test_v24_dfm_mq_pure_monthly_uses_mariano_murasawa_2010_ar1``
(idiosyncratic AR(1) default),
``test_v24_dfm_mq_mixed_m_q_handles_quarterly_nan_padded_input``
(NaN-padded quarterly input → quarterly index conversion),
``test_v24_dfm_single_frequency_falls_back_to_state_space_dfm``
(non-MQ default still uses ``DynamicFactor`` Kalman MLE), and
``test_v24_dfm_mq_failure_surfaces_in_diagnostic_attribute``
(no-monthly-anchor case → ``_mq_failure_reason`` set instead of silent
degradation).

### Honesty-pass fix (V2.3: VAR ops vs Coulombe & Göbel 2021)

The verification audit (``docs/architecture/v089_verification_results.md``
§ V2.3) found two L7 ops registered as operational since v0.2 #189
that did not match their named procedures:

* **`generalized_irf` was misnamed.** The op was named after Pesaran-
  Shin (1998) generalized IRF (order-invariant), but the runtime
  returned ``statsmodels orth_irfs`` (Cholesky orthogonalised IRFs --
  order-dependent). Two distinct algorithms.

  v0.8.9 splits this into two ops: **`orthogonalised_irf`** (operational,
  routes to the existing Cholesky path -- numerical output unchanged)
  and **`generalized_irf`** (future-gated, runtime raises
  ``NotImplementedError`` with a Pesaran-Shin paper reference and a
  redirect to ``orthogonalised_irf``). v0.9.x will add a real
  Pesaran-Shin runtime under the existing name.

  Replication recipes (``examples/recipes/replications/arctic_var.yaml``,
  ``recipes/paper_methods.varctic_arctic_amplification``) updated to
  use ``orthogonalised_irf``.

* **`historical_decomposition` was an importance proxy, not HD.** The
  v0.2 #189 runtime returned ``|orth_irfs|.sum × std(resid)`` per
  shock -- a time-invariant quantity that ignored the actual
  realisation of structural shocks. The Burbidge-Harrison (1985)
  historical decomposition expresses the *path* of each variable as a
  convolution of orthogonalised IRFs with the time series of
  recovered structural shocks.

  v0.8.9 rewrites the runtime in ``_var_impulse_frame`` to compute
  the canonical HD: Cholesky-decompose Σᵤ, recover structural shocks
  ``e*_t = P⁻¹ u_t``, convolve with the orth_irfs to produce the
  per-time-step contribution table ``hd[t, i, j]``, and surface the
  per-shock cumulative absolute contribution to the target variable.
  Two known-answer tests pin (i) the reconstruction-magnitude lower
  bound (total importance is on the order of the realised target's L1
  fluctuation), and (ii) path dependence (different residual
  realisations produce different importance vectors -- the previous
  proxy was nearly constant across draws).

This is a behaviour change: recipes using ``historical_decomposition``
will produce different importance numbers (the new ones are the
correct Burbidge-Harrison decomposition; the old ones were a proxy
score). Recipes using ``generalized_irf`` will need to switch to
``orthogonalised_irf`` (one-line edit) -- the Cholesky output is
numerically identical to the v0.2 implementation.

### Honesty-pass fix (`macroeconomic_random_forest` re-anchored to mrf-web)

The ``macroeconomic_random_forest`` family (operational since v0.2 #187)
previously shipped an **in-house ``_MRFWrapper``** that augmented ``X``
with a normalised time trend, fit a sklearn ``RandomForestRegressor``,
and attached a per-leaf ``LinearRegression``. The verification audit
(``docs/architecture/v089_verification_results.md`` § V2.2) found that
this implementation matched **only the per-leaf linear piece** of
Goulet Coulombe (2024) and was missing two paper-defining pieces:

* the random-walk kernel / Olympic-podium regularisation that gives the
  GTVPs their time-smoothness;
* the Block Bayesian Bootstrap (Taddy 2015 extension) ensemble that
  the paper uses to surface forecast intervals.

v0.8.9 ships a new ``_MRFExternalWrapper`` (in ``core/runtime.py``) that
delegates the algorithm to Ryan Lucas's reference implementation,
**vendored under ``macroforecast/_vendor/macro_random_forest/``** with
four surgical numpy 2.x / pandas 2.x compatibility patches (full list:
``macroforecast/_vendor/macro_random_forest/PATCHES.md``). **No
algorithmic changes** -- the numerical output of ``_ensemble_loop()``
matches the upstream package on environments where both can run.
Upstream URL: <https://github.com/RyanLucas3/MacroRandomForest>.

Vendoring (instead of an external ``[mrf]`` extra) avoids dragging
users through a separate PyPI install for what is in practice a hard
dependency for the ``macroeconomic_random_forest`` family. The MIT
licence is preserved alongside the source; see
``THIRD_PARTY_NOTICES.md`` at the repository root for the consolidated
attribution table.

* No new extra. The family works out of the box once
  ``pip install macroforecast`` is done.
* New params on the family: ``B`` (bootstrap iterations, default 50),
  ``ridge_lambda`` (0.1), ``rw_regul`` (0.75), ``mtry_frac`` (1/3),
  ``trend_push`` (1), ``quantile_rate`` (0.3), ``fast_rw`` (True),
  ``parallelise`` (False), ``n_cores`` (1). Old ``n_estimators`` is
  honoured as an alias for ``B``; ``max_depth`` is silently ignored
  (mrf-web uses RW + ridge regularisation instead of tree depth).
* L7 ``mrf_gtvp`` consumer rewired to read the GTVP β̂(t) series
  directly from ``_cached_betas`` (populated by the most recent
  ``predict`` call) -- shape ``(T, K+1)`` with column 0 = intercept.
  Importance now uses ``np.nanmean(|β|)`` because oos rows are not
  covered by the in-sample bootstrap and arrive as NaN.

This is a behaviour change: recipes using
``macroeconomic_random_forest`` will produce different forecasts (the
new implementation runs the full paper procedure, not just the leaf-
linear shortcut). The previous in-house wrapper is removed.

**Citation requirement**: research using this family must cite Goulet
Coulombe (2024) "The Macroeconomy as a Random Forest" (Journal of
Applied Econometrics, arXiv:2006.12724) and acknowledge the upstream
implementation by Ryan Lucas
(<https://github.com/RyanLucas3/MacroRandomForest>). Both citations
are listed in the OPTION_DOCS entry for ``macroeconomic_random_forest``
and surfaced in the encyclopedia regen.

### Honesty-pass fix (`scaled_pca` runtime rewrite)

The ``scaled_pca`` L3 op (operational since v0.1) previously
implemented a **row-wise ``|target|`` weighting** of observations,
which is **not** the paper's algorithm. Huang/Jiang/Li/Tong/Zhou
(2022) "Scaled PCA: A New Approach to Dimension Reduction"
(Management Science 68(3)) defines sPCA as a **column-wise predictive-
slope β scaling**: for each column j, fit univariate OLS of target on
the standardised column, scale the column by the resulting β_j, then
PCA.

v0.8.9 ships ``_scaled_pca_huang_zhou`` -- a paper-faithful
implementation matching the authors' MATLAB ``sPCAest.m`` to machine
precision (β coefficients agree at ~1e-16, factor directions identical).
Regression test
``test_v21_scaled_pca_matches_huang_zhou_2022_authors_matlab`` pins
the new behaviour. See
``docs/architecture/v089_verification_results.md`` § V2.1 for the
full audit.

This is a behaviour change: recipes that depended on the previous
(non-paper) row-weighted variant will produce different factors. For
the small number of users on that path, the previous algorithm is
documented retrospectively as ``target_row_weighted_pca`` (not a
registered op; trivial to recover via L3 ``scale + pca`` if needed).

### Phase 1 Tier 1 promotions (5 atomic primitives, operational)

Each promotion has a known-answer test in
``tests/core/test_v09_paper_coverage.py``:

* **`ridge.coefficient_constraint=nonneg`** -- non-negative ridge via
  ``scipy.optimize.nnls`` on the augmented system
  ``[X; sqrt(α)·I] β = [y; 0]``, β >= 0. Backbone for the Albacore-
  family Assemblage Regression (Goulet Coulombe et al. 2024 "Maximally
  Forward-Looking Core Inflation"). Helper class ``_NonNegRidge``.
* **`bagging.strategy=block`** -- moving-block bootstrap (Künsch 1989)
  inside ``_BaggingWrapper``: replaces i.i.d. resampling with
  consecutive ``block_length``-row blocks, preserving short-range
  serial dependence. Used for serially-correlated macro panels and
  the Taddy 2015 ext. cited by Coulombe (2024) MRF.
* **`dual_decomposition`** (linear families) -- representer-theorem
  closed-form ``w(xₜ) = X(X'X + αI)⁻¹xₜ`` for ridge / OLS / lasso
  (Goulet Coulombe / Goebel / Klieber 2024). Output frame carries
  inline portfolio diagnostics (HHI / short / turnover / leverage)
  via ``frame.attrs['portfolio_metrics']``. Non-linear extensions
  (kernel / RF leaf-co-occurrence) deferred to v0.9.x Phase 2.
  Helper function ``_dual_decomposition_frame``.
* **`blocked_oob_reality_check`** -- block-bootstrap variant of White
  (2000) reality check on per-origin loss differentials vs a named
  benchmark. Reject H0 when median bootstrap MSE_diff < 0 at α=0.05.
  Used for the v0.9.x HNN (Coulombe / Frenette / Klieber 2025 JAE)
  evaluation pipeline. Helper function
  ``_blocked_oob_reality_check_p_values``.
* **`asymmetric_trim`** (L2/L3) -- rank-space transformation: per-
  period sort of a ``(T x K)`` component panel into the corresponding
  matrix of order statistics. Asymmetric trimming emerges in the
  *downstream* nonneg ridge that learns rank-position weights; the op
  itself does the sort transformation only. Algorithm spec at
  ``docs/replications/maximally_forward_looking_algorithm_notes.md``.
  Helper function ``_asymmetric_trim``.

### Decomposition discipline (new architectural principle)

A method published in a paper gets a new L4 family / L3 op / L7 op only
when it is truly atomic. If the method decomposes into existing
primitives plus sub-axis options on existing families, it is captured as:

* a parametric sub-axis on the existing family (`ridge.prior`,
  `ridge.coefficient_constraint`, `decision_tree.split_shrinkage`,
  `bagging.strategy`, `extra_trees.max_features`, `mlp.architecture`,
  `mlp.loss`)
* a recipe pattern in `examples/recipes/replications/<paper>.yaml`
* a Python helper in `macroforecast.recipes.paper_methods`

This keeps the registry small and forces every paper to expose its
algorithmic content at the recipe level rather than hide it behind a
paper-named family option.

### Added (atomic primitives)

L4 family additions:

* **`mars`** -- Friedman (1991) Multivariate Adaptive Regression Splines.
  **Operational** via `pyearth` optional dep (`pip install
  macroforecast[mars]`); raises `NotImplementedError` with install hint
  when the extra is missing (mirrors xgboost / lightgbm / catboost
  pattern). Required as the base learner for Coulombe (2024) MARSquake
  recipe (`bagging(base_family=mars, ...)`).

L3 op additions:

* **`savitzky_golay_filter`** -- **operational**, wraps
  `scipy.signal.savgol_filter`. Used as the fixed-window baseline
  against AlbaMA's adaptive-window estimator (Coulombe & Klieber 2025).
* **`adaptive_ma_rf`** (AlbaMA) -- schema-only future. Coulombe & Klieber
  (2025) arXiv:2501.13222 §3 RF-driven adaptive moving-average window.

L2 / L3 op addition:

* **`asymmetric_trim`** -- **operational** (see Phase 1 Tier 1
  promotions). Layer scope expanded to ``(l2, l3)`` so the L3 DAG can
  dispatch it. Coulombe / Klieber / Barrette / Goebel (2024) Albacore-
  family rank-space transformation.

L5 op addition:

* **`blocked_oob_reality_check`** -- **operational** (see Phase 1
  Tier 1 promotions). HNN block-bootstrap variant of White (2000)
  reality check on per-origin loss differentials.

L7 op additions:

* **`dual_decomposition`** -- **operational for linear families** (see
  Phase 1 Tier 1 promotions above). Output artifact also carries
  inline portfolio diagnostics (HHI / short / turnover / leverage)
  from the same paper -- these are trivial numpy reductions on the
  dual weights and do not warrant their own L7 op (decomposition
  discipline). Non-linear extensions (kernel / RF) deferred to v0.9.x
  Phase 2.
* **`oshapley_vi`** + **`pbsv`** -- schema-only future. Borup et al.
  (2022) "Anatomy of OOS Forecasting Accuracy" SSRN 4278745. Runtime
  delegates to the `anatomy` PyPI package once the L7 adapter lands;
  `[anatomy]` extra registered.

### Added (sub-axis options on existing families)

Each sub-axis is documented in the corresponding family's OPTION_DOCS
prose (encyclopedia surfaces inline). Default values preserve existing
runtime behaviour; non-default values trigger a clear
`NotImplementedError` until the v0.9.x runtime promotion.

* **`extra_trees.max_features`** -- **operational** (sklearn pass-through).
  `max_features=1` implements Coulombe (2024) PRF baseline.
* **`ridge.prior`** -- schema-only future. RW kernel implements 2SRR.
* **`ridge.coefficient_constraint`** -- **operational** (see Phase 1
  Tier 1 promotions). `nonneg` value invokes ``_NonNegRidge`` for
  Albacore Assemblage Regression.
* **`decision_tree.split_shrinkage`** -- schema-only future. SLOTH.
* **`bagging.strategy`** -- `block` value **operational** (see Phase 1
  Tier 1); `sequential_residual` (Booging) remains future for v0.9.x
  Phase 2.
* **`mlp.architecture`** + **`mlp.loss`** (apply equally to lstm / gru
  / transformer) -- schema-only future. HNN dual hemispheres + emphasis loss.

### Added (recipe gallery)

* **`examples/recipes/replications/`** (new) -- one YAML per paper.
  * **9 operational**: `perfectly_random_forest`, `scaled_pca`,
    `macroeconomic_random_forest`, `ols_attention_demo`,
    `sparse_macro_factors`, `macroeconomic_data_transformations`
    (MARX), `ml_useful_macro`, `factor_midas_nowcast`, `arctic_var`.
  * **8 pre-promotion**: `booging`, `marsquake`, `adaptive_ma_rf`,
    `two_step_ridge`, `hemisphere_nn`, `anatomy_oos`,
    `dual_interpretation`, `maximally_forward_looking`,
    `slow_growing_trees`.

### Added (Python helper module)

* **`macroforecast.recipes.paper_methods`** (new) -- 19 helpers (PRF
  + 18 paper variants) returning recipe dicts ready for
  `macroforecast.run`. Helpers and YAML recipes are kept 1:1 in sync.

### Added (introspect API)

* **`macroforecast.scaffold.introspect.all_options(layer_id)`** -- new
  helper. Used by the orphan-detection test so future-status options
  can carry OPTION_DOCS prose ahead of their runtime promotion.
* L3 / L4 / L7 introspect fallback builders include `future`-status
  options with status carried through.

### Added (optional deps)

* **`[mars]`** = `pyearth>=0.1`
* **`[anatomy]`** = `anatomy` (Schwenk-Nebbe 2022; PyPI `anatomy 0.1.6`)

### Test additions

* **`tests/core/test_v09_paper_coverage.py`** -- 25 tests pinning:
  * `mars` operational with optional-dep error
  * 7 decomposable methods are NOT L4 families
  * each new atomic op layer_scope + status
  * 6 sub-axis future gates raise NotImplementedError with paper
    citation in message
  * `perfectly_random_forest` end-to-end via helper API and via YAML

### Promotion plan

13 algorithmic implementations land across v0.9.0 / v0.9.1 / v0.9.2 /
v0.9.3 in Tier-grouped milestones. Plan, paper references, algorithm
sketches, dependencies, risks: ``docs/architecture/v09_paper_coverage_plan.md``.

## [0.8.8] -- 2026-05-06 -- "user-friendliness pass (docs only)"

Single docs-only release that bundles five documentation upgrades. No
code changes; same 1035 tests; bit-exact replicate contract unchanged.

### Added
* **`docs/for_recipe_authors/custom_hooks.md` deep dive** -- every one
  of the five extension points (`custom_model`, `custom_preprocessor`,
  `target_transformer`, `custom_feature_block`,
  `custom_feature_combiner`) gets seven sections: decorator usage,
  required signature with type hints, input contract, output contract,
  worked example, common errors, and (for `custom_preprocessor`) a
  table comparing `applied_at='l2'` (pre-pipeline) vs `'l3'`
  (post-pipeline) covering leaf_config keys, runtime stages,
  cleaning_log entries, since-versions.
* **`docs/for_recipe_authors/partial_layer_execution.md` (new)** --
  user guide for running L1 / L2 / L3 / L4 / L5 in isolation via
  `materialize_l1` / `materialize_l2` / `materialize_l3_minimal` /
  `materialize_l4_minimal` / `materialize_l5_minimal` /
  `execute_l1_l2` / `execute_minimal_forecast` / `execute_node`. 9
  runnable snippets + schema tables for nine intermediate artifacts
  (L1DataDefinitionArtifact through L5EvaluationArtifact), with
  debugging use cases (outlier-policy inspection, L3 method-dev
  iteration).
* **`docs/troubleshooting.md` (new)** -- 10 common error scenarios
  with fixes: missing `leaf_config.target`, stale `pip install
  macroforecast` cache, `compare_models().compare()` chain on
  pre-`fit_main` versions, `replicate().sink_hashes_match=False`
  debugging, custom callable not registered,
  `mixed_frequency_representation` gate, missing extras,
  Encyclopedia drift CI failures, partial-layer inspection,
  where-to-ask. Linked from `docs/index.md` "Pick your path".

### Fixed (Simple Docs accuracy)
* **`ExperimentRunResult` / `ExperimentSweepResult` -> `ForecastResult`**
  across 5 simple_api/ pages (the v0.8.0+ rename was incomplete).
* **`result.variants` -> `result.metrics`** and
  **`result.compare("mse")` -> `result.ranking` /
  `result.mean(metric="mse")`** in 4 pages -- aligned with the actual
  v0.8.5 rich-accessor API.

### Fixed (Architecture page drift)
* **`docs/architecture/layer2/index.md`** -- the "Decision order"
  table listed 13 axis names from the pre-0.0-restart 8-layer
  registry (`tcode_policy`, `target_lag_block`,
  `factor_feature_block`, `level_feature_block`,
  `temporal_feature_block`, `rotation_feature_block`,
  `feature_block_combination`, `feature_selection_policy`,
  `feature_selection_semantics`, `feature_builder`,
  `x_lag_feature_block`, `target_normalization`,
  `horizon_target_construction`) that no longer exist in the L2
  `LayerImplementationSpec`. Rewritten as the actual 5 sub-layer ×
  15 axis table (`mixed_frequency_representation`,
  `sd_tcode_policy`, etc.) with an L2 custom-hook section pointing
  at `for_recipe_authors/custom_hooks.md`.
* **Stale numbered parent links** (`Parent: [4. Detail (code): Full]`
  / `Previous: [4.0 Layer 0: ...]` / `Next: [4.2 Layer 2: ...]`) on
  L0 / L1 / L2 / L1.* sub-pages collapsed to plain
  `[Architecture]` / `[Layer N: ...]` (the v0.6.3 number-prefix
  cleanup missed the parent-link strings).

### Notes
* No code, test, or schema changes; encyclopedia tree unchanged
  (drift CI green).
* The agent dispatch contracts for `custom_feature_block` and
  `custom_feature_combiner` documented here are the *actual* runtime
  contracts (`fn(frame, params)` / `fn(inputs, params)`), which
  replace the v0.1-era `FeatureBlockCallableContext` /
  `FeatureCombinerCallableContext` framing that was inaccurate after
  the 0.0 restart.
* `Experiment(...)` constructor today only drives official FRED
  datasets; custom inline panels still need to use
  `mf.run(yaml_recipe)`. This is documented in the worked examples
  and is a follow-up for a future minor.

## [0.8.6] -- 2026-05-06 -- "spec gap fixes: L2 pre-pipeline hook + fit_main + combined-dataset smoke + msfe→mse"

### Added
* **L2 pre-pipeline custom preprocessor hook** (Gap 1) -- new
  ``leaf_config.custom_preprocessor`` slot on L2. Runs *before* the
  canonical transform / outlier / impute / frame_edge stages so users
  can clean the raw L1 panel (drop bad columns, deflation,
  normalisation, custom resampling) before the official t-codes apply.
  Distinct from the v0.2.5 #251 ``custom_postprocessor`` slot which
  runs *after* the canonical pipeline. Both hooks dispatch through the
  same ``macroforecast.custom.register_preprocessor`` contract;
  ``Experiment.use_preprocessor(name, applied_at='l2')`` writes the
  new pre-pipeline slot, ``applied_at='l3'`` (default, unchanged)
  writes the existing post-pipeline slot. ``applied_at`` outside
  ``{'l2', 'l3'}`` raises ``ValueError``. Design Part 2 § L2 carries a
  new "Custom preprocessor hooks (pre vs post pipeline)" subsection
  documenting the two slots side-by-side.
* **Stable ``fit_main`` L4 fit-node id** (Gap 2) --
  ``Experiment.__init__`` and ``Experiment.compare_models([...])`` now
  rename the lone ``fit_<n>_<family>`` node generated by
  ``RecipeBuilder.l4.fit(...)`` to ``fit_main``. Predict-node inputs
  and the L4 ``sinks`` block (``l4_forecasts_v1`` /
  ``l4_model_artifacts_v1``) update atomically. Chained
  ``.compare("4_forecasting_model.nodes.fit_main.params.alpha", [...])``
  follow-ups now have a predictable dotted path independent of the
  original ``model_family=`` argument. Multi-fit (ensemble / horse-race)
  recipes are skipped automatically; an existing ``fit_main`` node
  short-circuits the rename so round-trips through
  ``to_recipe_dict()`` are idempotent.
* **``fred_md+fred_sd`` / ``fred_qd+fred_sd`` combined-dataset smoke**
  (Gap 3) -- new ``tests/integration/test_combined_dataset_smoke.py``
  locks in the L1 dispatch wired in ``_load_official_raw_result`` for
  the two combined-dataset strings. The test pre-populates the raw
  cache with the existing ``tests/fixtures/fred_md_sample.csv`` and
  ``fred_sd_sample.csv`` fixtures so no network access is required;
  the L1 sink is asserted to carry both national columns
  (``INDPRO``, ``RPI``, ``UNRATE``, ``CPIAUCSL``) and regional
  ``VAR_STATE`` columns (``UR_CA``, ``BPPRIVSA_CA``, ``UR_TX``,
  ``BPPRIVSA_TX``). Marked ``slow`` to mirror the existing integration
  suite.

### Changed
* **`primary_metric: msfe` → `primary_metric: mse`** in user-facing
  docs (Gap 4). The L5 schema accepts ``mse`` / ``rmse`` / ``mae``
  etc., not the legacy ``msfe`` alias from the macrocast era. Sweep
  affected ``docs/for_researchers/simple_api/*.md``,
  ``docs/for_recipe_authors/default_profiles.md``, and
  ``docs/navigator/replication_library.md``. The ``inverse_msfe`` /
  ``dmsfe`` L4 combine-method names are unrelated to this rename and
  are kept as-is.
* ``Experiment.use_preprocessor(applied_at='l2')`` no longer raises
  ``NotImplementedError``; the previous "reserved for v0.9" message
  has been removed and the docstring rewritten to document both
  dispatch points side-by-side.
* Simple-API docs (``compare_models.md`` / ``sweep_only_what_you_care.md``
  / ``simple_api/index.md``) reference ``fit_main`` in dotted-path
  examples.

### Tests
* ``tests/api/test_use_methods.py`` -- 3 new tests for the L2
  pre-pipeline hook (leaf_config wiring, end-to-end column-doubler
  preprocessor, ``applied_at`` validation).
* ``tests/api/test_experiment.py`` -- 4 new tests for ``fit_main``
  normalisation (init, ``compare_models``, chained ``.compare`` end-to-end,
  default ``model_family``).
* ``tests/integration/test_combined_dataset_smoke.py`` -- 3 new tests
  for ``fred_md+fred_sd`` / ``fred_qd+fred_sd`` loader dispatch and
  L1-sink merge.

## [0.8.5] -- 2026-05-02 -- "simple API completed (PR 2 of 2): .use_* hooks, ForecastResult rich, variants, two new axes"

### Added
* **`Experiment.use_*` hooks** -- six chainable methods that lower
  user-facing intent into the canonical recipe:
  - ``use_fred_sd_selection(states=, variables=)`` -- writes
    ``state_selection`` / ``sd_variable_selection`` axes (L1.D) plus
    ``selected_states`` / ``selected_sd_variables`` leaf lists.
  - ``use_fred_sd_state_group(group)`` -- L1.D ``fred_sd_state_group``
    axis (16 options, validated).
  - ``use_fred_sd_variable_group(group)`` -- L1.D
    ``fred_sd_variable_group`` axis (12 options, validated).
  - ``use_mixed_frequency_representation(mode)`` -- new L2.A
    ``mixed_frequency_representation`` axis (5 options).
  - ``use_sd_inferred_tcodes()`` -- new L2.B ``sd_tcode_policy=inferred``.
  - ``use_sd_empirical_tcodes(unit, code_map=, audit_uri=)`` -- new
    L2.B ``sd_tcode_policy=empirical`` plus its supporting leaf_config.
  - ``use_preprocessor(name, applied_at='l3')`` -- dispatches a
    ``mf.custom_preprocessor`` registration via the v0.2.5 #251
    runtime hook (``leaf_config.custom_postprocessor``); ``applied_at='l2'``
    raises ``NotImplementedError`` (reserved for v0.9 schema work).
  All return ``self`` for chaining; bad inputs raise ``ValueError``
  with the allowed-options list.
* **`Experiment.variant(name, **overrides)`** -- branches a named
  recipe variant. Variants land under ``recipe['variants'][name]``
  and are expanded to one cell per variant by ``execute_recipe`` in
  ``core/execution.py:_expand_variants``. Variants combine with
  ``compare_models`` / ``compare`` / ``sweep`` axes via the existing
  grid / zip combine modes; cell ids carry a ``__variant-<name>``
  suffix when a variant is active.
* **`ForecastResult` rich accessors** (replaces the v0.8.0 minimal shell):
  - ``forecasts`` -> per-cell ``l4_forecasts_v1`` rows concatenated
    with columns ``cell_id, model_id, target, horizon, origin,
    y_pred, y_pred_lo, y_pred_hi``.
  - ``metrics`` -> per-cell ``l5_evaluation_v1.metrics_table`` with a
    ``cell_id`` column.
  - ``ranking`` -> per-cell ``l5_evaluation_v1.ranking_table`` (empty
    DataFrame when no L5 ranking emitted).
  - ``mean(metric='mse')`` -> per-(model, target, horizon) average of
    one metric; useful one-liner for horse-race summaries.
  - ``read_json(name)`` / ``file_path(name)`` -- per-cell artifact
    accessors, fall back to the manifest root.
  - ``get(cell_id)`` -- pull one cell out by id, raise ``KeyError``
    on miss.
  All return empty DataFrames rather than raising when there is
  nothing to aggregate.
* **`mixed_frequency_representation` axis (L2.A)** -- 5 options:
  ``calendar_aligned_frame`` (default), ``drop_unknown_native_frequency``,
  ``drop_non_target_native_frequency``, ``native_frequency_block_payload``,
  ``mixed_frequency_model_adapter``. Generalises the FRED-SD-specific
  alignment rules to any mixed-frequency panel. Sub-layer L2.A renamed
  from "FRED-SD frequency alignment" to "Mixed frequency alignment".
  Gate: active when dataset includes FRED-SD (or a custom panel
  declares mixed frequency).
* **`sd_tcode_policy` axis (L2.B)** -- 3 options orthogonal to
  ``transform_policy``: ``none`` (default; FRED-SD source values left
  as published), ``inferred`` (national-analog research layer;
  records ``official: false``), ``empirical`` (variable-global /
  state-series stationarity audit map; requires ``sd_tcode_unit``,
  ``sd_tcode_code_map`` when ``unit=state_series``,
  ``sd_tcode_audit_uri``). Gate: active only when dataset includes
  FRED-SD.
* **OptionDoc entries** for each new axis option (8 total: 5 for
  ``mixed_frequency_representation``, 3 for ``sd_tcode_policy``).
* **Encyclopedia regenerated** -- 189 source-tree pages (was 187);
  two new axis pages
  (``docs/encyclopedia/l2/axes/mixed_frequency_representation.md``,
  ``sd_tcode_policy.md``) plus the canonical browse / index updates.
* **Design Part 2** -- ``plans/design/part2_l2_l3_l4.md`` documents
  both new axes (renamed L2.A heading + per-axis sub-section + gate
  table update).
* **Docs**: ``docs/for_researchers/planned_simple_api/`` ->
  ``docs/for_researchers/simple_api/``. Stripped the
  "API status note (current)" planning banner from each page.
  Replaced the index banner with an "every method documented here is
  implemented in v0.8.5" callout. Updated cross-doc links in
  ``docs/for_researchers/index.md`` (toctree + path table).
* **Tests** -- new file ``tests/api/test_use_methods.py`` with
  20 tests covering each ``.use_*`` validator path + variant() alone +
  variant × compare_models × sweep cross-products. Eight rich
  ForecastResult tests added to ``tests/api/test_forecast_result.py``.
  Updated ``test_experiment.py`` variant tests to assert recipe
  emission.

### Changed
* `pyproject.toml` + `macroforecast/__init__.py` -> 0.8.5.
* README + ``docs/install.md`` git pin -> ``@v0.8.5``.

## [0.8.0] -- 2026-05-02 -- "core public API: forecast() + Experiment + ForecastResult (PR 1 of 2)"

### Added
* **`mf.forecast(...)`** -- one-shot forecasting helper. Assembles the
  canonical default recipe (L0 fail_fast/seeded_reproducible/serial,
  L1 official-source path with target / horizons / sample window,
  L2 no-transform pass-through, L3 lag1 + target_construction,
  L4 single fit_model node with the requested family, L5 standard mse)
  via ``RecipeBuilder``, runs it through ``execute_recipe``, and wraps
  the result in a :class:`ForecastResult`. Supports ``fred_md``,
  ``fred_qd``, ``fred_sd`` (with explicit ``frequency=``),
  ``fred_md+fred_sd``, ``fred_qd+fred_sd``.
* **`mf.Experiment(...)`** -- builder class for one forecasting study.
  Methods: ``compare_models([f1, f2, ...])``, ``compare(axis_path, values)``,
  ``sweep(axis_path, values)`` (alias of ``compare``),
  ``to_recipe_dict()``, ``to_yaml()``, ``validate()``,
  ``run(output_directory=...)``, ``replicate(manifest_path)``.
  ``compare()`` walks dotted paths into the in-progress recipe dict
  (auto-creates intermediate dicts; addresses L3/L4 ``nodes`` lists by
  the entry's ``id`` field) and replaces the leaf with a
  ``{"sweep": [...]}`` marker.
* **`mf.ForecastResult`** (minimal shell) -- ``cells`` / ``succeeded`` /
  ``manifest_path`` / ``replicate()`` proxies over the underlying
  :class:`ManifestExecutionResult`.
* `tests/api/` -- 32 new tests across ``test_forecast.py``,
  ``test_experiment.py``, ``test_forecast_result.py`` covering: the
  default recipe wiring (L0 / L1 / L4 / L5 axes + start / end +
  horizons), dataset-frequency conflict detection, ``compare_models``
  expansion + ``sweep_values`` recording, generic ``compare()`` /
  ``sweep()`` axis paths, ``to_yaml`` round-trip through
  ``execute_recipe``, ``replicate()`` sink-hash match, ``_set_at``
  edge cases (empty path, traversal into scalar, list-by-id walk,
  sweep-marker overwrite).

### Deferred to v0.8.1 (PR 2 of 2)
* ``Experiment.use_fred_sd_inferred_tcodes()`` /
  ``.use_sd_empirical_tcodes()`` / ``.use_preprocessor()`` /
  ``.use_*`` family of one-call hooks.
* ``Experiment.variant(name, **overrides)`` -- currently raises
  ``NotImplementedError("variant() lands in v0.8.1")``.
* ``ForecastResult`` rich accessors: ``.forecasts`` / ``.metrics`` /
  ``.ranking`` / ``.read_json(...)`` / ``.file_path(...)`` / ``.mean()`` /
  ``.get(...)``.
* Docs migration: drop the ``planned_`` prefix on
  ``docs/for_researchers/planned_simple_api/`` and remove the
  per-page "API status note (current)" banners now that the API
  is real.

### Migration notes
* The new ``mf.forecast()`` / ``mf.Experiment`` surface is **additive** --
  ``mf.run("recipe.yaml")`` / ``mf.replicate(...)`` and the
  ``RecipeBuilder`` continue to work unchanged.
* The exclamation mark in the commit subject (``feat(api)!``) flags the
  new public surface for SemVer awareness; nothing existing is removed
  in v0.8.0.

## [0.7.0] -- 2026-05-06 -- "encyclopedia (replaces auto-emit reference)"

### Added
* **`docs/encyclopedia/`** -- source-committed markdown tree, one page
  per layer / sub-layer / axis (and per-option sections under each axis
  page), with three browse views: by layer, by axis (A-Z), by option
  *value* (A-Z). Generated from the live `LayerImplementationSpec`
  registry plus the `OPTION_DOCS` documentation registry under
  `macroforecast/scaffold/option_docs/`. 187 pages on first emit.
* `macroforecast/scaffold/render_encyclopedia.py` and
  `macroforecast/scaffold/__main__.py` -- the encyclopedia renderer plus
  a `python -m macroforecast.scaffold encyclopedia <out>` entry point.
* `macroforecast scaffold encyclopedia <out>` CLI subcommand on the
  top-level `macroforecast` console script.
* `tests/scaffold/test_render_encyclopedia.py` -- 11 tests covering
  page-count floor, per-layer index + axis pages, browse-by-option
  >= 30 model families, missing-OptionDoc TBD fallback, both CLI
  smoke routes.
* New section in [`docs/encyclopedia/public_api.md`](docs/encyclopedia/public_api.md)
  preserves the curated public Python API table that previously lived
  under `docs/reference/public_api.md`. Linked from
  `for_researchers/index.md` and `for_recipe_authors/index.md`.
* `ci-docs.yml`: new "Encyclopedia drift check" step. Re-emits the
  encyclopedia into a scratch dir and diffs against
  `docs/encyclopedia/`; the build fails if a contributor edits the
  schema or OptionDoc without re-running
  `python -m macroforecast.scaffold encyclopedia docs/encyclopedia/`.
* RELEASE_CHECKLIST.md gains an explicit reminder to regenerate the
  encyclopedia after any OptionDoc / LayerImplementationSpec edit.

### Changed
* `docs/index.md` "Pick your path" row now points at the encyclopedia
  rather than the removed reference index.
* Each `docs/architecture/layer{0..8}/index.md` gained a "See
  encyclopedia" footer cross-link to the matching
  `../../encyclopedia/l{N}/index.md`.
* README.md has a new "Browse the full encyclopedia at
  `docs/encyclopedia/`" pointer in the recipe-gallery section.

### Removed
* `docs/reference/` directory (the previous auto-emitted reference
  tree, including `public_api.md` and the per-build `lN.rst` files
  written by `_emit_optiondoc_reference()` in `docs/conf.py`). The
  curated `public_api.md` content is preserved at
  `docs/encyclopedia/public_api.md`.
* `_emit_optiondoc_reference()` build-time hook in `docs/conf.py`. The
  sphinx build no longer mutates the docs tree -- the encyclopedia is
  now source-committed and CI enforces sync.

### Migration notes
* If you had a local link to `docs/reference/<layer>.rst`, replace it
  with `docs/encyclopedia/<layer_id>/index.md` (per-layer landing) or
  `docs/encyclopedia/<layer_id>/axes/<axis>.md` (per-axis page).
* Bookmarks for `reference/public_api.md` should redirect to
  `encyclopedia/public_api.md`.

## [0.6.3] -- 2026-05-06 -- "openpyxl baseline + FRED-SD docs subdir + architecture number prefix cleanup"

### Changed
* **``openpyxl`` is now a core dependency**, not an optional extra.
  FRED-SD Excel workbook loading is a baseline code path; gating it
  behind ``[excel]`` made the user-visible "first FRED-SD recipe"
  story confusing for negligible install savings (the package itself
  is small). The ``[excel]`` extra is removed from
  ``[project.optional-dependencies]`` and from the ``[all]`` aggregate.

### Fixed (docs)
* **FRED-SD t-code policy pages moved under FRED-SD**:
  ``docs/for_researchers/{fred_sd_transform_policy,
  fred_sd_inferred_tcodes, fred_sd_inferred_tcode_review_v0_1}.md``
  ->
  ``docs/for_researchers/fred_datasets/fred_sd/{transform_policy,
  inferred_tcodes, inferred_tcode_review_v0_1}.md``.
  The flat ``fred_sd.md`` page also moved to
  ``fred_datasets/fred_sd/index.md`` so the SD subtree groups under a
  single page. The toctree at the top of ``fred_datasets/`` keeps
  ``fred_md`` / ``fred_qd`` flat (they have no sub-pages) and points
  at ``fred_sd`` (now a directory).
* **Number prefixes removed from headings**:
  - ``docs/for_researchers/fred_datasets/fred_md.md`` ``# 5.1 FRED-MD``
    -> ``# FRED-MD``; same for QD/SD.
  - ``docs/architecture/layer{0,1,2}/`` headings ``# 4.0 Layer 0`` /
    ``# 4.1 Layer 1`` / ``# 4.1.5 Raw Source Cleaning`` etc. ->
    ``# Layer 0`` / ``# Layer 1`` / ``# Raw Source Cleaning``. The
    folder hierarchy already encodes ordering; the text prefix was
    stale carryover from the pre-reorg flat ``detail/`` tree.
  - Parent links such as ``[5. FRED-Dataset](index.md)`` -> ``[FRED
    datasets](index.md)`` (or ``../index.md`` from ``fred_sd/``).

### Notes
* No code/test changes; same 953 tests, same recipe schema, same
  bit-exact replicate contract. Documentation hygiene release.

## [0.6.2] -- 2026-05-05 -- "docs reorganization (4 audiences + replications track)"

### Changed (docs only)
* **Docs IA reorganized into a 4-audience tree** plus reference and
  replications:

  * ``docs/for_researchers/`` (replaces ``docs/getting_started/`` and
    consolidates ``docs/fred_dataset/``; preserves the planned-API
    preview as ``planned_simple_api/`` inside this tree).
  * ``docs/for_recipe_authors/`` (replaces ``docs/user_guide/`` and
    absorbs ``docs/detail/custom_extensions.md`` ->
    ``custom_hooks.md``, ``target_transformer.md``,
    ``default_profiles.md``).
  * ``docs/for_contributors/`` (replaces ``docs/dev/``).
  * ``docs/architecture/`` (replaces ``docs/detail/`` for the L0-L8
    per-layer pages and the cross-cutting contracts; absorbs the old
    top-level ``foundation_core.md`` -> ``foundation.md`` and
    ``philosophy.md``).
  * ``docs/reference/`` carries the curated ``public_api.md`` (was
    ``docs/api/index.md``); the auto-emitted per-layer
    ``l{0..8}.rst`` pages from
    ``_emit_optiondoc_reference`` already land in this directory.
  * ``docs/replications/`` (NEW in v0.6.1) untouched.
  * ``docs/navigator/`` and ``docs/_html_extra/`` untouched.

* **Top-level ``docs/index.md``** rewritten as a 4-audience picker:
  "Pick your path" maps each role (researcher / recipe author /
  contributor / reference lookup / replications / navigator) to the
  right tree.

### Removed (docs only)
* ``docs/detail/decision_tree_navigator.md`` (redirect-only stub).
* ``docs/detail/contract_source_of_truth.md`` (refers to a registry
  layer that no longer exists).
* ``docs/detail/execution_engine.md`` (refers to the deleted
  ``macroforecast.execution`` legacy stack).
* ``docs/detail/experiment_object.md`` (refers to the deprecated
  ``mc.Experiment`` API that does not ship in v0.6.x).
* ``docs/detail/philosophy.md`` (older duplicate of the top-level
  ``philosophy.md``; kept the more current top-level copy at
  ``architecture/philosophy.md``).

### Archived (docs only)
* ``docs/_archive/post_pr70_runtime_roadmap.md`` (closed phases).
* ``docs/_archive/layer2_revision_plan.md`` (closed phases).
* ``docs/_archive/navigator_ui_redesign_plan.md`` (alternative UI
  track that did not ship).
* ``docs/_archive/source_rst_skeleton_v0.3/`` (orphaned RST skeleton
  pinned to release="0.3.0", unreferenced by ``.readthedocs.yaml``).

  ``docs/_archive/`` is excluded from the Sphinx build via
  ``exclude_patterns``; pages there are kept for blame/history but are
  not part of the live tree.

### Added (CI)
* ``ci-core.yml`` learns a "no stale audience-tree references" check
  that fails the build if any live doc still links to
  ``getting_started/``, ``user_guide/``, ``fred_dataset/``,
  ``simple/``, ``detail/``, ``api/``, or ``dev/`` (the trees the
  reorg removed).

### Notes
* No code changes. Same ``__version__`` lifecycle, same recipe schema,
  same bit-exact replicate contract. v0.6.2 is a documentation-only
  release; the test suite is unchanged at 953 passing.

## [0.6.1] -- 2026-05-05 -- "post-rename consistency sweep"

### Fixed
* **Stale ``v0.5.x`` strings** in ``macroforecast/__init__.py`` docstring,
  ``docs/api/index.md``, ``docs/getting_started/index.md``, and the API
  status banners in every ``docs/simple/*.md`` page. The package
  docstring no longer pins a version number ("**Public surface**" with
  no parenthesised version); API banners use "(current)" /
  "current YAML runtime" instead of "(v0.5.x)".
* **``release.yml`` NOTE comment** still claimed the ``macroforecast``
  PyPI namespace was held by an unrelated 2017 package. Replaced with
  the actual situation: the maintainer owns the namespace and v0.6.0
  was the first release published under it.
* **``.github/RELEASE_CHECKLIST.md`` "PyPI namespace status" section**
  rewritten the same way; gained a ``pip index versions`` post-tag
  check; the throwaway-venv install command now uses
  ``pip install macroforecast==X.Y.Z`` instead of the GitHub-tag fallback.
* **``CLAUDE.md`` header** still showed "Version 0.5.0" + "~785 tests";
  pinned the version to "see ``pyproject.toml`` / ``__version__``" and
  refreshed the test count to 953.

### Notes
* No code changes; same 953 tests, same recipe schema, same bit-exact
  replication contract. v0.6.0 narrative consistency patch.

## [0.6.0] -- 2026-05-05 -- "rename macrocast -> macroforecast"

### Changed (BREAKING)
* **Package rename**: ``macrocast`` -> ``macroforecast``. Both the
  PyPI distribution name and the importable Python module name change.
  ``import macrocast`` no longer resolves; use ``import macroforecast``.
  Convention alias in docs: ``import macroforecast as mf`` (was
  ``as mc``).
* **GitHub repo rename**: ``NanyeonK/macrocast`` ->
  ``NanyeonK/macroforecast``. GitHub auto-redirects old URLs for some
  time but new install commands and bookmarks should use the new name.
* **CLI script**: ``macrocast`` console script renamed to
  ``macroforecast`` (still backed by ``macroforecast.scaffold.cli:main``).
* **PyPI publish unblocked**: the user owns the ``macroforecast``
  PyPI namespace, so ``release.yml`` can now publish successfully on
  tag push (the v0.5.x ``macrocast`` namespace was held by an
  unrelated 2017 package; that warning is gone in v0.6.0).

### Migration

```diff
- import macrocast as mc
+ import macroforecast as mf

- result = mc.run("recipe.yaml")
+ result = mf.run("recipe.yaml")
```

```bash
# Old:
pip install "git+https://github.com/NanyeonK/macrocast.git@v0.5.3"
# New:
pip install macroforecast
# or pinned to a tagged release:
pip install "git+https://github.com/NanyeonK/macroforecast.git@v0.6.0"
```

### Notes
* No runtime behaviour changes; same 953 tests, same recipe schema,
  same bit-exact replication contract. This is a name-only release.
* The ``CHANGELOG`` historical entries below have been swept by sed
  along with the rest of the codebase; references to "macroforecast"
  in v0.5.x entries should be read as "macrocast" historically. The
  past PyPI namespace warnings were about the old ``macrocast`` name
  which is no longer relevant.

## [0.5.3] -- 2026-05-05 -- "version consistency + CI guardrails"

### Fixed
* **README + docs/install.md still pointed at v0.5.1**: ``@v0.5.1`` ->
  ``@v0.5.3`` for every recommended ``pip install
  "git+...@v..."`` command, citation line bumped, version-badge
  removed in favour of real CI badges so the README cannot drift again.
* **``docs/api/index.md`` claimed v0.1**: rewritten to "v0.5.x" and
  framed as a curated reference (it is not actually
  ``sphinx-apidoc`` autogenerated output).
* **``docs/getting_started/index.md`` mis-described Simple Docs**:
  "Existing simple API" / "older high-level Python facade" was
  incorrect — the simple ``mf.forecast`` / ``mf.Experiment`` shape
  is *planned*, not legacy. Row updated to "Planned simple API" and
  redirects to ``macroforecast.run`` / Detail Docs for v0.5.x.
* **API status note only on Simple Docs index**: every other page in
  ``docs/simple/`` (quickstart, run_experiment, compare_models,
  add_custom_model, add_custom_preprocessor, read_results,
  sweep_only_what_you_care, fred_sd) now has the same banner so a
  reader landing via search hits the warning before any sample code.
* **README static badges go stale silently**: replaced the
  ``tests-N passing`` / ``version-X.Y.Z`` static shields with live
  ``ci-core`` and ``ci-docs`` workflow badges. Test count moved to a
  short prose note under the badges.
* **README operational coverage was easy to over-read**: section now
  opens with a callout pointing at
  ``docs/getting_started/runtime_support.md`` for the exact
  end-to-end coverage matrix.
* **``docs/install.md`` core dependencies table missed scipy /
  matplotlib**: added rows so the requirements line and the
  table agree with ``pyproject.toml``.

### Added
* **``ci-core`` version-consistency check**: a step that asserts
  ``pyproject.toml::version`` and ``macroforecast/__init__.py::__version__``
  match, and that no ``@vX.Y.Z`` reference in ``README.md`` /
  ``docs/install.md`` is older than the current package version.
  This is what would have caught v0.5.2 shipping README pointing at
  v0.5.1.
* **``.github/RELEASE_CHECKLIST.md``**: pre-tag / tag / post-tag
  checklist plus the PyPI-namespace status note so future releases
  do not re-discover the same gotchas.
* **``release.yml`` NOTE comment**: explicit reminder that PyPI
  publish is gated on (a) the ``PYPI_API_TOKEN`` secret being
  registered and (b) the token having upload permission for the
  ``macroforecast`` PyPI project (which is currently held by an
  unrelated 2017 package).

### Changed
* This release is functionally equivalent to v0.5.2 — runtime
  behaviour, public API, and test count unchanged. The bump exists
  because v0.5.2 shipped with stale install instructions in its
  README/docs/install.md, and the user-facing fix has to live behind
  a new tag (``bit-exact replication`` ethos: do not silently
  force-push tags).

## [0.5.2] -- 2026-05-05 -- "external review fixes"

### Fixed
* **README install instructions wrong**: ``pip install macroforecast``
  installs an unrelated 2017 package (``macroforecast 0.0.2`` by Amir
  Sani). README + ``docs/install.md`` now warn about the namespace
  collision and recommend ``pip install
  "git+https://github.com/NanyeonK/macroforecast.git@v0.5.2"`` until the
  namespace is resolved.
* **README badges stale**: version ``0.3.0`` -> ``0.5.2``, tests
  ``785 passing`` -> ``953 passing``.
* **docs/install.md placeholders**: ``your-org/macroforecast`` ->
  ``NanyeonK/macroforecast``; expected test count ``291`` -> ``953``;
  ``scipy`` and ``matplotlib`` added to the requirements table to
  match the actual ``pyproject.toml`` dependency set.
* **Simple Docs reference a not-yet-shipped API**: Simple-track pages
  use ``mf.forecast(...)`` and ``mf.Experiment(...).compare_models([
  ...]).run()``; these are not yet exported from
  ``macroforecast.__all__``. Added an "API status note" banner at the top
  of ``docs/simple/index.md`` clarifying that the snippets describe
  the v0.6+ planned wrapper shape, and pointing users to the v0.5.x
  canonical entry surface (``macroforecast.run`` / ``macroforecast.replicate``
  / ``RecipeBuilder`` / ``python -m macroforecast scaffold``).

### Added
* **CLI script entry**: ``[project.scripts]`` now declares
  ``macroforecast = "macroforecast.scaffold.cli:main"`` so installs expose the
  ``macroforecast scaffold`` / ``macroforecast run`` / ``macroforecast replicate``
  / ``macroforecast validate`` shell commands directly. Previously users
  had to use ``python -m macroforecast ...``.

### Changed
* **PyPI publish auth**: ``release.yml`` switched from OIDC Trusted
  Publishing to a ``PYPI_API_TOKEN`` secret since the trusted
  publisher was not registered on PyPI's side. Tag pushes now
  authenticate via the API token in repo secrets.

## [0.5.1] -- 2026-05-05 -- "docs + CI hygiene patch"

### Fixed
* **ci-docs sphinx build**: package docstring in ``macroforecast/__init__.py``
  used ``Heading:`` followed by a hyphen-bullet list, which docutils
  parsed as a definition list with unexpected indentation. Switched to
  ``**Heading**`` plus a blank line before each list. Also added
  ``macroforecast.scaffold`` to the importable submodule list (it shipped
  in v0.3 but was not advertised in the docstring).
* **ci-docs trigger paths**: docstring-only changes in
  ``macroforecast/**`` previously did not re-run ``ci-docs`` (its trigger
  paths were limited to ``docs/**`` + ``pyproject.toml``), so a
  docstring fix could leave a known-broken sphinx build cached on
  ``main``. Added ``macroforecast/**`` to the workflow's ``paths`` filter.

### Notes
* Same code surface and behaviour as v0.5.0; this is a release-hygiene
  bump so the next ``release.yml`` invocation runs against a green
  ``ci-docs`` build.

## [0.5.0] -- 2026-05-05 -- "examples gauntlet + CLI surface"

### Fixed
* **Quick-start broken**: ``examples/recipes/l4_minimal_ridge.yaml``
  (referenced by ``CLAUDE.md`` Quick start) and three other isolated
  layer fragments (``l4_ensemble_ridge_xgb_vs_ar1.yaml`` /
  ``l6_standard.yaml`` / ``l6_full_replication.yaml``) failed with
  ``single_target requires leaf_config.target string``. All four are
  now self-contained end-to-end runnable on a synthetic panel; the
  ensemble + replication recipes use ``gradient_boosting`` (sklearn,
  always installed) instead of ``xgboost`` so they run on a stock
  install -- swap the family in if the corresponding extra is
  installed.
* **L6.B nested test runtime crash**: ``_l6_nested_results`` referenced
  an undefined ``hac_kernel`` local; added the
  ``leaf.get("dependence_correction", "newey_west")`` lookup. Surfaced
  by the ``l6_full_replication`` example which exercises L6.B for the
  first time end-to-end.
* **Wizard wrote the wrong diagnostic-layer recipe keys** under
  ``include_diagnostics=True`` -- ``1_5_data_diagnostics`` etc. instead
  of the runtime's canonical ``1_5_data_summary`` /
  ``2_5_pre_post_preprocessing`` / ``4_5_generator_diagnostics``.
  Fixed in ``scaffold/wizard.py`` ``_LAYER_KEYS``.
* **YAML parse failure**: ``examples/recipes/l1_with_regime.yaml`` was
  a multi-document file. Split into
  ``l1_with_regime.yaml`` + ``l1_estimated_markov_switching.yaml``.

### Added
* **``tests/test_examples_smoke.py``** -- regression guard that
  parametrises over every example yaml. Two layers: every recipe must
  parse + use canonical NEW layer keys; the curated ``runnable``
  subset must execute via ``macroforecast.run`` without error. 78 new
  tests.
* **CLI subcommands** (``macroforecast run`` / ``replicate`` / ``validate``).
  Previously only ``scaffold`` existed; users had to run
  ``python -c "import macroforecast; macroforecast.run(...)"`` for everything
  else.
* **``examples/recipes/goulet_coulombe_2021_replication.yaml``** --
  Goulet-Coulombe (2021) FRED-MD ridge baseline ported to the
  12-layer schema. Runnable end-to-end on the embedded sample panel.

### Changed
* **Archived the v0.0.0 8-layer schema corner**: 13 recipes (9 in
  ``examples/recipes/`` + 3 ``replications/`` + 1 ``templates/``) that
  used the deprecated ``recipe_id`` / ``path: { 1_data_task: ... }``
  wrapper moved to ``examples/recipes/archive_v0/`` with a README
  explaining the migration path. The smoke test skips this directory.
* **``RecipeBuilder`` docstring** now spells out that the per-layer
  namespaces deliberately mutate the shared dict rather than returning
  ``self``; users chain through ``b`` itself, not jQuery-style. No API
  change -- documentation only.

### Test coverage
* 944 passing (up from 866) / 13 skipped.

## [0.3.0] -- 2026-04-XX -- "third honesty pass + new families"

See ``CLAUDE.md`` ``v0.3 third honesty pass`` table.

## [0.25.0] -- 2026-03-XX -- "second honesty pass"

See ``CLAUDE.md`` ``v0.25 second honesty pass`` table.

## [0.2.0] -- 2026-02-XX -- "first honesty pass"

See ``CLAUDE.md`` ``v0.2 honesty-pass demotions`` table.

## [0.1.0] -- 2026-01-XX -- "initial 12-layer release"

Schema-runtime parity for L0..L8 + L1.5/L2.5/L3.5/L4.5 diagnostics.
