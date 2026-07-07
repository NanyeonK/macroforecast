# Changelog

Notable changes since the v0.0.0 schema reset. See ``CLAUDE.md`` for the
full per-version honesty-pass history embedded in repo documentation.

## [Unreleased]

- `models/timeseries.py`, `forecasting/policies/panel.py`, `pipeline/spec.py`
  (behavior change, issue #442): `var` now has a validated direct-projection
  mode for `direct` panel forecasts. The direct VAR target equation regresses
  `y[t+h]` on the origin-dated panel lag block
  `Y[t], ..., Y[t-p+1]`, so `h=1` matches the iterated VAR one-step forecast
  while longer horizons no longer collapse to stale persistence. Because this is
  a point target, not the horizon-average object, `var` is guarded under
  `direct_average` by the same `on_unsupported_direct` error/warn/reroute control
  used for other unsupported direct-like combinations. `var` leaves
  `DIRECT_POLICY_GUARD_MODELS` for plain `direct`. For the remaining guarded
  iterated/state-space
  models, `pipeline_spec(...)` now defaults to
  `on_unsupported_direct="error"` instead of warning: silent persistence-like
  forecasts must not be produceable by default. Deliberate weak benchmarks can
  pass `on_unsupported_direct="warn"`, or `on_unsupported_direct="reroute"` to
  run affected arm-target cells as `forecast_policy="recursive"` with recursive
  row labels. `forecasting.run(...)` with panel-input models under
  `forecast_policy="recursive"` no longer raises `ValueError`; the panel runner
  uses the model's native multi-step prediction path and emits recursive row
  labels. Added the generated
  `docs/guide/model_policy_matrix.md` and CI drift check.

- `pipeline/result_store.py`, `pipeline/run.py`, `pipeline/spec.py`,
  `preprocessing/cache.py`, `forecasting/preprocessing_stage.py` (feature, W9
  result store + Gap C): added `pipeline_spec(..., result_store=...)` for
  cross-run reuse of completed `(target, horizon, arm)` forecast cells. Stored
  cells live under `<store>/cells/<digest>.parquet` with a JSON manifest, are
  written via atomic replace, and are reused only when the resolved cell identity
  and the pipeline data fingerprint match. `PipelineReport.provenance` now
  includes a `result_store` block when enabled, with reused/computed/
  undigestible counts and version-mismatch details; mismatched package versions
  still reuse but warn once. Custom user callables are intentionally
  undigestible unless they carry an explicit `__mf_digest__` string. New
  `mf.pipeline.result_store_summary()` and `mf.pipeline.purge_result_store()`
  inspect and maintain stores. The existing `PreprocessorStore` also persists
  the horizon-independent prepared-base DataFrame tier when
  `preprocessing_cache_dir` is configured, so adding a new model to an existing
  horse race no longer re-fits or re-transforms the shared per-origin
  preprocessing base. Default `result_store=None` leaves the existing execution
  path unchanged.

- `data/vintage.py`, `forecasting/runner.py`, `pipeline/run.py`,
  `forecasting/checkpoint.py` (feature, per-origin vintages): added the
  `VintageSource` protocol, `VintageUnavailableError`, `fred_md_vintages()` /
  `fred_qd_vintages()`, `custom_vintages()`, `with_static_extras()`, and
  `VintagePanelSpec` so FRED-MD/QD or user-supplied studies can resolve one
  point-in-time `DataBundle` per forecast origin while keeping the non-vintage
  runner path unchanged. Custom vintages accept callable, mapping, and long
  ALFRED-style frames, with every snapshot normalized through the canonical
  panel contract and memoized by resolved vintage identifier. Static extras join
  genuinely non-revised columns onto every resolved snapshot and add a stable
  extra-panel fingerprint to the vintage ID. Vintage-aware runs now support
  serial and parallel pipeline execution, feature-matrix `direct`,
  `direct_average`, `path_average`, and `recursive` policies, panel-input
  models, full-panel preprocessing, and full/fixed feature scopes; the remaining
  guard is `retrain_every != 1`, which still needs future semantics for stale
  fit reuse across changing data content. `actuals_vintage="latest"` scores
  against the latest run snapshot, while `actuals_vintage="first_release"`
  resolves each target date `d` from the first vintage strictly after `d`, making
  `actuals_vintage_id` row-varying when needed. Forecast rows and lean
  checkpoints carry nullable `vintage_id` and `actuals_vintage_id` columns;
  `PipelineReport.provenance["vintage_source"]` records source kind,
  actuals policy, reference calendar, and the origin-vintage map, writing maps
  above 500 origins to `vintage_map.json` beside `checkpoint_dir` with a sha256
  sidecar reference. `leakage_audit["vintage_boundary_audit"]` surfaces the
  mandatory check that resolved vintage panels stop strictly before each origin.
  Vintage runs tag in-memory preprocessing/feature cache keys and
  `PreprocessorStore` namespaces with `metadata["vintage"]`; default
  non-vintage cache keys remain byte-identical. Change/growth-style
  macroforecast-side target transforms now emit a warning in vintage-aware runs
  so users pre-transform those targets within each vintage snapshot when that is
  the intended real-time estimand. See `docs/guide/vintages.md`.

- `pipeline/run.py`, `pipeline/spec.py`, `pipeline/rescore.py` (feature,
  Wave B lane B-2, self-certifying `PipelineReport`): `run_pipeline`'s
  provenance (`pipeline/run.py::_audit`) and `output.collect_provenance`
  (`output/core.py`) were two systems that never met -- the pipeline's
  `report.provenance` had `package_version`/`seed`/`targets`/`arms`/
  leakage-audit but no git SHA, no environment, and no data identity, while
  `collect_provenance` had all of that (git commit/branch/dirty, Python/
  platform, pinned `numpy`/`pandas`/`scipy`/`scikit-learn`/`statsmodels`
  versions) but only attached on the opt-in save/`write_run` path. A referee
  handed only the report artifact could not tell which macroforecast build
  produced it, nor pin the data vintage. By default
  (`pipeline_spec(..., provenance_level="full")`, the default)
  `report.provenance` now additionally carries: (1) `"environment"` -- reuses
  `output.collect_provenance` (no duplicated git/env/deps logic), pointed at
  the RUNNING macroforecast package's own checkout rather than the caller's
  cwd, so it resolves to that checkout's commit/branch/dirty from a source
  install and gracefully returns `None`s from a wheel install (no `.git` above
  site-packages); (2) `"data"` -- dataset name/source family/declared vintage
  from the `DataBundle` metadata, panel shape, date range, and a stable sha256
  content fingerprint over the panel's index+columns+values (full-content by
  default, measured at ~1ms for a FRED-MD-sized panel and ~8ms for a 2,000 x
  400 panel; a deterministic strided subsample only above 20,000,000 cells,
  labeled via `fingerprint["method"]`, never silently mistaken for a full
  digest); (3) `"spec_echo"` -- a plain JSON-able snapshot of the resolved
  spec's key choices (targets/policies, horizons, window cutoffs, arms/
  models, benchmark, evaluation config, seed, `n_jobs`/`model_threads`,
  cache/checkpoint dirs). `provenance_level="basic"` opts out to exactly the
  pre-existing dict shape. `rescore()` reports carry the same `"environment"`
  block (respecting the spec's `provenance_level`) plus the existing
  `rescored_from` marker; `"data"`/`"spec_echo"` are not attached to a
  rescored report since rescoring does not re-touch the original data.
  Forecasts/accuracy are BYTE-IDENTICAL to a pre-change golden fixture
  (`tests/pipeline/_golden/default_provenance_*.parquet`, captured at base
  commit `70ad5b0e`) -- provenance is purely additive, and neither
  `_run_cells`/`evaluate()` nor any numerical path was touched. See
  `docs/reference/pipeline.md` ("Provenance") and
  `tests/pipeline/test_default_provenance.py`.
- `models` (WP-B4, default-cost fix, closes the WP-A4 policy-matrix-scan
  finding): `hemisphere_nn` and `density_hnn` (torch-backed, bagged
  dual-head/density Hemisphere neural networks) could not finish a single
  `.dev-notes/policy_matrix_scan.py` worker call (all 4 policies, 140-obs toy
  panel) even at a 900s timeout with their out-of-the-box defaults. Same
  root cause as the six-model WP4 fix below: `model_selection=None` still
  runs a 20-trial random search over the model's "standard" search-space
  preset even with no explicit tuning request (`_selection_for_model` /
  `_fit_one_model_at_origin` in `forecasting/policies/base.py`), so an
  untuned fixed cost (`max_epochs`, `patience`) held at its literal default
  for every trial, multiplied by the "standard" preset's `neurons`/
  `n_estimators`/`prior_estimators` corners, got paid 20 times per origin per
  policy. Old -> new defaults (both `MODEL_SPECS[...].default_params` in
  `models/specs.py` and the public-function/regressor-class kwarg defaults in
  `models/neural.py`, which are independent of the spec layer for any
  direct, non-pipeline call):
  - `hemisphere_nn`: `n_estimators` `100`->`20`, `max_epochs` `100`->`40`,
    `patience` `15`->`8`. "standard" preset (own dedicated `_HNN_SPACES`,
    not shared with other models): `neurons` `(32, 64)`->`(16, 32)`,
    `n_estimators` `(5, 10)`->`(3, 5)`; `learning_rate` unchanged. "small"
    and "wide" are unchanged -- "wide" still reaches the old `(32, 64)` /
    `(5, 10)` corner for deep/explicit use.
  - `density_hnn`: `n_estimators` `100`->`20`, `prior_estimators` `50`->`10`,
    `max_epochs` `100`->`40`, `patience` `15`->`8`. `neurons` (400, the
    paper-faithful Aionx `DensityHNN` width) is UNCHANGED: `model_selection`
    left at `None` always overrides `neurons` via the "standard"/"wide"
    preset anyway, so it was never the scan's cost driver, and trimming it would
    only cost paper-fidelity for direct/no-search callers with no scan-cost
    benefit. "standard" preset (own dedicated `_DENSITY_HNN_SPACES`):
    `neurons` `(32, 64)`->`(16, 32)`, `n_estimators` `(5, 10)`->`(3, 5)`,
    `prior_estimators` `(3, 5)`->`(2, 3)`; `learning_rate` unchanged. "small"
    and "wide" are unchanged.
  Verified scan totals (4 policies, thread-pinned for clean numbers --
  `OMP_NUM_THREADS=1`/`MKL_NUM_THREADS=1`/`OPENBLAS_NUM_THREADS=1`,
  `torch.set_num_threads(1)`; per-policy seconds from
  `.dev-notes/policy_matrix_scan.py --model <name>`): `hemisphere_nn`
  360.12s (`direct` 70.15s, `direct_average` 69.67s, `path_average` 140.27s,
  `recursive` 80.03s) -> 90.16s (18.91s, 17.76s, 35.43s, 18.06s);
  `density_hnn` 240.62s (46.53s, 46.50s, 94.66s, 52.93s) -> 66.29s (13.75s,
  13.05s, 26.08s, 13.41s). Both now comfortably clear the 150s bar with the
  full 4-policy scan; individual policies are all well under 60s, and the
  new bagged member counts (20/10) are still real ensembles, not degenerate
  single fits. Deep/paper-faithful settings remain reachable by passing the old
  values explicitly (e.g. `hemisphere_nn(X, y, n_estimators=100,
  max_epochs=100, patience=15)`) or via the "wide" preset for search-driven
  runs. Trade-off, not free: fewer bag members widens `density_hnn`'s
  per-row predicted-variance right tail further -- WP-A4's anchor work
  (`tests/models/anchors/test_hnn_anchors.py`) already found and documented
  that tail as driven by base bagged-ensemble member disagreement (not the
  OOB volatility-rescaling step), so fewer members means more sampling noise
  in that cross-member disagreement estimate, i.e. a wider tail than before
  -- reachable back to the old width via explicit `n_estimators=100`.
  `tests/models/test_default_cost_budget.py` (WP4's single-fit cost-budget
  gate) gets two new tests, `test_hemisphere_nn_default_cost_budget` /
  `test_density_hnn_default_cost_budget`, torch-gated per-test
  (`pytest.importorskip("torch")`, matching `tests/models/test_models.py`'s
  existing mixed-file convention) since this file is not module-level
  torch-gated. `tests/models/test_models.py`'s
  `density_hnn_spec.default_params["prior_estimators"]` assertion updated
  `50`->`10` to match. The WP-A4 anchor tests
  (`tests/models/anchors/test_hnn_anchors.py`) use explicit small params
  throughout (e.g. `neurons=8, max_epochs=15, n_estimators=3`) and are
  unaffected -- re-verified green (7 passed).

- `pipeline/evaluate.py`, `pipeline/spec.py` (bugfix): `EvalSpec.metrics` and
  `EvalSpec.tests` were declared fields that `evaluate()` silently ignored --
  `accuracy_table` always emitted the hard-coded `rmse`/`relative_mse`/`r2_oos`
  columns and `significance_table`/`mcs_table` always ran DM/CW/MCS regardless
  of what a user passed. `EvalSpec(metrics=("mae",), tests=("dm",))` now
  actually restricts the output to those metrics/tests: `accuracy_table`
  computes exactly the listed metrics (string names resolved through
  `macroforecast.metrics.get_metric`, or a callable `metric(y_true, y_pred) ->
  float` named by `__name__`) per contender on the same pairwise-vs-benchmark
  sample it always used; `significance_table`/`mcs_table` run only the named
  tests, with `"cw"` still additionally gated by `cw_for_nested`. Unsupported
  test names now raise `ValueError` at `pipeline_spec(...)` build time instead
  of being silently dropped. The three default metrics keep their existing benchmark-relative
  formulas byte-for-byte, pinned by a golden test
  (`tests/pipeline/test_evalspec_threading.py`) captured from the pre-fix
  `evaluate()` output on a fixed master forecast frame. `rescore()` inherits
  all of this automatically since it just calls `evaluate()`.

- `pipeline/spec.py`, `pipeline/evaluate.py` (feature): new `EvalSpec.loss:
  Callable[[y_true, y_pred], ndarray] | None = None`, a per-observation loss
  threaded into the Diebold-Mariano loss differential and the Model Confidence
  Set's loss matrix (default `None` is squared error, the prior, only
  behavior), enabling custom-loss (e.g. asymmetric/linex) horse races for the
  IJF-economist persona. Because the Clark-West adjustment term is derived
  under quadratic loss, it is not a valid test under an arbitrary loss
  function: when `loss` is set and CW would otherwise run (`"cw"` requested,
  `cw_for_nested` true, and at least one nested contender), `significance_table`
  skips CW entirely and emits a `UserWarning` explaining why, rather than
  silently computing it against the wrong loss -- DM and MCS are loss-agnostic
  and are unaffected. See `docs/reference/pipeline.md` ("Custom metrics,
  significance tests, and loss").
- `pipeline/spec.py`, `pipeline/evaluate.py` (feature): `EvalSpec.tests` now
  wires the existing standalone forecast-comparison library into pipeline
  reports beyond the historical DM/CW/MCS trio. New opt-in pairwise
  `PipelineReport.significance` rows cover `"gw"`, `"enc_new"`, `"enc_t"`,
  `"pt"`, `"hm"`, `"ag"`, and `"gr"`; full-set benchmark comparisons
  `"spa"`, `"rc"`, and `"stepm"` append rows to `PipelineReport.mcs`. New
  `EvalSpec.test_options` validates per-test keyword options at spec-build time
  and threads bootstrap/window controls to the underlying public callables. The
  default `EvalSpec()` path remains byte-identical, pinned by the existing
  golden tests.
- `tests.py` (feature): added public `multi_horizon_spa_test(...)` implementing
  Quaedvlieg (2021) pairwise multi-horizon SPA in uniform (`"uspa"`) and
  average (`"aspa"`) modes. It accepts a horizon-wise loss-differential panel
  or two aligned loss panels, uses a Quadratic Spectral HAC original statistic,
  and uses the moving-block bootstrap/natural block-variance Algorithm 1 path
  with paper simulation defaults `block_length=3`, `n_boot=999`.
- `pipeline/spec.py`, `pipeline/evaluate.py` (feature): `EvalSpec.tests`
  accepts `"uspa"`/`"aspa"` for joint multi-horizon pairwise comparison across
  all horizons of each target/contender/benchmark triple. Results append
  `PipelineReport.significance` rows with `horizon="joint"`; single-horizon
  specs requesting either test fail fast at `pipeline_spec(...)` build time.
- `deps` (WP-A4, torch install + Tier-1 anchor coverage completion): CPU-only
  `torch` (`pip install torch --index-url
  https://download.pytorch.org/whl/cpu`) installed into the shared dev
  `.venv` (torch 2.12.1+cpu, ~2.0GB added), closing the last
  environment-blocked gap from WP-V2's anchor-coverage inventory. New
  `tests/models/anchors/test_hnn_anchors.py` (torch-gated via module-level
  `pytest.importorskip("torch")`, so `ci-core` -- which does not install
  `[deep]` -- collects and skips the whole file cleanly, verified by
  simulating torch-absence via meta-path/`sys.modules` import blocking
  rather than uninstalling) anchors both previously-`BLOCKED(no-torch)`
  Tier-1 zero-anchor models, `hemisphere_nn` and `density_hnn`: (1) a seeded
  determinism pin (bit-identical `fit`+`predict`+bagged-member
  `state_dict()` across repeated same-seed fits; different seed changes the
  result -- no silent RNG leak), (2) a closed-form zero-weight
  architecture-limit anchor (every `Linear` layer zeroed except the final
  head bias collapses each network to an input-independent constant,
  hand-verified to float32 precision; `density_hnn`'s volatility-head
  normalization additionally collapses to exactly its own
  `volatility_emphasis` parameter, independent of the raw pre-normalization
  bias -- both verified directly against `_TorchHemisphereNet`/
  `_TorchDensityHNNNet`, not assumed from the docstring), (3) a low-noise
  linear-DGP recovery smoke test (>= 50% RMSE improvement over a naive
  constant forecast; both models actually achieve 84-89%), and (4) for
  `density_hnn`, a density-output contract check (finite/positive variance
  always; a loose, median-based calibration-sanity band anchored on the
  model's own realized out-of-sample MSE, not the unobservable true DGP
  noise floor). 7 new tests, all green, 6.6s wall time. **Finding (reported,
  not xfail'd)**: `density_hnn`'s per-row predicted variance has a wide
  right tail even on a near-noiseless DGP; isolated by ablation
  (`rescale_volatility=False` shows the same tail, if anything larger) to
  the base bagged-ensemble disagreement across members, not the OOB
  log-linear volatility-rescaling step originally suspected -- documented
  in the new test file's module docstring, not hidden. Bonus: re-ran the
  `.dev-notes/policy_matrix_scan.py` empirical (model x policy) support scan
  for all 6 previously torch-blocked models (`lstm`, `gru`, `nn`,
  `transformer`, `hemisphere_nn`, `density_hnn`); see PR body for the full
  status table.
- `reporting` (`paper_accuracy_table`, publication on-ramp): new
  `mf.reporting.paper_accuracy_table(report, *, target=None, metric="rel_rmse",
  star_levels=..., mcs_mark="†", benchmark_row=True)` joins `report.accuracy` /
  `.significance` / `.mcs` -- the three separate long frames `run_pipeline`
  returns -- into the one wide models-by-horizons table ("Table 3") most macro
  forecasting papers publish: rel-RMSE (computed here as `sqrt(relative_mse)`,
  added as `"rel_rmse"` to the metric rename map), Diebold-Mariano
  significance stars, and a Model Confidence Set marker, one line to
  `.to_latex(booktabs=True)` / `.to_html()` / `.to_markdown()`. Multi-target
  reports return `dict[target, ReportTable]` (folding targets into one frame
  would force a ragged union of horizons); an explicit `target=...`, or a
  single-target report, returns the `ReportTable` directly. Tests:
  `tests/reporting/test_reporting.py` (hand-built fixture -> exact wide frame
  incl. stars/marks, LaTeX render smoke, missing-benchmark and single-horizon
  edge cases).

- `docs/guide` (`custom_data_tutorial`, publication on-ramp): new capstone
  tutorial "Your Data, Your Model, One Table"
  (`docs/guide/custom_data_tutorial.md`) connects `load_custom_csv`/
  `custom_dataset` -> `TargetSpec(transform=...)` -> `custom_model(...)` ->
  `run_pipeline` -> `paper_accuracy_table(...).to_latex()` end to end on a
  small synthetic panel; every code block was executed for real and the
  output shown is genuine. Closes the audited gap where the `custom_dataset`/
  `custom_model` reference pages each stopped one step short of a scored
  comparison (`custom_dataset` dead-ended at `reprocess`, `custom_model`'s
  "Runner Use" example used low-level `forecasting.run` -- and, incidentally,
  showed a multi-model dict that `forecasting.run` now rejects since `run` is
  atomic; fixed in the same pass); both reference pages now cross-link to the
  tutorial.

- `forecasting` (implicit default-feature-spec `UserWarning`, publication
  on-ramp): an arm with a supervised model (`random_forest`, a custom model,
  ...) and `features=None` silently resolves to
  `feature_spec(target=..., horizon=..., target_mode=..., target_transform=...)`
  with no explicit `predictors`/`lags`/`target_lags`. That FeatureSpec's
  `predictors=None` falls back at FIT time (`_resolve_predictors` in
  `feature_engineering/shared.py`) to `base.predictors`'s own default,
  `"all"` -- i.e. it uses EVERY other panel column at lags 0 and 1 with no
  feature engineering, and, because `target_lags` is never set on this path,
  NOT the target's own lags either. This is the opposite of what a
  signature-only reading of `feature_spec(predictors: ... = None)` suggests
  (verified empirically by tracing the resolved `FeatureSpec.predictors`
  through to the actual fitted `X` columns; documented in
  `docs/reference/feature_engineering.md`'s `predictors` row, but easy to miss
  without running it). `_feature_spec_for_policy` now emits one `UserWarning`
  (relies on Python's default warning-filter dedup, not custom bookkeeping,
  so a 60-cell pipeline grid warns once per call site rather than once per
  cell) when this branch fires for a supervised model; `ar`/`far` are carved
  out (their documented construction is an explicit target-lags-only
  `FeatureSpec`, matching the same ar/far carve-out in
  `pipeline.spec.DIRECT_POLICY_GUARD_MODELS`), and panel-/target-input models
  never reach this branch at all. Fixed `getting_started.md`'s "single
  forecast" quickstart, whose AR and RF arms both relied on this default: AR
  was silently NOT an autoregression (it was regressing on the whole FRED-MD
  panel excluding its own lags), and RF's implicit, untuned 254-feature
  (127 columns x 2 lags) matrix -- combined with `random_forest`'s own
  per-origin hyperparameter search, which runs by default whenever a search
  space exists and is not explicitly disabled -- made the quickstart take
  upwards of 40 minutes to run for real. Both arms now get explicit, curated
  `FeatureSpec`s and RF's per-origin search is disabled
  (`model_selection={"random_forest": None}`) for this quickstart, bringing
  it under 2.5 minutes; "A full study" below it is unaffected (both arms
  there already had explicit features). Test:
  `tests/forecasting/test_default_feature_spec_warning.py`.

- `docs/guide` (glossary/running, `recursive` policy, publication on-ramp):
  `docs/guide/glossary.md` had no `recursive policy` entry at all -- only
  `direct policy`/`direct_average policy`/`path_average policy` were defined,
  even though `recursive` (and its `forecast_policy="iterated"` code alias)
  has existed in the policy registry throughout. Added the term
  (cross-linked from `path_average policy`, which already referenced "the
  separate recursive policy" in prose) and a "Textbook mapping" table to
  `docs/guide/concepts/running.md` (direct -> `direct`; iterated/recursive ->
  `recursive` (alias `iterated`); the `*_average` variants as h-average
  forms). `running.md`'s `path_average` semantics (a direct per-step
  average, not an iterated one) were already corrected in #420; this pass
  only adds the missing `recursive` policy and its textbook cross-reference.

- `tests/parity` (WP-V1, R-parity verification harness): new `tests/parity/`
  directory (marker `rparity`, opt-in via `pytest tests/parity/ -m rparity`,
  excluded from `ci-core.yml` explicitly) anchors the highest-priority gaps
  from the WP-V0 anchor-coverage inventory (`.dev-notes/anchor_coverage/`)
  against R, via a subprocess-`Rscript` bridge (`tests/parity/conftest.py`)
  rather than `rpy2` -- the only `rpy2` wheel available here (3.6.7) fails to
  import against this host's R 4.3.3 build in both API and ABI mode
  (`undefined symbol: R_getVar`/`R_ParentEnv`), confirmed even with
  `R_HOME`/`LD_LIBRARY_PATH` set explicitly. 37 parity tests across 7 items:
  restored the deleted C59 R-crossref suite (`git show
  7050f5d2:tests/core/test_r_crossref_c59.py`, adapted to the current API) for
  `realized_garch` vs `rugarch::ugarchfit(model="realGARCH")` (all 9
  overlapping parameters match to atol=0.01 across two seeds -- promotes
  `realized_garch` from zero-anchor Tier-1 to a tightly-verified P-anchor) and
  `restricted_midas`/`midas_almon` vs `midasr::midas_r(weight_method=
  "nealmon")`; HAC long-run-variance kernel parity vs R `sandwich::kernHAC`
  for `dm_test`'s 3 usable kernels; `dm_test` vs `forecast::dm.test` (h=1/4,
  HLN-corrected and raw, power=1/2, 6/6 pass); `model_confidence_set`/
  `_mcs_statistic` vs R `MCS::MCSprocedure` (deterministic `GetD` core exact,
  survivor-set match); `_berkowitz_density_test` vs a clean-room R port of
  Berkowitz (2001); exact-value oracles for `gaussian_nll`/`crps`/`log_score`/
  `negative_log_score`/`qlike`/`smape` (crps also vs `scoringRules::crps_norm`
  and `crps_sample`); `mars` vs R `earth` prediction parity. Full outcome
  table: `.dev-notes/anchor_coverage/v1_parity_results.md`.
  **Suspected bugs found (reported here, NOT silently fixed per the WP-V1
  mandate)**: (1) `_long_run_variance(..., kernel="andrews")` -- and every
  public callable that forwards to it (`dm_test`, `gw_test`,
  `harvey_newbold_test` confirmed directly, `clark_west_test`/`cw_test`/
  `enc_t_test` likely) -- ALWAYS raises `ValueError: unknown HAC kernel
  'newey_west'` for any input: the code sets `kernel = "newey_west"` after
  computing the Andrews/Newey-West automatic bandwidth, but the weighted-sum
  branch below is spelled `"bartlett"`, not `"newey_west"`, so execution
  always falls through to the final `raise`. (2) `_long_run_variance`'s
  `"bartlett"` branch weights lag k as `1 - k/(bandwidth + 1)` (Newey-West
  1987 convention) while its own `"acf"`/`"parzen"` siblings -- and R
  `sandwich::kernHAC`'s Bartlett kernel -- use `1 - k/bandwidth` (Andrews
  1991 convention); confirmed to full double precision this is a real
  internal inconsistency, not a numerical artifact (both xfailed
  `strict=True` in `tests/parity/test_hac_kernels.py`, not silently
  loosened). Note: the public `dm_test(kernel="bartlett", horizon=h)` path
  is NOT affected by finding (2) -- its bandwidth is always `horizon - 1`,
  which happens to make macroforecast's formula coincide with
  `forecast::dm.test`'s own bartlett convention (verified: 6/6 pass in
  `test_dm_test.py`).

- `tests.py` (bugfix, HAC kernel finding 1 above): fixed
  `_long_run_variance(..., kernel="andrews")`, which had been unusable
  since it was introduced -- every call raised `ValueError: unknown HAC
  kernel 'newey_west'` because the code reassigned `kernel = "newey_west"`
  after computing the Andrews/Newey-West AR(1) plug-in bandwidth, but the
  linear-taper branch below is spelled `"bartlett"`, not `"newey_west"`.
  Reassigning to `"bartlett"` directly unbreaks `kernel="andrews"` for
  `dm_test`, `gw_test`, `harvey_newbold_test`, `clark_west_test`/`cw_test`,
  and `enc_t_test` (all now exercised end-to-end in
  `tests/parity/test_hac_kernels.py`, which also hand-computes the expected
  `_long_run_variance(kernel="andrews")` value independently and matches it
  to 1e-10). The former strict-`xfail` crash reproduction in that file is
  replaced by these positive regression tests; zero `xfail`s remain in
  `tests/parity/`.

- `tests.py` (docs/clarification, HAC kernel finding 2 above): documented,
  in `_long_run_variance`'s new docstring, the intentional bandwidth-taper
  asymmetry across its kernel branches -- `"bartlett"` uses the Newey-West
  (1987) taper `1 - k/(bandwidth + 1)` (kept, NOT aligned to `"acf"`/
  `"parzen"`, because it is required for the verified 6/6 `dm_test` parity
  with `forecast::dm.test`, whose own bartlett bandwidth is always
  `horizon - 1`), while `"acf"`/`"parzen"` use the Andrews (1991) taper
  `1 - k/bandwidth` (matching R `sandwich::kernHAC` for those two kernels).
  Behavior is byte-identical to before this entry -- only documentation and
  test coverage changed. The former strict-`xfail` R-parity mismatch for
  `"bartlett"` in `tests/parity/test_hac_kernels.py` is replaced by explicit
  documented-divergence assertions (same style as the `midas_almon`
  architecture-difference finding in `test_midas_almon.py`): our bartlett
  value is asserted to double precision against an independently
  hand-computed NW-1987 fixture, and separately asserted to deliberately
  differ from R `sandwich::kernHAC`'s Bartlett value, with a comment
  pointing at the `dm_test` parity dependency it protects.
- `tests/models/anchors/` + `tests/parity/` (WP-V2, model-anchors-v2): independent
  correctness anchors for the remaining Tier-1 (zero-anchor) MODELS identified by
  `.dev-notes/anchor_coverage/summary.md` -- `bvar_minnesota`,
  `bvar_normal_inverse_wishart`, `favar`, `dfm_mixed_mariano_murasawa`,
  `tvp_ridge`, `lgb_plus`, `lgba_plus`, `assemblage_regression`,
  `supervised_aggregation`. `hemisphere_nn`/`density_hnn` remain
  BLOCKED(no-torch) -- torch is deliberately not installed on this host; see
  `.dev-notes/anchor_coverage/v2_anchor_results.md` for the proposed anchor
  design once torch is available. `bvartools`/`FAVAR` R packages now build
  cleanly from CRAN on this host (`BVAR` also installs but was not needed),
  enabling live R parity tests alongside the from-formula clean-room ones.
  Full per-model anchor-type/tolerance/result table in
  `.dev-notes/anchor_coverage/v2_anchor_results.md`.

  - `bvar_minnesota`/`bvar_normal_inverse_wishart`: the Minnesota prior
    variance grid (own-lag/cross-lag/lag-decay) is checked deterministically
    (1e-12) against the documented `bvartools::minnesota_prior` formula two
    ways -- an independent from-scratch reimplementation, and a live R
    subprocess-Rscript parity test against `bvartools::minnesota_prior`
    itself. Separately, the Normal-Inverse-Wishart Gibbs sampler's posterior
    mean is shown (Monte Carlo, tolerance set from the sampler's own reported
    MCSE) to converge to the closed-form OLS estimate on the shared VAR
    design -- the exact special case where a diffuse coefficient prior makes
    the GLS estimator, for any sampled error covariance, algebraically
    identical to OLS (shared-regressor SUR equivalence).

    **FINDING (RESOLVED, see the correctness fix below)**:
    `bvar_normal_inverse_wishart`'s own default `s0=0.0` (an
    exactly-zero inverse-Wishart prior scale) made the Gibbs sampler
    numerically unstable -- posterior mean off by 3-4+ orders of magnitude,
    ~97% of post-burn-in draws divergent -- whenever the fitted VAR's
    residual covariance was even mildly near-singular (a realistic scenario,
    not a corner case: e.g. a FAVAR-style factor+target block where the
    factors explain the target well). `favar()`'s own default
    `varprior=None` silently resolved to this same fragile configuration
    with no warning. Originally marked `xfail(strict=True)` with the full
    isolating diagnosis in `test_bvar_minnesota_niw_anchors.py`; not
    tolerance-loosened away. Reported prominently in the WP-V2 results file.
    Fixed in the same PR -- see "BVAR-NIW default prior scale + diverging-
    draws guard" below.

  - `favar`: the fully deterministic factor-identification helpers
    (`_favar_extr_pc`/`_favar_facrot`/`_favar_olssvd`/`_favar_bgm`) now
    have both an independent PCA-via-SVD cross-check and live R parity
    against `FAVAR:::ExtrPC`/`facrot`/`olssvd`/`BGM` (the exact R
    functions the docstrings claim alignment with). A near-noiseless
    known-factor DGP oracle (small idiosyncratic noise + an explicit
    non-degenerate `varprior`, both required to avoid the `s0=0.0` finding
    above and a separate BGM-on-exactly-rank-deficient-data degeneracy)
    recovers the forecast to <1% of the target's in-sample range through the
    full `favar()` pipeline.

  - `dfm_mixed_mariano_murasawa`: confirmed (by reading the source) to be a
    thin wrapper around `statsmodels.tsa.statespace.dynamic_factor_mq
    .DynamicFactorMQ`, so the wrapper's own glue (column reordering,
    frequency inference) is anchored directly -- parameter/log-likelihood/
    fitted-value pass-through vs a direct `DynamicFactorMQ` call on an
    identically-shaped frame, including a column-order-shuffle variant -- plus
    a low-noise Mariano-Murasawa [1,2,3,2,1] mixed-frequency factor-recovery
    oracle (correlation with the true factor > 0.9).

  - `tvp_ridge`: `_dual_generalized_ridge` ("dualGRR") is shown, via an
    independent Woodbury-identity derivation (not read off the code), to
    solve the textbook generalized-ridge/Tikhonov estimator
    `theta = (Z'WZ + Lambda)^-1 Z'Wy`; matched to 1e-8 against a from-scratch
    reference solve, including a heterogeneous-weights case. A
    constant-parameter limit test confirms lambda1 -> infinity collapses the
    recovered path to a single plain (static-only) ridge fit with penalty
    lambda2, the textbook "TVP collapses to constant coefficients as state
    variance -> 0" limit.

  - `lgb_plus`/`lgba_plus`: the competition/alternating ensemble logic is
    clean-roomed against hand-rolled loops calling raw `lightgbm` directly
    (`lightgbm` extra now installed) -- for `lgb_plus` this includes
    reproducing the exact `np.random.default_rng(seed)` call sequence
    (subsample draw, then linear-candidate-feature draw, per step) so the
    random streams -- and therefore predictions -- match bit-for-bit; for
    `lgba_plus` (deterministic at `subsample=1.0`) no RNG replay is needed.
    Both get a same-seed determinism pin.

  - `assemblage_regression`/`supervised_aggregation`: both route through
    the shared `SupervisedAggregationRegressor` (confirmed by reading
    `assemblage.py`), anchored with hand-computable analytic fixtures --
    plain ridge, target-shrinkage ridge (both unconstrained closed forms),
    and simplex-/mean-match-constrained ridge (solved exactly via the
    bordered KKT linear system, with an explicit check that the `nonneg`
    inequality constraint is non-binding so the equality-only closed form is
    the correct target) -- plus a thin check that the public
    `assemblage_regression`/`supervised_aggregation` entry points route to
    the same verified estimator.

  - Cheap bonus: a git-archaeology pass over historically deleted test files
    (`git log --diff-filter=D --name-only -- 'tests/**'`) found dedicated,
    substantive (59-424 line) BVAR/FAVAR/DFM/HNN test files removed in the
    `2e62e740` "Clean semantic package structure" reorg (same commit V0
    flagged for the deleted R cross-ref suite). Spot-checked: these import a
    now-nonexistent `macroforecast.core.runtime` / `macroforecast.models.ops`
    (`OPERATIONAL_MODELS`/`FUTURE_MODELS`/`get_family_status`) registry API,
    so they are NOT directly restorable (`git show > file` would not import);
    restoration would need rewriting to the current module structure. Full
    candidate list with commits/line counts in the WP-V2 results file; not
    restored here per the task brief.

- **`models` (CORRECTNESS, follow-up to the WP-V2 finding above): BVAR-NIW
  default prior scale + diverging-draws guard.** Users who called
  `bvar_normal_inverse_wishart`/`bvar_minnesota`/`favar` with their documented
  defaults (i.e. never passed `s0`) got a *different, and previously silently
  wrong, result* on any near-singular-residual-covariance panel. This is a
  breaking behavior change for anyone relying on the old default numerically
  -- read this entry before upgrading if you fit these models without an
  explicit `s0`.

  - **Old default**: `s0=0.0` (an exactly-zero inverse-Wishart prior scale).
    On a near-singular VAR residual covariance this made the Gibbs sampler's
    Wishart-draw scale matrix equal the bare sample residual cross-product
    with no floor, and posterior means silently diverged by 3-4+ orders of
    magnitude (~97% of post-burn-in draws divergent; `favar()`'s default
    forecast on a realistic near-noiseless factor-augmented-VAR DGP was
    `-8.6e24` vs a true value of about `-1.0`) -- no error, warning, or NaN.

  - **New default**: `s0=None`, which resolves (in `_favar_bvar_draws`, via
    the new `_favar_default_niw_scale` helper) to a data-dependent diagonal
    scale `diag(sigma_1**2, ..., sigma_k**2)`, where `sigma_i` is each
    series' own univariate AR(`n_lag`)-OLS residual SD on the fit data (the
    same per-equation AR-residual helper already used, and independently
    anchored, for the Minnesota prior's cross-lag variance scaling). This is
    the standard natural-conjugate NIW-BVAR convention: under this module's
    posterior parameterization (`Sigma | data ~ InvWishart(S0 + SSE, tnum +
    nu0)`, `E[Sigma] = scale / (df - k - 1)`), the minimal proper integer
    degrees of freedom keeping a no-data prior mean finite is `k + 2`, and at
    that reference point `S0 = diag(sigma_i**2)` makes the prior's implied
    residual-covariance mean equal each equation's own AR/OLS variance
    estimate (Karlsson 2013, Handbook of Economic Forecasting Vol. 2B Ch. 15
    -- already this repo's own reference for the sibling Minnesota-prior
    sigma^2-scaling fix, see the "PR8" entry further down; Giannone, Lenza &
    Primiceri 2015, "Prior Selection for Vector Autoregressions", Review of
    Economics and Statistics 97(2)). `s0` stays fully exposed and an
    explicitly passed value (including `s0=0.0`, to reproduce the exact old
    behavior) is honored exactly as before. `nu0`'s own default (`0.0`) is
    unchanged -- only the `s0` default changed.

  - **New diverging-draws guard**: `_warn_if_bvar_draws_diverged` now checks,
    after every fit, what fraction of post-burn-in Gibbs coefficient draws
    have a coefficient-matrix Frobenius norm more than 50x the plain-OLS
    reference scale on the same shared VAR design; if more than 10% do, a
    `UserWarning` is raised ("near-singular residual covariance with
    (near-)zero prior scale s0 -- posterior draws diverged; set s0>0 or
    rescale data"). This never raises -- it is a visibility fix, not a
    behavior-restricting one -- and it still fires for an explicitly passed
    `s0=0.0` on a fragile panel (that path's dedicated regression test), so
    explicit callers who want the old numerics are still warned rather than
    silently misled. Bonus finding surfaced by this guard: an existing,
    unrelated smoke test (`test_bvar_and_target_timeseries_models_fit_and_
    predict`) fits `bvar_normal_inverse_wishart` on a fully deterministic
    (zero-noise) synthetic panel; even the new data-dependent `s0` is
    numerically tiny there (the AR-residual-variance estimate is itself
    near-zero on noiseless data), so the guard correctly still fires --
    previously this same fit silently produced a ~6.4e13-magnitude
    coefficient with no signal at all.

  - **`favar`**: no `favar`-specific code changed. `_parse_favar_varprior`'s
    two dict-building branches previously hardcoded `"s0": prior.get("s0",
    0.0)`, so `favar(varprior=None)` (favar's own default) always forwarded
    an explicit `s0=0.0`, bypassing whatever `bvar_normal_inverse_wishart`'s
    own default was. Both branches now use `prior.get("s0")` (defaulting to
    `None`), so `favar`'s default transparently inherits the same fixed
    default through the shared `_favar_bvar_draws` resolution -- proven by a
    new end-to-end test (`test_favar_default_varprior_recovers_forecast`)
    that asserts both forecast accuracy AND the absence of the new warning
    on the exact DGP that used to produce `-8.6e24`-scale garbage.

  - Tests: the WP-V2 `xfail(strict=True)` regression pin in
    `test_bvar_minnesota_niw_anchors.py` is now two positive tests --
    `test_niw_default_s0_matches_closed_form_ols_on_previously_diverging_
    fixture` (default-`s0` posterior mean matches the closed-form OLS anchor
    on the exact previously-diverging fixture, same MCSE-based tolerance as
    the other OLS-anchor tests) and `test_niw_explicit_s0_zero_still_
    diverges_and_warns` (explicit `s0=0.0` still diverges exactly as before,
    but now raises the guard's `UserWarning`) -- plus
    `test_favar_default_varprior_recovers_forecast` in
    `test_favar_anchors.py`. All pre-existing WP-V2 anchor tests (the
    explicit-`s0=1.0` diffuse-prior Gibbs-vs-OLS tests) are confirmed
    unaffected. `mypy` clean on all touched files; no new errors in the
    project-wide baseline (still the same 4 pre-existing, unrelated errors
    in `data_analysis/summary.py`/`models/linear.py`).

  - `macroforecast/models/specs.py`: the `bvar_minnesota`/
    `bvar_normal_inverse_wishart` `MODEL_SPECS` entries' `default_params`
    (consumed directly by `ModelSpec.fit()`/recipe-driven runs, which bypass
    the Python function's own keyword default) are updated to `s0=None` to
    match, so recipe/`forecasting.run()`-driven fits inherit the fix too.

- `tests/mc` (WP-V3, Monte Carlo size/power validation harness): new `tests/mc/`
  directory (marker `mc`, opt-in via `pytest tests/mc/ -m mc`, excluded from the
  `ci-core.yml` default run alongside `slow`) validates the STATISTICAL
  calibration of the inference-critical forecast-comparison tests -- empirical
  rejection rate under a TRUE null vs nominal alpha, judged against exact
  Clopper-Pearson 99% binomial bands (fixed before looking at results, never
  widened to pass), master seed 20260705 with `SeedSequence`-spawned independent
  per-replication streams, R=1000-5000 per design cell, full suite ~5m20s on a
  single core. This is complementary to WP-V1/V2 parity: parity proves we match
  a reference implementation, MC size proves the statistical behavior is right
  -- and catches distribution/HAC/df errors both sides of a parity check can
  share. 21 tests: 14 passed, 7 xfailed (`strict=True` -- a future fix flips
  them to loud XPASS). Correct calibration POSITIVELY verified for: `dm_test`
  at h=1 (n=50/200, HLN and none, alpha=.05/.10) and n=200 h=4, plus the
  Harvey-Leybourne-Newbold DIRECTION property (HLN strictly closer to nominal
  than uncorrected at n=50 h=4); `clark_west_test` under a nested null with
  demonstrably induced nested bias (does not over-reject; conservative exactly
  as Clark-West 2007 reports, with mid-range power .55 against a small true
  alternative); `model_confidence_set` retaining a genuinely dominant model at
  .947 >= 1-alpha; `pesaran_timmermann_test` at n=200;
  `superior_predictive_ability_test` on iid loss panels; `giacomini_white_test`
  at h=1. **Size-distortion findings (reported as strict xfails + diagnosis in
  `.dev-notes/anchor_coverage/v3_mc_results.md`, NOT silently absorbed)**:
  (1) `giacomini_white_test` is genuinely oversized for h>1 (.124 at
  alpha=.05/n=50/h=4; still .072-.092 at n up to 5000 -- does NOT vanish
  asymptotically; distortion is monotone in the number of Newey-West HAC lags
  h-1, mean statistic 2.29 vs chi2(2) mean 2.00 at n=5000) -- the HAC-Wald
  construction lacks any small-sample correction analogous to HLN; a real
  calibration issue for multi-horizon use worth a code fix. (2)
  `model_confidence_set` under-covers the equal-losers GLOBAL null: P(all 5
  tied models retained)=.818 vs the >=.90 floor, stable across n_boot
  1000/2000, n=100/200, alpha=.10/.05, with/without a common loss factor --
  isolates an over-rejecting first-step elimination test built on
  `_mcs_statistic` (the same private helper WP-V0 ranked as tests.py gap #1).
  (3) `superior_predictive_ability_test` over-rejects ~1.5-2x nominal whenever
  losses are serially correlated (AR(1) rho=.3 suffices), insensitive to
  p_value_type/studentize/block_length(1-20,auto)/n(100-500), while the iid
  design is in band -- tracks the `arch.bootstrap.SPA` backend's
  block-bootstrap calibration, not the macroforecast panel layer. (4) `dm_test`
  is oversized at n=50 h=4 (.065 HLN / .095 uncorrected at alpha=.05,
  re-confirmed at R=10000) -- too few effective non-overlapping blocks for 3
  HAC lags; shared with forecast::dm.test by construction (formula is
  parity-matched), documented as a usage caveat rather than a bug. (5)
  `pesaran_timmermann_test` mildly oversized at n=50 (.056 at alpha=.05,
  R=15000) -- PT-1992 asymptotic-normal reference at small n. Raw machine
  results: `tests/mc/_mc_raw_results.json` (rewritten by each `-m mc` session).
- `forecasting`/`pipeline` (performance, Gap A): the per-origin fitted feature
  builder (`FeatureSpec.fit()` -- the PCA/MARX/SIR-style numerical state) is now
  shared across arms of the same target in the serial pipeline path, exactly like
  the per-origin `FittedPreprocessor` already was. Previously the fitted builder
  lived in a `run()`-local variable, so two arms differing only in `model` -- the
  most common pipeline comparison -- refit the feature transform at every origin;
  `docs/reference/pipeline.md` over-claimed that model comparison recomputes "only
  the model fit/predict" (now corrected). New
  `macroforecast/forecasting/feature_stage.py` holds the sharing helper: the cache
  key is `("features", <sha256 of FeatureSpec.to_dict() + feature StagePolicy
  .to_dict()>, (scope, fit_start_pos, fit_end_pos))` -- a content digest (never
  object identity) plus the EXACT per-origin fit-sample row bounds, so an arm with
  its own `window` (different estimation rows at the same origin) or a `custom`
  feature scope can never be served another arm's fit; unresolvable bounds mean
  "do not share", never a wildcard. Horizon/forecast-policy dependence is carried
  by the digest itself (`_feature_spec_for_policy` bakes the resolved horizon,
  `target_mode`, and `target_transform` into the spec). Sharing rides the existing
  per-target `preprocessing_cache` dict (value type widened to `FittedPreprocessor
  | _PreparedStage | FittedFeatureBuilder`), which `pipeline/run.py` now builds
  unconditionally in the serial backend (previously only when spec-level
  `preprocessing` was set) so feature-only pipelines share too; an arm opts out of
  ALL sharing by overriding `preprocessing` OR `window`. `never`/interval feature
  update cadences keep their exact semantics (the cache only changes HOW a due
  refit is computed, not WHEN). The parallel backend (`preprocessing_cache=None`)
  and direct `run()` calls without a cache dict are byte-for-byte unchanged.
  Forecasts are pinned byte-identical to the unshared path by
  `tests/pipeline/test_feature_cache_sharing.py` (fit-counter: once per origin
  across arms; divergent-window no-wrong-share; `update="never"` single total fit)
  and the existing golden snapshot / serial==parallel / oracle gates.
- `forecasting`/`window` (CRITICAL correctness fix): panel-input models (`var`,
  `bvar_*`, `dfm_mixed_mariano_murasawa`, `midas_almon`) mislabeled the forecast
  horizon -- a `horizons=[2]` request emitted a multi-step path but tagged EVERY
  row `horizon=1`, and the requested horizon was never produced (`(horizon,
  origin)` was not a unique key). Root cause was two compounding bugs: (1)
  `WindowSpec.origins()` built the panel test slice as `[origin_pos, origin_pos +
  horizon - 1]`, so the origin's own date doubled as the first test row instead
  of being excluded from the test window; (2) `_panel_prediction_horizon` floored
  the computed distance to `max(1, ...)` to compensate, so the origin's own row
  (true distance 0) and the genuinely-1-step-away row both read horizon=1. Fix:
  `WindowSpec.origins`/`plan`/`iter_origins`/`validate` take a new
  `exclude_origin` keyword (default `False`, preserving byte-identical behavior
  for every existing caller); the panel runner (`_run_panel_models`) opts in.
  Under `exclude_origin=True` the origin is the LAST IN-SAMPLE date: the
  estimation window runs through the origin (`origin_pos - embargo`, previously
  `origin_pos - embargo - 1`) and the test slice is `[origin_pos + 1, origin_pos
  + horizon]`. Fitting through the origin is what makes the labels TRUE: panel
  models forecast positionally from the end of their fit data, so forecast step
  s now lands exactly on `origin + s` (pinned by a `_VAR` value oracle in the
  regression test, `abs=1e-12`). `_panel_prediction_horizon` no longer floors
  the distance, and `_fit_predict_panel_origin` emits ONLY the row matching the
  requested horizon (matching the supervised direct-policy contract of one row
  per `(origin, horizon)`); intermediate path steps are dropped so a
  multi-horizon request can no longer create duplicate `(horizon, origin)` keys.
  Supervised/target/volatility window plans are untouched (`exclude_origin`
  defaults off and only the panel call site sets it -- their `test_idx[0] ==
  origin_pos` "as-of" feature-row contract is unaffected). Behavior changes:
  panel forecast VALUES shift because the information set now includes the
  origin's own observation (each emitted value is a true h-step-ahead forecast
  conditioned on data through the origin); origins near the sample end now
  require data through `origin + h`, so panel origin coverage at longer horizons
  shifts to align with the supervised path; and for mixed-frequency panel
  models a horizon must land on a date where the target realises (the old
  window "supported" quarterly targets at h=1 only because the mislabeled
  first test row was the quarter-end origin itself). Fixes #423.
- `models` (favar unusable with defaults): `favar`'s default `fctmethod` was
  `"BBE"` with `slowcode=None`, but `_FAVAR.fit` hard-requires `slowcode` under
  BBE, so every trial in the default "standard" search space failed instantly
  with `SearchError: ... slowcode is required when fctmethod='BBE'`. Default
  `fctmethod` is now `"BGM"` (needs no `slowcode`); the BBE+missing-slowcode
  error is unchanged (pinned by a new regression test), and `fctmethod` is
  still NOT part of the search space (search dimensionality unchanged, as
  locked). Making the default trial actually run also exposed a second,
  previously invisible problem: `favar`'s Gibbs/Wishart posterior sampler
  (`nburn`/`nrep` = 5000/15000) combined with the *shared* `_FACTOR_SPACES`/
  `_AR_SPACES` "standard" grid (n_factors up to 8, n_lag up to 12 -- cheap for
  the OLS-based `far`/`ar`/`var` models that also use it, but not for a
  Bayesian sampler that rebuilds an `O((n_factors + 1) * n_lag)`-square
  posterior every MCMC draw) made even a single default-preset grid search
  take many minutes. `favar` now has its own dedicated search space
  (`_FAVAR_SPACES`, NOT shared with `far`/`ar`/`var`) with the same two tuned
  dimensions (`n_factors`, `n_lag`) but a much tighter "standard"/"wide" range,
  and cheapened `nburn`/`nrep` defaults: `5000`->`100` / `15000`->`200`. Old
  (paper-faithful) values remain reachable by passing them explicitly. Verified
  with the empirical policy-matrix scan: favar now reports `OK` on all 4
  policies (previously all 25/25 default-preset trials failed).
- `models` (dfm_unrestricted_midas prediction NaN): `predict_from_panel`
  rebuilt the DFM-factor design at prediction time and hard-rejected any NaN
  in it. With the default `factor_lags=(0,)` and `target_frequency="quarterly"`
  / `anchor_position="period_end"`, the lag-0 factor lookup date is the
  anchor's *enclosing quarter-end* month, which can fall several months past
  the last date the DFM's own predictors cover (e.g. forecasting April with
  `h=2` projects the anchor to the June quarter-end) -- so
  `dfm_factor1_lag0` was NaN and `direct`/`direct_average`/`path_average` all
  failed (`recursive` was, and remains, correctly rejected by the panel-input
  guard). `_dfm_unrestricted_midas_design` now extends the fitted
  `DynamicFactorMQ` state forward to the anchor date via the model's own
  `.extend()` Kalman-filter forecast (all-missing placeholder observations for
  the gap months) instead of reindexing to NaN -- i.e. it forecasts the
  factors forward using the fitted state, rather than manufacturing a missing
  value. Also added a fit-time validation: `target_frequency` must normalize
  to one of monthly/quarterly/annual, raising a clear `ValueError` instead of
  silently falling back to a monthly anchor projection. Verified with a
  regression test reproducing the original failure (monthly toy panel,
  default params, `direct` policy, `h=2`) and with the empirical policy-matrix
  scan (`direct`/`direct_average`/`path_average` now `OK`).

- `models` (pathological default costs): six models exceeded 150s on a 140-obs
  toy policy-matrix scan (`.dev-notes/policy_matrix_scan.py`) with their
  out-of-the-box defaults, either from the default params themselves or from
  the "standard" search-space preset that `model_selection=None` still runs
  (dispatch is out of scope; only per-trial cost changed). Old -> new:
  - `bvar_minnesota` / `bvar_normal_inverse_wishart`: `iter` `10000`->`300`,
    `burnin` `5000`->`100`; "standard" preset `n_lag` `(1, 2, 4)`->`(1, 2)`
    (own dedicated spaces dict, not shared with other models). With the
    original `iter=10000`/`burnin=5000` defaults, a single fit did not finish
    within 280s even at `n_lag=1` (the smallest combo); at an intermediate
    `iter=2000`/`burnin=1000` a single fit alone ran 12.8-41.4s depending on
    `n_lag` (1/2/4) -- still far too slow once multiplied across the
    "standard" grid, origins, and 4 policies. Verified scan total after the
    fix (4 policies, `iter=300`/`burnin=100`, `n_lag` capped at `(1, 2)`):
    28.41s.
  - `favar`: see the separate favar entry above (`nburn`/`nrep` and its new
    dedicated search space). Verified scan total 54.6s.
  - `macro_random_forest`: default `B` `50`->`25`; "standard" preset `B`
    `(25, 50, 100)`->`(10, 25)` (own dedicated spaces dict). Verified scan
    total 81.8s (was 241.9s uncontended, i.e. already over the 150s bar).
  - `lgb_plus`: default `n_ensemble` `10`->`3`, `n_steps` `200`->`30`;
    "standard" preset `n_ensemble` `(5, 10)`->`(3, 5)`, `n_steps`
    `(100, 200, 400)`->`(30, 50, 100)` (own dedicated spaces dict). Verified
    scan total 84.2s (was: did not finish within 300s before this change).
  - `mars`: default unchanged (`max_degree=1`, `max_terms=20`); "standard"
    search preset drops the `max_degree=2` x `max_terms=30` corner
    (`max_terms` `(10, 20, 30)`->`(10, 20)`); "wide" preset unchanged (still
    reaches `max_terms=30` x `max_degree=2` for deep/explicit use). That corner
    alone took 3.7s per fit in isolation (vs. 1.0s for the plain default), and
    with a 20-trial random search repeated across multiple origins and 4
    policies it was the single biggest cost driver in the "standard" preset.
    Verified scan total after the fix: 132.9s (this WP does not have a clean
    "before" scan total on record -- the corner was identified and removed
    based on the isolated per-fit timing above, not a full before/after scan
    comparison).
  - `restricted_midas`: `maxiter` `1000`->`200`, `tolerance` `1e-8`->`1e-6`.
    Verified scan total 38.6s (was 211.3s, i.e. already over the 150s bar).
  Deep/paper-faithful settings remain reachable by passing the old values
  explicitly (or, for `mars`, using the "wide" preset). A new
  `tests/models/test_default_cost_budget.py` (marked `@pytest.mark.slow`,
  this repo's opt-in gate for realistic-shape integration tests -- there is
  no registered `deep` marker) pins that a single default fit for each model
  stays inside a generous wall-clock budget, to catch a future default
  regressing back toward a pathological cost.
- `preprocessing`/`pipeline` (footgun fix + perf default): two defects in the #416
  on-disk `PreprocessorStore` shared-cache. (1) The store key hashed
  `(PreprocessSpec.to_dict(), target, origin_pos)` but OMITTED
  `preprocessing_policy.scope` -- sharing one `preprocessing_cache_dir` across runs
  that differ ONLY in scope (e.g. one `origin_available`, one `fit_window` for the
  same spec) could silently serve one run's fitted preprocessor to the other.
  `PreprocessorStore` now accepts an optional `namespace` (any JSON-serialisable
  value, folded into the key digest); `pipeline/run.py::_execute_cell` always
  constructs its store with a namespace derived from each arm's EFFECTIVE
  `preprocessing_policy` (`StagePolicy.to_dict()`, resolved exactly as `run()`
  itself resolves it), so pipeline-driven runs are safe by construction.
  `namespace=None` (the default for anyone constructing `PreprocessorStore`
  directly) reproduces the original digest exactly. **This changes the on-disk key
  for every pipeline-driven store entry** -- existing `preprocessing_cache_dir`
  directories from before this change are silently invalidated (a clean cache MISS,
  never a wrong-answer HIT) and will be recomputed once. (2) `n_jobs>1` previously
  left every worker with `preprocessing_cache=None` and no store unless the user
  knew to set `preprocessing_cache_dir` explicitly, so parallel runs silently lost
  all cross-arm/cross-horizon EM-fit dedup. `preprocessing_cache_dir` is now a
  three-state knob: an explicit `str` path is used as-is (unchanged); `None`
  (default) auto-creates a run-scoped temporary directory for the duration of the
  run when `n_jobs>1` (removed afterward; a no-op when `n_jobs==1`, which already
  dedupes via its own in-memory cache) so parallel EM dedup now works
  out-of-the-box; `False` is the new explicit opt-out sentinel (never auto-create a
  store, matching the pre-this-change parallel behavior). `pipeline_spec(...,
  preprocessing_cache_dir=True)` now raises -- it is neither a valid path nor the
  opt-out sentinel. New tests: `test_store_namespace_isolates_cross_scope_writes`,
  `test_shared_cache_dir_does_not_cross_contaminate_across_scopes`, and
  `test_auto_cache_dir_dedupes_across_cells_when_unset` in
  `tests/pipeline/test_preprocessing_share.py`. The existing serial==parallel and
  store-off==store-on golden pins are unchanged and still pass.
- `pipeline` (spec-build-time guard): `pipeline_spec(...)` now emits a
  `UserWarning` whenever a guarded iterated/state-space model is combined with
  `forecast_policy` `direct` or `direct_average`. These models forecast a horizon
  by ITERATING their own one-step dynamics (statsmodels-style target forecasters,
  panel VAR/DFM models, `favar`'s internal factor-VAR) rather than fitting a
  genuine h-step-ahead projection, so a direct-policy request can silently degrade
  toward a stale/persistence-like forecast at longer horizons -- the same defect
  fixed for `ar`/`far` above (GCLS replication Bug 3), documented at the time as a
  follow-up for the other iterated model families. The guard WARNS, it does not
  reject: deliberate use (e.g. an intentionally weak benchmark) stays possible.
  Guarded set (new `pipeline.DIRECT_POLICY_GUARD_MODELS`, derived from
  `list_model_specs()`: every `input_kind in {"target", "panel"}` model plus
  `favar`): `arima`, `auto_arima`, `ets`, `holt_winters`, `naive`,
  `random_walk_drift`, `seasonal_naive`, `stlf`, `theta_method` (target-kind);
  `var`, `bvar_minnesota`, `bvar_normal_inverse_wishart`,
  `dfm_mixed_mariano_murasawa`, `dfm_unrestricted_midas` (panel-kind); `favar`.
  `ar`/`far` are deliberately excluded -- they now have a validated
  direct-projection mode. New `tests/pipeline/test_direct_policy_guard.py`
  (warning-fires tests for a representative target-kind and panel-kind model, no
  warning for ar/far/supervised models or for recursive/path_average, a
  cross-check that the guarded set tracks `list_model_specs()`, and a
  forecasts-byte-identical pin proving the warning has zero effect on computed
  output). New docs section `docs/reference/forecasting.md#model-x-policy-compatibility`.
  `runner.py`, `window/`, and `models/` are untouched -- the guard lives entirely
  in `pipeline/spec.py` at spec-validation time.
- `pipeline` (convenience API): new `pipeline.rescore(checkpoint_dir, spec)`
  re-scores a saved pipeline run from its checkpoint directory alone -- no
  refitting. Evaluation was already pure-frame (`evaluate(master, spec)`) and
  `forecasting.checkpoint.load_checkpoint_frame` already reconstructed one cell's
  lean records, but re-scoring a full run meant hand-walking the
  `<checkpoint_dir>/<target>__<arm>/h<h>/` tree and reassembling the master frame
  by hand. `rescore` is that glue: it walks every (target, arm, horizon) cell the
  spec describes, loads the persisted per-origin parquet, reattaches the
  `arm`/`contender` labels (the lean schema does not store them), and runs the
  standard evaluation, returning the same `PipelineReport` type as `run_pipeline`
  with `forecasts`/`accuracy`/`significance`/`mcs` populated identically to a
  live run over the same forecasts. Live-run-only fields are absent by contract:
  `interpretation` is `None`, `failed_cells` is empty (a failed cell wrote no
  checkpoints and is indistinguishable from a never-run one), `empty_cells` is
  best-effort (arms with zero checkpoint rows), and `provenance`/`leakage_audit`
  carry a `rescored_from` marker. An empty or wrong directory raises an
  actionable `ValueError` instead of returning a silently empty report. New
  module `macroforecast/pipeline/rescore.py`, exported from
  `macroforecast.pipeline`; round-trip + failure-mode tests in
  `tests/pipeline/test_rescore.py`; how-to section in
  `docs/reference/pipeline.md`.

- `forecasting` (CRITICAL correctness fix, found by a new oracle): the `path_average`
  policy fitted its per-step `ar`/`far` with the LEGACY iterated estimator
  (`direct=False`) instead of the DIRECT s-step projection, so each path step
  persisted a stale value -- negligible at h1 (~1.2e-4) but growing to O(1) at h>=2.
  On a noiseless factor DGP where the direct policy is exact to machine precision,
  path-average `far` was off by ~0.4. The per-step fit and its IC order selection now
  set `direct=True` exactly like the direct policy. This restores the h1 direct==path
  invariant to BYTE-identical for `ar`/`far` (the tolerance is back to 1e-9 from the
  interim 1e-3), and it supersedes the earlier "finite-sample edge effect" explanation
  of that gap, which was this bug.
- `tests` (independent-reference matrix, step a): anchored the two families the
  coverage audit flagged as unanchored through a policy. New
  `tests/forecasting/test_far_policy_oracle.py` drives `far`/FM through the direct AND
  path policies on a rank-4 linear-state DGP and requires recovery of the realised
  future value to machine precision, plus far>>ar discrimination (far uses predictor
  factors a 2-lag AR cannot span). New `tests/models/test_adaptive_lasso_reference.py`
  gives `adaptive_lasso` its first prediction test: a clean-room reimplementation
  (ridge-init adaptive weights -> weighted lasso -> coefficient back-map) matched to
  1e-7, sparse-support recovery, and a direct-policy ground-truth recovery that AR
  cannot achieve.
- `tests` (trust-gap audit follow-up): closed the independent-reference gaps for the
  direct/path forecasting core. (1) New `tests/forecasting/test_path_average_multistep_oracle.py`
  anchors MULTI-STEP (h>=2) path-average against external oracles for the first time:
  a noiseless-AR(2) ground-truth (forecast == realised future mean), a byte-exact
  clean-room reproduction of the per-step direct OLS on noisy data, and a path!=direct
  discrimination guard so a future collapse cannot pass silently. (2) New
  `test_crosshorizon_em_factor_split_equals_whole` proves the cross-horizon base/forward
  transform split is byte-identical to the whole-window transform under `em_factor`
  imputation (the prior identity test only covered row-independent `mean`). (3) The h1
  direct==path invariant tolerance for `ar`/`far` is tightened 5e-3 -> 1e-3 with a
  corrected rationale: the ~1.2e-4 gap is a finite-sample edge effect of the two design
  constructions (direct projection on pipeline lag features vs iterated AR on the raw
  series), verified INDEPENDENT of `n_lag` (not order selection) and identical for far
  and ar (not PCA). (4) Doc-only: the `ar`/`far` direct-mode lag-0 descriptions now
  match the code -- `*_lag0` is the observed origin value (Stock-Watson direct
  regressor), NOT look-ahead; the previous prose claimed the opposite in five places.
- `forecasting`/`models` (CRITICAL correctness fix): under the `direct`/`direct_average`
  policy, the autoregressive models `ar` and `far` produced degenerate long-horizon
  forecasts. They rolled forward from the target's own history, and because the h-ahead
  target's freshest leak-free lag is origin-h stale (and the h-period average is
  near-unit-root) they persisted a stale value — forecasts worse than the unconditional
  mean (RMSE ≈ √2·target std, ≈ uncorrelated with the realised future). Since `far` is
  the usual factor benchmark, this distorted every direct relative metric and grew with
  the horizon. `ar`/`far` now have a direct-projection mode: under the direct policy they
  regress the h-ahead target on the fresh one-period lags (the `n_lag` most recent
  observed lags, leak-free) and predict per origin, with IC order selection in the same
  mode; `recursive`/`path_average` keep the (correct) iterated behaviour. Validated on the
  GCLS (2021) replication: UNRATE direct FM absolute RMSE at horizon 24 went from 0.1016
  (49% above the paper) to 0.0726 (~7%). The 11 other iterated/state-space models (VAR
  family, `favar`, statsmodels forecasters, mixed-frequency DFM, naive baselines) share
  the defect under the direct policy and are a documented follow-up.

- `models` (correctness follow-up to the above): under the `direct`/`direct_average`
  policy `far` silently dropped its factors and collapsed to plain `ar`. In direct mode
  every feature reaches the model lag-named (`predictor_lag1`, ...), and `_FAR`'s
  factor-block selector excluded every `*_lag*` column, so the predictor block was empty
  and no factors were fit. On the GCLS replication this made the direct FM benchmark
  byte-identical to AR (AR/FM = 1.000 for every target/horizon), disagreeing with the
  paper where AR = 1.04–1.11 × FM. `_FAR` now excludes only the target's OWN lag columns
  from the factor block; the predictor lags remain and drive the PCA. Recursive/path
  `far` was already correct and is unchanged. The direct FM benchmark and every direct
  relative-RMSE must be re-scored after this fix.

- `pipeline`/`forecasting`: opt-in shared on-disk preprocessing cache. Set
  `pipeline_spec(..., preprocessing_cache_dir=...)` (or pass `preprocessing_store=`
  to `forecasting.run`) and each per-`(PreprocessSpec, target, origin)`
  `FittedPreprocessor` is computed once and reused across cells AND worker
  processes, instead of every parallel cell recomputing the dominant-cost EM/factor
  imputation. New `macroforecast/preprocessing/cache.py::PreprocessorStore`
  (content-addressed, atomic writes, lock-free reads). Default (dir unset) is
  byte-for-byte the prior behavior; a regression test pins serial==parallel and
  store-off==store-on at 1e-10. The cache key encodes `(spec, target, origin_pos)`
  only, so do not share one store directory across runs that differ in
  `preprocessing_policy.scope` for the same spec.
- `pipeline`: new `evaluate_cross_policy(forecasts, benchmark=, benchmark_policy=)`
  helper scores every `(arm, forecast_policy)` contender against one benchmark fixed
  to a single policy (the common-denominator convention, e.g. the GCLS direct FM as
  the denominator for both the direct and the path-average tables). It qualifies the
  contender names by policy and returns a tidy table with `arm` and `forecast_policy`
  as their own columns, replacing the previous hand-built `SimpleNamespace` recipe.
  `accuracy_table` is refactored over an internal `_accuracy_against(master, bench)`;
  behaviour is unchanged.

- `forecasting` (performance, numerically transparent): the per-origin spec-level
  preprocessing TRANSFORM (the dominant-cost EM imputation + factor projection) is now
  reused across horizons. `_prepare_origin_panel` previously keyed its prepared-panel
  cache on `(origin_pos, horizon)` (the appended target row is horizon-dependent), so a
  multi-horizon `run()` re-ran the whole-window transform once per horizon even though
  the rows observable at the origin (`<= origin_pos`) transform identically for every
  horizon. The transform is now split: the horizon-independent base panel (rows
  `<= origin_pos`) is transformed once and cached under an origin-keyed
  `("prepared_base", origin_pos)` key, and only the tiny forward/target rows
  (`> origin_pos`) are transformed per horizon. On the GCLS grid (≈456 origins × 6
  horizons × 2 arms per target) this removes the ~6× per-horizon transform redundancy.
  The split is only taken on the shared-cache (serial spec-level) path; the parallel
  backend keeps the un-split whole-window transform, and a serial==parallel golden pins
  byte-identical forecasts. Cross-arm reuse (per-`(origin, horizon)` prepared stage) is
  unchanged.

- `tests.py`/`giacomini_white_test` (bugfix, WP-A1 following on WP-V3 finding
  2): **p-values for `horizon > 1` CHANGE under the new default**
  (`small_sample=True`) -- see the exact before/after numbers below; pass
  `small_sample=False` to reproduce the OLD p-values byte-for-byte. WP-V3's
  Monte Carlo size validation found the test genuinely oversized at h=4
  (.124 at alpha=.05/n=50; still .072-.092 out to n=5000 -- confirmed NOT a
  small-n artifact). WP-A1 root-caused it: the inline Bartlett-tapered
  Newey-West HAC (`weight = 1 - lag/h`) applied to lags 1..h-1 systematically
  discards a large, NON-VANISHING fraction of the *known* (finite-order,
  exactly h-1-dependent) autocovariance of an h-step loss differential --
  confirmed two ways: analytically (the taper weights predict
  `E[Omega_11]=11.0` vs the true `16.0` at h=4, exactly matching the measured
  downward bias) and empirically (plugging the TRUE population `Omega` into
  the same Wald construction gives `mean(stat)=1.995~=2.0`,
  `rate@alpha=.05=.050` -- i.e. the test's instrument choice and construction
  are otherwise correct; ALL of the size distortion traces to the `Omega`
  estimator). `dm_test` never had this problem because it already uses an
  UNTAPERED (`kernel="acf"`) sum over the same lags (matching R
  `forecast::dm.test`) -- correct here because the true ACF is exactly zero
  beyond h-1, so Bartlett tapering (whose purpose is guaranteeing PSD for
  GENERAL, not-known-finite-order processes) only discards real signal for
  no benefit. Fix (default `small_sample=True`): (a) sum UNTAPERED sample
  autocovariances over lags 0..h-1, matching `dm_test`'s own convention;
  (b) fall back to a smaller bandwidth (down to 0) if that untapered sum is
  not positive semi-definite -- untapered sums lose Newey-West's automatic
  PSD guarantee, so this mirrors `_long_run_variance`'s own existing
  non-positive-variance fallback; (c) reference the Wald statistic against
  `F(q, ESS-q)` (Hotelling-style, statistic scaled by `q`) instead of
  chi2(q) whenever a HAC lag was actually used, with
  `ESS = n/(1+2*bandwidth_used)` the standard effective-sample-size
  correction for serially dependent data -- this mops up the residual (much
  smaller than the taper bias, but still real at small n) finite-sample
  over-rejection of a Wald test built on an estimated multi-dimensional
  covariance. At horizon=1 (bandwidth=0, already well-calibrated) this
  reduces EXACTLY to the old chi2(q) reference -- verified by MC to
  introduce no regression. `small_sample=False` restores the exact pre-fix
  Bartlett-HAC + chi2(q) behavior. MC before/after
  (`tests/mc/test_giacomini_white_size.py`, identical seeds, R=5000): h=4
  n=50 alpha=.05 `.1210 -> .0450` (CI99 [.038,.053]), alpha=.10
  `.2062 -> .1030`; h=4 n=200 alpha=.05 `.0886 -> .0480`, alpha=.10
  `.1650 -> .1022`; h=1 unchanged at both n (still in band, e.g. n=50
  alpha=.05 `.0492`, n=200 alpha=.05 `.0456`). All four MC cells now PASS;
  the `xfail(strict=True)` tripwires at h=4 (n=50/200) are removed (no
  xfails remain for `giacomini_white_test`). New deterministic unit tests
  (`tests/correctness/test_add_gw_cpa.py`) pin the corrected h=4
  statistic/p-value on a fixed fixture and confirm `small_sample=False`
  reproduces the exact old formula bit-for-bit. `docs/reference/tests.md`'s
  `giacomini_white_test` entry is updated with the new default and a
  summary of the diagnosis.

- `tests/mc/test_mcs_coverage.py` (WP-A1 Step 0, following on WP-V3 finding
  3, **no code change**): the equal-losers global-null xfail
  (`P(all 5 tied models retained)=.818` vs a naive `>=.90` floor reading)
  was cross-checked against R's own `MCS::MCSprocedure` on the IDENTICAL
  design (subprocess-Rscript bridge, same pattern as `tests/parity/`,
  R=200 replications, `n_boot=500`,
  `MCSprocedure(alpha=.10, B=500, statistic="Tmax", k=5, min.k=3)`): R
  reproduces the SAME ~.82 coverage (`rate=.8200`,
  `CI99=[.7401,.8841]`), matching this package's own longer-run measurement
  (`rate=.8180`, `CI99=[.7846,.8483]`, n_reps=1000, WP-V3). Since the
  reference R implementation shows the identical under-coverage on the
  identical design, this is a genuine property of the Hansen-Lunde-Nason
  MCS sequential-elimination procedure itself under an exact global null
  with many tied models -- not a `macroforecast` defect -- so no code
  change was made. Converted the `xfail(strict=True)` into a
  documented-behavior regression band (`[0.70, 0.90)`, i.e. re-investigate
  if the rate ever drifts toward the naive `>=.90` floor -- an unexpected
  upstream fix -- or drops well below `.70` -- a new regression) rather
  than a `>=1-alpha` floor assertion; added an interpretation note to the
  `model_confidence_set` entry in `docs/reference/tests.md` (what `alpha`
  means under a global null with many ties: the `>=1-alpha` inclusion
  guarantee is confirmed for a genuinely-best model, but does not read
  literally as `P(all K retained)>=1-alpha` when all `K` are exactly tied).
  Design A (single dominant model, `>=1-alpha` floor, `.947`) is unaffected
  and unchanged.

- `pipeline/evaluate.py`, `pipeline/spec.py`, `forecasting/checkpoint.py`,
  `metrics.py` (feature, Phase 1 density pipeline): density/interval
  forecasting is now wired end to end through the MANAGED pipeline, not just
  the standalone `forecasting.run()`/`ForecastResult.evaluate()` path. Reality
  check first: `variance_prediction`/`quantile_predictions` emission onto the
  forecast table already existed (`forecasting/policies/base.py::
  _variance_series`/`_quantile_frame`, called from the direct policy) and
  `macroforecast.metrics` already had a full registry-driven table-level
  evaluator for them (`evaluate_forecasts`, with `crps`/`gaussian_nll`/
  `log_score`/`qlike`/`pinball_loss`/`coverage_rate`/`interval_width`/
  `interval_score` already in its metric registry) and `macroforecast.tests`
  already had the calibration diagnostics (`density_interval_tests`,
  `pit_autocorrelation_test`, `interval_coverage_test`). The actual gaps were
  narrower than assumed: (1) `pipeline/evaluate.py::evaluate()` never called
  any of it -- only `forecasts`/`accuracy`/`significance`/`mcs`; (2)
  `forecasting/checkpoint.py`'s lean schema was point-only, so a checkpointed/
  resumed run silently dropped `variance_prediction`/`quantile_predictions` for
  any origin recovered from disk; (3) `pipeline/rescore.py` reconstructs its
  ENTIRE master frame from the checkpoint, so a rescored report was ALWAYS
  point-only regardless of what the live run emitted.
  - New `metrics.metric_kind(name)` (registry plumbing, no new metric math):
    classifies a registered metric name as `"variance"`/`"volatility"`/
    `"quantile"`/`"relative"`/`"direction"`/`"point"` by the forecast-table
    column(s) its table-level evaluation needs. `metrics.DENSITY_METRIC_NAMES`
    is the union of the first three.
  - New `pipeline/evaluate.py::density_table(master, spec)` and
    `calibration_table(master, spec)`, both EvalSpec-gated (density metrics via
    `EvalSpec.metrics`, calibration tests via new `EvalSpec.tests` names
    `"berkowitz"`/`"pit_autocorr"`/`"coverage"`, added to `SUPPORTED_EVAL_TESTS`
    and new `pipeline/spec.py::CALIBRATION_EVAL_TESTS`) and populated into two
    new `evaluate()`/`PipelineReport` keys, `density`/`calibration`. Both are
    thin wrappers around the ALREADY-EXISTING `evaluate_forecasts`/
    `density_interval_tests`/`interval_coverage_test` machinery -- not
    reimplementations -- so a requested density metric with no matching column
    raises the same actionable `ValueError` those functions already raise.
    New `EvalSpec.calibration_alpha: float = 0.05` (independent of
    `mcs_alpha`). Defaults are unchanged: a default `EvalSpec` requests no
    density metric/calibration test, so `density`/`calibration` are empty
    frames and no forecast-frame column is even inspected -- pinned by two new
    golden fixtures (`tests/pipeline/_golden/density_defaults_*.parquet`,
    generated from the base commit) alongside the pre-existing
    `evalspec_defaults_*` golden (point-only case).
  - `forecasting/checkpoint.py::LEAN_FORECAST_COLUMNS` gains a fixed
    `variance_prediction` column (a plain float, `None` when a model does not
    emit one -- same convention as the rich table). Quantile predictions are a
    `{level: value}` mapping, not a scalar, so they are expanded into wide,
    per-origin `q_<pct>` columns instead (e.g. `q_05`/`q_50`/`q_95` for levels
    `0.05`/`0.5`/`0.95`) -- **a deliberate design deviation from a
    spec-declared quantile grid**: the level set is a per-model hyperparameter
    already fixed for the run, so it is derived empirically per origin rather
    than added as new pipeline-spec configuration surface.
    `load_checkpoint_frame` unions differing origin-file schemas via
    `pd.concat` (old checkpoints written before this column existed load fine
    and rescore as point-only) and reconstructs a `quantile_predictions`
    mapping column from the wide columns, so `rescore()` and a resumed
    `run()`'s in-memory merge (`forecasting/runner.py::
    _merge_checkpoint_records`) both end up with the SAME dict-based
    representation a live run's forecast table carries -- one quantile
    dispatch path regardless of source.
  - `pipeline/run.py::run_pipeline` and `pipeline/rescore.py::rescore` now
    thread `density`/`calibration` into the returned `PipelineReport` (new
    optional fields, appended at the end of the dataclass, default `None` for
    any hand-built report that does not pass them) -- outside this lane's
    original file scope (`pipeline/run.py` was not listed), but necessary: a
    report built through the standard `run_pipeline()` entry point would
    otherwise never expose either field despite `evaluate()` computing them.
  - Phase 1 scope, explicitly not addressed here: no synthetic residual-based
    variance fallback for models that do not natively emit one (columns stay
    absent/NaN, degrading gracefully); emission itself is unchanged and still
    direct/direct_average-policy only (recursive/path_average/panel/
    combinations emit explicit `None`, as before). See
    `docs/reference/pipeline.md` ("Density and interval forecasting").

## [0.9.5] -- 2026-06-27 -- "Replication robustness, Python 3.10 compatibility, type-clean"

**Correctness and compatibility:**

- `pipeline`: relative-RMSE / OOS-R2 are scored on each contender's PAIRWISE common
  sample with the benchmark, and `n_common` is per-contender. The previous single
  all-contender listwise sample let one short-coverage arm silently truncate every
  arm's evaluation window; ragged coverage now emits a `RuntimeWarning`. The joint
  sample is kept only for the Model Confidence Set.
- `forecasting`: information-criterion models (`ar`, `far`) select their AR order by the
  same BIC/AIC branch under both the direct and path-average policies, so horizon-1
  direct and path forecasts are identical by construction again.
- Python 3.10: fixed three 3.11+ APIs that broke on 3.10 -- `datetime.UTC` in
  `output/core.py` (now `timezone.utc`), `BaseException.add_note` in `data/loaders.py`
  (guarded), and `ModelFit.__getattr__` delegating dunder lookups to the estimator,
  which corrupted pickling of saved models on 3.10.
- `pipeline`: longest-processing-time-first cell scheduling (heaviest cells dispatched
  first) removes the tail bottleneck; bit-exact and result-invariant.
- typing: the package is mypy-clean (0 errors) and ships `py.typed`.
- docs: the GCLS (2021) replication is a single honest page (verification summary,
  eight-step build, and the Appendix B ground-truth tables); the user guide is a flat
  concept hub.

**Phase 3g-bis cascade complete**: 10 PRs (#360–#367 + #K). Legacy `core/layers/lN.py` and
`core/ops/lN_ops.py` fully removed. Circular import workaround eliminated.
`interpretation/` and `recipes/` converted to backward-compatible shims.

**Replication-driven robustness pass** (defects surfaced by the GCLS 2021
"transformations matter" and ML-Useful 2022 pseudo-out-of-sample replications, now
guarded by regression tests):

- `forecasting`: incremental per-origin checkpointing -- `run(checkpoint_path=...)`
  and `PipelineSpec.checkpoint_dir` write a lean forecast record per origin and resume
  after a crash. Single-horizon checkpoints are namespaced by horizon to prevent a
  cross-horizon collision that silently returned horizon-1 forecasts for every horizon.
- `forecasting`: multi-horizon runs share the per-origin preprocessing fit across
  horizons (keyed on origin position), and the `from_cutoffs(horizon=h)` validation
  embargo is re-derived per horizon -- a stale zero embargo could leak the h-step label
  into validation folds for `h > 1`.
- `forecasting`: the per-origin transformed panel no longer carries the full
  `macroforecast_metadata` on `.attrs`; it is held on the prepared-stage dataclass and
  re-attached only where consumed. This removes an `O(origins x arms x horizons)` cost
  of pandas deep-copying a large attrs dict on every operation, which made multi-horizon
  runs appear to hang.
- `forecasting`: model selection degrades gracefully when an early retrain origin cannot
  form a target-availability-safe validation split (reuses the last-tuned or default
  params, warns, tags `selection_degraded`) instead of aborting the whole run; it still
  raises when no origin can ever tune.
- `preprocessing`: `em_factor` imputation and PCA drop all-missing rows and columns, so
  ragged-start FRED-MD panels impute the dense sub-block instead of raising.
- `feature_engineering`: MARX fit drops predictor columns with no observation over the
  fit window (mirroring the PCA pattern), so an all-NaN series no longer empties the
  whole feature matrix under leak-free preprocessing.
- `pipeline`: the accuracy table scores each contender on its pairwise sample with the
  benchmark, so one short-coverage arm no longer truncates the others (see 0.9.5 above;
  the earlier all-contender common sample is kept only for the Model Confidence Set);
  the leakage audit validates every horizon and surfaces a crashed
  validation; Clark-West is emitted only for arms declaring `nested_in_benchmark`
  (Diebold-Mariano otherwise); a benchmark missing from a `(target, horizon)` slice is
  surfaced via `benchmark_present`; interpretation aggregation no longer averages
  misaligned rows; and a zero-row arm is reported via a warning rather than dropped.

## [0.9.5a1] -- 2026-06-03 -- "Vintage loader automation"

- `mf.data.load_fred_md(vintage="YYYY-MM")` now resolves the official
  McCracken-Ng historical FRED-MD archive automatically, caches the zip,
  extracts the requested vintage CSV, and returns the same `DataBundle` contract
  with the pandas panel in `bundle.panel`.
- `mf.data.load_fred_qd(vintage="YYYY-MM")` now uses the same automatic
  historical-archive path and accepts `local_zip_source=` as an explicit local
  override.
- FRED-SD vintage behavior is documented and regression-tested as the
  by-series workbook path: direct `series-YYYY-MM.xlsx` first, then the official
  by-series zip fallback for the requested vintage.
- The transformation-paper replication skeleton no longer requires users to
  pre-download a FRED-MD historical zip; `load_fred_md(vintage="2018-01")`
  records the exact archive/member source in artifact metadata.

### Deprecations scheduled for removal in v0.10.0

The following deprecated aliases emit `DeprecationWarning` in all v0.9.x releases
and will raise `TypeError` in v0.10.0.

- `Experiment(model_family=)` keyword argument -- use `model=` instead
- `Experiment(model_families=)` keyword argument -- use `models=` instead
- `mf.forecast(model_family=)` keyword argument -- use `model=` instead
- `build_default_recipe_dict(model_family=, model_families=, benchmark_family=)` -- use `model=, models=, benchmark_model=`
- `macroforecast.models.ops.OPERATIONAL_MODEL_FAMILIES` -- use `OPERATIONAL_MODELS`
- `macroforecast.models.ops.FUTURE_MODEL_FAMILIES` -- use `FUTURE_MODELS`
- L6 result dict key `decision_at_5pct` -- use `decision`

Note: axis renames (`custom_source_policy` -> `panel_composition`, `forecast_strategy` ->
`forecast_policy`, `quarterly_to_monthly_rule` -> `quarterly_to_monthly_policy`,
`monthly_to_quarterly_rule` -> `monthly_to_quarterly_policy`) are HARD CHANGES with
no alias support at the recipe level. Users must update recipe YAML files manually.
See `docs/explanation/deprecation_timeline.md` for the full deprecation reference.

### Bug fixes

- **PR9 (MEDIUM): `bvar_minnesota_fit` — three missing Litterman hyperparameters exposed**

  `bvar_minnesota_fit` in `macroforecast/api/functions/timeseries.py` only
  accepted `n_lag` and `lambda1`, omitting three of the four standard Litterman
  (1986) hyperparameters. Users could not reproduce published BVAR results
  without access to `lambda_cross`, `lambda_decay`, and `b_AR`.

  The `_BayesianVAR` backend and the recipe path (`_build_l4_model`) already
  accepted all four hyperparameters; only the standalone API was missing them.

  Fix: expanded `bvar_minnesota_fit` signature with three new keyword-only
  parameters and their corresponding defaults (matching `_BayesianVAR`
  defaults for backward compatibility):

  | Parameter | Default | Description |
  |-----------|---------|-------------|
  | `lambda_cross` | `0.5` | Cross-equation shrinkage (λ₂). `0` = no cross-lag information; `1` = cross lags same scale as own lags. |
  | `lambda_decay` | `1.0` | Lag decay exponent (λ₃). Prior variance for lag l scales as (λ₁ / l^{λ_decay})². |
  | `b_AR` | `1.0` | Prior mean for first own lag. `1.0` = random-walk (Litterman I(1) default); `0.9` = VARCTIC calibration (recipe default). |

  `BVARMinnesotaFitResult` was extended with matching fields (`lambda_cross`,
  `lambda_decay`, `b_AR`), and `summary()` now displays all four
  hyperparameters. Input validation raises `ValueError` for `lambda_cross < 0`,
  `lambda_decay <= 0`, and `b_AR` outside [0, 2].

  Backward compatibility: all existing calls remain valid and produce
  identical output. The new parameters carry defaults that reproduce the
  prior behaviour exactly.

  References: Litterman (1986) JBES 4(1); Karlsson (2013) Handbook of
  Economic Forecasting Vol. 2B, Ch. 15.

- **PR10: README minimal recipe — observation count extended; CI doctest added**

  The minimal recipe in `README.md` defined only 6 observations (Jan-Jun 2018).
  After the `lag` operation in L3 (`n_lag=1`) the aligned dataset had 5 rows.
  With `min_train_size: 4` the walk-forward loop requires strictly more than 5
  observations, so `mf.run()` raised
  `RuntimeError: minimal L4 runtime requires min_train_size < aligned observation count`.

  Fix: extended `date`, `y`, and `x1` arrays to 12 monthly entries (Jan-Dec 2018).
  After lag, 11 aligned observations remain, giving 7 OOS forecast origins with
  `min_train_size: 4`.

  To prevent future drift, a CI workflow (`.github/workflows/ci-readme.yml`) and
  helper script (`tools/run_readme_recipe.py`) were added. The workflow triggers
  on any push or pull request that touches `README.md` or `macroforecast/`, extracts
  the minimal recipe block, runs `mf.run()`, and fails the build if the recipe errors.

- **PR11: goulet_coulombe_2021_replication.yaml — 6 obsolete axis/op names updated**

  `examples/recipes/goulet_coulombe_2021_replication.yaml` used six names that
  were renamed in earlier refactors and have no alias support at the recipe
  level. Attempting `mf.run()` on the recipe raised
  `RuntimeError: unknown L1 axis 'custom_source_policy'` before any model was
  fitted.

  Changes applied (all in the recipe YAML, no runtime code touched):

  | Location | Old name | Canonical name |
  |----------|----------|----------------|
  | `0_meta` `fixed_axes` | `reproducibility_mode` | `reproducibility_policy` |
  | `0_meta` `fixed_axes` | `compute_mode` | `compute_policy` |
  | `1_data` `fixed_axes` | `custom_source_policy` | `panel_composition` |
  | `4_forecasting_model` nodes (×2) | `op: fit_model` | `op: fit` |
  | `4_forecasting_model` params (×2) | `family:` | `model:` |
  | `4_forecasting_model` params (×2) | `forecast_strategy:` | `forecast_policy:` |

  The statistical content is preserved exactly: ridge baseline on FRED-MD
  INDPRO, h ∈ {1, 3, 6, 12}, AR(BIC) benchmark, DM (HLN-corrected) test.

  Smoke: `mf.run('examples/recipes/goulet_coulombe_2021_replication.yaml')`
  completes with `cells = 1`.

- **PR12: docs sync — standalone callable count corrected + L6 .decision type fixed**

  Two documentation errors corrected in `docs/reference/api/standalone_functions/`.

  **Sub-fix A (callable count):** `index.md` reported "Total: 118 standalone
  callables." The actual count from `dir(macroforecast.functions)` (v0.9.5a0)
  is 132 pure function callables plus 61 result dataclasses. The drift arose
  from v0.2-v0.3 honesty-pass promotions adding ~14 functions (MRF GTVP,
  ridge variants, MIDAS families, conditional PFI, bagging, DMP, etc.).
  The per-layer table was also corrected: L4 fit 38 → 52 (deep NN, MIDAS,
  ridge variants, misc, ridge families all counted now). L2/L3/L5/L6/L7
  counts were accurate and unchanged.

  A new script `tools/count_callables.py` introspects `macroforecast.functions`
  and writes a full inventory to `docs/_audit/standalone-callable-inventory-<date>.md`.
  Run `python3 tools/count_callables.py` from the repo root after any API change.

  **Sub-fix B (.decision type):** `l6_tests.md` documented `.decision` as
  `str: 'reject' or 'fail to reject'` (lines 3 and 124). The actual return
  type is `bool` (`True` = reject H0 at 5%, `False` = fail to reject). The
  `.summary()` method formats this as the human-readable string. No code
  changed; docs only.

- **PR7 (HIGH): L2 temporal dispatch audit — rolling_window_per_origin leak path closed**

  `_validate_imputation` in `macroforecast/layers/l2_preprocessing/schema.py`
  did not reject the combination `imputation_temporal_rule: rolling_window_per_origin`
  with `imputation_policy: linear_interpolation`. The option name implies per-origin
  safety but the runtime has no per-origin rolling-window implementation: the
  else-branch in `materialize_l2` (runtime.py ~line 420) calls full-sample
  `_apply_imputation`, which applies `interpolate(method="linear")` over the
  full panel. Even with the PR6 `limit_direction=forward` safeguard, the full
  panel is used as the data source — a silent lookahead leak.

  Fix: hard validator rejection added for `rolling_window_per_origin +
  linear_interpolation`. The error message points to the two safe alternatives:
  `forward_fill` (inherently causal) and `expanding_window_per_origin` with
  `linear_interpolation` (fully per-origin).

  Additionally, a SOFT warning (not hard error) is now emitted by `validate_layer`
  when the user explicitly sets a stateful imputation or outlier policy alongside
  `block_recompute`. `block_recompute` is a legitimate full-sample-at-block-boundary
  approach but its statistics may span post-origin observations. Only explicitly-set
  stateful policies trigger the warning; causal-safe defaults and combos like
  `forward_fill + block_recompute` do not.

  Full per-origin rolling-window dispatch is deferred to v0.4. The `rolling_window_per_origin`
  schema option is retained with implementation behaviour equivalent to `block_recompute`
  (full-sample) for all non-rejected combinations.

  Dispatch matrix audit: `docs/_audit/l2-dispatch-audit-2026-05-27.md`.

  Tests: `tests/layers/test_l2_temporal_dispatch_audit.py` (20 new tests across 5
  classes: `TestRollingWindowLinearInterpolationHardReject`,
  `TestRollingWindowSafeCombosPasses`, `TestBlockRecomputeSoftWarning`,
  `TestExpandingWindowPerOriginUnaffected`, `TestFullSampleOnceRegressionGuard`).

- **PR8 (HIGH): BVAR Minnesota σ² scaling fix — two-pass OLS pre-estimate added**

  `_fit_multivariate_minnesota` in `macroforecast/core/runtime.py` computed the
  posterior precision and rhs without dividing by the per-equation error variance
  σ²_i. The Litterman (1986) Minnesota prior already absorbs σ²_i into V_i (the
  own-lag diagonal scales as λ₁² · σ²_i / lag^{2λ_decay}), so Vinv ~ 1/σ²_i.
  The canonical formula (Karlsson 2013, Handbook of Economic Forecasting,
  Vol. 2B, Eq. 15.8-15.9) requires:

      Precision_i = V_i^{-1} + Z'Z / σ²_i
      RHS_i       = V_i^{-1} m_i + Z' y_i / σ²_i

  Without the /σ²_i divisor on the likelihood terms, the data were over-weighted
  relative to the prior by a factor of σ²_i. With macro data where σ²_i >> 1
  (e.g. INDPRO levels with σ² ~ 50,000), the prior was effectively invisible at
  any finite λ₁ — the posterior collapsed to the OLS estimate regardless of the
  prior tightness setting.

  Fix: a two-pass scheme is inserted before the per-equation posterior loop.
  Pass 1 runs OLS per equation via `np.linalg.lstsq` to obtain σ²_i estimates
  (plug-in Empirical Bayes). Pass 2 computes the Bayesian posterior using these
  σ²_i values to divide both ZtZ and Z'y_i. The denominator is clamped to 1e-12
  to prevent division by zero in degenerate cases.

  Sanity check: with λ₁=500 (loose prior), BVAR posterior converges to OLS.
  With λ₁=0.01 (tight prior) and b_AR=1.0, all own-lag-1 coefficients are
  pulled close to 1.0. Both hold after the fix; the bug caused the prior to be
  invisible at λ₁=1 with high-variance data.

  Tests: `tests/core/test_bvar_sigma2_scaling.py` (10 new tests across 4
  classes: `TestHighVarianceShrinkage`, `TestOLSConvergence`,
  `TestPriorDominance`, `TestMonotoneConvergence`,
  `TestPosteriorCovarianceScaling`).

- **PR5: Berkowitz LR_3 AR(1) component added — df=2 → df=3 (serial-dependence detection)**

  `_density_interval_battery` in `macroforecast/core/runtime.py` computed the
  Berkowitz (2001) likelihood-ratio statistic using only the mean and variance
  of the transformed series (df=2). The docstring stated "AR(1) on the
  transformed series under H0 of i.i.d. N(0,1)" but the AR(1) serial-dependence
  component (rho) was entirely absent. A density forecast with autocorrelated
  PIT values — e.g. from a misspecified GARCH — would pass the old df=2 test
  while exhibiting precisely the serial dependence Berkowitz (2001) was designed
  to detect.

  Fix: replaced the Berkowitz block with the full LR_3 test from Berkowitz
  (2001, eq. 6). Under the AR(1) alternative z_t = mu + rho*z_{t-1} + eps_t,
  eps_t ~ N(0, sigma^2), the OLS MLE estimates (rho_hat, mu_hat, sigma^2_hat)
  are computed, the unrestricted log-likelihood conditions on the first
  observation, and LR_3 = -2[log L(H0) - log L(H1)] is evaluated against
  chi^2(3) (df=3). The result dict now includes `rho`, `mu`, `sigma`, and
  `df=3`. A fallback to LR_2 (df=2, rho=None) is retained for very small
  samples (n < 4).

- **PR6 (CRITICAL): linear_interpolation silent lookahead leak fixed — 3 locations in runtime.py**

  `pandas.DataFrame.interpolate(method="linear")` defaults to
  `limit_direction="both"` for numeric-index frames, meaning NaN cells at any
  position are filled using both past AND future non-NaN values. Three
  independent code paths in `macroforecast/core/runtime.py` used this
  interpolation without restricting the direction, silently leaking future
  observations into pre-cutoff imputed values in every study that used
  `imputation_policy: linear_interpolation`.

  **Location 1 — FRED-SD quarterly-to-monthly alignment**
  (`_apply_fred_sd_frequency_alignment`, ~line 15194):
  `series.interpolate(method="linear", limit_direction="both")` ran on the
  FULL panel before any per-origin split. A quarterly NaN for month 2 of a
  quarter was filled using the Q+1 value — a future observation unavailable
  at decision time. Fix: raise `ValueError` at runtime when
  `quarterly_to_monthly_policy="linear_interpolation"` is requested, directing
  the user to `step_forward` (causal) or `chow_lin` (regression-based)
  instead. The cleaning log records a `rejected_lookahead` step for the audit
  trail.

  **Location 2 — per-origin imputation** (`_apply_imputation_per_origin`,
  ~line 15494): `frame.interpolate(method="linear")` was called on the full
  frame including dates beyond `cutoff_ts`, causing NaN cells before the
  cutoff to be filled using post-cutoff values (e.g. a NaN at `t=2` with
  `cutoff_ts=3` would be interpolated using `t=4..N` values if the values
  inside the window were insufficient). Fix: interpolation is now restricted
  to `frame.loc[:cutoff_ts]` with `limit_direction="forward"`, so no
  pre-cutoff cell is ever filled by a post-cutoff value. When `cutoff_ts`
  is `None`, forward-only interpolation is applied to the full frame.

  **Location 3 — full-sample imputation** (`_apply_imputation`, ~line
  15593): same unconstrained `frame.interpolate(method="linear")` call.
  Fix: change to `limit_direction="forward"` and emit `UserWarning` so
  callers are aware that full-panel imputation is non-causal by design and
  that `imputation_temporal_rule=expanding_window_per_origin` is the
  preferred leak-free path.

  Leak evidence (pre-fix):
  ```
  # Per-origin: NaN at t=2, cutoff=3, post-cutoff values = [100, 200, 300, ...]
  # Pre-fix result at t=2: influenced by t>3 values (leak confirmed).
  # Post-fix result at t=2: == 2.0 (interpolated from t=1=1.0 and t=3=3.0 only).
  ```

  Tests: `tests/core/test_linear_interp_leak.py` (12 new tests across 3
  test classes: `TestPerOriginCutoffRespect`, `TestFredSdLinearInterpolationReject`,
  `TestFullSampleImputationWarning`).

- **PR4a (documentation): MCS docstring updated to accurately describe single-step max-test**

  `_mcs_from_per_origin_panel` in `macroforecast/core/runtime.py` previously
  claimed in its docstring to implement the Hansen-Lunde-Nason (2011) MCS
  iterative elimination procedure.  The actual implementation applies the
  T_max statistic once across all models and retains those below the bootstrap
  critical value — this is the single-step max-test, not full iterative MCS.

  The docstring now explicitly states: "NOT the full iterative
  Hansen-Lunde-Nason 2011 MCS elimination procedure."  An inline comment is
  also placed at the `mcs_set` computation for clarity.  The `stepm_rejected`
  output, which does use iterative Romano-Wolf StepM elimination, is
  unaffected.  Full iterative MCS is scheduled for Batch 2 PR13.

- **PR4b (critical): SPA benchmark guard — `spa_p_values` and `reality_check_p_values` return NaN with UserWarning when no benchmark specified**

  Hansen (2005) SPA requires a pre-specified null benchmark model.  The
  previous code used `argmin(means)` (the best-performing model in the sample)
  as an implicit benchmark, making the test circular and data-dependent.

  Fix: when `spa_benchmark_model` is absent from the sub-layer config dict,
  `_mcs_from_per_origin_panel` now emits:

      UserWarning: "SPA benchmark model not specified (spa_benchmark_model
      missing from sub_layers config). spa_p_values and reality_check_p_values
      will be NaN. Specify spa_benchmark_model (e.g. the AR benchmark model_id)
      for valid SPA / Reality Check p-values."

  and returns `float('nan')` for all `spa_p_values` and
  `reality_check_p_values` entries.

  When `spa_benchmark_model` is explicitly provided, the specified model index
  is used as the benchmark and a valid p-value in `[0, 1]` is returned.

  Note: `reality_check_p_values` continues to mirror `spa_p_values`; the
  White (2000) studentization that distinguishes Reality Check from SPA is a
  Batch 2 item (PR13), noted by a TODO comment at the assignment.

  Tests: `tests/core/test_mcs_spa.py` (4 new tests).

- **PR3 (critical): Clark-West one-sided p-value corrected for negative test statistic — 3 locations**

  The Clark-West (2006/2007) test is one-sided: H_a states that the large
  (unrestricted) model improves on the small (restricted) model.  The
  correct one-sided p-value is

      p_one = 1 - Phi(t_CW)

  which equals `p_two / 2` when `t_CW > 0` and `1 - p_two / 2` when
  `t_CW <= 0`.

  All three code paths that compute the CW one-sided p-value contained the
  same bug: when `stat <= 0` the code returned the two-sided `p_two`
  unchanged instead of `1 - p_two / 2`.  For a strongly negative stat
  (e.g. `t_CW = -10`), `p_two` is near zero, so the buggy code incorrectly
  rejected H_0 with near-certainty — exactly the opposite of the correct
  conclusion (strong evidence against H_a, fail to reject H_0).

  Example: `stat = -10.30`, `p_two = 7.3e-25`.
  - Pre-fix `pvalue = 7.3e-25` (false rejection of H_0).
  - Post-fix `pvalue = 1.0 - 3.6e-25 ≈ 1.000` (correct fail-to-reject).

  Fix applied at:
  1. `macroforecast/api/functions/tests.py` — `cw_test()` (~line 894).
  2. `macroforecast/api/functions/tests.py` — `enc_new_test()` (~line 991).
  3. `macroforecast/core/runtime.py` — recipe-engine CW path (~line 12788).

  The `enc_new_test` fix also covers Clark-McCracken (2001) encompassing
  test, which uses the same one-sided normal reference distribution.

  Tests: `tests/core/test_cw_one_sided.py` (6 new tests covering negative
  stat, strongly-negative stat near-1 assertion, enc_new symmetry property,
  positive stat regression, and None-input edge case).

- **PR2 (critical): Permutation importance deterministic reversal replaced with proper RNG — bit-exact replication restored**

  The L7 recipe-path helper `_permutation_importance_frame` used a deterministic
  reversal as its "permutation":

  ```python
  permuted[column] = list(reversed(permuted[column].tolist()))  # WRONG
  ```

  This made the `n_repeats` and `seed` / `random_state` parameters completely
  meaningless (every call returned the same result regardless of seed), and
  introduced systematic bias for time-ordered data (reversal is anti-correlated
  with the original ordering, not a random permutation).  The standard definition
  (Breiman 2001; Fisher, Rudin, Dominici 2019) requires an independent uniform
  random permutation per repeat, seeded by `random_state` for reproducibility.

  The buggy reversal also violated the bit-exact replication promise: the recipe
  path and the standalone `mf.functions.permutation_importance` API (which used a
  proper RNG when `random_state` was set) produced structurally different results
  for the same data and seed.

  Fix:
  - `_permutation_importance_frame` signature extended with `n_repeats: int = 1`
    and `seed: int = 0`.  The inner loop now calls
    `np.random.default_rng(seed).permutation(len(X_eval))` on each repeat.
    The returned DataFrame includes an `importances_` column with per-repeat
    values for variance estimation.
  - `_execute_l7_step` passes `n_repeats` and `seed` from the op `params` dict.
  - `mf.functions.permutation_importance` updated: `random_state=None` now uses
    `seed=0` (proper RNG, deterministic) instead of reversal, aligning the
    standalone API with the recipe path and restoring the replication promise.
  - `PermutationImportanceResult` gains an `importances_` field of shape
    `(n_features, n_repeats)` following the sklearn convention.

  **Impact**: any recipe or API call that used permutation importance will now
  receive statistically correct importance estimates.  Old numeric values are
  NOT preserved (they were wrong).  Seeds are now meaningful.

  Fix: `macroforecast/core/runtime.py` (~25 lines), `macroforecast/api/functions/importance.py` (~30 lines).
  Tests: `tests/core/test_permutation_importance.py` (9 new tests).

- **PR1 (critical): Kupiec (1995) POF likelihood-ratio formula corrected**

  The L6 density battery function `_density_interval_battery` computed the
  Kupiec proportion-of-failures (POF) likelihood-ratio statistic using a binary
  entropy difference formula instead of the correct log-likelihood ratio. The
  buggy expression was:

  ```
  LR = -2n [H(alpha) - H(p_hat)]   # WRONG: entropy difference
  ```

  which yields negative values whenever `p_hat > alpha` (violation rate exceeds
  nominal level), making the statistic structurally impossible as a chi-squared
  variate. The correct Kupiec (1995) formula is:

  ```
  LR_uc = 2 [x log(p_hat/alpha) + (n-x) log((1-p_hat)/(1-alpha))]  # CORRECT
  ```

  This is twice the log-LR of the unrestricted (p=p_hat) vs restricted (p=alpha)
  binomial model, guaranteed to be >= 0.

  **Impact**: `kupiec_pof.lr_statistic` was returning values like -27.15 (impossible)
  and `kupiec_pof.p_value` was returning 1.0 (from `chi2.cdf` of a negative argument),
  silently masking all VaR coverage violations. The fix produces the correct
  LR ~ 2.30, p ~ 0.13 for the canonical test case (n=200, x=15, alpha=0.05).

  Fix: `macroforecast/core/runtime.py`, lines 12933-12939 (4-line formula replacement).
  Tests: `tests/core/test_density_battery.py` (3 new tests: non-negativity, known-value
  numerical accuracy, and strong-rejection).

  Reference: Kupiec, P. (1995). "Techniques for Verifying the Accuracy of Risk
  Measurement Models." *Journal of Derivatives*, 3(2), 73-84.

### API (breaking -- deprecated, one-release window)

- **PR4 (hotfix): `model_family` → `model` rename with deprecation infrastructure**

  Public API parameters `model_family`, `model_families`, and `benchmark_family` across
  `macroforecast/api/` are renamed to `model`, `models`, and `benchmark_model` respectively.
  Old names are kept for one release as deprecated aliases that emit `DeprecationWarning`.
  All deprecated names will be removed in **v0.10.0**.

  Migration guide:

  | Old (deprecated) | New (canonical) |
  |-------------------|-----------------|
  | `mf.forecast(..., model_family="ridge")` | `mf.forecast(..., model="ridge")` |
  | `mf.Experiment(..., model_family="lasso")` | `mf.Experiment(..., model="lasso")` |
  | `build_default_recipe_dict(..., model_family=...)` | `build_default_recipe_dict(..., model=...)` |
  | `build_default_recipe_dict(..., model_families=...)` | `build_default_recipe_dict(..., models=...)` |
  | `build_default_recipe_dict(..., benchmark_family=...)` | `build_default_recipe_dict(..., benchmark_model=...)` |
  | `from macroforecast.models.ops import OPERATIONAL_MODEL_FAMILIES` | `import OPERATIONAL_MODELS` |
  | `from macroforecast.models.ops import FUTURE_MODEL_FAMILIES` | `import FUTURE_MODELS` |

  New module: `macroforecast.api._deprecations` centralizes all deprecation-shim logic
  (`resolve_model`, `resolve_models`, `resolve_benchmark_model`). Module-level `__getattr__`
  in `l4_models/ops.py` provides backward-compat for the constant renames.

### Fixes

- **PR2 (deep-audit CLI re-definition): fix broken `python -m macroforecast` entry point**

  `macroforecast/__main__.py` imported `macroforecast.scaffold.cli:main`, a module that
  does not exist (the `scaffold/` directory contains only `option_docs/`). Running
  `python -m macroforecast` raised `ModuleNotFoundError: No module named 'macroforecast.scaffold.cli'`.

  Fix: rewrite `__main__.py` to forward to `tools.docgen.cli:main`, the full CLI
  implementation that is already installed and provides subcommands `run`, `replicate`,
  `validate`, `scaffold`, and `encyclopedia`. Update `pyproject.toml` `[project.scripts]`
  entry from `tools.docgen.cli:main` to `macroforecast.__main__:main` so the installed
  console script resolves within the package namespace.

  New test file: `tests/cli/test_main_entry.py` (6 tests covering help, all four subcommands,
  and missing-recipe exit code).

### Docs

- **PR8 (deep-audit migration guide): comprehensive migration reference for v0.8.x / v0.9.0 upgraders**

  Created `docs/explanation/migration_guide.md` — the first dedicated migration guide in the
  repository. Covers all breaking changes introduced across the deep-audit PR series and prior
  version transitions.

  Contents:
  - v0.9.x → v0.10.0 upcoming removals (with before/after code snippets): `model_family=` →
    `model=`, `model_families=` → `models=`, `benchmark_family=` → `benchmark_model=`,
    `OPERATIONAL/FUTURE_MODEL_FAMILIES` → `OPERATIONAL/FUTURE_MODELS`, `decision_at_5pct` →
    `decision`.
  - v0.1 / v0.8.x → v0.9.x axis renames (hard changes, no alias): `custom_source_policy` →
    `panel_composition`, `forecast_strategy` → `forecast_policy`, `quarterly_to_monthly_rule` →
    `quarterly_to_monthly_policy`, `monthly_to_quarterly_rule` → `monthly_to_quarterly_policy`,
    `reproducibility_mode` → `reproducibility_policy`.
  - Module path migrations: `macroforecast.scaffold` → `tools.docgen`, `DEFAULT_FORECAST_POLICY`
    → `DEFAULT_FORECAST_STRATEGY`; shim-backed paths (`interpretation`, `recipes`).
  - Removed features table: wizard, navigator pages, audience-tree doc paths,
    `scripts/v01_smoke_check.py`, `l2_fred_sd_alignment.yaml`.
  - CLI entry point change: `macroforecast.__main__:main` now routes to `tools.docgen.cli:main`.
  - Recipe YAML migration checklist.
  - Links to `docs/explanation/deprecation_timeline.md` (PR6) for the full removal schedule.

  Navigation:
  - `docs/explanation/index.md` toctree updated to include `migration_guide`.
  - `README.md` "Upgrading?" notice added near version badge block, linking to migration guide.

- **PR7 (deep-audit sphinx + how_to executability): Sphinx -W zero warnings + how_to import audit**

  Problem 10 (Sphinx strict build) and Problem 11 (how_to import executability).

  Findings:
  - **41 imports audited** across `docs/how_to/`, `docs/tutorial/`, and `docs/getting_started.md`.
    40 PASS, 1 FAIL (fixed).
  - **Broken import fixed**: `DEFAULT_FORECAST_POLICY` → `DEFAULT_FORECAST_STRATEGY` in
    `docs/how_to/simple_api/quickstart.md`. `DEFAULT_FORECAST_POLICY` was never exported from
    `macroforecast.defaults`; the correct constant is `DEFAULT_FORECAST_STRATEGY`.
  - **Sphinx `-W` build**: already clean (0 warnings before and after fix). Exit code 0.
    CI gate at `.github/workflows/ci-docs.yml:58-60` already enforces `sphinx-build -W`
    on every PR — no new CI step needed.
  - **`docs-strict` Makefile target** added to `docs/Makefile` for local parity with the CI gate:
    `cd docs && make docs-strict` runs `sphinx-build -W -b html`.

  Deliverables:
  - `docs/how_to/simple_api/quickstart.md`: import corrected.
  - `docs/Makefile`: `docs-strict` target added.
  - `docs/_audit/howto-import-audit-2026-05-27.md` — full import scan table (41 entries).
  - `docs/_audit/sphinx-warning-audit-2026-05-27.md` — sphinx warning audit (0 warnings).

  Termination criteria:
  - TC11 (sphinx -W): PASS exit 0.
  - TC13 (all how_to imports executable): PASS (1 broken → 0 broken).

### Housekeeping

- **PR5 (deep-audit encyclopedia option drift): audit + drift gate test**

  Encyclopedia option drift audit (problem 8). Method: Python import of `AxisSpec` options
  from each `L{N}_LAYER_SPEC`, registry-based extraction of L3/L7 ops, `OPERATIONAL_MODELS |
  FUTURE_MODELS` for L4; compared against per-option sub-page directories under
  `docs/reference/encyclopedia/`.

  Findings:
  - **43 CODE_ONLY** items (valid code options without a dedicated encyclopedia sub-page),
    concentrated in L2 (10 options across 6 axes), L7 (26 ops), L4 (4 models), L3 (1 op),
    L5 (2 density metrics).
  - **7 DOCS_ONLY** items (L6 `equal_predictive_test` and `nested_test` sub-pages) — these
    are NOT true drift; L6 uses `options=()` in AxisSpec by design and validates accepted
    values at runtime. The docs pages are correct.
  - No deprecated axis names (`custom_source_policy`, `reproducibility_mode`, `fit_model`)
    found in any encyclopedia file.

  Deliverables:
  - `docs/_audit/encyclopedia-option-sync-2026-05-27.md` — full drift table.
  - `tests/tools/docgen/test_option_drift_gate.py` — drift gate test (10 parametrized cases,
    all pass). Asserts that drift does not grow beyond the 2026-05-27 baseline; any new
    undocumented options added to code will fail this test.

  All CODE_ONLY and DOCS_ONLY items are DEFERRED (follow-up PR).

- **PR6 (deep-audit problem 9): deprecation warning test coverage + timeline + axis alias check**

  Problem 9 (deprecation test coverage). Added test coverage for all three
  `macroforecast.api._deprecations` shims, the `OPERATIONAL/FUTURE_MODEL_FAMILIES`
  constant shims in `l4_models/ops.py`, and documented the deprecation timeline.

  Changes:
  - New test file: `tests/api/test_deprecations.py` (15 tests covering all deprecated
    parameter shims: `resolve_model`, `resolve_models`, `resolve_benchmark_model`,
    `OPERATIONAL_MODEL_FAMILIES`, `FUTURE_MODEL_FAMILIES` constant shims).
  - Migrated `tests/api/test_experiment.py`: replaced all `model_family=` fixtures with
    `model=` so the test suite is clean when run with `-W error::DeprecationWarning`.
  - Migrated `tests/integration/test_combined_dataset_smoke.py` line 92:
    `model_family="ridge"` -> `model="ridge"`.
  - New doc: `docs/explanation/deprecation_timeline.md` -- removal timeline for all v0.10.0
    removals plus hard-change axis renames.
  - Updated `docs/explanation/index.md` toctree to include `deprecation_timeline`.
  - Added "Deprecations scheduled for removal in v0.10.0" section to `[Unreleased]`.

  Axis alias finding: `custom_source_policy`, `forecast_strategy`,
  `quarterly_to_monthly_rule`, `monthly_to_quarterly_policy` are HARD CHANGES at the
  recipe YAML level. No alias helpers exist. Users receive an explicit `unknown axis`
  error at parse time. This is intentional -- documented in `deprecation_timeline.md`.

- **PR4 (deep-audit problem 7): rename `l2_fred_sd_alignment.yaml` → `l2_preprocessing_minimal.yaml`**

  The recipe file `examples/recipes/l2_fred_sd_alignment.yaml` used a `panel_composition: custom_panel_only`
  with a 12-row inline synthetic panel and demonstrated generic L2 preprocessing axes
  (`no_transform`, `none` outlier, `none_propagate` imputation, `keep_unbalanced` frame edge).
  It contained neither `sd_series_frequency_filter` nor `quarterly_to_monthly_policy`, so its
  filename and metadata description were misleading.

  Changes:
  - File renamed via `git mv` (history preserved).
  - File-level comment block updated to accurately describe the recipe contents.
  - `metadata.name`: `l2_fred_sd_alignment` → `l2_preprocessing_minimal`.
  - `metadata.description`: updated to describe generic L2 preprocessing axes; notes that
    FRED-SD alignment axes (`sd_series_frequency_filter`, `quarterly_to_monthly_policy`) are
    deferred to PR8+ (tracking: a dedicated FRED-SD demo recipe with real alignment axes is
    needed; see `docs/_audit/` for the outstanding gap).
  - `examples/recipes/README.md`: entry in the "In-progress" list updated to new filename.

  **TODO (PR8+):** Add a real `l2_fred_sd_alignment.yaml` demonstrating
  `sd_series_frequency_filter` and `quarterly_to_monthly_policy` with a synthetic
  mixed-frequency panel (monthly + quarterly series). The current recipe does not cover
  these axes; the filename previously implied it did.

- **PR3 (deep-audit tools/scripts audit): fix stale YAML keys in `audit_docs_vs_code.py`, stale
  module paths in `gen_standalone_docs.py`, archive `tools/research/`, delete obsolete
  `scripts/v01_smoke_check.py`**

  Problem 5 (tools/ scripts audit):

  - `tools/audit_docs_vs_code.py`: updated `RE_YAML_RECIPE_KEY` regex and `YAML_KEY_TO_LAYER`
    dict to use current L0 axis names `reproducibility_policy` (was `reproducibility_mode`) and
    `compute_policy` (was `compute_mode`). The scanner now correctly detects these keys in docs.

  - `tools/gen_standalone_docs.py`: updated `LAYER_MAP` module paths from
    `macroforecast.functions.*` to `macroforecast.api.functions.*` to match the current package
    layout. Previously all 132 callables emitted `WARN: unmapped module` because the old paths
    did not resolve. Added entries for `ridge_variants` and `midas` modules introduced since the
    original audit.

  - `tools/gen_encyclopedia_docs.py`: no changes needed. `fit_model` references are intentional
    (exclusion list comment; description of internal dispatch op).

  - `tools/generate_fred_dataset_docs.py`: no changes needed. No stale imports found.

  - `tools/research/README.md` (new): archive annotation per Decision C. Marks the directory
    as a research-phase audit trail, not part of build/CI/installed package.

  Problem 6 (`scripts/v01_smoke_check.py`): **deleted**.

  The script used four stale recipe keys (`custom_source_policy`, `reproducibility_mode`,
  `op: fit_model`, `params: {model_family: ridge}`) that fail schema validation under the
  current API. The choropleth render also produces a PDF file — not safe for CI. The 1345-test
  suite fully covers end-to-end execution; the v0.1 smoke check is obsolete. The `scripts/`
  directory is removed (it was the only file). No CI workflow referenced the script.

### Docs

- **PR1 (deep-audit Section A cleanup): fix dead scaffold path, stale `model_family=` example, and internal CI comment labels**

  Three fixes bundled from the Section A deep-audit pass:

  1. `.github/RELEASE_CHECKLIST.md` — replaced the dead `python -m macroforecast.scaffold encyclopedia docs/encyclopedia/` command (module does not exist) with the current `python -m tools.docgen encyclopedia docs/reference/encyclopedia/` command and updated the `git add` path to match.

  2. `examples/custom_fred_sd_mixed_frequency_model.py` — replaced deprecated `model_family=` keyword argument with canonical `model=` in the `mf.Experiment(...)` call. Eliminates `DeprecationWarning` for users running the example.

  3. `.github/workflows/ci-core.yml` — rewrote the four-line comment block explaining the `api/` exclusion in the stale-audience-tree check to remove the internal cycle label `C52` and the retired `navigator pages` term. The technical meaning of the comment is preserved.

- **PR7a (recipe cleanup): move 26 partial-layer recipes from `examples/recipes/` to `docs/recipe-snippets/`**

  Per PR6 audit (`docs/_audit/recipe-sweep-2026-05-26.md`): 26 recipes fail `mf.run()` with
  `single_target requires leaf_config.target string` because they are illustrative partial-layer
  snippets, not end-to-end runnable recipes. They are now in `docs/recipe-snippets/` (Diátaxis
  how-to / reference quadrant), identified as syntax demos rather than executable examples.

  `examples/recipes/` now holds only the 6 PASS recipes (directly runnable) and 5 Pattern-2
  recipes (stale `fixed_axes` L3 syntax — PR7b scope).

  `docs/recipe-snippets/README.md` added. `examples/recipes/README.md` updated to clarify
  the runnable-only boundary and cross-link to `docs/recipe-snippets/`. Docs cross-references
  in `docs/how_to/partial_layer_execution.md` and `docs/tutorial/replications/goulet_coulombe_2021.md`
  updated to new paths.

- **PR7b (recipe cleanup): migrate 5 L1/L2 recipes from stale `fixed_axes` L3 syntax to nodes/sinks DSL**

  Per PR6 audit (Pattern 2): five recipes in `examples/recipes/` contained a
  `3_feature_engineering:` block using the legacy `fixed_axes` sugar, which the current L3
  schema rejects (`L3 uses a step graph (nodes/sinks); fixed_axes sugar is not supported`).

  Each recipe was updated to include a complete, end-to-end runnable recipe skeleton:
  a `0_meta` block with deterministic seed, a `1_data` block with a 12-row
  `custom_panel_inline` synthetic dataset (preserving all original L1 axes such as
  `regime_definition` and `regime_estimation_temporal_rule`), a safe `2_preprocessing`
  block compatible with `custom_panel_only`, and minimal lag-1 ridge L3+L4 blocks using
  the current nodes/sinks DAG pattern.

  | Recipe | L1 axis of interest |
  |--------|---------------------|
  | `l1_minimal.yaml` | `panel_composition: custom_panel_only` baseline |
  | `l1_with_regime.yaml` | `regime_definition: external_nber` |
  | `l1_estimated_markov_switching.yaml` | `regime_definition: estimated_markov_switching` + `expanding_window_per_origin` |
  | `l2_minimal.yaml` | minimal `2_preprocessing` with explicit safe defaults |
  | `l2_preprocessing_minimal.yaml` (renamed from `l2_fred_sd_alignment.yaml` in PR4-deep-audit) | generic `2_preprocessing` axes: `no_transform`, `none` outlier, `none_propagate` imputation, `keep_unbalanced` frame edge |

  Post-migration sweep: 11 PASS / 0 FAIL across all active `examples/recipes/` recipes.
  Sweep audit committed to `docs/_audit/recipe-sweep-2026-05-27-post-pr7.md`.

- **PR6 (docs precision audit): committed full example-recipe sweep audit table (38 recipes, 4-category classification) to `docs/_audit/recipe-sweep-2026-05-26.md`; no recipe edits (check8 audit — PR7+ scope for repairs)**

  Sweep result: 6 PASS, 32 FAIL_SCHEMA, 0 FAIL_RUNTIME, 0 NEGATIVE_EXAMPLE.

  Three systematic failure patterns identified:

  | Pattern | Count | Root cause |
  |---------|-------|-----------|
  | `single_target requires leaf_config.target string` | 26 | Partial-layer docs recipes lack a runnable `1_data` + `target` block |
  | `L3 uses step graph; fixed_axes sugar is not supported` | 5 | Stale `fixed_axes` syntax on L3 sections that migrated to nodes/sinks DAG |
  | `unknown L1 axis 'custom_source_policy'` | 1 | Renamed axis: `custom_source_policy` → `panel_composition` |

  Recipe repairs are PR7+ scope.

- **PR3 (docs precision audit): purge internal cycle codes from user-facing docs and module docstrings**

  120+ occurrences of internal development cycle codes (`Cycle N`, `C<N>`) were removed
  from all user-facing surfaces: module docstrings (visible via `help()`), encyclopedia
  pages, how-to guides, reference tables, and `tools/` build scripts. Each reference was
  replaced with its public version equivalent (`v0.8.x`, `v0.9.3`, or `v0.9.5`) so that
  the `help()` surface and rendered docs present stable, meaningful version anchors rather
  than internal sprint identifiers.

  Mapping applied:

  | Cycle range | Version label |
  |-------------|---------------|
  | C12–C22 | `v0.8.x` |
  | C26–C29 | `v0.8.x` |
  | C35, C37–C38 | `v0.9.x` |
  | C41 | `v0.9.2b1` |
  | C45, C47–C50 | `v0.9.3` |
  | C57, C59 | `v0.9.4` |
  | C63–C64 | `v0.9.5` |

  Affected files (70 total): `macroforecast/core/runtime.py`,
  `macroforecast/layers/l1_data/option_docs.py`,
  `macroforecast/layers/l4_models/option_docs.py`,
  `tools/docgen/option_docs/{l5,l6,l7_a,types}.py`,
  `tools/docgen/{render_encyclopedia,introspect,cli}.py`,
  `tools/{gen_standalone_docs,gen_encyclopedia_docs}.py`,
  and 56 additional source and documentation files.

  Test scope (`tests/`) was not modified (separate maintenance track).

- **PR1 (docs precision audit): fix broken imports in three how-to guides**

  Three how-to pages referenced submodule paths that no longer exist following the
  Phase 3g-bis restructure. Copy-pasting the code blocks produced `ModuleNotFoundError`
  for any user.

  | File | Old import | New import |
  |------|-----------|-----------|
  | `docs/how_to/feature_selection_boruta.md` | `macroforecast.feature_selection` | `macroforecast.features.selection` |
  | `docs/how_to/advanced_recipes.md` | `macroforecast.feature_selection` | `macroforecast.features.selection` |
  | `docs/how_to/chow_lin_disaggregation.md` | `macroforecast.transforms` | `macroforecast.features.transforms` |

  Prose module references in the same pages updated to match. The
  `macroforecast.interpretation.GIRF` shim in `irf_pesaran_shin_girf.md`
  is operational and left unchanged.

  Smoke-tested: all three fixed imports import cleanly under the current package layout.

- **PR4 (docs site cleanup): unify layer numbering to compact L0–L8 / L1.5–L4.5 form**

  Body text and cross-reference navigation links throughout the docs site used
  the verbose "Layer N: ..." form inconsistently alongside the compact "LN" form.
  All body text and nav links now use the compact LN label. Page H1 titles
  (`# Layer N: ...` in explanation pages) and encyclopedia headings
  (`# Layer LN -- ...`) are intentionally kept in their canonical verbose forms.

  | Scope | Files changed |
  |-------|--------------|
  | Explanation architecture pages (nav links, body refs) | `layer0.md`–`layer8.md`, five `layer1_*.md` sub-pages, `index.md` |
  | 12-layer design overview | `12_layer_design.md` |
  | Philosophy page | `philosophy.md` |
  | How-to conventions | `how_to/conventions.md` |
  | Recipe schema references | `reference/recipe_schema/data.md`, `data_policies.md`, `output.md` |
  | Tutorial | `tutorial/04_custom_preprocessor.md` |

  Termination condition: every remaining `Layer [0-9]+:` / `Layer L[0-9]+` hit
  outside `tutorial/replications/` is a line-1 H1 title. PASS (18 hits, all H1).

- **PR4 (docs precision audit): replace TAXONOMY-deprecated 'family' L4 model references**

  Replaced ~20 occurrences of TAXONOMY-deprecated vocabulary across 13 encyclopedia and
  explanation files. The TAXONOMY bans `family` as an identifier for L4 model classes
  (e.g., "tree-family L4 model") but preserves legitimate statistics English uses
  (SHAP family, ETS family, design-doc quoted string "pick family, tune, repeat.").

  Rewrite map applied:

  | Deprecated pattern | Replacement |
  |--------------------|-------------|
  | `linear-family L4 model(s)` | `linear L4 model(s)` |
  | `linear L4 family` | `linear L4 model` |
  | `Linear-family attribution` | `Linear-model attribution` |
  | `lstm model family` | `lstm model` |
  | `transformer family` | `transformer model` |
  | `VAR family` | `VAR model` |
  | `every L4 family` | `every L4 model` |
  | `unified across families` | `unified across model classes` |
  | `family-specific op` | `model-specific op` |
  | `macroeconomic_random_forest L4 family` | `macroeconomic_random_forest L4 model` |
  | `base family` | `base model` |
  | `first-class L4 family` | `first-class L4 model` |
  | `tree-family L4 model` | `tree-based L4 model` |
  | `tree-family models` | `tree-based models` |
  | `Sklearn-family estimators` | `Sklearn estimators` |
  | `model-family sweep` | `model sweep` |
  | `ONNX … supported by the family` | `supported by the model` |
  | `PMML-compatible families only` | `PMML-compatible models only` |
  | `directed acyclic graph` (L3 topology) | `step graph` |

  Preserved unchanged: all design-doc quoted strings containing "pick family, tune,
  repeat."; SHAP family; SHAP-family ops; ETS family; pytorch families (framework
  enumeration); base_family YAML key; dependence-plot family; wavelet family name;
  family-wise (statistics); Albacore-family (model branding).

  Termination: `grep -rnE "(tree|linear|sklearn|VAR|lstm)[\- ]family" docs/` → 0 hits.
  `grep -rnE "directed acyclic graph" docs/` → 0 hits. (check4 audit, strict-30 cleanup)

- **PR2 (docs precision audit): fix YAML recipe block violations in 4 docs pages**

  Fixed 6 hits across 4 files identified by the check2 audit: deprecated `op: ridge` shorthand
  and `config:` key (replaced with canonical `op: fit` + `params:`), deprecated `model_family:`
  param key in L7 example (replaced with `model:`), stale `_rule` axis URL paths in L2 API
  reference (replaced with `_policy`), and stale `model_family: "ar"` prohibition in defaults
  reference (updated to current `model: "ar"` concern with `ar_p` guidance).

  | File | Lines | Change |
  |------|-------|--------|
  | `docs/explanation/recipe_to_run.md` | 89-91 | `op: ridge` + `config:` → `op: fit` + `params:` |
  | `docs/explanation/architecture/layer7.md` | 53 | `model_family: xgboost` → `model: xgboost` |
  | `docs/reference/api/standalone_functions/l2_clean.md` | 86, 100 | `_rule` → `_policy` in encyclopedia links |
  | `docs/reference/recipe_schema/defaults.md` | 16 | Replace stale `model_family: "ar"` with active `model: "ar"` / `ar_p` guidance |

- **PR2 (docs site cleanup): move architecture pages to explanation; encyclopedia is single lookup source**

  Architecture and design-narrative pages lived under `docs/reference/architecture/` but belong in `docs/explanation/` (Diátaxis: reference = look-up, explanation = narrative). Moved all pages; encyclopedia cross-links updated to point to the new location.

  | Action | Files |
  |--------|-------|
  | Moved layer pages (L0–L8 + sub-pages) | `docs/reference/architecture/layer{0-8}/*.md` → `docs/explanation/architecture/layer{N}.md` |
  | Moved top-level pages (7 files) | `foundation`, `philosophy`, `layer_boundary_contract`, `recipe_layers`, `artifacts_and_manifest`, `reproducibility`, `terminology` |
  | Created | `docs/explanation/architecture/index.md` (toctree for all moved pages) |
  | Removed | `docs/reference/architecture/` (directory deleted entirely) |
  | Updated toctrees | `docs/explanation/index.md` (add `architecture/index`), `docs/reference/index.md` (drop architecture entry, keep prose cross-link) |
  | Updated cross-refs (5 files) | `12_layer_design.md`, `bit_exact_replicate.md`, `honesty_pass.md`, `recipe_to_run.md`, `contributing.md`, `encyclopedia/index.md` |

  Termination conditions: `docs/reference/architecture/` does not exist. `docs/explanation/architecture/` exists. Sphinx build clean (no undefined labels, no broken references).

- **PR1 (docs site cleanup): delete navigator pages and cross-links**

  Navigator was removed from the public feature set in Phase 0 (TAXONOMY v1 §12).
  Deleted all 6 `docs/reference/api/navigator/` pages and removed all cross-references:

  | File | Action |
  |------|--------|
  | `docs/reference/api/navigator/index.md` | Deleted |
  | `docs/reference/api/navigator/tree_navigator.md` | Deleted |
  | `docs/reference/api/navigator/path_resolver.md` | Deleted |
  | `docs/reference/api/navigator/compatibility_engine.md` | Deleted |
  | `docs/reference/api/navigator/replication_library.md` | Deleted |
  | `docs/reference/api/navigator/yaml_execution.md` | Deleted |
  | `docs/reference/api/index.md` | Removed Navigator section and toctree entry |
  | `docs/reference/architecture/philosophy.md` | Replaced "Navigation before execution" with "Option reference"; removed Navigator link from Next section; updated "Contracts are enforced" |

  Termination condition 1: `docs/reference/api/navigator/` is empty. PASS.

- **PR5 (docs precision audit): remove redirect-only pages from hidden toctree in how_to/index.md**

  Five redirect-only pages (`add_dataset.md`, `custom_model.md`, `partial_execution.md`,
  `custom_hooks.md`, `user_data_workflow.md`) already carry `orphan: true` front-matter,
  but were listed in a `:hidden:` toctree block in `docs/how_to/index.md` (lines 48-56).
  A hidden toctree suppresses the block from the visible page body while still including
  the listed pages in the global sidebar navigation tree. Listing an orphan page in any
  toctree overrides the `orphan` directive, causing the redirect stubs to appear in the nav.
  Removed the entire hidden toctree block; the canonical destination pages
  (`add_custom_dataset`, `add_custom_model`, `partial_layer_execution`, `use_extension_points`)
  remain properly listed in the active Infrastructure guides toctree. (check6b audit)

### Internal

- **PR7 (hotfix FINAL): docs-code drift verification gate + tool import-path fixes**

  Final PR in the comprehensive hotfix cascade. Verification gate confirms all 15 TCs pass.
  Additionally resolved 21 pre-existing test failures discovered during verification:

  | Change | Files |
  |--------|-------|
  | `tools/docgen/builder.py`: `api.run` → `api.recipe.run` (post-restructure drift) | `builder.py` |
  | `tools/audit_docs_vs_code.py`: `_bootstrap_ops` old `core.ops.l5/l6/l8_ops` → `layers.*.ops` | `audit_docs_vs_code.py` |
  | `tools/gen_encyclopedia_docs.py`: same `_bootstrap_ops` import-path fix | `gen_encyclopedia_docs.py` |
  | `tests/api/`: `model_family=` → `model=` in 7 test call sites (post-PR4 drift) | `test_forecast.py`, `test_experiment.py` |
  | `tests/core/test_l3_5_selection_view_none.py`: fix import to `macroforecast.diagnostics.features.schema` | `test_l3_5_selection_view_none.py` |

  **Verification results:**

  | TC | Condition | Verdict |
  |----|-----------|---------|
  | TC1 | README/ARCHITECTURE no "12-layer" / "12 layers" | PASS |
  | TC2 | CHANGELOG no cycle codes | PASS |
  | TC3 | No `macroforecast.scaffold` in docs/macroforecast (excluding _build) | PASS |
  | TC4 | `explanation/architecture/` no `# Layer N:` headings | PASS |
  | TC5 | No `custom_source_policy` in macroforecast/ | PASS |
  | TC6 | `model_family` in api/ only in deprecation context | PASS |
  | TC7 | Deprecated constants only in `__getattr__` shim | PASS |
  | TC8 | No `model_family:` / `base_family:` in examples/recipes/ | PASS |
  | TC9 | `examples/recipes/archive_v0/` deleted | PASS |
  | TC10 | `tests/scaffold/` removed | PASS |
  | TC11 | No old `core.ops.l3/l4/l7_ops` imports in tests/ | PASS |
  | TC12 | Focused pytest (api + tools): 21 new passes, 0 regressions | PASS |
  | TC13 | `test_encyclopedia_op_coverage.py` — 95 passed | PASS |
  | TC14 | Encyclopedia regen: 313 pages, zero diff vs checked-in | PASS |
  | TC15 | L1-L8 axis counts: all match; L6 -4 meta axes (intentional); L3/L4/L7 op gaps expected | PASS |

  **Behavioral impact**: NONE to runtime. Tool import paths corrected. Test assertions fixed to match current API.

- **PR6 (hotfix): tests API vocab update + tests/scaffold/ → tests/tools/docgen/ move**

  Follow-up cleanup to PR4 and Phase 5 tooling rename:

  | Change | Files |
  |--------|-------|
  | `OPERATIONAL_MODEL_FAMILIES` → `OPERATIONAL_MODELS` | `test_bvar_minnesota.py`, `test_dfm_mariano_murasawa.py`, `test_factor_augmented_var.py`, `test_mrf_gtvp.py`, `test_status_honesty.py`, `test_v09_paper_coverage.py`, `test_l4_midas_family_c48.py`, `test_l4_realized_garch_c49.py` |
  | `FUTURE_MODEL_FAMILIES` → `FUTURE_MODELS` | same set |
  | `macroforecast.core.ops.l3_ops` → `macroforecast.features.ops` | `test_phase_c_top6.py` |
  | `macroforecast.core.ops.l4_ops.OPERATIONAL_MODEL_FAMILIES` → `macroforecast.models.ops.OPERATIONAL_MODELS` | `test_encyclopedia_op_coverage.py` docstring |
  | `tests/scaffold/` → `tests/tools/docgen/` (git mv) | 15 test files + `__init__.py` |
  | `ENC_ROOT = Path(__file__).parents[3]` (was `parents[2]` before move) | `test_encyclopedia_op_coverage.py` |
  | Add `conftest.py` at repo root: pinches project root to `sys.path[0]` | `conftest.py` (new) |
  | Add `tests/tools/docgen/conftest.py`: guards `tools.docgen` namespace against pytest importlib shadow | `tests/tools/docgen/conftest.py` (new) |
  | Add backward-compat `DeprecationWarning` test for old constant names | `test_status_honesty.py` |

  **Termination conditions:**

  | TC | Condition | Verdict |
  |----|-----------|---------|
  | TC10 | `tests/scaffold/` does not exist; `tests/tools/docgen/` exists | PASS |
  | TC11 | No stale `macroforecast.core.ops.l4_ops` / `l7_ops` imports in `tests/` | PASS |
  | Drift gate | `tests/tools/docgen/test_drift_gate_meta.py` — 1 passed | PASS |
  | Encyclopedia | `tests/tools/docgen/test_encyclopedia_op_coverage.py` — 95 passed | PASS |

  **Behavioral impact**: NONE. No source code modified. No test logic changed — vocab and import paths only.

- **Phase 3g-bis FINAL: cascade complete (TC1–TC7 verified)** (see PR #K)

  **Cascade summary — 10 PRs (B–J merged, #H no-op, #K this PR):**

  | PR | GitHub | Description | Status |
  |----|--------|-------------|--------|
  | #B | #360 | Eliminate registry.py circular-import workaround | MERGED |
  | #C | #361 | Delete legacy L0 schema body (`core/layers/l0.py`) | MERGED |
  | #D | #362 | Collocate L5 schema and ops | MERGED |
  | #E | #363 | Collocate L6 schema and ops | MERGED |
  | #F | #364 | Collocate L7 schema and ops (largest PR by line count) | MERGED |
  | #G | #365 | Collocate L8 schema and ops | MERGED |
  | #H | — | No-op (planner decision; no change required) | SKIPPED |
  | #I | #366 | Relocate `interpretation/` body to canonical `methods.py` | MERGED |
  | #J | #367 | `recipes/` docstring normalization | MERGED |
  | #K | — | Fix 6 stale test imports to enable TC7 | THIS PR |

  **Termination condition verification:**

  | TC | Condition | Verdict |
  |----|-----------|---------|
  | TC1 | `core/layers/` clean — no `lN.py` files remain | PASS |
  | TC2 | `core/ops/` clean — `universal.py`, `registry.py`, `diagnostic_ops.py` only (diag_ops kept per planner spec-H) | PASS |
  | TC3 | `interpretation/__init__.py` is 21-line backward-compat shim | PASS |
  | TC4 | `recipes/__init__.py` is 42-line backward-compat shim | PASS |
  | TC5 | No duplicate class definitions anywhere in the codebase | PASS |
  | TC6 | No "Deferred core imports" or "circular-dependency" comments in any source file | PASS |
  | TC7 | `pytest tests/` passes piecewise — 8 per-PR testers each reported 0 new failures; PR #K fixed 6 pre-existing stale test imports (out of scope for prior phases) to unblock collection errors and complete TC7 | PASS |

  **PR #K change record**: Fixed stale legacy import paths in 6 test files
  (`tests/core/test_l3_feature_selection_temporal_rule.py`,
  `tests/layers/test_l3.py`, `tests/layers/test_l3_5.py`,
  `tests/models/test_c64_baseestimator_selectors.py`,
  `tests/promotion/test_c63_1_chow_lin.py`,
  `tests/transforms/test_chow_lin_disaggregate.py`). These files held imports
  from modules deleted during Phases 3a–3f that were never cleaned up.
  Additional inline legacy imports discovered and fixed within the same 6 files
  (8 sites across `test_l3.py`, `test_l3_5.py`, and
  `test_chow_lin_disaggregate.py`). No source code was modified; no new tests
  were added; no test logic was changed — import paths only.

  **Behavioral impact**: NONE. Tester PASS (5/5 gates). 0 new failures introduced.

- **Phase 3g-bis PR #J: recipes/ docstring normalization** (see PR)
  - Modified: `macroforecast/recipes/__init__.py` — docstring rewrite, -22 lines (64 → 42).
    Removed redundant boilerplate; tightened module-level description.
  - No functional change. `paper_methods.py` was already relocated in Phase 3b;
    this PR contains no source-code edits.
  - **Behavioral impact**: NONE. Tester PASS (5/5 gates). 0 new failures.

- **Phase 3g-bis PR #I: relocate interpretation/ body to canonical methods.py** (see PR)
  - Modified: `macroforecast/interpretation/__init__.py` (225-line body reduced to
    21-line backward-compat re-export shim; all symbols re-exported from canonical
    location so `from macroforecast.interpretation import ...` continues to work).
  - Modified: `macroforecast/layers/l7_interpretation/methods.py` (39-line shim
    promoted to 226-line canonical body carrying GIRF, LSTMHiddenState, and helper
    implementations).
  - **Backward compatibility**: preserved. No import path changes required in
    downstream code or tests.
  - **Test edits**: none. 0 new test failures. Tester PASS (5/5 gates).
  - **Behavioral impact**: NONE. Body relocation only.

- **Phase 3g-bis PR #G: collocate L8 schema and ops** (see PR)
  - Deleted: `macroforecast/core/layers/l8.py` (-429 lines) — legacy schema
    body superseded by the Phase 3f collocated-layer restructure.
  - Deleted: `macroforecast/core/ops/l8_ops.py` (-19 lines) — legacy ops
    body; full implementation relocated to canonical location.
  - Canonical location (unchanged): `macroforecast/layers/l8_output/ops.py`
    (shim replaced with full body).
  - Modified: `macroforecast/core/runtime.py` (1 line: l8 layer source updated
    via side-effect import).
  - Test import paths updated in 1 file: `tests/layers/test_l8.py`.
  - **Behavioral impact**: none. Dead-file removal + shim-to-body promotion only.
    **Completes layer body cleanup**: after this PR, all `core/layers/lN.py` and
    `core/ops/lN_ops.py` legacy files are removed from the codebase.

- **Phase 3g-bis PR #F: collocate L7 schema and ops (biggest)** (see PR)
  - Deleted: `macroforecast/core/layers/l7.py` (-428 lines) — legacy schema
    body superseded by the Phase 3f collocated-layer restructure.
  - Deleted: `macroforecast/core/ops/l7_ops.py` (-711 lines) — legacy ops
    body; 717-line full implementation relocated to canonical location.
  - Canonical location (unchanged): `macroforecast/layers/l7_interpretation/ops.py`
    (shim replaced with full body; +717 LOC).
  - Modified: `macroforecast/core/runtime.py` (1 line: l7 layer source updated
    via side-effect import).
  - Test import paths updated in 11 files: all affected L7 test files updated
    to canonical `macroforecast.interpretation` paths.
  - **Behavioral impact**: none. Dead-file removal + shim-to-body promotion only.
    This is the largest single PR in the Phase 3g-bis cascade by line count.

- **Phase 3g-bis PR #E: collocate L6 schema and ops** (see PR)
  - Deleted: `macroforecast/core/layers/l6.py` (-499 lines) — legacy schema
    body superseded by the Phase 3f collocated-layer restructure.
  - Deleted: `macroforecast/core/ops/l6_ops.py` (-43 lines) — legacy ops
    body; full implementation moved to canonical location.
  - Canonical location (unchanged): `macroforecast/layers/l6_tests/ops.py`
    (shim replaced with full body).
  - Modified: `macroforecast/core/runtime.py` (1 line: l6 layer source updated).
  - Test import paths updated in 1 file: `tests/layers/test_l6.py` (2 import
    sites; legacy `macroforecast.core.layers.l6` replaced with
    `macroforecast.stat_tests.schema`).
  - **Behavioral impact**: none. Dead-file removal + shim-to-body promotion only.

- **Phase 3g-bis PR #D: collocate L5 schema and ops** (see PR)
  - Deleted: `macroforecast/core/layers/l5.py` (-417 lines) — legacy schema
    body superseded by the Phase 3f collocated-layer restructure.
  - Deleted: `macroforecast/core/ops/l5_ops.py` (-86 lines) — legacy ops
    body; full 87-line implementation moved to canonical location.
  - Canonical location (unchanged): `macroforecast/layers/l5_evaluation/ops.py`
    (shim replaced with full body).
  - Modified: `macroforecast/core/runtime.py` (1 line: l5 layer source updated).
  - Test import paths updated in 1 file: `tests/layers/test_l5.py` (lines 1
    and 149; legacy `macroforecast.core.layers.l5` replaced with
    `macroforecast.evaluation.schema`).
  - **Behavioral impact**: none. Dead-file removal + shim-to-body promotion only.

- **Phase 3g-bis PR #C: delete legacy L0 schema body** (see PR)
  - Deleted: `macroforecast/core/layers/l0.py` (373 lines) — legacy duplicate
    body superseded by the Phase 3f collocated-layer restructure.
  - Canonical location (unchanged): `macroforecast/layers/l0_meta/schema.py`.
  - Test import paths updated in 2 files: `tests/layers/test_l0.py` and
    `tests/core/test_parallel_unit_cells.py` (legacy `core.layers.l0` path
    replaced with `macroforecast.meta.schema`).
  - **Behavioral impact**: none. Dead-file removal only.

- **Phase 3g-bis: eliminate registry.py circular-import workaround** (see PR)
  - New file: `macroforecast/core/layers/_bootstrap.py` — holds all 13
    `register_layer()` calls, loaded as a side-effect import after registry
    machinery is fully initialized.
  - Modified: `macroforecast/core/layers/__init__.py` (side-effect import of
    `_bootstrap`), `macroforecast/core/layers/registry.py` (13 schema imports
    and 13 `register_layer()` calls removed).
  - Modified: 9 schema files — removed "Deferred core imports" comment blocks
    and `# noqa: E402` markers; all imports now at standard position.
  - **Behavioral impact**: none. Structural refactor only.

### Maintenance

- Remove internal workflow artifacts from public repository: untrack `CLAUDE.md`,
  `mailbox.md`, `implementation.md`; add `plans/design/` to `.gitignore`; strip
  agent identifiers from `ARCHITECTURE.md`; delete stale `docs/_archive/`
  entries (11 files). Strip `plans/design/` cross-references from source
  docstrings, README, contributing guide, and test fixtures to maintain
  repo consistency. **Behavioral impact**: NONE. No source logic changed.

### Removed

- L1 axis `official_transform_policy` (duplicates L2 `transform_policy`)
- L1 axis `official_transform_scope` (L2 transform is global by axis; no replacement)
- L1 axis `raw_missing_policy` (duplicates L2 `imputation_policy`)
- L1 axis `raw_outlier_policy` (duplicates L2 `outlier_policy`)
- Preprocessing parameter `tcode_application_scope` (absorbed into L2 `transform_policy` semantics)
- Documentation pages `docs/reference/architecture/layer1/raw_source_cleaning.md` and
  `docs/reference/architecture/layer1/official_transforms.md` (described the removed axes)
- `macroforecast.wizard` package and all Solara UI dependencies (`solara`,
  `[wizard]` optional extra) per the v0.10 restructure plan, Phase 0.
- `macroforecast.scaffold.wizard` companion module.
- `macroforecast.core.stages` and the `STAGE_BY_LAYER` / `StageLabel` /
  `stage_of` re-exports from `macroforecast.core`. Use `layer_id` directly.
- Stale entries in `defaults.DEFAULT_PREPROCESSING_AXES` that referenced
  L2 axes which no longer exist.

### Changed
- Applied locked TAXONOMY v1 vocabulary across recipes, source code, tests, and
  user-facing docs. Recipe axis renames:
  - `family:` -> `model:` (L4 axis)
  - `op: fit_model` -> `op: fit`
  - `custom_source_policy` -> `panel_composition` (L1)
  - `quarterly_to_monthly_rule` -> `quarterly_to_monthly_policy` (L2)
  - `monthly_to_quarterly_rule` -> `monthly_to_quarterly_policy` (L2)
  - `compute_mode` -> `compute_policy` (L0)
  - `reproducibility_mode` -> `reproducibility_policy` (L0)
  - `forecast_strategy` -> `forecast_policy` (L4)
  - `alpha_strategy` -> `alpha_search_policy` (L4)
  - `correction_method` -> `correction_policy` (L6)
  - `scaling_method` -> `scaling_policy` (L2/L3)
  - `target_geography_scope` -> `target_geography_policy` (L1)
  - `predictor_geography_scope` -> `predictor_geography_policy` (L1)
- File rename: `macroforecast/core/dag.py` -> `macroforecast/core/pipeline.py`.
  The internal `DAG` class symbol is retained inside the renamed file. Public
  uses should switch to `from macroforecast.core.pipeline import DAG`.
- Removed L2 `*_scope` axes (`outlier_scope`, `transform_scope`,
  `imputation_scope`, `frame_edge_scope`). The runtime now uses inlined
  defaults instead of axis-driven scope decisions. Recipes that previously
  set these axes will no longer apply scope filtering.
- `DEFAULT_MODEL_FAMILY` constant renamed to `DEFAULT_MODEL` in
  `macroforecast/defaults.py` and `macroforecast/api_high.py`.

### Breaking changes
- Recipes using the old axis names will fail validation. Apply the rename
  table above before upgrading.
- `from macroforecast.core.dag import DAG` no longer works. Use
  `from macroforecast.core.pipeline import DAG`.
- Standalone functions with renamed parameters: `sliced_inverse_regression_transform`
  uses `scaling_policy=` instead of `scaling_method=`, `ridge_variants` uses
  `alpha_search_policy=` instead of `alpha_strategy=`, `tests.py` result
  dataclasses use `correction_policy` instead of `correction_method`.

---

## [0.9.5a0] -- 2026-05-25 -- "Round 3: documentation + naming + refactor"

Round 3 shifts macroforecast from a contract-first YAML framework to
a standalone-first library API, cleans up internal DAG jargon from the public
surface, and completes a full Diátaxis documentation overhaul including rewritten
tutorials and per-algorithm how-tos. No algorithmic logic was changed.

2026-06-03 publishing pass: `0.9.5a0` is the alpha label for the current
callable-first package snapshot on `main`. This pass also adds the first
paper-replication settings page for Goulet Coulombe et al. (2021), including a
historical FRED-MD vintage policy for the `1980M01-2017M12` experiment.

### Breaking Changes

None.

### Added

#### Added — Cycle 63 (standalone-first promotion, L4/L3/L7)

- **Standalone model API** (`mf.models`): 22 previously private `_<Name>` classes
  promoted to public. `mf.models.linear` (8), `mf.models.tree` (6), `mf.models.neural`
  (2), `mf.models.timeseries` (4), `mf.models.factor` (2). Backward compat preserved
  via thin subclass pattern.
- **`mf.feature_selection`**: 5 sklearn-compatible wrappers: `Boruta`,
  `RecursiveFeatureElimination`, `LassoPathSelector`, `StabilitySelector`,
  `GeneticAlgorithmSelector`.
- **`mf.interpretation`**: `GIRF`, `LSTMHiddenState` promoted to public.
- 8 gap standalone callables added to `mf.functions.*` for newly promoted
  model families.
- 8 additional model promotions: `mf.models.tree` gains 6 classes
  (`SlowGrowingTree`, `SparsePCRTree`, `BoostedTree`, `AutoClipTree`,
  `CondPFITree`, `StagewiseTree`); `mf.models.neural` gains 2 (`LSTM`, `GRU`
  convenience subclasses via `SequenceModel`).
- 5 C63 selectors refactored to inherit from
  `sklearn.base.BaseEstimator + TransformerMixin`, adding `feature_names_in_` /
  `n_features_in_` tracking and full `sklearn.clone()` compatibility.
- 2 runtime bug fixes: `_SlowGrowingTree._build` infinite BFS capped at
  `max_depth=10`; `_BaggingWrapper.fit` `TypeError` on `base_params=None` guarded.
- **`mf.recipes`** canonical namespace: `run`, `run_file`, `replicate`,
  `forecast`, `Experiment`, `ForecastResult` re-exported. `mf.run is mf.recipes.run`
  identity preserved -- no user-facing behavior change.
- Tutorials 01-03 rewritten as standalone-first. `01_first_forecast.md`:
  `LinearAR(p=2)` on synthetic data (211 -> 95 lines). `02_full_study.md`: 3-model
  comparison with `TimeSeriesSplit` (491 -> 125 lines). `03_custom_model.md`:
  `BaseEstimator + RegressorMixin` subclass pattern (258 -> 130 lines).
- Six new per-algorithm how-tos: RealizedGARCH (Hansen-Huang-Shek 2012),
  Boruta (Kursa-Rudnicki 2010), BVARMinnesota (Litterman 1986), Chow-Lin GLS
  (Chow-Lin 1971), GIRF (Pesaran-Shin 1998), 4-way MIDAS comparison (Ghysels 2004,
  2007; Foroni 2015).
- `docs/how_to/advanced_recipes.md` (164 lines): YAML structure, custom-step
  extension via `mf.register_model`, recipe composition, manifest semantics, migration
  guide.

### Changed

- `macroforecast/__init__.py` `_LAZY_EXPORTS`: 8 orchestration symbols rerouted
  through `.recipes` instead of `.api` / `.api_high`. Module docstring updated.
- Public-facing DAG jargon reduced: `macroforecast/` 464 -> 10 occurrences (98%
  reduction); `docs/` non-archive 131 -> 7 (95% reduction). Internal `core/dag.py`
  module docstring annotated as internal-vocabulary. 1 internal wizard file renamed
  (`layer_dag.py` -> `layer_step_graph.py`). Public classes `DAG`, `DAGValidationError`,
  `validate_dag` preserved unchanged for backward compatibility.
- `docs/how_to/index.md` reorganized: 3 visible categories (per-algorithm /
  infrastructure / advanced) + preserved legacy block.

### Fixed

- Documentation restatement of `BaseEstimator` inheritance semantics:
  `isinstance(public_instance, _PrivateClass)` continues to hold in the forward
  direction; the reverse check (private inherits from public) is False, as
  expected with single inheritance.
- `mf.transforms.chow_lin_disaggregate` replaced with canonical Chow-Lin (1971)
  GLS implementation that preserves temporal totals. Previous OLS wrapper did not
  enforce the conservation constraint.
- `BVARMinnesota` enforces `prior='minnesota'` internally.

### Deprecated

None.

### Documentation

- R3-P0 audit: full class inventory (35 entries), naming convention spec,
  recipe-extraction spec, DAG cleanup plan, R3 execution roadmap.
- Tutorial API parameter names corrected from source: `LinearAR(p=2)` (not
  `n_lags`), `FactorAugmentedAR(p=2, n_factors=3)`, `predict(X_test)`.
- `tests/promotion/test_c68_howto_validation.py`: 50 parametrized validation
  cases (49 PASS + 1 skip).

### Packaging

- Version: `0.9.3b1` -> `0.9.5a0`. The version jump skips `0.9.4`; PEP 440 ordering
  `0.9.3b1 < 0.9.5a0` is valid. The pre-release suffix changes from `b1` to `a0`
  because `0.9.5a0` begins a new alpha snapshot series at a higher version, per
  user versioning policy. The algorithmic surface is unchanged since `0.9.3b1`; the
  `a0` suffix reflects the snapshot nature of this docs/refactor cut, not a
  regression in stability.
- `Development Status` Trove classifier remains `4 - Beta`. The `a0` suffix is a
  versioning choice; functional maturity has not regressed from `0.9.3b1`.
- Install documentation pins updated: `docs/tutorial/00_install.md` (4 occurrences),
  `README.md` (3 occurrences).

---

## [0.9.3b1] -- 2026-05-22 -- "Round 2 cross-review remediation"

Codex + MiniMax external cross-review identified P0/P1/P2 statistical and governance gaps in v0.9.2b2. This release closes them.

### Breaking Changes

- (HHS realized_garch) `_RealizedGARCHModel.fit()` no longer silently returns init params on convergence failure. Raises `RuntimeError` instead. Users relying on silent fallback must catch the exception.
- (Boruta) `_boruta_selection` no longer returns argmax(hit_count) when no feature is formally accepted. Returns empty DataFrame on null DGP. Fixes 100% false-positive rate.

### Fixed

- (Boruta P0/P1) **Critical**: false-positive rate on null data corrected from **100% to 3.3%**. Two-bug fix: argmax fallback removed + multi-shadow MISA calibration (`n_shadow_copies=6`).
- (HHS P0) Multi-start L-BFGS-B `bounds` enforcement, consistent `log_sigma_u` clip in objective and storage, convergence metadata exposed.
- (Custom model contract) `_CustomModelAdapter._invoke` context now includes `target` and `horizon` keys per Tutorial 03 docs (was: KeyError).
- (Encyclopedia drift) 2 missing pages added: `lstm_hidden_state` (L7), `chow_lin_disaggregation` (L3).
- (Tutorial 04) YAML key correction `preprocessor_name` (invalid) to `leaf_config.custom_preprocessor` (correct).

### Added

- (CI governance P0) `release.yml` trigger changed from `push.tags` (auto-publish on tag) to `workflow_dispatch` (manual). PyPI publish requires explicit user action via GitHub UI.
- (CI drift gate) `tests/tools/docgen/test_encyclopedia_op_coverage.py` — 95-item gate fails when operational op lacks reference page.
- (HHS recovery depth P1) 5-seed multi-seed recovery test with tightened tolerances (mu atol 0.05 to 0.02, omega 0.50 to 0.15, etc); T bumped 500 to 2000 for asymptotic SE compliance. (C56+C59)
- (R cross-reference P1) Real rpy2 bridge for Boruta/midasr/rugarch validation, gated via `pytest.importorskip`. New `[validation]` optional extra.
- (Tutorial 04 custom_preprocessor) Full narrative tutorial covering `register_preprocessor` API with synthetic data, debugging section.
- (How-to validate_against_r) Documentation for R cross-reference setup.
- (MIDAS encyclopedia clarity) Optimization method labels (NLS vs OLS) on Almon/Beta/Step/U-MIDAS pages.
- (Documentation polish) `12_layer_design.md` terminology unified (12 brand + 9+4=13 slot count reconciled). `bit_exact_replicate.md` forward-reference to caveats.

### Deferred

- (`register_metric` tutorial) API does not exist in `macroforecast.custom`; tutorial deferred to a future cycle pending API design.
- (Monte Carlo finite-sample suite) Bias/RMSE/coverage assessment for HHS/MIDAS/GIRF; deferred beyond Round 2.

### Packaging

- v0.9.3b1 fresh semantic version (0.9.3 > 0.9.2 final on PyPI). Pre-release suffix `b1` requires `pip install macroforecast --pre` to install via PyPI default.
- `pyproject.toml` metadata polish: Trove classifiers (Beta, Python 3.10/3.11/3.12, MIT, Topic, Audience), `authors`, `keywords`.
- `LICENSE` file added (MIT body).
- `macroforecast/py.typed` marker added (PEP 561 type-hint exposure).

---

## [0.9.2b2] -- 2026-05-22 -- "v0.9.3 algorithmic completion + Diátaxis docs overhaul"

v0.9.3 algorithmic honesty pass is complete: all four cycles promoted
paper-faithful implementations, leaving `FUTURE_MODEL_FAMILIES = ()` and
`FUTURE_OPS = ()`. Fourteen items promoted across four layers (L3 feature
selection, L4 MIDAS families, L4 Realized GARCH, L7 GIRF, L1/L2 real-time
vintage and temporal disaggregation). C51 closed four tracked issues. C52-C54
reorganized and populated the Diátaxis 4-tier documentation structure: 3
narrative tutorials, 6 task how-tos, 4 explanation pages, and a reference index
serving 319 auto-generated encyclopedia pages.

### Added

- **C47 — L3 Feature Selection (5 ops promoted to operational)**
  - `boruta_selection`: Kursa & Rudnicki (2010, JSS 36(11)) shadow-feature
    permutation test; Bonferroni-corrected two-sided binomial test; pure NumPy +
    sklearn `RandomForestRegressor`; no `boruta` package dependency.
  - `recursive_feature_elimination`: Guyon et al. (2002, Machine Learning 46)
    recursive backward elimination with optional RFECV; base estimators: ridge,
    lasso, svr_linear.
  - `lasso_path_selection`: Efron et al. (2004, AoS 32(2)) LARS path entry-order
    selector; selects first `n_features_to_select` features to enter the LARS
    active set; distinct from `feature_selection(method="lasso")` which ranks by
    LassoCV coefficient magnitude at convergence.
  - `stability_selection`: Meinshausen & Bühlmann (2010, JRSS-B 72(4)); lasso or
    elastic-net on `n_subsamples` random subsamples; features retained where
    selection probability exceeds `pi_thr`; defaults n_subsamples=100,
    subsample_fraction=0.5, pi_thr=0.6.
  - `genetic_algorithm_selection`: Goldberg (1989) binary-chromosome GA;
    tournament selection, single-point crossover, bit-flip mutation rate 1/N,
    elitism; fitness = CV negative MSE; pure NumPy, no `deap` dependency.
  - L3 operational op count: ≥32 → ≥37.

- **C48 — MIDAS Families (4 L4 families promoted to operational; count 42 → 46)**
  - `midas_almon`: Almon polynomial lag weights estimated by multi-start
    Nelder-Mead NLS; implements Ghysels, Santa-Clara & Valkanov (2004) §2 eq. (3).
  - `midas_beta`: Beta distribution kernel lag weights by multi-start NLS;
    implements Ghysels, Sinko & Valkanov (2007) §2.
  - `midas_step`: Piecewise-constant step-function lag weights by OLS; implements
    Foroni, Marcellino & Schumacher (2015) §2.2.
  - `dfm_unrestricted_midas`: U-MIDAS by OLS with optional AR(1) y-lag; implements
    FMS (2015) §3 eqs. (7), (20); lag order by BIC/AIC.
  - All four share per-origin seed contract (`random_state = base_seed +
    origin_position`) from #279.

- **C49 — Realized GARCH + Pesaran-Shin GIRF (2 items promoted)**
  - `realized_garch`: Hansen, Huang & Shek (2012, JAE 27(6): 877-906) joint-MLE
    Realized GARCH; three-equation system (return + log-variance + measurement);
    11-parameter vector via `scipy.optimize.minimize(method="L-BFGS-B")` with 3
    multi-starts; no `arch` dependency. Distinct from
    `realized_garch_with_rv_exog` (RV-as-exogenous-regressor via `arch` package).
  - `generalized_irf`: Pesaran & Shin (1998, Economics Letters 58(1): 17-29)
    order-invariant GIRF; formula `GIRF_h(j) = sigma_jj^{-1/2} * A_h * Sigma *
    e_j`; importance metric is L1 norm of target-variable response; order-
    invariance verified atol=1e-8. Distinct from `orthogonalised_irf` (Cholesky,
    order-dependent).
  - After C49: `FUTURE_MODEL_FAMILIES = ()`.

- **C50 — Final v0.9.3 Algorithmic Items (4 items promoted; after C50: `FUTURE_OPS = ()`)**
  - `real_time_alfred` (`vintage_policy`, L1.A): real-time ALFRED vintage policy
    operational; two modes: `alfred_mode=local` (pre-downloaded snapshots from
    `alfred_snapshot_dir`) or `alfred_mode=api` (ALFRED REST API via
    `alfred_api_key` / `FRED_API_KEY`). Reference: Croushore & Stark (2001, JoE
    105(1)); Federal Reserve Bank of St. Louis ALFRED API.
  - `chow_lin` (`quarterly_to_monthly_rule`, L2.A): regression-based temporal
    disaggregation; GLS with AR(1) errors using monthly indicator
    `leaf_config.chow_lin_indicator`; quarterly sum-constraint preserved by
    construction. Reference: Chow & Lin (1971, RES 53(4), doi:10.2307/1928739).
  - `keep_with_indicator` (`outlier_action`, L2.C): keeps original outlier value
    and appends binary `{col}__outlier_flag` column (1=flagged, 0=clean).
  - `lstm_hidden_state` (L7.A op): LSTM hidden-state activation heatmap; per-
    timestep `h_t` activations via PyTorch forward hook; requires
    `macroforecast[deep]`; raises `NotImplementedError` for transformer. Reference:
    Karpathy, Johnson & Fei-Fei (2015, arXiv:1506.02078).

- **C51 — Issue Cleanup**
  - `tests/tools/test_gen_encyclopedia.py`: `test_gen_no_third_party_imports`
    (closes #318).
  - `tests/tools/test_audit_docs.py`: `test_audit_no_third_party_imports`
    (closes #318).
  - `macroforecast/scaffold/option_docs/l2.py`: formal `ParameterDoc` for
    `chow_lin_indicator` (Flag-D).
  - `macroforecast/core/ops/l3_ops.py`: `chow_lin_disaggregation` promoted from
    `future` to `operational`.

- **C52-C54 — Diátaxis Documentation Overhaul**
  - `docs/tutorial/01_first_forecast.md`, `02_full_study.md`,
    `03_custom_model.md`: three new narrative tutorials replacing empty stubs
   ; `two_entry_points.md` renamed and polished.
  - `docs/how_to/add_custom_dataset.md`, `tune_hyperparameters.md`,
    `sweep_over_models.md`, `replicate_a_study.md`: four new how-to task
    recipes; two refactors, one rename.
  - `docs/explanation/12_layer_design.md`, `bit_exact_replicate.md`,
    `honesty_pass.md`, `recipe_to_run.md`: four new conceptual explanation pages
   .
  - `docs/reference/api/index.md`: new umbrella index for API sub-tier.
  - `tests/docs/test_tutorial_smoke.py`: CI smoke test extracting Python blocks
    from tutorials 01-03 and executing them in subprocess (C54; closes C53
    reviewer note 1).

### Changed

- **C47 — L3 Op Registry Refactor**
  - `macroforecast/core/ops/l3_ops.py`: removed `_future_selection_op` closure
    factory and 5-name `for` loop; replaced with 5 separate
    `@register_op(status="operational")` decorated functions with individual
    `params_schema` and `hard_rules`.
  - `macroforecast/core/ops/l7_ops.py`: removed `boruta_selection`,
    `recursive_feature_elimination`, `lasso_path_selection`, `stability_selection`
    from `FUTURE_OPS` (L3 ops must not gain L7 scope via tail registration loop).
  - `macroforecast/core/runtime.py`: 5 dispatch branches + 5 private helpers in
    `_execute_l3_op`; all follow #215/#279 seed-propagation contract.
  - `tests/layers/test_l7.py`, `tests/layers/test_l3.py`: tests renamed and
    thresholds updated to reflect new op counts.

- **C48 — L4 MIDAS Dispatch**
  - `macroforecast/core/ops/l4_ops.py`: `OPERATIONAL_MODEL_FAMILIES` gains 4
    entries; `FUTURE_MODEL_FAMILIES` reduced 5 → 1 (`realized_garch` only).
  - `macroforecast/core/runtime.py`: 4 dispatch branches + 4 private model
    classes (`_MidasAlmonModel`, `_MidasBetaModel`, `_MidasStepModel`,
    `_UnrestrictedMidasModel`); two-attribute design `_w_hat_effective` +
    zero-padded `_w_hat`.

- **C49 — Internal GARCH Family Rename**
  - `macroforecast/core/runtime.py`: `_GARCHFamily` internal variant renamed from
    `"realized_garch"` to `"rv_exog"` for the `realized_garch_with_rv_exog`
    branch (internal only; no external API change).
  - `ARCHITECTURE.md`: L4 operational count 46 → 47; L7 `FUTURE_OPS` count 2 → 1.

- **C50 — Layer Validator + ALFRED Optimizer Updates**
  - `macroforecast/core/layers/l1.py`: `VintagePolicy` Literal updated; validator
    accepts `real_time_alfred`; soft validation checks `alfred_snapshot_dir` (local)
    or `alfred_api_key` / `FRED_API_KEY` (api).
  - `macroforecast/core/layers/l2.py`: `chow_lin` + `keep_with_indicator` added to
    valid sets and `AxisSpec`; hard-reject guards removed.
  - `macroforecast/core/ops/l7_ops.py`: `lstm_hidden_state` in `OPERATIONAL_OPS`;
    `FUTURE_OPS = ()`.
  - `macroforecast/raw/alfred_adapter.py`: rolling-mode ALFRED vintage resolution
    vectorized O(N×snapshot) → O(snapshot+N).
  - Option docs for `l1.py`, `l2.py`, `l7_a.py`: new OptionDoc and ParameterDoc
    entries.

- **C51 — var_decomp Internal Rename + Tool Help Text**
  - `macroforecast/raw/alfred_adapter.py`: rolling-mode vectorization (also C50;
    confirmed operational in C51).
  - `tools/gen_encyclopedia_docs.py`: `--out` help text updated to document
    auto-create behavior (closes #319).

- **C52 — Diátaxis Docs Structure**
  - Existing content moved via `git mv` from 11 parallel directories to 4-tier
    structure; encyclopedia moved to `docs/reference/encyclopedia/`; architecture
    moved to `docs/reference/architecture/`; redirect stubs at retired URLs;
    CI drift check updated.

- **C53 — Tutorial and How-to Refactors**
  - `docs/tutorial/two_entry_points.md`: renamed from `03_two_entry_points.md`;
    broken links fixed.
  - `docs/tutorial/00_install.md`: broken quickstart link fixed.
  - `docs/how_to/add_custom_model.md`: refactored from `custom_function_quickstart`.
  - `docs/how_to/use_custom_hooks.md`: refactored from `custom_hooks.md`.
  - `docs/how_to/partial_layer_execution.md`: renamed from `partial_execution.md`
    + broken link fixes.
  - Tutorial index, how-to index: toctrees updated.

- **C54 — Reference Index + Landing Page**
  - `docs/reference/index.md`: card layout updated; encyclopedia in visible
    toctree; API links point to new umbrella index.
  - `docs/reference/encyclopedia/index.md`: auto-gen clarity sentence added.
  - `docs/index.md`: removed "Expanding in C54" placeholder.
  - `docs/explanation/index.md`: replaced stub with 4-page toctree.

### Fixed

- `docs/standalone_functions/l7_importance.md`: `shap_linear_importance` correctly
  requires `pip install macroforecast[shap]` (closes #310; C51).
- `docs/encyclopedia/l1/axes/information_set_type.md`: removed stale
  "future feature Cycle 14 K-4" references to `real_time_alfred` (C51, Flag-B).
- Issue #311: `test_r4_02_paper_table2_k60_midas_wins` timeout was resolved in C46
  via `@pytest.mark.slow`; issue closed with comment (C51, no code change).
- Issue #316: L3 op count clarified in ARCHITECTURE.md and CLAUDE.md (C51, no
  code change).

---

## [0.9.2b1] -- 2026-05-20 -- "introspection-based docs catalog + holiday pandas 2.x fix"

### Changed
- docs: all six per-layer pages under `docs/standalone_functions/` rewritten
  from `inspect.signature` and live result-dataclass `dir()` introspection.
  A previous v1 draft was REJECTED by cross-model review (Codex GPT-5.4) for
  hallucinated signatures and attribute names; the v2 cycle regenerates every
  callable entry from source via `tools/gen_standalone_docs.py`. No callable
  signature or result attribute appearing in docs is hand-authored.
- docs: `docs/conf.py` `myst_heading_anchors` raised from 3 to 4 so each H4
  callable block (`#### `name(...)``) gets a deep-link anchor.

### Fixed
- runtime: `_holiday_indicator` now imports `USFederalHolidayCalendar` from
  `pandas.tseries.holiday` (the v0.9.2 import path
  `pd.tseries.offsets.USFederalHolidayCalendar` was removed in pandas 2.x).
- tests: removed `@pytest.mark.xfail(strict=True)` decorators from three
  holiday-transform tests in
  `tests/functions/test_l3_final_b1_transforms.py`
  (`test_datetime_index_flag`, `test_datetime_panel_shape`,
  `test_bit_exact_datetime_vs_runtime`). They now pass with the runtime fix.

## [0.9.2b0] -- 2026-05-19 -- "docs visibility for standalone callables (paradigm beta)"

### Added
- New docs section `docs/standalone_functions/` — namespace overview + per-layer
  reference pages (l2_clean / l3_transforms / l4_fit / l5_metrics / l6_tests /
  l7_importance) cataloging the 117 standalone callables shipped in v0.9.2.
- New `docs/two_entry_points.md` decision guide: when to use the recipe DSL
  vs the standalone `mf.functions.<name>` callables.
- `README.md` "Quick standalone use" section showing 3-line examples for
  `ols_fit` / `mse` / `permutation_importance` before the recipe quickstart.
- `docs/index.md` "Pick your path" grid now includes a "🧩 Standalone
  functions" card linking to the new docs section.

### Changed
- Version 0.9.2 → 0.9.2b0 (PEP 440 beta) to publish docs-visibility update
  as a beta of v0.9.2 paradigm shift.

## [0.9.2] -- 2026-05-18 -- "paradigm shift: 109 new standalone callables across L2/L3/L4/L5/L6/L7 (sm.ols-style API)"

### Cycle 38 -- L7 importance standalone callables (8 ops)

**New standalone callables** in `mf.functions` (all return a frozen dataclass):

**importance.py (8 ops):**
`model_native_linear_coef_importance(result, X)` -> `NativeImportanceResult` (method="linear_coef")
`model_native_tree_importance(result, X)` -> `NativeImportanceResult` (method="tree_native")
`permutation_importance(result, X, y, *, n_repeats=10, random_state=None)` -> `PermutationImportanceResult`
`cond_permutation_importance(result, X, y, *, n_repeats=10, random_state=None)` -> `CondPermutationImportanceResult`
`partial_dependence_importance(result, X, *, grid_resolution=20)` -> `PDPImportanceResult`
`ale_importance(result, X, *, n_bins=20)` -> `ALEImportanceResult`
`shap_tree_importance(result, X)` -> `SHAPImportanceResult` (requires shap)
`shap_linear_importance(result, X)` -> `SHAPImportanceResult` (requires shap)

**UX pattern**: callables take a `FitResultBase`-conforming result object (any L4 standalone
callable return value) and extract `result._model` internally to call runtime helpers.

**`random_state=None`** for permutation ops uses deterministic reverse-order permutation
(bit-exact with `_permutation_importance_frame`). Integer seed draws random permutations.

**ALE**: cumsum + center (subtract mean of local effects) + L1 norm of centred ALE function.

**SHAP ops**: `import shap` inside function body only. `skipif(shap not installed)` at test level.

**predict() failures**: try/except returning 0.0 for PDP/ALE grid points that fail.

**Result dataclasses** (frozen): each exposes `.summary(top_n=10) -> str`.
- `NativeImportanceResult`: `importances_`, `feature_names_`, `method`
- `PermutationImportanceResult`: `importances_mean_`, `importances_std_`, `feature_names_`, `n_repeats`
- `CondPermutationImportanceResult`: similar + `method="strobl"`
- `PDPImportanceResult`: `importances_`, `feature_names_`, `pdp_values_`, `grid_values_`
- `ALEImportanceResult`: `importances_`, `feature_names_`, `ale_values_`
- `SHAPImportanceResult`: `shap_values_`, `expected_value_`, `feature_names_`, `explainer_type`

All 14 names (8 callables + 6 result types) added to `mf.functions.__all__`.

**OptionDoc updates** (`macroforecast/scaffold/option_docs/l7_a.py`):
- 8 entries updated with `op_page=True`, `op_func_name`, `data_args`, `return_type`, `returns_attrs`.
- New constants: `_L7_RESULT_X_DATA_ARGS`, `_L7_RESULT_XY_DATA_ARGS`.
- `_o()` factory extended to accept op_page / op_func_name / data_args / return_type / returns_attrs.

**Encyclopedia**: 300 -> 308 pages (8 new L7 op pages).

**Tests** (`tests/functions/test_l7_importance.py`): 64 passed, 2 skipped (no-shap guard).
- Bit-exact vs runtime helpers for all 6 non-SHAP ops.
- Family compat: linear ops on tree raise ValueError; tree ops on linear raise ValueError.
- `random_state=None` -> reverse-order deterministic (bit-exact with `_permutation_importance_frame`).
- ALE centering verified; L1 norm non-negative.
- predict-failure -> 0.0 contribution (PDP).
- SHAP: skipif when shap not installed.

**New file**: `macroforecast/functions/importance.py`.

### Cycle 37 -- L4 timeseries + misc family standalone-ization (20 ops)

**New standalone callables** in `mf.functions` (all return a frozen dataclass):

**timeseries.py (14 ops):**
`var_fit(X, y, *, n_lags=1)` -> `VARFitResult`
`bvar_minnesota_fit(X, y, *, n_lags=1, lambda1=0.2)` -> `BVARMinnesotaFitResult`
`bvar_niw_fit(X, y, *, n_lags=1)` -> `BVARNIWFitResult`
`ar_fit(X, y, *, n_lags=1)` -> `ARFitResult`
`far_fit(X, y, *, n_factors=3, n_lags=1)` -> `FARFitResult`
`pcr_fit(X, y, *, n_components=3)` -> `PCRFitResult`
`favar_fit(X, y, *, n_factors=3, n_lags=1)` -> `FAVARFitResult`
`garch11_fit(X, y)` -> `GARCH11FitResult` (requires arch)
`egarch_fit(X, y)` -> `EGARCHFitResult` (requires arch)
`realized_garch_fit(X, y, rv)` -> `RealizedGARCHFitResult` (requires arch)
`ets_fit(X, y)` -> `ETSFitResult`
`theta_fit(X, y)` -> `ThetaFitResult`
`holt_winters_fit(X, y)` -> `HoltWintersFitResult`
`dfm_fit(X, y, *, n_factors=3)` -> `DFMFitResult`

**misc.py (6 ops):**
`svr_linear_fit(X, y, *, C=1.0)` -> `SVRLinearFitResult`
`svr_rbf_fit(X, y, *, C=1.0, gamma="scale")` -> `SVRRBFFitResult`
`svr_poly_fit(X, y, *, C=1.0, degree=3)` -> `SVRPolyFitResult`
`knn_fit(X, y, *, n_neighbors=5)` -> `KNNFitResult`
`kernel_ridge_fit(X, y, *, alpha=1.0, kernel="rbf", gamma=None)` -> `KernelRidgeFitResult`
`mars_fit(X, y)` -> `MARSFitResult` (requires pyearth)

Each result dataclass exposes family-specific attrs + `._model` + `.predict(X)` + `.summary()`.
GARCH results also expose `.predict_variance(h_steps)`. `realized_garch_fit` accepts an
explicit `rv` (realised-variance) Series as third positional argument.

Paradigm (C28 lazy-import): each callable calls `_build_l4_model("<family>", params)` directly.

**New files**: `macroforecast/functions/timeseries.py` (14 ops) + `macroforecast/functions/misc.py` (6 ops).

All 40 names (20 callables + 20 result types) added to `mf.functions.__all__`.

**OptionDoc updates** (`macroforecast/scaffold/option_docs/l4.py`):
- 20 entries updated with `op_page=True`, `op_func_name`, `data_args=_L4_DATA_ARGS`, `return_type`, `returns_attrs`.

**Encyclopedia**: 280 -> 300 pages (20 new L4 op pages).

**Tests**: `tests/functions/test_l4_timeseries.py` (80 passed, 16 skipped arch-guarded),
`tests/functions/test_l4_misc.py` (41 passed, 6 skipped pyearth-guarded).

### Cycle 36 -- L4 deep family standalone-ization (4 ops)

**New standalone callables** in `mf.functions` (all return a frozen dataclass):

`mlp_fit(X, y, *, hidden_layer_sizes=(32,16), max_iter=500, random_state=0)` -> `MLPFitResult`
`lstm_fit(X, y, *, hidden_size=32, n_epochs=50, random_state=0)` -> `LSTMFitResult`
`gru_fit(X, y, *, hidden_size=32, n_epochs=50, random_state=0)` -> `GRUFitResult`
`transformer_fit(X, y, *, hidden_size=32, n_epochs=50, random_state=0)` -> `TransformerFitResult`

Each result dataclass exposes: `.n_params`, `.n_features_in_`, `.hidden_layer_sizes` (MLP) / `.hidden_size` (torch), `.epochs_used`, `.final_loss`, `._model`, `.predict(X)`, `.summary()`.

Paradigm (C28 lazy-import): MLP calls `_build_l4_model("mlp", params)` from `macroforecast.core.runtime` directly (bit-exact, rtol=1e-12). Torch families (LSTM/GRU/Transformer) replicate `_TorchSequenceModel` architecture identically in `_fit_torch_sequence` helper (atol=1e-5). Seeds set via `torch.manual_seed(random_state)` + `np.random.seed(random_state)` before fit. `final_loss` computed via `with torch.no_grad():` forward pass after training.

**New file**: `macroforecast/functions/deep.py` (4 ops).

All 8 names (4 callables + 4 result types) added to `mf.functions.__all__`.

**OptionDoc updates** (`macroforecast/scaffold/option_docs/l4.py`):
- 4 deep family entries updated with `op_page=True`, `op_func_name`, `data_args=_L4_DATA_ARGS`, `return_type`, `returns_attrs`.

**Encyclopedia**: 276 -> 280 pages (4 new L4 deep op pages).

**Tests** (`tests/functions/test_l4_deep_family.py`): see test file.
- Bit-exact MLP vs `_build_l4_model("mlp", params)` direct call (rtol=1e-12).
- atol=1e-5 for torch families (LSTM/GRU/Transformer).
- `.predict()` shape (n,) + dtype float.
- `.summary()` contains model_type, n_features, final_loss.
- Protocol structural conformance (`FitResultBase`).
- Input validation (max_iter, hidden_size, n_epochs, hidden_layer_sizes).
- Namespace wiring (`mf.functions.__all__`).

### Cycle 35 -- L4 tree/ensemble family standalone-ization (6 ops) + C34 backlog

**New standalone callables** in `mf.functions` (all return a frozen dataclass):

`random_forest_fit(X, y, *, n_estimators=200, max_depth=None, min_samples_leaf=1, random_state=0, n_jobs=1)` -> `RandomForestFitResult`
`extra_trees_fit(X, y, *, n_estimators=200, max_depth=None, min_samples_leaf=1, random_state=0, n_jobs=1)` -> `ExtraTreesFitResult`
`gradient_boosting_fit(X, y, *, n_estimators=200, learning_rate=0.1, max_depth=3, random_state=0)` -> `GradientBoostingFitResult`
`xgboost_fit(X, y, *, n_estimators=300, learning_rate=0.1, max_depth=6, subsample=1.0, random_state=0)` -> `XGBoostFitResult`
`lightgbm_fit(X, y, *, n_estimators=300, learning_rate=0.1, max_depth=-1, num_leaves=31, random_state=0)` -> `LightGBMFitResult`
`catboost_fit(X, y, *, n_estimators=300, learning_rate=0.1, max_depth=6, random_state=0)` -> `CatBoostFitResult`

Each result dataclass exposes: `.feature_importances_` (raw per-family importances), `.n_estimators_used`, `._model`, `.predict(X)`, `.summary()`.

Paradigm (C28 lazy-import pattern): each wrapper calls `_build_l4_model("<family>", params)` from `macroforecast.core.runtime` inside the function body -- no formula duplication. `n_jobs` overridden post-construction for RF/ET only. `num_leaves` and `subsample` set via `set_params()` post-construction for LightGBM/XGBoost (not exposed in `_build_l4_model`).

**New file**: `macroforecast/functions/tree.py` (6 ops).

All 6 names added to `mf.functions.__all__`.

**OptionDoc updates** (`macroforecast/scaffold/option_docs/l4.py`):
- 6 tree/ensemble family entries updated with `op_page=True`, `op_func_name`, `data_args=_L4_DATA_ARGS`, `return_type`, `returns_attrs`.

**Encyclopedia**: 270 -> ~276 pages (6 new L4 tree op pages).

**C34 backlog fixes**:
- `macroforecast/functions/clean.py` Notes docstring: replaced stale `.reindex(df_q.index)` description with accurate `other_agg.join(agg)` returning quarterly-indexed frame directly.
- CHANGELOG C34 entry: `iqr.replace(0, pd.NA)` -> `iqr.replace(0, np.nan)` (correctness constraint text).
- CHANGELOG C34 entry: test count corrected `131 passed` -> `142 passed`.

**Tests** (`tests/functions/test_l4_tree_family.py`): 73 passed.
- Bit-exact vs `_build_l4_model("<family>", params)` direct call on RNG-42 100x5 panel.
- `.predict()` correctness vs recipe path.
- `.summary()` content (family name, top-3 features).
- Protocol structural conformance (FitResultBase).
- LightGBM: max_depth=-1 valid; max_depth=0 raises ValueError; max_depth=-2 raises ValueError.
- CatBoost: prediction shape guaranteed 1-D after `.ravel()`.

### Cycle 34 -- L2 clean panel ops standalone-ization (14 ops)

**New standalone callables** in `mf.functions` (all return `pd.DataFrame`):

`iqr_outlier_clean(panel, *, threshold=10.0, action="flag_as_nan")` -> `pd.DataFrame`
`zscore_outlier_clean(panel, *, threshold=3.0, action="flag_as_nan")` -> `pd.DataFrame`
`winsorize_clean(panel, *, lower_quantile=0.01, upper_quantile=0.99)` -> `pd.DataFrame`
`em_factor_impute_clean(panel, *, n_factors=8, max_iter=20, tol=1e-4)` -> `pd.DataFrame`
`em_multivariate_impute_clean(panel, *, max_iter=20, tol=1e-4)` -> `pd.DataFrame`
`mean_impute_clean(panel)` -> `pd.DataFrame`
`forward_fill_clean(panel)` -> `pd.DataFrame`
`linear_interpolate_clean(panel)` -> `pd.DataFrame`
`truncate_to_balanced_clean(panel)` -> `pd.DataFrame`
`drop_unbalanced_series_clean(panel)` -> `pd.DataFrame`
`zero_fill_leading_clean(panel)` -> `pd.DataFrame`
`apply_tcode_transform(panel, tcode_map)` -> `pd.DataFrame`
`freq_align_quarterly_to_monthly_clean(panel, quarterly_columns, *, rule="step_backward")` -> `pd.DataFrame`
`freq_align_monthly_to_quarterly_clean(panel, monthly_columns, *, rule="quarterly_average")` -> `pd.DataFrame`

Paradigm (C29 lazy-import recipe-path): each wrapper imports the runtime helper
inside the function body -- no formula duplication.

**Critical correctness constraints**:
- `iqr_outlier_clean`: `iqr.replace(0, np.nan)` precedes mask computation (zero-IQR columns not flagged).
- `freq_align_quarterly_to_monthly_clean` step_backward: `.bfill().ffill()` order (NOT `.ffill().bfill()`).
- `zero_fill_leading_clean`: fills ALL NaN with 0 (name misleading but matches runtime).
- `em_multivariate_impute_clean` passes `n_factors=None` -> `rank = min(T, K) // 2`.

**New file**: `macroforecast/functions/clean.py` (14 ops).

All 14 names added to `mf.functions.__all__`.

**OptionDoc updates** (`macroforecast/scaffold/option_docs/l2.py`):
- `_e()` helper extended with `op_page`, `op_func_name`, `data_args`, `return_type`, `returns_attrs` kwargs.
- `_L2_PANEL_DATA_ARG` and `_L2_PANEL_TCODE_DATA_ARGS` shared data-arg constants added.
- 14 OptionDoc entries updated with `op_page=True`, `op_func_name`, `data_args=_L2_PANEL_DATA_ARG`,
  `return_type="pd.DataFrame"`.

**Encyclopedia**: 256 -> 270 pages (14 new L2 op pages).

**Tests** (`tests/functions/test_l2_clean.py`): 142 passed.
- Bit-exact vs runtime for all applicable ops.
- Correctness: shape, column/index preservation, outlier flagging, imputation fill, tcode transforms.
- Input validation: empty panel, threshold ranges, invalid actions/rules/tcodes.
- Namespace wiring: all 14 in `mf.functions.__all__`.

### Cycle 33 -- L3 final B1 transforms standalone (8 ops) + C32 backlog

**New standalone callables** in `mf.functions` (all return `pd.DataFrame`):

`sparse_pca_transform(panel, *, n_components=8)` -> `pd.DataFrame`
`sparse_pca_chen_rohe_transform(panel, *, n_components=4, zeta=0.0, max_iter=200, var_innovations=False, random_state=0)` -> `pd.DataFrame`
`varimax_transform(panel)` -> `pd.DataFrame`
`random_projection_transform(panel, *, n_components=8)` -> `pd.DataFrame`
`kernel_features_transform(panel, *, kind="rbf", gamma=1.0)` -> `pd.DataFrame`
`nystroem_transform(panel, *, n_components=32)` -> `pd.DataFrame`
`time_trend_transform(panel)` -> `pd.DataFrame`
`holiday_transform(panel)` -> `pd.DataFrame`

Paradigm (C29 lazy-import recipe-path): each wrapper imports the runtime helper
inside the function body -- no formula duplication. `time_trend_transform` uses
inline `np.arange(1, T+1)` (no runtime helper required for trivial generation).

`kernel_features_transform` returns a T_clean x T_clean Gram matrix (exact kernel,
not an approximation). See function Notes for large-panel guidance.

`holiday_transform` delegates to `_holiday_indicator` which uses
`USFederalHolidayCalendar`. Known runtime bug: `pd.tseries.offsets.USFederalHolidayCalendar`
was moved to `pd.tseries.holiday` in pandas >= 2.x; DatetimeIndex path raises
`AttributeError`. Non-DatetimeIndex path returns all zeros correctly (tracked in mailbox.md).

All 8 names added to `mf.functions.__all__`.

**OptionDoc updates** (`macroforecast/scaffold/option_docs/l3.py`):
- 8 canonical ops set to `op_page=True` with `op_func_name`, `data_args`, `return_type`, `returns_attrs`.
- 4 alias entries set to `op_page=False` (orphan pages deleted); catalog entries in `op.md` redirect readers to canonical pages:
  `varimax_rotation` -> `varimax_transform`, `kernel` -> `kernel_features_transform`,
  `nystroem_features` -> `nystroem_transform`, `polynomial` -> `polynomial_expansion_transform`.

**Encyclopedia**: 248 -> 256 pages (8 new canonical, 4 alias pages skipped).

**C32 backlog fixes**:
- **NOTE-A** (BLK-4 PLS clamp): `min(T_clean-1, K_clean-1)` -> `min(T_clean-1, K_clean)`.
  Test assertion updated: `<= 4` -> `== 5` for RNG-42 50x5 panel.
- **NOTE-B** (CHANGELOG drift): C32 entry test count corrected: 59 -> 84.

**Tests** (`tests/functions/test_l3_final_b1_transforms.py`): 66 passed, 3 xfailed.
- 66 passing: bit-exact vs runtime for all 8 ops (RNG-42, `rtol=1e-12, atol=1e-14`).
- Correctness: shape, column names, index preservation, input validation, namespace wiring.
- 3 xfailed: `TestHolidayTransform` DatetimeIndex tests expose pre-existing
  `_holiday_indicator` runtime bug (`pd.tseries.offsets` -> should be `pd.tseries.holiday`).

### Cycle 32 -- L3 supervised/mixed transforms standalone (6 ops)

**New standalone callables** in `mf.functions` (all return `pd.DataFrame`):

`scaled_pca_transform(panel, target, *, n_components=3)` -> `pd.DataFrame`
`supervised_pca_transform(panel, target, *, n_components=3)` -> `pd.DataFrame`
`partial_least_squares_transform(panel, target, *, n_components=3)` -> `pd.DataFrame`
`sliced_inverse_regression_transform(panel, target, *, n_components=3, n_slices=10)` -> `pd.DataFrame`
`dfm_transform(panel, *, n_factors=3)` -> `pd.DataFrame`
`feature_selection_transform(panel, target=None, *, n_features=0.5, method="variance")` -> `pd.DataFrame`

Paradigm (C29 lazy-import recipe-path): each wrapper imports the runtime helper
inside the function body -- no formula duplication.

Target alignment validation (supervised ops):
- `target.index.intersection(panel.index).empty` raises `ValueError` with message
  "no common index values" for `scaled_pca_transform`, `supervised_pca_transform`,
  `partial_least_squares_transform`, `sliced_inverse_regression_transform`.
- `feature_selection_transform`: `method in {"correlation", "lasso"}` and `target is None`
  raises `ValueError("requires target")`.

`dfm_transform` is fully unsupervised (no target argument). `feature_selection_transform`
accepts `target=None` (optional) and raises only when a supervised method needs it.

All 6 names added to `mf.functions.__all__`.

**OptionDoc updates** (`macroforecast/scaffold/option_docs/l3.py`):
- Added `_L3_SUPERVISED_DATA_ARGS` constant (panel_arg + required target_arg) for 4 supervised ops.
- Added `_L3_OPTIONAL_TARGET_DATA_ARGS` constant (panel_arg + optional_target_arg) for `feature_selection`.
- `dfm` uses existing `_L3_PANEL_DATA_ARG`-only data_args.
- All 6 ops set to `op_page=True` with `op_func_name` and full `data_args` + `return_type` population.

**Encyclopedia**: 242 → 248 pages (6 new L3 op pages).

**Tests** (`tests/functions/test_l3_supervised_transforms.py`): 84 tests, all pass (59 original + 25 from C32-fixup BLK items: BLK-3 SIR scaling_method / BLK-4 PLS clamp / BLK-5 DFM reindex / BLK-6 supervised_pca q-clamp).
- Bit-exact vs runtime helper for all 6 ops (RNG-42, `rtol=1e-12, atol=1e-14`).
- Target alignment `ValueError` on disjoint indices (4 supervised ops).
- `feature_selection_transform` method/target combo validation.
- `dfm_transform` no-target signature verified.
- Namespace wiring (`mf.functions.<name>`) for all 6.

### Cycle 31 -- L3 advanced transforms standalone (12 ops) + C30 backlog fixes

**New standalone callables** in `mf.functions` (all return `pd.DataFrame`):

`hp_filter_transform(panel, *, lambda_=1600)` -> `pd.DataFrame`
`hamilton_filter_transform(panel, *, h=8, p=4)` -> `pd.DataFrame`
`savitzky_golay_transform(panel, *, window=7, polyorder=3)` -> `pd.DataFrame`
`polynomial_expansion_transform(panel, *, degree=2)` -> `pd.DataFrame`
`interaction_terms_transform(panel)` -> `pd.DataFrame`
`pca_transform(panel, *, n_components=3)` -> `pd.DataFrame`
`maf_per_variable_pca_transform(panel, *, n_lags=12, n_components_per_var=2)` -> `pd.DataFrame`
`adaptive_ma_rf_transform(panel, *, n_estimators=100, min_samples_leaf=40, sided="two", random_state=0)` -> `pd.DataFrame`
`wavelet_transform(panel, *, wavelet="db4", n_levels=3)` -> `pd.DataFrame`
`fourier_transform(panel, *, n_terms=4, period=12)` -> `pd.DataFrame`
`asymmetric_trim_transform(panel)` -> `pd.DataFrame`
`season_dummy_transform(panel, *, season="quarter")` -> `pd.DataFrame`

Parameter alignment with runtime helpers (C29 paradigm: lazy import, no duplicate formulas):
- `hp_filter_transform`: `lambda_` maps to runtime `lam`
- `hamilton_filter_transform`: `h`/`p` map to runtime `n_horizon`/`n_lags`
- `savitzky_golay_transform`: `window` maps to runtime `window_length`
- `adaptive_ma_rf_transform`: exposes `n_estimators` + `min_samples_leaf` (no `max_depth` -- not in runtime)
- `wavelet_transform`: `wavelet` accepted for API consistency; runtime uses rolling-mean approximation

All 12 names added to `mf.functions.__all__`.

**OptionDoc updates** (`macroforecast/scaffold/option_docs/l3.py`):
- All 12 L3 advanced transform ops set to `op_page=True` with full parameter documentation

**Encyclopedia**: 12 new per-op pages under `docs/encyclopedia/l3/op/`
(hp_filter, hamilton_filter, savitzky_golay_filter, polynomial_expansion,
interaction, pca, maf_per_variable_pca, adaptive_ma_rf, wavelet,
fourier, asymmetric_trim, season_dummy). Total: 242 pages (was 230).

**C30 backlog fixes**:
- NOTE-1 (`log_diff_transform`): Added 2 sentences to Notes section documenting
  `np.log` vs recipe `pd.NA` guard divergence on non-positive cells.
  Identical pattern to `log_transform`. Numerics unchanged.
- NOTE-2 (`scale_transform`): Changed `raise NotImplementedError(...)` to
  `raise ValueError(f"Unknown method: {method!r}. Expected zscore/robust/minmax/winsorize/quantile.")`
  to match project convention (invalid user input = ValueError).
  Updated `tests/functions/test_l3_basic_transforms.py` assertion accordingly.

**Tests**: `tests/functions/test_l3_advanced_transforms.py` -- 103 tests, all pass.
Bit-exact assertions (rtol=1e-12, atol=1e-14) on 10 deterministic ops;
structural checks for RF-based op (adaptive_ma_rf_transform).
NOTE-1 docstring guard + NOTE-2 ValueError coverage included.


### Cycle 30 -- L3 basic transforms standalone (10 ops) + C29 docstring fix

**New standalone callables** in `mf.functions` (all return `pd.DataFrame`):

`diff_transform(panel, *, periods=1)` -> `pd.DataFrame`
`log_transform(panel)` -> `pd.DataFrame`
`log_diff_transform(panel, *, periods=1)` -> `pd.DataFrame`
`pct_change_transform(panel, *, periods=1)` -> `pd.DataFrame`
`cumsum_transform(panel)` -> `pd.DataFrame`
`ma_window_transform(panel, *, window=3)` -> `pd.DataFrame`
`lag_matrix(panel, *, n_lag=4, include_contemporaneous=False)` -> `pd.DataFrame`
`seasonal_lag_matrix(panel, *, seasonal_period=12, n_seasonal_lags=1)` -> `pd.DataFrame`
`ma_increasing_order_transform(panel, *, max_order=12)` -> `pd.DataFrame`
`scale_transform(panel, *, method="zscore")` -> `pd.DataFrame`

Each callable lazy-imports the corresponding runtime primitive from `macroforecast.core.runtime`
(`_as_frame`, `_diff_like`, `_pct_change_like`, `_lagged_predictors`,
`_seasonal_lagged_predictors`, `_ma_increasing_order`, `_scale_frame`)
to ensure bit-exact numerical results with the recipe-path dispatch.

All 10 names added to `mf.functions.__all__`.

**OptionDoc updates** (`macroforecast/scaffold/option_docs/l3.py`):
- Extended `_o()` helper to accept `op_page`, `op_func_name`, `data_args`, `return_type`, `returns_attrs`
- Added shared `_L3_PANEL_DATA_ARG` constant for the common `panel: pd.DataFrame` argument doc
- All 10 L3 basic transform ops set to `op_page=True` with full parameter documentation
- Updated import to include `ParameterDoc` and `REQUIRED`

**Encyclopedia**: 10 new per-op pages under `docs/encyclopedia/l3/op/`
(diff, log, log_diff, pct_change, cumsum, ma_window, lag,
seasonal_lag, ma_increasing_order, scale). Total: 230 pages (was 220).

**C29 backlog fix** (`macroforecast/functions/tests.py`): removed `_l6_dmp_multi_horizon`
from module docstring helper list (it is a private runtime function, not a helper exposed
by the standalone module).

**Tests**: `tests/functions/test_l3_basic_transforms.py` -- 77 tests, all pass.
Bit-exact assertions (rtol=1e-12, atol=1e-14) on all 10 ops.
Coverage: shape, NaN pattern, runtime parity, edge cases, namespace wiring.


### Cycle 29 — L6 statistical tests bulk standalone-ization (7 ops)

**New standalone callables** in `mf.functions` (all return a frozen dataclass with `.summary()`):

`dm_test(loss_a, loss_b, *, horizon, correction, kernel)` -> `DMTestResult`
`gw_test(loss_a, loss_b, *, horizon, correction, kernel)` -> `GWTestResult`
`dmp_test(loss_differentials, *, kernel)` -> `DMPTestResult`
`hn_test(e_a, e_b, *, horizon, kernel, small_sample)` -> `HNTestResult`
`cw_test(loss_small, loss_large, f_small, f_large, *, horizon, kernel)` -> `CWTestResult`
`enc_new_test(loss_small, loss_large, *, horizon, kernel)` -> `EncNewTestResult`
`enc_t_test(loss_small, loss_large, *, horizon, kernel)` -> `EncTTestResult`

Each callable lazy-imports the corresponding runtime primitive from `macroforecast.core.runtime`
(`_diebold_mariano_test`, `_harvey_newbold_test`, `_l6_dmp_multi_horizon`, `_long_run_variance`)
to ensure bit-exact numerical results with the recipe-path dispatch.

All 14 names (7 result classes + 7 callables) added to `mf.functions.__all__`.

**OptionDoc updates** (`macroforecast/scaffold/option_docs/l6.py`):
- Extended `_e()` helper to accept `op_page`, `op_func_name`, `data_args`, `return_type`, `returns_attrs`
- Shared `data_args` constants: `_L6_LOSS_PAIR_DATA_ARGS`, `_L6_DMP_DATA_ARGS`, `_L6_HN_DATA_ARGS`, `_L6_CW_DATA_ARGS`, `_L6_ENC_DATA_ARGS`
- All 7 L6 ops (4 L6.A + 3 L6.B) set to `op_page=True` with full parameter documentation
- Added L6.B nested_test section (clark_west, enc_new, enc_t, multi) which was previously empty

**L6.B nested_test axis fix** (`macroforecast/scaffold/introspect.py`):
- Added `L6_B_nested` `AxisInfo` to `_build_l6_fallback()` so encyclopedia renders clark_west, enc_new, enc_t, multi option pages
- Result: 220 encyclopedia pages (was 217; +3 nested_test option pages)

**Tests**: `tests/functions/test_l6_tests.py` — 60 tests, all pass.
Bit-exact assertions (rtol=1e-12, atol=1e-14) on stat and pvalue for all 7 ops.


### Cycle 28 -- L4 linear family standalone-ization (~7 ops)

**New standalone callables** in `mf.functions` (all return a `FitResultBase`-conformant frozen dataclass):

`ols_fit` -> `OLSFitResult`
`lasso_fit` -> `LassoFitResult`
`elastic_net_fit` -> `ElasticNetFitResult`
`lasso_path_fit` -> `LassoPathFitResult`
`bayesian_ridge_fit` -> `BayesianRidgeFitResult`
`huber_fit` -> `HuberFitResult`
`glmboost_fit` -> `GLMBoostFitResult`

All callables produce bit-exact output with the L4 recipe-path computation
(`_build_l4_model` in `macroforecast.core.runtime`).

Encyclopedia pages auto-emitted for all 7 new ops.

Ridge sub-axis variants (NonNeg, TwoStageRandomWalk, ShrinkToTarget, FusedDifference)
confirmed as sub-modes of the existing `ridge_fit`; no new callables needed.

**Backlog fixes**:
- B27-1: PT denom guard aligned to runtime behavior (clamped, not early-NaN).
- B27-2: `mse_reduction` OptionDoc description corrected (absolute difference, not ratio).


### Cycle 27 -- L5 metrics bulk standalone-ization (~13 ops)

**New standalone callables** in `mf.functions` (all return `float`):

Point metrics: `mse`, `rmse`, `mae`, `medae`, `mape`
Relative metrics: `relative_mse`, `relative_mae`, `mse_reduction`, `r2_oos`
Interval/coverage metrics: `interval_score`, `coverage_rate`
Direction metrics: `success_ratio`, `pesaran_timmermann_metric`

All callables produce results bit-exact with the L5 recipe-path computation
where a recipe path exists; `mape`, `interval_score`, and `coverage_rate` are
new canonical implementations (no prior recipe-path existed).

Encyclopedia pages auto-emitted for all 13 new ops plus `theil_u2` (previously
missing its per-op page).

Deferred: `log_score`, `crps` (B2 -- require predictive density objects).

**Note on `mse_reduction`**: the standalone matches the runtime computation
(absolute difference `benchmark_MSE - model_MSE`), not the ratio-based
description in earlier docs. The doc-string flags this explicitly.



## [0.9.1] -- 2026-05-18 -- "audit fixes C12-C22 + paradigm foundation (REQUIRED sentinel, FitResultBase)"

### Cycle 26 — Paradigm Foundation: REQUIRED sentinel + FitResultBase + per-op page v2

#### Breaking Changes
- `ParameterDoc.default` now defaults to `REQUIRED` sentinel instead of
  `None`.  **BREAKING (internal)**: code instantiating `ParameterDoc`
  without `default=` previously received `default=None`; it now receives
  `default=REQUIRED`.  All existing callers in the codebase use explicit
  `default=` and are unaffected.  External code must add `default=None`
  for genuine-None defaults.

#### Added
- `REQUIRED` singleton sentinel exported from
  `macroforecast.scaffold.option_docs.types`.
  `from macroforecast.scaffold.option_docs.types import REQUIRED`
- `OptionDoc.data_args: tuple[ParameterDoc, ...]` — positional data-input
  parameters for per-op encyclopedia pages. Defaults to `()`.
- `OptionDoc.return_type: str` — return-type annotation rendered as
  `-> {return_type}` in per-op signature block. Defaults to `""`.
- `OptionDoc.returns_attrs: tuple[tuple[str, str, str], ...]` — return-value
  attribute table for `## Returns` section. Defaults to `()`.
- `macroforecast.functions.FitResultBase` — `@runtime_checkable` Protocol
  declaring `.summary() -> str` and `.predict(X) -> np.ndarray` as the
  structural contract for all L4 fit-result objects.
- `RidgeFitResult.summary()` — minimal statsmodels-style text table
  showing alpha, predictor count, intercept, and coefficient vector.
  Std errors deferred to Cycle 28.

#### Docs
- `docs/encyclopedia/l4/family/ridge.md` signature now shows `X`, `y`
  positional args and `-> RidgeFitResult` return type; `## Returns`
  table lists all result attributes.
- `docs/encyclopedia/l5/point_metrics/theil_u1.md` signature now shows
  `-> float` return type; `## Returns` section emitted.
- Per-op page renderer (`_render_op_page`) updated: data_args rendered
  before `*,`; `## Returns` section added; `—` default in tables uses
  REQUIRED sentinel identity (not `is None`).

### Cycle 22 — POC: per-op page + mf.functions namespace (v0.10 candidate)

- OptionDoc.op_page field: render_encyclopedia emits a separate per-op Markdown page for each op that declares one.
- macroforecast/functions/ module: ridge_fit (RidgeFitResult, sklearn-style) + theil_u1 + theil_u2 standalone wrappers.
- l4/family/ridge.md + l5/point_metrics/theil_u1.md per-op pages with Function signature + Parameters sections.
- mf.functions exposed via _LAZY_MODULES (Cycle 16 K-1 lazy-namespace pattern).
- 17 new unit tests in tests/functions/ + tests/tools/docgen/test_op_page_render.py.
- Pattern validates for L3/L4/L5/L6/L7 migration in subsequent cycles.
- **v0.10.0 candidate**
- **C22 known limitations** (deferred to C23 pre-expansion fixup):
  - `ParameterDoc.default=None` sentinel conflicts with parameters that legitimately default to None (vol_model, random_state). Per-op page renderer misclassifies as required positional.
  - Rendered `## Function signature` omits data positional args (X, y / y_true, y_pred); not copy-pasteable.
  - Both flagged by C22 reviewer (a2c0179f89ea5a688). Bit-exact POC correctness unaffected.


### Cycle 21 -- L1.E + L1.F + L1.G batch -- L1 audit complete

- **C21-E** L1.E sample window: `sample_start_rule=fixed_date` -> `sample_start_date` ParameterDoc (partial-ISO acceptance per C12 F-P0-1); `sample_end_rule=fixed_date` -> `sample_end_date` ParameterDoc (ISO date, must-be-gte-start constraint documented).
- **C21-F** L1.F horizon_set: `single` -> `target_horizons` ParameterDoc (optional, defaults to [1]); `custom_list` -> `target_horizons` ParameterDoc (required, non-empty list); `range_up_to_h` -> `max_horizon` ParameterDoc (required positive int; expands to [1..max_horizon]). `max_horizon` added to `_KNOWN_LEAF_CONFIG_KEYS["1_data"]`.
- **C21-G** L1.G regime_definition: `external_user_provided` -> 3 ParameterDocs (regime_indicator_path XOR regime_dates_list, plus n_regimes); `estimated_markov_switching` -> n_regimes ParameterDoc; `estimated_threshold` -> threshold_variable (required) + n_thresholds ParameterDocs; `estimated_structural_break` -> max_breaks + break_ic_criterion ParameterDocs. 10 new keys added to `_KNOWN_LEAF_CONFIG_KEYS["1_data"]` (regime_indicator_path, regime_dates_list, n_regimes, threshold_variable, n_thresholds, max_breaks, break_ic_criterion, regime_rolling_window_size, block_recompute_interval, max_horizon).
- **L1 audit complete**: all 7 sub-layers (L1.A-L1.G) are Tier-1 with ParameterDoc populated for conditional leaf_config keys. Module docstring updated to reflect completion.
- 22 new unit tests in `tests/tools/docgen/test_l1efg_parameter_doc.py`.

### Cycle 20 — L1.D Geography encyclopedia complete

- **C20** L1.D 6 axes: ParameterDoc populated for 6 conditional leaf_config keys (target_states, predictor_states, sd_states, sd_variables, sd_state_group_members/sd_state_groups, sd_variable_group_members/sd_variable_groups). `_KNOWN_LEAF_CONFIG_KEYS["1_data"]` extended with 8 keys. Docstring updated to confirm L1.D Tier-1 coverage (all 6 axes).
- **C20 follow-up**: Added `target_state` (singular, target_geography_scope=single_state) to ParameterDoc + `_KNOWN_LEAF_CONFIG_KEYS["1_data"]`. Pre-existing gap surfaced by reviewer grep-audit.

### Cycle 19 — L1.C Predictor Universe encyclopedia complete

- **P-1** Fixed outdated docstring at top of `option_docs/l1.py` falsely claiming L1.C axes carry placeholder entries — all 8 axes are Tier-1 (reviewed 2026-05-05). Tier-1-complete sub-layer list updated to reflect actual coverage through Cycle 19.
- **P-2** ParameterDoc populated for L1.C 8 axes: 6 conditional leaf_config keys documented (variable_universe_columns, fixed_lag_periods, release_lag_per_series, outlier_iqr_threshold, zscore_threshold_value, winsorize_quantiles). Options without conditional leaf_config carry explicit `parameters=()`.
- **P-3** Cross-references added in descriptions for `raw_outlier_policy` options pointing to L2 `outlier_policy`/`outlier_action` (same surface, different stage: raw vs post-tcode), and for `raw_missing_policy/preserve_raw_missing` pointing to L2 `imputation_policy`. `_KNOWN_LEAF_CONFIG_KEYS["1_data"]` extended with the 6 conditional keys to suppress false unknown-key warnings.
- **C19 follow-up** Added 2 missed leaf_config keys per reviewer cross-model audit: `x_imputation` (missing_availability/impute_predictors_only) + `raw_x_imputation` (raw_missing_policy/impute_raw_predictors). Both added to ParameterDoc + `_KNOWN_LEAF_CONFIG_KEYS["1_data"]`. Also corrected `fixed_lag_periods` constraint (was "required", actually optional default 0).

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


#### Cycle 18 — L1.B Target definition encyclopedia complete

- **C18-A** `target_structure`: removed `multi_series_target` ghost option from `TargetStructure` Literal, validator set, and runtime canonicalization. Code had silent alias to `multi_target`; per user-driven L1.B review, alias removed for clarity. **BREAKING**: recipes using `target_structure: multi_series_target` now fail. Migration: switch to `multi_target`.
- **C18-B** `target_structure` ParameterDoc populated: `single_target` → `target: str`, `multi_target` → `targets: list[str]`.

#### Cycle 17.5 — fred_sd_frequency_policy reject/require enforcement (BREAKING)

- **C17.5** L1.A `fred_sd_frequency_policy`: `reject_mixed_known_frequency` and `require_single_known_frequency` now actually raise ValueError at L1 validation (was: silently aligned). Honors OptionDoc per LOW-A2 review finding. BREAKING for recipes that previously relied on silent alignment under reject/require policies; use `allow_mixed_frequency` to preserve old behavior.

#### Cycle 17 — L1.A Source Selection encyclopedia complete

- **O-1** Tier-2 -> Tier-1 OptionDoc promotion for `frequency` (2 ops + `'derived'` sentinel), `information_set_type` (2 ops), `vintage_policy/real_time_alfred` (Cycle 14 K-4 hard-reject documented with ALFRED future intent), `fred_sd_frequency_policy` (4 ops). 10 entries marked last_reviewed=2026-05-16.
- **O-2** ParameterDoc population for L1.A custom_source_policy: `custom_panel_only` gains 3 ParameterDoc entries (`custom_source_path`, `custom_panel_inline`, `custom_panel_records` -- mutually exclusive, required); `official_plus_custom` gains 2 entries (`custom_source_path`, `custom_merge_rule` -- both required; merge rule enum: `left_join` / `inner_join` / `outer_join`). `_KNOWN_LEAF_CONFIG_KEYS["1_data"]` extended with `custom_merge_rule`. All other L1.A options carry `parameters=()` (no conditional leaf_config). Encyclopedia regenerated (190 pages, no new pages -- same axis count). L1.B-L1.G ParameterDoc population deferred to next cycles.

#### Cycle 16 — L0 encyclopedia docs/code sync + Parameters section

- **N-1** L0 docs sync: random_seed default 42 (was: "default 0" mistext); removed phantom `n_workers_inner` examples; renamed `parallel_unit` description from "sub-axis" to "conditional leaf_config key" for consistency with implementation.
- **N-2** L0 `parallel_unit: cells` now operational. Was previously documented but rejected by validator (`PARALLEL_UNIT_OPTIONS` enum). Cell-level ProcessPoolExecutor honors deterministic per-cell seed schedule.
- **N-3** Encyclopedia structure: `OptionDoc` and `AxisSpec` now carry a `parameters: tuple[ParameterDoc, ...]` field; encyclopedia pages render `**Parameters**` table when non-empty. Distinguishes axis-as-categorical-switch from option-as-function-with-params. L0 3 axes populated (reproducibility_mode/seeded_reproducible: random_seed; compute_mode/parallel: parallel_unit + n_workers); L1-L8 will be filled in subsequent docs cycles. Validator `_KNOWN_LEAF_CONFIG_KEYS["0_meta"]` gains `parallel_unit` so L1-3 unknown-key warning no longer fires on legitimate L0 `fixed_axes` leaf_config.

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
* **Design Part 2** -- L2-L4 construction layer design documented.
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
* `tests/tools/docgen/test_render_encyclopedia.py` -- 11 tests covering
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
