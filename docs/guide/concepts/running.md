# Running

[Back to User Guide](../index.md)

`macroforecast.forecasting.run` is the atomic forecasting function. It accepts
one model, one data source, and a `WindowSpec`, then iterates over test origins
and returns a `ForecastResult`. For each origin it: applies preprocessing to the
estimation-window panel; builds features and targets from those rows; selects
hyperparameters via the validation window; fits the model; and generates
predictions for the test horizon.

`macroforecast.pipeline.run_pipeline` wraps `run` into a full POOS evaluation.
It enumerates every (arm, target, horizon) cell, calls `run` for each, collects
the master forecast frame, evaluates every contender against the benchmark with
relative MSE, DM/CW, and the Model Confidence Set, and returns a
`PipelineReport`.

## Forecast policies

The `forecast_policy` argument to `run` (and the policy resolved from a
`TargetSpec` t-code in `run_pipeline`) controls how h-step forecasts are built:

- **direct** (`forecast_policy="direct"`): fit one model for each horizon h
  separately, using `y[t+h]` as the target. The simplest and most common choice.
- **direct_average** (`forecast_policy="direct_average"`): the forecast object is
  the h-period cumulation (average) of the stationary transform, not the raw
  single-period value. This is the standard convention for growth-rate series
  (t-codes 2, 3, 5, 6, 7 in FRED-MD/QD) and matches how practitioners report
  average inflation or average growth over the horizon.
- **path_average** (`forecast_policy="path_average"`): fit h step-specific
  models, where step s forecasts the one-period object realized at t+s from
  information available at the origin, then average the h step forecasts. This is
  a direct multi-step design, not an iterated one; iterating a single model
  forward instead is the separate `recursive` policy.
- **recursive** (`forecast_policy="recursive"`, code alias `"iterated"`): fit
  one one-step-ahead model, then roll it forward h times, feeding each step's
  own prediction back in as the next step's lagged input (the textbook
  "iterated" multi-step forecast). Unlike `path_average`, a later step's
  forecast depends on an earlier step's prediction rather than only on
  origin-available data. See
  [`future_feature_policy`](../../reference/forecasting.md) for how exogenous
  (non-target) predictors are rolled forward under this policy.

At horizon 1, `direct_average` and `path_average` are the same forecast by
construction (averaging over a single step is that step), so the two policies
produce identical predictions there. They diverge only for h greater than 1,
where the h-period-average target and the averaged one-step path are genuinely
different objects. This holds across every model, including the
information-criterion autoregressions (`ar`, `far`), whose order is selected by
BIC/AIC on the same sample under both policies.

### Textbook mapping

`macroforecast`'s policy names do not always match the vocabulary of a
forecasting textbook. The table below lines them up:

| Textbook term | `forecast_policy` | What it does |
| --- | --- | --- |
| direct | `"direct"` | One model per horizon h, fit directly on the h-period-ahead value. |
| iterated / recursive | `"recursive"` (code alias `"iterated"`) | One one-step model, rolled forward h times, each step's prediction feeding the next. |
| direct, h-period-average object | `"direct_average"` | The direct idea, but the forecast object is the h-period average of the stationary transform. |
| h-average of h one-step models | `"path_average"` | h independent step-specific one-step-ahead models (never iterated), averaged. |

The `*_average` variants (`direct_average`, `path_average`) are both h-average
forecast objects; they differ in whether one model is fit on the h-period-average
target directly (`direct_average`) or h one-step models are averaged after the
fact (`path_average`). Neither feeds a prediction back into the next step's
inputs -- only `recursive` does that.

The t-code to policy mapping is documented in the
[Pipeline reference](../../reference/pipeline.md).

T-codes 3 and 6 describe second-difference series, but the forecast engine's
target transforms stop at `change` and `log_growth`. By default,
`resolve_target` therefore follows the package convention of forecasting the
corresponding first-difference object. Set `reduce_i2=False` on either
`TargetSpec` or `resolve_target` to refuse that substitution. Resolution then
raises an actionable error unless the target supplies an explicit `transform`
or `tcode_map` supplies an entry for the active code. A false value on either
surface wins; it cannot be overridden by a true value on the other surface.

Recursive custom models need extra care with exogenous predictors. A custom
supervised model with non-target features under `forecast_policy="recursive"`
must only use predictors that are genuinely available at each recursive step
(for example, lagged values). `pipeline_spec()` warns when it can see a custom
supervised model, a recursive target policy, and exogenous features in the same
arm.

## Parallel custom code

`pipeline_spec(..., n_jobs>1)` runs cells in worker processes. Any custom model,
feature, preprocessing, policy, or model-selection object carried by an arm must
therefore be pickleable. Define custom callables at module scope, or keep
`n_jobs=1` for notebook-local closures. The runner preflights these arm objects
before dispatch and raises an actionable `ValueError` instead of a raw
`PicklingError` from a worker pool.

## Key Callables

`mf.forecasting.run` executes one (model, data, window) cell and returns a
`ForecastResult`.

`mf.pipeline.run_pipeline` executes a full `PipelineSpec` and returns a
`PipelineReport`.

```python
import macroforecast as mf
from macroforecast.pipeline import pipeline_spec, run_pipeline, Arm, EvalSpec, TargetSpec

# Low-level: run one model for one target.
result = mf.forecasting.run(
    data_spec,
    model="ar",
    window=mf.window.from_cutoffs(test_start="1985-01-01", horizon=1),
    forecast_policy="direct",
    target="INDPRO",
    horizon=1,
)
forecasts_df = result.to_frame()

# High-level: run the full pipeline with multiple arms and automatic evaluation.
spec = pipeline_spec(
    data=bundle,
    targets=[TargetSpec(name="INDPRO")],
    horizons=[1, 3, 6, 12],
    window=mf.window.from_cutoffs(test_start="1985-01-01"),
    arms=[
        Arm(name="AR", model="ar", is_benchmark=True),
        Arm(name="RF", model="random_forest",
            preprocessing=mf.preprocessing.preprocess_spec(transform="official"),
            features=mf.feature_engineering.feature_spec(
                target="INDPRO",
                predictors="all",
                lags=None,
                feature_steps=[mf.feature_engineering.marx_step(name="MARX_X", max_lag=12)],
            )),
    ],
    evaluation=EvalSpec(benchmark="AR"),
)
report = run_pipeline(spec)
```

For a runnable end-to-end example, see the single-forecast and full-study
snippets in [Getting Started](../getting_started.md) and the step-by-step
pipeline in the [Replication Gallery](../gallery.md#a-complete-pipeline-step-by-step).

## Seeds, parallelism, and model storage

`pipeline_spec(..., seed=...)` is the run-level reproducibility knob for the
pipeline. During `run_pipeline(spec)`, it temporarily becomes the active
`mf.configure(random_seed=...)` value, so model-owned random search and parallel
workers see the same seed. Stochastic model fits derive a stable per-arm
`random_state` from `(seed, arm name)` only when the arm did not explicitly pass
`random_state` in `params`; explicit model params win. The report records these
effective values under `report.provenance["effective_seeds"]`.

`n_jobs="auto"` resolves to a concrete cell-worker count plus a per-worker model
thread budget. On Linux it honors CPU affinity; on macOS and Windows it falls
back to `os.cpu_count()`. Parallel workers cap common BLAS/OpenMP thread
environment variables before running forecast cells and receive the data payload
once per worker rather than once per cell.

Pipeline model persistence is opt-in. `pipeline_spec(...)` defaults to
`save_models=False`; pass `save_models=True` and `model_store=...` only when you
need fitted-model pickles. Large projected stores warn before execution. Remove
old model fits with `mf.pipeline.purge_model_store(...)`. The lower-level
`mf.forecasting.run(...)` keeps its historical `save_models=True` default.

## Incremental horse races

Long paper projects often grow a model comparison over many months. Use
`pipeline_spec(..., result_store="results/cells", preprocessing_cache_dir="results/prep")`
when you expect to add arms later:

```python
spec = pipeline_spec(
    data=bundle,
    targets=[TargetSpec("INDPRO")],
    horizons=[1, 3, 12],
    window=window,
    arms=[
        Arm("AR", model="ar", features=features),
        Arm("RF", model="random_forest", features=features),
    ],
    evaluation=EvalSpec(benchmark="AR"),
    result_store="cache/result_cells",
    preprocessing_cache_dir="cache/preprocessing",
)
first = run_pipeline(spec)

later = pipeline_spec(
    data=bundle,
    targets=[TargetSpec("INDPRO")],
    horizons=[1, 3, 12],
    window=window,
    arms=[
        Arm("AR", model="ar", features=features),
        Arm("RF", model="random_forest", features=features),
        Arm("GBM", model="gradient_boosting", features=features),
    ],
    evaluation=EvalSpec(benchmark="AR"),
    result_store="cache/result_cells",
    preprocessing_cache_dir="cache/preprocessing",
)
second = run_pipeline(later)
```

Every contender added this way needs its own name. Arm names and combination
names are one name space -- both are contenders in the same evaluation -- so
`pipeline_spec` rejects a combination named after an arm, or two combinations
named alike, at build time rather than at run time. It also rejects a repeated
horizon (compared after normalization to integers, so `1` and `1.0` are one
horizon) and two targets that resolve to the same public name, even when their
transform or policy differs. None of these are deduplicated for you: a repeated
label is a specification error, and the tables downstream are keyed by that
label alone. Valid input keeps its order.

The second run reuses the stored `(target, horizon, arm)` cells for `AR` and `RF`
and computes only `GBM`. The shared `preprocessing_cache_dir` also reuses the
prepared per-origin preprocessing base when the preprocessing spec is unchanged.
Result-store identities include the data content fingerprint, the effective
selection seed, arm/model/features/preprocessing choices, the effective stage
policies, and the backend package versions that own the arm's numerical fit.
Vintage-aware specs additionally hash the enumerable vintage labels, reference
calendar, and a bounded latest-vintage panel fingerprint. Stores created before
this identity hardening will miss and recompute cells once, then reuse normally
under the new digest.

The stage policies recorded are the RESOLVED ones — what the run actually fits
under, not the fields you did or did not set. A policy you leave unset is not "no
policy": the runner resolves it against `default_preprocessing_scope`,
`default_feature_scope`, and `default_selection_scope`, so the same spec fits
differently after `mf.meta.configure(...)` changes one of them. `origin_available`
and `full_panel` are the leak-aware and the whole-panel fit, so a digest that
could not tell them apart would serve one run's forecasts to the other. Changing
a default therefore misses the cells it can reach; setting a policy explicitly
pins them against that default moving. An arm that fits no preprocessing has no
preprocessing policy, and the identity says so rather than inventing a scope.
`PipelineReport.leakage_audit["preprocessing_policies"]` reports the same resolved
policy per arm, as a `StagePolicy` mapping (or `None` for an arm with no
preprocessing).

If the content fingerprint cannot be computed at all, the cell is not cacheable.
The failure is recorded in provenance rather than raised — collecting provenance
must never end a run — but the descriptor carries no content, so it cannot
identify a panel and every panel that failed the same way would otherwise share
one digest. Such a cell is recomputed, and `run_pipeline()` warns for it like any
other undigestible cell.

The content fingerprint covers every index value, column name, and cell, whatever
the panel's size. It used to fall back to a strided subsample above a cell cap,
which meant a cell the stride skipped could change while the digest stayed put —
so a stale forecast could be served for changed data. Large panels are now streamed
in row chunks instead: the bound is on memory, not on how much of the data is read,
and the chunk size cannot move the digest. Panels that were already hashed in full
keep their exact digest, so ordinary stores are not invalidated by this change; only
the oversized panels whose digest was unsound recompute once.

For custom code, reuse is opt-in. A custom model function, feature step,
preprocessing step, metric, or loss must carry a stable `__mf_digest__` string to be
stored. Without it, the cell is recomputed every run. If you edit the callable, do
not trust old results unless you also update `__mf_digest__` and force a miss. The
store is intended for a single writer; inspect it with
`mf.pipeline.result_store_summary(...)` and delete cells with
`mf.pipeline.purge_result_store(...)`.

That rule covers **every** callable identity reaches, not only the ones named above: a
`SearchSpec`'s `custom_func`, anything callable inside its `custom_params`, a callable
passed through `Arm(params=...)`, and — because the resolved stage policies are part of
the digest — a `custom_stage_policy(...)` selector and any callable inside a
`StagePolicy`'s `metadata`. The selector's marker is what separates two selectors that
share a module and qualname; without one, the cell is recomputed and the warning names
the stage, for example
`arm.stage_policies.feature_engineering.selector`. `StagePolicy.to_dict()` itself is
unchanged: it stays the readable export and still records a selector by name.

A stage policy's `update` cadence is identified the same way — by what it means rather
than by what it prints. A pandas `DateOffset` exports as its `freqstr`, and that label is
not an identity: **every** `CustomBusinessDay` prints `"C"` whatever holidays or week mask
it carries, so two policies that refit on different days used to share a cell. Identity
records the offset's concrete type, its multiplier, `normalize`, and its full constructor
state, including any `numpy.busdaycalendar` rendered as its week mask and holiday dates —
never as the address its repr would print. An offset whose state cannot be read, or which
holds a value this store cannot identify, makes the cell uncacheable with the field path
rather than being guessed at.

Two consequences. Cells whose stage policy uses a `DateOffset` cadence **miss once** and
are recomputed, because preserving their old digest would preserve the collision; every
other policy keeps its digest. And a `DateOffset` inside a policy's `metadata` is now
identifiable too, where it previously made the cell uncacheable. A function's name is not its identity — edit
the body, keep the name, and the old forecasts would otherwise still be served — so the
marker itself is part of the digest. Two functions sharing a module and qualname with
different markers are different cells; the same marker with the same configuration is
the same cell.

Values are identified by their contents. Plain NumPy arrays record dtype, shape and a
hash of every byte (a repr would elide the middle of a large array, so two arrays
differing only there used to collide); sets and frozensets record their elements in a
canonical order that does not depend on the hash seed, and keep their container type;
`bytes` and `bytearray` record an unambiguous encoding.

Two narrowings are worth knowing because they make a cell uncacheable rather than wrong:

- **Only a plain `numpy.ndarray` is identified, not its subclasses.** A subclass carries
  meaning the base array's buffer does not — a masked array's mask decides which elements
  exist at all, and is invisible to dtype, shape and bytes. Rather than guess which
  attributes of an unknown subclass matter, a subclass is recomputed.
- **Mapping keys must be strings.** Only a `str` has text that is both unambiguous and
  the same in every process: an arbitrary object falls back to a repr carrying its
  address, and an `int` key is indistinguishable from the string of the same digits. A
  mapping with any non-string key is recomputed, so use string keys in `params` and in
  any nested configuration.

Anything identity cannot record faithfully makes the cell uncacheable rather than
guessing. That covers an unsupported object type, a `to_dict()` that raises, a mapping
whose keys collide when written down, and a self-referential structure. The cell is
recomputed and `run_pipeline()` warns with the field path, for example
`arm.model_selection.custom_params.scorer is a custom callable without __mf_digest__`.
Nothing is ever identified by `repr()`, which used to be the fallback and made two
different values with matching reprs share a cell.

Both purge helpers delete files, so they fail closed. Filters are validated before
anything is enumerated or removed: a `before=` that cannot be parsed raises rather
than acting as no cutoff at all, and a `digests=`/`aliases=` entry that is not a plain
store entry name — empty, `.`, `..`, an absolute path, anything holding a path
separator — raises rather than addressing a file outside the store. One bad entry in a
list refuses the whole call, so a refused purge deletes nothing. A directory that
resolves outside the store, such as a symlinked alias, is never followed.

```python
mf.pipeline.purge_result_store(store, before="2026-01-01")   # ok
mf.pipeline.purge_result_store(store, before="last Tuesday") # ValueError, nothing deleted
```

Deletion itself stays best effort and idempotent — an entry that is already gone is
not an error. The returned number counts **entries**, not files: it is how many stored
cells (or model fits) had at least one of their files actually removed, so a filter
that matches nothing returns 0.

For custom models, prefer passing the digest through the constructor:

```python
model = mf.models.custom_model("my_model", fit_model, mf_digest="my-model-v1")
```

When a custom object lacks a digest, `run_pipeline()` emits a warning for that
undigestible cell and includes the reason from the result-store identity layer.

Checkpoint rescoring also verifies identity for new checkpoints. Each completed
cell writes a small manifest next to its `h<h>/origin_*.parquet` files. `rescore()`
refuses manifest-bearing cells whose stored digest no longer matches the current
spec/data identity; pass `allow_stale=True` only when intentionally scoring stale
forecasts. Older checkpoint directories that lack manifests still rescore, but
emit a warning because they can only be matched by directory name.

Selection-history logging is opt-in on checkpointed pipeline runs:

```python
spec = pipeline_spec(
    data=bundle,
    targets=[TargetSpec("INDPRO")],
    horizons=[1, 3],
    window=window,
    arms=[
        Arm(
            "RIDGE",
            model="ridge",
            features=mf.feature_engineering.feature_spec(
                target="INDPRO",
                predictors="all",
                feature_steps=[mf.feature_engineering.predictor_screen(top_k=40)],
            ),
        )
    ],
    evaluation=EvalSpec(benchmark="RIDGE"),
    checkpoint_dir="cache/checkpoints",
    selection_history=True,
)
report = run_pipeline(spec)

history = mf.pipeline.selection_history(report)
freq = mf.pipeline.selection_frequency_table(history)
```

Each completed origin writes `origin_<pos>_selection.jsonl` next to its forecast
parquet file. `selection_history(...)` also accepts a checkpoint directory or a
rescored report, returning tidy rows with `arm`, `origin`, `horizon`, `kind`,
`name`, and `value`; feature-screen selections use `kind="feature"` and selected
model parameters use `kind="param"`. When loading a checkpoint directory without
its spec, non-null target and arm labels stored in these sidecars take precedence
over the sanitized directory name. Ambiguous legacy directory-only labels remain
missing instead of being guessed. With the default
`selection_history=False`, no sidecars are written.

## Reference

- [Forecasting reference page](../../reference/forecasting.md) — `run`, `ForecastResult`, forecast policy options, and stage policy definitions.
- [Pipeline reference page](../../reference/pipeline.md) — `run_pipeline`, `pipeline_spec`, `PipelineReport`, `Arm`, `EvalSpec`, and t-code to policy mapping.
