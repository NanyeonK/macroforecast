# Advanced recipe orchestration

The goal is to explain recipe orchestration in depth for users who have
graduated from the standalone tutorial track. A recipe is a single YAML
file that fully specifies an L0–L8 layered pipeline and the
sweep markers needed to expand into independent cells. Recipes provide
bit-exact replication via the manifest, custom-step extension via
`mf.register_model`, and composition via sweep markers. Standalone code
remains the right choice for exploratory work, and recipes pay off when
the study needs to be replicated, audited, or shared.

## When to use recipes

We recommend the recipe path when any one of the following holds. The
study needs bit-exact replication. The study sweeps over more than a few
configurations. The study will be shared as a unit, for example as a
paper replication package or a CI artifact. The study depends on
cross-layer references such as L4 `is_benchmark`, L6.D `mcs_inclusion`,
or L3 `lineage`. We recommend standalone code (`mf.models.*`,
`macroforecast.features.selection.*`, and the other public submodules) when the
study is exploratory, when the user is iterating on a single estimator,
or when the input data does not match the FRED-MD, QD, or SD adapter
contracts.

## Basic recipe structure

A minimal L4 recipe selects a model and runs it on a default
dataset. The structure follows the canonical layer order (L0–L8). The recipe
gallery in `examples/recipes/` ships runnable templates including
`l4_minimal_ridge.yaml` and `l7_minimal_shap.yaml`.

```yaml
study_id: minimal_ridge
random_seed: 0

data:
  dataset: fred_md
  target: INDPRO
  horizon: 1

preprocessing:
  transform: difference
  outlier: winsor_1_99

4_forecasting_model:
  model: ridge
  refit_policy: every_origin
```

The call `mf.run("path/to/recipe.yaml")` materializes every layer,
expands sweep markers into independent cells, and returns a
`ManifestExecutionResult`.

## Custom-step registration

Custom estimators register through the top-level `mf.register_model`
decorator. The registered name then becomes a valid `model:` value in
the L4 layer.

```python
import macroforecast as mf

@mf.register_model("my_baseline", description="Tail-mean baseline forecaster")
def my_baseline(X_train, y_train, X_test, context):
    # Return a scalar, or a one-element sequence for compatibility.
    return float(y_train.tail(12).mean())
```

The recipe references the registered name verbatim.

```yaml
4_forecasting_model:
  model: my_baseline
  refit_policy: every_origin
```

Confirm the registration before `mf.run` by calling
`mf.list_custom_models()`. The same pattern applies to the four related
registries. `mf.register_preprocessor` registers an L2 matrix
post-processor. `mf.register_target_transformer` registers a wrapper
that fits at the L5 boundary. The call
`mf.register_feature_block(name, fn, block_kind='temporal')` registers
an L2 feature block, with valid `block_kind` values of `temporal`,
`rotation`, and `factor`. The call `mf.register_feature_combiner`
registers an L3 feature combiner. Registration lives in the current
Python process only, since YAML does not import Python modules. The
script that calls `mf.run` must therefore import the registration
module first.

## Recipe composition

The `{sweep: [...]}` marker at any layer parameter expands into
independent cells. Each cell inherits a deterministic seed derived from
the study's `random_seed` plus the cell's position, so cell-level
reproducibility is automatic.

```yaml
4_forecasting_model:
  model: {sweep: [ridge, lasso, elastic_net]}
  refit_policy: every_origin
```

`ManifestExecutionResult.cells` is a list of `RuntimeResult` objects,
one per cell. The leader uses `result.cells[i].sink_hashes` to confirm
each cell's artifact integrity at completion time. Sweep markers compose
freely across layers, so a recipe with two sweeps of length three and
four expands into twelve cells under the default grid mode.

## Manifest semantics

`mf.run` writes a `manifest.json` to the output directory. The manifest
records the recipe source, the resolved cell parameters, per-cell sink
hashes, and provenance fields (14 fields after v0.2). The companion
function `mf.replicate(manifest_path)` re-runs the stored recipe and
verifies that every sink hash matches bit-for-bit. The bit-exact
guarantee is what underwrites paper replication packages.

```python
result = mf.run("study.yaml", output_directory="out/")
replication = mf.replicate("out/manifest.json")
assert replication.sink_hashes_match
```

## Migration from standalone to recipe

Consider a standalone study that fits a Boruta selector and a Ridge
estimator.

```python
import macroforecast as mf
from macroforecast.features.selection import Boruta

X, y = load_my_data()
sel = Boruta(random_state=0).fit(X, y)
X_sel = sel.transform(X)
# ... fit ridge on X_sel, evaluate, and so on.
```

The equivalent recipe nominates `boruta_selection` as an L3 op and
`ridge` as the L4 model. The relevant lines are as follows.

```yaml
3_feature_engineering_dag:
  nodes:
    - id: boruta
      op: boruta_selection
      params:
        n_estimators_rf: 100
        alpha: 0.05

4_forecasting_model:
  model: ridge
```

The recipe form earns the user manifest-backed replication, sweep
support, and CI auditability, in exchange for the YAML overhead.

## See also

- {doc}`add_custom_model`
- {doc}`replicate_a_study`
- {doc}`tune_hyperparameters`
- {doc}`sweep_over_models`
- {doc}`reproducibility_policy`
