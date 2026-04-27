# 4. Detail (code)

Detail docs are the full layer-by-layer contract for macrocast. Use them when you need exact YAML, runtime artifacts, custom method hooks, or compatibility rules.

Each layer page follows the same structure:

- role;
- input contract;
- output contract;
- decision order;
- axis list;
- forced or derived choices;
- compatibility gates;
- runtime artifacts;
- YAML path shape;
- code path;
- related Navigator links.

## Layer listing

| Section | Layer | Owns |
|---|---|---|
| [4.1](layer0/index.md) | Layer 0: Study setup | Research design, runner unit, failure policy, reproducibility, compute layout. |
| [4.2](layer1/index.md) | Layer 1: Data task | Dataset, source adapter, target structure, data availability, raw missing/outlier policy, official transforms. |
| [4.3](layer2/index.md) | Layer 2: Representation | T-code use, missing/outlier handling after official frame, target construction, feature blocks, scaling, selection, custom preprocessing. |
| [4.4](layer3/index.md) | Layer 3: Forecast generator | Model family, benchmark family, forecast type/object, training windows, tuning, future-X paths. |
| [4.5](layer4/index.md) | Layer 4: Evaluation | Metrics, benchmark comparison scope, aggregation, ranking, regimes, OOS period. |
| [4.6](layer5/index.md) | Layer 5: Output / provenance | Export formats, saved objects, artifact granularity, manifest and sidecar contracts. |
| [4.7](layer6/index.md) | Layer 6: Statistical tests | Equal predictive ability, nested tests, multiple-model tests, density/direction tests, correction rules. |
| [4.8](layer7/index.md) | Layer 7: Interpretation / importance | Importance family, SHAP, local surrogate, partial dependence, grouping, stability, temporal output. |

```{toctree}
:maxdepth: 2
:caption: Layers

layer0/index
layer1/index
layer2/index
layer3/index
layer4/index
layer5/index
layer6/index
layer7/index
```

```{toctree}
:maxdepth: 1
:caption: Reference and Audits

terminology
philosophy
experiment_object
default_profiles
recipe_layers
layer_boundary_contract
layer_contract_ledger
layer_axis_census
layer_axis_migration_plan
package_runtime_gap_audit
post_pr70_runtime_roadmap
registry_axes
execution_engine
artifacts_and_manifest
custom_extensions
target_transformer
raw_panel_iterated_contract
reproducibility
layer0_meta_audit
layer1_data_task_audit
layer2_feature_representation
layer2_closure_ledger
layer2_layer3_detailed_design
layer2_layer3_sweep_contract
layer2_layer3_grid_examples
layer2_revision_plan
layer3_training_audit
preprocessing_layer_audit
fred_sd_transform_policy
fred_sd_inferred_tcodes
fred_sd_inferred_tcode_review_v0_1
layer_audit
```
