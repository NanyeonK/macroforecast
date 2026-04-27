# Replication Library

The Replication Library stores paper-style routes as exact tree paths plus runnable YAML.

It is not limited to byte-identical replication packages. Each entry must state deviations from the original paper so users know whether they are running a package-native route, an approximate reproduction, or a strict replication.

## CLI

List entries:

```bash
macrocast-navigate replications
```

Inspect and write one YAML:

```bash
macrocast-navigate replications goulet-coulombe-2021-fred-md-ridge \
  --write-yaml recipes/gc2021-ridge.yaml
```

## Navigator App

The static Navigator App receives each replication entry with both runnable
YAML text and a parsed recipe object. Clicking `Load path` on an entry replaces
the browser tree state with that replication route, shows the exact path and
declared deviations, refreshes disabled-branch reasons, and regenerates the
YAML preview. The downloaded YAML should still be resolved with the CLI before
execution.

## Entry Contract

Every entry contains:

| Field | Meaning |
|---|---|
| `paper_name` | Paper or route name. |
| `short_description` | One-paragraph purpose. |
| `exact_tree_path` | Canonical path choices that define the route. |
| `recipe_yaml` | Full runnable YAML text. |
| `recipe` | Parsed recipe object used by the browser app to load the route without a Python process. |
| `command` | One-line CLI command. |
| `notebook_snippet` | Minimal notebook entry point. |
| `expected_outputs` | Artifact files users should expect. |
| `deviations_from_original_paper` | Explicit differences from the paper or original replication package. |

## Built-In Entries

| ID | Route | Purpose |
|---|---|---|
| `goulet-coulombe-2021-fred-md-ridge` | Goulet Coulombe et al. (2021), FRED-MD ridge-style path | Official transformations, train-only standardization, raw-panel ridge generator, AR-BIC benchmark, MSFE, and DM test. |
| `fred-sd-midasr-almonp-direct` | FRED-SD mixed-frequency MIDAS runtime route | Fixture-safe route for Layer 2 native-frequency adapter payloads plus Layer 3 `model_family=midasr`, `midasr_weight_family=almonp`. |
| `synthetic-replication-roundtrip` | Fixture-safe synthetic route | Small route for verifying YAML generation, path resolution, and one-line execution. |

## Templates Versus Entries

Replication entries are runnable from YAML alone. Custom model recipes are
different: the YAML can select `model_family=my_registered_model`, but the
callable must already be registered in the Python process with
`@mc.custom_model(...)`.

Custom extension examples therefore live as templates rather than built-in
replication entries. The current FRED-SD mixed-frequency template is:

```text
examples/custom_fred_sd_mixed_frequency_model.py
examples/recipes/templates/fred-sd-custom-mixed-frequency-model.yaml
```

Use the Python file in a script or notebook to register the custom model, then
compile or run the YAML route. The Navigator should display this as a custom
route template, not as a paper replication entry, because the model code is
researcher-owned.

## Goulet-Coulombe-Style Path

The package-native route uses:

```text
1_data_task.dataset=fred_md
1_data_task.official_transform_policy=apply_official_tcode
2_preprocessing.tcode_policy=official_tcode_then_extra_preprocess
2_preprocessing.scaling_policy=standard
3_training.feature_builder=raw_feature_panel
3_training.model_family=ridge
3_training.benchmark_family=ar_bic
4_evaluation.primary_metric=msfe
6_stat_tests.equal_predictive=dm
```

The key Layer 2 detail is that `t-code + standardize` is not `official_tcode_only`. It is `official_tcode_then_extra_preprocess` with train-only scaling.

## FRED-SD MIDASR Path

This entry exercises the package-owned mixed-frequency route:

```text
1_data_task.dataset=fred_sd
1_data_task.frequency=monthly
2_preprocessing.fred_sd_mixed_frequency_representation=mixed_frequency_model_adapter
3_training.feature_builder=raw_feature_panel
3_training.model_family=midasr
3_training.midasr_weight_family=almonp
3_training.forecast_type=direct
```

It is a runtime recipe, not a paper-identical replication. Run it against the
small fixture or replace `--local-raw-source` with an official FRED-SD
workbook:

```bash
macrocast-navigate replications fred-sd-midasr-almonp-direct \
  --write-yaml /tmp/fred-sd-midasr-almonp.yaml

macrocast-navigate run /tmp/fred-sd-midasr-almonp.yaml \
  --local-raw-source tests/fixtures/fred_sd_sample.csv \
  --output-root /tmp/macrocast-fred-sd-midasr
```

## Synthetic Round Trip

The synthetic entry is designed to run in the test fixture environment:

```bash
macrocast-navigate replications synthetic-replication-roundtrip \
  --write-yaml /tmp/synthetic-replication.yaml

macrocast-navigate run /tmp/synthetic-replication.yaml \
  --local-raw-source tests/fixtures/fred_md_ar_sample.csv \
  --output-root /tmp/macrocast-synthetic
```
