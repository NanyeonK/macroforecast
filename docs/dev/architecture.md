# Architecture

## 8-Layer Canonical Order

macrocast enforces a fixed layer order for every forecasting study:

```
Stage 0: Meta / Grammar        — study_mode, experiment_unit, comparison_contract
Stage 1: Data / Task            — dataset, frequency, forecast_type, horizons
Stage 2: Preprocessing          — tcode, missing, outlier, scaling, governance
Stage 3: Training               — framework, model, feature_builder, tuning
Stage 4: Evaluation             — metrics, benchmarks, regime analysis
Stage 5: Output / Provenance    — export format, saved objects, manifest
Stage 6: Statistical Tests      — DM, CW, MCS, diagnostics
Stage 7: Importance             — SHAP, permutation, PDP, stability
```

## Key principles

### Grammar first, content later
Stage 0 fixes the structural language of a study (what is fixed, what varies, how fairness is defined) BEFORE later registries are populated with content.

### One path = one fully specified study
Every recipe defines a complete study. No hidden defaults. If a choice matters, it must be in the recipe.

### Represent before execute
The registry can represent more choices than the runtime can execute. A value marked `registry_only` is grammatically valid but has no execution code yet. This lets the design space be defined upfront and incrementally operationalized.

### Separate model execution from benchmark execution
Models and benchmarks are executed by separate code paths. The benchmark always uses the same training data as the model to ensure fair comparison.

## Pipeline flow

```
YAML recipe
  -> Compiler (validate axes, build Stage 0, build preprocess contract)
  -> Execution (load data, build features, OOS loop, metrics, artifacts)
  -> Artifacts (predictions.csv, metrics.json, stat_test.json, importance.json, manifest.json)
```

## Registry architecture

```
macrocast/registry/
  base.py           — BaseRegistryEntry, AxisDefinition (per-axis typed entries)
  build.py          — auto-discovery loader (scans stage0/, data/, training/, etc.)
  stage0/           — 7 grammar axes
  data/             — 29 data/task axes
  preprocessing/    — 24 preprocessing axes
  training/         — 28 training axes
  evaluation/       — 18 evaluation axes
  output/           — 4 output axes
  tests/            — 2 stat test axes
  importance/       — 13 importance axes
```

**See also:** [User Guide](../user_guide/index.md) | [API Reference](../api/index.md)
