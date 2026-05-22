# Philosophy

macroforecast is designed for research workflows where the difficult part is not calling a model function. The difficult part is choosing a valid forecasting path, keeping comparisons fair, and making every deviation auditable.

## Navigation before execution

The package has many possible choices across data, representation, model, evaluation, output, statistical tests, and interpretation. API docs alone cannot explain which combinations are valid. The Navigator is therefore a first-class documentation surface, not a demo.

## Defaults are explicit

Defaults are part of the research design. They must be visible, versioned, and written to artifacts. A simple run should remain short, but the resolved manifest must show what was actually used.

## Layers own decisions

Each layer owns a different class of decisions:

- L0: study setup and execution grammar.
- L1: data source, target, predictor universe.
- L2: cleaning and preprocessing.
- L3: feature engineering and target construction (DAG).
- L4: forecasting model fit, tuning, and ensembling (DAG).
- L5: evaluation, benchmarking, ranking.
- L6: statistical tests.
- L7: interpretation and importance (DAG).
- L8: export and provenance.

Layer boundaries matter because they prevent silent leakage, hidden preprocessing, and ambiguous model comparisons.

## Contracts are enforced

Some choices are free. Some are derived. Some are disabled because another selection makes them invalid. The docs and Navigator should show the same constraint system.

## Custom methods are first-class

Method researchers should be able to add a custom Layer 2 representation or Layer 3 generator and compare it against built-ins without editing package internals.

## Artifacts are part of the API

Predictions, metrics, manifests, sidecar reports, and provenance files are not incidental outputs. They are how a run is audited, replicated, and compared.

## Next

- [Navigator](../navigator/index.md)
- [Researchers](../for_researchers/index.md)
- [Recipe authors](../for_recipe_authors/index.md)
- [Architecture index](index.md)
