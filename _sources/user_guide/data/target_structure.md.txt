# Target Structure (1.2)

Layer 1 only declares the target shape inside the official data frame. It does
not choose the study runner. Runner ownership stays in Layer 0 through
`research_design` and `experiment_unit`.

## 1.2.1 `target_structure`

`target_structure` says whether the recipe has one target series or multiple
target series.

Values:

- `single_target`: one target series. Requires
  `leaf_config.target`.
- `multi_target`: two or more target series. Requires
  `leaf_config.targets`.

Compatibility:

- Legacy recipes may still use `task`.
- The compiler canonicalizes `task` to `target_structure`.
- New generated recipes, manifests, and docs should use `target_structure`.

Layer 0 connection:

- `experiment_unit` is derived from `target_structure` plus the sweep shape.
- Multi-target `experiment_unit` values require
  `target_structure=multi_target`.
- Single-target `experiment_unit` values require
  `target_structure=single_target`.

Recipe usage:

```yaml
path:
  1_data_task:
    fixed_axes:
      target_structure: multi_target
    leaf_config:
      targets: [INDPRO, UNRATE, CPIAUCSL]
      horizons: [1, 3, 6]
```

Single-target usage:

```yaml
path:
  1_data_task:
    fixed_axes:
      target_structure: single_target
    leaf_config:
      target: INDPRO
      horizons: [1, 3, 6]
```

## Boundary

These axes are no longer Layer 1 target-structure choices:

- `forecast_type`: Layer 3, because direct vs iterated is forecast-generation
  logic.
- `forecast_object`: Layer 3, because mean/median/quantile is model output
  contract.
- `predictor_family` and `feature_builder`: Layer 2, because they choose the
  feature representation handed to the forecast generator.
- `horizon_target_construction`: Layer 2, because level/diff/logdiff target
  construction is target representation. Coulombe-style direct average and
  path-average growth/difference targets are also Layer 2 target representation
  choices, even where runtime support is still registry-only.
- `multi_target_architecture`: Layer 0, through `experiment_unit`.

Layer 1 owns target identity and target cardinality. Layer 0 owns how that
cardinality is executed.
