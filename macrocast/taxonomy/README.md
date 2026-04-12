# macrocast.taxonomy

Purpose
- Tree-structured registry space for forecasting-package choices.
- YAML truth for enumerated choices, path grammar, and extension slots.
- Numeric/free parameters belong in recipe configs, not enum registries.

Canonical layer order
- 0_meta
- 1_data
- 2_target_x
- 3_preprocess
- 4_training
- 5_evaluation
- 6_stat_tests
- 7_importance
- 8_output_provenance

Design rules
- One path = one fully specified forecasting study.
- Enum choices live in taxonomy registries.
- Numeric/free parameters live in recipe/run configs.
- Fixed / sweep / conditional / derived axes are separated.
- Every major family should permit custom extensions.
