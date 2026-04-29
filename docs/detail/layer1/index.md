# 4.1 Layer 1: Data Task

- Parent: [4. Detail (code): Full](../index.md)
- Previous: [4.0 Layer 0: Study Scope](../layer0/index.md)
- Current: Layer 1
- Next: [4.2 Layer 2: Representation / Research Preprocessing](../layer2/index.md)

Layer 1 owns the official data task. It decides which data source is used, which target structure is being forecast, what information is available, and how raw source-level missing/outlier and official-transform policies are handled before representation construction.

## Decision order

| Group | Axes |
|---|---|
| Source and frame | `dataset`, `source_adapter`, `frequency`, `information_set_type` |
| FRED-SD source selection | `fred_sd_frequency_policy`, `fred_sd_state_group`, `fred_sd_variable_group` |
| Target and universe | `target_structure`, `variable_universe` |
| Availability and timing | `missing_availability`, `release_lag_rule`, `contemporaneous_x_rule` |
| Raw source cleaning | `raw_missing_policy`, `raw_outlier_policy` |
| Official transforms | `official_transform_policy`, `official_transform_scope` |

## Layer contract

Input:
- source request and target request.

Output:
- `layer1_official_frame_v1`;
- source availability contract;
- data reports for availability, release lag, missing policy, and FRED-SD source metadata when relevant.

## Canonical names

Layer 1 is canonical-only. Recipes should use the axis IDs in the decision-order table and the values documented on each axis page. Removed aliases for source dispatch, information-set regime, and target shape are rejected during registry validation, so generated YAML, docs, and manifests stay on one vocabulary.

## Related reference

- [Layer 1 Data Task Audit](../layer1_data_task_audit.md)
- [Data Source and Frame](../../user_guide/data/source.md)
- [Target Structure](../../user_guide/data/target_structure.md)
- [Data Handling Policies](../../user_guide/data/policies.md)
