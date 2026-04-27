# 4.6 Layer 5: Output / Provenance

- Parent: [4. Detail (code): Full](../index.md)
- Previous: [4.5 Layer 4: Evaluation](../layer4/index.md)
- Current: Layer 5
- Next: [4.7 Layer 6: Statistical Tests](../layer6/index.md)

Layer 5 owns output format, saved objects, provenance fields, and artifact granularity. It is the boundary where execution becomes auditable files.

## Decision order

| Group | Axes |
|---|---|
| Format | `export_format` |
| Saved objects | `saved_objects` |
| Provenance | `provenance_fields` |
| Granularity | `artifact_granularity` |

## Naming migration

Layer 5 keeps artifact choices explicit in YAML. Older punctuation-based
format IDs still compile through `registry_naming_v1`.

| Axis | Legacy value | Canonical value |
|---|---:|---:|
| `export_format` | `json+csv` | `json_csv` |

## Layer contract

Input:
- run artifacts from Layers 1-4 and optional Layer 6-7 outputs.

Output:
- `artifact_manifest.json`;
- `manifest.json`;
- selected exports such as JSON, CSV, or parquet.

## Related reference

- [Artifacts and Manifest](../artifacts_and_manifest.md)
- [Layer Contract Ledger](../layer_contract_ledger.md)
