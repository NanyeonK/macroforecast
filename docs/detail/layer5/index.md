# 4.6 Layer 5: Output / Provenance

Layer 5 owns output format, saved objects, provenance fields, and artifact granularity. It is the boundary where execution becomes auditable files.

## Decision order

| Group | Axes |
|---|---|
| Format | `export_format` |
| Saved objects | `saved_objects` |
| Provenance | `provenance_fields` |
| Granularity | `artifact_granularity` |

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
