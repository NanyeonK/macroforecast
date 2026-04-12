# TP-006 Benchmark Redesign

This step changes benchmark handling from early resolved ids toward family plus options.

Current first pass:
- `config/meta/benchmarks.yaml` now has `benchmark_families` and `benchmark_variants`
- `resolve_meta_config(..., benchmark_registry=...)` resolves family + options into a benchmark id
- compiled specs now carry `benchmark_options` as explicit contract field

Current limitation:
- old config defaults still include resolved ids in places
- benchmark options are not yet fully exposed through every user-facing path
- this is the migration step toward full benchmark/model symmetry
