# Infra: Cache Discipline

## 1. What

Separation between **per-study shared raw cache** (FRED vintage downloads, macro
data pulls) and **per-variant artifact directories** (predictions, metrics,
manifests). Sweep studies can contain dozens of variants that all need identical
upstream data, so re-downloading per variant is wasteful and risks vintage drift
between siblings in the same study. This infra establishes a single discipline: raw
data lives once at the study root under a shared cache directory, while each
variant writes a complete, self-contained artifact directory that mirrors the
single-path output layout.

## 2. Used by phases

- phase-00 — introduces the `cache_root` parameter on `execute_recipe`
- phase-01 — `SweepRunner` passes a shared `cache_root` to every variant call so
  FRED downloads happen exactly once per study

## 3. API spec

```python
from pathlib import Path

def execute_recipe(
    ...,
    output_root: str | Path,
    cache_root: str | Path | None = None,
) -> ExecutionResult:
    effective_cache = (
        Path(cache_root)
        if cache_root is not None
        else (Path(output_root) / ".raw_cache")
    )
    # pass `effective_cache` to every raw loader / FRED adapter
    ...
```

Layout on disk:

```
<output_root>/
  .raw_cache_shared/        # study-level, shared across variants
    fred_md_2026-04.csv
    fred_qd_2026-04.csv
  variants/
    v-a1b2c3d4/
      predictions.csv
      metrics.json
      manifest.json
      failures.json        # only if failed
    v-e5f6g7h8/
      ...
  study_manifest.json
  decomposition_result.json  # phase-07 output
  bundle/                    # phase-08 output
```

## 4. Implementation notes

Default behaviour is preserved for single-path callers: when `cache_root=None`,
the cache lives under the run's own `output_root/.raw_cache`, which is the current
behaviour. No existing caller breaks.

`SweepRunner` sets `cache_root = <study_root>/.raw_cache_shared` and passes it to
every variant invocation. The shared directory is created once up front; variant
runs see it read-mostly (writes happen only on first cache miss and are idempotent).

Per-variant artifact directories remain the single-path layout — the same files
that `execute_recipe` writes today in `output_root`. No variant ever writes outside
its own `variants/<variant_id>/` directory except for the manifest update to
`study_manifest.json`, which is owned by the runner, not the variant.

Cache invariants: filename includes vintage or hash of source URL; no variant
mutates a cached file in place; concurrent variant runs in the future (phase-02+)
will need a file lock, not yet in scope.

## 5. Test requirements

`tests/test_execution_cache.py`:

- `execute_recipe` with `cache_root=None` writes to `<output_root>/.raw_cache`
- Explicit `cache_root` overrides default and no files land under `.raw_cache`
- Re-running the same recipe hits the cache (no second download)

`tests/test_sweep_cache_share.py`:

- Two variants in the same study see the same on-disk cache file (inode identity
  on Unix, or equivalent content hash assertion)
- Exactly one FRED download occurs per vintage per study, verified by mocking the
  FRED client and asserting call count

## 6. Owner / ADR references

Owned by phase-00 (parameter introduction) and phase-01 (sweep wiring). No
dedicated ADR. The discipline is referenced by the study_manifest schema
(artifact_dir paths are relative to `<output_root>`) and by the release plan (the
cache is never shipped in bundles or PyPI wheels).
