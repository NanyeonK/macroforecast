# Phase 0 Audit — Single-Path Execution Sweep-Safety

Date: 2026-04-17
Scope: `macrocast/execution/build.py` (2368 LOC), `macrocast/tuning/engine.py`, `macrocast/raw/cache.py`, `macrocast/registry/build.py`
Purpose: identify hidden global state, RNG determinism issues, cache/imports side effects that would leak across N repeated `execute_recipe()` calls in a Phase 1 sweep loop.

---

## Summary

| Concern | Severity | Fix required before Phase 1? |
|--------|:--------:|:--------------:|
| Hardcoded `random_state=42` across 20+ model calls | **High** | Yes — blocks valid sweep statistics |
| Double-write of `manifest.json` and `tuning_result.json` | **Medium** | Yes — code-smell + wasted I/O |
| `_last_tp` semantic ambiguity in multi-target loop | **Low** | Defer (fix when multi-target gains more use) |
| Cache root tied to `output_root` — sweep variants re-download | **Medium** | Yes — add sweep-level cache share |
| ThreadPoolExecutor in `parallel_by_model` nesting with future sweep parallelism | **Low** | Defer (policy decision for Phase 1) |
| Registry `lru_cache` | Safe | No |
| `raw/cache.py` | Safe | No |
| Module-level warnings/pd options | Safe | No |

---

## 1. RNG / seed issues (HIGH severity)

### Finding

`macrocast/execution/build.py` hardcodes `random_state=42` (or `random_seed=42`) in 20+ model instantiations:

- lines 557, 577, 587, 592, 597, 602, 607, 612 — autoreg model variants (RF, SVR, ET, GBM, XGB, LGB, CB, MLP)
- lines 648, 672, 684, 690, 696, 702, 708, 714 — raw-panel model variants (same families)
- line 1429 — RandomForest in importance computation
- line 1576, 1640 — permutation_importance `random_state=42`
- line 1592 — `np.random.default_rng(42)` hardcoded

Parametric seed exists in a few places (lines 1088, 1722 use `np.random.default_rng(seed)` where `seed` is passed), but the vast majority use literal 42.

### Why it matters for Phase 1 sweep runner

Phase 1 will execute the same model family under `N` axis variants (e.g., `preprocessing_scaling ∈ {none, standard, robust}`). With hardcoded seed 42:

- Every variant's RF trains with identical bootstrap draws
- Every variant's permutation importance uses identical permutation paths
- Between-variant variance is artificially suppressed → sweep statistics under-estimate real variability
- DM / MCS / reality-check assumptions about independence across variants break

This makes sweep output **scientifically invalid** unless fixed.

### Recommended fix (Phase 0 scope)

1. Introduce `_resolve_seed(recipe, context)` helper that reads `reproducibility_spec` from provenance_payload and returns a deterministic per-variant seed (e.g., `hash(variant_id) & 0xFFFFFFFF`).
2. Replace all literal `random_state=42` with `_resolve_seed(...)` calls.
3. `reproducibility_spec.seed_policy` already defined in registry — wire it into execute_recipe (currently only stored in manifest, not consumed).
4. Document: single-path calls without explicit seed get seed=42 (back-compat); sweep-owned calls get variant-derived seeds.

### Files touched

- `macrocast/execution/build.py` (20+ lines to replace)
- Possibly new: `macrocast/execution/seed_policy.py` (centralized seed resolution)

---

## 2. Double-write of manifest.json and tuning_result.json (MEDIUM severity)

### Finding

In `execute_recipe()`:

- **Line 2208-2212** writes `tuning_result.json` then `manifest.json`
- **Line 2343-2352** writes `tuning_result.json` then `manifest.json` **again**

The second write has the current state (stat_test + importance additions). The first write is **stale** (pre-stat-test, pre-importance). On disk the second overwrites the first, so artifacts are correct — but:

- Wasted I/O (2× file writes per run)
- If a crash occurs between writes, disk holds stale manifest
- Bug prone — any code path adding new manifest fields must ensure placement after first write

### Why it matters for sweep runner

N variants × 2 wasted writes = 2N I/O ops. Not catastrophic, but sweep runtime overhead adds up.

### Recommended fix (Phase 0 scope)

Remove the first write (lines 2208-2212 `_write_json(...manifest.json)` and the tuning_result write that precedes it). Keep only the end-of-function write.

### Files touched

- `macrocast/execution/build.py` lines 2199-2212

---

## 3. `_last_tp` semantic ambiguity (LOW severity)

### Finding

In the multi-target loop (lines 2108-2139), `_last_tp` is overwritten each iteration. Whatever the last target's tuning payload happens to be gets persisted in the manifest. For multi-target recipes this is confusing — each target may have different tuning behavior.

### Recommendation

Defer. Multi-target joint tuning is Phase 5b scope. For Phase 1 sweep runner (fixed-target within variant), single-target `_last_tp` is fine.

---

## 4. Cache root tied to output_root (MEDIUM severity)

### Finding

Line 2094:
```python
raw_result = _load_raw_for_recipe(recipe, local_raw_source, output_root / ".raw_cache")
```

Raw data cache is **per-output-root**, not global. If Phase 1 sweep runner emits each variant to `output_root/variant_<hash>/`, each variant gets its own `.raw_cache/` → each variant re-downloads FRED data.

### Why it matters

- 100-variant sweep × N-MB FRED download = large unnecessary I/O + network use
- FRED vintage cache should be **study-level**, not variant-level

### Recommended fix (Phase 0 scope)

Introduce a `cache_root` parameter on `execute_recipe()` that defaults to current behavior for single-path, but can be overridden by sweep runner to a shared study-level cache dir.

### Files touched

- `macrocast/execution/build.py` `execute_recipe()` signature + line 2094
- `macrocast/execution/types.py` if cache_root becomes part of ExecutionSpec

### Default cache discipline

Follow `raw/cache.py` convention: `get_raw_cache_root()` already defaults to `~/.macrocast/raw` when called without a path. Sweep runner can pass this directly. Single-path callers continue to opt into per-output cache via existing default.

---

## 5. ThreadPoolExecutor in parallel_by_model (LOW severity)

### Finding

Lines 2112-2129 use `ThreadPoolExecutor` for `compute_mode == "parallel_by_model"`. If a Phase 1 sweep runner itself parallelizes (`compute_mode = parallel_by_variant`), we'd nest thread pools.

### Recommendation

Defer. Phase 1 sweep runner should start serial-by-variant; parallelism is a Phase 5+ topic. When nesting becomes concrete, decide policy (ban inner parallelism, cap outer × inner workers, etc.).

---

## 6. Safe patterns (verified)

### Module-level constants (`build.py` lines 38-41)

```python
_EXECUTION_ARCHITECTURE = "separate_model_and_benchmark_executors"
_DEFAULT_MINIMUM_TRAIN_SIZE = 5
_DEFAULT_MAX_AR_LAG = 3
_LAG_SELECTION = "bic"
```

All immutable strings/ints, no mutation risk. **Safe.**

### Registry auto-loader (`registry/build.py`)

Uses `@lru_cache(maxsize=1)` for `_discover_axis_definitions()` and `_axis_registry()`. Results are read-only; `get_axis_registry()` returns `dict(...)` copy. **Safe.**

### Raw data cache (`raw/cache.py`)

All file-based, no in-memory state. Cache operations are idempotent file creation. **Safe.**

### No module-level side effects

Grep confirmed: no `warnings.filterwarnings` / `pd.set_option` / `sklearn.set_config` / `np.set_printoptions` at module level. Only context-managed `warnings.catch_warnings()` inside functions. **Safe.**

### Tuning engine (`tuning/engine.py`)

No module-level mutable state. Pure functions with explicit parameters. **Safe.** In fact well-suited for Phase 1 reuse.

---

## Phase 0 Action Items

### P0 blockers for Phase 1 (must fix):

1. **Replace hardcoded `random_state=42` with seed policy wiring** — new `seed_policy.py`, replace 20+ sites in `build.py`
2. **Remove duplicate manifest/tuning_result writes** — clean up lines 2199-2212
3. **Add `cache_root` parameter to `execute_recipe()`** for sweep-level cache sharing

### P1 (nice-to-have, defer if needed):

4. Audit `macrocast/execution/deep_training.py` for same seed-hardcoding pattern (likely present given same models)
5. Add a regression test: 100 consecutive `execute_recipe()` calls with same recipe → identical artifact hash (proves determinism given same seed)
6. Add a second test: 100 consecutive calls with distinct seeds → distinct artifact hashes (proves seed is actually wired)

### P2 (Phase 5+ when relevant):

7. Decide parallelism nesting policy when sweep runner gains parallel_by_variant mode

---

## Estimated effort

| Item | Hours |
|------|------|
| Seed policy wiring + 20+ site replacement | 4-6h |
| Double-write cleanup | 0.5h |
| Cache_root parameter | 1h |
| Regression tests | 2h |
| **Phase 0 total** | **~1-2 days** |

Phase 0 does not require architectural changes — it's surgical cleanup. Safe to do as a single PR before starting Phase 1.

---

## Readiness gate for Phase 1

Phase 1 (sweep runner) should NOT start until:

- [ ] Seed policy fixed for all model families
- [ ] Deterministic-replay regression test green
- [ ] Cache_root parameter landed
- [ ] Manifest double-write removed
- [ ] Full existing test suite (291 tests) still green

Once these are done, `execute_recipe()` can be confidently called N times by a sweep runner without cross-contamination or statistical invalidity.
