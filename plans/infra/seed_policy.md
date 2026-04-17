# Infra: Seed Policy

## 1. What

Deterministic per-variant seed resolution that replaces the hardcoded `random_state=42`
literal scattered across `execute_recipe` and its downstream callers. The seed policy
centralises how a reproducibility mode (declared once in the recipe spec) is translated
into concrete integer seeds passed to estimators, samplers, permutation importance, and
deep model initialisers. By routing every randomness-consuming call site through a
single `resolve_seed(...)` function, we guarantee that (a) identical recipes produce
identical artifacts across machines, (b) variants in a controlled-variation study get
distinct but deterministic seeds derived from their `variant_id`, and (c) exploratory
runs can opt into non-determinism without code changes.

## 2. Used by phases

- phase-00 — introduces `seed_policy.py` module and wires the recipe-level field
- phase-01 — SweepRunner invokes resolver per variant with `variant_id` key
- phase-05a — deep model families call resolver for `torch.manual_seed` and CUDA seeds
- phase-06 — ablation engine seeds per (ablation_group, variant) pair
- phase-07 — decomposition re-runs use the same resolver with recorded `variant_id`

## 3. API spec

```python
# macrocast/execution/seed_policy.py
import hashlib
import numpy as np

VALID_MODES = {
    "strict_reproducible",
    "seeded_reproducible",
    "best_effort",
    "exploratory",
}

def resolve_seed(
    *,
    recipe_id: str,
    variant_id: str | None = None,
    reproducibility_spec: dict,
    model_family: str | None = None,
) -> int:
    mode = reproducibility_spec.get("reproducibility_mode", "seeded_reproducible")
    base_seed = int(reproducibility_spec.get("seed", 42))
    if mode == "strict_reproducible":
        key = f"{recipe_id}|{variant_id or 'main'}|{model_family or ''}"
        digest = hashlib.sha256(key.encode()).hexdigest()[:8]
        return int(digest, 16) & 0x7FFFFFFF
    if mode in ("seeded_reproducible", "best_effort"):
        return base_seed
    if mode == "exploratory":
        return int(np.random.randint(0, 2**31 - 1))
    raise ValueError(f"unknown reproducibility_mode: {mode}")
```

The 4 modes:

- `strict_reproducible` — hash-derived per-variant seed; bit-identical outputs guaranteed
- `seeded_reproducible` — single `base_seed` for the whole run; default for single-path
- `best_effort` — same as seeded, but BLAS/GPU non-determinism tolerated
- `exploratory` — fresh random seed each call; used for ad-hoc sanity checks

## 4. Implementation notes

Every caller must pass `reproducibility_spec` explicitly — no module-level fallback.
Known call sites in `execute_recipe` (approx. 20+ occurrences of `random_state=` or
`default_rng(...)`) at lines 557, 577, 587, and later across the fit/predict loop.
`sklearn.inspection.permutation_importance`, `numpy.random.default_rng`, and in
phase-05a the `torch.manual_seed` + `torch.cuda.manual_seed_all` pair all go through
`resolve_seed`. For CUDA, the resolved int is also set on `torch.backends.cudnn`
determinism flags when mode is `strict_reproducible`.

The resolver is pure and cheap — no caching needed. Variant IDs must be stable across
re-runs (see study_manifest schema).

## 5. Test requirements

`tests/test_seed_policy.py` must cover:

- all 4 modes return expected types and ranges
- `strict_reproducible` is deterministic for a fixed `(recipe_id, variant_id, model_family)`
- distinct `variant_id`s produce distinct seeds under strict mode (collision probability ~0)
- `seeded_reproducible` ignores `variant_id` and returns `base_seed`
- `exploratory` returns varying values across calls
- unknown mode raises `ValueError` with the offending value in the message
- integration test: two successive `execute_recipe` runs with `strict_reproducible`
  produce byte-identical `predictions.csv`

## 6. Owner / ADR references

Owned by phase-00 (module introduction) with follow-up ownership in phase-01 (sweep
integration) and phase-05a (torch wiring). Related to ADR-005 (reproducibility
contract) — this file is the operational realisation of that ADR's 4-mode matrix.
