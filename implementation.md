# implementation.md — Cycle 64 Bug Fix Retry

## Summary

Two pre-existing bugs in `macroforecast/core/runtime.py` uncovered by the
C64 tester are fixed here, along with one test update. Both bugs affected the
newly promoted public model classes (`SlowGrowingTree` and `Bagging`).

---

## Files Modified

### macroforecast/core/runtime.py

**Bug A — `_SlowGrowingTree.__init__` max_depth default (line ~5923)**

Changed `max_depth: Any = None` to `max_depth: Any = 10`.

Root cause: the `_build()` BFS only terminates when either (a) the
Herfindahl index H >= herfindahl_threshold, (b) max_depth is reached, or
(c) no split improves SSE. With soft weights (eta < 1), every node retains
all n rows at non-zero weight. H = Σ(ω_i²)/(Σω_i)² stays near 1/n (for
n=80 this is ~0.013), far below the threshold=0.25. No split can trigger
the Herfindahl stop, so the only reliable depth bound is max_depth. With
the old default of None, the BFS grew without bound and hit the 60s timeout.

Fix chosen: Option 1 (add max_depth=10 fallback). This matches the depth
range used in standard CART literature and stops the BFS after at most 2^10
= 1024 nodes in the worst case. Users wanting deeper trees can pass
max_depth=None explicitly.

**Bug B — `_BaggingWrapper.fit` TypeError on base_params=None (line ~8042)**

Changed `params = dict(self.base_params)` to
`params = dict(self.base_params) if self.base_params is not None else {}`.

Root cause: `_BaggingWrapper.__init__` sets `self.base_params = dict(base_params or {})`
which guards None during direct construction. However, sklearn's
`BaseEstimator.set_params()` and `clone()` bypass `__init__` and set
attributes directly, so `self.base_params` can be set to None after
construction. The `fit()` method then called `dict(None)` which raises
TypeError. The guard in `fit()` makes it robust regardless of how
`base_params` was set.

### macroforecast/models/tree.py

**Bug A companion fix — `SlowGrowingTree.__init__` max_depth default**

The public `SlowGrowingTree` class inherits from `_SlowGrowingTree` but
overrides `__init__` and restores raw parameter values after `super().__init__()`
(the sklearn clone()-safe pattern documented in tree.py's module docstring).
This means `self.max_depth = max_depth` at the end of the public class's
`__init__` overwrites the private class's default of 10 with the public
class's default of None.

Both defaults had to be updated: `_SlowGrowingTree.__init__` (runtime.py)
and `SlowGrowingTree.__init__` (tree.py).

Updated the docstring for `max_depth` in `SlowGrowingTree` to explain why
`None` is not recommended for SGT (soft weights keep H low, so max_depth
is the primary depth bound).

### tests/promotion/test_c63_promotion.py

**C63 `__all__` count update: 22 → 30**

`test_A4_models_all_count_is_22` hardcoded 22. After C64 promoted 8 more
classes (tree.py: 6 + neural.py: 2), `mf.models.__all__` has 30 entries
(14+3+2+3+6+2). Updated to 30 and renamed the test method to
`test_A4_models_all_count_is_30` for clarity. Added explanation of the
C64 additions in the docstring.

---

## Commits

1. `971abe92` — `fix(runtime): _SlowGrowingTree._build infinite BFS — add max_depth=10 default`
   Includes both Bug A (`max_depth=10` default in `_SlowGrowingTree.__init__` and
   `SlowGrowingTree.__init__`) and Bug B (`base_params is not None` guard in
   `_BaggingWrapper.fit`). Both were runtime.py changes committed atomically.

2. `713fa341` — `test(c63): update mf.models __all__ count after C64 promotions (22 → 30)`

---

## Test Results

```
tests/promotion/ — 388 passed, 7 deselected (slow/deep/heavy)
```

Specifically verified:
- `TestSlowGrowingTree::test_ST3_smoke_fit_predict` — PASSED (was timeout)
- `TestBagging::test_BA3_smoke_fit_predict` — PASSED (was TypeError)
- All other C63 + C64 promotion tests still pass

---

## Design Choices

- Bug A fix: Option 1 (max_depth=10) was chosen over Option 2 (Herfindahl
  comparison inversion) because the Herfindahl check is logically correct —
  the paper says "split when H < H-bar (low concentration = many effective
  rows)". With soft weights the Herfindahl condition is vacuously true, so
  a depth cap is the appropriate structural fix. Option 3 (no-improvement
  plateau) was not chosen because `_best_split` already returns None when
  no SSE improvement is possible, but this only fires when the tree is
  already over-fitted and homogeneous, not early enough for default data.

- Both defaults (private and public class) must agree. Changing only the
  private class would be silently overridden by the sklearn restore pattern
  in the public class.

---

## Known Limitations

- `max_depth=None` with SGT and soft weights (eta < 1) will still be
  unbounded in theory. This is documented in the updated `SlowGrowingTree`
  docstring as "not recommended". For the default use case (eta=0.1,
  herfindahl_threshold=0.25), max_depth=10 gives reasonable tree depth.

- The uv.lock file was updated by `uv pip install pytest pytest-timeout`
  during smoke validation. These are test-only dependencies and do not
  affect production use.
