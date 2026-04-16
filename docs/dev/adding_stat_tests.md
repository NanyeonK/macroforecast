# Adding a New Statistical Test

## Process

### Step 1: Registry entry
Add to `macrocast/registry/tests/stat_test.py`.

### Step 2: Test implementation
Add a function in `macrocast/execution/build.py` following the pattern:

```python
def _compute_my_test(predictions, recipe):
    # Extract model and benchmark errors
    model_errors = predictions["error"].to_numpy()
    benchmark_errors = predictions["benchmark_error"].to_numpy()
    # Compute test statistic and p-value
    ...
    return {"test": "my_test", "statistic": stat, "p_value": pval}
```

Register in `stat_dispatch` dictionary.

### Step 3: Tests
Add to `STAT_TESTS` list in `tests/test_full_operational_sweep.py`.

### Step 4: Documentation
- `docs/user_guide/stat_tests.md` — when to use, which models
- `docs/math/stat_tests.md` — H0, test statistic, distribution, reference
