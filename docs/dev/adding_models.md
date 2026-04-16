# Adding a New Model Family

## 5-step process

### Step 1: Registry entry
Add to `macrocast/registry/training/model_family.py`:
```python
EnumRegistryEntry(
    id="my_new_model",
    description="My new model",
    status="operational",
    priority="A",
),
```

### Step 2: Executor functions
Add two executor functions in `macrocast/execution/build.py`:

```python
def _run_my_new_model_autoreg_executor(train, horizon, recipe, contract, raw_frame=None, origin_idx=None, start_idx=0):
    lag_order, _, _, model, _tp = _fit_autoreg_sklearn(train, recipe, "my_new_model", MyModel())
    return {"y_pred": _recursive_predict_sklearn(model, train, horizon, lag_order), "selected_lag": lag_order, "selected_bic": math.nan, "tuning_payload": _tp}

def _run_my_new_model_raw_panel_executor(train, horizon, recipe, contract, raw_frame=None, origin_idx=None, start_idx=0):
    _, _, X_pred, model, _tp = _fit_raw_panel_model(raw_frame, recipe, horizon, start_idx, origin_idx, contract, "my_new_model", MyModel())
    return {"y_pred": float(model.predict(X_pred)[0]), "selected_lag": 0, "selected_bic": math.nan, "tuning_payload": _tp}
```

Register both in the dispatch tables (`_AUTOREG_MODEL_DISPATCH`, `_RAW_PANEL_MODEL_DISPATCH`).

### Step 3: HP space
Add to `macrocast/tuning/hp_spaces.py`:
```python
"my_new_model": {
    "param1": HPDistribution("log_float", 1e-4, 1e4, log=True),
},
```

### Step 4: Tests
Add parametrized test entries in `tests/test_full_operational_sweep.py`.

### Step 5: Documentation
- `docs/user_guide/models.md` — add entry with description, HP, when to use
- `docs/math/` — if the model has novel aspects worth formalizing
