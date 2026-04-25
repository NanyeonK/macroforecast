# Add A Custom Model

Custom models are registered in Python. No registry file edit is needed.

```python
import macrocast as mc

@mc.custom_model("my_model")
def my_model(X_train, y_train, X_test, context):
    model = MyModel()
    model.fit(X_train, y_train)
    return model.predict(X_test)
```

Use the registered name exactly like a built-in model:

```python
result = (
    mc.Experiment(
        dataset="fred_md",
        target="INDPRO",
        start="1980-01",
        end="2019-12",
        horizons=[1, 3, 6],
    )
    .compare_models(["ridge", "my_model"])
    .run()
)
```

MVP contract:

```python
fn(X_train, y_train, X_test, context) -> prediction
```

`X_train`, `y_train`, and `X_test` are already split by the execution engine. `X_test` is one row. Return a scalar or a one-element sequence/array.

`context` includes:

- `model_name`
- `feature_builder`
- `target`
- `horizon`
- `feature_names`
- `contract_version`

The manifest records the model as custom:

```python
manifest = result.variant(result.per_variant_results[0].variant_id).manifest
manifest["model_spec"]["custom_model"]
```

Keep custom model code leakage-free. Fit only on `X_train` and `y_train`; do not read future rows or full-sample statistics.
