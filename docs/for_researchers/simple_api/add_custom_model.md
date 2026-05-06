# Add A Custom Model

Custom models are registered in Python. No registry file edit is needed.

```python
import macroforecast as mf

@mf.custom_model("my_model")
def my_model(X_train, y_train, X_test, context):
    model = MyModel()
    model.fit(X_train, y_train)
    return model.predict(X_test)
```

Use the registered name exactly like a built-in model:

```python
result = (
    mf.Experiment(
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

- `contract_version`
- `model_name`
- `target`
- `horizon`
- `feature_names`
- `feature_runtime_builder`
- `block_order`
- `block_roles`
- `alignment`
- `leakage_contract`
- `mode`

Some routes add optional fields. FRED-SD mixed-frequency routes add
`context["auxiliary_payloads"]` with native-frequency block metadata.

```python
@mf.custom_model("my_fred_sd_mixed_frequency_model")
def my_fred_sd_mixed_frequency_model(X_train, y_train, X_test, context):
    payloads = context.get("auxiliary_payloads", {})
    blocks = payloads["fred_sd_native_frequency_block_payload"]
    native_frequency = blocks["column_to_native_frequency"]
    # Fit a leakage-free mixed-frequency research model here.
    return float(y_train[-1])

result = (
    mf.Experiment(
        dataset="fred_sd",
        target="UR_CA",
        start="2000-01",
        end="2020-12",
        horizons=[1],
        frequency="monthly",
        feature_builder="raw_feature_panel",
        model_family="my_fred_sd_mixed_frequency_model",
    )
    .use_fred_sd_selection(states=["CA", "TX"], variables=["UR", "NQGSP"])
    .use_fred_sd_mixed_frequency_adapter()
    .run(local_raw_source="tests/fixtures/fred_sd_sample.csv")
)
```

See `examples/custom_fred_sd_mixed_frequency_model.py` and
`examples/recipes/templates/fred-sd-custom-mixed-frequency-model.yaml`.
When using YAML, import the Python module that registers the model before
running the recipe. YAML selects `model_family`; it does not register the
callable.

The manifest records the model as custom:

```python
manifest = result.variant(result.per_variant_results[0].variant_id).manifest
manifest["model_spec"]["custom_model"]
```

Keep custom model code leakage-free. Fit only on `X_train` and `y_train`; do not read future rows or full-sample statistics.
