# Add A Custom Preprocessor

Custom preprocessors run inside the same split discipline as built-in preprocessing.

```python
import macrocast as mc

@mc.custom_preprocessor("center_x")
def center_x(X_train, y_train, X_test, context):
    location = X_train.mean(axis=0)
    return X_train - location, X_test - location
```

Use the registered name in an experiment:

```python
result = (
    mc.Experiment(
        dataset="fred_md",
        target="INDPRO",
        start="1980-01",
        end="2019-12",
        horizons=[1, 3, 6],
        model_family="ridge",
    )
    .use_preprocessor("center_x")
    .run()
)
```

MVP contract:

```python
fn(X_train, y_train, X_test, context) -> (X_train_new, X_test_new)
```

Rules:

- Fit preprocessing decisions on `X_train` only.
- Return transformed `X_train` and `X_test`.
- Do not transform `y_train`.
- Use `y_train` only as read-only context for target-aware feature preprocessing.
- Keep the number of training rows aligned with `y_train`.

Use a fixed custom preprocessor while comparing models:

```python
result = (
    mc.Experiment(
        dataset="fred_md",
        target="INDPRO",
        start="1980-01",
        end="2019-12",
        model_family="ridge",
    )
    .use_preprocessor("center_x")
    .compare_models(["ridge", "lasso"])
    .run()
)
```

Preprocessor sweeps are blocked in the simple runtime until the preprocessing layer audit is complete. Target transformations are a separate contract. The MVP preprocessor API is predictor-only by design.
