# Experiment Object

`Experiment` is the public object that should hide internal recipe complexity while preserving reproducibility.

Target responsibilities:

- collect minimum required user inputs
- resolve a named default profile
- accept model comparisons
- accept targeted sweeps
- accept custom extension names
- lower to one recipe or a sweep plan
- run through the execution engine
- expose forecasts, metrics, artifacts, and manifest paths

Target shape:

```python
exp = (
    mc.Experiment(
        dataset="fred_md",
        target="INDPRO",
        horizons=[1, 3, 6],
        start="1980-01",
        end="2019-12",
    )
    .compare_models(["ar", "ridge"])
    .sweep({"scaling": ["none", "standard"]})
)
```

Implementation note:

The `Experiment` object should be a facade over existing compiler and execution components. It should not duplicate the registry or runtime logic.
