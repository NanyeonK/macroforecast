# Paper Tables

[Back to User Guide](index.md)

`macroforecast.reporting` turns a `PipelineReport` into publication-facing
tables without changing the evaluation state. The helpers are post-processing
adapters over report frames, so they work the same way on a freshly run report,
a rescored report, or a report loaded from disk.

## Accuracy Horse-Race Table

Use `paper_accuracy_table` for the standard models-by-horizons accuracy table:
relative RMSE, Diebold-Mariano stars, and Model Confidence Set markers.

```python
table = mf.reporting.paper_accuracy_table(report, target="INDPRO")
print(table.data)
print(table.to_latex(booktabs=True))
```

The function reads `report.accuracy`, `report.significance`, and `report.mcs`.
If the report has multiple targets, pass `target=...`; if it was scored with
`EvalSpec.subsamples`, pass `subsample=...` to select one evaluation window.

## Pairwise Test Matrix

Use `pairwise_test_table` when a paper prints a K-by-K matrix of pairwise test
p-values or statistics across all models, rather than benchmark-vs-contender
rows.

```python
matrix = mf.reporting.pairwise_test_table(
    report,
    target="INDPRO",
    horizon=4,
    models=["AR", "FM", "RF"],
    test="dm",
    value="p_value",
    test_options={"hac_lags": 4},
)
print(matrix)
print(matrix.to_latex())
```

Rows and columns are model names. The diagonal is missing by construction. For
DM p-values the matrix is symmetric; for DM statistics the sign flips when row
and column are swapped. Set `stars=True` to render p-values with significance
markers before calling `to_latex()`.

The adapter recomputes each cell from `report.forecasts` by calling the same
public test functions in `macroforecast.tests` that the pipeline uses. It does
not add pipeline state and does not require rerunning the forecasting stage.

## Reference

- [Reporting reference page](../reference/reporting.md) — table helper signatures.
- [Evaluation guide](concepts/evaluation.md) — test selection and `test_options`.
