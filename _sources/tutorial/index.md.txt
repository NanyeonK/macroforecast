# Tutorials

New to macroforecast? The tutorials below start with the standalone model classes in
`macroforecast.models`. No YAML is required for tutorials 01 through 03. Each tutorial
ends with a short section on graduating to the recipe pipeline when reproducibility or
systematic sweeps are needed.

Tutorial 01 fits a linear AR model on a synthetic series in under five minutes. Tutorial
02 compares three model families using scikit-learn's `TimeSeriesSplit`. Tutorial 03 shows
how to subclass `BaseEstimator` and `RegressorMixin` to define a custom forecasting model.
For installation instructions, start with tutorial 00. For a conceptual comparison of the
standalone and recipe entry points, see {doc}`two_entry_points`.

```{toctree}
:maxdepth: 1

00_install
01_first_forecast
02_full_study
03_custom_model
04_custom_preprocessor
two_entry_points
replications/index
```
