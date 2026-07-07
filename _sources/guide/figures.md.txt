# Paper Figures

[Back to User Guide](index.md)

Install the plotting extra before using these helpers:

```bash
pip install "macroforecast[plots]"
```

The plotting helpers live under `mf.reporting` and take the same
`PipelineReport` surface as `paper_accuracy_table`: the master forecast frame
from `report.forecasts`, plus the benchmark metadata recorded by the pipeline.
Each helper accepts `ax=` when you want to draw into an existing matplotlib
axis, and `savefig="figure.pdf"` or `savefig="figure.png"` when the paper
artifact should be written directly.

## Cumulative loss differential

The cumulative squared-error differential (CSSED) plot is the standard horse-race
companion to a relative-accuracy table. It cumulates benchmark loss minus
contender loss over forecast target dates, so sustained upward movement means
the contender is reducing loss relative to the benchmark. Referees look for
whether the advantage is persistent, concentrated in a short episode, or
reversed outside the period emphasized by the table.

```python
fig = mf.reporting.cumulative_loss_differential_plot(
    report,
    target="INDPRO",
    horizon=1,
    benchmark="AR",
    contenders=["RF", "Ridge"],
    shade=[("2007-12-01", "2009-06-01")],
    savefig="cssed_indpro_h1.pdf",
)
```

## Giacomini-Rossi fluctuation

The Giacomini-Rossi fluctuation plot shows the rolling loss-differential test
path with the two-sided Table 1 critical-value bands. It is useful when an
average test statistic hides time variation in forecast performance. Referees
usually check whether the path crosses the bands, whether the crossing is
isolated, and whether the window ratio and horizon-aware HAC lag choice are
reported consistently with the study design.

```python
fig = mf.reporting.fluctuation_test_plot(
    report,
    target="INDPRO",
    horizon=3,
    benchmark="AR",
    contender="RF",
    window_ratio=0.5,
    savefig="gr_fluctuation_indpro_h3.pdf",
)
```

## PIT histogram

The probability integral transform (PIT) histogram is the basic density-forecast
calibration exhibit. Under a calibrated Gaussian predictive density, the PITs
should look uniform; the plotted band is the binomial sampling range around the
uniform count. Referees look for U-shapes, spikes near zero or one, and other
departures that imply under-dispersion, over-dispersion, or biased predictive
distributions. The run must include variance-emitting forecasts.

```python
fig = mf.reporting.pit_histogram_plot(
    report,
    target="INDPRO",
    horizon=1,
    model="DensityHNN",
    bins=10,
    savefig="pit_densityhnn_indpro_h1.png",
)
```

## Forecast path

The actual-versus-forecast path plot is the sanity-check exhibit: it overlays
the realized target series with one or more model forecasts over the same target
dates. This plot is not a formal test, but referees use it to spot phase shifts,
stale forecasts, scale mistakes, and episodes where a table average is driven by
one break. When the selected model emits `variance_prediction`, `variance_band=`
adds an approximate 95 percent Gaussian band.

```python
fig = mf.reporting.forecast_path_plot(
    report,
    target="INDPRO",
    horizon=1,
    models=["AR", "RF", "DensityHNN"],
    start="2000-01-01",
    variance_band="DensityHNN",
    savefig="forecast_path_indpro_h1.pdf",
)
```
