# Data

[Back to User Guide](../index.md)

`macroforecast.data` is the data entry point for the package. It loads official
or user-supplied data, normalizes it to one pandas panel contract, and attaches
source metadata. The main output is always a `DataBundle` or `DataSpec`. This
module does not apply stationarity transforms, outlier rules, imputation, feature
engineering, model fitting, or evaluation.

The standard panel is a `pandas.DataFrame` with a `DatetimeIndex` named `date`
and one macro series per column. Dataset metadata is kept separately and also
mirrored in `panel.attrs["macroforecast_metadata"]`. `mf.data.load_*()` returns a
`DataBundle(panel, metadata)`. `mf.data.spec(...)` attaches target, horizon,
sample-window, and predictor choices to that panel.

## Key Callables

`mf.data.load_fred_md` downloads or reads the FRED-MD monthly panel, stores
official t-codes in metadata, and returns a `DataBundle`.

`mf.data.spec` builds a `DataSpec` from an already-loaded bundle, recording
which target to forecast, which horizons to use, what date range is active, and
which predictor columns are included.

```python
import macroforecast as mf

# Load the FRED-MD monthly panel (downloads on first call, then caches).
bundle = mf.data.load_fred_md()

# Attach study-level choices: target, horizons, date range, predictors.
data_spec = mf.data.spec(
    bundle,
    target="INDPRO",
    horizons=[1, 3, 6, 12],
    start="1960-01",
    end="2024-12",
    predictors="all",
)
```

## Executed walkthrough

Loading the panel and inspecting its shape, span, and first rows:

```python
bundle = mf.data.load_fred_md()
panel = bundle.panel
print(panel.shape)                          # (rows, series)
print(panel.index.min().date(), panel.index.max().date())
print(panel.iloc[:3, :4])
```

```text
(708, 128)
1959-01-01 2017-12-01
                 RPI  W875RX1  DPCERA3M086SBEA    CMRMTSPLx
date
1959-01-01  2289.932   2151.8           18.191  255861.8850
1959-02-01  2299.790   2160.4           18.380  257783.6485
1959-03-01  2314.456   2176.2           18.555  256866.3717
```

The panel here is the FRED-MD 2018-01 vintage, so it carries 128 series through
2017-12. Exact dimensions and values depend on the vintage you load. Attaching
study choices returns a `DataSpec` that records the target and horizons:

```python
data_spec = mf.data.spec(bundle, target="INDPRO", horizons=[1, 3, 6, 12])
print(data_spec.target, data_spec.horizons)
```

```text
INDPRO (1, 3, 6, 12)
```

## Real-valued panels

A canonical panel is real-valued. Complex columns are rejected by both `as_panel`
and `validate_panel`, including under `strict=False`: the forecasting, statistical, and
model paths downstream are defined over real values, so a complex column has nowhere to
go that keeps its meaning. Split the real and imaginary parts into separate columns if
both matter.

Before that rejection existed the loss happened quietly further in. Complex passed the
numeric check, and the infinity check and the content fingerprint both cast to float, so
two panels differing only in their imaginary values validated and then shared one content
identity. The fingerprint no longer drops them — it hashes complex values in full, as a
backstop for callers that reach it without going through a canonical panel — but that is
a backstop and not a licence. The rejection above is the contract.

`strict=False` still relaxes what it always relaxed: unparseable dates and non-numeric
strings that a permissive load may coerce. It does not widen the value domain.

## Reference

- [Data reference page](../../reference/data.md) — full function list and output contracts.
- [FRED Datasets](../../datasets/index.md) — dataset-specific pages for FRED-MD, FRED-QD, and FRED-SD.
