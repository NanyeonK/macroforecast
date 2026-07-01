"""Panel-input models (VAR family) must run through the runner.

The runner's panel branch fits panel-input models by passing a ``DataBundle``
(``model_spec(DataBundle(panel, metadata))``). ``var``/``bvar_*`` call
``as_frame`` on that argument, which previously did not unwrap a ``DataBundle``
and raised ``TypeError: float() argument ... not 'DataBundle'``. Only the DFM
worked (it unwraps via ``_coerce_panel_with_metadata``). This pins that the VAR
family runs end-to-end through ``forecasting.run``.
"""
import numpy as np
import pandas as pd
import pytest

import macroforecast as mf


def _bundle(n=180):
    idx = pd.date_range("1990-01-01", periods=n, freq="MS")
    rng = np.random.default_rng(0)
    panel = pd.DataFrame({f"s{i}": np.cumsum(rng.normal(size=n)) for i in range(4)}, index=idx)
    panel["Y"] = 0.5 * panel["s0"] + rng.normal(size=n)
    panel.index.name = "date"
    return mf.data.custom_dataset(panel, transform_codes={c: 1 for c in panel.columns})


@pytest.mark.parametrize("model", ["var", "bvar_minnesota", "bvar_normal_inverse_wishart"])
def test_var_family_runs_through_runner(model):
    win = mf.window.from_cutoffs(
        test_start="2000-01-01", test_end="2000-06-01", mode="expanding",
        val_method="last_block", retrain_every=1,
    )
    report = mf.forecasting.run(
        _bundle(), model, window=win, features=None, target="Y",
        horizons=[6], forecast_policy="direct_average",
    )
    fc = report.to_frame().dropna(subset=["prediction"])
    assert not fc.empty  # produced forecasts, did not crash on the DataBundle
