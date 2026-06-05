"""F1/F2: the estimation/test embargo convention is surfaced, not silently leaked."""
import pandas as pd

import macroforecast as mf


def _validate(horizon, embargo=0):
    w = mf.window.from_cutoffs(
        test_start=40, horizon=horizon, embargo=embargo,
        val_method="expanding", val_min_train_size=10,
    )
    return w.validate(pd.RangeIndex(60))


def test_default_embargo_warns_for_multistep():
    rep = _validate(horizon=3)
    assert rep["ok"]  # warning, not error
    assert any("pseudo-out-of-sample" in w for w in rep["warnings"])


def test_no_warning_for_one_step():
    rep = _validate(horizon=1)
    assert not any("pseudo-out-of-sample" in w for w in rep["warnings"])


def test_strict_realtime_embargo_silences_warning():
    # embargo = horizon - 1 makes the last training label observable at the origin
    rep = _validate(horizon=3, embargo=2)
    assert not any("pseudo-out-of-sample" in w for w in rep["warnings"])
