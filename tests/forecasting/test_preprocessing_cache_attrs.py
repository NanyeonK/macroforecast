"""Regression test for the shared-preprocessing-cache ``.attrs`` deepcopy blow-up.

A multi-horizon pseudo-out-of-sample run shares one ``preprocessing_cache`` across
all horizons (``runner._run_multiple_horizons``). Each per-origin transformed panel
is stored in that cache as a ``_PreparedStage`` and is then reindexed / dropna'd /
fed to feature fit+transform once per origin AND reused across arms and horizons.

The defect was that those cached panels carried the full ``macroforecast_metadata``
payload (EM/factor + standardization state, ~40 KB) on ``panel.attrs``. pandas
deep-copies ``.attrs`` on essentially every operation via ``__finalize__``, so the
per-origin hot loop paid an O(origins x arms x horizons) deepcopy of that large
metadata. With enough origins the run appeared to hang inside ``copy.deepcopy ->
_deepcopy_dict``. The metadata does not grow per origin; it is simply large, and
the cost is paid once per pandas operation on every cached panel.

The fix (``runner._detach_panel_metadata``) keeps the full metadata on the
``_PreparedStage`` dataclass and re-attaches it only where consumed, leaving the
cached panel's ``.attrs`` holding just the small transform-code map. These tests
assert that the cached panels carry no heavy metadata on ``.attrs`` and that the
per-panel ``.attrs`` size stays bounded regardless of how many origins are run.
"""
from __future__ import annotations

import copy
import pickle

import numpy as np
import pandas as pd

import macroforecast as mf
from macroforecast.forecasting.runner import _PreparedStage


def _bundle(n: int = 200, n_cols: int = 30):
    idx = pd.date_range("1990-01-31", periods=n, freq="ME", name="date")
    rng = np.random.default_rng(0)
    data = rng.standard_normal((n, n_cols)).cumsum(axis=0)
    # A few missing cells so the EM/factor imputation (the source of the heavy
    # ``transform_state`` metadata) actually engages.
    data[rng.integers(0, n, 20), rng.integers(1, n_cols, 20)] = np.nan
    cols = ["TARGET"] + [f"X{i}" for i in range(n_cols - 1)]
    frame = pd.DataFrame(data, index=idx, columns=cols)
    codes = {c: 1 for c in cols}
    return mf.data.custom_dataset(frame, transform_codes=codes)


def _window(test_start: str, test_end: str, *, horizon: int = 1):
    return mf.window.from_cutoffs(
        estimation_start="1990-01",
        test_start=test_start,
        test_end=test_end,
        mode="expanding",
        horizon=horizon,
        embargo=0,
        retrain_every=12,
        val_method="last_block",
        val_size=12,
    )


def _features():
    return mf.feature_engineering.feature_spec(
        target="TARGET", lags=1, target_lags=(0, 1)
    )


def _cached_panels(cache: dict) -> list[pd.DataFrame]:
    return [v.panel for v in cache.values() if isinstance(v, _PreparedStage)]


def _attrs_bytes(panel: pd.DataFrame) -> int:
    return len(pickle.dumps(dict(panel.attrs)))


def _run(cache, *, test_end: str, horizons):
    return mf.forecasting.run(
        _bundle(),
        "ridge",
        window=_window("2000-01", test_end, horizon=1),
        preprocessing=mf.preprocessing.preprocess_spec(
            impute="em_factor", standardize="zscore"
        ),
        features=_features(),
        target="TARGET",
        horizons=horizons,
        preprocessing_cache=cache,
        save_models=False,
    )


def test_cached_prepared_panels_do_not_carry_heavy_metadata_attrs() -> None:
    """Cached per-origin panels must not carry ``macroforecast_metadata`` on attrs.

    That payload is what pandas deep-copies on every panel operation in the
    per-origin hot loop; leaving it on the cached panel reintroduces the hang.
    """
    cache: dict = {}
    _run(cache, test_end="2003-01", horizons=[1, 3])

    panels = _cached_panels(cache)
    assert panels, "expected the shared cache to hold prepared-stage panels"

    for panel in panels:
        # The heavy metadata must live on the dataclass, never on the panel attrs.
        assert "macroforecast_metadata" not in panel.attrs
        # Deep-copying the cached panel's attrs must be cheap and bounded. The only
        # surviving key is the small transform-code map.
        assert _attrs_bytes(panel) < 4000
        copy.deepcopy(panel.attrs)  # must not blow up

    # The full metadata is still available to callers via the dataclass field.
    stages = [v for v in cache.values() if isinstance(v, _PreparedStage)]
    assert all(isinstance(s.panel_metadata, dict) for s in stages)


def test_cached_attrs_size_does_not_grow_with_number_of_origins() -> None:
    """Per-panel attrs size must be bounded and independent of origin count.

    A short run (few origins) and a longer run (many more origins) must produce the
    same bounded per-panel ``.attrs`` size. On the buggy code the panels carried the
    full ~40 KB metadata regardless; this test pins the size down so a regression
    that re-attaches the heavy payload (or one that makes it accumulate across
    origins) fails loudly instead of merely getting slow.
    """
    short_cache: dict = {}
    _run(short_cache, test_end="2002-06", horizons=[1, 3])
    long_cache: dict = {}
    _run(long_cache, test_end="2005-01", horizons=[1, 3])

    short_panels = _cached_panels(short_cache)
    long_panels = _cached_panels(long_cache)
    assert short_panels and long_panels
    # The longer run genuinely processes more origins.
    assert len(long_panels) > len(short_panels)

    short_max = max(_attrs_bytes(p) for p in short_panels)
    long_max = max(_attrs_bytes(p) for p in long_panels)

    # Bounded and stable: running far more origins does not enlarge any single
    # cached panel's deep-copied attrs.
    assert long_max < 4000
    assert long_max == short_max
