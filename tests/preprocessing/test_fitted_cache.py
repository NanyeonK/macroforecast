"""Tests for the content-addressed on-disk FittedPreprocessor store.

TDD: these tests were written BEFORE the implementation in
macroforecast/preprocessing/cache.py.
"""
from __future__ import annotations

import os
import pickle
import struct
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from macroforecast.preprocessing.cache import PreprocessorStore, UndigestiblePreprocessorSpec
from macroforecast.preprocessing.specs import PreprocessSpec, preprocess_spec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_spec(**options: Any) -> PreprocessSpec:
    """Create a PreprocessSpec with the given options."""
    return preprocess_spec(**options)


def _dummy_panel() -> pd.DataFrame:
    """Tiny panel suitable for fitting a cheap PreprocessSpec."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=20, freq="MS")
    data = pd.DataFrame(
        {
            "date": dates,
            "y": rng.normal(0, 1, 20),
            "x1": rng.normal(0, 1, 20),
            "x2": rng.normal(0, 1, 20),
        }
    )
    return data


def _fit_real_preprocessor(spec: PreprocessSpec):
    """Fit a real FittedPreprocessor on a small panel via the real API."""
    import macroforecast as mf

    df = _dummy_panel()
    panel = mf.data.as_panel(df, date="date")
    bundle = mf.data.DataBundle(
        panel,
        {"dataset": "custom", "source_family": "custom", "frequency": "monthly"},
    )
    return spec.fit(bundle)


# ---------------------------------------------------------------------------
# 1. Basic round-trip
# ---------------------------------------------------------------------------


def test_store_put_get_roundtrip(tmp_path: Path) -> None:
    """get on a fresh store returns None; after put, get returns an equal object."""
    store = PreprocessorStore(tmp_path)
    key = store.key(_make_spec(impute="mean"), target="y", origin_pos=10)

    # Miss on empty store.
    assert store.get(key) is None

    # Put a simple picklable object.
    payload = {"hello": "world", "nums": [1, 2, 3]}
    store.put(key, payload)

    # Hit returns equal object.
    result = store.get(key)
    assert result == payload

    # A second get also succeeds (not a one-shot).
    assert store.get(key) == payload


def test_store_frame_roundtrip_preserves_index_columns_and_dtypes(tmp_path: Path) -> None:
    """Prepared-base frame payloads round-trip through parquet + JSON metadata."""

    store = PreprocessorStore(tmp_path)
    spec = _make_spec(impute="mean", standardize="zscore")
    key = store.frame_key(
        spec,
        target="y",
        cache_key=("origin_pos", 4),
        kind="prepared_base",
    )
    frame = pd.DataFrame(
        {
            "x": pd.Series([1.0, 2.5, 3.0], dtype="float64").to_numpy(),
            "flag": pd.Series([1, 0, 1], dtype="int64").to_numpy(),
        },
        index=pd.date_range("2020-01-01", periods=3, freq="MS", name="date"),
    )
    frame.attrs["macroforecast_transform_codes"] = {"x": 1, "flag": 1}

    store.put_frame(key, frame)
    loaded = store.get_frame(key)

    assert loaded is not None
    pd.testing.assert_frame_equal(loaded, frame)
    assert loaded.attrs == frame.attrs

    # A torn/corrupt metadata sidecar is a miss, not an exception.
    (tmp_path / f"{key}.json").write_text("{")
    assert store.get_frame(key) is None


# ---------------------------------------------------------------------------
# 2. Spec-hash collision avoidance
# ---------------------------------------------------------------------------


def test_store_distinguishes_spec_hash(tmp_path: Path) -> None:
    """Different (spec, target, origin_pos) tuples must yield different keys."""
    store = PreprocessorStore(tmp_path)

    spec_a = _make_spec(impute="mean", standardize="zscore")
    spec_b = _make_spec(impute="forward_fill", standardize="none")

    key_a = store.key(spec_a, target="y", origin_pos=10)
    key_b = store.key(spec_b, target="y", origin_pos=10)
    key_c = store.key(spec_a, target="z", origin_pos=10)  # different target
    key_d = store.key(spec_a, target="y", origin_pos=20)  # different origin

    # All four keys are distinct.
    assert len({key_a, key_b, key_c, key_d}) == 4

    # Put under key_a; all others remain None.
    store.put(key_a, "value_a")
    assert store.get(key_a) == "value_a"
    assert store.get(key_b) is None
    assert store.get(key_c) is None
    assert store.get(key_d) is None

    # Put under key_b; key_a unchanged, key_c / key_d still None.
    store.put(key_b, "value_b")
    assert store.get(key_a) == "value_a"
    assert store.get(key_b) == "value_b"
    assert store.get(key_c) is None
    assert store.get(key_d) is None


def test_store_uses_mf_digest_for_custom_callable_keys(tmp_path: Path) -> None:
    store = PreprocessorStore(tmp_path)

    func_a = lambda panel, metadata=None: panel  # noqa: E731
    func_b = lambda panel, metadata=None: panel  # noqa: E731
    func_a.__mf_digest__ = "lambda-a"
    func_b.__mf_digest__ = "lambda-b"

    spec_a = _make_spec(custom_steps=[{"name": "step", "func": func_a}])
    spec_b = _make_spec(custom_steps=[{"name": "step", "func": func_b}])

    assert store.key(spec_a, target="y", origin_pos=1) != store.key(
        spec_b,
        target="y",
        origin_pos=1,
    )


def test_preprocess_spec_rejects_undigested_lambda_custom_step() -> None:
    with pytest.raises(ValueError, match="anonymous lambda.*__mf_digest__"):
        _make_spec(custom_steps=[lambda panel, metadata=None: panel])


def test_store_refuses_named_custom_callable_without_digest(tmp_path: Path) -> None:
    def custom_step(panel: pd.DataFrame, metadata=None) -> pd.DataFrame:
        return panel

    spec = _make_spec(custom_steps=[{"name": "custom", "func": custom_step}])
    store = PreprocessorStore(tmp_path)

    with pytest.raises(UndigestiblePreprocessorSpec, match="without __mf_digest__"):
        store.key(spec, target="y", origin_pos=1)


# ---------------------------------------------------------------------------
# 3. Atomic write
# ---------------------------------------------------------------------------


def test_store_atomic_write(tmp_path: Path) -> None:
    """Corrupt / partial files are treated as misses, not exceptions.

    Also verifies that a successful put:
    - leaves no stray tmp files behind
    - is immediately readable via get
    """
    store = PreprocessorStore(tmp_path)
    spec = _make_spec(impute="mean")
    key = store.key(spec, target="gdp", origin_pos=5)

    # --- Simulate a partial/corrupt write at the final path. ---
    # Derive the final path the same way the implementation would.
    final_path = tmp_path / f"{key}.pkl"
    # Write garbage (truncated pickle header).
    final_path.write_bytes(b"\x80\x04")  # truncated pickle stream

    # get must treat this as a miss, not raise.
    result = store.get(key)
    assert result is None

    # --- Real put must overwrite the garbage and succeed. ---
    store.put(key, {"clean": True})
    result2 = store.get(key)
    assert result2 == {"clean": True}

    # No stray .tmp files remain after a successful put.
    tmp_files = list(tmp_path.glob("*.tmp"))
    assert tmp_files == [], f"Stray tmp files: {tmp_files}"

    # --- Verify temp-file + atomic-replace pattern via monkeypatching. ---
    # We patch os.replace to record the src and dst, then restore it,
    # confirming that a temporary file was created and atomically moved.
    replacements: list[tuple[str, str]] = []
    original_replace = os.replace

    def _spy_replace(src: str, dst: str) -> None:
        replacements.append((src, dst))
        original_replace(src, dst)

    os.replace = _spy_replace  # type: ignore[assignment]
    try:
        key2 = store.key(spec, target="gdp", origin_pos=99)
        store.put(key2, "atomic_payload")
    finally:
        os.replace = original_replace  # type: ignore[assignment]

    assert len(replacements) == 1, "Expected exactly one os.replace call per put"
    src_path, dst_path = replacements[0]
    # dst must be the final key path; src must differ (temp file).
    assert dst_path.endswith(f"{key2}.pkl")
    assert src_path != dst_path
    # Temp file is gone after replace.
    assert not Path(src_path).exists()
    # Final file is readable.
    assert store.get(key2) == "atomic_payload"


# ---------------------------------------------------------------------------
# 4. Real FittedPreprocessor round-trip
# ---------------------------------------------------------------------------


def test_store_roundtrips_real_fitted_preprocessor(tmp_path: Path) -> None:
    """A real FittedPreprocessor must survive a store/load cycle intact."""
    spec = _make_spec(
        impute="mean",
        standardize="zscore",
        outliers="none",
        transform="none",
    )
    fitted = _fit_real_preprocessor(spec)

    store = PreprocessorStore(tmp_path)
    key = store.key(spec, target="y", origin_pos=0)

    store.put(key, fitted)
    loaded = store.get(key)

    assert loaded is not None
    # The loaded object is a FittedPreprocessor with the same spec options.
    assert loaded.spec.options == fitted.spec.options
    # The processed training panel matches.
    pd.testing.assert_frame_equal(
        loaded.processed_train.panel,
        fitted.processed_train.panel,
    )
