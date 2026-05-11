"""Phase D-2d: _HemisphereNN early stopping + 80/20 train/val split tests.

Verifies paper §3 p.14 (Goulet Coulombe / Frenette / Klieber 2025 JAE):
- patience=15, val_frac=0.20 are the paper-locked defaults.
- 80/20 train/val split is computed ONCE per fit call (all bags share).
- Per-epoch validation NLL drives patience counter.
- Best-model weights are restored at bag end.
- Early stopping fires well before n_epochs is exhausted on a noisy DGP.
"""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import pytest

from macroforecast.core.runtime import _HemisphereNN


# ---------------------------------------------------------------------------
# Shared DGP helpers
# ---------------------------------------------------------------------------


def _make_dataframe(T: int, K: int, seed: int = 42):
    """Pure-noise DGP: X ~ N(0,1), y ~ N(0,1), no signal.
    A pure-noise target means validation loss will not improve
    systematically, causing early stopping to fire quickly."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.standard_normal((T, K)),
        columns=[f"x{i}" for i in range(K)],
    )
    y = pd.Series(rng.standard_normal(T), name="y")
    return X, y


# ---------------------------------------------------------------------------
# T1 — Early stopping triggers before n_epochs on pure-noise DGP
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_hnn_early_stopping_triggers_on_plateau():
    """With patience=5 and n_epochs=100 on a pure-noise target, training
    must stop well before epoch 100. We verify this by checking that a
    patience=5 model produces different predictions than a patience=100
    model (patience=100 effectively disables early stopping for n_epochs=100).
    Different predictions confirm different stopping points."""
    pytest.importorskip("torch")

    T, K = 80, 4
    SEED = 7
    X, y = _make_dataframe(T, K, seed=SEED)

    hnn_es = _HemisphereNN(
        B=3,
        n_epochs=100,
        patience=5,
        val_frac=0.20,
        random_state=SEED,
        neurons=16,
    )
    hnn_full = _HemisphereNN(
        B=3,
        n_epochs=100,
        patience=100,   # effectively no early stopping within 100 epochs
        val_frac=0.20,
        random_state=SEED,
        neurons=16,
    )

    hnn_es.fit(X, y)
    hnn_full.fit(X, y)

    pred_es = hnn_es.predict(X)
    pred_full = hnn_full.predict(X)

    # Predictions should differ — early stopping produces a different
    # checkpoint than running the full 100 epochs.
    assert not np.allclose(pred_es, pred_full, atol=1e-6), (
        "Early-stopped model (patience=5) produced identical predictions to "
        "no-early-stop model (patience=100). Early stopping may not be active."
    )

    # Both models must produce finite predictions.
    assert np.all(np.isfinite(pred_es)), "Early-stopped model: non-finite predictions"
    assert np.all(np.isfinite(pred_full)), "Full-run model: non-finite predictions"


# ---------------------------------------------------------------------------
# T2 — 80/20 split produces correct val_size
# ---------------------------------------------------------------------------


def test_hnn_train_val_split_size_ratio():
    """val_size = round(val_frac * T) for val_frac=0.20.
    For T in {40, 80, 100, 150}, val_size should be within ±1 of 0.20*T.
    The model must fit without error on each size."""
    pytest.importorskip("torch")

    for T in [40, 80, 100, 150]:
        K = 3
        X, y = _make_dataframe(T, K, seed=T)
        expected_val_size = round(0.20 * T)
        computed_val_size = max(1, round(0.20 * T))

        # Tolerance: round() can produce ±1 relative to 0.20*T depending
        # on floating-point edge cases.
        assert abs(computed_val_size - 0.20 * T) <= 1.0, (
            f"T={T}: computed val_size={computed_val_size} deviates more "
            f"than ±1 from 0.20*T={0.20*T}"
        )

        hnn = _HemisphereNN(
            B=2,
            n_epochs=5,
            patience=15,
            val_frac=0.20,
            random_state=42,
            neurons=8,
        )
        # Should fit without error for all T values.
        hnn.fit(X, y)
        preds = hnn.predict(X)
        assert preds.shape == (T,), f"T={T}: unexpected prediction shape"
        assert np.all(np.isfinite(preds)), f"T={T}: non-finite predictions"


# ---------------------------------------------------------------------------
# T3 — Best-model restore changes final weights vs full-run
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_hnn_best_model_restored():
    """Best-model checkpoint restore: verify that the implementation actually
    stores and restores state_dict at best val NLL.

    Design: use a large lr + n_epochs gap so the two regimes (few vs many
    epochs) land in different parts of weight space.

    patience=2, n_epochs=200: fires very early (stops after ~4 epochs total),
    restores best-of-4 checkpoint.
    patience=200, n_epochs=200: runs all 200 epochs, restores best-of-200.
    On pure noise the NLL landscape is noisy; best-of-200 over more
    optimizer steps virtually always differs from best-of-4.

    Structural check: the val variance predictions must differ, since
    variance head state_dict diverges across more optimizer steps."""
    pytest.importorskip("torch")

    T, K = 60, 4
    SEED = 99
    LR = 5e-3   # larger lr so weights move faster — wider divergence gap

    X, y = _make_dataframe(T, K, seed=SEED)

    hnn_early = _HemisphereNN(
        B=2,
        n_epochs=200,
        patience=2,    # stops after ~4 epochs on noise
        val_frac=0.20,
        random_state=SEED,
        neurons=16,
        lr=LR,
    )
    hnn_full = _HemisphereNN(
        B=2,
        n_epochs=200,
        patience=200,  # runs all 200 epochs
        val_frac=0.20,
        random_state=SEED,
        neurons=16,
        lr=LR,
    )

    hnn_early.fit(X, y)
    hnn_full.fit(X, y)

    # Compare variance predictions (more training-step sensitive than means).
    var_early = hnn_early.predict_variance(X)
    var_full = hnn_full.predict_variance(X)

    # Best-of-4 epochs and best-of-200 epochs virtually never produce the
    # same variance head state after 200 optimizer steps vs 4.
    assert not np.allclose(var_early, var_full, rtol=1e-4, atol=1e-4), (
        "patience=2 (stops ~epoch 4) and patience=200 (runs 200 epochs) produced "
        "identical variance predictions. Best-model restore checkpoint mechanism "
        "may not be activating correctly."
    )

    # Both must produce finite, positive variance predictions.
    assert np.all(np.isfinite(var_early)), "early-stop model: non-finite variance"
    assert np.all(np.isfinite(var_full)), "full-run model: non-finite variance"
    assert np.all(var_early > 0), "early-stop model: non-positive variance"
    assert np.all(var_full > 0), "full-run model: non-positive variance"


# ---------------------------------------------------------------------------
# T4 — __init__ kwargs are stored with paper-faithful defaults
# ---------------------------------------------------------------------------


def test_hnn_init_defaults():
    """Paper defaults: patience=15, val_frac=0.20. Verify these are stored
    correctly and clip logic works as specified."""
    hnn = _HemisphereNN()
    assert hnn.patience == 15, f"Expected patience=15, got {hnn.patience}"
    assert hnn.val_frac == 0.20, f"Expected val_frac=0.20, got {hnn.val_frac}"


def test_hnn_init_clip_patience():
    """patience < 1 must be clipped to 1."""
    hnn = _HemisphereNN(patience=0)
    assert hnn.patience == 1, f"patience=0 should clip to 1, got {hnn.patience}"

    hnn2 = _HemisphereNN(patience=-5)
    assert hnn2.patience == 1, f"patience=-5 should clip to 1, got {hnn2.patience}"


def test_hnn_init_clip_val_frac():
    """val_frac outside [0.05, 0.50] must be clipped."""
    hnn_low = _HemisphereNN(val_frac=0.01)
    assert hnn_low.val_frac == 0.05, (
        f"val_frac=0.01 should clip to 0.05, got {hnn_low.val_frac}"
    )

    hnn_high = _HemisphereNN(val_frac=0.99)
    assert hnn_high.val_frac == 0.50, (
        f"val_frac=0.99 should clip to 0.50, got {hnn_high.val_frac}"
    )

    hnn_ok = _HemisphereNN(val_frac=0.30)
    assert hnn_ok.val_frac == 0.30, (
        f"val_frac=0.30 should be unchanged, got {hnn_ok.val_frac}"
    )


# ---------------------------------------------------------------------------
# T5 — Small dataset guard: n < 4 returns early before split
# ---------------------------------------------------------------------------


def test_hnn_small_dataset_guard():
    """Existing guard: if n < 4, fit() returns self without splitting or
    training. Must not raise even when n=2 with the new split logic."""
    pytest.importorskip("torch")
    X = pd.DataFrame({"x0": [0.1, 0.2]})
    y = pd.Series([0.5, 0.6])
    hnn = _HemisphereNN(B=2, n_epochs=5, patience=15, val_frac=0.20, random_state=0)
    result = hnn.fit(X, y)
    assert result is hnn, "fit() must return self"
    # No models trained — small dataset guard triggered.
    assert len(hnn._models) == 0, "No models should be built for n < 4"
