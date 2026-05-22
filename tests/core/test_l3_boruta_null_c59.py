"""Cycle 59 — Boruta null calibration and signal calibration tests.

Produced by the StatsClaw tester pipeline (Cycle 59). Tests are derived
exclusively from test-spec.md (Cycle 59). The tester does NOT read spec.md
or implementation.md — results are validated purely against behavioral
contracts.

Coverage:
  NULL  — test_boruta_null_baseline: shuffled-label DGP, 30 seeds, FP rate <= 5%
  SIG   — test_boruta_signal_calibrated: tightened C47 threshold,
          recall >= 3/4 AND precision >= 0.5 on the 4-relevant-feature DGP

References:
  Kursa, M.B. & Rudnicki, W.R. (2010) "Feature Selection with the Boruta
  Package." Journal of Statistical Software 36(11): 1-13.
  doi:10.18637/jss.v036.i11
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from macroforecast.core.runtime import _boruta_selection


# ---------------------------------------------------------------------------
# DGP helpers
# ---------------------------------------------------------------------------

def _make_shuffled_label_panel(
    n_obs: int = 120,
    n_features: int = 20,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.Series]:
    """Null DGP: X is random noise, y is completely independent of X.

    Achieved by shuffling the feature matrix rows with a distinct permutation
    from the target — no feature has any true signal. This is the canonical
    null hypothesis for Boruta: any selected feature is a false positive.

    Args:
        n_obs: Number of observations (rows).
        n_features: Number of candidate features.
        seed: Base random seed.

    Returns:
        (frame, y): DataFrame of predictors and a target Series.
    """
    rng = np.random.default_rng(seed)
    idx = pd.RangeIndex(n_obs)
    X = rng.standard_normal((n_obs, n_features))
    cols = [f"z{i}" for i in range(n_features)]
    frame = pd.DataFrame(X, index=idx, columns=cols)
    # Target is drawn independently — shuffle rows of X differently
    y_raw = rng.standard_normal(n_obs)
    # Explicitly permute y to guarantee zero correlation with any column
    y_perm = rng.permutation(y_raw)
    y = pd.Series(y_perm, index=idx, name="y_null")
    return frame, y


def _make_signal_panel(
    n_obs: int = 120,
    n_features: int = 20,
    n_relevant: int = 4,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Signal DGP: y depends on the first n_relevant features; rest are noise.

    Matches the fixture used in C47 (test_l3_feature_selection_c47.py) so
    the tightened threshold (recall >= 3/4 AND precision >= 0.5) is evaluated
    on the same data distribution.

    y = x0 + x1 + 0.5 * x2 + 0.3 * x3 + noise(sigma=0.5)
    Remaining columns (x4..x{n_features-1}) are pure Gaussian noise.
    """
    rng = np.random.default_rng(seed)
    idx = pd.RangeIndex(n_obs)
    X = rng.standard_normal((n_obs, n_features))
    cols = [f"x{i}" for i in range(n_features)]
    frame = pd.DataFrame(X, index=idx, columns=cols)
    noise = rng.standard_normal(n_obs) * 0.5
    y = (
        frame["x0"]
        + frame["x1"]
        + 0.5 * frame["x2"]
        + 0.3 * frame["x3"]
        + noise
    )
    y.name = "y"
    return frame, y


# ---------------------------------------------------------------------------
# NULL — False-positive rate test (30 seeds)
# ---------------------------------------------------------------------------

class TestBorutaNullBaseline:
    """NULL: Shuffled-label DGP — any accepted feature is a false positive.

    Test-spec.md §A — test_boruta_null_baseline:
      - DGP: n_obs=120, n_features=20, y independent of X
      - 30 independent seeds (0..29)
      - For each seed, run Boruta with max_iter=50, n_estimators_rf=50
      - FP rate = (seeds with >= 1 accepted feature) / 30
      - Acceptance criterion: FP rate <= 5%  (i.e. at most 1 seed out of 30)

    Tolerances from test-spec.md (used exactly as specified):
      - FP threshold: 0.05 (atol, hard maximum)
      - Note: 5% = 1.5 seeds, so in practice <= 1 seed with FP is allowed.
    """

    N_SEEDS: int = 30
    N_OBS: int = 120
    N_FEATURES: int = 20
    MAX_ITER: int = 50
    N_ESTIMATORS: int = 50
    ALPHA: float = 0.05
    FP_THRESHOLD: float = 0.05  # tolerance from test-spec.md — MUST NOT be relaxed

    def test_boruta_null_baseline(self) -> None:
        """FP rate across 30 shuffled-label seeds must be <= 5%.

        Tolerance (from test-spec.md): FP rate <= 0.05
        Used exactly: atol = 0.05
        """
        fp_seeds: list[int] = []

        for seed in range(self.N_SEEDS):
            frame, y = _make_shuffled_label_panel(
                n_obs=self.N_OBS,
                n_features=self.N_FEATURES,
                seed=seed,
            )
            result = _boruta_selection(
                frame,
                target=y,
                params={
                    "n_estimators_rf": self.N_ESTIMATORS,
                    "max_iter": self.MAX_ITER,
                    "alpha": self.ALPHA,
                    "include_tentative": False,  # strict: accepted only, no tentative
                    "random_state": seed,
                },
            )
            # Any non-empty selection on the null DGP is a false positive
            # (edge-case fallback: even if no feature passes, the implementation
            # returns the best-hit feature — count that as a false positive too
            # only if the fallback triggered, i.e. all features were tentative/rejected
            # but the fallback returned 1 anyway)
            # We count FP as: any feature *formally accepted* (status==1), i.e.
            # include_tentative=False so result is only accepted features OR the
            # fallback-1-feature. In both cases, if result.shape[1] >= 1 on the
            # null DGP, it is a FP (conservative definition, per test-spec.md).
            #
            # NOTE: The Boruta fallback (argmax hit_count when nothing accepted)
            # fires when truly nothing is accepted. To distinguish, we set
            # include_tentative=False and trust that the fallback is infrequent
            # enough that the overall rate stays below 5%.
            if result.shape[1] >= 1:
                fp_seeds.append(seed)

        fp_rate: float = len(fp_seeds) / self.N_SEEDS

        assert fp_rate <= self.FP_THRESHOLD, (
            f"Boruta null-DGP FP rate = {fp_rate:.3f} "
            f"(seeds with >= 1 accepted feature: {fp_seeds}) "
            f"> threshold {self.FP_THRESHOLD} from test-spec.md. "
            f"Tolerance used: atol=0.05 (unchanged from spec)."
        )


# ---------------------------------------------------------------------------
# SIG — Signal calibration test (tightened C47 threshold)
# ---------------------------------------------------------------------------

class TestBorutaSignalCalibrated:
    """SIG: Tightened C47 threshold — recall >= 3/4 AND precision >= 0.5.

    Test-spec.md §A — test_boruta_signal_calibrated:
      - DGP: same 4-relevant-feature panel as C47 (n_obs=120, n_features=20,
             n_relevant=4, seed=42)
      - Run Boruta with n_estimators_rf=100, max_iter=100, alpha=0.05
      - Relevant features: x0, x1, x2, x3
      - Acceptance criteria (from test-spec.md, NOT relaxed):
          * recall  = |selected ∩ relevant| / |relevant|  >= 3/4 = 0.75
          * precision = |selected ∩ relevant| / |selected| >= 0.5

    C47 had a weaker threshold (recall >= 2/4 = 0.5). C59 tightens to 3/4.
    """

    N_OBS: int = 120
    N_FEATURES: int = 20
    N_RELEVANT: int = 4
    SEED: int = 42
    N_ESTIMATORS: int = 100
    MAX_ITER: int = 100
    ALPHA: float = 0.05
    RECALL_THRESHOLD: float = 3.0 / 4.0  # 0.75 — from test-spec.md, exact
    PRECISION_THRESHOLD: float = 0.5      # from test-spec.md, exact

    def test_boruta_signal_calibrated(self) -> None:
        """Recall >= 3/4 AND precision >= 0.5 on the 4-relevant-feature DGP.

        Tolerances from test-spec.md (used exactly as specified):
          - recall threshold:    3/4 = 0.75 (hard minimum)
          - precision threshold: 0.5 (hard minimum)
        """
        frame, y = _make_signal_panel(
            n_obs=self.N_OBS,
            n_features=self.N_FEATURES,
            n_relevant=self.N_RELEVANT,
            seed=self.SEED,
        )
        result = _boruta_selection(
            frame,
            target=y,
            params={
                "n_estimators_rf": self.N_ESTIMATORS,
                "max_iter": self.MAX_ITER,
                "alpha": self.ALPHA,
                "include_tentative": False,
                "random_state": self.SEED,
            },
        )

        relevant: set[str] = {f"x{i}" for i in range(self.N_RELEVANT)}
        selected: set[str] = set(result.columns)
        n_relevant_selected: int = len(relevant & selected)
        n_selected: int = len(selected)

        recall: float = n_relevant_selected / self.N_RELEVANT
        precision: float = n_relevant_selected / n_selected if n_selected > 0 else 0.0

        assert recall >= self.RECALL_THRESHOLD, (
            f"Boruta signal recall = {recall:.3f} ({n_relevant_selected}/{self.N_RELEVANT} "
            f"relevant features recovered: {relevant & selected}) "
            f"< threshold {self.RECALL_THRESHOLD} from test-spec.md. "
            f"Selected: {selected}. "
            f"Tolerance used: recall >= 3/4 = 0.75 (unchanged from spec)."
        )

        assert precision >= self.PRECISION_THRESHOLD, (
            f"Boruta signal precision = {precision:.3f} "
            f"({n_relevant_selected} relevant out of {n_selected} selected) "
            f"< threshold {self.PRECISION_THRESHOLD} from test-spec.md. "
            f"Selected: {selected}. "
            f"Tolerance used: precision >= 0.5 (unchanged from spec)."
        )
