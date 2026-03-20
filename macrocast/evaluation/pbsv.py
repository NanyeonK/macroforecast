"""Performance-Based Shapley Value (PBSV) for out-of-sample forecast evaluation.

Implements the PBSV decomposition from:
  Coulombe, Boldea, Renneson, Spierdijk, Spierdijk (2022).
  "Anatomy of Out-of-Sample Gains" (CBRSS 2022).
  Reference implementation: sander-sn/anatomy.

Two complementary tools are provided:

1. **oShapley-VI** (out-of-sample Shapley Variable Importance, Eq. 16):
   Quantifies how much each predictor group contributes to OOS forecast accuracy.
   Uses the Shapley value formula over 2^N subsets, weighted by OOS loss reduction.

2. **PBSV** (Performance-Based Shapley Value, Eq. 19-25):
   Decomposes the total OOS gain of a model over the benchmark into additive
   predictor contributions, evaluated at every test date.  Output is a
   (T_test, N_groups) matrix of time-varying contributions.

Both tools are model-agnostic and loss-agnostic: they work with any callable
that maps (X_train, y_train, X_test) → y_hat_test.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

# Predictor group: a list of column indices belonging to one group
PredictorGroup = list[int]

# Forecast callable: (X_train, y_train, X_test) → y_hat (T_test,)
ForecastFn = Callable[
    [NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]],
    NDArray[np.floating],
]


# ---------------------------------------------------------------------------
# Shapley weight for coalition of size k out of N
# ---------------------------------------------------------------------------


def _shapley_weight(n: int, k: int) -> float:
    """Shapley kernel weight: k!(n-k-1)! / n!"""
    from math import factorial

    return factorial(k) * factorial(n - k - 1) / factorial(n)


# ---------------------------------------------------------------------------
# oShapley Variable Importance
# ---------------------------------------------------------------------------


def oshapley_vi(
    forecast_fn: ForecastFn,
    X_train: NDArray[np.floating],
    y_train: NDArray[np.floating],
    X_test: NDArray[np.floating],
    y_test: NDArray[np.floating],
    groups: Sequence[PredictorGroup],
    loss_fn: Callable | None = None,
    baseline_value: float = 0.0,
) -> NDArray[np.floating]:
    """Out-of-sample Shapley Variable Importance (oShapley-VI).

    Computes the Shapley value of each predictor group with respect to the
    OOS loss function.  Uses exact computation over all 2^N subsets — feasible
    for N ≤ 15 groups; use approximation for larger N.

    Parameters
    ----------
    forecast_fn : callable (X_train, y_train, X_test) → y_hat
        Model fitting and prediction function.  Must accept X matrices with
        arbitrary column subsets.
    X_train : array of shape (T_train, N_total)
        Training features.
    y_train : array of shape (T_train,)
        Training targets.
    X_test : array of shape (T_test, N_total)
        Test features.
    y_test : array of shape (T_test,)
        Realised test values.
    groups : list of lists of int
        Column indices for each predictor group.  len(groups) = N_groups.
    loss_fn : callable (y_true, y_hat) → scalar, optional
        Loss function.  Defaults to negative MSFE (so higher = better).
    baseline_value : float
        Score when no predictors are included.  Default 0 (null forecast).

    Returns
    -------
    phi : array of shape (N_groups,)
        Shapley value for each predictor group.
    """
    if loss_fn is None:

        def loss_fn(yt, yh):
            return -float(np.mean((yt - yh) ** 2))  # negative MSFE (higher = better)

    N = len(groups)
    phi = np.zeros(N)

    # Cache coalition scores to avoid recomputing the same subset multiple times
    score_cache: dict = {}

    def coalition_score(coalition_idx: tuple[int, ...]) -> float:
        """Score for a coalition (subset of group indices)."""
        if coalition_idx in score_cache:
            return score_cache[coalition_idx]
        if len(coalition_idx) == 0:
            score_cache[coalition_idx] = baseline_value
            return baseline_value
        # Gather columns belonging to this coalition
        cols = sorted(set(c for gi in coalition_idx for c in groups[gi]))
        X_tr_sub = X_train[:, cols]
        X_te_sub = X_test[:, cols]
        try:
            y_hat = forecast_fn(X_tr_sub, y_train, X_te_sub)
            s = loss_fn(y_test, y_hat)
        except Exception:
            s = baseline_value
        score_cache[coalition_idx] = s
        return s

    # Shapley formula: average marginal contribution over all orderings
    all_idx = list(range(N))
    for i in range(N):
        phi_i = 0.0
        for size in range(N):
            # All coalitions of size `size` not containing i
            for subset in itertools.combinations([j for j in all_idx if j != i], size):
                w = _shapley_weight(N, len(subset))
                s_with = coalition_score(tuple(sorted(subset + (i,))))
                s_without = coalition_score(tuple(sorted(subset)))
                phi_i += w * (s_with - s_without)
        phi[i] = phi_i

    return phi


# ---------------------------------------------------------------------------
# PBSV: Performance-Based Shapley Value
# ---------------------------------------------------------------------------


def compute_pbsv(
    forecast_fn: ForecastFn,
    X_train_seq: list[NDArray[np.floating]],
    y_train_seq: list[NDArray[np.floating]],
    X_test_seq: NDArray[np.floating],
    y_test_seq: NDArray[np.floating],
    groups: Sequence[PredictorGroup],
    loss_fn: Callable | None = None,
    baseline_value: float = 0.0,
) -> NDArray[np.floating]:
    """Performance-Based Shapley Value decomposition.

    Decomposes the OOS performance gain into additive predictor-group
    contributions for each test date t (CBRSS 2022, Eq. 19-25).

    The PBSV at test date t for group i is the Shapley value of group i
    computed on the OOS loss at that specific date, using the training
    window up to t.

    Parameters
    ----------
    forecast_fn : callable
        Model fitting function.  Called once per coalition per test date.
    X_train_seq : list of arrays, one per test date
        X_train_seq[t] has shape (T_train_t, N_total).  Each element is the
        expanding (or rolling) training window for test date t.
    y_train_seq : list of arrays (T_train_t,)
        Corresponding training targets.
    X_test_seq : array of shape (T_test, N_total)
        Test feature matrix; one row per test date.
    y_test_seq : array of shape (T_test,)
        Realised test values.
    groups : sequence of lists of int
        Predictor group column indices.
    loss_fn : callable or None
        Loss function.  Defaults to negative MSFE (one observation).
    baseline_value : float
        Score when no predictors are included.

    Returns
    -------
    pbsv : array of shape (T_test, N_groups)
        PBSV[t, i] = Shapley value of group i at test date t.
    """
    if loss_fn is None:

        def loss_fn(yt, yh):
            return -float(np.mean((yt - yh) ** 2))

    T_test = len(y_test_seq)
    N_groups = len(groups)
    pbsv = np.zeros((T_test, N_groups))

    for t in range(T_test):
        X_tr_t = X_train_seq[t]
        y_tr_t = y_train_seq[t]
        x_te_t = X_test_seq[t : t + 1]  # single test row, shape (1, N_total)
        y_te_t = y_test_seq[t : t + 1]

        # For each test date, compute oShapley-VI on the single-observation loss
        def _fn_t(X_tr, y_tr, X_te):
            return forecast_fn(X_tr, y_tr, X_te)

        phi_t = oshapley_vi(
            forecast_fn=_fn_t,
            X_train=X_tr_t,
            y_train=y_tr_t,
            X_test=x_te_t,
            y_test=y_te_t,
            groups=groups,
            loss_fn=loss_fn,
            baseline_value=baseline_value,
        )
        pbsv[t] = phi_t

    return pbsv


# ---------------------------------------------------------------------------
# MAS: Model Accordance Score
# ---------------------------------------------------------------------------


def model_accordance_score(
    pbsv_model: NDArray[np.floating],
    pbsv_benchmark: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Model Accordance Score (MAS).

    MAS measures how consistently the model agrees with the benchmark in
    the sign of predictor contributions over time (CBRSS 2022).

    MAS[i] = (1/T) * sum_t sign(PBSV_model[t,i]) == sign(PBSV_bench[t,i])

    Parameters
    ----------
    pbsv_model : array of shape (T, N_groups)
    pbsv_benchmark : array of shape (T, N_groups)

    Returns
    -------
    mas : array of shape (N_groups,)
        Agreement rate in [0, 1] per predictor group.
    """
    agreement = np.sign(pbsv_model) == np.sign(pbsv_benchmark)
    return agreement.mean(axis=0)
