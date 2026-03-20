"""Dual observation weights for ML forecasts (CGK 2024).

Implements the dual weight representation from:
  Coulombe, Goulet-Coulombe, Kichian (2024).
  "Dual Weights" (philgoucou/dual).

Each forecast ŷ_t = sum_s w_{ts} * y_s where w_{ts} is the weight assigned
to training observation s when forecasting at t.  The weight matrix W
(T_test, T_train) reveals which historical episodes the model draws on.

Model-specific dual weight implementations:

  - **KRR** (Kernel Ridge Regression):
    w_t = k_t(K + λI)^{-1}  (row vector)
    where K_{ss'} = k(x_s, x_{s'}) and k_t = [k(x_s, x_t)]_s.
    Exact closed form — no refit needed if kernel matrix is cached.

  - **Random Forest / XGBoost**:
    w_ts = (number of trees where obs s and t fall in the same leaf) / (tree count)
    Normalised to sum to 1 per forecast date.

  - **Neural Network** (linear dual, penultimate layer):
    w_t = phi_t (Phi Phi')^{-1} phi_s / T_train
    where phi_i is the penultimate layer activation for observation i.
    Ridge-regularised for numerical stability.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# KRR dual weights
# ---------------------------------------------------------------------------


def krr_dual_weights(
    X_train: NDArray[np.floating],
    X_test: NDArray[np.floating],
    alpha: float,
    gamma: float,
    kernel: str = "rbf",
) -> NDArray[np.floating]:
    """Compute KRR dual weight matrix W (T_test, T_train).

    For each test observation t: w_t = k_t @ (K + alpha*I)^{-1}

    The resulting weight matrix satisfies:
        ŷ_test = W @ y_train

    Parameters
    ----------
    X_train : array of shape (T_train, N)
    X_test  : array of shape (T_test, N)
    alpha   : float  Regularisation parameter (same as KernelRidge.alpha).
    gamma   : float  RBF kernel bandwidth (same as KernelRidge.gamma).
    kernel  : str    Kernel type.  Only "rbf" is currently supported.

    Returns
    -------
    W : array of shape (T_test, T_train)
    """
    if kernel != "rbf":
        raise NotImplementedError("Only RBF kernel is currently supported.")

    T_train = X_train.shape[0]

    # RBF kernel: K(x, x') = exp(-gamma * ||x - x'||^2)
    def _rbf(A: NDArray, B: NDArray) -> NDArray:
        # Pairwise squared Euclidean distances
        # A: (m, N), B: (n, N) → (m, n)
        sq_norms_A = np.sum(A**2, axis=1, keepdims=True)  # (m, 1)
        sq_norms_B = np.sum(B**2, axis=1, keepdims=True)  # (n, 1)
        sq_dist = sq_norms_A + sq_norms_B.T - 2 * A @ B.T
        sq_dist = np.maximum(sq_dist, 0)  # numerical clip
        return np.exp(-gamma * sq_dist)

    K_train = _rbf(X_train, X_train)  # (T_train, T_train)
    K_test = _rbf(X_test, X_train)  # (T_test,  T_train)

    # Solve (K + alpha * I) @ A = K_test.T for A, then W = A.T
    regularised = K_train + alpha * np.eye(T_train)
    # W.T = (K + αI)^{-1} @ K_test.T → W = K_test @ (K + αI)^{-1}
    W = np.linalg.solve(regularised.T, K_test.T).T  # (T_test, T_train)

    return W


# ---------------------------------------------------------------------------
# Random Forest / XGBoost dual weights (leaf co-membership)
# ---------------------------------------------------------------------------


def tree_dual_weights(
    model,
    X_train: NDArray[np.floating],
    X_test: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Dual weights for tree ensemble models via leaf co-membership.

    For each tree b and test observation t, find the leaf l_{bt} and count
    how many training observations fall in the same leaf.  The weight for
    training observation s at test date t is:

        w_ts = (1/B) * sum_b 1[leaf_b(x_t) == leaf_b(x_s)] / |leaf_b(x_t)|

    Parameters
    ----------
    model : fitted RandomForestRegressor or XGBRegressor
        Must have `apply(X)` method that returns leaf indices (T, n_trees).
    X_train : array of shape (T_train, N)
    X_test  : array of shape (T_test, N)

    Returns
    -------
    W : array of shape (T_test, T_train)
    """
    # Get leaf indices: (T, n_trees)
    # RandomForest and XGBoost both support apply()
    leaves_train = model.apply(X_train)  # (T_train, n_trees)
    leaves_test = model.apply(X_test)  # (T_test,  n_trees)

    if leaves_train.ndim == 1:
        leaves_train = leaves_train[:, np.newaxis]
        leaves_test = leaves_test[:, np.newaxis]

    T_train = X_train.shape[0]
    T_test = X_test.shape[0]
    n_trees = leaves_train.shape[1]

    W = np.zeros((T_test, T_train))

    for b in range(n_trees):
        lt = leaves_test[:, b]  # (T_test,)
        ls = leaves_train[:, b]  # (T_train,)

        for ti in range(T_test):
            # Training obs in same leaf as test obs ti
            same_leaf = ls == lt[ti]
            leaf_size = same_leaf.sum()
            if leaf_size > 0:
                W[ti] += same_leaf.astype(float) / leaf_size

    W /= n_trees
    return W


# ---------------------------------------------------------------------------
# Neural Network dual weights (penultimate layer linear dual)
# ---------------------------------------------------------------------------


def nn_dual_weights(
    activations_train: NDArray[np.floating],
    activations_test: NDArray[np.floating],
    ridge_alpha: float = 1e-4,
) -> NDArray[np.floating]:
    """Dual weights for feedforward neural networks via penultimate layer.

    The readout layer computes ŷ = phi(x) @ w where phi(x) is the penultimate
    layer activation.  The linear dual representation is:

        w_ts = phi_t @ (Phi Phi' + alpha*I)^{-1} @ phi_s / T_train

    which yields ŷ_t = W_t @ y_train (approximately, for the linear readout).

    Parameters
    ----------
    activations_train : array of shape (T_train, H)
        Penultimate layer activations for training observations.
    activations_test  : array of shape (T_test, H)
        Penultimate layer activations for test observations.
    ridge_alpha : float
        Regularisation for numerical stability of the matrix inverse.

    Returns
    -------
    W : array of shape (T_test, T_train)
    """
    T_train, H = activations_train.shape

    # Phi: (T_train, H); PhiPhi': (H, H)
    Phi = activations_train
    PhiPhi_reg = Phi.T @ Phi + ridge_alpha * np.eye(H)

    # W[t, s] = phi_t @ (PhiPhi' + αI)^{-1} @ phi_s / T_train
    # = activations_test @ solve(PhiPhi_reg, Phi.T) / T_train
    # Shape: (T_test, H) @ (H, T_train) = (T_test, T_train)
    PhiPhi_inv_PhiT = np.linalg.solve(PhiPhi_reg, Phi.T)  # (H, T_train)
    W = (activations_test @ PhiPhi_inv_PhiT) / T_train

    return W


# ---------------------------------------------------------------------------
# Weight matrix summary statistics
# ---------------------------------------------------------------------------


def effective_history_length(W: NDArray[np.floating]) -> NDArray[np.floating]:
    """Effective number of training observations used per forecast date.

    Computed as the inverse of the Herfindahl-Hirschman Index of the weights:
        EHL_t = 1 / sum_s w_{ts}^2

    Parameters
    ----------
    W : array of shape (T_test, T_train)

    Returns
    -------
    ehl : array of shape (T_test,)
    """
    # Normalise rows to sum to 1 (if they don't already)
    row_sums = W.sum(axis=1, keepdims=True)
    W_norm = W / np.where(row_sums != 0, row_sums, 1)
    hhi = np.sum(W_norm**2, axis=1)
    return 1.0 / np.where(hhi > 0, hhi, np.nan)


def top_analogies(
    W: NDArray[np.floating],
    train_dates: NDArray,
    test_dates: NDArray,
    k: int = 5,
) -> list:
    """Return the top-k historical analogies for each test date.

    Parameters
    ----------
    W : array of shape (T_test, T_train)
    train_dates : array of shape (T_train,)  Date labels for training obs.
    test_dates  : array of shape (T_test,)   Date labels for test obs.
    k : int  Number of top analogies to return per test date.

    Returns
    -------
    list of dicts, one per test date:
        {"test_date": ..., "analogies": [(train_date, weight), ...]}
    """
    results = []
    for ti in range(len(test_dates)):
        row = W[ti]
        top_idx = np.argsort(row)[::-1][:k]
        analogies = [(train_dates[si], float(row[si])) for si in top_idx]
        results.append({"test_date": test_dates[ti], "analogies": analogies})
    return results
