"""Model Confidence Set (MCS) — Hansen, Lunde, Nason (2011).

Implements the MCS procedure with block bootstrap to account for
time-series dependence in forecast errors.

Reference:
  Hansen, P.R., Lunde, A., and Nason, J.M. (2011).
  "The Model Confidence Set." Econometrica, 79(2), 453-497.

Algorithm:
  1. Compute pairwise loss differential d_{ij,t} = L(e_{i,t}) - L(e_{j,t}).
  2. Compute the MCS t-statistic T_{max} = max_{i,j} t_{ij}.
  3. Block bootstrap to obtain the null distribution of T_{max}.
  4. Eliminate the model with the largest average loss differential.
  5. Repeat until the null is not rejected (p > alpha).

The surviving set is the MCS at level alpha.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass
class MCSResult:
    """Result of the MCS procedure.

    Attributes
    ----------
    included : list of str
        Model IDs in the MCS (p-value > alpha).
    excluded : list of str
        Model IDs eliminated.
    p_values : dict
        Elimination p-value for each model {model_id: p_value}.
        Models in the MCS have p_value = 1.0 (never eliminated).
    mcs_alpha : float
        Significance level used.
    """

    included: list[str]
    excluded: list[str]
    p_values: dict[str, float]
    mcs_alpha: float


def mcs(
    loss_df: pd.DataFrame,
    alpha: float = 0.10,
    block_size: int = 12,
    n_bootstrap: int = 1000,
    loss_col: str = "squared_error",
    model_col: str = "model_id",
    date_col: str = "forecast_date",
    seed: int = 42,
) -> MCSResult:
    """Model Confidence Set with block bootstrap.

    Parameters
    ----------
    loss_df : pd.DataFrame
        Long-format table with columns [model_id, forecast_date, squared_error]
        (or equivalent column names via parameters).
    alpha : float
        Significance level for the MCS.  Default 0.10.
    block_size : int
        Block length for the stationary block bootstrap.  Set to the
        approximate autocorrelation horizon of the loss differentials.
        Default 12 (monthly data, one year).
    n_bootstrap : int
        Number of bootstrap replications.
    loss_col : str
        Column name for the loss series.
    model_col : str
        Column name for model identifiers.
    date_col : str
        Column name for forecast dates.
    seed : int
        Random seed.

    Returns
    -------
    MCSResult
    """
    rng = np.random.default_rng(seed)

    # Pivot to (T, n_models) loss matrix
    pivot = loss_df.pivot(index=date_col, columns=model_col, values=loss_col)
    pivot = pivot.sort_index().dropna(how="any")

    L = pivot.values  # (T, M)
    model_names = list(pivot.columns)
    T, M = L.shape

    surviving = list(range(M))  # indices of models still in the set
    p_values = {name: 1.0 for name in model_names}
    eliminated = []

    while len(surviving) > 1:
        S = len(surviving)
        L_s = L[:, surviving]  # (T, S)

        # Pairwise loss differentials d_{ij} and their means
        # t_ij = d_bar_ij / se(d_ij)
        # Test statistic: T_max = max_{i != j} |t_ij|
        t_stat, worst_idx = _mcs_t_max(L_s)

        # Block bootstrap null distribution
        t_boot = _block_bootstrap_t_max(L_s, block_size, n_bootstrap, rng)

        p_val = float(np.mean(t_boot >= t_stat))

        if p_val > alpha:
            # Reject nothing — all survivors are in the MCS
            break

        # Eliminate model with highest average loss differential
        # (the model that most consistently loses to the others)
        avg_d = np.array(
            [
                np.mean(
                    [L_s[:, i].mean() - L_s[:, j].mean() for j in range(S) if j != i]
                )
                for i in range(S)
            ]
        )
        elim_local = int(np.argmax(avg_d))
        elim_global = surviving[elim_local]
        elim_name = model_names[elim_global]

        p_values[elim_name] = p_val
        eliminated.append(elim_name)
        surviving.pop(elim_local)

    included = [model_names[i] for i in surviving]
    excluded = eliminated

    return MCSResult(
        included=included,
        excluded=excluded,
        p_values=p_values,
        mcs_alpha=alpha,
    )


def _mcs_t_max(L: NDArray[np.floating]) -> tuple[float, int]:
    """Compute the MCS T_max statistic and index of worst model."""
    T, S = L.shape
    t_max = -np.inf
    worst = 0

    for i in range(S):
        for j in range(S):
            if i == j:
                continue
            d = L[:, i] - L[:, j]
            d_bar = d.mean()
            # HAC standard error (Newey-West with bandwidth = T^{1/3})
            bw = max(1, int(T ** (1 / 3)))
            se = _nw_se(d, bw)
            if se > 0:
                t = abs(d_bar / se)
                if t > t_max:
                    t_max = t
                    worst = i

    return float(t_max), worst


def _block_bootstrap_t_max(
    L: NDArray[np.floating],
    block_size: int,
    n_boot: int,
    rng: np.random.Generator,
) -> NDArray[np.floating]:
    """Block bootstrap distribution of T_max under H0."""
    T, S = L.shape
    t_boot = np.zeros(n_boot)

    # Demean each model (impose H0: equal losses in expectation)
    L_dm = L - L.mean(axis=0, keepdims=True)

    for b in range(n_boot):
        # Stationary block bootstrap: random block starts, geometrically distributed lengths
        L_b = _draw_block_bootstrap(L_dm, block_size, T, rng)
        t_boot[b], _ = _mcs_t_max(L_b)

    return t_boot


def _draw_block_bootstrap(
    L: NDArray[np.floating],
    block_size: int,
    T: int,
    rng: np.random.Generator,
) -> NDArray[np.floating]:
    """Draw one block bootstrap sample of length T."""
    rows = []
    while len(rows) < T:
        start = rng.integers(0, T)
        length = rng.geometric(1.0 / block_size)
        for k in range(length):
            rows.append((start + k) % T)
    rows = rows[:T]
    return L[rows]


def _nw_se(d: NDArray[np.floating], bw: int) -> float:
    """Newey-West (1987) HAC standard error for a scalar time series."""
    T = len(d)
    d_dm = d - d.mean()
    gamma0 = np.dot(d_dm, d_dm) / T
    nw_var = gamma0
    for lag in range(1, bw + 1):
        gamma_l = np.dot(d_dm[lag:], d_dm[:-lag]) / T
        w = 1.0 - lag / (bw + 1)
        nw_var += 2 * w * gamma_l
    return float(np.sqrt(max(nw_var, 0) / T))
