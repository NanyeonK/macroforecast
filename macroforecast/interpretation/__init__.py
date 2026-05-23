"""Public L7 interpretation class wrappers.

Exposes two class-based wrappers for L7 interpretation operations that
operate on fitted model artifacts. Each class provides a ``compute(...)``
method that delegates numeric work to the existing private runtime helpers
in ``macroforecast.core.runtime``.

Classes
-------
- :class:`GIRF`             -- Pesaran-Shin (1998) Generalized IRF.
- :class:`LSTMHiddenState`  -- Karpathy (2015) LSTM hidden-state importance.

Usage::

    from macroforecast.interpretation import GIRF, LSTMHiddenState

    # For a fitted VAR result (from macroforecast.functions.var_fit):
    girf = GIRF()
    importance_df = girf.compute(fitted_var, n_periods=12)

    # For a fitted LSTM result (from macroforecast.functions.lstm_fit):
    lstm_interp = LSTMHiddenState()
    importance_df = lstm_interp.compute(fitted_lstm, X)

Cycle 63 -- L7 interpretation class wrappers (GIRF + LSTMHiddenState).
"""
from __future__ import annotations

from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_model_artifact(fitted_result: Any, X: pd.DataFrame | None = None) -> Any:
    """Build a minimal ModelArtifact from a FitResult for L7 dispatch.

    The runtime L7 helpers expect a ModelArtifact with a ``fitted_object``
    attribute holding the raw fitted estimator. This factory wraps any
    FitResult (which stores the fitted estimator as ``._model``) into the
    required type.

    Parameters
    ----------
    fitted_result :
        A FitResult object exposing ``._model`` (the fitted estimator).
        May also be a bare fitted estimator if it does not have ``._model``.
    X :
        Optional feature matrix; used to extract feature names.

    Returns
    -------
    ModelArtifact
    """
    from macroforecast.core.types import ModelArtifact

    # Extract the raw fitted object: FitResult wraps in ._model; bare estimators
    # are passed through directly.
    if hasattr(fitted_result, "_model"):
        fitted_obj = fitted_result._model
    else:
        fitted_obj = fitted_result

    # Infer feature names.
    feature_names: tuple[str, ...] = ()
    if X is not None and isinstance(X, pd.DataFrame):
        feature_names = tuple(str(c) for c in X.columns)
    elif hasattr(fitted_obj, "feature_names_in_"):
        feature_names = tuple(str(n) for n in fitted_obj.feature_names_in_)

    # Infer framework from type module path.
    mod = type(fitted_obj).__module__ or ""
    if mod.startswith("xgboost"):
        framework: str = "xgboost"
    elif mod.startswith("lightgbm"):
        framework = "lightgbm"
    elif mod.startswith("statsmodels"):
        framework = "statsmodels"
    elif mod.startswith("torch"):
        framework = "torch"
    else:
        framework = "sklearn"

    return ModelArtifact(
        model_id="_standalone_l7",
        family="_standalone",
        fitted_object=fitted_obj,
        framework=framework,  # type: ignore[arg-type]
        feature_names=feature_names,
    )


# ---------------------------------------------------------------------------
# GIRF
# ---------------------------------------------------------------------------

class GIRF:
    """Pesaran-Shin (1998) Generalized Impulse Response Function.

    Computes the order-invariant generalized IRF for a fitted VAR model.
    The importance metric per variable j is the L1 norm of the target
    variable's response to a shock in j across all horizons h = 0, ..., H.

    Formula:
        GIRF_h(j) = sigma_jj^{-1/2} * A_h * Sigma * e_j

    where A_h is the reduced-form MA coefficient matrix at horizon h (from
    ``irf.irfs``, NOT the Cholesky-orthogonalized ``orth_irfs``), Sigma is
    the residual covariance matrix, and e_j is the unit vector at position j.

    Falls back to tree-based permutation importance when the model is not a
    fitted statsmodels VAR.

    References
    ----------
    Pesaran, Shin (1998) "Generalized impulse response analysis in linear
    multivariate models." Economics Letters 58(1).
    """

    def compute(
        self,
        fitted_var: Any,
        n_periods: int = 12,
    ) -> pd.DataFrame:
        """Compute the GIRF importance frame.

        Parameters
        ----------
        fitted_var :
            A fitted VAR result. Accepts:
            - A FitResult from :func:`~macroforecast.functions.var_fit`
              (exposes ``._model`` with the statsmodels VAR results).
            - A raw fitted statsmodels VAR results object.
        n_periods : int
            Horizon length for impulse response accumulation. Default 12.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ``["feature", "importance", "method"]``
            where ``importance`` is the L1 norm of the GIRF response for
            each variable across horizons 0..H.
        """
        from macroforecast.core.runtime import _var_girf_frame

        model_artifact = _build_model_artifact(fitted_var)
        return _var_girf_frame(model_artifact, n_periods=n_periods)


# ---------------------------------------------------------------------------
# LSTMHiddenState
# ---------------------------------------------------------------------------

class LSTMHiddenState:
    """Karpathy (2015) LSTM/GRU hidden-state activation importance.

    Registers a forward hook on the sequence cell (nn.LSTM or nn.GRU) of
    the fitted ``_TorchSequenceModel`` to capture output activations h_t
    (shape: batch x seq_len x hidden_size) during a single forward pass
    over the OOS feature matrix X.

    Importance per hidden unit = mean(|h_t|) across all OOS observations.

    Gates on torch availability (raises ``NotImplementedError`` with a
    ``macroforecast[deep]`` install hint when torch is unavailable) and on
    model family (raises ``NotImplementedError`` for transformer family,
    where hidden-state attribution is less meaningful).

    References
    ----------
    Karpathy (2015) "Visualizing and Understanding Recurrent Networks."
    arXiv:1506.02078.
    """

    def compute(
        self,
        fitted_lstm: Any,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute LSTM hidden-state importance.

        Parameters
        ----------
        fitted_lstm :
            A fitted LSTM or GRU result. Accepts:
            - A FitResult from :func:`~macroforecast.functions.lstm_fit` or
              :func:`~macroforecast.functions.gru_fit` (exposes ``._model``).
            - A raw fitted ``_TorchSequenceModel`` instance.
        X : pd.DataFrame
            Out-of-sample feature matrix used for the forward pass.
            Shape (n_samples, n_features). NaN values are filled with 0.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns
            ``["feature", "importance", "coefficient", "method"]`` where
            ``feature`` is ``"hidden_unit_i"``, ``importance`` is
            ``mean(|h_t|)``, ``coefficient`` is ``None``, and ``method``
            is ``"lstm_hidden_state"``.

        Raises
        ------
        NotImplementedError
            If torch is not installed or the model family is "transformer".
        """
        from macroforecast.core.runtime import _lstm_hidden_state_frame

        if not isinstance(X, pd.DataFrame):
            import numpy as np
            X = pd.DataFrame(
                X, columns=[f"x{i}" for i in range(X.shape[1])]
            )

        model_artifact = _build_model_artifact(fitted_lstm, X)
        return _lstm_hidden_state_frame(model_artifact, X)


__all__ = [
    "GIRF",
    "LSTMHiddenState",
]
