"""Abstract base classes for macrocast estimators.

All Python-side forecasting models implement one of two interfaces:

* ``MacrocastEstimator`` -- standard cross-sectional interface, input shape (T, N).
  Used by KRR, SVR, RF, XGBoost, and NN.
* ``SequenceEstimator`` -- sequence interface, input shape (T, L, N) where L is
  the look-back window length.  Used by LSTM.

Both interfaces enforce direct forecasting (one model per horizon h).  The
estimator receives the feature matrix Z_t constructed by FeatureBuilder; it has
no knowledge of the horizon, which is baked into the target vector y_{t+h}.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from macrocast.pipeline.components import Nonlinearity

# ---------------------------------------------------------------------------
# Standard estimator  (T, N) → scalar
# ---------------------------------------------------------------------------


class MacrocastEstimator(ABC):
    """Interface for all cross-sectional forecasting models.

    Subclasses must implement ``fit``, ``predict``, and expose the
    ``nonlinearity_type`` class attribute so that experiment runners can
    route models to the correct decomposition bucket.

    Parameters are intentionally left to subclasses; the interface
    imposes no hyperparameter contract.
    """

    #: Declared by every concrete subclass — identifies the Nonlinearity
    #: component for the four-part decomposition framework.
    nonlinearity_type: Nonlinearity

    @abstractmethod
    def fit(
        self, X: NDArray[np.floating], y: NDArray[np.floating]
    ) -> MacrocastEstimator:
        """Fit the model on the training feature matrix.

        Parameters
        ----------
        X : array of shape (T, N)
            Feature matrix where T is the number of training observations and
            N is the feature dimension (AR lags, PCA factors, or raw predictors
            depending on the FeatureBuilder configuration).
        y : array of shape (T,)
            Target values.  For horizon h, these are y_{t+h} aligned with rows
            of X indexed at time t (direct forecasting convention).

        Returns
        -------
        self
            Returns the fitted estimator to allow chaining.
        """

    @abstractmethod
    def predict(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """Generate point forecasts for the provided feature matrix.

        Parameters
        ----------
        X : array of shape (T_test, N)
            Out-of-sample feature matrix.

        Returns
        -------
        y_hat : array of shape (T_test,)
            Point forecast for each row.
        """


# ---------------------------------------------------------------------------
# Sequence estimator  (T, L, N) → scalar  (LSTM only in v1)
# ---------------------------------------------------------------------------


class SequenceEstimator(ABC):
    """Interface for sequence models (LSTM).

    Input tensors have shape (T, L, N) where L is the look-back window
    length (a tuning parameter).  The target vector y has shape (T,) aligned
    at the *last* time step of each window (direct forecasting: window ending
    at t predicts y_{t+h}).

    All other conventions from MacrocastEstimator apply.
    """

    nonlinearity_type: Nonlinearity

    @abstractmethod
    def fit(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating],
    ) -> SequenceEstimator:
        """Fit the sequence model.

        Parameters
        ----------
        X : array of shape (T, L, N)
            Sequence feature tensor.  L is the look-back length; N is the
            number of features per time step.
        y : array of shape (T,)
            Targets aligned with the last step of each window.

        Returns
        -------
        self
        """

    @abstractmethod
    def predict(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """Generate point forecasts from a sequence tensor.

        Parameters
        ----------
        X : array of shape (T_test, L, N)
            Out-of-sample sequence tensor.

        Returns
        -------
        y_hat : array of shape (T_test,)
        """
