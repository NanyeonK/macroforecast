"""Pipeline component definitions for the four-part decomposition framework.

Coulombe, Leroux, Stevanovic, and Surprenant (2022) decompose ML forecast gains
into four orthogonal treatment components: nonlinearity, regularization,
cross-validation scheme, and loss function.  Each component is an enum-like
object -- never a plain string flag -- so experiment configurations are both
type-safe and human-readable.

Two distinct window concepts are kept separate:

* ``CVScheme`` -- inner hyperparameter-selection loop (K-fold, POOS-CV, BIC).
* ``Window``   -- outer pseudo-OOS evaluation loop (expanding, rolling).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# ---------------------------------------------------------------------------
# Nonlinearity
# ---------------------------------------------------------------------------


class Nonlinearity(Enum):
    """Functional form of the forecasting model.

    LINEAR covers all models that are linear in parameters (AR, ARDI, Ridge,
    LASSO, etc.).  All other values designate nonlinear models handled on the
    Python side.
    """

    LINEAR = "linear"
    KRR = "krr"  # Kernel Ridge Regression (RBF)
    SVR_RBF = "svr_rbf"  # SVR with RBF kernel (loss fn comparison)
    SVR_LINEAR = "svr_linear"  # Linear SVR (loss fn comparison, no kernel)
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    NEURAL_NET = "neural_net"  # Feedforward, ReLU, 1-2 hidden layers
    LSTM = "lstm"  # Sequence model (SequenceEstimator)
    GRADIENT_BOOSTING = "gradient_boosting"


# ---------------------------------------------------------------------------
# Regularization
# ---------------------------------------------------------------------------


class Regularization(Enum):
    """Dimensionality-reduction / penalty strategy.

    NONE through BOOGING map to specific estimation procedures.  Models marked
    ``# R`` are implemented in macrocastR and produce results via the parquet
    exchange; the rest are Python-side.

    FACTORS (PCA diffusion index / ARDI) is the empirically dominant choice
    per CLSS 2022: factor model regularization outperforms all sparse
    alternatives in the data-rich environment.
    """

    NONE = "none"  # OLS, no penalty (data-poor baseline)
    RIDGE = "ridge"  # L2 penalty -- R: glmnet(alpha=0)
    LASSO = "lasso"  # L1 penalty -- R: glmnet(alpha=1)
    ADAPTIVE_LASSO = "adaptive_lasso"  # Weighted L1 -- R: glmnet(penalty.factor=)
    GROUP_LASSO = "group_lasso"  # Group L1 by FRED category -- R: grpreg
    ELASTIC_NET = "elastic_net"  # L1+L2 mix -- R: glmnet(alpha=0.5)
    FACTORS = "factors"  # PCA diffusion index (ARDI) -- R + Python
    TVP_RIDGE = "tvp_ridge"  # Time-varying params via Ridge -- R
    BOOGING = "booging"  # Bootstrap aggregating + pruning -- R


# ---------------------------------------------------------------------------
# CV Scheme  (inner HP-selection loop)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _CVSchemeBase:
    """Base class for all cross-validation scheme configurations.

    Subclasses are frozen dataclasses so they can be used as dictionary keys
    and compared for equality without surprises.
    """


@dataclass(frozen=True)
class _KFoldCV(_CVSchemeBase):
    """K-fold cross-validation.

    Preferred per CLSS 2022: exploits the full time series for HP ranking,
    unlike POOS-CV which uses only the end of the training sample.

    Parameters
    ----------
    k : int
        Number of folds.  Default 5.
    """

    k: int = 5


@dataclass(frozen=True)
class _POOSScheme(_CVSchemeBase):
    """Pseudo-OOS (expanding one-step-ahead) cross-validation."""


@dataclass(frozen=True)
class _BICScheme(_CVSchemeBase):
    """Bayesian Information Criterion for HP selection.

    Applicable to linear models only (AR lag order, Ridge penalty).
    """


class CVScheme:
    """Factory and namespace for CV scheme objects.

    Usage
    -----
    >>> CVScheme.BIC
    _BICScheme()
    >>> CVScheme.POOS
    _POOSScheme()
    >>> CVScheme.KFOLD(k=5)
    _KFoldCV(k=5)
    """

    BIC: _BICScheme = _BICScheme()
    POOS: _POOSScheme = _POOSScheme()

    @staticmethod
    def KFOLD(k: int = 5) -> _KFoldCV:
        """Return a K-fold CV configuration with *k* folds."""
        return _KFoldCV(k=k)


# Public type alias for annotations
CVSchemeType = _CVSchemeBase


# ---------------------------------------------------------------------------
# Loss Function
# ---------------------------------------------------------------------------


class LossFunction(Enum):
    """In-sample loss function optimised during model training.

    L2 is the default and empirically preferred (CLSS 2022).
    EPSILON_INSENSITIVE is the SVR-type loss; comparing KRR (L2) vs SVR_RBF
    (EPSILON_INSENSITIVE) with identical RBF kernels isolates the loss
    function treatment effect.
    """

    L2 = "l2"
    EPSILON_INSENSITIVE = "epsilon_insensitive"


# ---------------------------------------------------------------------------
# Window  (outer pseudo-OOS evaluation loop)
# ---------------------------------------------------------------------------


class Window(Enum):
    """Outer pseudo-OOS evaluation window strategy.

    Distinct from ``CVScheme``, which governs the inner HP-selection loop.

    EXPANDING uses all available history back to the start of the sample and
    is the macrocast default (matches CLSS 2022).  ROLLING discards distant
    history and is relevant when structural breaks make early data
    uninformative.
    """

    EXPANDING = "expanding"
    ROLLING = "rolling"
