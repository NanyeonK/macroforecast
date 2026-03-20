"""macrocast.pipeline — Layer 2: Forecasting experiment and model grid."""

from macrocast.pipeline.components import (
    CVScheme,
    CVSchemeType,
    LossFunction,
    Nonlinearity,
    Regularization,
    Window,
)
from macrocast.pipeline.estimator import MacrocastEstimator, SequenceEstimator
from macrocast.pipeline.experiment import FeatureSpec, ForecastExperiment, ModelSpec
from macrocast.pipeline.features import FeatureBuilder
from macrocast.pipeline.models import (
    GBModel,
    KRRModel,
    LSTMModel,
    NNModel,
    RFModel,
    SVRLinearModel,
    SVRRBFModel,
    XGBoostModel,
)
from macrocast.pipeline.r_models import (
    ARDIModel,
    ARModel,
    AdaptiveLassoModel,
    BoogingModel,
    ElasticNetModel,
    GroupLassoModel,
    LassoModel,
    RModelEstimator,
    RidgeModel,
    TVPRidgeModel,
)
from macrocast.pipeline.horserace import HorseRaceGrid
from macrocast.pipeline.results import ForecastRecord, ResultSet

__all__ = [
    # components
    "CVScheme",
    "CVSchemeType",
    "LossFunction",
    "Nonlinearity",
    "Regularization",
    "Window",
    # estimator ABCs
    "MacrocastEstimator",
    "SequenceEstimator",
    # features
    "FeatureBuilder",
    # results
    "ForecastRecord",
    "ResultSet",
    # Python models
    "KRRModel",
    "SVRRBFModel",
    "SVRLinearModel",
    "RFModel",
    "XGBoostModel",
    "GBModel",
    "NNModel",
    "LSTMModel",
    # R model bridge
    "RModelEstimator",
    "ARModel",
    "ARDIModel",
    "RidgeModel",
    "LassoModel",
    "AdaptiveLassoModel",
    "GroupLassoModel",
    "ElasticNetModel",
    "TVPRidgeModel",
    "BoogingModel",
    # experiment
    "ModelSpec",
    "FeatureSpec",
    "ForecastExperiment",
    # horse race grid
    "HorseRaceGrid",
]
