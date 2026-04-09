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
from macrocast.pipeline.horserace import HorseRaceGrid
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
from macrocast.pipeline.registry import (
    get_feature_defaults,
    get_model_defaults,
    load_feature_registry,
    load_model_registry,
    validate_feature_model_compatibility,
    validate_feature_registry,
    validate_model_registry,
)
from macrocast.pipeline.r_models import (
    AdaptiveLassoModel,
    ARDIModel,
    ARModel,
    BoogingModel,
    BVARModel,
    ElasticNetModel,
    GroupLassoModel,
    LassoModel,
    RidgeModel,
    RModelEstimator,
    TVPRidgeModel,
)
from macrocast.pipeline.results import ForecastRecord, ResultSet

__all__ = [
    "CVScheme",
    "CVSchemeType",
    "LossFunction",
    "Nonlinearity",
    "Regularization",
    "Window",
    "MacrocastEstimator",
    "SequenceEstimator",
    "FeatureBuilder",
    "ForecastRecord",
    "ResultSet",
    "KRRModel",
    "SVRRBFModel",
    "SVRLinearModel",
    "RFModel",
    "XGBoostModel",
    "GBModel",
    "NNModel",
    "LSTMModel",
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
    "BVARModel",
    "ModelSpec",
    "FeatureSpec",
    "ForecastExperiment",
    "HorseRaceGrid",
    'get_feature_defaults',
    'get_model_defaults',
    'load_feature_registry',
    'load_model_registry',
    'validate_feature_model_compatibility',
    'validate_feature_registry',
    'validate_model_registry',
]
