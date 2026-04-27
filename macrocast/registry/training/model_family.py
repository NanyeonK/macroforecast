from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='model_family',
    layer='3_training',
    axis_type='enum',
    default_policy='sweep',
    entries=(
        EnumRegistryEntry(id='ar', description='ar', status='operational', priority="A"),
        EnumRegistryEntry(id='ols', description='ols', status='operational', priority="A"),
        EnumRegistryEntry(id='ridge', description='ridge', status='operational', priority="A"),
        EnumRegistryEntry(id='lasso', description='lasso', status='operational', priority="A"),
        EnumRegistryEntry(id='elasticnet', description='elasticnet', status='operational', priority="A"),
        EnumRegistryEntry(id='bayesianridge', description='bayesianridge', status='operational', priority="A"),
        EnumRegistryEntry(id='huber', description='huber', status='operational', priority="A"),
        EnumRegistryEntry(id='adaptivelasso', description='adaptivelasso', status='operational', priority="A"),
        EnumRegistryEntry(id='svr_linear', description='svr linear', status='operational', priority="A"),
        EnumRegistryEntry(id='svr_rbf', description='svr rbf', status='operational', priority="A"),
        EnumRegistryEntry(id='componentwise_boosting', description='componentwise boosting', status='operational', priority="A"),
        EnumRegistryEntry(id='boosting_ridge', description='boosting ridge', status='operational', priority="A"),
        EnumRegistryEntry(id='boosting_lasso', description='boosting lasso', status='operational', priority="A"),
        EnumRegistryEntry(id='pcr', description='pcr', status='operational', priority="A"),
        EnumRegistryEntry(id='pls', description='pls', status='operational', priority="A"),
        EnumRegistryEntry(id='factor_augmented_linear', description='factor augmented linear', status='operational', priority="A"),
        EnumRegistryEntry(id='quantile_linear', description='quantile linear', status='operational', priority="A"),
        EnumRegistryEntry(id='midas_almon', description='midas almon distributed lag', status='operational_narrow', priority="B"),
        EnumRegistryEntry(id='randomforest', description='randomforest', status='operational', priority="A"),
        EnumRegistryEntry(id='extratrees', description='extratrees', status='operational', priority="A"),
        EnumRegistryEntry(id='gbm', description='gbm', status='operational', priority="A"),
        EnumRegistryEntry(id='xgboost', description='xgboost', status='operational', priority="A"),
        EnumRegistryEntry(id='lightgbm', description='lightgbm', status='operational', priority="A"),
        EnumRegistryEntry(id='catboost', description='catboost', status='operational', priority="A"),
        EnumRegistryEntry(id='mlp', description='mlp', status='operational', priority="A"),
        EnumRegistryEntry(id='lstm', description='lstm', status='operational', priority="A"),
        EnumRegistryEntry(id='gru', description='gru', status='operational', priority="A"),
        EnumRegistryEntry(id='tcn', description='tcn', status='operational', priority="A"),
    ),
    compatible_with={},
    incompatible_with={},
    component="nonlinearity",
)
