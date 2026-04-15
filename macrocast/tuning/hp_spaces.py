from __future__ import annotations

from .types import HPDistribution

MODEL_HP_SPACES = {
    "ridge": {"alpha": HPDistribution("log_float", 1e-4, 1e4, log=True)},
    "lasso": {"alpha": HPDistribution("log_float", 1e-6, 1e2, log=True)},
    "elasticnet": {
        "alpha": HPDistribution("log_float", 1e-6, 1e2, log=True),
        "l1_ratio": HPDistribution("float", 0.01, 0.99),
    },
    "adaptivelasso": {
        "gamma": HPDistribution("float", 0.5, 3.0),
        "init_estimator": HPDistribution("categorical", choices=("ridge", "ols")),
        "alpha": HPDistribution("log_float", 1e-5, 1e1, log=True),
    },
    "svr_linear": {
        "C": HPDistribution("log_float", 0.01, 1000.0, log=True),
        "epsilon": HPDistribution("log_float", 0.001, 1.0, log=True),
    },
    "svr_rbf": {
        "C": HPDistribution("log_float", 0.01, 1000.0, log=True),
        "epsilon": HPDistribution("log_float", 0.001, 1.0, log=True),
        "gamma": HPDistribution("categorical", choices=("scale", "auto", 0.001, 0.01, 0.1)),
    },
    "componentwise_boosting": {
        "n_iterations": HPDistribution("int", 50, 300),
        "learning_rate": HPDistribution("log_float", 0.01, 0.3, log=True),
    },
    "boosting_ridge": {
        "n_iterations": HPDistribution("int", 50, 300),
        "learning_rate": HPDistribution("log_float", 0.01, 0.3, log=True),
        "ridge_alpha": HPDistribution("log_float", 1e-3, 1e2, log=True),
    },
    "boosting_lasso": {
        "n_iterations": HPDistribution("int", 50, 300),
        "learning_rate": HPDistribution("log_float", 0.01, 0.3, log=True),
        "lasso_alpha": HPDistribution("log_float", 1e-5, 1e1, log=True),
    },
    "randomforest": {"n_estimators": HPDistribution("int", 100, 400), "max_depth": HPDistribution("int", 3, 10)},
    "extratrees": {"n_estimators": HPDistribution("int", 100, 400), "max_depth": HPDistribution("int", 3, 10)},
    "gbm": {
        "n_estimators": HPDistribution("int", 50, 300),
        "learning_rate": HPDistribution("log_float", 0.01, 0.3, log=True),
        "max_depth": HPDistribution("int", 2, 5),
    },
    "xgboost": {
        "n_estimators": HPDistribution("int", 50, 300),
        "learning_rate": HPDistribution("log_float", 0.01, 0.3, log=True),
        "max_depth": HPDistribution("int", 2, 6),
    },
    "lightgbm": {
        "n_estimators": HPDistribution("int", 50, 300),
        "learning_rate": HPDistribution("log_float", 0.01, 0.3, log=True),
        "num_leaves": HPDistribution("int", 8, 64),
    },
    "catboost": {
        "iterations": HPDistribution("int", 50, 300),
        "learning_rate": HPDistribution("log_float", 0.01, 0.3, log=True),
        "depth": HPDistribution("int", 2, 8),
    },
    "mlp": {
        "hidden_layer_sizes": HPDistribution("categorical", choices=((32,), (64,), (64, 32))),
        "alpha": HPDistribution("log_float", 1e-6, 1e-1, log=True),
        "learning_rate_init": HPDistribution("log_float", 1e-4, 1e-1, log=True),
    },
    "pcr": {"n_components": HPDistribution("int", 1, 10)},
    "pls": {"n_components": HPDistribution("int", 1, 10)},
    "huber": {
        "epsilon": HPDistribution("float", 1.05, 3.0),
        "alpha": HPDistribution("log_float", 1e-6, 1e2, log=True),
    },
}
