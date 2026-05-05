from __future__ import annotations

import numpy as np

SCORERS = {
    "validation_mse": lambda y, yhat: float(np.mean((y - yhat) ** 2)),
    "validation_rmse": lambda y, yhat: float(np.sqrt(np.mean((y - yhat) ** 2))),
    "validation_mae": lambda y, yhat: float(np.mean(np.abs(y - yhat))),
}


def get_scorer(objective: str):
    if objective not in SCORERS:
        raise ValueError(f"unknown tuning_objective: {objective}")
    return SCORERS[objective]
