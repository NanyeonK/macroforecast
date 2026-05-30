from __future__ import annotations

import numpy as np
import pandas as pd


def xy(n: int = 48) -> tuple[pd.DataFrame, pd.Series]:
    idx = pd.date_range("2000-01-31", periods=n, freq="ME")
    x1 = np.linspace(0.0, 1.0, n)
    x2 = np.cos(np.arange(n) / 4.0)
    X = pd.DataFrame({"x1": x1, "x2": x2}, index=idx)
    y = pd.Series(0.5 + 2.5 * x1 - 0.4 * x2, index=idx, name="target")
    return X, y


class PanelNoLeakFit:
    def __init__(self, panel: pd.DataFrame, target: str) -> None:
        predictors = [column for column in panel.columns if column != target]
        if "actual" in predictors:
            raise ValueError("target leaked into panel predictors")
        self._value = float(panel[target].mean())

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(self._value, index=X.index, name="prediction")


def panel_no_leak_fit(
    panel: pd.DataFrame,
    *,
    target: str | None = None,
    k: int = 1,
) -> PanelNoLeakFit:
    if target is None:
        raise ValueError("panel target was not supplied")
    return PanelNoLeakFit(panel, target)


class ScoreFit:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(self.value, index=X.index, name="prediction")


def score_model(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    score_value: float = 0.0,
    **_: object,
) -> ScoreFit:
    return ScoreFit(score_value)


def failing_model(X: pd.DataFrame, y: pd.Series, *, alpha: float = 0.0) -> ScoreFit:
    raise RuntimeError(f"intentional failure alpha={alpha}")


def first_prediction(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(pd.Series(y_pred).iloc[0])
