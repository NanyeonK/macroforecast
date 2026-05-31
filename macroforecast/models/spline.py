from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from macroforecast.models.types import ModelFit
from macroforecast.models.utils import fit_estimator

HingeSide = Literal["left", "right"]


@dataclass(frozen=True)
class _HingeFactor:
    feature: str
    knot: float
    side: HingeSide

    def evaluate(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame[self.feature].to_numpy(dtype=float)
        if self.side == "right":
            return np.maximum(0.0, values - self.knot)
        return np.maximum(0.0, self.knot - values)

    def name(self) -> str:
        op = "max(0,x-k)" if self.side == "right" else "max(0,k-x)"
        return f"{op}[{self.feature};{self.knot:.6g}]"


@dataclass(frozen=True)
class _BasisTerm:
    factors: tuple[_HingeFactor, ...] = ()

    @property
    def degree(self) -> int:
        return len(self.factors)

    def extend(self, factor: _HingeFactor) -> "_BasisTerm":
        return _BasisTerm((*self.factors, factor))

    def evaluate(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.factors:
            return np.ones(len(frame), dtype=float)
        values: np.ndarray = np.ones(len(frame), dtype=float)
        for factor in self.factors:
            values *= factor.evaluate(frame)
        return values

    def name(self) -> str:
        if not self.factors:
            return "intercept"
        return " * ".join(factor.name() for factor in self.factors)


class MARSRegressor:
    """Package-native multivariate adaptive regression splines estimator.

    This is an additive/low-order hinge-basis implementation for the package's
    callable API. It uses forward pair insertion and optional backward pruning
    by generalized cross-validation. It does not claim bit-level equivalence to
    proprietary or unmaintained MARS backends.
    """

    def __init__(
        self,
        *,
        max_terms: int = 20,
        max_degree: int = 1,
        n_knots: int = 10,
        min_improvement: float = 1e-6,
        penalty: float = 2.0,
        prune: bool = True,
    ) -> None:
        self.max_terms = max(1, int(max_terms))
        self.max_degree = max(1, int(max_degree))
        self.n_knots = max(2, int(n_knots))
        self.min_improvement = float(min_improvement)
        self.penalty = float(penalty)
        self.prune = bool(prune)
        self.feature_names_in_: tuple[str, ...] = ()
        self.x_mean_: pd.Series | None = None
        self.terms_: list[_BasisTerm] = [_BasisTerm()]
        self.basis_names_: tuple[str, ...] = ("intercept",)
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.gcv_: float | None = None
        self.n_terms_: int = 1

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MARSRegressor":
        frame = X.astype(float).copy()
        self.feature_names_in_ = tuple(str(column) for column in frame.columns)
        self.x_mean_ = frame.mean(axis=0)
        frame = frame.fillna(self.x_mean_).fillna(0.0)
        target = pd.Series(y, index=frame.index).astype(float)
        terms = [_BasisTerm()]
        current_rss, _ = self._fit_terms(frame, target, terms)
        candidate_knots = self._candidate_knots(frame)
        while len(terms) < self.max_terms:
            best: tuple[float, list[_BasisTerm]] | None = None
            parent_terms = [term for term in terms if term.degree < self.max_degree]
            for parent in parent_terms:
                for feature, knots in candidate_knots.items():
                    for knot in knots:
                        left = parent.extend(_HingeFactor(feature, float(knot), "left"))
                        right = parent.extend(_HingeFactor(feature, float(knot), "right"))
                        if left in terms or right in terms:
                            continue
                        trial_terms = [*terms, left, right]
                        if len(trial_terms) > self.max_terms:
                            trial_terms = trial_terms[: self.max_terms]
                        rss, _ = self._fit_terms(frame, target, trial_terms)
                        if best is None or rss < best[0]:
                            best = (rss, trial_terms)
            if best is None:
                break
            improvement = current_rss - best[0]
            scale = max(abs(current_rss), 1.0)
            if improvement <= self.min_improvement * scale:
                break
            current_rss = best[0]
            terms = best[1]
        if self.prune and len(terms) > 2:
            terms = self._prune_terms(frame, target, terms)
        _, coef = self._fit_terms(frame, target, terms)
        self.terms_ = terms
        self.basis_names_ = tuple(term.name() for term in terms)
        self.intercept_ = float(coef[0])
        self.coef_ = np.asarray(coef[1:], dtype=float)
        self.gcv_ = self._gcv(frame, target, terms)
        self.n_terms_ = len(terms)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.x_mean_ is None:
            return np.zeros(len(X), dtype=float)
        frame = X.reindex(columns=list(self.feature_names_in_), fill_value=np.nan).astype(float)
        frame = frame.fillna(self.x_mean_).fillna(0.0)
        design = self._design(frame, self.terms_)
        coef = np.r_[self.intercept_, np.asarray(self.coef_ if self.coef_ is not None else [], dtype=float)]
        if len(coef) != design.shape[1]:
            return np.full(len(frame), self.intercept_, dtype=float)
        return design @ coef

    def _candidate_knots(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        grid = np.linspace(0.1, 0.9, self.n_knots)
        out: dict[str, np.ndarray] = {}
        for column in frame.columns:
            values = frame[column].dropna().to_numpy(dtype=float)
            if len(np.unique(values)) < 3:
                continue
            knots = np.unique(np.quantile(values, grid))
            lo: float = float(np.min(values))
            hi: float = float(np.max(values))
            out[str(column)] = knots[(knots > lo) & (knots < hi)]
        return out

    def _fit_terms(
        self,
        frame: pd.DataFrame,
        target: pd.Series,
        terms: list[_BasisTerm],
    ) -> tuple[float, np.ndarray]:
        design = self._design(frame, terms)
        coef = np.linalg.pinv(design) @ target.to_numpy(dtype=float)
        residual = target.to_numpy(dtype=float) - design @ coef
        rss = float(residual @ residual)
        return rss, coef

    def _prune_terms(
        self,
        frame: pd.DataFrame,
        target: pd.Series,
        terms: list[_BasisTerm],
    ) -> list[_BasisTerm]:
        best_terms = list(terms)
        best_gcv = self._gcv(frame, target, best_terms)
        improved = True
        while improved and len(best_terms) > 2:
            improved = False
            candidate: tuple[float, list[_BasisTerm]] | None = None
            for idx in range(1, len(best_terms)):
                trial = [term for pos, term in enumerate(best_terms) if pos != idx]
                gcv = self._gcv(frame, target, trial)
                if candidate is None or gcv < candidate[0]:
                    candidate = (gcv, trial)
            if candidate is not None and candidate[0] <= best_gcv:
                best_gcv, best_terms = candidate
                improved = True
        return best_terms

    def _gcv(self, frame: pd.DataFrame, target: pd.Series, terms: list[_BasisTerm]) -> float:
        rss, _ = self._fit_terms(frame, target, terms)
        n_obs = max(1, len(frame))
        effective = len(terms) + self.penalty * max(0, len(terms) - 1) / 2.0
        denom = max((1.0 - effective / n_obs) ** 2, 1e-12)
        return float((rss / n_obs) / denom)

    @staticmethod
    def _design(frame: pd.DataFrame, terms: list[_BasisTerm]) -> np.ndarray:
        return np.column_stack([term.evaluate(frame) for term in terms])


def mars(
    X: Any,
    y: Any | None = None,
    *,
    max_terms: int = 20,
    max_degree: int = 1,
    n_knots: int = 10,
    min_improvement: float = 1e-6,
    penalty: float = 2.0,
    prune: bool = True,
) -> ModelFit:
    """Fit package-native multivariate adaptive regression splines."""

    params = {
        "max_terms": int(max_terms),
        "max_degree": int(max_degree),
        "n_knots": int(n_knots),
        "min_improvement": float(min_improvement),
        "penalty": float(penalty),
        "prune": bool(prune),
        "implementation_note": "Package-native hinge-basis MARS-style estimator.",
    }
    return fit_estimator(
        MARSRegressor(
            max_terms=int(max_terms),
            max_degree=int(max_degree),
            n_knots=int(n_knots),
            min_improvement=float(min_improvement),
            penalty=float(penalty),
            prune=bool(prune),
        ),
        X,
        y,
        model="mars",
        metadata=params,
    )


__all__ = ["MARSRegressor", "mars"]
