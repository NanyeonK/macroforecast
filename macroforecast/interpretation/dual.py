from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from macroforecast.interpretation.core import (
    data_portfolio_diagnostics as _data_portfolio_diagnostics,
    episode_group_weights as _episode_group_weights,
    observation_weights as _observation_weights,
    outcome_contributions as _outcome_contributions,
    top_episodes as _top_episodes,
)


@dataclass(frozen=True)
class DualInterpretationResult:
    """Paper-aligned dual interpretation output bundle.

    Goulet Coulombe, Goebel, and Klieber's DualML code reports observation
    weights, observation contributions, and data-portfolio diagnostics as
    connected objects. The result container keeps that relation explicit while
    still exposing output-ready tables through ``to_tables``.
    """

    weights: pd.DataFrame
    contributions: pd.DataFrame | None = None
    diagnostics: pd.DataFrame | None = None
    top_observations: pd.DataFrame | None = None
    group_weights: pd.DataFrame | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    metadata_schema: dict[str, Any] = field(
        default_factory=lambda: {"kind": "dual_interpretation_result", "version": 1}
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata_schema": dict(self.metadata_schema),
            "metadata": dict(self.metadata),
            "tables": {
                "observation_weights": _table_summary(self.weights),
                "observation_contributions": _table_summary(self.contributions),
                "forecast_diagnostics": _table_summary(self.diagnostics),
                "top_observations": _table_summary(self.top_observations),
                "group_observation_weights": _table_summary(self.group_weights),
            },
        }

    def to_tables(self, *, prefix: str = "dual") -> dict[str, pd.DataFrame]:
        """Return output-ready tables with paper-aligned names."""

        safe_prefix = _safe_table_name(prefix)
        tables: dict[str, pd.DataFrame] = {
            f"{safe_prefix}_observation_weights": _attach_dual_table_schema(
                self.weights.copy(),
                kind="dual_observation_weights_table",
                section="observation_weights",
            )
        }
        if self.contributions is not None:
            tables[f"{safe_prefix}_observation_contributions"] = _attach_dual_table_schema(
                self.contributions.copy(),
                kind="dual_observation_contributions_table",
                section="observation_contributions",
            )
        if self.diagnostics is not None:
            tables[f"{safe_prefix}_forecast_diagnostics"] = _attach_dual_table_schema(
                self.diagnostics.copy(),
                kind="dual_forecast_diagnostics_table",
                section="forecast_diagnostics",
            )
        if self.top_observations is not None:
            tables[f"{safe_prefix}_top_observations"] = _attach_dual_table_schema(
                self.top_observations.copy(),
                kind="dual_top_observations_table",
                section="top_observations",
            )
        if self.group_weights is not None:
            tables[
                f"{safe_prefix}_group_observation_weights"
            ] = _attach_dual_table_schema(
                self.group_weights.copy(),
                kind="dual_group_observation_weights_table",
                section="group_observation_weights",
            )
        tables[f"{safe_prefix}_metadata"] = _attach_dual_table_schema(
            _metadata_frame(self.metadata),
            kind="dual_metadata_table",
            section="metadata",
        )
        return tables


def dual_interpretation(
    model: Any | None,
    X_train: pd.DataFrame,
    y_train: pd.Series | Sequence[float],
    X_test: pd.DataFrame | None = None,
    *,
    method: str = "auto",
    lambda_: float = 1e-8,
    kernel: str = "linear",
    sigma: float = 1.0,
    add_intercept: bool = False,
    ridge_penalty_scale: str = "n_train",
    normalize: bool = False,
    center: bool = False,
    include_base: bool = False,
    top_n: int = 10,
    top_sort_by: str = "abs_weight",
    top_q: float = 0.05,
    groups: Mapping[str, Sequence[Any]] | None = None,
    include_contributions: bool = True,
    include_diagnostics: bool = True,
    include_top_observations: bool = True,
    include_group_weights: bool | None = None,
) -> DualInterpretationResult:
    """Run the ridge/KRR/RF DualML interpretation path in one callable."""

    weights = observation_weights(
        model,
        X_train,
        X_test,
        method=method,
        lambda_=lambda_,
        kernel=kernel,
        sigma=sigma,
        add_intercept=add_intercept,
        ridge_penalty_scale=ridge_penalty_scale,
        normalize=normalize,
    )
    contributions = (
        observation_contributions(
            weights,
            y_train,
            center=center,
            include_base=include_base,
        )
        if include_contributions
        else None
    )
    weight_source = contributions if contributions is not None else weights
    diagnostics = (
        forecast_diagnostics(weights, top_q=top_q) if include_diagnostics else None
    )
    top = (
        top_observations(
            weight_source,
            y_train=None,
            n=top_n,
            sort_by=top_sort_by,
        )
        if include_top_observations
        else None
    )
    make_group_weights = bool(groups) if include_group_weights is None else bool(include_group_weights)
    grouped = (
        group_observation_weights(weight_source, groups or {}, y_train=None)
        if make_group_weights
        else None
    )
    schema = weights.attrs.get("macroforecast_metadata_schema", {})
    return DualInterpretationResult(
        weights=weights,
        contributions=contributions,
        diagnostics=diagnostics,
        top_observations=top,
        group_weights=grouped,
        metadata={
            "paper": "Dual Interpretation of Machine Learning Forecasts",
            "authors": "Goulet Coulombe, Goebel, and Klieber",
            "year": 2024,
            "arxiv": "2412.13076",
            "method": schema.get("method", method),
            "implemented_routes": ["ridge", "krr", "random_forest"],
            "unsupported_routes": [
                "boosted_tree_axil",
                "lgb_plus_channel_weights",
                "nn_embedding_ridge",
                "classification_log_odds",
            ],
            "include_contributions": bool(include_contributions),
            "include_diagnostics": bool(include_diagnostics),
            "include_top_observations": bool(include_top_observations),
            "include_group_weights": bool(make_group_weights),
            "top_n": int(top_n),
            "top_q": float(top_q),
            "center": bool(center),
            "normalize": bool(normalize),
        },
    )


def dual_from_forecast_result(
    result: Any,
    model: Any | None,
    X_train: pd.DataFrame,
    y_train: pd.Series | Sequence[float],
    X_test: pd.DataFrame | None = None,
    *,
    attach: bool = True,
    sidecar_name: str = "dual",
    **kwargs: Any,
) -> Any:
    """Build a dual interpretation sidecar for a completed forecast result.

    A forecast table cannot reconstruct the exact train/test feature matrices
    used by the fitted model. The caller therefore passes the fitted model,
    training features, training target, and forecast-row features explicitly.
    """

    dual = dual_interpretation(model, X_train, y_train, X_test, **kwargs)
    dual.metadata["forecast_result"] = {
        "forecast_rows": int(len(result.to_frame())) if hasattr(result, "to_frame") else None,
        "metadata_schema": getattr(result, "metadata", {}).get("metadata_schema")
        if hasattr(result, "metadata")
        else None,
    }
    if not attach:
        return dual
    if not hasattr(result, "with_sidecar"):
        raise TypeError("result must be a ForecastResult when attach=True")
    return result.with_sidecar(sidecar_name, dual)


def observation_weights(
    model: Any | None,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame | None = None,
    *,
    method: str = "auto",
    lambda_: float = 1e-8,
    kernel: str = "linear",
    sigma: float = 1.0,
    add_intercept: bool = False,
    ridge_penalty_scale: str = "n_train",
    normalize: bool = False,
) -> pd.DataFrame:
    """Return DualML observation/data-portfolio weights."""

    return _observation_weights(
        model,
        X_train,
        X_test,
        method=method,
        lambda_=lambda_,
        kernel=kernel,
        sigma=sigma,
        add_intercept=add_intercept,
        ridge_penalty_scale=ridge_penalty_scale,
        normalize=normalize,
    )


def observation_contributions(
    weights: pd.DataFrame,
    y_train: pd.Series | Sequence[float],
    *,
    center: bool = False,
    include_base: bool = False,
) -> pd.DataFrame:
    """Convert observation weights into observation-level forecast contributions."""

    table = _outcome_contributions(
        weights,
        y_train,
        center=center,
        include_base=include_base,
    )
    table.attrs["macroforecast_metadata_schema"] = {
        **dict(table.attrs.get("macroforecast_metadata_schema", {})),
        "kind": "observation_contributions",
    }
    return table


def forecast_diagnostics(weights: pd.DataFrame, *, top_q: float = 0.05) -> pd.DataFrame:
    """Return concentration, short-position, leverage, and turnover diagnostics."""

    table = _data_portfolio_diagnostics(weights, top_q=top_q)
    table.attrs["macroforecast_metadata_schema"] = {
        **dict(table.attrs.get("macroforecast_metadata_schema", {})),
        "kind": "forecast_diagnostics",
    }
    return table


def top_observations(
    weights: pd.DataFrame,
    *,
    y_train: pd.Series | Sequence[float] | None = None,
    n: int = 10,
    sort_by: str = "abs_weight",
) -> pd.DataFrame:
    """Return the largest historical observations per forecast row."""

    table = _top_episodes(weights, y_train=y_train, n=n, sort_by=sort_by)
    table.attrs["macroforecast_metadata_schema"] = {
        **dict(table.attrs.get("macroforecast_metadata_schema", {})),
        "kind": "top_observations",
    }
    return table


def group_observation_weights(
    weights: pd.DataFrame,
    groups: Mapping[str, Sequence[Any]],
    *,
    y_train: pd.Series | Sequence[float] | None = None,
) -> pd.DataFrame:
    """Aggregate observation weights over named historical groups."""

    table = _episode_group_weights(weights, groups, y_train=y_train)
    table.attrs["macroforecast_metadata_schema"] = {
        **dict(table.attrs.get("macroforecast_metadata_schema", {})),
        "kind": "group_observation_weights",
    }
    return table


outcome_contributions = observation_contributions
data_portfolio_diagnostics = forecast_diagnostics
top_episodes = top_observations
episode_group_weights = group_observation_weights


def _attach_dual_table_schema(
    table: pd.DataFrame,
    *,
    kind: str,
    section: str,
) -> pd.DataFrame:
    schema = dict(table.attrs.get("macroforecast_metadata_schema", {}))
    metadata = dict(schema.get("metadata", {})) if isinstance(schema.get("metadata"), Mapping) else {}
    metadata.update({"section": section, "source": "DualInterpretationResult"})
    table.attrs["macroforecast_metadata_schema"] = {
        "kind": kind,
        "version": 1,
        "method": schema.get("method", "dual_interpretation"),
        "model": schema.get("model"),
        "n_features": schema.get("n_features", 0),
        "columns": [str(column) for column in table.columns],
        "reference": {
            "class": "paper_formula_adapter",
            "reference": "Dual Interpretation of Machine Learning Forecasts, Goulet Coulombe, Goebel, and Klieber (2024)",
            "alignment": "output table wrapper around DualML observation weights, contributions, diagnostics, and groups",
        },
        "metadata": metadata,
    }
    return table


def _metadata_frame(metadata: Mapping[str, Any]) -> pd.DataFrame:
    rows = [{"key": str(key), "value": value} for key, value in metadata.items()]
    return pd.DataFrame(rows, columns=["key", "value"])


def _safe_table_name(value: str) -> str:
    text = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(value).strip())
    text = "_".join(part for part in text.split("_") if part)
    return text.lower() or "dual"


def _table_summary(table: pd.DataFrame | None) -> dict[str, Any] | None:
    if table is None:
        return None
    return {
        "shape": [int(table.shape[0]), int(table.shape[1])],
        "columns": [str(column) for column in table.columns],
        "metadata_schema": dict(table.attrs.get("macroforecast_metadata_schema", {})),
    }


__all__ = [
    "DualInterpretationResult",
    "data_portfolio_diagnostics",
    "dual_from_forecast_result",
    "dual_interpretation",
    "episode_group_weights",
    "forecast_diagnostics",
    "group_observation_weights",
    "observation_contributions",
    "observation_weights",
    "outcome_contributions",
    "top_episodes",
    "top_observations",
]
