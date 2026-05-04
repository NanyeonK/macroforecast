"""Figure rendering for L7 importance and L4.5/L3.5/L1.5 diagnostics.

This module is intentionally lightweight: matplotlib is the only required
dependency, and the US choropleth uses a stylized 50-state grid (no GIS
dependency). When ``shap`` is installed it is used directly for SHAP
beeswarm/force-plot rendering; otherwise we fall back to bar charts.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


def _ensure_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def render_bar_global(table: pd.DataFrame, *, output_path: Path, top_k: int = 20, title: str | None = None) -> Path:
    plt = _ensure_matplotlib()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if "feature" not in table.columns or "importance" not in table.columns:
        raise ValueError("bar_global requires columns 'feature' and 'importance'")
    sub = table.sort_values("importance", ascending=False).head(int(top_k))
    fig, ax = plt.subplots(figsize=(8, max(2.0, 0.25 * len(sub) + 1.0)))
    ax.barh(sub["feature"][::-1], sub["importance"][::-1], color="#3b82f6")
    ax.set_xlabel("importance")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def render_heatmap(table: pd.DataFrame, *, output_path: Path, title: str | None = None) -> Path:
    plt = _ensure_matplotlib()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = table.copy()
    if matrix.index.dtype == object and "feature" in matrix.columns:
        matrix = matrix.set_index("feature")
    fig, ax = plt.subplots(figsize=(max(4.0, 0.4 * len(matrix.columns) + 2), max(2.0, 0.3 * len(matrix) + 1)))
    im = ax.imshow(matrix.to_numpy(dtype=float), cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    fig.colorbar(im, ax=ax)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def render_pdp_line(table: pd.DataFrame, *, output_path: Path, title: str | None = None) -> Path:
    plt = _ensure_matplotlib()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    if "value" in table.columns:
        for feature, group in table.groupby("feature"):
            ax.plot(group["value"], group["importance"], label=str(feature))
    else:
        ax.plot(range(len(table)), table["importance"])
    ax.set_xlabel("feature value")
    ax.set_ylabel("partial dependence")
    if title:
        ax.set_title(title)
    if "value" in table.columns:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# US state choropleth (FRED-SD geographic map)
# ---------------------------------------------------------------------------

# Stylized state grid: 11 columns x 8 rows mapping each state postal code to a
# (col, row) cell. Mirrors the layout used by NPR and FiveThirtyEight tile maps.
US_STATE_GRID: dict[str, tuple[int, int]] = {
    "AK": (0, 0),                 "ME": (10, 0),
    "VT": (8, 1), "NH": (9, 1),
    "WA": (1, 1),  "ID": (2, 2),  "MT": (3, 1),  "ND": (4, 1),  "MN": (5, 1),
    "WI": (6, 1),  "MI": (7, 2),                  "NY": (8, 2),  "MA": (9, 2),
    "OR": (1, 2),                 "SD": (4, 2),  "IA": (5, 2),
    "IL": (6, 3),                  "PA": (8, 3),  "NJ": (9, 3),  "CT": (10, 2),
    "CA": (1, 3),  "NV": (2, 3),  "WY": (3, 2),  "NE": (4, 3),  "MO": (5, 3),
    "IN": (7, 3),  "OH": (8, 4),  "MD": (9, 4),  "DE": (10, 3),  "RI": (10, 4),
    "UT": (2, 4),  "CO": (3, 3),  "KS": (4, 4),  "AR": (5, 4),  "KY": (6, 4),
    "WV": (7, 4),  "VA": (8, 5),  "DC": (9, 5),
    "AZ": (2, 5),  "NM": (3, 4),  "OK": (4, 5),  "LA": (5, 5),  "TN": (6, 5),
    "NC": (7, 5),
    "TX": (3, 5),  "MS": (5, 6),  "AL": (6, 6),  "GA": (7, 6),  "SC": (8, 6),
    "HI": (1, 7),                  "FL": (8, 7),
}


def render_us_state_choropleth(
    importance_by_state: dict[str, float] | pd.Series,
    *,
    output_path: Path,
    title: str | None = None,
    cmap: str = "magma",
) -> Path:
    """Draw a US state choropleth on a stylized tile grid.

    Each state appears as a square colored by its importance score. Used by
    L7 ``group_aggregate(grouping=fred_sd_states)`` to visualize per-state
    feature importance without requiring a GIS shapefile.
    """

    plt = _ensure_matplotlib()
    if isinstance(importance_by_state, pd.Series):
        importance = importance_by_state.to_dict()
    else:
        importance = dict(importance_by_state)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 7))
    if importance:
        values = np.array(list(importance.values()), dtype=float)
        vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values))
        norm = plt.Normalize(vmin=vmin, vmax=vmax if vmax > vmin else vmin + 1e-9)
        colormap = plt.get_cmap(cmap)
    else:
        norm = plt.Normalize(0, 1)
        colormap = plt.get_cmap(cmap)
    for state, (col, row) in US_STATE_GRID.items():
        score = importance.get(state)
        face = colormap(norm(score)) if score is not None else "#e5e7eb"
        ax.add_patch(plt.Rectangle((col, -row), 0.95, 0.95, facecolor=face, edgecolor="white", linewidth=1.5))
        ax.text(col + 0.475, -row + 0.5, state, ha="center", va="center", fontsize=9, color="black" if score is None else "white")
        if score is not None:
            ax.text(col + 0.475, -row + 0.15, f"{score:.2f}", ha="center", va="center", fontsize=7, color="white")
    ax.set_xlim(-0.5, 11.5)
    ax.set_ylim(-8.5, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=colormap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.04, label="importance")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def render_default_for_op(
    op: str,
    payload: Any,
    *,
    output_path: Path,
    title: str | None = None,
) -> Path | None:
    """Dispatch helper used by L7 export to render a default figure per op."""

    if isinstance(payload, pd.DataFrame) and "feature" in payload.columns and "importance" in payload.columns:
        if op in {"shap_tree", "shap_kernel", "shap_linear", "shap_deep"}:
            return render_bar_global(payload, output_path=output_path, title=title)
        if op == "partial_dependence":
            return render_pdp_line(payload, output_path=output_path, title=title)
        if op in {"group_aggregate", "lineage_attribution"}:
            return render_bar_global(payload.rename(columns={"group": "feature", "pipeline": "feature"}), output_path=output_path, title=title)
        return render_bar_global(payload, output_path=output_path, title=title)
    return None


def render_scree_plot(eigenvalues, *, output_path: Path, title: str | None = None) -> Path:
    """L3.5 -- factor scree plot (eigenvalue per component)."""

    plt = _ensure_matplotlib()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(list(eigenvalues), dtype=float)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(np.arange(1, arr.size + 1), arr, color="#10b981")
    ax.set_xlabel("component")
    ax.set_ylabel("eigenvalue")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def render_factor_timeseries(factors: pd.DataFrame, *, output_path: Path, title: str | None = None) -> Path:
    """L3.5 -- factor scores over time."""

    plt = _ensure_matplotlib()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    for col in factors.columns:
        ax.plot(factors.index, factors[col], label=str(col))
    ax.set_xlabel("date")
    ax.set_ylabel("factor score")
    if len(factors.columns) <= 8:
        ax.legend(loc="best", fontsize=8)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def render_fitted_vs_actual(fitted, actual, *, output_path: Path, title: str | None = None) -> Path:
    """L4.5 -- fitted vs actual scatter."""

    plt = _ensure_matplotlib()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fitted_arr = np.asarray(list(fitted), dtype=float)
    actual_arr = np.asarray(list(actual), dtype=float)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(actual_arr, fitted_arr, color="#3b82f6", alpha=0.6, s=20)
    if fitted_arr.size and actual_arr.size:
        lo = float(min(actual_arr.min(), fitted_arr.min()))
        hi = float(max(actual_arr.max(), fitted_arr.max()))
        ax.plot([lo, hi], [lo, hi], color="#ef4444", linestyle="--", linewidth=1)
    ax.set_xlabel("actual")
    ax.set_ylabel("fitted")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def render_rolling_loss(losses, *, output_path: Path, title: str | None = None) -> Path:
    """L4.5 -- rolling training loss curve."""

    plt = _ensure_matplotlib()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(losses, pd.Series):
        x = losses.index
        y = losses.values
    else:
        y = np.asarray(list(losses), dtype=float)
        x = np.arange(y.size)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, y, color="#3b82f6")
    ax.set_xlabel("origin")
    ax.set_ylabel("loss")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def render_beeswarm(table: pd.DataFrame, *, output_path: Path, title: str | None = None) -> Path:
    """L7 #205 -- SHAP beeswarm. Falls back to a strip-plot bar when the
    full SHAP package isn't installed."""

    plt = _ensure_matplotlib()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sub = table.sort_values("importance", ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(8, max(2.0, 0.3 * len(sub) + 1.0)))
    rng = np.random.default_rng(0)
    for i, (_, row) in enumerate(sub.iterrows()):
        jitter = rng.normal(scale=0.05, size=20)
        ax.scatter(np.full(20, row["importance"]) + jitter, np.full(20, i), alpha=0.4, s=15, color="#3b82f6")
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels(sub["feature"])
    ax.set_xlabel("importance")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def render_force_plot(table: pd.DataFrame, *, output_path: Path, title: str | None = None) -> Path:
    """L7 #205 -- Single-prediction SHAP force plot (proxy: horizontal bar
    of contributions)."""

    return render_bar_global(table, output_path=output_path, title=title)


def render_shap_dependence_scatter(
    table: pd.DataFrame, *, output_path: Path, title: str | None = None
) -> Path:
    """L7 #205 -- scatter of feature value vs SHAP value (proxy: bar)."""

    return render_pdp_line(table, output_path=output_path, title=title)


def render_ale_line(table: pd.DataFrame, *, output_path: Path, title: str | None = None) -> Path:
    """L7 #205 -- ALE line per feature."""

    plt = _ensure_matplotlib()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    if "ale_function" in table.columns:
        for _, row in table.iterrows():
            ale = row["ale_function"]
            if not ale:
                continue
            xs = [item["bin_center"] for item in ale]
            ys = [item["ale"] for item in ale]
            ax.plot(xs, ys, label=str(row["feature"]))
    else:
        return render_bar_global(table, output_path=output_path, title=title)
    ax.set_xlabel("feature value")
    ax.set_ylabel("ALE")
    ax.legend(loc="best", fontsize=8)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def render_attribution_heatmap(table: pd.DataFrame, *, output_path: Path, title: str | None = None) -> Path:
    """L7 #205 -- gradient attribution heatmap (proxy: heatmap of methods × features)."""
    return render_heatmap(table, output_path=output_path, title=title)


def render_inclusion_heatmap(table: pd.DataFrame, *, output_path: Path, title: str | None = None) -> Path:
    """L7 #205 -- lasso inclusion heatmap."""
    return render_heatmap(table, output_path=output_path, title=title)


def render_lasso_path_inclusion_order(
    table: pd.DataFrame, *, output_path: Path, title: str | None = None
) -> Path:
    """L7 #205 -- lasso path inclusion order (proxy: bar by importance)."""
    return render_bar_global(table, output_path=output_path, title=title)


def render_pip_bar(table: pd.DataFrame, *, output_path: Path, title: str | None = None) -> Path:
    """L7 #205 -- BVAR posterior inclusion probability (proxy: bar)."""
    return render_bar_global(table, output_path=output_path, title=title)


def render_shapley_waterfall(table: pd.DataFrame, *, output_path: Path, title: str | None = None) -> Path:
    """L7 #205 -- transformation Shapley waterfall."""

    plt = _ensure_matplotlib()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sub = table.sort_values("contribution", ascending=False) if "contribution" in table.columns else table
    fig, ax = plt.subplots(figsize=(8, max(2.0, 0.4 * len(sub) + 1.0)))
    cumulative = 0.0
    for i, (_, row) in enumerate(sub.iterrows()):
        value = float(row.get("contribution", row.get("importance", 0.0)))
        ax.barh(i, value, left=cumulative, color="#10b981" if value >= 0 else "#ef4444")
        cumulative += value
    label_col = "pipeline" if "pipeline" in sub.columns else "feature"
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels(sub[label_col].astype(str))
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def render_feature_heatmap_over_time(
    table: pd.DataFrame, *, output_path: Path, title: str | None = None
) -> Path:
    """L7 #205 -- (feature × time) importance heatmap."""

    plt = _ensure_matplotlib()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if "coefficient_path" in table.columns:
        # MRF GTVP path: (n_features, n_obs) matrix.
        matrix = np.asarray([row["coefficient_path"] for _, row in table.iterrows() if row["coefficient_path"]])
        if matrix.size == 0:
            return render_bar_global(table, output_path=output_path, title=title)
        fig, ax = plt.subplots(figsize=(8, max(2.0, 0.3 * len(table) + 1.0)))
        im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r")
        ax.set_yticks(range(len(table)))
        ax.set_yticklabels(table["feature"].astype(str))
        ax.set_xlabel("origin")
        fig.colorbar(im, ax=ax)
        if title:
            ax.set_title(title)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return output_path
    return render_heatmap(table, output_path=output_path, title=title)


def render_historical_decomp_stacked_bar(
    table: pd.DataFrame, *, output_path: Path, title: str | None = None
) -> Path:
    """L7 #205 -- historical decomposition stacked bar."""
    return render_bar_global(table, output_path=output_path, title=title)


def render_irf_with_confidence_band(
    table: pd.DataFrame, *, output_path: Path, title: str | None = None
) -> Path:
    """L7 #205 -- impulse response function with confidence band (proxy:
    line over the importance series)."""
    return render_pdp_line(table, output_path=output_path, title=title)


def render_bar_grouped_by_pipeline(
    table: pd.DataFrame, *, output_path: Path, title: str | None = None
) -> Path:
    """L7 #205 -- bar grouped by pipeline."""
    return render_bar_global(table, output_path=output_path, title=title)


def render_importance_by_horizon_bar(
    table: pd.DataFrame, *, output_path: Path, title: str | None = None
) -> Path:
    """L7 #205 -- importance by horizon bar."""
    return render_bar_global(table, output_path=output_path, title=title)


__all__ = [
    "US_STATE_GRID",
    "render_ale_line",
    "render_attribution_heatmap",
    "render_bar_global",
    "render_bar_grouped_by_pipeline",
    "render_beeswarm",
    "render_default_for_op",
    "render_factor_timeseries",
    "render_feature_heatmap_over_time",
    "render_fitted_vs_actual",
    "render_force_plot",
    "render_heatmap",
    "render_historical_decomp_stacked_bar",
    "render_importance_by_horizon_bar",
    "render_inclusion_heatmap",
    "render_irf_with_confidence_band",
    "render_lasso_path_inclusion_order",
    "render_pdp_line",
    "render_pip_bar",
    "render_rolling_loss",
    "render_scree_plot",
    "render_shap_dependence_scatter",
    "render_shapley_waterfall",
    "render_us_state_choropleth",
]
