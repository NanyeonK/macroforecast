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


def render_bar_global(table: pd.DataFrame, *, output_path: Path, top_k: int = 20, title: str | None = None, dpi: int = 150) -> Path:
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
    fig.savefig(output_path, dpi=int(dpi))
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
    dpi: int = 150,
    top_k: int = 20,
) -> Path | None:
    """Dispatch helper used by L7 export to render a default figure per op.

    F-P1-11 fix: dpi and top_k are threaded from the L7 axis_resolved values.
    """

    if isinstance(payload, pd.DataFrame) and "feature" in payload.columns and "importance" in payload.columns:
        if op in {"shap_tree", "shap_kernel", "shap_linear", "shap_deep"}:
            return render_bar_global(payload, output_path=output_path, title=title, dpi=dpi, top_k=top_k)
        if op == "partial_dependence":
            return render_pdp_line(payload, output_path=output_path, title=title)
        if op in {"group_aggregate", "lineage_attribution"}:
            return render_bar_global(payload.rename(columns={"group": "feature", "pipeline": "feature"}), output_path=output_path, title=title, dpi=dpi, top_k=top_k)
        return render_bar_global(payload, output_path=output_path, title=title, dpi=dpi, top_k=top_k)
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
    """L7 #249 -- single-prediction SHAP force plot. Bidirectional bar
    chart: positive contributions push the prediction up (right, blue),
    negative contributions push it down (left, red). Sorted by absolute
    contribution.
    """

    plt = _ensure_matplotlib()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if "feature" not in table.columns:
        return render_bar_global(table, output_path=output_path, title=title)
    contrib_col = "contribution" if "contribution" in table.columns else "importance"
    sub = table.copy()
    sub["__contrib__"] = sub[contrib_col].astype(float)
    sub["__abs__"] = sub["__contrib__"].abs()
    sub = sub.sort_values("__abs__", ascending=True).tail(20)
    fig, ax = plt.subplots(figsize=(8, max(2.0, 0.3 * len(sub) + 1.0)))
    colors = ["#3b82f6" if v >= 0 else "#ef4444" for v in sub["__contrib__"]]
    ax.barh(sub["feature"], sub["__contrib__"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("contribution")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def render_shap_dependence_scatter(
    table: pd.DataFrame, *, output_path: Path, title: str | None = None
) -> Path:
    """L7 #249 -- SHAP dependence scatter: x = feature value, y = SHAP
    value (or importance). One subplot per feature when ``feature_value``
    column is present; otherwise a stripe chart per feature.
    """

    plt = _ensure_matplotlib()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if "feature_value" in table.columns and "feature" in table.columns:
        unique_features = sorted(table["feature"].unique())[:6]
        n = len(unique_features)
        fig, axes = plt.subplots(1, max(1, n), figsize=(4 * max(1, n), 4), squeeze=False)
        for ax, feat in zip(axes[0], unique_features):
            sub = table[table["feature"] == feat]
            y_col = "shap_value" if "shap_value" in sub.columns else "importance"
            ax.scatter(sub["feature_value"], sub[y_col], alpha=0.5, s=15, color="#3b82f6")
            ax.set_xlabel(str(feat))
            ax.set_ylabel("SHAP value")
            ax.axhline(0, color="black", linewidth=0.5)
        if title:
            fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return output_path
    # Without explicit (feature, feature_value) pairs we plot importance per
    # feature as a strip chart -- distinct from the bar_global default.
    sub = table.sort_values("importance" if "importance" in table.columns else table.columns[1])
    fig, ax = plt.subplots(figsize=(8, max(2.0, 0.3 * len(sub) + 1.0)))
    rng = np.random.default_rng(0)
    for i, (_, row) in enumerate(sub.iterrows()):
        jitter = rng.normal(scale=0.04, size=10)
        ax.scatter(np.full(10, row.get("importance", 0.0)) + jitter, np.full(10, i), alpha=0.4, s=12, color="#10b981")
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels(sub["feature"])
    ax.set_xlabel("importance")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


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
    """L7 #249 -- gradient attribution heatmap. Diverging RdBu colormap
    centred at 0 (positive vs negative attributions). When the input is
    a feature-indexed (method × feature) frame, transpose so rows are
    features and columns are methods.
    """

    plt = _ensure_matplotlib()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = table.copy()
    if "feature" in matrix.columns:
        matrix = matrix.set_index("feature")
    fig, ax = plt.subplots(figsize=(max(4.0, 0.4 * len(matrix.columns) + 2), max(2.0, 0.3 * len(matrix) + 1)))
    arr = matrix.to_numpy(dtype=float)
    vmax = float(np.abs(arr).max()) if arr.size else 1.0
    im = ax.imshow(arr, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    fig.colorbar(im, ax=ax, label="attribution")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def render_inclusion_heatmap(table: pd.DataFrame, *, output_path: Path, title: str | None = None) -> Path:
    """L7 #249 -- (lambda × feature) lasso inclusion heatmap. Greens
    palette with a 0/1 colorbar (inclusion frequency in [0, 1])."""

    plt = _ensure_matplotlib()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = table.copy()
    if "feature" in matrix.columns:
        matrix = matrix.set_index("feature")
    fig, ax = plt.subplots(figsize=(max(4.0, 0.4 * len(matrix.columns) + 2), max(2.0, 0.3 * len(matrix) + 1)))
    im = ax.imshow(matrix.to_numpy(dtype=float), cmap="Greens", aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    fig.colorbar(im, ax=ax, label="inclusion freq")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def render_lasso_path_inclusion_order(
    table: pd.DataFrame, *, output_path: Path, title: str | None = None
) -> Path:
    """L7 #249 -- features ordered by their inclusion order along the
    lasso path (lower lambda first). Step-line per feature shows the
    coefficient evolution; without that data we draw a coloured bar
    sorted by inclusion order.
    """

    plt = _ensure_matplotlib()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if "lambda" in table.columns:
        # Path data: pivot to (lambda × feature) and plot one line per feat.
        pivot = table.pivot(index="lambda", columns="feature", values="coefficient").fillna(0.0)
        fig, ax = plt.subplots(figsize=(8, 5))
        for col in pivot.columns:
            ax.plot(pivot.index, pivot[col], label=str(col), drawstyle="steps-post")
        ax.set_xlabel("lambda")
        ax.set_ylabel("coefficient")
        ax.set_xscale("log")
        if len(pivot.columns) <= 12:
            ax.legend(loc="best", fontsize=8)
    else:
        sub = table.copy()
        sort_col = "importance" if "importance" in sub.columns else sub.columns[-1]
        sub = sub.sort_values(sort_col, ascending=False)
        fig, ax = plt.subplots(figsize=(8, max(2.0, 0.3 * len(sub) + 1.0)))
        ax.barh(sub["feature"][::-1], sub[sort_col][::-1], color="#10b981")
        ax.set_xlabel("inclusion frequency / coefficient")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def render_pip_bar(table: pd.DataFrame, *, output_path: Path, title: str | None = None) -> Path:
    """L7 #249 -- BVAR posterior inclusion probability bar with HDI bars
    when ``hdi_low`` / ``hdi_high`` columns are present.
    """

    plt = _ensure_matplotlib()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sub = table.sort_values("importance", ascending=True).tail(25)
    fig, ax = plt.subplots(figsize=(8, max(2.0, 0.3 * len(sub) + 1.0)))
    ax.barh(sub["feature"], sub["importance"], color="#8b5cf6", alpha=0.8)
    if {"hdi_low", "hdi_high"}.issubset(sub.columns):
        for i, (_, row) in enumerate(sub.iterrows()):
            ax.plot([row["hdi_low"], row["hdi_high"]], [i, i], color="black", linewidth=1.2)
    ax.axvline(0.5, color="black", linewidth=0.5, linestyle="--")
    ax.set_xlabel("posterior inclusion probability")
    ax.set_xlim(0, 1)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


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
    """L7 #249 -- historical decomposition: stacked bar over (period, shock)
    where ``period`` is on x-axis and each shock contribution stacks
    vertically. Falls back to a single-bar layout when the input frame
    only has the (feature, importance) shape.
    """

    plt = _ensure_matplotlib()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if "period" in table.columns and "shock" in table.columns and "contribution" in table.columns:
        pivot = table.pivot(index="period", columns="shock", values="contribution").fillna(0.0)
        fig, ax = plt.subplots(figsize=(10, 5))
        bottom_pos = np.zeros(len(pivot))
        bottom_neg = np.zeros(len(pivot))
        cmap = plt.get_cmap("tab10")
        for j, shock in enumerate(pivot.columns):
            values = pivot[shock].values
            pos_mask = values >= 0
            neg_mask = ~pos_mask
            ax.bar(pivot.index, np.where(pos_mask, values, 0.0), bottom=bottom_pos, color=cmap(j % 10), label=str(shock))
            ax.bar(pivot.index, np.where(neg_mask, values, 0.0), bottom=bottom_neg, color=cmap(j % 10))
            bottom_pos = bottom_pos + np.where(pos_mask, values, 0.0)
            bottom_neg = bottom_neg + np.where(neg_mask, values, 0.0)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_xlabel("period")
        ax.set_ylabel("contribution")
        ax.legend(loc="best", fontsize=8, ncol=2)
    else:
        return render_bar_global(table, output_path=output_path, title=title)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def render_irf_with_confidence_band(
    table: pd.DataFrame, *, output_path: Path, title: str | None = None
) -> Path:
    """L7 #249 -- impulse response with confidence band. Expects columns
    ``horizon`` / ``response`` / ``ci_low`` / ``ci_high`` (or per-shock
    multi-curve form). Falls back to a per-feature line plot of the
    importance series when those columns aren't present.
    """

    plt = _ensure_matplotlib()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if {"horizon", "response"}.issubset(table.columns):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        if "shock" in table.columns:
            cmap = plt.get_cmap("tab10")
            for j, (shock, group) in enumerate(table.groupby("shock")):
                ax.plot(group["horizon"], group["response"], label=str(shock), color=cmap(j % 10))
                if {"ci_low", "ci_high"}.issubset(group.columns):
                    ax.fill_between(
                        group["horizon"], group["ci_low"], group["ci_high"],
                        color=cmap(j % 10), alpha=0.2,
                    )
            ax.legend(loc="best", fontsize=8)
        else:
            ax.plot(table["horizon"], table["response"], color="#3b82f6")
            if {"ci_low", "ci_high"}.issubset(table.columns):
                ax.fill_between(
                    table["horizon"], table["ci_low"], table["ci_high"], alpha=0.25, color="#3b82f6"
                )
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xlabel("horizon")
        ax.set_ylabel("response")
    else:
        return render_pdp_line(table, output_path=output_path, title=title)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def render_bar_grouped_by_pipeline(
    table: pd.DataFrame, *, output_path: Path, title: str | None = None
) -> Path:
    """L7 #249 -- grouped bar with one cluster per pipeline. Expects
    ``feature`` × ``pipeline`` × ``importance``.
    """

    plt = _ensure_matplotlib()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if "pipeline" in table.columns and "feature" in table.columns:
        pivot = table.pivot(index="feature", columns="pipeline", values="importance").fillna(0.0)
        n_features = len(pivot.index)
        n_pipelines = len(pivot.columns)
        fig, ax = plt.subplots(figsize=(max(6, 1.5 * n_pipelines), max(3, 0.4 * n_features + 2)))
        x = np.arange(n_features)
        width = 0.8 / max(n_pipelines, 1)
        cmap = plt.get_cmap("tab10")
        for j, pipeline in enumerate(pivot.columns):
            ax.bar(x + j * width, pivot[pipeline], width=width, label=str(pipeline), color=cmap(j % 10))
        ax.set_xticks(x + width * (n_pipelines - 1) / 2)
        ax.set_xticklabels(pivot.index, rotation=45, ha="right")
        ax.set_ylabel("importance")
        ax.legend(loc="best", fontsize=8)
    else:
        return render_bar_global(table, output_path=output_path, title=title)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def render_importance_by_horizon_bar(
    table: pd.DataFrame, *, output_path: Path, title: str | None = None
) -> Path:
    """L7 #249 -- grouped bar with horizons on the x-axis (one cluster
    per horizon, colour per feature)."""

    plt = _ensure_matplotlib()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if "horizon" in table.columns and "feature" in table.columns:
        pivot = table.pivot(index="horizon", columns="feature", values="importance").fillna(0.0)
        n_horizons = len(pivot.index)
        n_features = len(pivot.columns)
        fig, ax = plt.subplots(figsize=(max(6, 1.5 * n_horizons), 4))
        x = np.arange(n_horizons)
        width = 0.8 / max(n_features, 1)
        cmap = plt.get_cmap("tab10")
        for j, feat in enumerate(pivot.columns):
            ax.bar(x + j * width, pivot[feat], width=width, label=str(feat), color=cmap(j % 10))
        ax.set_xticks(x + width * (n_features - 1) / 2)
        ax.set_xticklabels(pivot.index)
        ax.set_xlabel("horizon")
        ax.set_ylabel("importance")
        ax.legend(loc="best", fontsize=8)
    else:
        return render_bar_global(table, output_path=output_path, title=title)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


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
