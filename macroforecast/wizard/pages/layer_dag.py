"""DAG placeholder page for L3, L4, L7.

Phase 3 will replace this with a React Flow embedded DAG editor.
"""
from __future__ import annotations

try:
    import solara
except ImportError as exc:
    raise ImportError(
        "macroforecast wizard requires the [wizard] extra. "
        "Install with: pip install 'macroforecast[wizard]'"
    ) from exc

_DOCS_URL = "https://macroforecast.readthedocs.io/"


@solara.component
def LayerDagPlaceholder(layer_id: str) -> None:
    """Informational placeholder for DAG-configured layers (L3/L4/L7).

    Args:
        layer_id: The layer identifier (e.g. "l3").
    """
    with solara.Column(
        style=(
            "align-items:center; justify-content:center; padding:48px; "
            "min-height:300px;"
        )
    ):
        with solara.Column(
            style=(
                "background:#fefce8; border:1px solid #fde047; border-radius:8px; "
                "padding:32px; max-width:500px; align-items:center; text-align:center;"
            )
        ):
            solara.Text(
                f"DAG editor for {layer_id.upper()} is coming in Phase 3.",
                style="font-size:18px; font-weight:bold; color:#713f12; margin-bottom:12px;",
            )
            solara.Text(
                "Use the YAML preview to configure this layer manually.",
                style="color:#92400e; margin-bottom:16px;",
            )
            solara.Text(
                "The DAG editor will provide a visual node graph for configuring "
                f"the {layer_id.upper()} layer's pipeline operations, with drag-and-drop "
                "node placement and connection management.",
                style="color:#78350f; font-size:13px; margin-bottom:20px;",
            )
            with solara.Row(style="gap:8px;"):
                solara.Text("Documentation:", style="color:#6b7280; font-size:13px;")
                solara.Button(
                    "macroforecast docs",
                    href=_DOCS_URL,
                    target="_blank",
                    style="font-size:13px;",
                )


__all__ = ["LayerDagPlaceholder"]
