"""macroforecast.wizard — browser-based recipe authoring wizard.

Provides a local Solara web UI for configuring macroforecast recipes with
a 3-pane layout: layer rail, form workspace, and YAML preview.

Quick start::

    macroforecast wizard --port 8765

Or programmatically::

    from macroforecast.wizard import launch
    launch(port=8765, open_browser=True)

Requires the ``[wizard]`` extra::

    pip install 'macroforecast[wizard]'
"""
from __future__ import annotations

try:
    import solara  # noqa: F401
except ImportError as exc:
    raise ImportError(
        "macroforecast wizard requires the [wizard] extra. "
        "Install with: pip install 'macroforecast[wizard]'"
    ) from exc

from .app import WizardApp
from .cli import launch

__all__ = ["WizardApp", "launch"]
