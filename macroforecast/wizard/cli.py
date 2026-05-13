"""Wizard CLI entry point and programmatic launch function.

The ``launch()`` function starts a blocking Solara server.
``_cmd_wizard()`` is wired into ``macroforecast/scaffold/cli.py``.
"""
from __future__ import annotations

import os
import sys
import webbrowser
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse


def launch(
    port: int = 8765,
    *,
    open_browser: bool = True,
    recipe_path: str | None = None,
) -> None:
    """Start the Solara wizard server and optionally open a browser tab.

    This call is blocking — it returns when the user stops the server
    (Ctrl+C).

    Args:
        port: TCP port to bind the server on.
        open_browser: If True, open the browser tab after the server
            starts.
        recipe_path: Optional path to an existing YAML recipe to load on
            startup.  Raises ``FileNotFoundError`` if the path does not
            exist.
    """
    try:
        import solara  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "macroforecast wizard requires the [wizard] extra. "
            "Install with: pip install 'macroforecast[wizard]'"
        ) from exc

    from macroforecast.wizard.state import RecipeState

    if recipe_path is not None:
        RecipeState.load_from_path(recipe_path)

    # Set the SOLARA_APP env var so the Solara server knows which component
    # to render.  This mirrors how ``solara run <module:Component>`` works.
    os.environ["SOLARA_APP"] = "macroforecast.wizard.app:WizardApp"

    url = f"http://localhost:{port}"
    print(f"[macroforecast wizard] Starting server at {url}", file=sys.stderr)

    # Start browser opener in a daemon thread
    server_started = threading.Event()
    failed = threading.Event()

    if open_browser:
        def _open_browser() -> None:
            server_started.wait(timeout=15)
            if not failed.is_set():
                webbrowser.open(url)

        threading.Thread(target=_open_browser, daemon=True).start()

    # Build and run uvicorn directly, the same way the solara CLI does
    try:
        import uvicorn
        from solara.server import settings as solara_settings
        solara_settings.main.mode = "development"

        LOGGING_CONFIG: dict = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "()": "uvicorn.logging.DefaultFormatter",
                    "fmt": "%(levelprefix)s %(message)s",
                    "use_colors": None,
                },
            },
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
            },
            "loggers": {
                "solara": {"handlers": ["default"], "level": "ERROR"},
                "uvicorn": {"handlers": ["default"], "level": "ERROR"},
                "uvicorn.error": {"level": "ERROR"},
            },
        }

        config = uvicorn.Config(
            app="solara.server.starlette:app",
            host="localhost",
            port=port,
            log_config=LOGGING_CONFIG,
        )
        server_obj = uvicorn.Server(config)

        # Signal "started" once uvicorn is ready
        def _signal_ready() -> None:
            # Poll for the server's started attribute
            deadline = time.time() + 15
            while time.time() < deadline:
                if getattr(server_obj, "started", False):
                    server_started.set()
                    return
                time.sleep(0.1)
            # If we never saw started=True, signal anyway (best effort)
            server_started.set()

        threading.Thread(target=_signal_ready, daemon=True).start()

        server_obj.run()

    except KeyboardInterrupt:
        print("[macroforecast wizard] Server stopped.", file=sys.stderr)
        sys.exit(0)
    finally:
        failed.set()


def _cmd_wizard(args: "argparse.Namespace") -> int:
    """Subcommand handler for ``macroforecast wizard``.

    Wired into ``macroforecast/scaffold/cli.py::main()``.
    """
    try:
        import macroforecast.wizard as _wiz  # noqa: F401
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    recipe_file: str | None = getattr(args, "recipe", None)
    port: int = getattr(args, "port", 8765)
    no_browser: bool = getattr(args, "no_browser", False)

    launch(port=port, open_browser=not no_browser, recipe_path=recipe_file)
    return 0


__all__ = ["launch", "_cmd_wizard"]
