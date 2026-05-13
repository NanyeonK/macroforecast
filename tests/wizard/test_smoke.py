"""Smoke tests — Scenarios SM-01..SM-03.

SM-01 is marked @pytest.mark.slow and requires a running Solara server.
SM-02 and SM-03 are subprocess-based and run in the default pytest run.
"""
from __future__ import annotations

import subprocess
import sys
import signal
import tempfile
import os
import time

import pytest

MACROFORECAST_CLI = os.path.join(
    os.path.dirname(sys.executable), "macroforecast"
)
# Fallback to module invocation if the CLI script is not found
if not os.path.exists(MACROFORECAST_CLI):
    MACROFORECAST_CLI = None


def _run_cli(*args, timeout=15, capture_output=True) -> subprocess.CompletedProcess:
    """Run the macroforecast CLI with the given args."""
    if MACROFORECAST_CLI is not None:
        cmd = [MACROFORECAST_CLI] + list(args)
    else:
        cmd = [sys.executable, "-m", "macroforecast"] + list(args)
    return subprocess.run(
        cmd,
        capture_output=capture_output,
        text=True,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# SM-01: Server starts, health check passes, stops cleanly
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_sm01_server_starts_and_responds():
    """SM-01: wizard server starts within 8s, serves HTTP 200, stops cleanly."""
    import urllib.request
    import urllib.error

    port = 8799

    if MACROFORECAST_CLI is not None:
        cmd = [MACROFORECAST_CLI, "wizard", "--no-browser", f"--port={port}"]
    else:
        cmd = [sys.executable, "-m", "macroforecast", "wizard", "--no-browser", f"--port={port}"]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        url = f"http://localhost:{port}"
        got_200 = False
        deadline = time.time() + 8.0
        while time.time() < deadline:
            try:
                resp = urllib.request.urlopen(url, timeout=1)
                if resp.status in (200, 301, 302):
                    got_200 = True
                    break
            except (urllib.error.URLError, ConnectionRefusedError, OSError):
                pass
            time.sleep(0.5)

        assert got_200, (
            f"Server at {url} did not return HTTP 200 within 8 seconds. "
            f"Process returncode={proc.poll()}"
        )

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    # Subprocess should exit cleanly (0 or negative from signal)
    rc = proc.returncode
    assert rc is not None, "Process did not exit"
    # On SIGTERM: returncode is -signal.SIGTERM on Unix or 1 on Windows
    assert rc <= 0 or rc == 1, (
        f"Unexpected exit code {rc} after SIGTERM (expected 0 or negative or 1)"
    )


# ---------------------------------------------------------------------------
# SM-02: CLI help text includes "wizard" subcommand
# ---------------------------------------------------------------------------

def test_sm02_cli_help_includes_wizard():
    """SM-02: macroforecast --help output contains 'wizard'."""
    result = _run_cli("--help", timeout=10)
    combined = result.stdout + result.stderr
    assert "wizard" in combined, (
        f"'wizard' not found in CLI help output.\nstdout={result.stdout!r}\nstderr={result.stderr!r}"
    )


# ---------------------------------------------------------------------------
# SM-03: CLI wizard loads a recipe file without crashing
# ---------------------------------------------------------------------------

def test_sm03_wizard_loads_recipe_file(tmp_path):
    """SM-03: macroforecast wizard --no-browser with a recipe file doesn't crash on startup."""
    yaml_content = "0_meta:\n  fixed_axes:\n    failure_policy: fail_fast\n"
    recipe_file = tmp_path / "minimal.yaml"
    recipe_file.write_text(yaml_content, encoding="utf-8")

    port = 8800

    if MACROFORECAST_CLI is not None:
        cmd = [MACROFORECAST_CLI, "wizard", "--no-browser", f"--port={port}", str(recipe_file)]
    else:
        cmd = [sys.executable, "-m", "macroforecast", "wizard", "--no-browser", f"--port={port}", str(recipe_file)]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        # Give it 3 seconds to start (or crash)
        time.sleep(3)
        poll = proc.poll()

        if poll is not None and poll != 0:
            stdout, stderr = proc.communicate(timeout=2)
            # Check that failure is NOT due to FileNotFoundError or ImportError
            combined = stdout + stderr
            assert "FileNotFoundError" not in combined, (
                f"Process crashed with FileNotFoundError:\n{combined}"
            )
            assert "ImportError" not in combined, (
                f"Process crashed with ImportError:\n{combined}"
            )
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
