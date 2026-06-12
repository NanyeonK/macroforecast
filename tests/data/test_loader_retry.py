"""Tests for transient-failure retry behavior in the official-source fetch path.

These tests are fully offline: they monkeypatch the ``urlopen`` symbol used by
``macroforecast.data.loaders`` and the ``time.sleep`` used for backoff so they
never touch the network and run instantly.
"""

from __future__ import annotations

from urllib.error import HTTPError

import pytest

from macroforecast.data import loaders
from macroforecast.data.errors import RawNetworkError


class _FakeResponse:
    """Minimal stand-in for a urlopen response usable as a context manager."""

    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False


def test_read_official_url_retries_transient_connection_reset(monkeypatch) -> None:
    """A transient ConnectionResetError on the first calls is retried, then succeeds."""
    payload = b"sasdate,INDPRO\nTransform:,5\n1/1/2000,100\n"
    calls = {"n": 0}

    def flaky_urlopen(_request, **_kwargs):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise ConnectionResetError("connection reset by peer")
        return _FakeResponse(payload)

    monkeypatch.setattr(loaders, "urlopen", flaky_urlopen)
    monkeypatch.setattr(loaders.time, "sleep", lambda _seconds: None)

    result = loaders._read_official_url("https://example.test/current.csv")

    assert result == payload
    assert calls["n"] == 3  # retried twice before succeeding


def test_read_official_url_does_not_retry_http_404(monkeypatch) -> None:
    """A deterministic HTTP 404 is not retried and propagates immediately."""
    calls = {"n": 0}
    sleeps = {"n": 0}

    def not_found_urlopen(_request, **_kwargs):
        calls["n"] += 1
        raise HTTPError("https://example.test/missing.csv", 404, "Not Found", {}, None)

    monkeypatch.setattr(loaders, "urlopen", not_found_urlopen)
    monkeypatch.setattr(loaders.time, "sleep", lambda _seconds: sleeps.__setitem__("n", sleeps["n"] + 1))

    with pytest.raises(HTTPError) as excinfo:
        loaders._read_official_url("https://example.test/missing.csv")

    assert excinfo.value.code == 404
    assert calls["n"] == 1  # single attempt, no retry
    assert sleeps["n"] == 0  # never backed off


def test_read_official_url_exhausts_attempts_on_persistent_failure(monkeypatch) -> None:
    """A persistent retryable error exhausts attempts and raises a clear RawNetworkError."""
    calls = {"n": 0}
    sleeps = {"n": 0}

    def always_reset_urlopen(_request, **_kwargs):
        calls["n"] += 1
        raise ConnectionResetError("connection reset by peer")

    monkeypatch.setattr(loaders, "urlopen", always_reset_urlopen)
    monkeypatch.setattr(loaders.time, "sleep", lambda _seconds: sleeps.__setitem__("n", sleeps["n"] + 1))

    with pytest.raises(RawNetworkError) as excinfo:
        loaders._read_official_url("https://example.test/current.csv")

    message = str(excinfo.value)
    assert str(loaders._DOWNLOAD_RETRY_ATTEMPTS) in message
    assert "transient" in message.lower()
    assert isinstance(excinfo.value.__cause__, ConnectionResetError)
    assert calls["n"] == loaders._DOWNLOAD_RETRY_ATTEMPTS
    # One fewer sleep than attempts (no backoff after the final failure).
    assert sleeps["n"] == loaders._DOWNLOAD_RETRY_ATTEMPTS - 1
