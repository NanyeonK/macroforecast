from __future__ import annotations


class RawDataError(Exception):
    """Base exception for raw-data operations."""


class RawVersionFormatError(RawDataError):
    """Raised when a vintage string has invalid format."""


class RawVersionUnavailableError(RawDataError):
    """Raised when a requested version is unavailable."""


class RawDownloadError(RawDataError):
    """Raised when a raw file cannot be fetched."""


class RawParseError(RawDataError):
    """Raised when a raw file cannot be parsed."""


class RawManifestError(RawDataError):
    """Raised when manifest persistence fails."""
