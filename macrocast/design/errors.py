from __future__ import annotations


class DesignError(Exception):
    """Base exception for stage0 operations."""


class DesignNormalizationError(DesignError):
    """Raised when stage0 inputs cannot be normalized."""


class DesignValidationError(DesignError):
    """Raised when stage0 inputs violate required structure."""


class DesignCompletenessError(DesignError):
    """Raised when a stage0 frame is structurally incomplete for execution."""


class DesignRoutingError(DesignError):
    """Raised when no valid route owner can be derived from a stage0 frame."""
