from __future__ import annotations


class PreprocessContractError(Exception):
    """Base exception for preprocessing contract operations."""


class PreprocessValidationError(PreprocessContractError):
    """Raised when preprocessing semantics are invalid or governance rules are violated."""
