class MetaRegistryError(ValueError):
    """Base exception for meta registry issues."""


class AxisClassificationError(MetaRegistryError):
    """Raised when an axis appears in multiple incompatible classes."""


class BenchmarkRegistryError(MetaRegistryError):
    """Raised when benchmark registry entries are invalid."""


class PresetResolutionError(MetaRegistryError):
    """Raised when a preset cannot be resolved or violates policy."""


class IllegalOverrideError(MetaRegistryError):
    """Raised when an override violates axis-class policy."""


class NamingPolicyError(MetaRegistryError):
    """Raised when naming policy is malformed."""
