"""Study-level orchestration for horse-race sweeps (Phase 1)."""

from .ablation import (
    ABLATION_REPORT_SCHEMA_VERSION,
    AblationSpec,
    execute_ablation,
)
from .manifest import (
    STUDY_MANIFEST_SCHEMA_VERSION,
    VariantManifestEntry,
    build_study_manifest,
    validate_study_manifest,
)
from .replication import (
    REPLICATION_DIFF_SCHEMA_VERSION,
    ReplicationResult,
    execute_replication,
)

__all__ = [
    "ABLATION_REPORT_SCHEMA_VERSION",
    "AblationSpec",
    "REPLICATION_DIFF_SCHEMA_VERSION",
    "ReplicationResult",
    "STUDY_MANIFEST_SCHEMA_VERSION",
    "VariantManifestEntry",
    "build_study_manifest",
    "execute_ablation",
    "execute_replication",
    "validate_study_manifest",
]
