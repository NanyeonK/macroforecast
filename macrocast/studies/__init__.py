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
from .multi_target import (
    SEPARATE_RUNS_MANIFEST_SCHEMA_VERSION,
    SeparateRunsResult,
    execute_separate_runs,
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
    "SEPARATE_RUNS_MANIFEST_SCHEMA_VERSION",
    "STUDY_MANIFEST_SCHEMA_VERSION",
    "SeparateRunsResult",
    "VariantManifestEntry",
    "build_study_manifest",
    "execute_ablation",
    "execute_replication",
    "execute_separate_runs",
    "validate_study_manifest",
]
