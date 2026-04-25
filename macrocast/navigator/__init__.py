"""Decision-tree navigation, compatibility resolution, and replication recipes."""

from .core import (
    NAVIGATOR_SCHEMA_VERSION,
    build_navigation_view,
    canonical_path,
    compatibility_view,
    load_recipe,
)
from .replications import (
    REPLICATION_LIBRARY_VERSION,
    get_replication_entry,
    list_replication_entries,
    replication_recipe_yaml,
    write_replication_recipe,
)
from .ui_data import (
    NAVIGATOR_UI_DATA_SCHEMA_VERSION,
    axis_catalog,
    navigator_ui_data,
    write_navigator_ui_data,
)

__all__ = [
    "NAVIGATOR_SCHEMA_VERSION",
    "NAVIGATOR_UI_DATA_SCHEMA_VERSION",
    "REPLICATION_LIBRARY_VERSION",
    "axis_catalog",
    "build_navigation_view",
    "canonical_path",
    "compatibility_view",
    "get_replication_entry",
    "list_replication_entries",
    "load_recipe",
    "navigator_ui_data",
    "replication_recipe_yaml",
    "write_navigator_ui_data",
    "write_replication_recipe",
]
