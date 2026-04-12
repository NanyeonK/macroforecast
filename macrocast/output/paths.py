from __future__ import annotations

from pathlib import Path
from typing import Any


def safe_slug(value: str) -> str:
    return value.replace('/', '_').replace(' ', '_').replace(':', '_')


def build_run_tree_path(*, recipe_id: str | None = None, taxonomy_path: dict[str, Any] | None = None, run_id: str) -> Path:
    if recipe_id:
        return Path('runs') / 'recipes' / safe_slug(recipe_id) / safe_slug(run_id)
    if taxonomy_path:
        ordered = [f"{k}={safe_slug(str(v))}" for k, v in taxonomy_path.items()]
        return Path('runs') / 'paths' / Path(*ordered) / safe_slug(run_id)
    return Path('runs') / 'ad_hoc' / safe_slug(run_id)


def ensure_output_dirs(base_dir: str | Path, run_id: str, *, recipe_id: str | None = None, taxonomy_path: dict[str, Any] | None = None) -> dict[str, Path]:
    base_dir = Path(base_dir).expanduser()
    run_tree = build_run_tree_path(recipe_id=recipe_id, taxonomy_path=taxonomy_path, run_id=run_id)
    base = base_dir / run_tree
    paths = {
        'root': base,
        'forecasts': base / 'forecasts',
        'evaluation': base / 'evaluation',
        'tests': base / 'tests',
        'interpretation': base / 'interpretation',
        'manifests': base / 'manifests',
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths
