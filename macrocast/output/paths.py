from __future__ import annotations

from pathlib import Path


def ensure_output_dirs(base_dir: str | Path, run_id: str) -> dict[str, Path]:
    base = Path(base_dir).expanduser() / run_id
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
