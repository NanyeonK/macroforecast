# macroforecast wizard (Solara web UI)

`macroforecast wizard` — Solara-based 3-pane web UI for YAML recipe
authoring, replacing the deprecated stdlib CLI wizard
(`macroforecast/scaffold/wizard.py`).

## Quick start

    pip install 'macroforecast[wizard]'
    macroforecast wizard --port 8765

Open `http://localhost:8765` in a browser.

## Layout

- **Left rail**: layer navigation, color coded by `STAGE_BY_LAYER` stage
  (see `macroforecast/core/stages.py`)
- **Center workspace**: layer form (L0 currently; L1/L2/L5/L6 in P2b/c)
- **Right pane**: live YAML preview

## Phase status (as of 2026-05-13)

| Phase | Scope | Status |
|---|---|---|
| P2a | Shell + L0 form + YAML preview + CLI entry | DONE (`c0addcf0`) |
| P2b | L1/L2 form + bidirectional YAML edit + Starlette upper bound | planned |
| P2c | L5/L6 form + Mosaic Cube overview polish + LR-04 fix | planned |
| P3  | L3/L4/L7 DAG editor (React Flow embed) | planned |

See `ARCHITECTURE.md` "Phase Wizard P2a" section for full design + Mermaid.

## Test

    pytest tests/wizard/ -v -m "not slow"

## CLI flags

- `--port <n>` (default 8765)
- `--no-browser` (don't auto-open browser)
- `[recipe.yaml]` positional — load existing recipe
