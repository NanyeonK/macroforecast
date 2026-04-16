# Contributing

## Repository structure

```
macroforecast/
  macrocast/         — package source (~12,000 LOC)
  tests/             — test suite (~4,300 LOC, 291 tests)
  docs/              — public documentation
  plans/             — internal planning (not public)
  examples/          — example recipes (YAML)
  CLAUDE.md          — project guide for AI-assisted development
  README.md          — package overview
```

## Running tests

```bash
cd ~/project/macroforecast
python3 -m pytest tests/ -x -q           # full suite (~3 min)
python3 -m pytest tests/ -x -q -k "not sweep"  # fast subset (~30s)
python3 -m pytest tests/test_full_operational_sweep.py -v  # 96 sweep tests (~5 min)
```

## Code style

- Python 3.10+ with `from __future__ import annotations`
- `@dataclass(frozen=True)` for all data objects
- Explicit error classes per module (not generic ValueError)
- Every new model/test/importance method needs both code and registry entry
- Train-only preprocessing fit is mandatory (no data leakage)

## Adding new functionality

- [Adding Models](adding_models.md) — new model families
- [Adding Statistical Tests](adding_stat_tests.md) — new forecast comparison tests

## Documentation rule

Every code surface that becomes part of the package must be documented in `docs/` with:
- User Guide entry (when to use, how it works)
- API Reference entry (function signatures)
- Mathematical Background if applicable (formulas, references)
