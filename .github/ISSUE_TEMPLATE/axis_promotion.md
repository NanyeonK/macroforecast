---
name: Axis Promotion
about: Promote a registry_only or planned value to operational
title: "Promote <axis>=<value> to operational"
labels: ["axis-promotion", "phase-XX"]
---

## Current status

From Coverage Ledger (`plans/coverage_ledger.md`):
- Layer: X
- Axis: `<axis_name>`
- Value: `<value>`
- Current: `registry_only` | `planned` | `future` | `absent`

## Target

- Target version: v0.X or v1.1 or v2
- Target phase: phase-XX
- Rationale: <from Coverage Ledger row>

## Implementation requirements

- [ ] Registry entry: `macroforecast/registry/<layer>/<axis>.py` (exists / needs creation)
- [ ] Runtime adapter: `<path>.py` (wiring in execute_recipe or module)
- [ ] Test: `tests/test_<axis>_<value>.py`
- [ ] Docs: (user-facing? docs/user_guide/<axis>.md)

## Coverage Ledger resolution

After merge, update `plans/coverage_ledger.md`:
- Layer X axis Y value Z → status `operational`

## Related sub-task

- Part of: #<phase sub-task issue>
