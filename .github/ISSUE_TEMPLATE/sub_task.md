---
name: Sub-task
about: Specific deliverable within a phase
title: "[PHASE-XX.Y] <deliverable>"
labels: ["sub-task", "phase-XX", "priority-P?"]
---

## Parent epic

#<phase epic issue number>

## Plan reference

`plans/phases/phase_XX_<name>.md` — Sub-Tasks row XX.Y

## Description

<!-- Copy from plan Sub-Tasks table row -->

## Acceptance criteria

<!-- Derived from plan §7 Acceptance Gate items relevant to this sub-task -->
- [ ] ...
- [ ] ...

## Files to touch

<!-- From plan §5 File Layout -->
- `path/to/new_file.py` (new)
- `path/to/existing_file.py` (modify)

## Test requirements

<!-- From plan §6 Test Strategy -->
- `tests/test_<name>.py` — what it validates

## Dependencies

- Blocked by: #<prerequisite sub-task if any>
- Infra files used: `plans/infra/<name>.md`
- ADRs referenced: `ADR-XXX`

## Definition of Done

- [ ] Implementation matches plan §4 API spec
- [ ] New tests green (`pytest tests/<test_file> -x`)
- [ ] Existing test suite (291+ regression) green
- [ ] Docs page updated if user-facing
- [ ] PR linked to this issue
- [ ] Code review approved
