---
name: Phase Epic
about: Parent issue for a full phase of the roadmap
title: "[PHASE-XX] <phase name>"
labels: ["epic", "phase-XX", "priority-P?"]
---

## Phase plan

Full plan: [`plans/phases/phase_XX_<name>.md`](../../plans/phases/)

## Goal

<!-- Copy from plan §1 -->

## Sub-tasks

<!-- Auto-populated on phase kickoff. One checkbox per sub-task issue. -->
- [ ] #TBD Sub-task XX.1
- [ ] #TBD Sub-task XX.2

## Gate criteria

<!-- Copy from plan §7 Acceptance Gate -->
- [ ] All P0/P1 sub-tasks closed
- [ ] Full test suite green
- [ ] RTD docs build green
- [ ] CHANGELOG updated

## Version target

v0.X

## Blocked by

- #<previous phase epic>

## Blocks

- #<next phase epic>

## Cross-references

- Infra files: `plans/infra/*.md`
- ADRs: `plans/infra/adr/*.md`
- Coverage Ledger rows resolved: Layer X axis Y
