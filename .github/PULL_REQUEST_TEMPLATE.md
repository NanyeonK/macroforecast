## Phase / Issue

Closes #<issue number>
Phase: `phase-XX` (v0.X target)

## Summary

<!-- 1-2 sentences — what this PR does and why -->

## Plan reference

`plans/phases/phase_XX_<name>.md` Sub-Tasks row XX.Y

## Changes

- [ ] **Code:** files touched
- [ ] **Tests:** tests added / modified
- [ ] **Docs:** docs pages updated
- [ ] **Registry:** axes promoted (reference `plans/coverage_ledger.md`)
- [ ] **Infra:** `plans/infra/*.md` updated if cross-cutting contract changed

## Acceptance criteria

<!-- Copy from linked issue -->
- [ ] ...
- [ ] ...

## Breaking change?

- [ ] No
- [ ] Yes — migration documented in `plans/infra/` or `CHANGELOG.md`, and `DeprecationWarning` added per ADR-006

## Test evidence

<details>
<summary>`pytest tests/ -x` output</summary>

```
<paste relevant pytest output here>
```

</details>

## Docs evidence (if docs changed)

<details>
<summary>`sphinx-build -W` output</summary>

```
<paste build output>
```

</details>

## Reviewer checklist

- [ ] Implementation matches plan §4 API spec
- [ ] Tests cover happy path + error cases
- [ ] No `random_state=42` hardcoded (use `resolve_seed()`)
- [ ] Coverage Ledger updated if axis status changed
