---
orphan: true
---

# Docs Structure Plan 2026-05-27

## Goal

Make the docs match the semantic package structure now used by the public API:
`meta`, `data`, `preprocessing`, `features`, `models`, `evaluation`,
`stat_tests`, `interpretation`, `output`, and `diagnostics`.

The docs should let users start from callable Python APIs or YAML recipe blocks
without learning internal numbering first. Generated numbered pages remain
available for exhaustive lookup, but they should not dominate the main
navigation.

## Target IA

```text
docs/
├── index.md
├── tutorial/
├── how_to/
├── explanation/
│   └── architecture/
├── reference/
│   ├── index.md
│   ├── meta.md
│   ├── data.md
│   ├── preprocessing.md
│   ├── features.md
│   ├── models.md
│   ├── evaluation.md
│   ├── stat_tests.md
│   ├── interpretation.md
│   ├── output.md
│   ├── diagnostics.md
│   ├── generated/
│   └── standalone_functions/
├── recipe-snippets/
└── _audit/
```

## Current Step Completed

- Added semantic `reference/meta.md`.
- Rewrote semantic `reference/data.md`.
- Moved generated option encyclopedia under `reference/generated/`.
- Updated `reference/index.md` so semantic pages are primary and generated
  pages are secondary.

## Next Steps

1. Build the remaining semantic reference pages in package order:
   `preprocessing`, `features`, `models`, `evaluation`, `stat_tests`,
   `interpretation`, `output`, `diagnostics`.
2. Rename or rewrite architecture pages from `layer0`, `layer1`, ... to
   semantic filenames once each package contract is stable.
3. Delete or regenerate `recipe-snippets/` after the recipe key contract is
   settled. Do not mechanically patch stale snippets one by one.
4. Rewrite tutorial and how-to YAML examples after the semantic recipe surface
   is stable.
5. Keep generated encyclopedia pages in `reference/generated/` and refresh
   docgen paths to emit there directly.

## Policy

- Main navigation should expose semantic package concepts first.
- Generated pages are exhaustive lookup, not onboarding.
- Old numbering language can remain inside generated pages until docgen is
  updated, but hand-written docs should prefer semantic names.
- Do not patch stale recipes piecemeal unless they are part of an active,
  verified tutorial or test.
