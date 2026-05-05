# Docs Conventions

Rules for writing macroforecast documentation. Enforced repository-wide; new docs that don't follow these rules should be rejected at review.

The goal: a package that gives researchers **optional** choices with **sensible defaults**. Most users stay on defaults and don't need to read a line of docs beyond the quickstart. Docs exist for the moments when a researcher *wants* to deviate, and at that moment we owe them a per-choice explanation that is faithful to the actual runtime.

## R1 — Per-axis page structure

Every axis lives on exactly one page. The page heading is `Name (section.number)`, e.g. `Task & Target (1.2)`. Within the page, each axis gets the following blocks in this fixed order:

```
## N.n `axis_name`

**선택 질문** (selection question): [one sentence — what research question does this axis let the user answer?]

**기본값**: `default_value` — [one line: which research conventions sit at this default; "대부분의 연구자는 이대로 두면 됩니다" phrasing is fine]

### Value catalog

| Value | Status | 언제 쓰는가 | 검증 |
|---|---|---|---|
| `default_value` | operational (default) | base-rate research use | `manifest.json["data_task_spec"]["axis_name"]` |
| `another_value` | operational | when-to-pick sentence | specific dispatch site / manifest key / predictions column |
| ... | ... | ... | ... |

### Compatibility guards (only if the axis has compile-time guards)

- Guard sentence: "axis=X requires feature_builder∈{…} — else blocked_by_incompatibility."

### Functions & features

- Runtime dispatch function + module path.
- leaf_config input channels (when a non-default value reads a user field).
- Manifest fields that record the resolved choice.

### Dropped values (only if values were dropped)

- `value`: one-line reason, link to PR.

### Recipe usage

```yaml
# 1 example per non-default combination worth demonstrating.
```
```

The `언제 쓰는가` column is non-negotiable — readers must be able to scan values and know whether a given choice applies to their study.

The `검증` column is non-negotiable — every value must tell the reader where to look at runtime to confirm the choice actually took effect. Typical entries: `manifest.data_task_spec.<key>`, `predictions.csv::<column>`, `blocked_reasons[*]`.

## R2 — Single source of truth

Each axis has exactly one deep-dive location. Cross-references are links, never re-explanations. Top-level index and overview pages carry **one-line summaries + a link** — no duplicated paragraphs.

Apply the same rule to leaf_config input fields: if `release_lag_per_series` is documented on the `release_lag_rule` page, every other page that mentions it just links there.

## R3 — Default-first writing

Open each axis with the default. Frame the user choice as "leave this alone unless you want X." The catalog table's first row is always the default, with its `언제 쓰는가` cell explicitly calling that out: "Default. Use this unless …".

## R4 — Docs-code mapping accuracy

The Functions & features block cites the exact module path + function name that implements each value. Anyone reading the docs must be able to open that file and verify the dispatch with one grep. If docs reference `_apply_foo(rule, spec)` the function must exist there with that signature. When the code changes, the docs change in the same PR.

When writing docs you may find code that is complex or inconsistent (long dispatch chains, ad-hoc defaults in multiple places, stale fallbacks). Flag those for simplification rather than papering over them in prose.

## R5 — Completed-stage-only surface

Docs only expose stages whose per-axis walk is finished. Stages in progress live in plan files under `plans/` and in source comments, not in user-facing docs. Layer 0 (Design) and Layer 1 (Data) are complete as of v0.9.4. Layer 2+ axes are documented only via the in-progress plan files, not in `for_researchers/` or `for_recipe_authors/`.

If a completed axis has values scheduled for a future runtime, mark them `registry_only` with a one-line v1.x commitment — never operational until the runtime lands.

## Style notes

- No section-sign symbol. Write `Task & Target (1.2)` or just `1.2`, not `1.2`. Applies to docs, plans, ledger, source comments, test names — every file in the repo.
- Headings are sentence case for blocks (`### Value catalog`), title case for page titles.
- Tables beat prose for catalogs. Prose beats tables for reasoning.
- Code examples are minimal YAML with comments — never a full recipe unless the example demonstrates interactions across multiple axes.
- Every non-default recipe example must mention in a comment **why** the non-default choice is picked.

## Template

The current canonical example is `docs/for_recipe_authors/data/source.md`. New axis pages should mirror its structure. Deviations need a one-paragraph rationale.
