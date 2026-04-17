# Macrocast Stage 0 — Grammar First

Status: reboot-stage architecture note
Date: 2026-04-14
Priority: highest

## 0.0 Purpose of Stage 0

The first sentence to lock is:

> Stage 0 is not the stage where registry content is filled in.
> Stage 0 is the stage where the grammar and contract that registries must follow are fixed.

This is the architectural purpose of Stage 0.

If this sentence is not locked, every later layer will start growing YAML in an undisciplined way.
At that point the package stops being a forecasting package with a clear execution grammar and turns into a graveyard of configuration files.

## Why this must come first

The package does not need a large option universe first.
It needs a disciplined language first.

What kills package coherence is not "too few options".
What kills package coherence is:
- registries inventing fields independently
- layers adding ad hoc YAML keys to solve local problems
- route-specific exceptions becoming top-level schema structure
- package logic being reverse-engineered from accumulated config content

So Stage 0 must fix how choices move inside the package before it fixes what the choices are.

## Stage 0 owns grammar, not inventory

Stage 0 must decide:
- what counts as a fixed choice
- what counts as a varying choice
- what counts as a route selector
- what counts as a compatibility mirror only
- what objects later layers are allowed to emit
- what fields registries may define
- what fields registries may not define
- what later layers must inherit rather than redefine

Stage 0 must not begin by asking:
- which dataset values exist
- which model values exist
- which benchmark values exist
- which paper-specific options exist

Those are registry-content questions.
They come after the grammar is locked.

## The key distinction

There are two very different problems:

1. Content problem
- which values are available in a registry
- example: which dataset ids, which benchmark ids, which model ids

2. Grammar problem
- how those values are allowed to participate in package execution
- example: fixed vs varying, single-path vs bundle, own-analysis default vs explicit override, route ownership, inheritance rules

Stage 0 is about problem (2), not problem (1).

## Architectural consequence

Every later registry should be readable as content constrained by a prior grammar.
Never the reverse.

Bad direction:
- registry content appears first
- grammar is inferred later from existing YAML shape

Correct direction:
- Stage 0 grammar is fixed first
- registries are authored to fit that grammar

## What Stage 0 should lock before any registry expansion

Minimum Stage 0 outputs should answer these questions:

1. What is the package's execution grammar?
- one-path run
- controlled variation
- bundle/orchestrated multi-run
- explicit override path

2. Which choices are structural?
- route ownership
- fixed design spine
- varying design grammar
- comparison grammar
- execution posture

3. Which fields are semantic categories rather than registry payload?
- replication input
- dataset implication
- derived design shape
- fixed design
- varying design
- comparison contract
- execution posture

4. What is allowed to appear in registries later?
- only content that plugs into these grammar slots
- no free-floating YAML branches outside the Stage 0 contract

## Practical rule for rebuild

Before creating or restoring any large registry tree:
- define the Stage 0 grammar objects
- define their contracts
- define allowed downstream extension points
- define forbidden ad hoc fields

Only then populate registries.

## Package-level warning

If Stage 0 is skipped or weakened, the rebuild will drift into one of these failure modes:
- registry sprawl
- paper-specific logic leaking into package core
- route logic encoded as content ids
- duplicated semantics across layers
- execution behavior determined by YAML accidents rather than package contracts

## Immediate implication for current reboot

The next big-picture planning step is not registry filling.
It is to write the Stage 0 grammar contract for the rebuilt package.

That contract must define:
- what Stage 0 objects exist
- what they mean
- how later layers consume them
- what later registries are allowed to express

## One-line doctrine

Stage 0 fixes the language of the package before the package starts speaking in registry content.
