# `reproducibility_mode`

[Back to L0](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``reproducibility_mode`` on sub-layer ``l0_a`` (layer ``l0``).

## Sub-layer

**l0_a**

## Axis metadata

- Default: `'seeded_reproducible'`
- Sweepable: False
- Status: operational
- Leaf-config keys: `random_seed`

## Operational status summary

- Operational: 2 option(s)
- Future: 0 option(s)

## Options

### `seeded_reproducible`  --  operational

Fix a deterministic seed and propagate it through every RNG.

The cell-loop reads ``leaf_config.random_seed`` (default ``0``) and applies it to ``random.seed``, ``numpy.random.seed``, ``torch.manual_seed`` (when torch is installed), and the ``PYTHONHASHSEED`` environment variable for the current process. Each L4 estimator inherits the seed via its ``params.random_state`` key (issue #215); per-fit-node ``random_state`` overrides the L0 seed when present.

This is the only mode under which ``macroforecast.replicate(manifest)`` can verify bit-exact sink hashes. Use it for every study you intend to publish, share, or compare against later.

**When to use**

Default. Required for any study where bit-exact replication matters (papers, internal benchmarks, regression tests, comparisons across package versions).

**When NOT to use**

Stochastic exploration where the explicit goal is to characterise the variability of a procedure across seeds. Pick ``exploratory`` and re-run; the manifest will record that the seed was unfixed.

**References**

* Stodden, McNutt, Bailey et al. (2016) 'Enhancing reproducibility for computational methods', Science 354(6317). (doi:10.1126/science.aah6168)
* macroforecast PR #215 -- L0 random_seed propagation into L4 random_state.

**Related options**: [`exploratory`](#exploratory)

**Examples**

*Reproducible study (default)*

```yaml
0_meta:
  fixed_axes:
    reproducibility_mode: seeded_reproducible
  leaf_config:
    random_seed: 42

```

_Last reviewed 2026-05-04 by macroforecast author._

### `exploratory`  --  operational

Do not fix stochastic seeds; each run draws fresh randomness.

Skips the global RNG seeding that ``seeded_reproducible`` performs. Each cell pulls its own randomness from the OS entropy pool; downstream estimators that take an explicit ``random_state`` still use whatever the recipe sets per node, but the L0 default of ``0`` is *not* propagated.

``replicate()`` cannot guarantee bit-exact sink hashes under this mode -- the recipe still re-runs and produces structurally identical manifests, but the numeric forecasts will differ run-over-run.

**When to use**

Sensitivity studies where you want to measure how much variability the random components introduce. Wrap the run in a sweep over several executions and compare the spread.

**When NOT to use**

Anything you intend to publish or share. The reviewer will not be able to reproduce your manifest.

**References**

* macroforecast design Part 1, L0 §A: 'exploratory mode trades reproducibility for unbiased sampling of stochastic variability.'

**Related options**: [`seeded_reproducible`](#seeded-reproducible)

**Examples**

*Sensitivity sweep over fresh seeds*

```yaml
0_meta:
  fixed_axes:
    reproducibility_mode: exploratory

```

_Last reviewed 2026-05-04 by macroforecast author._
