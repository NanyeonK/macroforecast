# Replications

This track shows end-to-end examples of running a research-grade
macroforecast study from raw FRED data through bit-exact replicable
manifest. Three layers:

1. **Bundled example walkthroughs** -- detailed walk through one of
   `examples/recipes/*.yaml`, explaining every layer choice and the
   resulting artifact tree.
2. **Bundled paper baselines** -- a published paper's ridge / model
   horse-race ported to macroforecast schema. Recipe ships in the
   repo, runs end-to-end, replicates bit-exact. The user swaps the
   bundled smoke panel for real FRED data when reproducing the
   paper's figures.
3. **Research replications** -- four studies the project maintainer
   conducted with macroforecast. Each replication includes the exact
   recipe YAML, command line, expected artifacts, and pointers back to
   the published paper / preprint.

Use these as a template when you want to write your own
publication-grade replication study.

## Walkthroughs

- [Example walkthrough -- minimal ridge](example_walkthrough.md)

## Bundled paper baselines

- [Goulet-Coulombe (2021) -- The Macroeconomy as a Random Forest](goulet_coulombe_2021.md)

## Research replications

Coming in v0.9.1 (additional 4+ replication walkthroughs covering published
macro-forecasting papers).

```{toctree}
:hidden:
:maxdepth: 1

example_walkthrough
goulet_coulombe_2021
```
