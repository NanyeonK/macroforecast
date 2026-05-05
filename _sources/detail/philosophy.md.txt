# Philosophy

macroforecast exists to help macroeconomic forecasting researchers compare methods under a common, auditable framework.

Core philosophy:

- run a reasonable default forecast without configuring every detail
- sweep only the choices the researcher cares about
- keep information sets, splits, benchmarks, and evaluation rules fixed unless explicitly changed
- let custom preprocessing, models, benchmarks, and metrics run beside built-ins
- record every resolved choice in reproducible artifacts

The package can expose a simple API because the internal system keeps a detailed recipe and registry model underneath it.

Vocabulary:

- `Experiment`: user-facing workspace
- `Recipe`: one fully specified executable path
- `SweepPlan`: materialized set of recipe variants
- `Run`: one concrete execution
- `Manifest`: resolved choices and provenance
- `Artifact`: saved output from a run or sweep
