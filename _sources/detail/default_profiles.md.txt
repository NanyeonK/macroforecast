# Default Profiles

Defaults are part of the public contract. They must be named, versioned, and written to the manifest.

Draft profile:

```text
macrocast-default-v1
```

Current MVP profile:

| Choice | Value |
|--------|-------|
| `information_set_type` | `final_revised_data` |
| `target_structure` | `single_target` |
| `framework` | `expanding` |
| `benchmark_family` | `zero_change` |
| `feature_builder` | `autoreg_lagged_target` |
| `model_family` | `ar` |
| `primary_metric` | `msfe` |
| `stat_test` | `none` |
| `importance_method` | `none` |
| preprocessing | dataset t-code transforms only, no extra preprocessing |
| sample period | explicit `start` and `end` required |
| data vintage | current vintage when omitted |
| `reproducibility_mode` | `seeded_reproducible` |
| `failure_policy` | `fail_fast` |
| `compute_mode` | `serial` |
| `random_seed` | `42` |

This default still writes legacy-compatible
`feature_builder=autoreg_lagged_target`; the compiler also records canonical
Layer 2 block/runtime provenance. New full recipes may use explicit feature
blocks directly.

The profile also keeps the current Layer 3 compatibility names:
`model_family=ar` means candidate forecast generator family `ar`, and
`benchmark_family=zero_change` means the `zero_change` generator is assigned
the benchmark/baseline role.

Frequency resolution:

- `fred_md` means monthly.
- `fred_qd` means quarterly.
- `fred_sd` alone requires `frequency`.
- `fred_md+fred_sd` means monthly.
- `fred_qd+fred_sd` means quarterly.
- monthly data converted to quarterly uses a 3-month average with a warning.
- quarterly data converted to monthly uses linear interpolation with a warning.

The MVP profile is intentionally narrow. It should expand only after the `Experiment` API and layer audit are stable.

The final profile should also specify:

- default information set
- default sample split
- default benchmark
- default model
- default preprocessing
- default metric
- default statistical test behavior
- default importance behavior
- default artifact policy
- default reproducibility mode

Design requirements:

- no hidden defaults in artifacts
- old default profiles remain reproducible
- changing a default profile is a versioned change
- users can inspect and override defaults
