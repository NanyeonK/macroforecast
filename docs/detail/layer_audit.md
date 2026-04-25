# Layer Audit Checklist

Use this checklist when reviewing each internal recipe layer against the `Experiment` philosophy.

For each layer, answer:

1. What does this layer mean to a forecasting researcher?
2. Which choices need beginner-facing aliases?
3. Which choices should be defaulted by `macrocast-default-v1`?
4. Which choices are safe to sweep from the simple API?
5. Which choices are advanced-only?
6. Which choices are custom extension points?
7. Which values are truly operational?
8. Which values require optional dependencies?
9. Which values are parsed but `not_supported` because no runner/runtime contract exists?
10. How is this layer recorded in the manifest?

Layer order:

```text
0_meta
1_data_task
2_preprocessing
3_training
4_evaluation
5_output_provenance
6_stat_tests
7_importance
```

Audit output should classify each axis as:

- simple default
- simple alias
- simple sweep
- custom extension
- advanced documented
- internal only
- planned or deprecated
