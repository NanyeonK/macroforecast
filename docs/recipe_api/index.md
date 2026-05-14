# Recipe API

The Recipe API is the YAML interface for full forecasting studies. Use it when the high-level Simple API is too small: multi-layer DAGs, explicit data policy, model comparisons, statistical tests, interpretation, and output provenance.

A recipe is a top-level mapping whose layer keys are:

`0_meta`, `1_data`, `2_preprocessing`, `3_feature_engineering`, `4_forecasting_model`, `5_evaluation`, `6_statistical_tests`, `7_interpretation`, `8_output`.

Diagnostic layers may also appear:

`1_5_data_summary`, `2_5_pre_post_preprocessing`, `3_5_feature_diagnostics`, `4_5_generator_diagnostics`.

Start here:

- [Recipe Gallery](gallery.md): runnable examples.
- [Recipe Layer Contract](layer_contract.md): layer keys, DAG node shape, and one complete recipe.
- [Data Layer](data.md): source, target, horizon, and FRED-SD geography choices.
- [Data Policies](data_policies.md): missingness, outliers, release lags, and same-period predictors.
- [Recipe Defaults](defaults.md): package-level defaults from `macroforecast.defaults`.
- [Runtime Support Matrix](runtime_support.md): what executes today.
- [Understanding Recipe Output](output.md): artifact directory and manifest layout.
- [FRED Datasets in Recipes](fred_datasets.md): FRED-MD, FRED-QD, and FRED-SD reference status.
