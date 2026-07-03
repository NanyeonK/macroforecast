# Glossary

[Back to User Guide](index.md)

The terms below name the core abstractions of `macroforecast`. Each definition
is short, and the concept and reference pages carry the full treatment. Entries
are sorted alphabetically.

```{glossary}
:sorted:

Arm
    A target-agnostic recipe of a preprocessing choice, a feature set, and a
    single model. Applied to one target and one horizon it forms one
    {term}`cell` and appears in the report as exactly one {term}`contender`. An
    arm carries one model, so comparing models means adding more arms. See
    [Models and Arms](concepts/models_and_arms.md).

benchmark
    The arm that every other arm is scored against. It is named in
    {term}`EvalSpec` and is usually an autoregression or a factor model.
    Relative accuracy and the comparison tests are all defined against it.

cell
    The execution unit of the pipeline, one {term}`Arm` applied to one target
    over the window for one horizon group. Each cell is run by a single `run()`
    call, so the total cell count is arms times targets times horizons.

Clark-West test
    A forecast-comparison test that adjusts Diebold-Mariano for the
    finite-sample bias that appears when a larger model nests a smaller
    benchmark. It is valid only when the benchmark is nested in the contender,
    declared with `nested_in_benchmark=True`. See
    [Evaluation](concepts/evaluation.md).

contender
    One entry in the accuracy and test tables. Every {term}`Arm` is one
    contender, and `arm.name` is its label.

cross-validation selection
    A hyperparameter strategy that scores each candidate on the validation
    splits and keeps the one with the lowest mean validation loss. Contrast with
    {term}`information-criterion selection`.

DataBundle
    The object returned by the `mf.data.load_*` loaders. It pairs the canonical
    {term}`panel` with its source metadata, including the official {term}`t-code`
    for each series.

DataSpec
    A panel plus the study choices attached by `mf.data.spec`, namely the
    target, the horizons, the active date range, and the predictor columns. See
    [Data](concepts/data.md).

Diebold-Mariano test
    A forecast-comparison test of whether two forecasts have equal expected
    accuracy. It is valid for any pair of forecasts, nested or not. Contrast with
    the {term}`Clark-West test`.

direct policy
    A multi-step strategy that fits one model per horizon using the value that
    many steps ahead as the target. It is the simplest construction and the most
    common. See [Running](concepts/running.md).

direct_average policy
    A variant of the {term}`direct policy` whose forecast object is the
    horizon-length average of the stationary transform rather than the
    single-period value. It is the standard convention for growth-rate series.

EM-factor imputation
    The default missing-value method for FRED-style panels. It fills gaps with an
    expectation-maximization algorithm built on common factors, refit on the rows
    available at each origin so no future data leaks in. See
    [Preprocessing](concepts/preprocessing.md).

embargo
    The number of origins held out between the training tail and the forecast
    origin. The default `embargo=0` follows the pseudo-out-of-sample convention,
    and `embargo=horizon-1` enforces a strict real-time gap.

estimation window
    The pre-test training sample and how it evolves across origins. An expanding
    window grows by one period at each origin, a rolling window keeps a fixed
    trailing size, and a fixed window stays anchored between set dates. It is one
    part of a {term}`WindowSpec`.

EvalSpec
    The evaluation and significance-testing configuration. It names the
    {term}`benchmark`, the accuracy metrics to compute, the tests to run, the
    grouping dimensions, and the Model Confidence Set settings. See
    [Evaluation](concepts/evaluation.md).

factor
    A latent component extracted from many predictors, usually by principal
    components, that summarizes their common movement. Factors are the basis of
    the F feature family and of factor models.

feature families
    The five recurring predictor designs. F is principal-component or sparse
    factors, X is raw lags of individual series, MARX is the moving-average lag
    cross that is the standard macro design, MAF is maximum-autocorrelation
    factors, and Level is untransformed level columns passed through unchanged.
    See [Features](concepts/features.md).

feature step
    One reusable construction operation placed in the `feature_steps` list of a
    {term}`FeatureSpec`, such as `marx_step`, `pca_step`, or `lag_step`. Stateful
    steps are refit inside each training window.

FeatureSpec
    A declaration of feature construction stored without running. The runner fits
    it inside each estimation window and applies it to the matching test rows, so
    operations such as PCA never see test data. Built with
    `mf.feature_engineering.feature_spec`.

forecast origin
    A test date at which a forecast is made from the data available up to that
    date. A run steps through a sequence of origins, mimicking real-time
    forecasting.

ForecastResult
    The output of `mf.forecasting.run` for one model. It holds the full forecast
    table with date, origin, horizon, model, prediction, and actual, and exposes
    `.to_frame()` and `.evaluate()`.

horizon
    The number of periods ahead a forecast targets. A study usually evaluates
    several horizons together, for example one, three, six, and twelve months.

information-criterion selection
    A hyperparameter strategy that picks the model minimizing an information
    criterion such as BIC or AIC on the estimation sample, without a separate
    validation holdout. Contrast with {term}`cross-validation selection`.

leak-free
    A property of the workflow whereby no observation dated at or after a
    {term}`forecast origin` enters the training data for that origin. Stateful
    steps are refit on the rows available at each origin. The opposite, using
    future information at decision time, is look-ahead.

Model Confidence Set
    The set of models that cannot be statistically separated from the best model
    at a chosen significance level. The pipeline reports membership for each
    target and horizon. See [Evaluation](concepts/evaluation.md).

model string
    The short name that selects a model family, such as `"ar"`, `"lasso"`, or
    `"random_forest"`. It is passed as `Arm(model=...)` and resolves against the
    model registry. See [Models and Features](model_overview.md).

ModelSpec
    A description of a single model, its canonical name, optional fixed
    parameters, and optional search space. It is needed only when fixed
    parameters or a custom search space are wanted, since most families are
    reachable by a {term}`model string`.

nesting
    The relationship where a benchmark is a special case of a larger contender.
    Declaring `nested_in_benchmark=True` on an arm licenses the
    {term}`Clark-West test` for it.

panel
    The canonical data contract, a `pandas.DataFrame` with a `DatetimeIndex`
    named date and one macro series per column, with dataset metadata stored
    alongside. See [Data](concepts/data.md).

path_average policy
    A multi-step strategy that fits one step-specific model per step of the
    horizon, each forecasting the one-period object at that step from information
    available at the origin, then averages the step forecasts. It is a direct
    multi-step construction, not an iterated one; iterating a single model forward
    is the separate recursive policy. Contrast with the {term}`direct policy`.

PipelineReport
    The object returned by `run_pipeline`. It carries `.accuracy` for relative
    accuracy by target and horizon, `.significance` for the comparison tests,
    `.mcs` for Model Confidence Set membership, and `.forecasts` for the full
    forecast frame.

PipelineSpec
    The validated, frozen configuration produced by `pipeline_spec`. It holds the
    resolved targets, the window, every arm, the evaluation spec, optional shared
    preprocessing, and parallelism settings, and is passed to `run_pipeline`.

POOS
    Pseudo-out-of-sample evaluation, the standard macro protocol in which a model
    is fit and forecast over a sequence of {term}`forecast origin` points on a
    fixed final-vintage dataset. With `embargo=0` the last training label may
    realize at or after the origin.

PreprocessSpec
    A declaration of preprocessing choices, namely the transform, outlier rule,
    imputation, and standardization, stored without running. The runner applies
    it per origin and refits stateful steps on the available rows so the path
    stays {term}`leak-free`. Built with `mf.preprocessing.preprocess_spec`.

PreprocessedData
    The output of `mf.preprocessing.reprocess`, a stationary and cleaned panel
    with its preprocessing metadata. It is the full-sample path used for
    exploration.

R-squared out-of-sample
    An accuracy metric, `r2_oos`, giving the share of benchmark forecast-error
    variance the contender removes. A positive value means the contender beats
    the benchmark.

relative MSE
    The ratio of contender mean squared error to {term}`benchmark` mean squared
    error. A value below one means the contender wins. It is the default
    `relative_mse` metric and the usual report quantity in the macro literature.

relative RMSE
    The square root of {term}`relative MSE`. Some studies report it, and it is
    not the same quantity as relative MSE, so the two should not be confused.

retrain and retune cadence
    How often the runner refits a model and reselects its hyperparameters.
    `retrain_every` controls refitting and `retune_every` controls reselection,
    both as a number of origins or a pandas offset, and `retune_on_retrain` ties
    reselection to refitting.

RMSE
    Root mean squared forecast error over the test origins, the base accuracy
    metric before any benchmark ratio is taken.

StagePolicy
    A declaration of where a stateful operation may fit its parameters. The
    `full_panel` scope fits on all data and introduces look-ahead, so it is for
    exploration only. The `origin_available` scope fits on all data up to the
    current origin and is the leak-free default for preprocessing. The
    `fit_window` scope fits on the estimation-window rows only and is the default
    for features and model selection. The `fixed_reference` scope fits once and
    reuses the parameters at later origins.

TargetSpec
    A declaration of a forecast target and how its forecast object is defined.
    The name identifies the panel column, and the optional transform and policy
    override the defaults derived from the {term}`t-code`. See
    [Running](concepts/running.md).

t-code
    An integer from the McCracken-Ng FRED transformation classification that
    encodes the stationarity transform for a series. The codes run from 1 (level)
    through 2 (first difference), 3 (second difference), 4 (log level), 5 (log
    first difference, the growth rate), 6 (log second difference), to 7
    (percentage change). The pipeline uses it to derive the forecast policy and
    target transform.

validation design
    How hyperparameter-selection splits are formed inside the estimation sample.
    Options include last_block, poos, expanding, rolling_blocks, blocked_kfold,
    and random_kfold, the last reserved for reproducing papers that used random
    folds. It is one part of a {term}`WindowSpec`.

WindowSpec
    The time frame passed across selection, model, and evaluation stages. It
    combines an {term}`estimation window`, a {term}`validation design`, a test
    window of origins and horizons, and an alignment rule for joining features
    and targets, together with the {term}`retrain and retune cadence`. See
    [Windows](concepts/windows.md).
```
