# Custom Evaluation Tests

[Back to custom extensions](index.md)

This page is generated from the live callable signatures.

## Callable Reference

### custom_test

Qualified name: `macroforecast.tests.custom_test`

#### Signature

```python
macroforecast.tests.custom_test(name: str, func: Callable[..., Any], *args: Any, alternative: str = "two_sided", alpha: float = 0.05, correction_policy: str | None = None, metadata: Mapping[str, Any] | None = None, **params: Any) -> TestResult
```

#### Description

Run a user-supplied forecast test and coerce it to ``TestResult``.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `name` | positional or keyword | `str` | `required` |
| `func` | positional or keyword | `Callable[..., Any]` | `required` |
| `args` | var positional | `Any` | `required` |
| `alternative` | keyword only | `str` | `"two_sided"` |
| `alpha` | keyword only | `float` | `0.05` |
| `correction_policy` | keyword only | `str \| None` | `None` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `params` | var keyword | `Any` | `required` |

#### Returns

`TestResult`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.tests.custom_test(...)
```

### EvalSpec

Qualified name: `macroforecast.pipeline.spec.EvalSpec`

#### Signature

```python
macroforecast.pipeline.EvalSpec(benchmark: str, metrics: tuple[str | Callable[..., float], ...] = ('rmse', 'relative_mse', 'r2_oos'), tests: tuple[str, ...] = ('dm', 'cw', 'mcs'), by: tuple[str, ...] = ('target', 'horizon'), primary_axis: str = "contender", cw_for_nested: bool = True, mcs_alpha: float = 0.1, mcs_method: str = "iterative", multiple_testing: str | None = None, subsamples: Mapping[str, SubsampleWindow] | None = None, dm_kwargs: Mapping[str, Any] = <factory>, loss: Callable[[Any, Any], Any] | None = None, test_options: Mapping[str, Mapping[str, Any]] = <factory>, calibration_alpha: float = 0.05) -> None
```

#### Description

Automatic evaluation and significance-testing configuration.

``metrics`` entries are either a name resolved through the metric registry
(:func:`macroforecast.metrics.get_metric`, e.g. ``"mae"``, ``"mape"``) or a
callable ``metric(y_true, y_pred) -> float`` named by its ``__name__``.
``accuracy_table`` computes one column per listed metric, per contender, on
the same pairwise-vs-benchmark sample it always has. The three defaults
(``"rmse"``, ``"relative_mse"``, ``"r2_oos"``) keep their existing
benchmark-relative formulas regardless of how they are requested.

``loss`` is a per-observation loss ``loss(y_true, y_pred) -> ndarray`` used
by the Diebold-Mariano loss differential and the Model Confidence Set's loss
matrix; ``None`` (default) is squared error, the prior, only behavior. Since
the Clark-West adjustment is derived under quadratic loss, setting a custom
``loss`` makes ``significance_table`` skip CW (with a ``UserWarning``)
rather than compute it against the wrong loss.

``tests`` lists which significance tests actually run; unsupported names
raise at :func:`pipeline_spec` build time (see ``SUPPORTED_EVAL_TESTS``).
Pairwise contender-vs-benchmark tests are ``"dm"``, ``"cw"``, ``"gw"``,
``"enc_new"``, ``"enc_t"``, ``"gr"``, and ``"mz"``. ``"mz"`` is the
Mincer-Zarnowitz actual-on-forecast rationality regression. ``"pt"``, ``"hm"``, and
``"ag"`` are directional-accuracy tests for the contender's own sign
forecasts, evaluated on the same benchmark-aligned sample for consistency
with the pairwise tests. Joint multi-horizon pairwise tests are ``"uspa"``
and ``"aspa"``; they require at least two horizons and use ``"joint"`` as
their significance-table horizon sentinel. Full-set benchmark comparisons are ``"mcs"``,
``"spa"``, ``"rc"``, and ``"stepm"``; they populate ``PipelineReport.mcs``.
``"spa"``, ``"rc"``, and ``"stepm"`` require the ``arch`` extra
(``pip install "macroforecast[arch]"``).
``"berkowitz"``/``"pit_autocorr"``/``"coverage"`` are PIT-based calibration
diagnostics (Phase 1 density pipeline) -- they populate
``PipelineReport.calibration`` rather than ``significance``/``mcs`` and,
like every other test name, are opt-in only (absent from the default).
``test_options`` maps a requested test name to keyword options for that
test's underlying public callable. Option blocks are validated when
:func:`pipeline_spec` is built: the key must appear in ``tests``, every
option name must be accepted by that test's callable, and the option must
not be one the evaluator supplies itself. The refused set is per test, and
each is reported with a pointer to its real owner rather than accepted and
then overwritten:

* set-comparison (``"mcs"``, ``"spa"``, ``"rc"``, ``"stepm"``) -- these take
  their loss panel's column names as keywords and the pipeline builds that
  panel, so ``loss``, ``model``, ``origin``, ``target``, ``horizon``, plus
  ``benchmark`` for ``"spa"``/``"rc"``/``"stepm"``, belong to
  ``EvalSpec.loss``/``EvalSpec.benchmark`` or to the internal schema;
* pairwise (``"dm"``, ``"cw"``, ``"gw"``, ``"enc_t"``) -- ``horizon`` is the
  horizon of the cell being evaluated, from ``pipeline_spec(horizons=...)``,
  and ``"dm"``'s ``input_type`` is fixed because the pipeline passes losses
  it has already computed;
* dispatch-by-name (``"pt"``, ``"hm"``, ``"ag"``, ``"gr"`` via ``method``;
  ``"uspa"``, ``"aspa"`` via ``statistic``) -- the requested test name in
  ``tests`` already chooses it.

Everything the caller genuinely controls stays valid: ``alpha``,
``hac_lags``, ``threshold``, ``kernel``, ``correction``, ``small_sample``,
``cw_adjustment``, ``critical_value``, ``instruments``, ``"gr"``'s
``lag_truncate``, and every bootstrap parameter (``n_boot``,
``block_length``, ``bootstrap_method``, ``studentize``, an explicit
``random_state``).

Density/interval accuracy metrics -- ``"crps"``, ``"gaussian_nll"``,
``"log_score"``, ``"negative_log_score"``, ``"qlike"``, ``"pinball_loss"``,
``"coverage_rate"``, ``"interval_width"``, ``"interval_score"`` -- are
requested the SAME way as any other ``metrics`` entry; they land in
``PipelineReport.density`` instead of ``accuracy`` because they need a
``variance_prediction``/``quantile_predictions`` column rather than plain
``(y_true, y_pred)`` (see ``macroforecast.metrics.metric_kind``). Requesting
one on a forecast frame that carries no such column raises the same
actionable ``ValueError`` :func:`macroforecast.metrics.evaluate_forecasts`
already raises. Absent from the defaults, so a default-EvalSpec run never
computes them.

``calibration_alpha`` is the significance level of the ``"berkowitz"`` and
``"pit_autocorr"`` tests -- the level at which their null of a correctly
specified predictive density is rejected. It does **not** govern
``"coverage"``: that test checks an interval against ITS OWN nominal level,
which ``calibration_table`` derives from the widest symmetric quantile pair
present in ``quantile_predictions`` (a 5%/95% pair is a 90% interval, so
``alpha=0.10``), because the nominal coverage of an interval is a property
of the interval rather than a reporting choice. ``calibration_alpha`` also
does not affect ``mcs_alpha``.

``subsamples`` optionally maps names to :class:`SubsampleWindow` values.
These are evaluation-window splits of an already-produced POOS forecast
frame: target-date rows are filtered before scoring and testing, without
refitting models or creating new forecast cells.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `benchmark` | positional or keyword | `str` | `required` |
| `metrics` | positional or keyword | `tuple[str \| Callable[..., float], ...]` | `("rmse", "relative_mse", "r2_oos")` |
| `tests` | positional or keyword | `tuple[str, ...]` | `("dm", "cw", "mcs")` |
| `by` | positional or keyword | `tuple[str, ...]` | `("target", "horizon")` |
| `primary_axis` | positional or keyword | `str` | `"contender"` |
| `cw_for_nested` | positional or keyword | `bool` | `True` |
| `mcs_alpha` | positional or keyword | `float` | `0.1` |
| `mcs_method` | positional or keyword | `str` | `"iterative"` |
| `multiple_testing` | positional or keyword | `str \| None` | `None` |
| `subsamples` | positional or keyword | `Mapping[str, SubsampleWindow] \| None` | `None` |
| `dm_kwargs` | positional or keyword | `Mapping[str, Any]` | `<factory>` |
| `loss` | positional or keyword | `Callable[[Any, Any], Any] \| None` | `None` |
| `test_options` | positional or keyword | `Mapping[str, Mapping[str, Any]]` | `<factory>` |
| `calibration_alpha` | positional or keyword | `float` | `0.05` |

#### Returns

`None`

#### Minimal Use

```python
import macroforecast as mf
# Construct with the signature above:
# mf.pipeline.EvalSpec(...)
```

#### Dataclass Fields

| Field | Type | Default |
| --- | --- | --- |
| `benchmark` | `str` | `required` |
| `metrics` | `tuple[str \| Callable[..., float], ...]` | `("rmse", "relative_mse", "r2_oos")` |
| `tests` | `tuple[str, ...]` | `("dm", "cw", "mcs")` |
| `by` | `tuple[str, ...]` | `("target", "horizon")` |
| `primary_axis` | `str` | `"contender"` |
| `cw_for_nested` | `bool` | `True` |
| `mcs_alpha` | `float` | `0.1` |
| `mcs_method` | `str` | `"iterative"` |
| `multiple_testing` | `str \| None` | `None` |
| `subsamples` | `Mapping[str, SubsampleWindow] \| None` | `None` |
| `dm_kwargs` | `Mapping[str, Any]` | `default_factory` |
| `loss` | `Callable[[Any, Any], Any] \| None` | `None` |
| `test_options` | `Mapping[str, Mapping[str, Any]]` | `default_factory` |
| `calibration_alpha` | `float` | `0.05` |
