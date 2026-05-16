# `information_set_type`

[Back to L1](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``information_set_type`` on sub-layer ``l1_a`` (layer ``l1``).

## Sub-layer

**l1_a**

## Axis metadata

- Default: `'final_revised_data'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 2 option(s)
- Future: 0 option(s)

## Options

### `final_revised_data`  --  operational

Use the final, currently-published revised data series.

Standard pseudo-OOS evaluation protocol: at each forecast origin the entire time series uses today's revised (currently-published) data -- that is, revisions that occurred after the origin date are still incorporated. The model never sees the data as it existed in real time.

This is the canonical approach used by McCracken & Ng (2016) and most published forecasting benchmark studies. It is fast, simple, and directly comparable across papers. The acknowledged limitation -- noted by Stark & Croushore (2002) and Faust & Wright (2009) -- is that it overstates real-time forecast accuracy for heavily-revised series (e.g., GDP, payrolls) because subsequent revisions correct early-vintage measurement error that a real forecaster would have faced.

Pairs naturally with ``vintage_policy: current_vintage``.

**When to use**

Benchmark and methods studies where vintage realism is not the primary focus; replication of published FRED-MD/QD benchmarks; any study comparing models on the same revised data.

**When NOT to use**

Real-time evaluation papers where data revisions materially affect conclusions -- use ``real_time_alfred`` when it becomes available (currently a future feature, Cycle 14 K-4).

**References**

* Stark & Croushore (2002) 'Forecasting with a real-time data set for macroeconomists', Journal of Macroeconomics 24(4). (doi:10.1016/S0164-0704(02)00041-0)
* Faust & Wright (2009) 'Comparing Greenbook and reduced form forecasts using a large realtime dataset', Journal of Business & Economic Statistics 27(4). (doi:10.1198/jbes.2009.06043)
* McCracken & Ng (2016) 'FRED-MD: A Monthly Database for Macroeconomic Research', Journal of Business & Economic Statistics 34(4). (doi:10.1080/07350015.2015.1086655)

**Related options**: [`pseudo_oos_on_revised_data`](#pseudo-oos-on-revised-data), [`vintage_policy`](#vintage-policy)

_Last reviewed 2026-05-16 by macroforecast author._

### `pseudo_oos_on_revised_data`  --  operational

Pseudo out-of-sample using revised series; explicit acknowledgement of using post-hoc data.

Numerically identical to ``final_revised_data`` in all released versions (v0.9.x and earlier): both options produce the same forecasts from the same revised data. The distinction is purely semantic -- selecting ``pseudo_oos_on_revised_data`` records the explicit recipe-author acknowledgement that revised data is being used for out-of-sample evaluation.

This axis value is exposed so that future versions can route real-time vintage requests through the same axis without breaking existing recipes. Studies that compare pseudo-OOS-on-revised against real-time ALFRED vintages (once Cycle 14 K-4 is implemented) will use this option to label the revised-data branch explicitly.

Pairs with ``vintage_policy: current_vintage``.

**When to use**

Studies explicitly contrasting pseudo-OOS-on-revised-data vs real-time vintage performance (once ``real_time_alfred`` is implemented); recipe scripts that want to make the revised-data protocol visible in the YAML rather than relying on the default.

**References**

* Stark & Croushore (2002) 'Forecasting with a real-time data set for macroeconomists', Journal of Macroeconomics 24(4). (doi:10.1016/S0164-0704(02)00041-0)
* Faust & Wright (2009) 'Comparing Greenbook and reduced form forecasts using a large realtime dataset', Journal of Business & Economic Statistics 27(4). (doi:10.1198/jbes.2009.06043)

**Related options**: [`final_revised_data`](#final-revised-data), [`vintage_policy`](#vintage-policy)

_Last reviewed 2026-05-16 by macroforecast author._
