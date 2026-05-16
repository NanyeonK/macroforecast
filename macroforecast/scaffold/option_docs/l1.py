"""L1 data definition -- per-option documentation.

L1 is the largest layer: 26 axes spanning data source, target structure,
variable universe, geography (FRED-SD), sample window, horizons, and
regime definition. This module ships Tier-1 documentation for all L1
axes that have been through a reviewer-stamped review pass.

Tier-1-complete sub-layers (all axes Tier-1 unless noted):
* L1.A -- all 6 axes (custom_source_policy, dataset, vintage_policy,
  frequency, information_set_type, fred_sd_frequency_policy) -- Cycle 17
* L1.B -- target_structure (1 axis) -- Cycle 18
* L1.C -- all 8 axes (variable_universe, missing_availability,
  raw_missing_policy, raw_outlier_policy, release_lag_rule,
  contemporaneous_x_rule, official_transform_policy,
  official_transform_scope) -- Cycle 19
* L1.D -- all 6 axes (target_geography_scope, predictor_geography_scope,
  fred_sd_state_group, fred_sd_variable_group, state_selection,
  sd_variable_selection) -- Cycle 20
* L1.E -- sample_start_rule / sample_end_rule
* L1.F -- horizon_set
* L1.G -- regime_definition
"""
from __future__ import annotations

from . import register
from .types import CodeExample, OptionDoc, ParameterDoc, Reference

_REVIEWED = "2026-05-04"
_REVIEWER = "macroforecast author"


# Common references reused across L1 entries.
_REF_DESIGN_L1 = Reference(
    citation="macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'",
)
_REF_MCCRACKEN_NG_2016 = Reference(
    citation="McCracken & Ng (2016) 'FRED-MD: A Monthly Database for Macroeconomic Research', Journal of Business & Economic Statistics 34(4).",
    doi="10.1080/07350015.2015.1086655",
)
_REF_MCCRACKEN_NG_2020 = Reference(
    citation="McCracken & Ng (2020) 'FRED-QD: A Quarterly Database for Macroeconomic Research', Federal Reserve Bank of St. Louis Review.",
)
_REF_HAMILTON_1989 = Reference(
    citation="Hamilton (1989) 'A new approach to the economic analysis of nonstationary time series and the business cycle', Econometrica 57(2).",
    doi="10.2307/1912559",
)
_REF_TONG_1990 = Reference(
    citation="Tong (1990) 'Non-linear Time Series: A Dynamical System Approach', Oxford University Press.",
)
_REF_BAI_PERRON_1998 = Reference(
    citation="Bai & Perron (1998) 'Estimating and testing linear models with multiple structural changes', Econometrica 66(1).",
    doi="10.2307/2998540",
)
_REF_NBER = Reference(
    citation="National Bureau of Economic Research, 'US Business Cycle Expansions and Contractions'.",
    url="https://www.nber.org/research/business-cycle-dating",
)
_REF_ALFRED = Reference(
    citation="Federal Reserve Bank of St. Louis, 'ALFRED: Archival Federal Reserve Economic Data' -- real-time vintage archive of FRED series.",
    url="https://alfred.stlouisfed.org/",
)
_REF_CROUSHORE_STARK_2001 = Reference(
    citation="Croushore & Stark (2001) 'A real-time data set for macroeconomists', Journal of Econometrics 105(1).",
    doi="10.1016/S0304-4076(01)00072-0",
)
_REF_STARK_CROUSHORE_2002 = Reference(
    citation="Stark & Croushore (2002) 'Forecasting with a real-time data set for macroeconomists', Journal of Macroeconomics 24(4).",
    doi="10.1016/S0164-0704(02)00041-0",
)
_REF_FAUST_WRIGHT_2009 = Reference(
    citation="Faust & Wright (2009) 'Comparing Greenbook and reduced form forecasts using a large realtime dataset', Journal of Business & Economic Statistics 27(4).",
    doi="10.1198/jbes.2009.06043",
)
_REF_FRED_SD = Reference(
    citation="macroforecast PR #251 (use_sd_inferred_tcodes) -- FRED-SD integration, state-level frequency-policy design.",
)


def _entry(
    sublayer: str,
    axis: str,
    option: str,
    summary: str,
    description: str,
    when_to_use: str,
    *,
    when_not_to_use: str = "",
    references: tuple[Reference, ...] = (_REF_DESIGN_L1,),
    related_options: tuple[str, ...] = (),
    examples: tuple[CodeExample, ...] = (),
    last_reviewed: str = _REVIEWED,
    reviewer: str = _REVIEWER,
) -> OptionDoc:
    return OptionDoc(
        layer="l1",
        sublayer=sublayer,
        axis=axis,
        option=option,
        summary=summary,
        description=description,
        when_to_use=when_to_use,
        references=references,
        when_not_to_use=when_not_to_use,
        related_options=related_options,
        examples=examples,
        last_reviewed=last_reviewed,
        reviewer=reviewer,
    )


# ---------------------------------------------------------------------------
# L1.A custom_source_policy
# ---------------------------------------------------------------------------

_L1A_SOURCE_OFFICIAL_ONLY = _entry(
    "l1_a", "custom_source_policy", "official_only",
    summary="Use the McCracken-Ng curated FRED-MD/QD/SD vintages.",
    description=(
        "Loads the bundled FRED snapshot via macroforecast's raw adapter -- no "
        "network access at runtime, no per-user data file. Vintages are "
        "pinned in ``macroforecast/raw/datasets/`` so two users on the same "
        "package version see identical raw inputs.\n\n"
        "This is the canonical recipe path: every published replication "
        "script, every example in the gallery, and every CI check uses "
        "``official_only`` so cross-user comparability is bit-exact."
    ),
    when_to_use=(
        "Reproducing or extending published macro forecasting work; "
        "running benchmarks where readers need to repeat the study from "
        "the recipe alone; default for any FRED-based analysis."
    ),
    when_not_to_use=(
        "Forecasting on non-FRED panels (firm-level data, country-specific "
        "series); needs a vintage newer than the bundled snapshot."
    ),
    references=(_REF_DESIGN_L1, _REF_MCCRACKEN_NG_2016, _REF_MCCRACKEN_NG_2020),
    related_options=("custom_panel_only", "official_plus_custom", "dataset", "vintage_policy"),
    examples=(
        CodeExample(
            title="FRED-MD baseline",
            code=(
                "1_data:\n"
                "  fixed_axes:\n"
                "    custom_source_policy: official_only\n"
                "    dataset: fred_md\n"
                "  leaf_config:\n"
                "    target: CPIAUCSL\n"
            ),
        ),
    ),
)

_L1A_SOURCE_CUSTOM_PANEL = _entry(
    "l1_a", "custom_source_policy", "custom_panel_only",
    summary="Load a single user-supplied panel (CSV / Parquet / inline dict).",
    description=(
        "Bypasses the FRED adapter entirely. The user provides:\n\n"
        "* an inline ``custom_panel_inline`` dict (small synthetic panels), or\n"
        "* a ``custom_source_path`` pointing to a CSV / Parquet file.\n\n"
        "The L1 runtime applies no schema-level validation beyond 'has a "
        "date column and at least the requested target series'. Variable "
        "metadata that the McCracken-Ng panel ships (group tags, t-codes, "
        "release dates) is unavailable, so axes that depend on it -- "
        "``official_transform_policy``, ``fred_sd_state_group``, etc. -- "
        "are inactive."
    ),
    when_to_use=(
        "Forecasting on proprietary firm panels, country-specific series, "
        "or any data not in FRED. Also the standard path for unit tests "
        "and tutorial recipes that ship deterministic synthetic data."
    ),
    when_not_to_use=(
        "When McCracken-Ng's curation (t-codes, group tags) is part of the "
        "study design -- ``official_only`` or ``official_plus_custom`` "
        "preserves it."
    ),
    references=(_REF_DESIGN_L1,),
    related_options=("official_only", "official_plus_custom"),
    examples=(
        CodeExample(
            title="Inline panel for a unit test",
            code=(
                "1_data:\n"
                "  fixed_axes:\n"
                "    custom_source_policy: custom_panel_only\n"
                "  leaf_config:\n"
                "    target: y\n"
                "    custom_panel_inline:\n"
                "      date: [2020-01-01, 2020-02-01]\n"
                "      y:    [1.0, 2.0]\n"
                "      x1:   [0.5, 1.0]\n"
            ),
        ),
    ),
)

_L1A_SOURCE_OFFICIAL_PLUS_CUSTOM = _entry(
    "l1_a", "custom_source_policy", "official_plus_custom",
    summary="Merge the official FRED panel with a user-supplied auxiliary panel.",
    description=(
        "Loads the FRED vintage (per ``dataset``) and joins a user CSV / "
        "Parquet on the date index. Requires ``custom_source_path`` plus "
        "``custom_merge_rule`` (one of ``inner_join`` / ``left_join`` / "
        "``outer_join``) so the merge contract is explicit.\n\n"
        "This is the canonical extension path for studies that want McCracken-Ng "
        "predictors plus a few additional series (e.g., proprietary survey "
        "indicators, alternative-data nowcast inputs)."
    ),
    when_to_use=(
        "Augmenting FRED-based studies with a small number of additional "
        "predictors that are not in the official panel."
    ),
    when_not_to_use=(
        "Pure custom panels (use ``custom_panel_only``); pure official "
        "panels (use ``official_only``); mixing two FRED vintages (the "
        "merge rule expects one FRED + one custom)."
    ),
    references=(_REF_DESIGN_L1,),
    related_options=("official_only", "custom_panel_only"),
    examples=(
        CodeExample(
            title="FRED-MD plus a single proprietary series",
            code=(
                "1_data:\n"
                "  fixed_axes:\n"
                "    custom_source_policy: official_plus_custom\n"
                "    dataset: fred_md\n"
                "  leaf_config:\n"
                "    target: CPIAUCSL\n"
                "    custom_source_path: data/proprietary_indicator.parquet\n"
                "    custom_merge_rule: left_join\n"
            ),
        ),
    ),
)



# ParameterDoc for custom_panel_only: three mutually-exclusive leaf_config keys.
# OptionDoc is frozen; replace() produces a new instance with parameters populated.
_L1A_SOURCE_CUSTOM_PANEL = _L1A_SOURCE_CUSTOM_PANEL.__class__(
    **{**_L1A_SOURCE_CUSTOM_PANEL.__dict__,
       "parameters": (
           ParameterDoc(
               name="custom_source_path",
               type="str | Path",
               default=None,
               constraint=(
                   "Exactly one of {custom_source_path, custom_panel_inline, "
                   "custom_panel_records} must be set."
               ),
               description=(
                   "Filesystem path (CSV or Parquet) to user-provided panel data. "
                   "The path is resolved relative to the recipe working directory."
               ),
           ),
           ParameterDoc(
               name="custom_panel_inline",
               type="dict",
               default=None,
               constraint=(
                   "Exactly one of {custom_source_path, custom_panel_inline, "
                   "custom_panel_records} must be set."
               ),
               description=(
                   "Inline panel as a dict with key 'date' (list of ISO date strings) "
                   "and one key per series (name -> list of float). Convenient for "
                   "unit tests and small synthetic examples without a file on disk."
               ),
           ),
           ParameterDoc(
               name="custom_panel_records",
               type="list[dict]",
               default=None,
               constraint=(
                   "Exactly one of {custom_source_path, custom_panel_inline, "
                   "custom_panel_records} must be set."
               ),
               description=(
                   "Row-records form of the panel. Each dict must have a 'date' key "
                   "plus one key per series. Equivalent to pandas 'records' orient."
               ),
           ),
       ),
    }
)

# ParameterDoc for official_plus_custom: custom_source_path + custom_merge_rule.
_L1A_SOURCE_OFFICIAL_PLUS_CUSTOM = _L1A_SOURCE_OFFICIAL_PLUS_CUSTOM.__class__(
    **{**_L1A_SOURCE_OFFICIAL_PLUS_CUSTOM.__dict__,
       "parameters": (
           ParameterDoc(
               name="custom_source_path",
               type="str | Path",
               default=None,
               constraint="Required when custom_source_policy=official_plus_custom.",
               description=(
                   "Filesystem path (CSV or Parquet) to the auxiliary panel to merge "
                   "onto the official FRED panel. Joined on the date index."
               ),
           ),
           ParameterDoc(
               name="custom_merge_rule",
               type="str",
               default=None,
               constraint=(
                   "Required when custom_source_policy=official_plus_custom. "
                   "Must be one of: 'left_join', 'inner_join', 'outer_join'."
               ),
               description=(
                   "How to merge the official FRED panel (left) with the custom panel "
                   "(right) on the date index. 'left_join' keeps all FRED dates; "
                   "'inner_join' keeps only dates present in both panels; 'outer_join' "
                   "keeps all dates from either panel, filling missing values with NaN."
               ),
           ),
       ),
    }
)

# ---------------------------------------------------------------------------
# L1.A dataset
# ---------------------------------------------------------------------------

def _dataset_entry(option: str, summary: str, description: str, when_to_use: str, refs: tuple[Reference, ...]) -> OptionDoc:
    return _entry(
        "l1_a", "dataset", option,
        summary=summary,
        description=description,
        when_to_use=when_to_use,
        references=refs,
        related_options=("custom_source_policy", "frequency", "horizon_set"),
    )


_L1A_FRED_MD = _dataset_entry(
    "fred_md",
    "FRED-MD: 130+ monthly US macro series (1959-).",
    (
        "The McCracken & Ng (2016) Monthly Database for Macroeconomic "
        "Research. Curated set of ~130 macroeconomic and financial series "
        "with stable transformation codes, group tags, and a single "
        "vintage per month.\n\n"
        "Default for monthly forecasting work; pairs with "
        "``horizon_set: standard_md`` (h ∈ {1, 3, 6, 9, 12, 18, 24}) and "
        "``frequency: monthly``."
    ),
    "Monthly inflation, employment, industrial-production, and term-structure forecasting.",
    (_REF_MCCRACKEN_NG_2016,),
)

_L1A_FRED_QD = _dataset_entry(
    "fred_qd",
    "FRED-QD: 250+ quarterly US macro series (1959-).",
    (
        "The McCracken & Ng (2020) Quarterly Database for Macroeconomic "
        "Research. Larger variable count than FRED-MD; quarterly cadence "
        "matches GDP / NIPA-style targets.\n\n"
        "Default for quarterly forecasting; pairs with "
        "``horizon_set: standard_qd`` (h ∈ {1, 2, 4, 8}) and "
        "``frequency: quarterly``."
    ),
    "GDP, consumption, investment, productivity nowcasting / forecasting.",
    (_REF_MCCRACKEN_NG_2020,),
)

_L1A_FRED_SD = _dataset_entry(
    "fred_sd",
    "FRED-SD: state-level US series with geographic axes.",
    (
        "State-level macro panel covering ~50 states + DC. Activates the "
        "L1.D geography axes (target_geography_scope / predictor_geography_scope) "
        "and the L7 ``us_state_choropleth`` figure type for spatial "
        "interpretation.\n\n"
        "FRED-SD ships with mixed monthly + quarterly frequencies; the "
        "L2.A frequency-alignment rules (issue #202) handle the mixed case."
    ),
    "State-level employment / payroll / housing forecasting; geographic-importance studies.",
    (_REF_DESIGN_L1,),
)

_L1A_FRED_MD_SD = _dataset_entry(
    "fred_md+fred_sd",
    "Joint FRED-MD + FRED-SD panel.",
    (
        "Concatenates the FRED-MD national series with FRED-SD state-level "
        "series on the date index. Useful when a study needs both national "
        "context (FRED-MD) and state-level granularity (FRED-SD) -- e.g., "
        "a state-level employment forecast conditioned on national CPI."
    ),
    "Studies where state-level targets need national-aggregate predictors.",
    (_REF_DESIGN_L1,),
)

_L1A_FRED_QD_SD = _dataset_entry(
    "fred_qd+fred_sd",
    "Joint FRED-QD + FRED-SD panel (quarterly + state-level mixed).",
    (
        "Concatenates FRED-QD with FRED-SD. Triggers the L2.A "
        "frequency-alignment rules because FRED-QD is quarterly while "
        "much of FRED-SD is monthly."
    ),
    "Quarterly state-level studies (rare; use only when the target is quarterly state-level).",
    (_REF_DESIGN_L1,),
)


# ---------------------------------------------------------------------------
# L1.A frequency
# ---------------------------------------------------------------------------

_L1A_FREQ_MONTHLY = _entry(
    "l1_a", "frequency", "monthly",
    summary="Monthly observation frequency.",
    description=(
        "Pinned monthly frequency. Sets the canonical sampling cadence to "
        "one calendar month per observation, so horizon h=1 means "
        "one-month-ahead and ``standard_md`` horizons h ∈ {1, 3, 6, 9, 12, "
        "18, 24} are interpreted in months.\n\n"
        "Compatible with ``dataset=fred_md`` and "
        "``dataset=fred_md+fred_sd``. When ``frequency`` is unset, the "
        "default ``'derived'`` sentinel resolves to ``monthly`` for FRED-MD "
        "datasets via ``_derived_frequency()`` at L1 normalization -- "
        "setting it explicitly is redundant for FRED-MD but required for "
        "custom panels that carry monthly observations."
    ),
    when_to_use=(
        "Monthly macro forecasting (industrial production, payrolls, CPI, "
        "etc.); custom panels with monthly observations; explicit override "
        "of the FRED-MD default for documentation clarity."
    ),
    references=(_REF_MCCRACKEN_NG_2016,),
    related_options=("quarterly", "dataset", "horizon_set"),
    last_reviewed="2026-05-16",
    reviewer="macroforecast author",
)

_L1A_FREQ_QUARTERLY = _entry(
    "l1_a", "frequency", "quarterly",
    summary="Quarterly observation frequency.",
    description=(
        "Pinned quarterly frequency. Sets the canonical sampling cadence to "
        "one calendar quarter per observation, so horizon h=1 means "
        "one-quarter-ahead and ``standard_qd`` horizons h ∈ {1, 2, 4, 8} "
        "are interpreted in quarters.\n\n"
        "Compatible with ``dataset=fred_qd`` and "
        "``dataset=fred_qd+fred_sd``. The ``'derived'`` default resolves to "
        "``quarterly`` when ``dataset=fred_qd`` via ``_derived_frequency()``. "
        "Setting it explicitly is required for custom panels that carry "
        "quarterly observations."
    ),
    when_to_use=(
        "Quarterly macro forecasting (GDP, productivity, NIPA-style targets); "
        "quarterly custom panels; FRED-QD-based studies."
    ),
    references=(_REF_MCCRACKEN_NG_2020,),
    related_options=("monthly", "dataset", "horizon_set"),
    last_reviewed="2026-05-16",
    reviewer="macroforecast author",
)


# ---------------------------------------------------------------------------
# L1.A vintage_policy
# ---------------------------------------------------------------------------

_L1A_VINTAGE_CURRENT = _entry(
    "l1_a", "vintage_policy", "current_vintage",
    summary="Use the latest available vintage of the dataset.",
    description=(
        "Loads the most recent FRED-MD/QD/SD snapshot bundled with the "
        "package. No real-time vintage tracking; revisions that happened "
        "after the snapshot date are not reflected.\n\n"
        "This is the only operational option in v1.0. Real-time vintages "
        "(ALFRED-style) are tracked as a future axis -- see GitHub issues "
        "#XXX."
    ),
    when_to_use="Default for any pseudo-out-of-sample study using revised data.",
    when_not_to_use="Real-time forecasting evaluations -- those need ALFRED vintages (future feature; see real_time_alfred).",
    related_options=("real_time_alfred", "information_set_type"),
    last_reviewed="2026-05-16",
    reviewer="macroforecast author",
)

_L1A_VINTAGE_REAL_TIME_ALFRED = _entry(
    "l1_a", "vintage_policy", "real_time_alfred",
    summary="Real-time ALFRED vintage policy (not yet implemented).",
    description=(
        "ALFRED (Archival FRED) is the St. Louis Fed's real-time data "
        "archive. It stores historical vintages of every FRED series, "
        "allowing researchers to reconstruct the information set that was "
        "actually available at any past date -- before subsequent data "
        "revisions occurred.\n\n"
        "Future macroforecast support will pull the historical-as-of "
        "vintage for each forecast origin from the ALFRED API, enabling "
        "true real-time replication studies where the model never sees "
        "data that was not yet released at the forecast origin.\n\n"
        "**Current behavior**: selecting ``real_time_alfred`` raises a "
        "hard ``ValueError`` at recipe validation with the message "
        "``'real_time_alfred is not yet implemented; future feature. "
        "Use current_vintage (default).'`` (Cycle 14 K-4). "
        "No partial execution occurs."
    ),
    when_to_use=(
        "Future. For now, use ``current_vintage`` and document the "
        "data-revision context via ``data_revision_tag`` in manifest "
        "provenance (Cycle 14 K-3 auto-captures ``fred-md@YYYY-MM``)."
    ),
    when_not_to_use=(
        "Any current recipe -- this option is hard-rejected at validation "
        "in all released versions up to and including v0.9.x."
    ),
    references=(_REF_ALFRED, _REF_CROUSHORE_STARK_2001),
    related_options=("current_vintage",),
    last_reviewed="2026-05-16",
    reviewer="macroforecast author",
)


# ---------------------------------------------------------------------------
# L1.A information_set_type
# ---------------------------------------------------------------------------

_L1A_INFOSET_FINAL = _entry(
    "l1_a", "information_set_type", "final_revised_data",
    summary="Use the final, currently-published revised data series.",
    description=(
        "Standard pseudo-OOS evaluation protocol: at each forecast origin "
        "the entire time series uses today's revised (currently-published) "
        "data -- that is, revisions that occurred after the origin date "
        "are still incorporated. The model never sees the data as it "
        "existed in real time.\n\n"
        "This is the canonical approach used by McCracken & Ng (2016) and "
        "most published forecasting benchmark studies. It is fast, simple, "
        "and directly comparable across papers. The acknowledged "
        "limitation -- noted by Stark & Croushore (2002) and Faust & "
        "Wright (2009) -- is that it overstates real-time forecast accuracy "
        "for heavily-revised series (e.g., GDP, payrolls) because "
        "subsequent revisions correct early-vintage measurement error that "
        "a real forecaster would have faced.\n\n"
        "Pairs naturally with ``vintage_policy: current_vintage``."
    ),
    when_to_use=(
        "Benchmark and methods studies where vintage realism is not the "
        "primary focus; replication of published FRED-MD/QD benchmarks; "
        "any study comparing models on the same revised data."
    ),
    when_not_to_use=(
        "Real-time evaluation papers where data revisions materially "
        "affect conclusions -- use ``real_time_alfred`` when it becomes "
        "available (currently a future feature, Cycle 14 K-4)."
    ),
    references=(_REF_STARK_CROUSHORE_2002, _REF_FAUST_WRIGHT_2009, _REF_MCCRACKEN_NG_2016),
    related_options=("pseudo_oos_on_revised_data", "vintage_policy"),
    last_reviewed="2026-05-16",
    reviewer="macroforecast author",
)

_L1A_INFOSET_PSEUDO = _entry(
    "l1_a", "information_set_type", "pseudo_oos_on_revised_data",
    summary="Pseudo out-of-sample using revised series; explicit acknowledgement of using post-hoc data.",
    description=(
        "Numerically identical to ``final_revised_data`` in all released "
        "versions (v0.9.x and earlier): both options produce the same "
        "forecasts from the same revised data. The distinction is purely "
        "semantic -- selecting ``pseudo_oos_on_revised_data`` records the "
        "explicit recipe-author acknowledgement that revised data is being "
        "used for out-of-sample evaluation.\n\n"
        "This axis value is exposed so that future versions can route "
        "real-time vintage requests through the same axis without breaking "
        "existing recipes. Studies that compare pseudo-OOS-on-revised "
        "against real-time ALFRED vintages (once Cycle 14 K-4 is "
        "implemented) will use this option to label the revised-data "
        "branch explicitly.\n\n"
        "Pairs with ``vintage_policy: current_vintage``."
    ),
    when_to_use=(
        "Studies explicitly contrasting pseudo-OOS-on-revised-data vs "
        "real-time vintage performance (once ``real_time_alfred`` is "
        "implemented); recipe scripts that want to make the revised-data "
        "protocol visible in the YAML rather than relying on the default."
    ),
    references=(_REF_STARK_CROUSHORE_2002, _REF_FAUST_WRIGHT_2009),
    related_options=("final_revised_data", "vintage_policy"),
    last_reviewed="2026-05-16",
    reviewer="macroforecast author",
)


# ---------------------------------------------------------------------------
# L1.B target_structure
# ---------------------------------------------------------------------------

_L1B_TARGET_SINGLE = _entry(
    "l1_b", "target_structure", "single_target",
    summary="Forecast one target series at a time.",
    description=(
        "The recipe declares one ``target`` in L1 leaf_config. All "
        "downstream layers (feature DAG, model, evaluation) operate on "
        "that single series.\n\n"
        "This is the dominant pattern for benchmark studies because most "
        "forecasting literature reports per-target metrics; multi-series "
        "studies typically compose multiple single-target runs in a sweep."
    ),
    when_to_use="Default. Any standard forecasting benchmark.",
    related_options=("multi_target",),
    examples=(
        CodeExample(
            title="Forecast CPI inflation",
            code="1_data:\n  leaf_config:\n    target: CPIAUCSL\n",
        ),
    ),
)

_L1B_TARGET_MULTI = _entry(
    "l1_b", "target_structure", "multi_target",
    summary="Forecast multiple target series jointly within one cell.",
    description=(
        "The recipe declares ``targets: [a, b, c]`` in leaf_config. The L4 "
        "model is fit per-(target, horizon) tuple; the L5 metrics table "
        "carries one row per (model, target, horizon, origin).\n\n"
        "Useful for vector-target methods (VAR, FAVAR, BVAR) and for "
        "studies that compute cross-target metrics (e.g., portfolio MSE)."
    ),
    when_to_use="VAR-style joint forecasting; cross-target evaluation; replicating papers that report joint metrics.",
    when_not_to_use="Independent per-target studies -- those are usually clearer as separate sweep cells.",
    related_options=("single_target",),
)

# ParameterDoc for single_target: leaf_config.target key.
_L1B_TARGET_SINGLE = _L1B_TARGET_SINGLE.__class__(
    **{**_L1B_TARGET_SINGLE.__dict__,
       "parameters": (
           ParameterDoc(
               name="target",
               type="str",
               default=None,
               constraint="Required when target_structure=single_target.",
               description=(
                   "FRED series ID of the variable to forecast (e.g., CPIAUCSL, UNRATE). "
                   "Must be present in the chosen dataset after any transformation step."
               ),
           ),
       ),
    }
)

# ParameterDoc for multi_target: leaf_config.targets key.
_L1B_TARGET_MULTI = _L1B_TARGET_MULTI.__class__(
    **{**_L1B_TARGET_MULTI.__dict__,
       "parameters": (
           ParameterDoc(
               name="targets",
               type="list[str]",
               default=None,
               constraint="Required when target_structure=multi_target. Must have length >= 1.",
               description=(
                   "List of FRED series IDs to forecast jointly. Each element is a "
                   "separate target; the model is fit per-(target, horizon) tuple."
               ),
           ),
       ),
    }
)


# ---------------------------------------------------------------------------
# L1.C variable_universe
# ---------------------------------------------------------------------------

def _variable_universe(option: str, summary: str, description: str, when_to_use: str) -> OptionDoc:
    return _entry(
        "l1_c", "variable_universe", option,
        summary=summary, description=description, when_to_use=when_to_use,
        references=(_REF_DESIGN_L1, _REF_MCCRACKEN_NG_2016),
        related_options=("missing_availability", "official_transform_policy"),
    )


_L1C_VARS_ALL = _variable_universe(
    "all_variables",
    "Use every series in the chosen dataset.",
    (
        "FRED-MD/QD ships ~130 / ~250 series respectively. ``all_variables`` "
        "uses every one of them as predictors (target excluded). Standard "
        "for high-dimensional forecasting comparisons (PCR, lasso, factor "
        "models)."
    ),
    "Default. Any high-dimensional benchmark following McCracken-Ng.",
)

_L1C_VARS_CORE = _variable_universe(
    "core_variables",
    "Restrict to McCracken-Ng's curated 'core' subset (~30 series).",
    (
        "Smaller predictor set covering output, prices, money/credit, "
        "interest rates, and labor. Useful when a study wants a "
        "low-dimensional benchmark or replicates a paper that used the "
        "core set explicitly."
    ),
    "Low-dimensional benchmark; comparison against published 'core' panel results.",
)

_L1C_VARS_CATEGORY = _variable_universe(
    "category_variables",
    "Restrict to one McCracken-Ng category (e.g., 'output_and_income').",
    (
        "Uses one of the 8 (FRED-MD) / 14 (FRED-QD) category groupings as "
        "the predictor set. Requires ``leaf_config.variable_category`` "
        "naming the chosen category."
    ),
    "Within-category importance studies; testing whether one block alone is sufficient.",
)

_L1C_VARS_TARGET_SPEC = _variable_universe(
    "target_specific_variables",
    "Use a custom predictor list keyed to the target.",
    (
        "Requires ``leaf_config.target_specific_columns: {target: [predictors...]}``. "
        "Different targets see different predictor sets. Useful when "
        "domain knowledge says only certain series are relevant for a "
        "given target (e.g., housing-target studies use housing-block "
        "predictors)."
    ),
    "Domain-specific studies where each target has a known predictor block.",
)

_L1C_VARS_EXPLICIT = _variable_universe(
    "explicit_variable_list",
    "Use exactly the columns listed in leaf_config.variable_universe_columns.",
    (
        "Most flexible option. The recipe author supplies the full predictor "
        "column list in leaf_config; macroforecast filters the panel to that "
        "list verbatim. No grouping logic, no category lookup."
    ),
    "Replication scripts that need an exact predictor set; ablations.",
)
_L1C_VARS_EXPLICIT = _L1C_VARS_EXPLICIT.__class__(
    **{**_L1C_VARS_EXPLICIT.__dict__,
       "parameters": (
           ParameterDoc(
               name="variable_universe_columns",
               type="list[str]",
               default=None,
               constraint="Required when variable_universe=explicit_variable_list; must be non-empty.",
               description=(
                   "Explicit list of column names from the data source to use as the predictor "
                   "universe. Validator rejects missing or empty list."
               ),
           ),
       ),
    }
)


# ---------------------------------------------------------------------------
# L1.C official_transform_policy
# ---------------------------------------------------------------------------

_L1C_TRANSFORM_OFFICIAL = _entry(
    "l1_c", "official_transform_policy", "apply_official_tcode",
    summary="Apply McCracken-Ng's series-by-series stationarity transforms.",
    description=(
        "Each FRED-MD/QD series ships with a transformation code (t-code) "
        "1-7 that maps to a stationarity transform: 1=level, 2=Δlevel, "
        "5=Δlog, 6=Δ²log, etc. ``apply_official_tcode`` runs the canonical "
        "transform per series so downstream estimators see stationary "
        "inputs.\n\n"
        "This is the canonical preprocessing path for the McCracken-Ng "
        "benchmark family. Every published replication on FRED-MD/QD uses "
        "it."
    ),
    when_to_use="Default for FRED-MD/QD studies. Canonical replication path.",
    when_not_to_use="Studies that want to compare alternative transform schemes (use ``keep_official_raw_scale`` and apply transforms in L2 manually).",
    references=(_REF_DESIGN_L1, _REF_MCCRACKEN_NG_2016),
    related_options=("keep_official_raw_scale", "official_transform_scope"),
)

_L1C_TRANSFORM_RAW = _entry(
    "l1_c", "official_transform_policy", "keep_official_raw_scale",
    summary="Skip the canonical t-codes; keep raw level data.",
    description=(
        "Series stay on their native scale (levels, ratios, indices). "
        "Useful for tree-based models that don't need stationarity, or "
        "for studies that apply alternative transforms in L2 / L3."
    ),
    when_to_use="Tree / forest models that don't require stationarity; alternative-transform studies.",
    references=(_REF_DESIGN_L1,),
    related_options=("apply_official_tcode",),
)


# ---------------------------------------------------------------------------
# L1.D target_geography_scope
# ---------------------------------------------------------------------------

_L1D_GEO_SINGLE = _entry(
    "l1_d", "target_geography_scope", "single_state",
    summary="Single FRED-SD state target (e.g., California payrolls).",
    description=(
        "Selects one US state as the target. Requires "
        "``leaf_config.target_state`` (two-letter postal code). Predictors "
        "default to ``match_target`` (same state)."
    ),
    when_to_use="State-level case studies (e.g., CA / TX / NY-specific forecasts).",
    references=(_REF_DESIGN_L1,),
    related_options=("all_states", "selected_states", "predictor_geography_scope"),
)

_L1D_GEO_ALL = _entry(
    "l1_d", "target_geography_scope", "all_states",
    summary="Forecast every state's series jointly (50+DC targets).",
    description=(
        "Treats every state series as a target. The L5 metrics table "
        "carries one row per (model, state, horizon, origin) and the L7 "
        "``us_state_choropleth`` figure type maps importance scores to "
        "the geographic layout.\n\n"
        "This is the standard FRED-SD configuration for cross-state "
        "comparison studies."
    ),
    when_to_use="Geographic-importance studies; cross-state benchmark comparisons.",
    references=(_REF_DESIGN_L1,),
    related_options=("single_state", "selected_states", "fred_sd_state_group"),
)

_L1D_GEO_SELECTED = _entry(
    "l1_d", "target_geography_scope", "selected_states",
    summary="Forecast a user-supplied subset of states.",
    description=(
        "Like ``all_states`` but restricted to ``leaf_config.target_states "
        "= [postal_codes...]`` or to a named ``fred_sd_state_group`` "
        "(census regions / divisions, BEA regions, etc.)."
    ),
    when_to_use="Region-specific studies (Northeast vs. Midwest), Census-division comparisons.",
    references=(_REF_DESIGN_L1,),
    related_options=("all_states", "fred_sd_state_group"),
)
# Cycle 20 L1.D fix: ParameterDoc for target_states
_L1D_GEO_SELECTED = _L1D_GEO_SELECTED.__class__(
    **{**_L1D_GEO_SELECTED.__dict__,
       "parameters": (
           ParameterDoc(
               name="target_states",
               type="list[str]",
               default=None,
               constraint="non-empty list required; each element a valid US state code or DC",
               description="Explicit target state list when target_geography_scope=selected_states.",
           ),
       ),
    }
)


# ---------------------------------------------------------------------------
# L1.E sample_start_rule / sample_end_rule
# ---------------------------------------------------------------------------

_L1E_START_MAX_BAL = _entry(
    "l1_e", "sample_start_rule", "max_balanced",
    summary="Start at the first date where every requested series is observed.",
    description=(
        "Computes the latest first-observation date across every column "
        "in the panel and trims earlier rows. Guarantees a balanced "
        "panel without imputing leading missing values.\n\n"
        "Default for studies that mix series with different start dates "
        "(common on FRED-MD because some series only begin in the 1980s)."
    ),
    when_to_use="Default for FRED-MD/QD studies with mixed start dates.",
    when_not_to_use="Custom panels where every series shares the same start date (use ``earliest_available`` to keep all rows).",
    references=(_REF_DESIGN_L1,),
    related_options=("earliest_available", "fixed_date"),
)

_L1E_START_EARLIEST = _entry(
    "l1_e", "sample_start_rule", "earliest_available",
    summary="Start at the panel's earliest date; tolerates leading missing values.",
    description=(
        "Keeps every row; lets the L1.C ``raw_missing_policy`` and L2 "
        "imputation handle leading NaNs. Useful when the L2 EM-factor "
        "imputer can recover early observations and dropping them would "
        "lose informative history."
    ),
    when_to_use="Studies that want maximum sample length and trust L2 imputation to handle leading NaNs.",
    references=(_REF_DESIGN_L1,),
    related_options=("max_balanced", "fixed_date"),
)

_L1E_START_FIXED = _entry(
    "l1_e", "sample_start_rule", "fixed_date",
    summary="Pin the start date in leaf_config (e.g., 1985-01-01).",
    description=(
        "Requires ``leaf_config.sample_start_date`` (ISO date). The L1 "
        "loader trims to that date verbatim. Useful for replication "
        "scripts that need an exact sample window matching a published "
        "paper."
    ),
    when_to_use="Replication scripts; ablation studies over alternative start dates.",
    references=(_REF_DESIGN_L1,),
    related_options=("max_balanced", "earliest_available"),
)

_L1E_END_LATEST = _entry(
    "l1_e", "sample_end_rule", "latest_available",
    summary="End at the panel's last date.",
    description=(
        "Default. Uses the most recent observation in the bundled vintage."
    ),
    when_to_use="Default. Studies that want to use the full available history.",
    references=(_REF_DESIGN_L1,),
    related_options=("fixed_date",),
)

_L1E_END_FIXED = _entry(
    "l1_e", "sample_end_rule", "fixed_date",
    summary="Pin the end date in leaf_config (e.g., 2019-12-31).",
    description=(
        "Requires ``leaf_config.sample_end_date``. Useful for "
        "pseudo-out-of-sample evaluation where the recipe wants to "
        "exclude the COVID window or stop at a paper's reported sample."
    ),
    when_to_use="Pre-COVID benchmark studies; matching a paper's reported sample window.",
    references=(_REF_DESIGN_L1,),
    related_options=("latest_available",),
)


# ---------------------------------------------------------------------------
# L1.F horizon_set
# ---------------------------------------------------------------------------

_L1F_HORIZON_STD_MD = _entry(
    "l1_f", "horizon_set", "standard_md",
    summary="Standard FRED-MD horizons: {1, 3, 6, 9, 12, 18, 24} months.",
    description=(
        "The canonical multi-horizon set used in the McCracken-Ng / Stock-Watson "
        "tradition for monthly forecasting. Models are fit per-horizon (when "
        "``forecast_strategy = direct``) and metrics report per-(model, "
        "horizon) rows."
    ),
    when_to_use="Default for monthly studies. Comparable to published monthly benchmarks.",
    references=(_REF_DESIGN_L1, _REF_MCCRACKEN_NG_2016),
    related_options=("standard_qd", "single", "custom_list", "range_up_to_h"),
)

_L1F_HORIZON_STD_QD = _entry(
    "l1_f", "horizon_set", "standard_qd",
    summary="Standard FRED-QD horizons: {1, 2, 4, 8} quarters.",
    description="Quarterly counterpart of ``standard_md``.",
    when_to_use="Default for quarterly (FRED-QD) studies.",
    references=(_REF_DESIGN_L1, _REF_MCCRACKEN_NG_2020),
    related_options=("standard_md", "single", "custom_list"),
)

_L1F_HORIZON_SINGLE = _entry(
    "l1_f", "horizon_set", "single",
    summary="A single horizon (defaults to h=1).",
    description=(
        "Forecasts only one horizon per cell. Sets ``leaf_config.target_horizons "
        "= [N]`` to override the default of 1. Faster than multi-horizon "
        "studies and clearer metrics tables when the study's question is "
        "single-horizon."
    ),
    when_to_use="One-shot studies (h=1 nowcasting, h=12 long-horizon ablation).",
    references=(_REF_DESIGN_L1,),
    related_options=("standard_md", "standard_qd", "custom_list"),
)

_L1F_HORIZON_CUSTOM = _entry(
    "l1_f", "horizon_set", "custom_list",
    summary="User-supplied horizon list (any non-empty integer set).",
    description=(
        "Requires ``leaf_config.target_horizons: [int...]``. Useful for "
        "non-standard horizon comparisons (e.g., {1, 2, 3, 6, 12} or "
        "{6, 12, 24, 36})."
    ),
    when_to_use="Replication of papers with non-standard horizon sets; ablation studies.",
    references=(_REF_DESIGN_L1,),
    related_options=("standard_md", "standard_qd", "range_up_to_h"),
)

_L1F_HORIZON_RANGE = _entry(
    "l1_f", "horizon_set", "range_up_to_h",
    summary="Every horizon from 1 to leaf_config.max_horizon (inclusive).",
    description=(
        "Equivalent to ``custom_list`` with ``[1, 2, ..., max_horizon]``. "
        "Useful for direct-h forecasting where the study wants dense "
        "horizon coverage (e.g., 1-12 months)."
    ),
    when_to_use="Dense horizon studies with direct-h forecasting.",
    references=(_REF_DESIGN_L1,),
    related_options=("custom_list", "standard_md"),
)


# ---------------------------------------------------------------------------
# L1.G regime_definition
# ---------------------------------------------------------------------------

_L1G_REGIME_NONE = _entry(
    "l1_g", "regime_definition", "none",
    summary="No regime structure -- the cell-loop sees a single time-invariant world.",
    description=(
        "Default. The L1 regime metadata sink is empty; downstream layers "
        "that conditionally reference regime info (L4 regime_wrapper, L5 "
        "by_regime decomposition, L6.C cpa conditioning) are inactive."
    ),
    when_to_use="Default for any study without an explicit regime structure.",
    references=(_REF_DESIGN_L1,),
    related_options=("external_nber", "external_user_provided", "estimated_markov_switching"),
)

_L1G_REGIME_NBER = _entry(
    "l1_g", "regime_definition", "external_nber",
    summary="NBER recession dates loaded from the bundled USREC series.",
    description=(
        "Loads the NBER official recession indicator (USREC) and emits a "
        "two-state regime label ('expansion' / 'recession') per "
        "observation. No estimation -- the labels come directly from "
        "NBER's published business-cycle dates.\n\n"
        "Pairs with L4 ``regime_wrapper = separate_fit`` for "
        "regime-conditional forecasting and with L5 ``by_regime`` "
        "decomposition for state-dependent metrics."
    ),
    when_to_use="Standard recession-conditioning studies; comparing models' performance during recessions vs expansions.",
    references=(_REF_DESIGN_L1, _REF_NBER),
    related_options=("none", "external_user_provided"),
    examples=(
        CodeExample(
            title="NBER-conditioned ridge",
            code=(
                "1_data:\n"
                "  fixed_axes:\n"
                "    regime_definition: external_nber\n"
            ),
        ),
    ),
)

_L1G_REGIME_USER = _entry(
    "l1_g", "regime_definition", "external_user_provided",
    summary="User-supplied regime label series.",
    description=(
        "Requires ``leaf_config.regime_indicator_path`` (CSV / Parquet "
        "with date + label columns) **or** ``leaf_config.regime_dates_list`` "
        "(inline date ranges per label). The L1 runtime aligns the labels "
        "to the panel index without estimation.\n\n"
        "Useful for custom regime definitions: monetary-policy-stance "
        "labels, fiscal-stance labels, geographic-dummy regimes, etc."
    ),
    when_to_use="Custom regime studies (monetary stance, fiscal stance, alternative recession-dating chronologies).",
    references=(_REF_DESIGN_L1,),
    related_options=("none", "external_nber"),
)

_L1G_REGIME_MS = _entry(
    "l1_g", "regime_definition", "estimated_markov_switching",
    summary="Hamilton (1989) Markov-switching regime estimated from the target.",
    description=(
        "Fits ``statsmodels.tsa.regime_switching.MarkovRegression`` with "
        "``leaf_config.n_regimes`` (default 2) regimes and switching variance. "
        "Returns smoothed posterior probabilities, the argmax label per "
        "observation, and the estimated transition matrix.\n\n"
        "The estimation is per-origin (no leakage) via "
        "``regime_estimation_temporal_rule``: ``expanding_window_per_origin`` "
        "(default) or ``rolling_window_per_origin`` / ``block_recompute``."
    ),
    when_to_use="Inflation-regime studies; recession-prediction; any study that wants endogenous regime estimation.",
    references=(_REF_DESIGN_L1, _REF_HAMILTON_1989),
    related_options=("estimated_threshold", "estimated_structural_break", "regime_estimation_temporal_rule", "n_regimes"),
)

_L1G_REGIME_THRESH = _entry(
    "l1_g", "regime_definition", "estimated_threshold",
    summary="Tong (1990) SETAR threshold model -- regime by threshold variable.",
    description=(
        "Estimates ``n_regimes`` regimes by grid-searching the threshold "
        "on a chosen ``threshold_variable`` (defaults to lagged target). "
        "The regime per observation is determined by which side of the "
        "threshold the variable falls on.\n\n"
        "Tong's original SETAR uses AR(p) per regime; macroforecast's runtime "
        "fits AR(``threshold_ar_p``) per partition and selects the "
        "threshold minimising joint SSR."
    ),
    when_to_use="Self-exciting threshold dynamics (e.g., interest-rate stance regimes); non-linear AR studies.",
    references=(_REF_DESIGN_L1, _REF_TONG_1990),
    related_options=("estimated_markov_switching", "estimated_structural_break"),
)

_L1G_REGIME_BREAK = _entry(
    "l1_g", "regime_definition", "estimated_structural_break",
    summary="Bai-Perron (1998) global LSE break detection.",
    description=(
        "Detects up to ``leaf_config.max_breaks`` structural breaks via "
        "the Bai (1997) dynamic-programming exact recursion + BIC selection. "
        "Each segment between breaks becomes a regime."
    ),
    when_to_use="Studies hypothesising structural change (e.g., pre/post Volcker, pre/post Great Moderation).",
    references=(_REF_DESIGN_L1, _REF_BAI_PERRON_1998),
    related_options=("estimated_markov_switching", "estimated_threshold", "max_breaks"),
)


# ---------------------------------------------------------------------------
# Long-tail axes (Tier-1 entries; one helper that records summary +
# description + when_to_use + references + last_reviewed for every
# remaining option).
# ---------------------------------------------------------------------------

def _t1(sublayer: str, axis: str, option: str,
        summary: str, description: str, when_to_use: str,
        *, when_not_to_use: str = "",
        related: tuple[str, ...] = (),
        references: tuple[Reference, ...] = (_REF_DESIGN_L1,)) -> OptionDoc:
    return OptionDoc(
        layer="l1", sublayer=sublayer, axis=axis, option=option,
        summary=summary, description=description, when_to_use=when_to_use,
        when_not_to_use=when_not_to_use, related_options=related,
        references=references,
        last_reviewed="2026-05-05", reviewer="macroforecast author",
    )


# L1.A fred_sd_frequency_policy
_L1A_FRED_SD_FREQ = (
    _entry(
        "l1_a", "fred_sd_frequency_policy", "report_only",
        summary="Log frequency mismatches in manifest; do not gate execution.",
        description=(
            "Default policy. When the FRED-SD pull contains variables with "
            "differing declared frequencies (e.g., monthly QCEW payroll "
            "series alongside quarterly income data), the runtime logs a "
            "diagnostic entry in the L1.5 manifest but allows the panel "
            "to proceed unchanged. No records are dropped and no error is "
            "raised.\n\n"
            "Alignment of the mixed-frequency panel is deferred entirely to "
            "L2.A (frequency-alignment rules). This is the appropriate "
            "choice for exploratory or default pipelines where the recipe "
            "author has not yet decided how to handle the frequency "
            "mismatch -- the manifest diagnostic surfaces the issue "
            "without blocking execution."
        ),
        when_to_use=(
            "Default for most FRED-SD recipes; exploratory work where "
            "mixed-frequency status should be visible in the manifest but "
            "should not stop execution; any pipeline that handles alignment "
            "in L2.A."
        ),
        when_not_to_use=(
            "Recipes that require strict frequency homogeneity -- use "
            "``reject_mixed_known_frequency`` or "
            "``require_single_known_frequency`` to gate early."
        ),
        references=(_REF_FRED_SD,),
        related_options=(
            "allow_mixed_frequency",
            "reject_mixed_known_frequency",
            "require_single_known_frequency",
            "mixed_frequency_representation",
        ),
        last_reviewed="2026-05-16",
        reviewer="macroforecast author",
    ),
    _entry(
        "l1_a", "fred_sd_frequency_policy", "allow_mixed_frequency",
        summary="Explicitly permit mixed frequencies; downstream layers must handle alignment.",
        description=(
            "Records an explicit recipe-author decision to accept a "
            "mixed-frequency FRED-SD panel. Unlike ``report_only``, "
            "selecting this option signals to downstream layers (L2.A "
            "frequency-alignment, L3 feature DAG) that the mixed-frequency "
            "structure is intentional and should be handled -- not silently "
            "passed through.\n\n"
            "The actual frequency-alignment logic (e.g., temporal "
            "aggregation of monthly series to quarterly, or Kalman-filter "
            "mixed-frequency representation) is delegated to "
            "``mixed_frequency_representation`` in L2.A. Use this option "
            "when the recipe is designed to combine monthly FRED-SD "
            "predictors (e.g., QCEW payrolls) with quarterly FRED-SD "
            "targets (e.g., GSP) and an explicit alignment strategy is "
            "configured in L2."
        ),
        when_to_use=(
            "Standard FRED-SD pipelines that intentionally combine monthly "
            "and quarterly state series; recipes where "
            "``mixed_frequency_representation`` is configured in L2.A."
        ),
        when_not_to_use=(
            "Pipelines that want to hard-reject mixed frequencies rather "
            "than align them -- use ``reject_mixed_known_frequency``."
        ),
        references=(_REF_FRED_SD,),
        related_options=(
            "report_only",
            "reject_mixed_known_frequency",
            "require_single_known_frequency",
            "mixed_frequency_representation",
        ),
        last_reviewed="2026-05-16",
        reviewer="macroforecast author",
    ),
    _entry(
        "l1_a", "fred_sd_frequency_policy", "reject_mixed_known_frequency",
        summary="Hard-reject if pulled variables span more than one declared frequency.",
        description=(
            "Safety gate: raises a ``ValueError`` at L1 validation if any "
            "two variables in the FRED-SD pull carry different *known* "
            "frequency declarations (e.g., one series is declared monthly "
            "and another declared quarterly). Variables with unknown "
            "frequency (i.e., series for which FRED-SD does not declare a "
            "frequency) are tolerated -- only explicit mismatches between "
            "known frequencies trigger the error.\n\n"
            "Useful when the recipe author expects a single-frequency panel "
            "and wants to fail loudly if FRED-SD upstream changes (new "
            "series additions, metadata corrections) introduce an unexpected "
            "frequency mix. The error message names the conflicting series "
            "and their declared frequencies."
        ),
        when_to_use=(
            "Defensive recipes where frequency homogeneity is part of the "
            "study design; CI checks that should fail loudly if new FRED-SD "
            "series at a different frequency are inadvertently pulled."
        ),
        when_not_to_use=(
            "Pipelines designed to work with mixed frequencies (use "
            "``allow_mixed_frequency``); pipelines where unknown-frequency "
            "series should also be rejected (use "
            "``require_single_known_frequency``)."
        ),
        references=(_REF_FRED_SD,),
        related_options=(
            "report_only",
            "allow_mixed_frequency",
            "require_single_known_frequency",
        ),
        last_reviewed="2026-05-16",
        reviewer="macroforecast author",
    ),
    _entry(
        "l1_a", "fred_sd_frequency_policy", "require_single_known_frequency",
        summary="Enforce single frequency; reject if any variable has unknown or differing frequency.",
        description=(
            "Strictest setting. The L1 gate passes only if every variable "
            "in the FRED-SD pull (a) has a declared known frequency, and "
            "(b) all declared frequencies are identical. Two distinct "
            "failure modes raise ``ValueError``:\n\n"
            "1. A variable carries frequency ``'unknown'`` in FRED-SD "
            "   metadata -- this would pass ``reject_mixed_known_frequency`` "
            "   but fails here.\n"
            "2. Two or more variables carry different *known* frequencies "
            "   (same condition as ``reject_mixed_known_frequency``).\n\n"
            "This is the appropriate gate for strictly mono-frequency studies "
            "(e.g., monthly-only payroll analyses) that must also enforce "
            "that all series have a documented cadence -- no 'we don't know "
            "the frequency' series are permitted."
        ),
        when_to_use=(
            "Strictly mono-frequency studies (e.g., monthly-only state-level "
            "employment analyses); pipelines that must guarantee every "
            "series has a known frequency declaration in the FRED-SD "
            "metadata."
        ),
        when_not_to_use=(
            "Pipelines that include FRED-SD series with undeclared "
            "frequencies (use ``reject_mixed_known_frequency`` to only block "
            "explicit mismatches, not unknown frequencies); mixed-frequency "
            "pipelines (use ``allow_mixed_frequency``)."
        ),
        references=(_REF_FRED_SD,),
        related_options=(
            "report_only",
            "allow_mixed_frequency",
            "reject_mixed_known_frequency",
        ),
        last_reviewed="2026-05-16",
        reviewer="macroforecast author",
    ),
)

# L1.C missing_availability
_L1C_MISSING_IMPUTE_PREDICTORS = _t1(
    "l1_c", "missing_availability", "impute_predictors_only",
    "Impute predictor missings at L1; never impute the target.",
    "Restricts imputation to the predictor block at L1 stage and forbids any target imputation in subsequent layers. Avoids accidentally back-filling the target via L2.D.",
    "Recipes where the target should be the ground-truth signal and never imputed.",
    related=("keep_available_rows",),
)
_L1C_MISSING_IMPUTE_PREDICTORS = _L1C_MISSING_IMPUTE_PREDICTORS.__class__(
    **{**_L1C_MISSING_IMPUTE_PREDICTORS.__dict__,
       "parameters": (
           ParameterDoc(
               name="x_imputation",
               type="str",
               default=None,
               constraint="required; one of ['bfill', 'ffill', 'mean', 'median'].",
               description=(
                   "Imputation method applied to predictor missings at L1. Used only when "
                   "missing_availability=impute_predictors_only."
               ),
           ),
       ),
    }
)
_L1C_MISSING = (
    _t1("l1_c", "missing_availability", "require_complete_rows",
        "Drop any row containing a missing value.",
        "Strict listwise-deletion rule applied at L1 before L2 imputation. Useful when the recipe author prefers to lose rows rather than rely on imputation; produces a smaller, fully-observed panel.",
        "Studies where imputation is methodologically inappropriate; sensitivity analyses against imputation effects.",
        when_not_to_use="When the panel is sparsely observed -- listwise deletion can leave too few rows.",
        related=("keep_available_rows", "impute_predictors_only")),
    _t1("l1_c", "missing_availability", "keep_available_rows",
        "Keep every row that has the target observed.",
        "Default; passes interior predictor NaNs through to L2.D for imputation. Ensures the maximum sample size while letting downstream imputation handle holes.",
        "Default for FRED-MD / -QD recipes where L2.D EM imputation is the canonical workflow.",
        related=("require_complete_rows", "impute_predictors_only")),
    _L1C_MISSING_IMPUTE_PREDICTORS,
    _t1("l1_c", "missing_availability", "zero_fill_leading_predictor_gaps",
        "Zero-fill leading predictor NaNs; preserve interior gaps.",
        "Replaces leading NaNs (before the predictor's first observation) with zero so the panel has a uniform start date. Interior NaNs pass through to L2.D unchanged.",
        "FRED-SD panels where some series start later but the user wants a balanced start date.",
        when_not_to_use="When zero is a meaningful value for the predictor -- choose ``preserve_raw_missing`` instead.",
        related=("require_complete_rows",)),
)

# L1.C raw_missing_policy
_L1C_RAW_MISSING_IMPUTE = _t1(
    "l1_c", "raw_missing_policy", "impute_raw_predictors",
    "Impute raw predictor NaNs at L1 (before any L2 stage).",
    "Runs a simple per-series imputation (mean / median / forward-fill) at L1. Useful when L2.D is disabled or when the user wants to pre-clean raw data before the t-code stage.",
    "Pipelines that use ``no_transform`` t-codes and need cleaning at L1.",
    related=("preserve_raw_missing", "drop_raw_missing_rows"),
)
_L1C_RAW_MISSING_IMPUTE = _L1C_RAW_MISSING_IMPUTE.__class__(
    **{**_L1C_RAW_MISSING_IMPUTE.__dict__,
       "parameters": (
           ParameterDoc(
               name="raw_x_imputation",
               type="str",
               default=None,
               constraint="required; one of ['bfill', 'ffill', 'mean', 'median'].",
               description=(
                   "Imputation method applied to raw predictor NaNs at L1. Used only when "
                   "raw_missing_policy=impute_raw_predictors."
               ),
           ),
       ),
    }
)
_L1C_RAW_MISSING = (
    _t1("l1_c", "raw_missing_policy", "preserve_raw_missing",
        "Pass raw NaN values through unchanged.",
        "Default; raw missingness flows into L2.D imputation. Required for the McCracken-Ng EM-factor imputation workflow. "
        "See also: L2 ``imputation_policy`` (same surface, different stage: raw vs post-tcode).",
        "Default; required when L2.D will run EM-factor or similar global imputation.",
        related=("zero_fill_leading_predictor_missing_before_tcode", "impute_raw_predictors", "drop_raw_missing_rows")),
    _t1("l1_c", "raw_missing_policy", "zero_fill_leading_predictor_missing_before_tcode",
        "Zero-fill leading predictor NaNs prior to t-code application.",
        "Important for level-difference t-codes that fail when leading NaNs are interspersed with observed values. The zero-fill creates a clean prefix for differencing.",
        "Tcode 1 / 2 / 5 / 6 pipelines where leading NaNs would propagate after differencing.",
        when_not_to_use="When zero is a meaningful value for the predictor.",
        related=("preserve_raw_missing",)),
    _L1C_RAW_MISSING_IMPUTE,
    _t1("l1_c", "raw_missing_policy", "drop_raw_missing_rows",
        "Drop rows containing any raw missing predictor.",
        "Aggressive listwise deletion at the raw stage. Reduces panel size before any cleaning runs.",
        "Sensitivity analyses; sanity checks against imputation effects.",
        when_not_to_use="When the panel is small -- you'll lose a lot of rows.",
        related=("preserve_raw_missing",)),
)

# L1.C raw_outlier_policy
_L1C_RAW_OUTLIER_PRESERVE = _t1(
    "l1_c", "raw_outlier_policy", "preserve_raw_outliers",
    "Pass raw outliers through to L2.C.",
    "Default; relies on L2.C McCracken-Ng IQR detection and the configured ``outlier_action`` to handle "
    "extreme values. See also: L2 ``outlier_policy`` / ``outlier_action`` (same surface, different stage: "
    "raw vs post-tcode).",
    "Default; the canonical workflow.",
    related=("winsorize_raw", "iqr_clip_raw", "mad_clip_raw", "zscore_clip_raw", "set_raw_outliers_to_missing"),
)
_L1C_RAW_OUTLIER_WINSORIZE = _t1(
    "l1_c", "raw_outlier_policy", "winsorize_raw",
    "Winsorise raw series at quantile cutpoints (default p1 / p99).",
    "Caps extreme values at the specified quantile before t-coding. Preserves observation count but compresses "
    "tails. Configured via ``leaf_config.winsorize_quantiles`` (default [0.01, 0.99]). Compare: L2 "
    "``outlier_policy=winsorize`` operates on the post-tcode panel.",
    "Heavy-tailed financial / macro series where extreme observations would dominate downstream estimates.",
    related=("preserve_raw_outliers", "iqr_clip_raw"),
)
_L1C_RAW_OUTLIER_WINSORIZE = _L1C_RAW_OUTLIER_WINSORIZE.__class__(
    **{**_L1C_RAW_OUTLIER_WINSORIZE.__dict__,
       "parameters": (
           ParameterDoc(
               name="winsorize_quantiles",
               type="list[float, float]",
               default="[0.01, 0.99]",
               constraint="0 <= low < high <= 1; both elements required.",
               description=(
                   "Lower and upper quantile clip thresholds. Defaults to symmetric 1%/99% winsorization. "
                   "Values outside [low, high] quantile bounds are clipped to the bound value."
               ),
           ),
       ),
    }
)
_L1C_RAW_OUTLIER_IQR = _t1(
    "l1_c", "raw_outlier_policy", "iqr_clip_raw",
    "Clip raw observations beyond k×IQR thresholds.",
    "Clips values outside ``Q1 - k·IQR``, ``Q3 + k·IQR`` (k default 10.0, matching McCracken-Ng). Robust to "
    "non-Gaussian distributions. Configured via ``leaf_config.outlier_iqr_threshold``. Compare: L2 "
    "``outlier_policy=mccracken_ng_iqr`` uses the same k but on the post-tcode panel.",
    "Robust outlier handling on non-normal series.",
    related=("winsorize_raw", "mad_clip_raw", "zscore_clip_raw"),
)
_L1C_RAW_OUTLIER_IQR = _L1C_RAW_OUTLIER_IQR.__class__(
    **{**_L1C_RAW_OUTLIER_IQR.__dict__,
       "parameters": (
           ParameterDoc(
               name="outlier_iqr_threshold",
               type="float",
               default="10.0",
               constraint=">0",
               description=(
                   "IQR multiplier above which raw observations are clipped. McCracken-Ng default is 10.0. "
                   "Observations satisfying |x - median| > k * IQR are clipped to the band boundary."
               ),
           ),
       ),
    }
)
_L1C_RAW_OUTLIER_MAD = _t1(
    "l1_c", "raw_outlier_policy", "mad_clip_raw",
    "Clip raw observations beyond k×MAD thresholds.",
    "Median Absolute Deviation -based clipping; even more robust than IQR. Default k = 3 maps to roughly 3σ "
    "for normal data.",
    "Highly non-Gaussian series with sparse outliers.",
    related=("iqr_clip_raw", "zscore_clip_raw"),
)
_L1C_RAW_OUTLIER_ZSCORE = _t1(
    "l1_c", "raw_outlier_policy", "zscore_clip_raw",
    "Clip raw observations beyond k standard deviations.",
    "Standard z-score rule (typically k = 3). Cheapest option but assumes approximate normality. Configured via "
    "``leaf_config.zscore_threshold_value``.",
    "Approximately Gaussian series; quick baseline.",
    when_not_to_use="Heavy-tailed series -- use ``iqr_clip_raw`` or ``mad_clip_raw``.",
    related=("iqr_clip_raw", "mad_clip_raw"),
)
_L1C_RAW_OUTLIER_ZSCORE = _L1C_RAW_OUTLIER_ZSCORE.__class__(
    **{**_L1C_RAW_OUTLIER_ZSCORE.__dict__,
       "parameters": (
           ParameterDoc(
               name="zscore_threshold_value",
               type="float",
               default="3.0",
               constraint=">0",
               description=(
                   "Z-score threshold; observations with |z| > threshold are clipped to the threshold boundary. "
                   "z is computed as (x - mean) / std over the series."
               ),
           ),
       ),
    }
)
_L1C_RAW_OUTLIER_SET_MISSING = _t1(
    "l1_c", "raw_outlier_policy", "set_raw_outliers_to_missing",
    "Set raw outliers to NaN and defer to L2.D imputation.",
    "Replaces flagged outliers with NaN. The L2.D imputation method then fills the resulting gaps; preserves "
    "observation count for downstream stages.",
    "Pipelines where outliers should be re-imputed coherently with other missing data.",
    related=("preserve_raw_outliers", "winsorize_raw"),
)
_L1C_RAW_OUTLIER = (
    _L1C_RAW_OUTLIER_PRESERVE,
    _L1C_RAW_OUTLIER_WINSORIZE,
    _L1C_RAW_OUTLIER_IQR,
    _L1C_RAW_OUTLIER_MAD,
    _L1C_RAW_OUTLIER_ZSCORE,
    _L1C_RAW_OUTLIER_SET_MISSING,
)

# L1.C release_lag_rule
_L1C_RELEASE_LAG_IGNORE = _t1(
    "l1_c", "release_lag_rule", "ignore_release_lag",
    "Treat every observation as available at its calendar period.",
    "Pseudo-real-time mode: ignores the release-lag distinction; every variable is assumed to be available the "
    "moment the period closes.",
    "Backtests where real-time vintage data is unavailable.",
    related=("fixed_lag_all_series", "series_specific_lag"),
)
_L1C_RELEASE_LAG_FIXED = _t1(
    "l1_c", "release_lag_rule", "fixed_lag_all_series",
    "Apply a single release lag to every series.",
    "All series shift by ``leaf_config.fixed_lag_periods`` periods. Approximates real-time availability without "
    "per-series detail.",
    "Coarse real-time approximations.",
    related=("series_specific_lag",),
)
_L1C_RELEASE_LAG_FIXED = _L1C_RELEASE_LAG_FIXED.__class__(
    **{**_L1C_RELEASE_LAG_FIXED.__dict__,
       "parameters": (
           ParameterDoc(
               name="fixed_lag_periods",
               type="int",
               default=None,
               constraint=">=0; optional; defaults to 0 if not set.",
               description=(
                   "Uniform release lag in periods applied to every predictor series. A value of 1 means each "
                   "series is available one period after the period it was observed."
               ),
           ),
       ),
    }
)
_L1C_RELEASE_LAG_SPECIFIC = _t1(
    "l1_c", "release_lag_rule", "series_specific_lag",
    "Use per-series release lags from leaf_config.",
    "Honours the published release-lag table in ``leaf_config.release_lag_per_series``. Most accurate option "
    "for true real-time studies.",
    "Real-time / nowcasting studies that respect publication delays.",
    related=("fixed_lag_all_series",),
)
_L1C_RELEASE_LAG_SPECIFIC = _L1C_RELEASE_LAG_SPECIFIC.__class__(
    **{**_L1C_RELEASE_LAG_SPECIFIC.__dict__,
       "parameters": (
           ParameterDoc(
               name="release_lag_per_series",
               type="dict[str, int]",
               default=None,
               constraint="Required when release_lag_rule=series_specific_lag; non-empty dict.",
               description=(
                   "Per-series release lag in periods. Maps series name to a non-negative integer. "
                   "Series not present in the dict are treated as zero-lag (available immediately)."
               ),
           ),
       ),
    }
)
_L1C_RELEASE_LAG = (
    _L1C_RELEASE_LAG_IGNORE,
    _L1C_RELEASE_LAG_FIXED,
    _L1C_RELEASE_LAG_SPECIFIC,
)

# L1.C contemporaneous_x_rule
_L1C_CONTEMP = (
    _t1("l1_c", "contemporaneous_x_rule", "allow_same_period_predictors",
        "Permit predictors observed in the same period as the target.",
        "Default; predictor ``x_t`` and target ``y_t`` are both available at time t. Used for nowcasting where contemporaneous information is exploited.",
        "Default; standard fitting / nowcasting flow.",
        related=("forbid_same_period_predictors",)),
    _t1("l1_c", "contemporaneous_x_rule", "forbid_same_period_predictors",
        "Require predictors to be at least one period stale.",
        "Forces predictors to be lagged ``y_t`` is forecast from ``x_{t-1}, x_{t-2}, ...``. Cleanest causal interpretation.",
        "Pure forecasting setups where contemporaneous information would create look-ahead.",
        related=("allow_same_period_predictors",)),
)

# L1.C official_transform_scope
_L1C_OFFICIAL_SCOPE = (
    _t1("l1_c", "official_transform_scope", "target_only",
        "Apply official t-codes only to the target column.",
        "Restricts McCracken-Ng tcode application to ``y``. Predictors flow through untransformed.",
        "When predictors are already pre-transformed.",
        related=("predictors_only", "target_and_predictors", "none")),
    _t1("l1_c", "official_transform_scope", "predictors_only",
        "Apply official t-codes only to predictor columns.",
        "Used when the user supplies an externally-transformed target.",
        "When the target is pre-engineered (e.g. growth rate).",
        related=("target_only", "target_and_predictors")),
    _t1("l1_c", "official_transform_scope", "target_and_predictors",
        "Apply official t-codes to both target and predictors.",
        "Default; canonical McCracken-Ng workflow.",
        "Default; FRED-MD / -QD recipes.",
        related=("target_only", "predictors_only", "none")),
    _t1("l1_c", "official_transform_scope", "none",
        "Skip official t-codes entirely.",
        "Disables L1's official tcode application. Used together with ``transform_policy = no_transform`` or ``custom_tcode``.",
        "Custom panels with bespoke transforms.",
        related=("target_and_predictors",)),
)

# L1.D predictor_geography_scope (FRED-SD)
_L1D_PRED_GEO = (
    _t1("l1_d", "predictor_geography_scope", "match_target",
        "Use the same geography scope as the target.",
        "Default; predictor states match the L1.D ``target_geography_scope``. Ensures spatial coherence for state-level forecasts.",
        "Default for state-level forecasts.",
        related=("all_states", "selected_states", "national_only")),
    _t1("l1_d", "predictor_geography_scope", "all_states",
        "Use predictors from every state regardless of target geography.",
        "All-50-states predictor block. Useful when cross-state spillovers matter and the target is a single state.",
        "Spillover / cross-state interaction studies.",
        related=("match_target", "selected_states")),
    _t1("l1_d", "predictor_geography_scope", "selected_states",
        "Use predictors from a user-supplied state list.",
        "Reads ``leaf_config.predictor_states`` and restricts the predictor block to that subset.",
        "Custom regional studies (e.g. neighbouring states).",
        related=("match_target", "all_states")),
    _t1("l1_d", "predictor_geography_scope", "national_only",
        "Use only national-aggregate predictors.",
        "Strips state-level predictors and keeps only national series. Reduces panel dimension when state-level features are noise.",
        "When national variables alone explain target variation.",
        when_not_to_use="State-level forecasts where regional predictors carry signal.",
        related=("all_states", "match_target")),
)
# Cycle 20 L1.D fix: ParameterDoc for predictor_states
_L1D_PRED_GEO_SELECTED = next(
    d for d in _L1D_PRED_GEO
    if d.axis == "predictor_geography_scope" and d.option == "selected_states"
)
_L1D_PRED_GEO_SELECTED_PATCHED = _L1D_PRED_GEO_SELECTED.__class__(
    **{**_L1D_PRED_GEO_SELECTED.__dict__,
       "parameters": (
           ParameterDoc(
               name="predictor_states",
               type="list[str]",
               default=None,
               constraint="non-empty list required",
               description="Explicit predictor state list. Independent of target_states; permits cross-state-pair studies.",
           ),
       ),
    }
)
_L1D_PRED_GEO = tuple(
    _L1D_PRED_GEO_SELECTED_PATCHED if d.option == "selected_states" else d
    for d in _L1D_PRED_GEO
)

# L1.D fred_sd_state_group (16 Census Bureau region / division groupings)
_REF_CENSUS_REGIONS = Reference(
    citation="US Census Bureau (2020) 'Geographic Levels: Regions and Divisions', US Census Bureau Geography Division.",
    url="https://www.census.gov/programs-surveys/economic-census/guidance-geographies/levels.html",
)

_STATE_GROUP_DOCS: dict[str, tuple[str, str, str]] = {
    "all_states": (
        "All 50 states + DC (51 jurisdictions).",
        (
            "Default. Includes every US state and the District of "
            "Columbia. Use as the broadest possible FRED-SD panel; "
            "subset thereafter via state_selection if specific "
            "filtering is needed."
        ),
        "Default; comprehensive 51-jurisdiction panel.",
    ),
    "census_region_northeast": (
        "Census Northeast Region (9 states): CT, ME, MA, NH, NJ, NY, PA, RI, VT.",
        (
            "Census Bureau's Region 1. Combines New England (CT, ME, "
            "MA, NH, RI, VT) and Mid-Atlantic (NJ, NY, PA) divisions. "
            "Heavily-populated, services-dominated regional economy."
        ),
        "Northeastern regional studies; comparing services-heavy economies.",
    ),
    "census_region_midwest": (
        "Census Midwest Region (12 states): IL, IN, IA, KS, MI, MN, MO, NE, ND, OH, SD, WI.",
        (
            "Census Bureau's Region 2. Combines East North Central (IL, "
            "IN, MI, OH, WI) and West North Central (IA, KS, MN, MO, "
            "NE, ND, SD) divisions. Manufacturing-heavy 'Rust Belt' + "
            "agricultural Plains economies."
        ),
        "Manufacturing-belt and Plains regional studies.",
    ),
    "census_region_south": (
        "Census South Region (16 states + DC): AL, AR, DE, DC, FL, GA, KY, LA, MD, MS, NC, OK, SC, TN, TX, VA, WV.",
        (
            "Census Bureau's Region 3. Combines South Atlantic (DE, "
            "DC, FL, GA, MD, NC, SC, VA, WV), East South Central (AL, "
            "KY, MS, TN), and West South Central (AR, LA, OK, TX) "
            "divisions. Largest Census region by population; mix of "
            "energy (TX, LA, OK) and Sun Belt service economies."
        ),
        "Southern regional studies; Sun Belt vs Rust Belt comparisons.",
    ),
    "census_region_west": (
        "Census West Region (13 states): AK, AZ, CA, CO, HI, ID, MT, NV, NM, OR, UT, WA, WY.",
        (
            "Census Bureau's Region 4. Combines Mountain (AZ, CO, ID, "
            "MT, NV, NM, UT, WY) and Pacific (AK, CA, HI, OR, WA) "
            "divisions. Tech-heavy Pacific Coast + commodity / "
            "tourism Mountain economies."
        ),
        "Pacific Coast tech and Mountain West commodity studies.",
    ),
    "census_division_new_england": (
        "Census New England Division (6 states): CT, ME, MA, NH, RI, VT.",
        (
            "Census Bureau's Division 1. Tight-knit historical "
            "region with finance / education / biotech "
            "concentration."
        ),
        "Finance / education / biotech regional studies.",
    ),
    "census_division_middle_atlantic": (
        "Census Middle Atlantic Division (3 states): NJ, NY, PA.",
        (
            "Census Bureau's Division 2. Hosts the New York "
            "metropolitan financial centre; largest population "
            "Census division."
        ),
        "Financial-centre regional studies (NY metro).",
    ),
    "census_division_east_north_central": (
        "Census East North Central Division (5 states): IL, IN, MI, OH, WI.",
        (
            "Census Bureau's Division 3. Great Lakes manufacturing "
            "belt; the historical 'Industrial Heartland' of the US."
        ),
        "Manufacturing / Rust Belt regional studies.",
    ),
    "census_division_west_north_central": (
        "Census West North Central Division (7 states): IA, KS, MN, MO, NE, ND, SD.",
        (
            "Census Bureau's Division 4. Agricultural Great Plains "
            "with grain / livestock concentration."
        ),
        "Agricultural / commodity regional studies.",
    ),
    "census_division_south_atlantic": (
        "Census South Atlantic Division (8 states + DC): DE, DC, FL, GA, MD, NC, SC, VA, WV.",
        (
            "Census Bureau's Division 5. Atlantic Seaboard from "
            "Delaware to Florida; mix of government (DC, VA), tech "
            "(NC, MD), and Sun Belt service economies (FL, GA)."
        ),
        "Atlantic Seaboard regional studies.",
    ),
    "census_division_east_south_central": (
        "Census East South Central Division (4 states): AL, KY, MS, TN.",
        (
            "Census Bureau's Division 6. Tennessee Valley region; "
            "automotive-supplier and traditional manufacturing "
            "concentration."
        ),
        "Tennessee Valley / Auto-Alley regional studies.",
    ),
    "census_division_west_south_central": (
        "Census West South Central Division (4 states): AR, LA, OK, TX.",
        (
            "Census Bureau's Division 7. Energy-dominated regional "
            "economy (TX, LA, OK oil & gas)."
        ),
        "Energy-sector regional studies.",
    ),
    "census_division_mountain": (
        "Census Mountain Division (8 states): AZ, CO, ID, MT, NV, NM, UT, WY.",
        (
            "Census Bureau's Division 8. Mountain West; mining, "
            "tourism (NV, CO, UT), and tech-corridor (CO, UT) "
            "economies."
        ),
        "Mountain West regional studies.",
    ),
    "census_division_pacific": (
        "Census Pacific Division (5 states): AK, CA, HI, OR, WA.",
        (
            "Census Bureau's Division 9. Pacific Coast tech "
            "concentration (CA, WA, OR) + non-contiguous states "
            "(AK, HI)."
        ),
        "Pacific Coast tech and non-contiguous-state studies.",
    ),
    "contiguous_48_plus_dc": (
        "Contiguous 48 states + DC (excludes AK, HI).",
        (
            "Drops Alaska and Hawaii from the all-states panel. "
            "Useful when the analysis assumes a contiguous geographic "
            "structure (e.g. spatial econometrics with adjacency "
            "weights)."
        ),
        "Continental US studies; spatial econometrics with adjacency matrices.",
    ),
    "custom_state_group": (
        "User-supplied state list (leaf_config.custom_state_list).",
        (
            "Bespoke regional groupings -- e.g. 'oil-producing states' "
            "(TX, OK, ND, NM, LA), 'eurozone-equivalent BEA regions', "
            "or 'states with right-to-work laws'. Reads the explicit "
            "state list from ``leaf_config.custom_state_list``."
        ),
        "Bespoke regional groupings not captured by Census definitions.",
    ),
}
_L1D_STATE_GROUP = tuple(
    _t1("l1_d", "fred_sd_state_group", option, summary,
        (
            f"FRED-SD state grouping: {desc}\n\n"
            f"This option selects which state-level series enter the "
            f"predictor / target panels. The grouping does not affect "
            f"national-aggregate variables; combine with "
            f"``predictor_geography_scope`` to control whether "
            f"predictors follow the target's geographic scope or use "
            f"a different state set."
        ),
        when,
        related=tuple(k for k in _STATE_GROUP_DOCS if k != option)[:3],
        references=(_REF_DESIGN_L1, _REF_MCCRACKEN_NG_2020, _REF_CENSUS_REGIONS))
    for option, (summary, desc, when) in _STATE_GROUP_DOCS.items()
)
# Cycle 20 L1.D fix: ParameterDoc for custom_state_group (OR: sd_state_group_members / sd_state_groups)
_L1D_STATE_GROUP_CUSTOM = next(
    d for d in _L1D_STATE_GROUP
    if d.axis == "fred_sd_state_group" and d.option == "custom_state_group"
)
_L1D_STATE_GROUP_CUSTOM_PATCHED = _L1D_STATE_GROUP_CUSTOM.__class__(
    **{**_L1D_STATE_GROUP_CUSTOM.__dict__,
       "parameters": (
           ParameterDoc(
               name="sd_state_group_members",
               type="list[str]",
               default=None,
               constraint="exactly one of {sd_state_group_members, sd_state_groups} required",
               description="Flat list of US state codes constituting the custom group.",
           ),
           ParameterDoc(
               name="sd_state_groups",
               type="dict[str, list[str]]",
               default=None,
               constraint="exactly one of {sd_state_group_members, sd_state_groups} required",
               description="Named subgroups: maps group-label to state-code list. For multi-named-subgroup studies.",
           ),
       ),
    }
)
_L1D_STATE_GROUP = tuple(
    _L1D_STATE_GROUP_CUSTOM_PATCHED if d.option == "custom_state_group" else d
    for d in _L1D_STATE_GROUP
)

# L1.D fred_sd_variable_group (12 categorical groupings)
_VAR_GROUP_DOCS: dict[str, tuple[str, str, str]] = {
    "all_sd_variables": (
        "All FRED-SD state-level variable categories.",
        (
            "Default. Includes every variable category in the "
            "FRED-SD groups manifest. Use as the broadest possible "
            "predictor block; subset via sd_variable_selection if "
            "specific filtering is needed."
        ),
        "Default; broadest predictor block.",
    ),
    "labor_market_core": (
        "Core labour-market series (employment, unemployment, hours).",
        (
            "Includes nonfarm employment, unemployment rate, labour-"
            "force participation, and average hours. Standard "
            "labour-market battery used in most state-level "
            "macroeconomic studies."
        ),
        "Labour-market focused studies; Sahm-rule recession analysis at state level.",
    ),
    "employment_sector": (
        "Sectoral employment series (NAICS supersector breakdowns).",
        (
            "Sectoral employment counts (manufacturing, construction, "
            "services, government, etc.). Useful when industry mix "
            "explains target variation."
        ),
        "Industry-level employment studies; structural-transformation analysis.",
    ),
    "income": (
        "Personal income / earnings series.",
        (
            "Includes per-capita personal income, total state income, "
            "and components (wages, transfers, dividends). Slow-"
            "moving but persistent predictor of state economic "
            "activity."
        ),
        "Consumer / household income studies; transfer-payment analysis.",
    ),
    "housing": (
        "State housing series (permits, prices, starts).",
        (
            "Building permits, housing starts, house-price indices. "
            "Leading indicator of state economic activity; central to "
            "any housing-cycle analysis."
        ),
        "Housing-cycle studies; foreclosure / mortgage-market analysis.",
    ),
    "trade": (
        "Trade / commerce series.",
        (
            "Retail sales, wholesale trade, port activity. State-level "
            "trade-flow indicators where available."
        ),
        "Trade-flow studies; port-region economic analysis.",
    ),
    "gsp_output": (
        "Gross state product / output series.",
        (
            "BEA gross state product (GSP), the state-level analogue "
            "of national GDP. Released quarterly with publication "
            "lag; main aggregate state-level output measure."
        ),
        "Aggregate output studies; state-level GDP forecasting.",
    ),
    "direct_analog_high_confidence": (
        "Variables with direct national analog (high-confidence cross-frequency join).",
        (
            "FRED-SD variables that map directly onto a known FRED-MD / "
            "-QD national series at the same definition. The cleanest "
            "subset for cross-frequency studies that need national-"
            "state correspondence."
        ),
        "Cross-frequency studies needing direct national-state mapping.",
    ),
    "provisional_analog_medium": (
        "Variables with provisional national analog (medium-confidence join).",
        (
            "FRED-SD variables that *approximately* map onto a "
            "national series but with some definition mismatch "
            "(coverage gap, methodology change, etc.). Use with "
            "caution; the join is provisional."
        ),
        "Sensitivity analyses on the analog mapping.",
        ),
    "no_reliable_analog": (
        "Variables without a reliable national analog.",
        (
            "FRED-SD-only series that have no clean correspondence "
            "to a FRED-MD / -QD national variable. Useful for "
            "state-only studies that exclude national benchmarks."
        ),
        "State-only studies; spatial-econometric panels that ignore national aggregates.",
    ),
    "semantic_review_outputs": (
        "Outputs of the FRED-SD semantic review process.",
        (
            "Variables flagged through the FRED-SD semantic-review "
            "pipeline (audit-trail diagnostics produced by the "
            "FRED-SD construction process). Mostly used for "
            "diagnostic provenance, not as predictors."
        ),
        "Audit-trail diagnostics for the FRED-SD construction process.",
    ),
    "custom_sd_variable_group": (
        "User-supplied variable list (leaf_config.custom_sd_variables).",
        (
            "Bespoke variable selections -- e.g. 'manufacturing + "
            "trade only' or 'a specific BLS series list'. Reads the "
            "explicit variable list from "
            "``leaf_config.custom_sd_variables``."
        ),
        "Bespoke variable selections not captured by built-in groupings.",
    ),
}
_L1D_VAR_GROUP = tuple(
    _t1("l1_d", "fred_sd_variable_group", option, summary,
        (
            f"FRED-SD variable category: {desc}\n\n"
            f"Restricts the predictor block to series tagged with this "
            f"category in the FRED-SD groups manifest. Combine with "
            f"``fred_sd_state_group`` to control geography and with "
            f"``sd_variable_selection`` to restrict further within "
            f"this category."
        ),
        when,
        related=tuple(k for k in _VAR_GROUP_DOCS if k != option)[:3],
        references=(_REF_DESIGN_L1, _REF_MCCRACKEN_NG_2020))
    for option, (summary, desc, when) in _VAR_GROUP_DOCS.items()
)
# Cycle 20 L1.D fix: ParameterDoc for custom_sd_variable_group (OR: sd_variable_group_members / sd_variable_groups)
_L1D_VAR_GROUP_CUSTOM = next(
    d for d in _L1D_VAR_GROUP
    if d.axis == "fred_sd_variable_group" and d.option == "custom_sd_variable_group"
)
_L1D_VAR_GROUP_CUSTOM_PATCHED = _L1D_VAR_GROUP_CUSTOM.__class__(
    **{**_L1D_VAR_GROUP_CUSTOM.__dict__,
       "parameters": (
           ParameterDoc(
               name="sd_variable_group_members",
               type="list[str]",
               default=None,
               constraint="exactly one of {sd_variable_group_members, sd_variable_groups} required",
               description="Flat list of FRED-SD variable names constituting the custom group.",
           ),
           ParameterDoc(
               name="sd_variable_groups",
               type="dict[str, list[str]]",
               default=None,
               constraint="exactly one of {sd_variable_group_members, sd_variable_groups} required",
               description="Named subgroups for variables: maps group-label to variable-name list.",
           ),
       ),
    }
)
_L1D_VAR_GROUP = tuple(
    _L1D_VAR_GROUP_CUSTOM_PATCHED if d.option == "custom_sd_variable_group" else d
    for d in _L1D_VAR_GROUP
)

# L1.D state_selection / sd_variable_selection (binary axes)
_L1D_STATE_SEL = (
    _t1("l1_d", "state_selection", "all_states",
        "Auto-select every state in ``fred_sd_state_group``.",
        "Skips per-state cherry-picking; uses the full set defined by the active state group.",
        "Default; state-group already does the filtering.",
        related=("selected_states",)),
    _t1("l1_d", "state_selection", "selected_states",
        "Use the explicit per-state list in leaf_config.",
        "Reads ``leaf_config.selected_states`` -- a subset of the active state-group, allowing fine-grained control.",
        "Custom regional studies that need a non-standard state subset.",
        related=("all_states",)),
)
# Cycle 20 L1.D fix: ParameterDoc for sd_states
_L1D_STATE_SEL_SELECTED = next(
    d for d in _L1D_STATE_SEL
    if d.axis == "state_selection" and d.option == "selected_states"
)
_L1D_STATE_SEL_SELECTED_PATCHED = _L1D_STATE_SEL_SELECTED.__class__(
    **{**_L1D_STATE_SEL_SELECTED.__dict__,
       "parameters": (
           ParameterDoc(
               name="sd_states",
               type="list[str]",
               default=None,
               constraint="non-empty list required",
               description="Filter FRED-SD panel to listed states only. Applied AFTER state_group resolution (i.e., intersect).",
           ),
       ),
    }
)
_L1D_STATE_SEL = tuple(
    _L1D_STATE_SEL_SELECTED_PATCHED if d.option == "selected_states" else d
    for d in _L1D_STATE_SEL
)
_L1D_VAR_SEL = (
    _t1("l1_d", "sd_variable_selection", "all_sd_variables",
        "Auto-select every variable in ``fred_sd_variable_group``.",
        "Default; uses the full set defined by the active variable group.",
        "Default broad-spectrum study.",
        related=("selected_sd_variables",)),
    _t1("l1_d", "sd_variable_selection", "selected_sd_variables",
        "Use the explicit per-series list in leaf_config.",
        "Reads ``leaf_config.selected_sd_variables`` -- a subset of the active variable group.",
        "Targeted studies that focus on specific FRED-SD series.",
        related=("all_sd_variables",)),
)
# Cycle 20 L1.D fix: ParameterDoc for sd_variables
_L1D_VAR_SEL_SELECTED = next(
    d for d in _L1D_VAR_SEL
    if d.axis == "sd_variable_selection" and d.option == "selected_sd_variables"
)
_L1D_VAR_SEL_SELECTED_PATCHED = _L1D_VAR_SEL_SELECTED.__class__(
    **{**_L1D_VAR_SEL_SELECTED.__dict__,
       "parameters": (
           ParameterDoc(
               name="sd_variables",
               type="list[str]",
               default=None,
               constraint="non-empty list required",
               description="Filter FRED-SD variables to listed names only. Applied AFTER variable_group resolution.",
           ),
       ),
    }
)
_L1D_VAR_SEL = tuple(
    _L1D_VAR_SEL_SELECTED_PATCHED if d.option == "selected_sd_variables" else d
    for d in _L1D_VAR_SEL
)

# L1.G regime_estimation_temporal_rule
_L1G_TEMP_RULE = (
    _t1("l1_g", "regime_estimation_temporal_rule", "expanding_window_per_origin",
        "Re-estimate regimes on every expanding window.",
        "Default for ``estimated_*`` regime methods. Avoids look-ahead by re-fitting the regime model on data through each origin date.",
        "Default; OOS-safe regime estimation.",
        related=("rolling_window_per_origin", "block_recompute")),
    _t1("l1_g", "regime_estimation_temporal_rule", "rolling_window_per_origin",
        "Re-estimate regimes on a fixed-length rolling window.",
        "Uses the most-recent ``params.window`` observations only. Useful when regime structure drifts over time.",
        "Drifting / non-stationary regime structure.",
        related=("expanding_window_per_origin", "block_recompute")),
    _t1("l1_g", "regime_estimation_temporal_rule", "block_recompute",
        "Re-estimate every leaf_config.regime_recompute_interval origins.",
        "Cheap approximation to per-origin re-fits. Caches the regime classification between recompute boundaries.",
        "Long sweeps where per-origin regime re-fits are infeasible.",
        related=("expanding_window_per_origin", "rolling_window_per_origin")),
)

register(*_L1A_FRED_SD_FREQ)
register(*_L1C_MISSING)
register(*_L1C_RAW_MISSING)
register(*_L1C_RAW_OUTLIER)
register(*_L1C_RELEASE_LAG)
register(*_L1C_CONTEMP)
register(*_L1C_OFFICIAL_SCOPE)
register(*_L1D_PRED_GEO)
register(*_L1D_STATE_GROUP)
register(*_L1D_VAR_GROUP)
register(*_L1D_STATE_SEL)
register(*_L1D_VAR_SEL)
register(*_L1G_TEMP_RULE)


# Register the manually-written entries.
register(
    _L1A_SOURCE_OFFICIAL_ONLY, _L1A_SOURCE_CUSTOM_PANEL, _L1A_SOURCE_OFFICIAL_PLUS_CUSTOM,
    _L1A_FRED_MD, _L1A_FRED_QD, _L1A_FRED_SD, _L1A_FRED_MD_SD, _L1A_FRED_QD_SD,
    _L1A_FREQ_MONTHLY, _L1A_FREQ_QUARTERLY,
    _L1A_VINTAGE_CURRENT, _L1A_VINTAGE_REAL_TIME_ALFRED,
    _L1A_INFOSET_FINAL, _L1A_INFOSET_PSEUDO,
    _L1B_TARGET_SINGLE, _L1B_TARGET_MULTI,
    _L1C_VARS_ALL, _L1C_VARS_CORE, _L1C_VARS_CATEGORY, _L1C_VARS_TARGET_SPEC, _L1C_VARS_EXPLICIT,
    _L1C_TRANSFORM_OFFICIAL, _L1C_TRANSFORM_RAW,
    _L1D_GEO_SINGLE, _L1D_GEO_ALL, _L1D_GEO_SELECTED,
    _L1E_START_MAX_BAL, _L1E_START_EARLIEST, _L1E_START_FIXED,
    _L1E_END_LATEST, _L1E_END_FIXED,
    _L1F_HORIZON_STD_MD, _L1F_HORIZON_STD_QD, _L1F_HORIZON_SINGLE,
    _L1F_HORIZON_CUSTOM, _L1F_HORIZON_RANGE,
    _L1G_REGIME_NONE, _L1G_REGIME_NBER, _L1G_REGIME_USER,
    _L1G_REGIME_MS, _L1G_REGIME_THRESH, _L1G_REGIME_BREAK,
)
