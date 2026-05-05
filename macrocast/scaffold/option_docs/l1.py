"""L1 data definition -- per-option documentation.

L1 is the largest layer: 26 axes spanning data source, target structure,
variable universe, geography (FRED-SD), sample window, horizons, and
regime definition. This module ships Tier-1 documentation for the
**core authoring axes** -- the ones every recipe needs to set
explicitly. Long-tail axes (release_lag_rule, fred_sd_frequency_policy,
etc.) carry placeholder entries that surface the schema description in
the wizard until a follow-up Tier-1 review pass.

Tier-1-complete entries (manually written, reviewer-stamped):
* L1.A custom_source_policy / dataset / vintage_policy
* L1.B target_structure
* L1.C variable_universe / official_transform_policy
* L1.D target_geography_scope / predictor_geography_scope
* L1.E sample_start_rule / sample_end_rule
* L1.F horizon_set
* L1.G regime_definition

Long-tail axes are scaffolded with machine-readable summaries; their
``last_reviewed`` field is empty so the v1.0 docs gauntlet flags them.
"""
from __future__ import annotations

from . import register
from .types import CodeExample, OptionDoc, Reference

_REVIEWED = "2026-05-04"
_REVIEWER = "macrocast author"


# Common references reused across L1 entries.
_REF_DESIGN_L1 = Reference(
    citation="macrocast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'",
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
        "Loads the bundled FRED snapshot via macrocast's raw adapter -- no "
        "network access at runtime, no per-user data file. Vintages are "
        "pinned in ``macrocast/raw/datasets/`` so two users on the same "
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
    summary="Sample at monthly cadence; pairs with FRED-MD / monthly custom panels.",
    description=(
        "Sets the canonical sampling frequency to monthly. Affects "
        "horizon resolution (1 = one month ahead), L2 frequency-alignment "
        "rules (only applicable when datasets mix), and the "
        "``standard_md`` horizon set.\n\n"
        "The default is ``derived``: macrocast infers the frequency from "
        "``dataset`` (fred_md → monthly, fred_qd → quarterly). Setting "
        "frequency explicitly is required for custom panels."
    ),
    when_to_use=(
        "Custom panels with monthly observations; explicit override of the "
        "FRED-MD default for clarity."
    ),
    related_options=("quarterly", "dataset", "horizon_set"),
)

_L1A_FREQ_QUARTERLY = _entry(
    "l1_a", "frequency", "quarterly",
    summary="Sample at quarterly cadence; pairs with FRED-QD / quarterly custom panels.",
    description=(
        "Sets the sampling frequency to quarterly. Activates the "
        "``standard_qd`` horizon set (h ∈ {1, 2, 4, 8} quarters) and "
        "monthly→quarterly aggregation rules in L2.A when the panel "
        "mixes frequencies."
    ),
    when_to_use=(
        "GDP / NIPA-style targets; quarterly custom panels; FRED-QD-based "
        "studies."
    ),
    related_options=("monthly", "dataset", "horizon_set"),
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
    when_not_to_use="Real-time forecasting evaluations -- those need ALFRED vintages.",
    related_options=("information_set_type",),
)


# ---------------------------------------------------------------------------
# L1.A information_set_type
# ---------------------------------------------------------------------------

_L1A_INFOSET_FINAL = _entry(
    "l1_a", "information_set_type", "final_revised_data",
    summary="Each origin sees fully revised data; standard pseudo-OOS protocol.",
    description=(
        "At every walk-forward origin, the model has access to the *current* "
        "revised values for every observation up to that origin. This is "
        "the standard pseudo-out-of-sample protocol used by McCracken-Ng "
        "and most published forecasting comparisons.\n\n"
        "Pros: simple, comparable across studies, no real-time data dep. "
        "Cons: optimistic about real-time forecast performance because "
        "later revisions correct early-vintage measurement error."
    ),
    when_to_use="Default for any benchmark study; comparable to published work.",
    related_options=("pseudo_oos_on_revised_data", "vintage_policy"),
)

_L1A_INFOSET_PSEUDO = _entry(
    "l1_a", "information_set_type", "pseudo_oos_on_revised_data",
    summary="Pseudo-OOS with revised data -- equivalent to final_revised_data for v1.0.",
    description=(
        "Synonym for ``final_revised_data`` in v1.0 (no ALFRED vintage "
        "tracking yet). Both options produce identical forecasts; the axis "
        "is exposed so future versions can route real-time vintages "
        "without breaking existing recipes."
    ),
    when_to_use="When the recipe wants to make the pseudo-OOS protocol explicit (e.g., for clarity in published replication scripts).",
    related_options=("final_revised_data",),
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
        "column list in leaf_config; macrocast filters the panel to that "
        "list verbatim. No grouping logic, no category lookup."
    ),
    "Replication scripts that need an exact predictor set; ablations.",
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
        "Tong's original SETAR uses AR(p) per regime; macrocast's runtime "
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
# Long-tail axes (placeholder Tier-1 entries -- machine-readable summaries
# only). Each is flagged with empty ``last_reviewed`` so the v1.0 docs
# gauntlet treats them as needing human review.
# ---------------------------------------------------------------------------

def _placeholder(
    sublayer: str, axis: str, option: str, summary: str, description: str = "",
) -> OptionDoc:
    return OptionDoc(
        layer="l1", sublayer=sublayer, axis=axis, option=option,
        summary=summary,
        description=(description or summary),
        when_to_use=(
            "Refer to the ``" + axis + "`` axis documentation for the canonical use case."
        ),
        references=(_REF_DESIGN_L1,),
        last_reviewed="",
        reviewer="machine-generated baseline -- needs human review",
    )


# fred_sd_frequency_policy
register(
    _placeholder("l1_a", "fred_sd_frequency_policy", "report_only", "Report mixed frequency without rejecting."),
    _placeholder("l1_a", "fred_sd_frequency_policy", "allow_mixed_frequency", "Allow the mixed-frequency panel; alignment runs in L2.A."),
    _placeholder("l1_a", "fred_sd_frequency_policy", "reject_mixed_known_frequency", "Reject when mixed-frequency would be ambiguous."),
    _placeholder("l1_a", "fred_sd_frequency_policy", "require_single_known_frequency", "Hard-require a single declared frequency."),
)


# missing_availability
register(
    _placeholder("l1_c", "missing_availability", "require_complete_rows", "Drop any row with a missing value."),
    _placeholder("l1_c", "missing_availability", "keep_available_rows", "Keep rows with at least the target observed."),
    _placeholder("l1_c", "missing_availability", "impute_predictors_only", "Impute predictor missings; do not impute target."),
    _placeholder("l1_c", "missing_availability", "zero_fill_leading_predictor_gaps", "Zero-fill leading missing predictor values; preserve interior NaN."),
)


# raw_missing_policy
register(
    _placeholder("l1_c", "raw_missing_policy", "preserve_raw_missing", "Pass NaN through to L2; let L2 imputation handle it."),
    _placeholder("l1_c", "raw_missing_policy", "zero_fill_leading_predictor_missing_before_tcode", "Zero-fill before applying the official t-codes."),
    _placeholder("l1_c", "raw_missing_policy", "impute_raw_predictors", "Impute raw missing predictors at L1."),
    _placeholder("l1_c", "raw_missing_policy", "drop_raw_missing_rows", "Drop rows with raw missing values."),
)


# raw_outlier_policy
register(
    _placeholder("l1_c", "raw_outlier_policy", "preserve_raw_outliers", "Pass outliers through to L2."),
    _placeholder("l1_c", "raw_outlier_policy", "winsorize_raw", "Winsorize raw series at quantile cutpoints."),
    _placeholder("l1_c", "raw_outlier_policy", "iqr_clip_raw", "Clip raw observations beyond IQR-multiple thresholds."),
    _placeholder("l1_c", "raw_outlier_policy", "mad_clip_raw", "Clip raw observations beyond MAD-multiple thresholds."),
    _placeholder("l1_c", "raw_outlier_policy", "zscore_clip_raw", "Clip raw observations beyond z-score thresholds."),
    _placeholder("l1_c", "raw_outlier_policy", "set_raw_outliers_to_missing", "Set raw outliers to NaN; let L2 imputation handle them."),
)


# release_lag_rule
register(
    _placeholder("l1_c", "release_lag_rule", "ignore_release_lag", "Treat all observations as available at their period."),
    _placeholder("l1_c", "release_lag_rule", "fixed_lag_all_series", "Apply a single release lag to every series."),
    _placeholder("l1_c", "release_lag_rule", "series_specific_lag", "Use per-series release lags from leaf_config."),
)


# contemporaneous_x_rule
register(
    _placeholder("l1_c", "contemporaneous_x_rule", "allow_same_period_predictors", "Permit predictors observed in the same period as the target."),
    _placeholder("l1_c", "contemporaneous_x_rule", "forbid_same_period_predictors", "Require predictors to be lagged at least one period."),
)


# official_transform_scope
register(
    _placeholder("l1_c", "official_transform_scope", "target_only", "Apply t-codes only to the target."),
    _placeholder("l1_c", "official_transform_scope", "predictors_only", "Apply t-codes only to predictors."),
    _placeholder("l1_c", "official_transform_scope", "target_and_predictors", "Apply t-codes to both target and predictors (default)."),
    _placeholder("l1_c", "official_transform_scope", "none", "Skip official t-codes entirely."),
)


# predictor_geography_scope
register(
    _placeholder("l1_d", "predictor_geography_scope", "match_target", "Use the same geography scope as the target."),
    _placeholder("l1_d", "predictor_geography_scope", "all_states", "Use predictors from every state regardless of target."),
    _placeholder("l1_d", "predictor_geography_scope", "selected_states", "Use predictors from a user-supplied state list."),
    _placeholder("l1_d", "predictor_geography_scope", "national_only", "Use only national aggregates as predictors."),
)


# fred_sd_state_group: derive the actual option list from introspect so
# we don't hand-curate (and drift from) the schema.
from .. import introspect as _introspect_module

for _axis in _introspect_module.axes("l1"):
    if _axis.name == "fred_sd_state_group":
        for _option in _axis.options:
            register(_placeholder(
                "l1_d", "fred_sd_state_group", _option.value,
                f"State group: {_option.value.replace('_', ' ')}.",
            ))
        break


# regime_estimation_temporal_rule
register(
    _placeholder("l1_g", "regime_estimation_temporal_rule", "expanding_window_per_origin", "Re-estimate regimes per origin using all data up to that origin."),
    _placeholder("l1_g", "regime_estimation_temporal_rule", "rolling_window_per_origin", "Re-estimate per origin using a fixed-length rolling window."),
    _placeholder("l1_g", "regime_estimation_temporal_rule", "block_recompute", "Re-estimate every leaf_config.regime_recompute_interval origins."),
)


# Register the manually-written entries.
register(
    _L1A_SOURCE_OFFICIAL_ONLY, _L1A_SOURCE_CUSTOM_PANEL, _L1A_SOURCE_OFFICIAL_PLUS_CUSTOM,
    _L1A_FRED_MD, _L1A_FRED_QD, _L1A_FRED_SD, _L1A_FRED_MD_SD, _L1A_FRED_QD_SD,
    _L1A_FREQ_MONTHLY, _L1A_FREQ_QUARTERLY,
    _L1A_VINTAGE_CURRENT,
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
