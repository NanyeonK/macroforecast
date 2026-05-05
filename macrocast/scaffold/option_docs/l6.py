"""L6 statistical tests -- per-option documentation.

L6 implements the classical forecast-evaluation tests. Operational
options are spread across three sub-layers:

* L6.A equal-predictive-ability (DM / GW / DMP),
* L6.C conditional predictive ability (Giacomini-Rossi / Rossi-Sekhposyan),
* L6.D multiple-model selection (MCS / SPA / Reality Check / StepM).

L6.B (nested), L6.E (density), L6.F (direction), L6.G (residual) are
populated as tier-1 entries by their schema axes but presently expose
no ``operational`` test_id options separate from the four sub-layers
listed above.
"""
from __future__ import annotations

from . import register
from .types import OptionDoc, Reference

_REVIEWED = "2026-05-05"
_REVIEWER = "macrocast author"

_REF_DESIGN_L6 = Reference(
    citation="macrocast design Part 3, L6: 'tests must report (statistic, p-value, kernel, lag) and respect HAC dependence-correction.'",
)


def _e(sublayer: str, axis: str, option: str, summary: str, description: str, when_to_use: str,
       *, when_not_to_use: str = "", references: tuple[Reference, ...] = (_REF_DESIGN_L6,),
       related: tuple[str, ...] = ()) -> OptionDoc:
    return OptionDoc(
        layer="l6", sublayer=sublayer, axis=axis, option=option,
        summary=summary, description=description, when_to_use=when_to_use,
        when_not_to_use=when_not_to_use, references=references,
        related_options=related,
        last_reviewed=_REVIEWED, reviewer=_REVIEWER,
    )


# L6.A equal_predictive_test
register(
    _e("L6_A_equal_predictive", "equal_predictive_test", "dm_diebold_mariano",
       "Diebold-Mariano (1995) equal-predictive-ability test with Newey-West HAC SE.",
       (
           "Pairwise test of equal expected loss between two forecasts. "
           "Implements DM with HLN small-sample correction (Harvey-"
           "Leybourne-Newbold 1997) and a configurable HAC kernel "
           "(``newey_west`` default, ``andrews`` / ``parzen`` available). "
           "Two-sided alternative tests equality of MSE / MAE losses."
       ),
       "Pairwise comparison of two non-nested forecasts.",
       when_not_to_use="Nested-model comparisons -- use Clark-West (L6.B) instead.",
       references=(_REF_DESIGN_L6,
           Reference(citation="Diebold & Mariano (1995) 'Comparing Predictive Accuracy', JBES 13(3): 253-263."),
           Reference(citation="Harvey, Leybourne & Newbold (1997) 'Testing the equality of prediction mean squared errors', IJF 13(2): 281-291.")),
       related=("gw_giacomini_white", "dmp_multi_horizon", "multi")),
    _e("L6_A_equal_predictive", "equal_predictive_test", "gw_giacomini_white",
       "Giacomini-White (2006) conditional equal-predictive-ability test.",
       (
           "Generalises DM to test conditional predictive ability "
           "given a vector of predictors. Robust to non-stationary "
           "performance differentials and works with rolling / "
           "expanding-window forecasts."
       ),
       "Conditional / regime-dependent forecast comparisons.",
       references=(_REF_DESIGN_L6,
           Reference(citation="Giacomini & White (2006) 'Tests of Conditional Predictive Ability', Econometrica 74(6): 1545-1578.")),
       related=("dm_diebold_mariano", "multi")),
    _e("L6_A_equal_predictive", "equal_predictive_test", "dmp_multi_horizon",
       "Diebold-Mariano-Pesaran joint multi-horizon test.",
       (
           "HAC-adjusted stacked DM test that evaluates equality of "
           "predictive ability across all forecast horizons "
           "simultaneously. v0.3 implementation following Pesaran-"
           "Timmermann."
       ),
       "Joint significance across multiple horizons (avoids per-horizon p-value adjustment).",
       references=(_REF_DESIGN_L6,
           Reference(citation="Pesaran & Timmermann (2007) 'Selection of estimation window in the presence of breaks', JoE 137(1): 134-161.")),
       related=("dm_diebold_mariano",)),
    _e("L6_A_equal_predictive", "equal_predictive_test", "multi",
       "Run DM + GW + DMP and stack the results.",
       (
           "Multi-test convenience option; emits a single output table "
           "with one row per test. Useful as a robustness check."
       ),
       "Comprehensive equal-predictive-ability audits.",
       related=("dm_diebold_mariano", "gw_giacomini_white", "dmp_multi_horizon")),
)


# L6.C conditional predictive ability
register(
    _e("L6_C_cpa", "cpa_test", "giacomini_rossi_2010",
       "Giacomini-Rossi (2010) rolling-window fluctuation test.",
       (
           "Rolling-window analogue of the GW test that tracks the "
           "evolution of predictive ability over time. v0.25 ships the "
           "simulated-CV table for ``(m/T, alpha)`` pairs used to "
           "compute exact critical values."
       ),
       "Detecting whether predictive ability is stable across the OOS sample.",
       references=(_REF_DESIGN_L6,
           Reference(citation="Giacomini & Rossi (2010) 'Forecast Comparisons in Unstable Environments', JAE 25(4): 595-620.")),
       related=("rossi_sekhposyan", "multi")),
    _e("L6_C_cpa", "cpa_test", "rossi_sekhposyan",
       "Rossi-Sekhposyan (2011/2016) one-time / instabilities tests.",
       (
           "Companion suite of conditional predictive ability tests "
           "based on monitoring statistics over the OOS sample. Detects "
           "structural breaks in relative forecast performance."
       ),
       "Detecting one-off regime shifts in predictive ability.",
       references=(_REF_DESIGN_L6,
           Reference(citation="Rossi & Sekhposyan (2016) 'Forecast Rationality Tests in the Presence of Instabilities', JAE 31(3): 507-532.")),
       related=("giacomini_rossi_2010",)),
    _e("L6_C_cpa", "cpa_test", "multi",
       "Run all CPA tests and stack the results.",
       "Multi-test convenience option; emits one row per CPA test.",
       "Comprehensive CPA audits.",
       related=("giacomini_rossi_2010", "rossi_sekhposyan")),
)


# L6.D multiple-model tests
register(
    _e("L6_D_multiple_model", "multiple_model_test", "mcs_hansen",
       "Hansen-Lunde-Nason Model Confidence Set (2011).",
       (
           "Default multiple-comparison test. Returns the set of "
           "models that contain the best at confidence level 1 - α "
           "via stationary-bootstrap (Politis-White 2004) iterated "
           "elimination. v0.25 uses the auto-tuned block length."
       ),
       "Identifying the small set of equally-best models out of many candidates.",
       references=(_REF_DESIGN_L6,
           Reference(citation="Hansen, Lunde & Nason (2011) 'The Model Confidence Set', Econometrica 79(2): 453-497.")),
       related=("spa_hansen", "reality_check_white", "step_m_romano_wolf")),
    _e("L6_D_multiple_model", "multiple_model_test", "spa_hansen",
       "Hansen Superior Predictive Ability test (2005).",
       (
           "Tests whether any candidate beats the benchmark; "
           "studentises losses and uses a centred-bootstrap p-value. "
           "Compared to RC, less sensitive to poor models."
       ),
       "Testing whether the best candidate beats a fixed benchmark.",
       references=(_REF_DESIGN_L6,
           Reference(citation="Hansen (2005) 'A Test for Superior Predictive Ability', JBES 23(4): 365-380.")),
       related=("mcs_hansen", "reality_check_white")),
    _e("L6_D_multiple_model", "multiple_model_test", "reality_check_white",
       "White's Reality Check (2000).",
       (
           "Tests whether the best of N candidates beats a fixed "
           "benchmark. Original multiple-comparison test; SPA improves "
           "by studentising."
       ),
       "Foundational reality-check; compatibility with older studies.",
       references=(_REF_DESIGN_L6,
           Reference(citation="White (2000) 'A Reality Check for Data Snooping', Econometrica 68(5): 1097-1126.")),
       related=("spa_hansen",)),
    _e("L6_D_multiple_model", "multiple_model_test", "step_m_romano_wolf",
       "Romano-Wolf StepM (2005) multiple-testing procedure.",
       (
           "Step-down procedure that controls FWER asymptotically. "
           "Returns ranked subset of candidates that beat the benchmark "
           "at level α."
       ),
       "Identifying which specific models in a large pool beat the benchmark.",
       references=(_REF_DESIGN_L6,
           Reference(citation="Romano & Wolf (2005) 'Stepwise Multiple Testing as Formalized Data Snooping', Econometrica 73(4): 1237-1282.")),
       related=("mcs_hansen", "spa_hansen")),
)
