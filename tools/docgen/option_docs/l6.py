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

Cycle 29: L6.A and L6.B ops updated with op_page=True, op_func_name,
data_args, return_type, returns_attrs.
"""

from __future__ import annotations

from . import register
from .types import OptionDoc, ParameterDoc, Reference, REQUIRED

_REVIEWED = "2026-05-05"
_REVIEWER = "macroforecast author"

_REF_DESIGN_L6 = Reference(
    citation="macroforecast design Part 3, L6: 'tests must report (statistic, p-value, kernel, lag) and respect HAC dependence-correction.'",
)


# Shared data_args tuples for per-op encyclopedia pages (Cycle 29).

_L6_LOSS_PAIR_DATA_ARGS: tuple[ParameterDoc, ...] = (
    ParameterDoc(
        name="loss_a",
        type="np.ndarray",
        default=REQUIRED,
        description="Per-period losses for model A (e.g. squared errors).",
    ),
    ParameterDoc(
        name="loss_b",
        type="np.ndarray",
        default=REQUIRED,
        description="Per-period losses for model B.",
    ),
)

_L6_DMP_DATA_ARGS: tuple[ParameterDoc, ...] = (
    ParameterDoc(
        name="loss_differentials",
        type="list[np.ndarray] or np.ndarray",
        default=REQUIRED,
        description="Per-period loss differentials, one array per horizon or pre-stacked.",
    ),
)

_L6_HN_DATA_ARGS: tuple[ParameterDoc, ...] = (
    ParameterDoc(
        name="e_a",
        type="np.ndarray",
        default=REQUIRED,
        description="Forecast errors for model A (actual - forecast_a).",
    ),
    ParameterDoc(
        name="e_b",
        type="np.ndarray",
        default=REQUIRED,
        description="Forecast errors for model B (actual - forecast_b).",
    ),
)

_L6_CW_DATA_ARGS: tuple[ParameterDoc, ...] = (
    ParameterDoc(
        name="loss_small",
        type="np.ndarray",
        default=REQUIRED,
        description="Squared losses for the small (restricted) model.",
    ),
    ParameterDoc(
        name="loss_large",
        type="np.ndarray",
        default=REQUIRED,
        description="Squared losses for the large (unrestricted) model.",
    ),
    ParameterDoc(
        name="f_small",
        type="np.ndarray",
        default=REQUIRED,
        description="Point forecasts for the small model.",
    ),
    ParameterDoc(
        name="f_large",
        type="np.ndarray",
        default=REQUIRED,
        description="Point forecasts for the large model.",
    ),
)

_L6_ENC_DATA_ARGS: tuple[ParameterDoc, ...] = (
    ParameterDoc(
        name="loss_small",
        type="np.ndarray",
        default=REQUIRED,
        description="Squared losses for the small model.",
    ),
    ParameterDoc(
        name="loss_large",
        type="np.ndarray",
        default=REQUIRED,
        description="Squared losses for the large model.",
    ),
)


def _e(
    sublayer: str,
    axis: str,
    option: str,
    summary: str,
    description: str,
    when_to_use: str,
    *,
    when_not_to_use: str = "",
    references: tuple[Reference, ...] = (_REF_DESIGN_L6,),
    related: tuple[str, ...] = (),
    op_page: bool = False,
    op_func_name: str = "",
    data_args: tuple[ParameterDoc, ...] = (),
    return_type: str = "",
    returns_attrs: tuple[tuple[str, str, str], ...] = (),
) -> OptionDoc:
    return OptionDoc(
        layer="l6",
        sublayer=sublayer,
        axis=axis,
        option=option,
        summary=summary,
        description=description,
        when_to_use=when_to_use,
        when_not_to_use=when_not_to_use,
        references=references,
        related_options=related,
        op_page=op_page,
        op_func_name=op_func_name,
        data_args=data_args,
        return_type=return_type,
        returns_attrs=returns_attrs,
        last_reviewed=_REVIEWED,
        reviewer=_REVIEWER,
    )


# L6.A equal_predictive_test
register(
    _e(
        "L6_A_equal_predictive",
        "equal_predictive_test",
        "dm_diebold_mariano",
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
        references=(
            _REF_DESIGN_L6,
            Reference(
                citation="Diebold & Mariano (1995) 'Comparing Predictive Accuracy', JBES 13(3): 253-263."
            ),
            Reference(
                citation="Harvey, Leybourne & Newbold (1997) 'Testing the equality of prediction mean squared errors', IJF 13(2): 281-291."
            ),
        ),
        related=("gw_giacomini_white", "dmp_multi_horizon", "multi"),
        op_page=True,
        op_func_name="dm_test",
        data_args=_L6_LOSS_PAIR_DATA_ARGS,
        return_type="DMTestResult",
        returns_attrs=(
            ("stat", "float or None", "DM test statistic"),
            ("pvalue", "float or None", "Two-sided p-value"),
            ("decision", "bool", "Reject H0 at 5%"),
            ("n_obs", "int", "Observations used"),
            ("hln_correction", "bool", "HLN correction applied"),
        ),
    ),
    _e(
        "L6_A_equal_predictive",
        "equal_predictive_test",
        "gw_giacomini_white",
        "Giacomini-White (2006) conditional equal-predictive-ability test.",
        (
            "Generalises DM to test conditional predictive ability "
            "given a vector of predictors. Robust to non-stationary "
            "performance differentials and works with rolling / "
            "expanding-window forecasts."
        ),
        "Conditional / regime-dependent forecast comparisons.",
        references=(
            _REF_DESIGN_L6,
            Reference(
                citation="Giacomini & White (2006) 'Tests of Conditional Predictive Ability', Econometrica 74(6): 1545-1578."
            ),
        ),
        related=("dm_diebold_mariano", "multi"),
        op_page=True,
        op_func_name="gw_test",
        data_args=_L6_LOSS_PAIR_DATA_ARGS,
        return_type="GWTestResult",
        returns_attrs=(
            ("stat", "float or None", "GW test statistic"),
            ("pvalue", "float or None", "Two-sided p-value"),
            ("decision", "bool", "Reject H0 at 5%"),
            ("n_obs", "int", "Observations used"),
            ("hln_correction", "bool", "HLN correction applied"),
        ),
    ),
    _e(
        "L6_A_equal_predictive",
        "equal_predictive_test",
        "dmp_multi_horizon",
        "Diebold-Mariano-Pesaran joint multi-horizon test.",
        (
            "HAC-adjusted stacked DM test that evaluates equality of "
            "predictive ability across all forecast horizons "
            "simultaneously. v0.3 implementation following Pesaran-"
            "Timmermann."
        ),
        "Joint significance across multiple horizons (avoids per-horizon p-value adjustment).",
        references=(
            _REF_DESIGN_L6,
            Reference(
                citation="Pesaran & Timmermann (2007) 'Selection of estimation window in the presence of breaks', JoE 137(1): 134-161."
            ),
        ),
        related=("dm_diebold_mariano",),
        op_page=True,
        op_func_name="dmp_test",
        data_args=_L6_DMP_DATA_ARGS,
        return_type="DMPTestResult",
        returns_attrs=(
            ("stat", "float or None", "DMP test statistic"),
            ("pvalue", "float or None", "Two-sided p-value"),
            ("decision", "bool", "Reject H0 at 5%"),
            ("n_obs_stacked", "int", "Stacked observations"),
        ),
    ),
    _e(
        "L6_A_equal_predictive",
        "equal_predictive_test",
        "harvey_newbold_encompassing",
        "Harvey-Leybourne-Newbold (1998) forecast-encompassing test.",
        (
            "Tests the null that forecast f_1 encompasses f_2 -- i.e. "
            "the optimal linear combination of the two forecasts puts "
            "zero weight on f_2's error. Constructs ``d_t = e_a (e_a - "
            "e_b)`` from the per-period forecast errors and tests its "
            "mean against zero with a Newey-West HAC long-run variance "
            "and an HLN small-sample correction at horizon h>1. "
            "Asymmetric by construction (f_1 encompasses f_2 ≠ f_2 "
            "encompasses f_1)."
        ),
        "Deciding whether one forecast contains all the information of another.",
        when_not_to_use="Symmetric equal-MSE comparison -- use ``dm_diebold_mariano`` instead.",
        references=(
            _REF_DESIGN_L6,
            Reference(
                citation="Harvey, Leybourne & Newbold (1998) 'Tests for Forecast Encompassing', JBES 16(2): 254-259."
            ),
        ),
        related=("dm_diebold_mariano", "gw_giacomini_white", "multi"),
        op_page=True,
        op_func_name="hn_test",
        data_args=_L6_HN_DATA_ARGS,
        return_type="HNTestResult",
        returns_attrs=(
            ("stat", "float or None", "HN test statistic"),
            ("pvalue", "float or None", "One-sided p-value"),
            ("decision", "bool", "Reject H0 at 5%"),
            ("n_obs", "int", "Observations used"),
            ("encompassing", "str", "Direction: a_over_b"),
        ),
    ),
    _e(
        "L6_A_equal_predictive",
        "equal_predictive_test",
        "multi",
        "Run DM + GW + DMP and stack the results.",
        (
            "Multi-test convenience option; emits a single output table "
            "with one row per test. Useful as a robustness check."
        ),
        "Comprehensive equal-predictive-ability audits.",
        related=(
            "dm_diebold_mariano",
            "gw_giacomini_white",
            "dmp_multi_horizon",
            "harvey_newbold_encompassing",
        ),
    ),
)


# L6.C conditional predictive ability
register(
    _e(
        "L6_C_cpa",
        "cpa_test",
        "giacomini_rossi_2010",
        "Giacomini-Rossi (2010) rolling-window fluctuation test.",
        (
            "Rolling-window analogue of the GW test that tracks the "
            "evolution of predictive ability over time. v0.25 ships the "
            "simulated-CV table for ``(m/T, alpha)`` pairs used to "
            "compute exact critical values."
        ),
        "Detecting whether predictive ability is stable across the OOS sample.",
        references=(
            _REF_DESIGN_L6,
            Reference(
                citation="Giacomini & Rossi (2010) 'Forecast Comparisons in Unstable Environments', JAE 25(4): 595-620."
            ),
        ),
        related=("rossi_sekhposyan", "multi"),
    ),
    _e(
        "L6_C_cpa",
        "cpa_test",
        "rossi_sekhposyan",
        "Rossi-Sekhposyan (2011/2016) one-time / instabilities tests.",
        (
            "Companion suite of conditional predictive ability tests "
            "based on monitoring statistics over the OOS sample. Detects "
            "structural breaks in relative forecast performance."
        ),
        "Detecting one-off regime shifts in predictive ability.",
        references=(
            _REF_DESIGN_L6,
            Reference(
                citation="Rossi & Sekhposyan (2016) 'Forecast Rationality Tests in the Presence of Instabilities', JAE 31(3): 507-532."
            ),
        ),
        related=("giacomini_rossi_2010",),
    ),
    _e(
        "L6_C_cpa",
        "cpa_test",
        "multi",
        "Run all CPA tests and stack the results.",
        "Multi-test convenience option; emits one row per CPA test.",
        "Comprehensive CPA audits.",
        related=("giacomini_rossi_2010", "rossi_sekhposyan"),
    ),
)


# L6.D multiple-model tests
register(
    _e(
        "L6_D_multiple_model",
        "multiple_model_test",
        "mcs_hansen",
        "Hansen-Lunde-Nason Model Confidence Set (2011).",
        (
            "Default multiple-comparison test. Returns the set of "
            "models that contain the best at confidence level 1 - α "
            "via stationary-bootstrap (Politis-White 2004) iterated "
            "elimination. v0.25 uses the auto-tuned block length."
        ),
        "Identifying the small set of equally-best models out of many candidates.",
        references=(
            _REF_DESIGN_L6,
            Reference(
                citation="Hansen, Lunde & Nason (2011) 'The Model Confidence Set', Econometrica 79(2): 453-497."
            ),
        ),
        related=("spa_hansen", "reality_check_white", "step_m_romano_wolf"),
    ),
    _e(
        "L6_D_multiple_model",
        "multiple_model_test",
        "spa_hansen",
        "Hansen Superior Predictive Ability test (2005).",
        (
            "Tests whether any candidate beats the benchmark; "
            "studentises losses and uses a centred-bootstrap p-value. "
            "Compared to RC, less sensitive to poor models."
        ),
        "Testing whether the best candidate beats a fixed benchmark.",
        references=(
            _REF_DESIGN_L6,
            Reference(
                citation="Hansen (2005) 'A Test for Superior Predictive Ability', JBES 23(4): 365-380."
            ),
        ),
        related=("mcs_hansen", "reality_check_white"),
    ),
    _e(
        "L6_D_multiple_model",
        "multiple_model_test",
        "reality_check_white",
        "White's Reality Check (2000).",
        (
            "Tests whether the best of N candidates beats a fixed "
            "benchmark. Original multiple-comparison test; SPA improves "
            "by studentising."
        ),
        "Foundational reality-check; compatibility with older studies.",
        references=(
            _REF_DESIGN_L6,
            Reference(
                citation="White (2000) 'A Reality Check for Data Snooping', Econometrica 68(5): 1097-1126."
            ),
        ),
        related=("spa_hansen",),
    ),
    _e(
        "L6_D_multiple_model",
        "multiple_model_test",
        "step_m_romano_wolf",
        "Romano-Wolf StepM (2005) multiple-testing procedure.",
        (
            "Step-down procedure that controls FWER asymptotically. "
            "Returns ranked subset of candidates that beat the benchmark "
            "at level α."
        ),
        "Identifying which specific models in a large pool beat the benchmark.",
        references=(
            _REF_DESIGN_L6,
            Reference(
                citation="Romano & Wolf (2005) 'Stepwise Multiple Testing as Formalized Data Snooping', Econometrica 73(4): 1237-1282."
            ),
        ),
        related=("mcs_hansen", "spa_hansen"),
    ),
)


# ---------------------------------------------------------------------------
# L6.B nested_test
# ---------------------------------------------------------------------------

_REF_CLARK_WEST_2007 = Reference(
    citation="Clark & West (2007) 'Approximately Normal Tests for Equal Predictive Accuracy in Nested Models', JoE 138(2): 291-311.",
)
_REF_CLARK_MCCRACKEN_2001 = Reference(
    citation="Clark & McCracken (2001) 'Tests of Equal Forecast Accuracy and Encompassing for Nested Models', JoE 105(2): 1-28.",
)
_REF_ERICSSON_1992 = Reference(
    citation="Ericsson (1992) 'Parameter Constancy, Mean Square Forecast Errors, and Measuring Forecast Performance', JoE 52(1-2): 113-153.",
)

register(
    _e(
        "L6_B_nested",
        "nested_test",
        "clark_west",
        "Clark-West (2007) MSE-adjusted nested-model predictive ability test.",
        (
            "Tests whether the large (unrestricted) model significantly "
            "outperforms the small (restricted, nested) model. Constructs "
            "the CW-adjusted statistic "
            "``f_t = (loss_small - loss_large) + (f_small - f_large)^2``, "
            "removing the negative expected value bias that standard DM "
            "has in nested comparisons. One-sided test (H_a: large model "
            "improves on small); ``hln=False``."
        ),
        "Testing whether a larger model with additional regressors beats the restricted benchmark.",
        when_not_to_use=(
            "Non-nested model comparisons -- use DM / GW (L6.A) instead. "
            "Forecast combination (use HN encompassing instead)."
        ),
        references=(
            _REF_DESIGN_L6,
            _REF_CLARK_WEST_2007,
        ),
        related=("enc_new", "enc_t", "multi"),
        op_page=True,
        op_func_name="cw_test",
        data_args=_L6_CW_DATA_ARGS,
        return_type="CWTestResult",
        returns_attrs=(
            ("stat", "float or None", "CW test statistic"),
            ("pvalue", "float or None", "One-sided p-value"),
            ("decision", "bool", "Reject H0 at 5%"),
            ("n_obs", "int", "Observations used"),
            ("cw_adjustment", "bool", "CW penalty applied"),
        ),
    ),
    _e(
        "L6_B_nested",
        "nested_test",
        "enc_new",
        "Enc-New forecast encompassing test (Clark-McCracken 2001).",
        (
            "Tests whether the large model's forecast contains information "
            "beyond the small (nested) model. Uses raw loss improvement "
            "``f_t = loss_small - loss_large`` without CW adjustment, "
            "then applies one-sided DM inference. Complementary to the "
            "Clark-West test when the user does not want the CW penalty."
        ),
        "Testing forecast encompassing in nested model settings without the CW adjustment term.",
        when_not_to_use="When the CW adjustment for bias is desired -- use clark_west instead.",
        references=(
            _REF_DESIGN_L6,
            _REF_CLARK_MCCRACKEN_2001,
        ),
        related=("clark_west", "enc_t", "multi"),
        op_page=True,
        op_func_name="enc_new_test",
        data_args=_L6_ENC_DATA_ARGS,
        return_type="EncNewTestResult",
        returns_attrs=(
            ("stat", "float or None", "Enc-New statistic"),
            ("pvalue", "float or None", "One-sided p-value"),
            ("decision", "bool", "Reject H0 at 5%"),
            ("n_obs", "int", "Observations used"),
        ),
    ),
    _e(
        "L6_B_nested",
        "nested_test",
        "enc_t",
        "Enc-T forecast encompassing test (Ericsson 1992 t-form).",
        (
            "Ericsson (1992) t-form of the encompassing test. Identical "
            "computation to enc_new in the current implementation "
            "(raw loss improvement, one-sided DM inference, no CW "
            "adjustment). The distinction is the conceptual labelling: "
            "enc_t is cast as a t-statistic on the mean loss improvement. "
            "Both enc_new and enc_t share the same runtime dispatch branch."
        ),
        "Encompassing tests in contexts where the Ericsson t-form labelling is preferred.",
        when_not_to_use="When CW adjustment is needed -- use clark_west instead.",
        references=(
            _REF_DESIGN_L6,
            _REF_ERICSSON_1992,
        ),
        related=("clark_west", "enc_new", "multi"),
        op_page=True,
        op_func_name="enc_t_test",
        data_args=_L6_ENC_DATA_ARGS,
        return_type="EncTTestResult",
        returns_attrs=(
            ("stat", "float or None", "Enc-T statistic"),
            ("pvalue", "float or None", "One-sided p-value"),
            ("decision", "bool", "Reject H0 at 5%"),
            ("n_obs", "int", "Observations used"),
        ),
    ),
    _e(
        "L6_B_nested",
        "nested_test",
        "multi",
        "Run clark_west + enc_new + enc_t and stack the results.",
        "Multi-test convenience option; emits one row per nested test.",
        "Comprehensive nested-model evaluation audits.",
        related=("clark_west", "enc_new", "enc_t"),
    ),
)
