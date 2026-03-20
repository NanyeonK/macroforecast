"""macrocast.evaluation — Layer 3: Forecast evaluation suite."""

from macrocast.evaluation.horserace import (
    HorseRaceResult,
    best_spec_table,
    dm_vs_benchmark_table,
    horserace_summary,
    mcs_membership_table,
    relative_msfe_table,
)
from macrocast.evaluation.decomposition import (
    DecompositionResult,
    decompose_treatment_effects,
)
from macrocast.evaluation.dm import DMResult, dm_test
from macrocast.evaluation.dual import (
    effective_history_length,
    krr_dual_weights,
    nn_dual_weights,
    top_analogies,
    tree_dual_weights,
)
from macrocast.evaluation.mcs import MCSResult, mcs
from macrocast.evaluation.metrics import csfe, mae, msfe, oos_r2, relative_msfe
from macrocast.evaluation.pbsv import (
    compute_pbsv,
    model_accordance_score,
    oshapley_vi,
)
from macrocast.evaluation.regime import RegimeResult, regime_conditional_msfe
from macrocast.evaluation.marginal import (
    MarginalEffect,
    marginal_contribution,
    marginal_contribution_all,
    oos_r2_panel,
)

__all__ = [
    # metrics
    "msfe",
    "mae",
    "relative_msfe",
    "csfe",
    "oos_r2",
    # decomposition
    "decompose_treatment_effects",
    "DecompositionResult",
    # PBSV
    "oshapley_vi",
    "compute_pbsv",
    "model_accordance_score",
    # dual weights
    "krr_dual_weights",
    "tree_dual_weights",
    "nn_dual_weights",
    "effective_history_length",
    "top_analogies",
    # MCS
    "mcs",
    "MCSResult",
    # DM test
    "dm_test",
    "DMResult",
    # regime
    "regime_conditional_msfe",
    "RegimeResult",
    # horse race
    "HorseRaceResult",
    "relative_msfe_table",
    "best_spec_table",
    "mcs_membership_table",
    "dm_vs_benchmark_table",
    "horserace_summary",
    # marginal contribution (CLSS 2021 Eq. 11-12)
    "oos_r2_panel",
    "MarginalEffect",
    "marginal_contribution",
    "marginal_contribution_all",
]
