import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
from macroforecast.analysis import axis_contribution


def _tiny_master() -> pd.DataFrame:
    dates = pd.date_range("2020-01-31", periods=6, freq="ME", name="date")
    targets = ["INDPRO", "UNRATE"]
    horizons = [1, 3]
    arms = [
        ("base", 0, "kfold"),
        ("nl", 1, "kfold"),
        ("poos", 0, "poos"),
        ("nl_poos", 1, "poos"),
    ]
    rng = np.random.default_rng(123)
    rows = []
    for target_idx, target in enumerate(targets):
        for horizon in horizons:
            for date_idx, date in enumerate(dates):
                cell = 2.5 + 0.35 * target_idx + 0.08 * horizon + 0.04 * date_idx
                for arm_idx, (arm, nl, cv) in enumerate(arms):
                    poos = cv == "poos"
                    noise = 0.015 * rng.standard_normal()
                    squared_error = cell + 0.65 * nl - 0.30 * poos + noise
                    rows.append(
                        {
                            "target": target,
                            "horizon": horizon,
                            "date": date,
                            "origin": date - pd.offsets.MonthEnd(horizon),
                            "arm": arm,
                            "contender": arm,
                            "actual": 0.0,
                            "prediction": np.sqrt(squared_error),
                            "tag_NL": nl,
                            "tag_CV": cv,
                        }
                    )
    return pd.DataFrame(rows)


def _statsmodels_crosscheck(master: pd.DataFrame):
    sm = pytest.importorskip("statsmodels.api")
    work = master.sort_values(
        ["date", "origin", "target", "horizon", "contender", "arm"],
        kind="mergesort",
    ).reset_index(drop=True)
    y = np.square(work["actual"].to_numpy(dtype=float) - work["prediction"].to_numpy(dtype=float))
    x = pd.DataFrame(
        {
            "const": 1.0,
            "tag_NL=1": work["tag_NL"].astype(float),
            "tag_CV=poos": (work["tag_CV"] == "poos").astype(float),
        }
    )
    fe_key = work[["target", "horizon", "date"]].astype("string").agg("\x1f".join, axis=1)
    categories = sorted(pd.unique(fe_key), key=lambda value: str(value))
    for idx, level in enumerate(categories[1:], start=1):
        x[f"fe_{idx}"] = (fe_key == level).astype(float)
    return sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": 1, "use_correction": False})


def test_axis_contribution_matches_statsmodels_ols_hac_with_three_way_fe():
    master = _tiny_master()

    got = axis_contribution(
        master,
        features=["NL", "CV"],
        outcome="squared_error",
        hac_lags=1,
    )
    expected = _statsmodels_crosscheck(master)
    by_term = got.set_index("term")

    for term in ["tag_NL=1", "tag_CV=poos"]:
        assert by_term.loc[term, "coef"] == pytest.approx(expected.params[term], abs=1e-10)
        assert by_term.loc[term, "se"] == pytest.approx(expected.bse[term], abs=1e-10)
    assert set(got["fe_spec"]) == {"joint(target,horizon,date)"}
    assert got.attrs["macroforecast_axis_contribution"]["hac_lags"] == 1
    assert mf.axis_contribution is axis_contribution


def test_axis_contribution_interactions_with_date_indexed_state_series():
    master = _tiny_master()
    state = pd.Series(
        np.linspace(-1.0, 1.0, master["date"].nunique()),
        index=sorted(master["date"].unique()),
        name="macro_u",
    )

    got = axis_contribution(
        master,
        features=["NL"],
        outcome="squared_error",
        interactions={"macro_u": state},
        hac_lags=1,
    )

    by_term = got.set_index("term")
    assert "tag_NL=1" in by_term.index
    assert "tag_NL=1:state_macro_u" in by_term.index
    assert by_term.loc["tag_NL=1:state_macro_u", "interaction"] == "macro_u"


def test_axis_contribution_r2_outcome_matches_hand_computed_reference():
    master = _tiny_master()

    got = axis_contribution(
        master,
        features=["NL"],
        outcome="r2",
        fixed_effects=(),
        hac_lags=0,
        reference="base",
    )

    work = master.sort_values(
        ["date", "origin", "target", "horizon", "contender", "arm"],
        kind="mergesort",
    ).reset_index(drop=True)
    squared = np.square(
        work["actual"].to_numpy(dtype=float) - work["prediction"].to_numpy(dtype=float)
    )
    ref = work.loc[work["contender"] == "base", ["target", "horizon"]].copy()
    ref["squared"] = squared[work["contender"] == "base"]
    denom = ref.groupby(["target", "horizon"], dropna=False)["squared"].mean()
    expected = 1.0 - squared / denom.reindex(
        pd.MultiIndex.from_frame(work[["target", "horizon"]])
    ).to_numpy(dtype=float)

    plot_frame = got.attrs["macroforecast_axis_contribution_plot_frame"]
    np.testing.assert_allclose(plot_frame["outcome_value"].to_numpy(dtype=float), expected)
    assert set(got["reference"]) == {"base"}
    assert got.attrs["macroforecast_axis_contribution"]["outcome_metadata"]["formula"] == (
        "1 - squared_error / mean_reference_squared_error_by_target_horizon"
    )
