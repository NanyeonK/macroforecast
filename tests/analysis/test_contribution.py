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


def _panel_covariance_master(
    shocks: np.ndarray,
    *,
    seed: int,
    beta: float = 0.6,
    noise_sd: float = 0.25,
) -> pd.DataFrame:
    dates = pd.date_range("2000-01-31", periods=len(shocks), freq="ME", name="date")
    arms = [("base_a", 0), ("treated_a", 1), ("base_b", 0), ("treated_b", 1)]
    rng = np.random.default_rng(seed)
    rows = []
    for date_idx, date in enumerate(dates):
        for arm, tag in arms:
            centered_tag = float(tag) - 0.5
            error = float(shocks[date_idx]) * centered_tag + noise_sd * rng.standard_normal()
            squared_error = 10.0 + beta * float(tag) + error
            rows.append(
                {
                    "target": "Y",
                    "horizon": 1,
                    "date": date,
                    "origin": date,
                    "arm": arm,
                    "contender": arm,
                    "actual": 0.0,
                    "prediction": np.sqrt(squared_error),
                    "tag_X": tag,
                }
            )
    return pd.DataFrame(rows)


def _ar1_shocks(
    periods: int,
    rng: np.random.Generator,
    *,
    rho: float = 0.65,
) -> np.ndarray:
    shocks = np.zeros(periods, dtype=float)
    innovations = rng.standard_normal(periods)
    for idx, innovation in enumerate(innovations):
        shocks[idx] = (rho * shocks[idx - 1] if idx else 0.0) + innovation
    return shocks


def _mc_axis_se(
    *,
    periods: int,
    reps: int,
    lag: int,
    serial: bool,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(123)
    coefs: list[float] = []
    dk_ses: list[float] = []
    hc0_ses: list[float] = []
    for rep in range(reps):
        shocks = _ar1_shocks(periods, rng) if serial else rng.standard_normal(periods)
        frame = _panel_covariance_master(shocks, seed=1000 + rep)
        dk = axis_contribution(
            frame,
            features=["X"],
            outcome="squared_error",
            fixed_effects=("date",),
            vcov="driscoll_kraay",
            hac_lags=lag,
        ).set_index("term")
        hc0 = axis_contribution(
            frame,
            features=["X"],
            outcome="squared_error",
            fixed_effects=("date",),
            vcov="hc0",
        ).set_index("term")
        coefs.append(float(dk.loc["tag_X=1", "coef"]))
        dk_ses.append(float(dk.loc["tag_X=1", "se"]))
        hc0_ses.append(float(hc0.loc["tag_X=1", "se"]))
    return float(np.std(coefs, ddof=1)), float(np.mean(dk_ses)), float(np.mean(hc0_ses))


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
        vcov="hac",
    )
    expected = _statsmodels_crosscheck(master)
    by_term = got.set_index("term")

    for term in ["tag_NL=1", "tag_CV=poos"]:
        assert by_term.loc[term, "coef"] == pytest.approx(expected.params[term], abs=1e-10)
        assert by_term.loc[term, "se"] == pytest.approx(expected.bse[term], abs=1e-10)
    assert set(got["fe_spec"]) == {"joint(target,horizon,date)"}
    assert got.attrs["macroforecast_axis_contribution"]["hac_lags"] == 1
    assert got.attrs["macroforecast_axis_contribution"]["vcov"] == "hac"
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


def test_axis_contribution_coefficients_identical_across_vcov_choices():
    master = _tiny_master()

    fits = {
        key: axis_contribution(
            master,
            features=["NL", "CV"],
            outcome="squared_error",
            hac_lags=1,
            vcov=key,
        )
        for key in ["hc0", "hac", "driscoll_kraay", "cluster"]
    }
    base = fits["driscoll_kraay"].set_index("term").sort_index()["coef"].to_numpy()

    for key, fit in fits.items():
        coefs = fit.set_index("term").sort_index()["coef"].to_numpy()
        assert np.array_equal(coefs, base), key


def test_axis_contribution_cluster_matches_hand_built_date_cr0():
    master = _panel_covariance_master(
        np.array([0.5, -0.25, 0.75, -0.5]),
        seed=222,
    )

    got = axis_contribution(
        master,
        features=["X"],
        outcome="squared_error",
        fixed_effects=(),
        vcov="cluster",
    ).set_index("term")

    work = master.sort_values(
        ["date", "origin", "target", "horizon", "contender", "arm"],
        kind="mergesort",
    ).reset_index(drop=True)
    y = np.square(work["actual"].to_numpy(dtype=float) - work["prediction"].to_numpy(dtype=float))
    x = np.column_stack([np.ones(len(work)), work["tag_X"].to_numpy(dtype=float)])
    bread = np.linalg.inv(x.T @ x)
    beta = bread @ (x.T @ y)
    resid = y - x @ beta
    scores = x * resid[:, None]
    score_frame = pd.DataFrame(scores)
    score_frame["date"] = pd.to_datetime(work["date"])
    group_scores = score_frame.groupby("date", sort=True).sum(numeric_only=True).to_numpy()
    expected_vcov = bread @ (group_scores.T @ group_scores) @ bread

    assert got.loc["tag_X=1", "coef"] == pytest.approx(beta[1], abs=1e-12)
    assert got.loc["tag_X=1", "se"] == pytest.approx(np.sqrt(expected_vcov[1, 1]), abs=1e-12)
    assert got.attrs["macroforecast_axis_contribution"]["covariance"] == "cluster_date_cr0"


def test_axis_contribution_dk_cross_section_mc_tracks_empirical_se():
    empirical, dk_se, hc0_se = _mc_axis_se(periods=48, reps=80, lag=0, serial=False)

    assert dk_se > hc0_se
    assert hc0_se < 0.75 * empirical
    assert dk_se == pytest.approx(empirical, rel=0.25)


def test_axis_contribution_dk_serial_mc_tracks_empirical_se():
    empirical, dk_se, hc0_se = _mc_axis_se(periods=72, reps=80, lag=5, serial=True)

    assert dk_se > hc0_se
    assert hc0_se < 0.5 * empirical
    assert dk_se == pytest.approx(empirical, rel=0.45)


def test_axis_contribution_single_date_dk_and_cluster_fall_back_to_hc0():
    master = _panel_covariance_master(np.array([0.75]), seed=333)

    hc0 = axis_contribution(
        master,
        features=["X"],
        outcome="squared_error",
        fixed_effects=(),
        vcov="hc0",
    ).set_index("term")
    dk = axis_contribution(
        master,
        features=["X"],
        outcome="squared_error",
        fixed_effects=(),
        vcov="driscoll_kraay",
    ).set_index("term")
    cluster = axis_contribution(
        master,
        features=["X"],
        outcome="squared_error",
        fixed_effects=(),
        vcov="cluster",
    ).set_index("term")

    assert dk.loc["tag_X=1", "se"] == pytest.approx(hc0.loc["tag_X=1", "se"], abs=1e-12)
    assert cluster.loc["tag_X=1", "se"] == pytest.approx(hc0.loc["tag_X=1", "se"], abs=1e-12)
    assert dk.attrs["macroforecast_axis_contribution"]["single_cluster_fallback"] is True
    assert cluster.attrs["macroforecast_axis_contribution"]["single_cluster_fallback"] is True


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
