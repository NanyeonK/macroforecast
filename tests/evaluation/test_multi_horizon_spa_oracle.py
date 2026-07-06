from __future__ import annotations

import numpy as np
import pandas as pd

import macroforecast as mf


def test_multi_horizon_spa_constant_dominance_oracle() -> None:
    n_obs = 6
    dominant = pd.DataFrame({"h1": np.ones(n_obs), "h2": np.ones(n_obs)})
    dominated = -dominant
    expected = np.sqrt(n_obs) / 1e-6

    for statistic in ("uspa", "aspa"):
        positive = mf.tests.multi_horizon_spa_test(
            dominant,
            statistic=statistic,
            n_boot=9,
            block_length=3,
            random_state=1,
        )
        negative = mf.tests.multi_horizon_spa_test(
            dominated,
            statistic=statistic,
            n_boot=9,
            block_length=3,
            random_state=1,
        )

        assert np.isclose(positive.statistic, expected)
        assert positive.p_value == 0.0
        assert positive.decision is True
        assert np.isclose(negative.statistic, -expected)
        assert negative.p_value == 1.0
        assert negative.decision is False
