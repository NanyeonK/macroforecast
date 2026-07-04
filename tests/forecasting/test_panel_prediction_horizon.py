"""The panel-model horizon helper must return the true positional distance.

Regression history (issue #423):

1. ``_panel_prediction_horizon`` originally returned
   ``positions[1] - positions[0] + 1``, so every panel prediction was tagged
   with a horizon one larger than the real number of steps from the origin
   to the target date.
2. A follow-up patch (``6e3a2ec2``) dropped the "+ 1" but added a
   ``max(1, ...)`` floor to compensate for a SEPARATE bug: the panel test
   window (``WindowSpec.origins()``) still included the origin's own date as
   the first test row, so its distance-0 row needed flooring up to 1 -- which
   collided with the genuinely-1-step-away row, so every row of a multi-step
   path read horizon=1.

The real fix is at the window level: ``WindowSpec.origins(exclude_origin=True)``
now starts the panel test slice at ``origin_pos + 1``, so the origin never
appears in ``base_index`` in production. With that fixed, this helper needs no
floor at all -- it is simply ``positions[1] - positions[0]``, with 0 being the
correct (if no longer reachable in production) answer for "date == origin".
"""
import pandas as pd

from macroforecast.forecasting.runner import _panel_prediction_horizon


def test_horizon_when_origin_is_before_the_test_panel():
    # origin is the last training date; the test panel starts one step later.
    # This is the shape WindowSpec.origins(exclude_origin=True) now produces
    # for every panel run (#423): the origin itself is never in base_index.
    base = pd.date_range("2020-02-01", periods=4, freq="MS")  # origin + 1..4 months
    origin = pd.Timestamp("2020-01-01")
    for steps, date in enumerate(base, start=1):
        assert _panel_prediction_horizon(date, origin=origin, base_index=base, default=-1) == steps


def test_horizon_when_origin_is_in_the_test_panel():
    # This input shape (origin literally inside base_index) can no longer
    # arise from a real panel run post-#423 -- WindowSpec.origins() excludes
    # the origin from the panel test slice. The helper is still exercised
    # directly here to pin the true (unfloored) positional distance: the
    # origin's own row is horizon 0, not clamped up to 1. Clamping to 1 was
    # exactly the bug that made every row of a multi-step path read
    # horizon=1 (see module docstring).
    base = pd.date_range("2020-01-01", periods=4, freq="MS")
    origin = pd.Timestamp("2020-01-01")
    expected = [0, 1, 2, 3]
    for exp, date in zip(expected, base):
        assert _panel_prediction_horizon(date, origin=origin, base_index=base, default=-1) == exp


def test_falls_back_to_default_when_date_not_locatable():
    base = pd.date_range("2020-02-01", periods=3, freq="MS")
    origin = pd.Timestamp("2020-01-01")
    missing = pd.Timestamp("1999-01-01")
    assert _panel_prediction_horizon(missing, origin=origin, base_index=base, default=7) == 7
