"""
Two drawdown functions, two `start` conventions, one bar apart.

`metrics.diagnostics.drawdown_periods` dates an episode from the PEAK.
`analysis.diagnostics.drawdown_profile` dates it from the FIRST BAR
UNDERWATER. Both are reachable from the agent surface and both return
something called `DrawdownEpisode`, with different fields.

Neither is wrong -- each matches the duration its own result reports, peak
-> recovery in one and underwater -> recovery in the other. This test exists
because that made the difference invisible: they agree on trough and on
depth, which is exactly the pattern that reads as "these are the same
measurement" right up until someone compares the starts.

If this test fails, one of the two conventions changed. That may be correct,
but it is a change to numbers the agent surface reports and it needs to be
deliberate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.analysis.diagnostics import drawdown_profile
from standard_quant_tools.metrics.diagnostics import drawdown_periods


def _curve(seed: int = 0) -> pd.Series:
    """A curve with one unambiguous drawdown: rise, fall, full recovery."""
    index = pd.date_range("2024-01-01", periods=61, freq="D")
    values = np.empty(61)
    values[:11] = np.linspace(1.00, 1.20, 11)
    values[11:26] = np.linspace(1.20, 0.95, 15)
    values[26:46] = np.linspace(0.95, 1.25, 20)
    values[46:] = np.linspace(1.25, 1.10, 15)
    return pd.Series(values, index=index)


class TestTheTwoConventions:
    def test_they_agree_on_trough_and_depth(self):
        equity = _curve()
        returns = equity.pct_change().dropna()
        aligned = equity.loc[returns.index[0] :]

        peak_based = drawdown_periods(aligned).iloc[0]
        underwater_based = drawdown_profile(returns)["worst_drawdowns"][0]

        assert str(peak_based["trough"].date()) == underwater_based["trough"][:10]
        assert float(peak_based["depth"]) == pytest.approx(
            underwater_based["depth"], abs=1e-6
        )

    def test_the_starts_are_exactly_one_bar_apart(self):
        """The whole point. Not an error, not a NaN -- one bar."""
        equity = _curve()
        returns = equity.pct_change().dropna()
        aligned = equity.loc[returns.index[0] :]

        peak_based = pd.Timestamp(drawdown_periods(aligned).iloc[0]["start"])
        underwater_based = pd.Timestamp(
            drawdown_profile(returns)["worst_drawdowns"][0]["start"]
        )

        assert underwater_based - peak_based == pd.Timedelta(days=1), (
            f"peak-based start {peak_based.date()}, underwater-based start "
            f"{underwater_based.date()}. These are one daily bar apart by "
            f"construction; if that changed, one convention moved."
        )

    def test_the_peak_based_start_really_is_the_peak(self):
        equity = _curve()
        returns = equity.pct_change().dropna()
        aligned = equity.loc[returns.index[0] :]

        start = pd.Timestamp(drawdown_periods(aligned).iloc[0]["start"])
        # The RUNNING max, not the global one -- this curve recovers to a
        # higher high later, which is what makes it a recovered drawdown.
        assert aligned.loc[start] == aligned.cummax().loc[start]
        assert aligned.loc[start] == pytest.approx(1.20)

    def test_the_underwater_based_start_really_is_below_the_peak(self):
        equity = _curve()
        returns = equity.pct_change().dropna()

        reconstructed = (1.0 + returns).cumprod()
        drawdown = reconstructed / reconstructed.cummax() - 1.0
        start = pd.Timestamp(drawdown_profile(returns)["worst_drawdowns"][0]["start"])

        assert drawdown.loc[start] < 0
        previous = drawdown.index[drawdown.index.get_loc(start) - 1]
        assert drawdown.loc[previous] == 0.0
