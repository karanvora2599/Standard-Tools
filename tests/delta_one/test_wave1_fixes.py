"""
Three fields that could not say what they claimed.

`hedge_effectiveness.tracking_error` was the hedged series' own sigma --
`volatility_after` under a second name. `index_basket.missing_symbols`
could never be non-empty, because a missing price was refused before the
list was built. `reset_spread_monitor` carried the degenerate-baseline
retry accumulators through a reset. Each is planted.
"""

import numpy as np
import pytest

from standard_quant_tools.delta_one.baskets import index_basket
from standard_quant_tools.delta_one.hedging import hedge_effectiveness, tracking_error
from standard_quant_tools.delta_one.streaming import (
    new_spread_monitor,
    reset_spread_monitor,
    update_spread_monitor,
)
from standard_quant_tools.error import ValidationError


class TestTrackingError:
    def test_it_is_the_instrument_s_active_sigma_not_the_hedged_sigma(self):
        rng = np.random.default_rng(0)
        hedge = rng.normal(scale=0.01, size=250)
        portfolio = 0.6 * hedge + rng.normal(scale=0.004, size=250)
        result = hedge_effectiveness(
            portfolio_returns=portfolio,
            hedge_returns=hedge,
            hedge_ratio=-0.6,
            window=60,
        )
        expected = tracking_error(portfolio, hedge, periods_per_year=252)
        assert result["tracking_error"] == pytest.approx(expected)
        # The one-for-one active sigma is larger than the hedged sigma at
        # the fitted ratio (0.4 of the hedge's variance is left in it);
        # the two fields were identical before.
        assert result["tracking_error"] > 1.3 * result["volatility_after"]


class TestMissingConstituents:
    def test_a_null_price_is_named_and_left_out_not_refused(self):
        result = index_basket(
            [
                {"symbol": "AAA", "price": 100.0, "weight": 0.5},
                {"symbol": "BBB", "price": None, "weight": 0.3},
                {"symbol": "CCC", "price": 50.0, "weight": 0.2},
            ]
        )
        assert result["missing_symbols"] == ["BBB"]
        assert result["n_constituents"] == 2
        assert result["basket_value"] == pytest.approx(0.5 * 100 + 0.2 * 50)
        assert any("no price" in w and "BBB" in w for w in result["warnings"])
        with pytest.raises(ValidationError, match="no `price`"):
            index_basket([{"symbol": "AAA", "weight": 1.0}])
        with pytest.raises(ValidationError, match="every constituent's price is null"):
            index_basket([{"symbol": "AAA", "price": None, "weight": 1.0}])


class TestMonitorReset:
    def test_a_reset_clears_the_degenerate_retry_accumulators(self):
        state = new_spread_monitor(warmup=12)
        # A flat spread makes the baseline degenerate, which starts the retry.
        state = update_spread_monitor(
            state, primary=[100.0] * 15, reference=[100.0] * 15
        )["state"]
        state["degenerate_n"] = 3
        state["degenerate_mean"] = 1.0
        state["degenerate_m2"] = 2.0
        for keep in (False, True):
            fresh = reset_spread_monitor(state, keep_baseline=keep)
            assert fresh["degenerate_n"] == 0
            assert fresh["degenerate_mean"] == 0.0
            assert fresh["degenerate_m2"] == 0.0
