"""
Sizing one position, as a library function the tool calls.

The stop-distance and share-count arithmetic lived inside
`get_position_size`, so its refusals -- a stop or a share count past the
float range, which used to escape as a NaN worst-case loss or a bare
OverflowError from `int()` -- existed only at the tool. `size_position`
holds the arithmetic now, and the tool calls it, so a direct caller sizes
and refuses the same way; it also refuses a position VALUE past the float
range, which the tool used to report as an infinity. The inputs the tool's
schema checks are checked again here. See the CHANGELOG entry of 2026-09-28.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.error import ValidationError
from standard_quant_tools.portfolio.position_sizing import size_position

NAN, INF = float("nan"), float("inf")


class TestTheArithmetic:
    def test_fixed_risk_buys_the_risk_budget_behind_the_stop(self):
        """The null case: $1,000 of risk behind a $4 stop is 250 shares."""
        result = size_position(100_000.0, 100.0, 2.0, atr_multiplier=2.0)
        assert result["stop_distance"] == pytest.approx(4.0)
        assert result["shares_fixed_risk"] == 250
        assert result["max_loss_fixed_risk"] == pytest.approx(1_000.0)
        assert result["position_value_fixed_risk"] == pytest.approx(25_000.0)
        assert result["recommended_sizing"] == "fixed_risk"
        assert result["kelly_fraction"] is None

    def test_a_positive_edge_recommends_half_kelly(self):
        """b = 2, p = 0.55: f = (2 x 0.55 - 0.45) / 2 = 0.325."""
        result = size_position(
            100_000.0, 100.0, 2.0, win_rate=0.55, avg_win_pct=0.06, avg_loss_pct=0.03
        )
        assert result["kelly_fraction"] == pytest.approx(0.325)
        assert result["shares_half_kelly"] == 162
        assert result["recommended_sizing"] == "half_kelly"
        assert result["recommended_shares"] == 162

    def test_no_edge_falls_back_to_fixed_risk(self):
        result = size_position(
            100_000.0, 100.0, 2.0, win_rate=0.30, avg_win_pct=0.02, avg_loss_pct=0.05
        )
        assert result["kelly_fraction"] == 0.0
        assert result["recommended_sizing"] == "fixed_risk"

    def test_a_flat_market_sizes_nothing_rather_than_dividing_by_zero(self):
        result = size_position(100_000.0, 100.0, 0.0)
        assert result["stop_distance"] == 0.0
        assert result["shares_fixed_risk"] == 0


class TestOverflowIsRefusedByName:
    def test_a_stop_past_the_float_range(self):
        with pytest.raises(ValidationError, match="atr_multiplier=1e.308 .*stop"):
            size_position(100_000.0, 100.0, 20.0, atr_multiplier=1e308)

    def test_a_share_count_past_the_float_range(self):
        with pytest.raises(ValidationError, match="not a finite number of shares"):
            size_position(
                1e308, 100.0, 20.0, risk_per_trade_pct=1.0, atr_multiplier=1e-10
            )

    def test_a_position_value_past_the_float_range(self):
        """The count fits; the count times the price does not."""
        with pytest.raises(ValidationError, match="worth more than the float range"):
            size_position(1e308, 100.0, 1.0, risk_per_trade_pct=1.0, atr_multiplier=1.0)

    def test_a_payoff_ratio_past_the_float_range(self):
        with pytest.raises(ValidationError, match="finite payoff ratio"):
            size_position(
                100_000.0,
                100.0,
                2.0,
                win_rate=0.5,
                avg_win_pct=1e10,
                avg_loss_pct=1e-310,
            )


class TestTheSchemaRulesHoldForADirectCaller:
    @pytest.mark.parametrize(
        "kwargs, name",
        [
            ({"account_equity": 0.0}, "account_equity"),
            ({"account_equity": NAN}, "account_equity"),
            ({"account_equity": INF}, "account_equity"),
            ({"risk_per_trade_pct": 0.0}, "risk_per_trade_pct"),
            ({"risk_per_trade_pct": 1.5}, "risk_per_trade_pct"),
            ({"atr_multiplier": -2.0}, "atr_multiplier"),
            ({"win_rate": 2.0}, "win_rate"),
            ({"avg_win_pct": -0.1}, "avg_win_pct"),
            ({"avg_loss_pct": 0.0}, "avg_loss_pct"),
            ({"last_atr": -1.0}, "last_atr"),
        ],
    )
    def test_is_refused_naming_the_input(self, kwargs, name):
        args = {"account_equity": 100_000.0, "last_close": 100.0, "last_atr": 2.0}
        args.update(kwargs)
        with pytest.raises(ValidationError, match=name):
            size_position(**args)


class TestTheToolIsThisFunction:
    def test_the_tool_reports_what_the_library_computes(self, monkeypatch):
        from unittest.mock import MagicMock

        from standard_quant_tools.agent.models import PositionSizerInput
        from standard_quant_tools.agent.runtimes.portfolio.tools import (
            get_position_size,
        )
        from standard_quant_tools.data.factory import DataFactory

        dates = pd.bdate_range("2022-01-03", periods=120)
        close = np.linspace(100.0, 120.0, 120)
        frame = pd.DataFrame(
            {
                "Open": close,
                "High": close + 10.0,
                "Low": close - 10.0,
                "Close": close,
                "Volume": np.full(120, 1e6),
            },
            index=dates,
        )
        provider = MagicMock()
        provider.get_ohlcv.return_value = frame
        monkeypatch.setattr(DataFactory, "get_provider", lambda *a, **kw: provider)

        tool = get_position_size(
            PositionSizerInput(
                symbol="AAA",
                start_date="2022-01-03",
                end_date="2022-06-30",
                account_equity=100_000.0,
                win_rate=0.55,
                avg_win_pct=0.06,
                avg_loss_pct=0.03,
            )
        )
        library = size_position(
            100_000.0,
            tool.last_close,
            tool.atr,
            win_rate=0.55,
            avg_win_pct=0.06,
            avg_loss_pct=0.03,
        )
        assert tool.shares_fixed_risk == library["shares_fixed_risk"]
        assert tool.shares_half_kelly == library["shares_half_kelly"]
        assert tool.recommended_sizing == library["recommended_sizing"]
