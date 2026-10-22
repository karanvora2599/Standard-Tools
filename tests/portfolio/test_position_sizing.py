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

Kelly is held to its definition: the position f that maximises
g(f) = p ln(1 + f W) + q ln(1 - f L), found numerically and compared with
the closed form. The library used to report (b p - q) / b -- which is
f* x L, the fraction of equity an average loss costs at full Kelly -- and
buy it as a position. See the CHANGELOG entry of 2026-10-22.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize_scalar

from standard_quant_tools.error import ValidationError
from standard_quant_tools.portfolio.position_sizing import (
    kelly_log_growth,
    kelly_notional_fraction,
    size_position,
)

NAN, INF = float("nan"), float("inf")

#: The planted edge: a 55% win rate, wins of 6% and losses of 3% of the
#: position. f* = 0.55 / 0.03 - 0.45 / 0.06 = 10.8333...
EDGE = {"win_rate": 0.55, "avg_win_pct": 0.06, "avg_loss_pct": 0.03}
F_STAR = 0.55 / 0.03 - 0.45 / 0.06


def _numeric_argmax(p: float, win: float, loss: float) -> float:
    """The growth-optimal fraction found by search, not by the formula."""
    upper = (1.0 / loss) * (1.0 - 1e-12) if p < 1 else 1e6
    found = minimize_scalar(
        lambda f: -(p * math.log1p(f * win) + (1 - p) * math.log1p(-f * loss)),
        bounds=(0.0, upper),
        method="bounded",
        options={"xatol": 1e-12, "maxiter": 10_000},
    )
    assert found.success
    return float(found.x)


class TestTheArithmetic:
    def test_fixed_risk_buys_the_risk_budget_behind_the_stop(self):
        """The null case: $1,000 of risk behind a $4 stop is 250 shares."""
        result = size_position(100_000.0, 100.0, 2.0, atr_multiplier=2.0)
        assert result["stop_distance"] == pytest.approx(4.0)
        assert result["shares_fixed_risk"] == 250
        assert result["max_loss_fixed_risk"] == pytest.approx(1_000.0)
        assert result["position_value_fixed_risk"] == pytest.approx(25_000.0)
        assert result["recommended_sizing"] == "fixed_risk"
        assert result["kelly_notional_fraction"] is None
        assert result["warnings"] == []

    def test_a_flat_market_sizes_nothing_rather_than_dividing_by_zero(self):
        result = size_position(100_000.0, 100.0, 0.0)
        assert result["stop_distance"] == 0.0
        assert result["shares_fixed_risk"] == 0


class TestKellyIsTheGrowthOptimalPosition:
    """
    p = 0.55, W = 6%, L = 3%: full Kelly is a position of 10.83 times
    equity, half-Kelly 5.42. The library used to size 162 shares of a $100
    stock on $100,000 -- 16.2% of equity, f* x L / 2 -- a thirty-third of
    half-Kelly. Since the CHANGELOG entry of 2026-10-22 it reports both
    fractions and sizes half-Kelly notional, capped at max_leverage.
    """

    @pytest.mark.parametrize(
        "p, win, loss",
        [
            (0.55, 0.06, 0.03),
            (0.55, 0.05, 0.02),
            (0.60, 0.02, 0.02),
            (0.45, 0.10, 0.04),
            (0.51, 0.50, 0.50),
            (0.90, 0.01, 0.08),
        ],
    )
    def test_the_closed_form_is_the_numeric_argmax(self, p, win, loss):
        f_star = kelly_notional_fraction(p, win, loss)
        assert f_star > 0
        assert abs(f_star - _numeric_argmax(p, win, loss)) < 1e-6

    @pytest.mark.parametrize(
        "p, win, loss",
        [(0.55, 0.06, 0.03), (0.30, 0.02, 0.05), (0.45, 0.10, 0.04), (0.8, 0.3, 0.9)],
    )
    def test_the_risk_fraction_is_the_number_the_library_used_to_report(
        self, p, win, loss
    ):
        """(b p - q) / b with b = W / L, the old `kelly_fraction` before it
        was floored at zero, is f* x L."""
        payoff = win / loss
        old = (payoff * p - (1 - p)) / payoff
        result = size_position(
            100_000.0, 100.0, 2.0, win_rate=p, avg_win_pct=win, avg_loss_pct=loss
        )
        assert result["kelly_risk_fraction"] == pytest.approx(old, abs=1e-12)
        assert result["kelly_notional_fraction"] == pytest.approx(old / loss, abs=1e-9)

    def test_half_kelly_is_sized_as_a_position_and_capped_at_one(self):
        """At the default cap of 1x the position is the whole account:
        1,000 shares of a $100 stock on $100,000, not 162."""
        result = size_position(100_000.0, 100.0, 2.0, **EDGE)
        assert result["kelly_notional_fraction"] == pytest.approx(F_STAR)
        assert result["kelly_risk_fraction"] == pytest.approx(0.325)
        assert result["half_kelly_notional_fraction"] == pytest.approx(F_STAR / 2)
        assert result["kelly_position_fraction"] == 1.0
        assert result["kelly_capped_by_max_leverage"] is True
        assert result["shares_half_kelly"] == 1_000
        assert result["recommended_sizing"] == "half_kelly"
        assert result["recommended_shares"] == 1_000
        assert any(
            "max_leverage=1 caps the position" in w for w in result["warnings"]
        ), result["warnings"]

    def test_a_higher_cap_buys_half_kelly_itself(self):
        """5.4167 x $100,000 / $100 = 5,416.67 shares, rounded down."""
        result = size_position(100_000.0, 100.0, 2.0, max_leverage=10.0, **EDGE)
        assert result["kelly_capped_by_max_leverage"] is False
        assert result["shares_half_kelly"] == 5_416
        assert result["portfolio_pct_half_kelly"] == pytest.approx(5.416)
        assert not any("caps the position" in w for w in result["warnings"])

    def test_the_growth_per_trade_at_each_size(self):
        """g(f*) = 0.098557, g(f*/2) = 0.074977, g(1) = 0.018341: half
        Kelly keeps three quarters of the growth, and the capped position
        less than a fifth."""
        result = size_position(100_000.0, 100.0, 2.0, **EDGE)

        def g(f):
            return 0.55 * math.log(1 + 0.06 * f) + 0.45 * math.log(1 - 0.03 * f)

        assert result["log_growth_per_trade_full_kelly"] == pytest.approx(g(F_STAR))
        assert result["log_growth_per_trade_half_kelly"] == pytest.approx(g(F_STAR / 2))
        assert result["log_growth_per_trade_recommended"] == pytest.approx(g(1.0))
        assert g(F_STAR) == pytest.approx(0.098557, abs=1e-6)
        assert g(1.0) == pytest.approx(0.018341, abs=1e-6)

    def test_no_edge_is_zero_shares_and_a_warning_not_fixed_risk(self):
        """p W - q L = 0.3 x 0.02 - 0.7 x 0.05 = -0.029. The old sizer fell
        back to the fixed-risk position without a word -- recommending a
        trade its own statistics said loses money."""
        result = size_position(
            100_000.0, 100.0, 2.0, win_rate=0.30, avg_win_pct=0.02, avg_loss_pct=0.05
        )
        assert result["kelly_notional_fraction"] == pytest.approx(
            0.3 / 0.05 - 0.7 / 0.02
        )
        assert result["kelly_position_fraction"] == 0.0
        assert result["shares_half_kelly"] == 0
        assert result["recommended_sizing"] == "half_kelly"
        assert result["recommended_shares"] == 0
        assert result["shares_fixed_risk"] == 250  # still computed, not offered
        assert result["log_growth_per_trade_full_kelly"] is None
        assert any("give no edge" in w and "-0.029" in w for w in result["warnings"])

    def test_a_win_rate_of_one_is_unbounded_and_sized_at_the_cap(self):
        result = size_position(
            100_000.0, 100.0, 2.0, win_rate=1.0, avg_win_pct=0.02, avg_loss_pct=0.05
        )
        assert result["kelly_notional_fraction"] is None
        assert result["kelly_risk_fraction"] is None
        assert result["shares_half_kelly"] == 1_000
        assert result["kelly_capped_by_max_leverage"] is True
        assert any("no trade in the sample lost" in w for w in result["warnings"])

    def test_no_winning_trade_is_unbounded_below_and_zero_shares(self):
        result = size_position(
            100_000.0, 100.0, 2.0, win_rate=0.5, avg_win_pct=0.0, avg_loss_pct=0.05
        )
        assert result["kelly_notional_fraction"] is None
        assert result["shares_half_kelly"] == 0
        assert any("unbounded below" in w for w in result["warnings"])

    def test_a_half_kelly_worth_less_than_one_share_says_so(self):
        result = size_position(
            1_000.0, 900.0, 2.0, win_rate=0.51, avg_win_pct=0.5, avg_loss_pct=0.5
        )
        assert result["kelly_position_fraction"] == pytest.approx(0.02)
        assert result["recommended_shares"] == 0
        assert any("less than one share" in w for w in result["warnings"])


class TestHowSureTheFractionIs:
    def test_n_trades_gives_the_wilson_interval_and_its_kelly_ends(self):
        """22 wins in 40 trades: the 95% Wilson interval is [0.3983, 0.6929]
        (centre (p + z^2/2n) / (1 + z^2/n), half-width
        z sqrt(p q / n + z^2 / 4n^2) / (1 + z^2/n)), and f* at its ends is
        p/L - q/W at each."""
        result = size_position(100_000.0, 100.0, 2.0, n_trades=40, **EDGE)
        low, high = result["win_rate_interval_95"]
        assert low == pytest.approx(0.398293, abs=1e-5)
        assert high == pytest.approx(0.692944, abs=1e-5)
        f_low, f_high = result["kelly_notional_fraction_interval_95"]
        assert f_low == pytest.approx(low / 0.03 - (1 - low) / 0.06)
        assert f_high == pytest.approx(high / 0.03 - (1 - high) / 0.06)
        assert f_low < F_STAR < f_high

    def test_an_edge_the_sample_does_not_establish_is_flagged(self):
        """p = 0.55 with W = L: f* = 0.1, and at 20 trades the interval's
        low end has no edge at all."""
        result = size_position(
            100_000.0,
            100.0,
            2.0,
            win_rate=0.55,
            avg_win_pct=0.05,
            avg_loss_pct=0.05,
            n_trades=20,
        )
        assert result["kelly_notional_fraction_interval_95"][0] < 0
        assert any("not established at 95%" in w for w in result["warnings"])

    def test_a_large_sample_establishes_it(self):
        result = size_position(
            100_000.0,
            100.0,
            2.0,
            win_rate=0.55,
            avg_win_pct=0.05,
            avg_loss_pct=0.05,
            n_trades=5_000,
        )
        assert result["kelly_notional_fraction_interval_95"][0] > 0
        assert not any("not established" in w for w in result["warnings"])


class TestGrowth:
    def test_a_loss_that_takes_the_account_is_minus_infinity(self):
        assert kelly_log_growth(40.0, 0.5, 0.05, 0.03) == -math.inf

    def test_no_position_is_no_growth(self):
        assert kelly_log_growth(0.0, 0.55, 0.06, 0.03) == 0.0


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
            ({"max_leverage": 0.0}, "max_leverage"),
            ({"max_leverage": INF}, "max_leverage"),
            ({**EDGE, "n_trades": 0}, "n_trades"),
            ({**EDGE, "n_trades": 2.5}, "n_trades"),
        ],
    )
    def test_is_refused_naming_the_input(self, kwargs, name):
        args = {"account_equity": 100_000.0, "last_close": 100.0, "last_atr": 2.0}
        args.update(kwargs)
        with pytest.raises(ValidationError, match=name):
            size_position(**args)

    def test_part_of_the_kelly_statistics_is_refused(self):
        """A win rate alone used to be dropped without a word, and the
        position sized as if Kelly had never been asked for."""
        with pytest.raises(
            ValidationError, match="win_rate given without avg_win_pct, avg_loss_pct"
        ):
            size_position(100_000.0, 100.0, 2.0, win_rate=0.55)

    def test_n_trades_without_the_statistics_is_refused(self):
        with pytest.raises(ValidationError, match="n_trades=40 was given without"):
            size_position(100_000.0, 100.0, 2.0, n_trades=40)


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
                max_leverage=3.0,
                n_trades=60,
                **EDGE,
            )
        )
        library = size_position(
            100_000.0,
            tool.last_close,
            tool.atr,
            max_leverage=3.0,
            n_trades=60,
            **EDGE,
        )
        assert tool.shares_fixed_risk == library["shares_fixed_risk"]
        assert tool.shares_half_kelly == library["shares_half_kelly"]
        assert tool.recommended_sizing == library["recommended_sizing"]
        assert tool.kelly_notional_fraction == pytest.approx(F_STAR, abs=1e-6)
        assert tool.kelly_risk_fraction == pytest.approx(0.325, abs=1e-6)
        assert tool.kelly_position_fraction == 3.0
        assert tool.kelly_capped_by_max_leverage is True
        assert tool.win_rate_interval_95 == pytest.approx(
            library["win_rate_interval_95"], abs=1e-6
        )
        assert tool.warnings == library["warnings"]
        assert any("max_leverage=3 caps" in w for w in tool.warnings)

    def test_a_null_kelly_figure_is_explained_beside_it(self, monkeypatch):
        """A win rate of 1 leaves f* unbounded: null in the result, with the
        reason in its warnings."""
        from unittest.mock import MagicMock

        from standard_quant_tools.agent.models import PositionSizerInput
        from standard_quant_tools.agent.runtimes.portfolio.tools import (
            get_position_size,
        )
        from standard_quant_tools.data.factory import DataFactory

        dates = pd.bdate_range("2022-01-03", periods=60)
        close = np.full(60, 100.0)
        frame = pd.DataFrame(
            {
                "Open": close,
                "High": close + 1.0,
                "Low": close - 1.0,
                "Close": close,
                "Volume": np.full(60, 1e6),
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
                end_date="2022-03-31",
                account_equity=100_000.0,
                win_rate=1.0,
                avg_win_pct=0.02,
                avg_loss_pct=0.05,
            )
        )
        assert tool.kelly_notional_fraction is None
        assert any("kelly_notional_fraction" in w for w in tool.warnings)
