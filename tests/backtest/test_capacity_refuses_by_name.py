"""
The capacity diagnostics refuse, by name, what `get_capacity_report` does.

The tool's schema refuses a repeated ticker and a target weight that is
not finite; the library functions behind it did not. A repeated ticker
collapsed into one entry while the universe read as longer than it was, a
NaN weight made its ticker's capacity NaN and chose it as the binding
constraint, and an infinite weight reported a capacity of exactly 0.0.
A NaN participation rate, and a NaN or infinite share count, came back as
NaN or infinite answers that read as measurements. Each is now a
ValidationError naming the input, with an ordinary portfolio as the null
case. See the CHANGELOG entry of 2026-09-28.
"""

from __future__ import annotations

import pytest

from standard_quant_tools.backtest.constraints import (
    capacity_report,
    days_to_liquidate,
    sector_exposure,
)
from standard_quant_tools.error import ValidationError

NAN, INF = float("nan"), float("inf")
ADV = {"A": 1e7, "B": 2e7}


class TestCapacityReport:
    def test_a_repeated_ticker_is_refused(self):
        with pytest.raises(ValidationError, match=r"tickers repeats \['A'\]"):
            capacity_report(["A", "B", "A"], ADV, {"A": 0.5, "B": 0.5}, 0.1)

    @pytest.mark.parametrize("bad", [NAN, INF, -INF])
    def test_a_non_finite_target_weight_is_refused_by_name(self, bad):
        with pytest.raises(
            ValidationError, match=r"target_weights is not finite at \['A'\]"
        ):
            capacity_report(["A", "B"], ADV, {"A": bad, "B": 0.5}, 0.1)

    def test_a_nan_participation_rate_is_refused(self):
        """NaN passes `<= 0`, and every capacity came back NaN."""
        with pytest.raises(ValidationError, match="max_participation"):
            capacity_report(["A", "B"], ADV, {"A": 0.5, "B": 0.5}, NAN)

    def test_an_ordinary_portfolio_binds_on_its_thinnest_name(self):
        """The null case."""
        result = capacity_report(["A", "B"], ADV, {"A": 0.5, "B": 0.5}, 0.1)
        assert result["binding_ticker"] == "A"
        assert result["max_account_size"] == pytest.approx(2e6)


class TestDaysToLiquidate:
    @pytest.mark.parametrize("bad", [NAN, INF, -INF])
    def test_a_non_finite_share_count_is_refused(self, bad):
        with pytest.raises(ValidationError, match="shares must be a finite number"):
            days_to_liquidate(bad, 1e6, 0.1)

    def test_a_finite_position_is_measured(self):
        """The null case."""
        assert days_to_liquidate(500_000, 1_000_000, 0.1) == pytest.approx(5.0)


class TestSectorExposure:
    def test_a_non_finite_weight_is_refused_by_name(self):
        """One NaN made its whole sector's total NaN."""
        with pytest.raises(ValidationError, match=r"weights is not finite at \['A'\]"):
            sector_exposure({"A": NAN, "B": 0.5}, {"A": "Tech", "B": "Tech"})

    def test_finite_weights_sum_by_sector(self):
        """The null case."""
        totals = sector_exposure({"A": 0.3, "B": 0.2}, {"A": "Tech", "B": "Tech"})
        assert totals == {"Tech": pytest.approx(0.5)}
