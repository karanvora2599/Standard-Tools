"""
The covered-interest-parity FX forward and the hedge it gives, against
answers worked by hand.

THE PLANTED QUOTE. EURUSD spot 1.1000, USD 5.30% and EUR 3.90% a year,
both money markets ACT/360, 90 days to delivery, so each leg's year
fraction is 90/360 = 0.25 and

    F = 1.1 x (1 + 0.053 x 0.25) / (1 + 0.039 x 0.25)
      = 1.1 x 1.01325 / 1.00975 = 1.103813 (to six places),

+38.13 pips at 0.0001 a pip, and a USD-based holder of EUR assets selling
EUR forward picks up (F / S - 1) / 0.25 = 1.3865% a year -- the rate
differential discounted, 0.014 / 1.00975.

Beyond it: equal growth on both legs gives F = S bit for bit; a basis of
zero is parity exactly and a basis b is parity at the foreign rate plus b;
quoting the pair the other way round changes no premium; the hedged return
is the cash flows of the asset plus the forward, an independent oracle; the
legs take their own day counts, and the date-based ones need a date. Every
refusal names its argument.
"""

from __future__ import annotations

import datetime as dt
import math
from typing import Any, Dict

import pydantic
import pytest

from standard_quant_tools.agent.runtimes import resolve
from standard_quant_tools.agent.runtimes.finance.currency_models import (
    FxForwardHedgeInput,
)
from standard_quant_tools.delta_one.daycount import year_fraction
from standard_quant_tools.error import ValidationError
from standard_quant_tools.finance import (
    covered_interest_forward,
    currency_pair,
    fx_forward_hedge,
    leg_year_fraction,
    pip_size_for,
)
from standard_quant_tools.mcp.catalog import _reads_market_data
from standard_quant_tools.mcp.schemas import dereference

from ..surface import synth

PLANTED: Dict[str, Any] = {
    "pair": "EURUSD",
    "spot": 1.1,
    "days": 90,
    "base_currency": "USD",
    "base_rate": 0.053,
    "foreign_rate": 0.039,
    "base_day_count": "ACT/360",
    "foreign_day_count": "ACT/360",
}


def _priced(**changes: Any) -> Dict[str, Any]:
    return fx_forward_hedge(**{**PLANTED, **changes})


class TestThePlantedQuote:
    def test_the_forward_points_and_pickup_are_the_worked_ones(self):
        """1.103813, +38.13 pips and 1.3865% a year, as worked above."""
        result = _priced()
        assert round(result["forward"], 6) == 1.103813
        assert round(result["forward_points"], 2) == 38.13
        assert round(result["hedge_carry"] * 100, 4) == 1.3865
        assert result["year_fractions"] == {"USD": 0.25, "EUR": 0.25}
        assert result["pip_size"] == 0.0001
        assert result["foreign_currency"] == "EUR"

    def test_the_pickup_is_the_rate_differential_discounted(self):
        """Under parity with one day count, (F/S - 1)/t is (r_base -
        r_foreign) / (1 + r_foreign t): the differential, discounted one
        period at the foreign rate."""
        result = _priced()
        assert result["rate_differential"] == pytest.approx(0.014, abs=1e-17)
        assert result["hedge_carry"] == pytest.approx(
            0.014 / (1 + 0.039 * 0.25), rel=1e-14
        )

    def test_the_forward_is_the_parity_expression_bit_for_bit(self):
        """A basis of zero is covered interest parity exactly: the same
        expression, nothing added to the foreign rate."""
        expected = 1.1 * ((1 + 0.053 * 0.25) / (1 + 0.039 * 0.25))
        assert _priced()["forward"] == expected
        assert _priced(basis_bps=0.0)["forward"] == expected
        assert (
            covered_interest_forward(
                1.1,
                first_rate=0.039,
                first_years=0.25,
                second_rate=0.053,
                second_years=0.25,
            )
            == expected
        )


class TestParity:
    @pytest.mark.parametrize("rate", [0.0, 0.0125, 0.053, -0.005, 0.4])
    @pytest.mark.parametrize("days", [1, 30, 90, 365, 730])
    def test_zero_rate_differential_gives_a_forward_equal_to_spot(self, rate, days):
        """Equal rates on equal day counts grow both legs alike, and the ratio
        is formed before it multiplies the spot, so F is S exactly."""
        result = _priced(base_rate=rate, foreign_rate=rate, days=days)
        assert result["forward"] == 1.1
        assert result["forward_points"] == 0.0
        assert result["hedge_carry"] == 0.0

    def test_equal_rates_on_different_day_counts_are_not_parity_at_spot(self):
        """GBP accrues ACT/365F and USD ACT/360, so 4% on both for 182 days
        grows the GBP leg by 1 + 0.04 x 182/365 and the USD leg by
        1 + 0.04 x 182/360, and GBPUSD's forward sits above spot."""
        result = _priced(
            pair="GBPUSD",
            spot=1.27,
            days=182,
            base_rate=0.04,
            foreign_rate=0.04,
            base_day_count="ACT/360",
            foreign_day_count="ACT/365F",
        )
        expected = 1.27 * ((1 + 0.04 * 182 / 360) / (1 + 0.04 * 182 / 365))
        assert result["forward"] == pytest.approx(expected, rel=1e-15)
        assert result["forward"] > 1.27
        assert result["year_fractions"] == {"USD": 182 / 360, "GBP": 182 / 365}
        assert any("count days differently" in w for w in result["warnings"])

    @pytest.mark.parametrize("basis", [-35.0, -12.5, 8.0])
    def test_a_basis_is_parity_at_the_foreign_rate_plus_the_basis(self, basis):
        """The basis is added to the foreign leg: the forward with a basis of
        b bps is parity with the foreign rate raised by b / 10,000."""
        with_basis = _priced(basis_bps=basis)
        shifted = _priced(foreign_rate=0.039 + basis / 10_000.0)
        assert with_basis["forward"] == shifted["forward"]
        assert with_basis["rates"]["EUR"] == 0.039 + basis / 10_000.0
        assert with_basis["rates"]["USD"] == 0.053

    def test_a_negative_basis_raises_a_base_hedgers_pickup_by_about_itself(self):
        """-25 bps on the EUR leg prices the forward as if EUR paid 3.65%, so
        a USD hedger selling EUR forward picks up (0.053 - 0.0365) / (1 +
        0.0365 x 0.25) = 1.6351% a year, 24.86 bps more than at parity --
        the basis, less its own discounting."""
        plain = _priced()["hedge_carry"]
        widened = _priced(basis_bps=-25.0)["hedge_carry"]
        assert widened == pytest.approx(
            (0.053 - 0.0365) / (1 + 0.0365 * 0.25), rel=1e-14
        )
        assert round((widened - plain) * 10_000, 2) == 24.86

    def test_the_hedge_premium_does_not_depend_on_how_the_pair_is_quoted(self):
        """EURUSD at 1.1 and USDEUR at 1/1.1 are the same market: the
        forwards are reciprocal, and the premium and pickup of the hedge are
        the same numbers whichever way round the pair is written."""
        direct = _priced()
        inverse = _priced(pair="USDEUR", spot=1 / 1.1)
        assert inverse["forward"] == pytest.approx(1 / direct["forward"], rel=1e-15)
        assert inverse["forward_premium"] == pytest.approx(
            direct["forward_premium"], rel=1e-13
        )
        assert inverse["hedge_carry"] == pytest.approx(direct["hedge_carry"], rel=1e-13)

    def test_a_jpy_quote_is_counted_in_hundredths_and_the_foreign_leg_is_second(self):
        """USDJPY 150.00 for a USD hedger of JPY assets: JPY is the quote, so
        a pip is 0.01, F = S (1 + r_JPY t) / (1 + r_USD t), and the hedger's
        premium is S / F - 1 (units of USD per JPY rise as USDJPY falls)."""
        result = _priced(
            pair="USDJPY",
            spot=150.0,
            base_rate=0.053,
            foreign_rate=0.001,
        )
        forward = 150.0 * ((1 + 0.001 * 0.25) / (1 + 0.053 * 0.25))
        assert result["forward"] == forward
        assert result["pip_size"] == 0.01
        assert "JPY" in result["pip_size_rule"]
        assert result["forward_points"] == pytest.approx((forward - 150.0) / 0.01)
        assert result["forward_premium"] == pytest.approx(150.0 / forward - 1)
        assert result["hedge_carry"] > 0  # USD pays more than JPY: a pickup
        assert pip_size_for("JPY") == 0.01 and pip_size_for("USD") == 0.0001

    def test_a_given_pip_size_is_used_and_said_to_be(self):
        result = _priced(pip_size=0.00001)
        assert result["pip_size"] == 0.00001
        assert result["pip_size_rule"] == "given"
        assert round(result["forward_points"], 1) == 381.3


class TestTheHedgedReturn:
    @pytest.mark.parametrize("hedge_ratio", [0.0, 0.5, 1.0, 1.25])
    @pytest.mark.parametrize("asset_return,end_spot", [(0.02, 1.08), (-0.1, 1.15)])
    def test_it_is_the_cash_flows_of_the_asset_and_the_forward(
        self, hedge_ratio, asset_return, end_spot
    ):
        """An independent oracle: one EUR of an asset bought at 1.1 USD ends
        worth (1 + r_L) EUR at the delivery spot, and the forward selling h
        EUR at F pays h (F - S_1) USD; the hedged return is their sum over
        the 1.1 USD paid. h = 0 is the unhedged return."""
        result = _priced(
            asset_return=asset_return,
            spot_at_maturity=end_spot,
            hedge_ratio=hedge_ratio,
        )
        forward = result["forward"]
        oracle = (
            (1 + asset_return) * end_spot + hedge_ratio * (forward - end_spot)
        ) / 1.1 - 1
        unhedged = (1 + asset_return) * end_spot / 1.1 - 1
        period = result["period"]
        assert period["hedged_return"] == pytest.approx(oracle, abs=1e-15)
        assert period["unhedged_return"] == pytest.approx(unhedged, abs=1e-15)
        assert period["hedged_return"] - period["unhedged_return"] == pytest.approx(
            period["hedge_return"], abs=1e-15
        )
        if hedge_ratio == 0.0:
            assert period["hedged_return"] == period["unhedged_return"]

    @pytest.mark.parametrize("hedge_ratio", [0.0, 1.0])
    def test_it_is_the_cash_flows_when_the_foreign_currency_is_quoted_second(
        self, hedge_ratio
    ):
        """USDJPY for a USD holder of a JPY asset: the cash flows are in USD
        per JPY, 1 / spot. JPY rallying from 150 to 140 is a +7.14% FX
        return to the holder, and the forward selling JPY at 1 / F gives it
        back."""
        result = _priced(
            pair="USDJPY",
            spot=150.0,
            foreign_rate=0.001,
            asset_return=0.03,
            spot_at_maturity=140.0,
            hedge_ratio=hedge_ratio,
        )
        x0, x1, xf = 1 / 150.0, 1 / 140.0, 1 / result["forward"]
        oracle = ((1 + 0.03) * x1 + hedge_ratio * (xf - x1)) / x0 - 1
        period = result["period"]
        assert period["fx_return"] == pytest.approx(150 / 140 - 1, rel=1e-15)
        assert period["hedged_return"] == pytest.approx(oracle, abs=1e-15)

    def test_a_full_hedge_leaves_the_cross_term_unhedged(self):
        """With h = 1 the currency's move is gone except through the asset's
        own gain: hedged = r_L + f + r_L x r_FX, the cross term left because
        the hedge was sized on the starting value."""
        result = _priced(asset_return=0.05, spot_at_maturity=1.0)
        period = result["period"]
        r_fx = 1.0 / 1.1 - 1
        assert period["fx_return"] == pytest.approx(r_fx, abs=1e-16)
        assert period["cross_return"] == pytest.approx(0.05 * r_fx, abs=1e-16)
        assert period["hedged_return"] == pytest.approx(
            0.05 + result["forward_premium"] + 0.05 * r_fx, abs=1e-15
        )

    def test_a_period_needs_both_the_asset_return_and_the_end_spot(self):
        with pytest.raises(ValidationError, match="spot_at_maturity"):
            _priced(asset_return=0.02)
        with pytest.raises(ValidationError, match="asset_return"):
            _priced(spot_at_maturity=1.08)
        assert _priced()["period"] is None


class TestDayCounts:
    def test_each_leg_takes_its_own_convention(self):
        """USD ACT/360 against GBP ACT/365F over 90 days: 0.25 and 90/365."""
        result = _priced(
            pair="GBPUSD", spot=1.27, foreign_day_count="ACT/365F", foreign_rate=0.0475
        )
        assert result["year_fractions"]["USD"] == 0.25
        assert result["year_fractions"]["GBP"] == 90 / 365
        assert result["day_counts"] == {"USD": "ACT/360", "GBP": "ACT/365F"}
        assert result["forward"] == 1.27 * (
            (1 + 0.053 * 0.25) / (1 + 0.0475 * (90 / 365))
        )

    @pytest.mark.parametrize("convention", ["ACT/ACT", "30/360"])
    def test_a_date_based_convention_needs_the_spot_date(self, convention):
        with pytest.raises(ValidationError, match="start_date"):
            _priced(foreign_day_count=convention)

    def test_with_a_spot_date_the_fraction_is_the_library_day_count(self):
        """ACT/ACT across a year end splits at it: 2027-12-01 plus 90 days
        is 31/365 of 2027 and 59/366 of 2028, as daycount counts it."""
        result = _priced(foreign_day_count="ACT/ACT", start_date="2027-12-01")
        end = dt.date(2027, 12, 1) + dt.timedelta(days=90)
        expected = year_fraction("2027-12-01", end, convention="ACT/ACT")
        assert result["year_fractions"]["EUR"] == expected
        assert expected == pytest.approx(31 / 365 + 59 / 366, rel=1e-15)
        assert result["maturity_date"] == end.isoformat()
        assert leg_year_fraction(90, "ACT/360", "2027-12-01") == 0.25

    def test_thirty_360_from_a_thirtieth_to_a_thirty_first_has_no_annual_rate(self):
        """30/360 counts no days from 30 January to 31 January, so the base
        leg's year fraction is zero and the pickup has no annual rate: null,
        with its reason, rather than a division by zero."""
        result = resolve("finance").dispatch(
            "price_fx_forward_hedge",
            {
                **PLANTED,
                "days": 1,
                "base_day_count": "30/360",
                "start_date": "2026-01-30",
            },
        )
        assert result["hedge_carry"] is None
        assert any("hedge_carry is null" in w for w in result["warnings"])

    def test_a_forward_over_a_year_says_how_it_was_priced(self):
        result = _priced(days=730)
        assert any("more than a year" in w for w in result["warnings"])


class TestRefusals:
    @pytest.mark.parametrize(
        "pair,match",
        [
            ("EURO", "not a currency pair"),
            ("EUREUR", "twice"),
            ("GBpUSD", "pence"),
            ("EURXYZ", "not an active ISO 4217"),
            ("DEMUSD", "withdrawn"),
        ],
    )
    def test_a_pair_that_is_not_two_currencies_is_refused(self, pair, match):
        with pytest.raises(ValidationError, match=match):
            currency_pair(pair)

    def test_the_pair_may_be_written_with_a_slash(self):
        assert currency_pair("eur/usd") == ("EUR", "USD")

    def test_a_base_currency_outside_the_pair_is_refused_with_the_pair_to_price(self):
        with pytest.raises(ValidationError, match="price EURGBP"):
            _priced(base_currency="GBP")

    @pytest.mark.parametrize(
        "field,value",
        [
            ("spot", 0.0),
            ("spot", math.inf),
            ("base_rate", 53.0),
            ("foreign_rate", math.nan),
            ("basis_bps", 20_000.0),
            ("hedge_ratio", -0.5),
            ("days", 0),
            ("pip_size", 2.0),
        ],
    )
    def test_a_value_outside_its_range_is_refused_by_name(self, field, value):
        with pytest.raises(ValidationError, match=field):
            _priced(**{field: value})

    def test_a_leg_that_cannot_grow_has_no_forward(self):
        """-9 a year (in range) over two years makes the foreign leg's growth
        1 - 18 < 0: no forward exists, and it is refused, not priced."""
        with pytest.raises(ValidationError, match="not above zero"):
            _priced(foreign_rate=-9.0, days=730)


class TestTheTool:
    def test_it_dispatches_through_the_finance_runtime(self):
        result = resolve("finance").dispatch("price_fx_forward_hedge", PLANTED)
        assert round(result["forward"], 6) == 1.103813
        assert round(result["forward_points"], 2) == 38.13
        assert result["period"] is None

    def test_the_input_refuses_a_base_currency_outside_the_pair_by_location(self):
        with pytest.raises(pydantic.ValidationError) as caught:
            FxForwardHedgeInput(**{**PLANTED, "base_currency": "JPY"})
        assert "neither currency of EURUSD" in str(caught.value)

    def test_a_spot_date_that_is_not_iso_is_refused_as_every_date_is(self):
        with pytest.raises(pydantic.ValidationError, match="not an ISO date"):
            FxForwardHedgeInput(**{**PLANTED, "start_date": "01/12/2027"})

    def test_the_input_normalises_the_pair(self):
        model = FxForwardHedgeInput(**{**PLANTED, "pair": "eur/usd"})
        assert model.pair == "EURUSD"

    def test_it_takes_no_ticker_and_is_synthesizable(self):
        """No field reads as a ticker, so the tool is not open-world and stays
        inside the determinism layer; the fuzzer can build its input."""
        schema = dereference(FxForwardHedgeInput.model_json_schema())
        assert not _reads_market_data(schema)
        arguments, reason = synth.synthesize(FxForwardHedgeInput)
        assert arguments is not None, reason
