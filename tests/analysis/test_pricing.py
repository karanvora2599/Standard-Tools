"""
Option pricing: four models behind one field, checked against arithmetic.

WHY THESE TESTS DO NOT COMPARE AGAINST RECORDED NUMBERS. A recorded price
pins whatever the implementation did on the day it was written, including
its mistakes. Every check here is against something that must hold
independently of the implementation:

- **Put-call parity.** C - P = S - K·e^(-rT), exactly, for any European
  model. A sign error anywhere in the call or put branch breaks it.
- **Convergence.** The binomial lattice must approach Black-Scholes on a
  European option as steps rise. A lattice that does not converge to the
  closed form on the case where both apply is wired wrong, and the error
  must fall like 1/steps.
- **Monotonicity.** A call is worth more when the underlying is higher, when
  volatility is higher, and when there is more time. These are properties of
  an option, not of a formula.
- **Early exercise is a premium, never a discount.** American ≥ European,
  always, because the American holder can always choose not to exercise.
- **Bounds.** A call is worth at least its discounted intrinsic value and
  never more than the underlying. Any price outside that is arbitrage.

THE ONE COMPARISON AGAINST ANOTHER IMPLEMENTATION is against this library's
own `black_scholes_price`, and its purpose is the opposite: to prove the new
path did NOT move an existing number.
"""

import math

import pytest

from standard_quant_tools.analysis.options import black_scholes_price
from standard_quant_tools.analysis.pricing import MODELS, price_option
from standard_quant_tools.error import ValidationError

BASE = dict(
    spot=100.0,
    strike=100.0,
    time_to_expiry=1.0,
    volatility=0.2,
    risk_free_rate=0.05,
)


def _price(**overrides):
    return price_option(**{**BASE, **overrides})["price"]


class TestItDidNotMoveAnExistingNumber:
    @pytest.mark.parametrize("option_type", ["call", "put"])
    @pytest.mark.parametrize("spot", [80.0, 100.0, 125.0])
    def test_black_scholes_matches_the_librarys_own(self, option_type, spot):
        mine = _price(spot=spot, option_type=option_type)
        theirs = black_scholes_price(
            spot, 100.0, 1.0, 0.05, 0.2, option_type=option_type
        )
        theirs = theirs["price"] if isinstance(theirs, dict) else theirs
        assert mine == pytest.approx(theirs, abs=1e-10)


class TestPutCallParity:
    """C - P = S - K e^(-rT). Exact, for every European model, and the
    single check most likely to catch a sign error."""

    @pytest.mark.parametrize("spot", [70.0, 100.0, 140.0])
    @pytest.mark.parametrize("t", [0.1, 1.0, 3.0])
    def test_black_scholes(self, spot, t):
        call = _price(spot=spot, time_to_expiry=t, option_type="call")
        put = _price(spot=spot, time_to_expiry=t, option_type="put")
        assert call - put == pytest.approx(spot - 100.0 * math.exp(-0.05 * t), abs=1e-9)

    @pytest.mark.parametrize("spot", [70.0, 100.0, 140.0])
    def test_black_76_parity_is_discounted(self, spot):
        """On a forward, parity discounts BOTH legs: C - P = e^(-rT)(F - K).
        Getting this wrong is how a forward gets carried twice."""
        call = _price(spot=spot, model="black_76", option_type="call")
        put = _price(spot=spot, model="black_76", option_type="put")
        assert call - put == pytest.approx(math.exp(-0.05) * (spot - 100.0), abs=1e-9)

    @pytest.mark.parametrize("spot", [-20.0, 0.0, 50.0])
    def test_bachelier_parity_holds_through_zero(self, spot):
        """The model exists for this range, so parity has to hold in it."""
        call = price_option(
            spot=spot,
            strike=10.0,
            time_to_expiry=0.5,
            volatility=15.0,
            risk_free_rate=0.02,
            option_type="call",
            model="bachelier",
        )["price"]
        put = price_option(
            spot=spot,
            strike=10.0,
            time_to_expiry=0.5,
            volatility=15.0,
            risk_free_rate=0.02,
            option_type="put",
            model="bachelier",
        )["price"]
        assert call - put == pytest.approx(
            math.exp(-0.02 * 0.5) * (spot - 10.0), abs=1e-9
        )

    @pytest.mark.parametrize("option_type", ["call", "put"])
    @pytest.mark.parametrize("t", [0.25, 1.5, 4.0])
    def test_black_76_on_the_forward_equals_black_scholes_on_the_spot(
        self, option_type, t
    ):
        """
        THE identity that defines the relationship, and the one test that
        catches the double-count.

        Black-76 on F = S*e^((r-q)T) must equal Black-Scholes on S, exactly.
        That is what "the forward already contains the carry" means.

        Written after a mutation survived: making black_76 carry the forward
        a second time passed all 65 other tests, because put-call parity on a
        forward is C - P = e^(-rT)(F - K) whatever the drift is, so parity
        cannot see it. The error also GROWS with time to expiry, which is why
        the horizons below run out to four years -- a near-dated test would
        miss it too.
        """
        spot, strike, vol, rate, dividend = 100.0, 105.0, 0.25, 0.05, 0.02
        forward = spot * math.exp((rate - dividend) * t)
        spot_price = price_option(
            spot=spot,
            strike=strike,
            time_to_expiry=t,
            volatility=vol,
            risk_free_rate=rate,
            dividend_yield=dividend,
            option_type=option_type,
        )["price"]
        forward_price = price_option(
            spot=forward,
            strike=strike,
            time_to_expiry=t,
            volatility=vol,
            risk_free_rate=rate,
            option_type=option_type,
            model="black_76",
        )["price"]
        assert forward_price == pytest.approx(spot_price, abs=1e-10), (
            "black_76 on the forward diverged from black_scholes on the spot, "
            "which means the carry is being applied twice or not at all"
        )

    def test_a_dividend_yield_enters_parity(self):
        call = _price(option_type="call", dividend_yield=0.03)
        put = _price(option_type="put", dividend_yield=0.03)
        assert call - put == pytest.approx(
            100.0 * math.exp(-0.03) - 100.0 * math.exp(-0.05), abs=1e-9
        )


class TestTheLatticeConverges:
    def test_it_approaches_black_scholes(self):
        """A lattice that does not converge to the closed form on the case
        where both apply is wired wrong."""
        closed = _price(option_type="call")
        errors = [
            abs(_price(option_type="call", model="binomial", steps=n) - closed)
            for n in (50, 200, 800)
        ]
        assert errors == sorted(errors, reverse=True), errors
        assert errors[-1] < 0.01

    def test_the_error_falls_about_like_one_over_steps(self):
        """CRR is first-order in the step count. Quadrupling the steps should
        quarter the error, roughly -- a convergence that is much slower says
        the probability or the up/down factors are off."""
        closed = _price(option_type="call")
        coarse = abs(_price(option_type="call", model="binomial", steps=50) - closed)
        fine = abs(_price(option_type="call", model="binomial", steps=200) - closed)
        assert 2.0 < coarse / fine < 8.0, coarse / fine

    def test_a_lattice_too_coarse_for_the_rate_is_refused(self):
        """No arbitrage-free probability exists when the growth per step
        escapes the up/down range. Better to say so than to return a price
        computed with a probability outside [0, 1]."""
        with pytest.raises(ValidationError, match="arbitrage-free"):
            price_option(
                spot=100.0,
                strike=100.0,
                time_to_expiry=1.0,
                volatility=0.01,
                risk_free_rate=0.5,
                model="binomial",
                steps=10,
            )


class TestEarlyExerciseIsAPremium:
    @pytest.mark.parametrize("spot", [70.0, 90.0, 110.0])
    def test_american_put_is_never_worth_less_than_european(self, spot):
        """The American holder can always decline to exercise, so the
        premium is bounded below by zero for every input."""
        european = price_option(
            spot=spot,
            strike=100.0,
            time_to_expiry=1.0,
            volatility=0.3,
            risk_free_rate=0.08,
            option_type="put",
            model="binomial",
            steps=300,
        )["price"]
        american = price_option(
            spot=spot,
            strike=100.0,
            time_to_expiry=1.0,
            volatility=0.3,
            risk_free_rate=0.08,
            option_type="put",
            model="binomial",
            american=True,
            steps=300,
        )["price"]
        assert american >= european - 1e-9

    def test_a_deep_in_the_money_put_shows_a_real_premium(self):
        """Not merely equal: the case where early exercise is worth
        something must actually be worth something, or the lattice is
        computing the European value and calling it American."""
        kwargs = dict(
            spot=60.0,
            strike=100.0,
            time_to_expiry=1.0,
            volatility=0.3,
            risk_free_rate=0.10,
            option_type="put",
            model="binomial",
            steps=300,
        )
        european = price_option(**kwargs)["price"]
        american = price_option(**kwargs, american=True)["price"]
        assert american > european + 0.5

    def test_american_on_a_european_model_is_refused_by_name(self):
        with pytest.raises(ValidationError) as exc:
            _price(model="black_scholes", american=True)
        message = str(exc.value)
        assert "binomial" in message
        assert "early-exercise premium" in message


class TestPropertiesOfAnOptionRatherThanOfAFormula:
    @pytest.mark.parametrize("model", ["black_scholes", "black_76", "binomial"])
    def test_a_call_rises_with_the_underlying(self, model):
        prices = [
            _price(spot=s, model=model, option_type="call") for s in (80, 100, 120)
        ]
        assert prices == sorted(prices)

    @pytest.mark.parametrize("model", ["black_scholes", "black_76", "binomial"])
    def test_a_put_falls_with_the_underlying(self, model):
        prices = [
            _price(spot=s, model=model, option_type="put") for s in (80, 100, 120)
        ]
        assert prices == sorted(prices, reverse=True)

    @pytest.mark.parametrize("option_type", ["call", "put"])
    def test_more_volatility_is_worth_more(self, option_type):
        prices = [
            _price(volatility=v, option_type=option_type) for v in (0.1, 0.2, 0.4)
        ]
        assert prices == sorted(prices)

    @pytest.mark.parametrize("option_type", ["call", "put"])
    def test_more_time_is_worth_more_on_a_forward(self, option_type):
        """Stated on black_76 rather than black_scholes: a European PUT on a
        spot underlying can lose value with time, because the discounting of
        the strike eventually outruns the added optionality. On a forward
        both legs discount together and the monotonicity is clean."""
        prices = [
            _price(time_to_expiry=t, model="black_76", option_type=option_type)
            for t in (0.25, 1.0, 3.0)
        ]
        assert prices == sorted(prices)

    def test_a_call_is_worth_at_least_its_discounted_intrinsic(self):
        """Below this is an arbitrage, not a model choice."""
        for spot in (80.0, 100.0, 130.0):
            price = _price(spot=spot, option_type="call")
            floor = max(0.0, spot - 100.0 * math.exp(-0.05))
            assert price >= floor - 1e-9

    def test_a_call_is_never_worth_more_than_the_underlying(self):
        for spot in (80.0, 100.0, 130.0):
            assert _price(spot=spot, option_type="call") <= spot + 1e-9

    def test_delta_is_bounded(self):
        for spot in (50.0, 100.0, 200.0):
            call = price_option(**{**BASE, "spot": spot}, option_type="call")
            put = price_option(**{**BASE, "spot": spot}, option_type="put")
            assert 0.0 <= call["delta"] <= 1.0
            assert -1.0 <= put["delta"] <= 0.0

    def test_gamma_is_positive_for_both(self):
        for option_type in ("call", "put"):
            assert price_option(**BASE, option_type=option_type)["gamma"] > 0


class TestBachelierIsWhyTheModelFieldExists:
    def test_it_prices_a_negative_underlying(self):
        """Not hypothetical. WTI settled at -$37.63 on 20 April 2020."""
        result = price_option(
            spot=-5.0,
            strike=0.0,
            time_to_expiry=0.25,
            volatility=20.0,
            risk_free_rate=0.01,
            option_type="call",
            model="bachelier",
        )
        assert result["price"] > 0

    def test_it_prices_a_negative_strike(self):
        result = price_option(
            spot=-5.0,
            strike=-10.0,
            time_to_expiry=0.25,
            volatility=20.0,
            risk_free_rate=0.01,
            option_type="call",
            model="bachelier",
        )
        assert result["price"] > 0

    @pytest.mark.parametrize("model", ["black_scholes", "black_76", "binomial"])
    def test_the_lognormal_models_refuse_and_name_the_alternative(self, model):
        with pytest.raises(ValidationError) as exc:
            price_option(
                spot=-5.0,
                strike=10.0,
                time_to_expiry=0.25,
                volatility=0.3,
                risk_free_rate=0.01,
                model=model,
            )
        assert "bachelier" in str(exc.value)

    def test_a_negative_strike_is_refused_by_the_lognormal_models(self):
        with pytest.raises(ValidationError, match="bachelier"):
            _price(strike=-10.0)

    def test_at_the_money_it_has_the_closed_form_value(self):
        """At S=K the Bachelier call is exactly sigma*sqrt(T/2pi), discounted.
        A closed form is a better check than a recorded number."""
        vol, t, rate = 15.0, 0.5, 0.02
        price = price_option(
            spot=50.0,
            strike=50.0,
            time_to_expiry=t,
            volatility=vol,
            risk_free_rate=rate,
            option_type="call",
            model="bachelier",
        )["price"]
        expected = math.exp(-rate * t) * vol * math.sqrt(t / (2 * math.pi))
        assert price == pytest.approx(expected, rel=1e-9)


class TestItRefusesNonsense:
    @pytest.mark.parametrize(
        "bad",
        [
            {"time_to_expiry": 0.0},
            {"time_to_expiry": -1.0},
            {"volatility": 0.0},
            {"volatility": -0.2},
            {"option_type": "straddle"},
            {"model": "heston"},
        ],
    )
    def test_bad_inputs_raise(self, bad):
        with pytest.raises(ValidationError):
            _price(**bad)

    def test_an_unknown_model_lists_the_known_ones(self):
        with pytest.raises(ValidationError) as exc:
            _price(model="sabr")
        for known in MODELS:
            assert known in str(exc.value)

    def test_an_expired_option_says_it_needs_no_model(self):
        with pytest.raises(ValidationError, match="does not need a model"):
            _price(time_to_expiry=0.0)


class TestThroughTheTool:
    """The plan's rule: the model families are a FIELD, not seven tools."""

    def test_no_new_tool_was_added(self):
        from standard_quant_tools.agent.runtimes import resolve

        # The tool MOVED to `derivatives` when that runtime reached
        # twelve tools. What this test pins is unchanged: every
        # pricing model is reachable through ONE tool, and no second
        # pricing tool was added alongside it.
        derivatives = resolve("derivatives")
        assert "price_option" not in derivatives.dispatch_table
        assert "get_option_pricing" in derivatives.dispatch_table
        assert "get_option_pricing" not in resolve("research").dispatch_table

    @pytest.mark.parametrize("model", ["black_scholes", "black_76", "binomial"])
    def test_every_model_is_reachable_through_the_one_tool(self, model):
        from standard_quant_tools.agent.runtimes import resolve

        result = resolve("derivatives").dispatch(
            "get_option_pricing", {**BASE, "model": model, "option_type": "call"}
        )
        assert result["price"] > 0
        assert result["model"] == model

    def test_the_lattice_reports_no_theta_rather_than_a_fabricated_one(self):
        from standard_quant_tools.agent.runtimes import resolve

        result = resolve("derivatives").dispatch(
            "get_option_pricing", {**BASE, "model": "binomial"}
        )
        assert result["greeks"]["theta"] is None
        assert result["notes"], "an absent greek should say why"

    def test_the_default_path_is_untouched(self):
        """No existing call may move by a cent."""
        from standard_quant_tools.agent.runtimes import resolve

        result = resolve("derivatives").dispatch("get_option_pricing", dict(BASE))
        assert result["price"] == pytest.approx(10.450584, abs=1e-6)
        assert result["greeks"]["theta"] is not None
