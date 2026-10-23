"""
Dollar greeks across a book of options and underlying positions on several
underlyings, and delta-gamma P&L against full revaluation.

Each check is against something the book cannot have computed itself:

- one option's dollar greeks are the Black-Scholes-Merton closed forms,
  written out here with math.erf, times the multiplier and the quantity --
  for a call and a put, a dividend payer, and a line given by expiry date;
- a conversion -- long a call, short the put at the same strike and
  expiry, short the underlying a contract controls -- has a book delta of
  zero and a gamma and vega of exactly zero by put-call parity, is worth
  -K e^{-rT} per unit, and its full revaluation is flat in spot;
- the delta-gamma error shrinks like the cube of the shock, at the rate
  speed x dS^3 / 6 predicts;
- the beta-weighted delta is sum_i beta_i x dollar delta_i, and the book is
  the sum of its underlyings, which are the sums of their lines.

The multiplier is required: a line without one is refused rather than
priced at 100.
"""

from __future__ import annotations

import math

import pydantic
import pytest

from standard_quant_tools.agent.runtimes import resolve
from standard_quant_tools.analysis.option_book import aggregate_option_book
from standard_quant_tools.analysis.options_batch import black_scholes_greeks_batch
from standard_quant_tools.error import ValidationError


def _N(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _closed_form(S, K, T, sigma, r, q, call):
    """Black-Scholes-Merton price and greeks, per unit: vega per vol point,
    theta per calendar day, rho per 1% of rate."""
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    gq, gr = math.exp(-q * T), math.exp(-r * T)
    if call:
        price = S * gq * _N(d1) - K * gr * _N(d2)
        delta = gq * _N(d1)
        theta = (
            -S * _pdf(d1) * sigma * gq / (2 * math.sqrt(T))
            + q * S * gq * _N(d1)
            - r * K * gr * _N(d2)
        )
        rho = K * T * gr * _N(d2)
    else:
        price = K * gr * _N(-d2) - S * gq * _N(-d1)
        delta = -gq * _N(-d1)
        theta = (
            -S * _pdf(d1) * sigma * gq / (2 * math.sqrt(T))
            - q * S * gq * _N(-d1)
            + r * K * gr * _N(-d2)
        )
        rho = -K * T * gr * _N(-d2)
    gamma = gq * _pdf(d1) / (S * sigma * math.sqrt(T))
    vega = S * gq * _pdf(d1) * math.sqrt(T)
    return {
        "price": price,
        "delta": delta,
        "gamma": gamma,
        "vega": vega / 100.0,
        "theta": theta / 365.0,
        "rho": rho / 100.0,
    }


@pytest.fixture(scope="module")
def derivatives():
    return resolve("derivatives")


def _line(**overrides):
    line = {
        "underlying": "XYZ",
        "option_type": "call",
        "strike": 105.0,
        "time_to_expiry": 0.5,
        "volatility": 0.25,
        "quantity": 7,
        "multiplier": 100,
    }
    line.update(overrides)
    return line


class TestOneLineIsTheClosedForm:
    @pytest.mark.parametrize("kind", ["call", "put"])
    def test_its_dollar_greeks_are_the_formulas_times_multiplier_and_quantity(
        self, derivatives, kind
    ):
        """A dividend payer, 7 contracts of 100 (and -7 for the put): every
        dollar greek is the closed form per unit x 100 x quantity, the
        dollar gamma x S^2 / 100, to 1e-12 relative."""
        S, K, T, sigma, r, q = 100.0, 105.0, 0.5, 0.25, 0.045, 0.02
        quantity = 7 if kind == "call" else -7
        out = derivatives.dispatch(
            "aggregate_option_book",
            {
                "options": [_line(option_type=kind, quantity=quantity)],
                "spots": {"XYZ": S},
                "rates": {"XYZ": r},
                "dividend_yields": {"XYZ": q},
            },
        )
        cf = _closed_form(S, K, T, sigma, r, q, kind == "call")
        n = 100 * quantity
        line = out["lines"][0]
        for greek in ("price", "delta", "gamma", "vega", "theta", "rho"):
            assert line[greek] == pytest.approx(cf[greek], rel=1e-12), greek
        totals = out["totals"]
        assert totals["dollar_delta"] == pytest.approx(cf["delta"] * S * n, rel=1e-12)
        assert totals["dollar_gamma"] == pytest.approx(
            cf["gamma"] * S * S / 100 * n, rel=1e-12
        )
        assert totals["dollar_vega"] == pytest.approx(cf["vega"] * n, rel=1e-12)
        assert totals["dollar_theta"] == pytest.approx(cf["theta"] * n, rel=1e-12)
        assert totals["dollar_rho"] == pytest.approx(cf["rho"] * n, rel=1e-12)
        assert totals["market_value"] == pytest.approx(cf["price"] * n, rel=1e-12)
        assert out["by_underlying"][0]["delta_units"] == pytest.approx(
            cf["delta"] * n, rel=1e-12
        )
        assert out["theta_basis"].startswith("per calendar day")

    def test_an_expiry_date_is_its_actual_days_over_365(self, derivatives):
        """182 calendar days from the valuation date is 182 / 365 years,
        the basis the per-day theta uses."""
        out = derivatives.dispatch(
            "aggregate_option_book",
            {
                "options": [
                    _line(time_to_expiry=None, expiry="2027-04-23"),
                ],
                "spots": {"XYZ": 100.0},
                "rates": {"XYZ": 0.04},
                "valuation_date": "2026-10-23",
            },
        )
        assert out["lines"][0]["time_to_expiry"] == 182 / 365
        cf = _closed_form(100.0, 105.0, 182 / 365, 0.25, 0.04, 0.0, True)
        assert out["lines"][0]["delta"] == pytest.approx(cf["delta"], rel=1e-12)

    def test_the_multiplier_scales_every_dollar_greek(self):
        """A mini contract of 10 is a tenth of the standard 100 on every
        dollar greek: the multiplier is a fact of the contract."""
        common = dict(spots={"XYZ": 100.0}, rates={"XYZ": 0.03})
        big = aggregate_option_book([_line()], **common)["totals"]
        mini = aggregate_option_book([_line(multiplier=10)], **common)["totals"]
        for name, value in big.items():
            if name != "beta_weighted_delta":
                assert mini[name] == pytest.approx(value / 10, rel=1e-14), name


class TestAConversionIsFlat:
    def _conversion(self, shocks=None):
        S, K, T, r = 100.0, 100.0, 0.75, 0.05
        lines = [
            _line(option_type="call", strike=K, time_to_expiry=T, quantity=3),
            _line(option_type="put", strike=K, time_to_expiry=T, quantity=-3),
        ]
        return (
            aggregate_option_book(
                lines,
                {"XYZ": S},
                {"XYZ": r},
                underlying_units={"XYZ": -300},
                spot_shocks=shocks,
            ),
            K,
            T,
            r,
        )

    def test_delta_is_zero_and_gamma_and_vega_exactly_zero(self):
        """Long a call, short the put, short the 300 shares the three
        contracts control: by put-call parity the options are a forward on
        300 shares, so the book's delta is zero (to rounding on $30,000)
        and its gamma and vega are exactly zero -- the call's and the put's
        are the same number."""
        out, K, T, r = self._conversion()
        totals = out["totals"]
        assert abs(totals["dollar_delta"]) < 1e-9
        assert totals["dollar_gamma"] == 0.0
        assert totals["dollar_vega"] == 0.0
        # What is left is a bond: -K e^{-rT} per unit, on 300 units.
        assert totals["market_value"] == pytest.approx(
            -K * math.exp(-r * T) * 300, rel=1e-12
        )

    def test_its_full_revaluation_does_not_move_with_spot(self):
        """At every shock up to 50% the revalued conversion is worth what
        it was: full P&L zero to rounding on the notional."""
        out, *_ = self._conversion(shocks=[-0.5, -0.1, 0.1, 0.5])
        for row in out["scenarios"]:
            assert abs(row["full_pnl"]) < 1e-8, row
            assert abs(row["delta_gamma_pnl"]) < 1e-8, row


class TestTheDeltaGammaErrorIsThirdOrder:
    def test_it_shrinks_like_the_cube_of_the_shock(self):
        """A long call, revalued against delta-gamma: halving the shock cuts
        the error by about 8, and at the smallest shock it is the speed
        term speed x dS^3 / 6 x multiplier x quantity to within 2%."""
        S, K, T, sigma, r = 100.0, 100.0, 0.25, 0.3, 0.02
        shocks = [0.0025, 0.005, 0.01, 0.02]
        out = aggregate_option_book(
            [_line(strike=K, time_to_expiry=T, volatility=sigma, quantity=1)],
            {"XYZ": S},
            {"XYZ": r},
            spot_shocks=shocks,
        )
        errors = {row["shock"]: row["error"] for row in out["scenarios"]}
        for small, large in zip(shocks, shocks[1:]):
            assert errors[large] / errors[small] == pytest.approx(8.0, rel=0.1)
        speed = float(
            black_scholes_greeks_batch(S, K, T, sigma, r, 0.0, True, greeks="speed")[
                "speed"
            ]
        )
        dS = S * shocks[0]
        assert errors[shocks[0]] == pytest.approx(-(speed * dS**3 / 6) * 100, rel=0.02)

    def test_a_zero_shock_has_no_error_and_a_null_relative_error_with_its_reason(
        self, derivatives
    ):
        out = derivatives.dispatch(
            "aggregate_option_book",
            {
                "options": [_line()],
                "spots": {"XYZ": 100.0},
                "rates": {"XYZ": 0.03},
                "spot_shocks": [0.0, 0.05],
            },
        )
        zero = out["scenarios"][0]
        assert zero["full_pnl"] == 0.0 and zero["error"] == 0.0
        assert zero["relative_error"] is None
        assert any(
            "relative_error" in w and "nothing to be relative" in w
            for w in out["warnings"]
        )


class TestTheBookAddsUp:
    def _book(self, betas=True):
        options = [
            _line(underlying="AAA", strike=50, quantity=-12, volatility=0.4),
            _line(underlying="AAA", option_type="put", strike=45, quantity=5),
            _line(underlying="BBB", strike=210, quantity=4, multiplier=50),
            _line(
                underlying="CCC",
                option_type="put",
                strike=30,
                quantity=-20,
                time_to_expiry=0.1,
            ),
        ]
        return aggregate_option_book(
            options,
            {"AAA": 52.0, "BBB": 200.0, "CCC": 31.5, "DDD": 80.0},
            {"AAA": 0.05, "BBB": 0.05, "CCC": 0.05},
            dividend_yields={"AAA": 0.01},
            underlying_units={"AAA": 800, "DDD": -150},
            betas={"AAA": 1.3, "BBB": 0.9, "CCC": 1.6, "DDD": 0.4} if betas else None,
        )

    def test_the_beta_weighted_delta_is_the_sum_of_beta_times_dollar_delta(self):
        out = self._book()
        rows = out["by_underlying"]
        for row in rows:
            assert row["beta_weighted_delta"] == row["beta"] * row["dollar_delta"]
        assert out["totals"]["beta_weighted_delta"] == pytest.approx(
            math.fsum(r["beta"] * r["dollar_delta"] for r in rows), rel=1e-15
        )

    def test_the_book_is_its_underlyings_and_they_are_their_lines(self):
        out = self._book()
        rows = {r["underlying"]: r for r in out["by_underlying"]}
        assert list(rows) == ["AAA", "BBB", "CCC", "DDD"]
        spots = {"AAA": 52.0, "BBB": 200.0, "CCC": 31.5, "DDD": 80.0}
        units = {"AAA": 800, "DDD": -150}
        for u, row in rows.items():
            lines = [x for x in out["lines"] if x["underlying"] == u]
            assert row["n_options"] == len(lines)
            assert row["dollar_delta"] == pytest.approx(
                math.fsum(x["dollar_delta"] for x in lines)
                + units.get(u, 0) * spots[u],
                rel=1e-14,
            )
            assert row["dollar_gamma"] == pytest.approx(
                math.fsum(x["dollar_gamma"] for x in lines), rel=1e-14, abs=0
            )
        for name in ("dollar_delta", "dollar_gamma", "dollar_vega", "dollar_theta"):
            assert out["totals"][name] == pytest.approx(
                math.fsum(r[name] for r in rows.values()), rel=1e-14
            )
        for j, row in enumerate(out["scenarios"]):
            assert row["full_pnl"] == pytest.approx(
                math.fsum(r["scenarios"][j]["full_pnl"] for r in rows.values()),
                rel=1e-14,
            )
        # An underlying held only outright has a dollar delta of units x
        # spot, no rate, and no delta-gamma error.
        assert rows["DDD"]["dollar_delta"] == -150 * 80.0
        assert rows["DDD"]["rate"] is None
        assert all(s["error"] == 0.0 for s in rows["DDD"]["scenarios"])

    def test_the_default_grid_is_plus_and_minus_1_2_5_and_10_percent(self):
        out = self._book()
        assert out["spot_shocks"] == [-0.1, -0.05, -0.02, -0.01, 0.01, 0.02, 0.05, 0.1]
        assert len(out["scenarios"]) == 8

    def test_without_betas_the_beta_weighted_delta_is_null_with_its_reason(
        self, derivatives
    ):
        out = derivatives.dispatch(
            "aggregate_option_book",
            {
                "options": [_line()],
                "spots": {"XYZ": 100.0},
                "rates": {"XYZ": 0.03},
            },
        )
        assert out["totals"]["beta_weighted_delta"] is None
        assert out["by_underlying"][0]["beta"] is None
        assert any("no betas were given" in w for w in out["warnings"])
        assert any("non-payers" in w for w in out["warnings"])


class TestRefusals:
    def test_a_line_without_a_multiplier_is_refused_not_priced_at_100(
        self, derivatives
    ):
        line = _line()
        del line["multiplier"]
        with pytest.raises(pydantic.ValidationError, match="convention"):
            derivatives.dispatch(
                "aggregate_option_book",
                {"options": [line], "spots": {"XYZ": 100.0}, "rates": {"XYZ": 0.0}},
            )
        with pytest.raises(ValidationError, match="no multiplier"):
            aggregate_option_book([line], {"XYZ": 100.0}, {"XYZ": 0.0})

    @pytest.mark.parametrize(
        "kwargs, phrase",
        [
            (dict(rates={}), "no rate for"),
            (dict(spots={"OTHER": 10.0}), "spots does not price"),
            (dict(betas={"OTHER": 1.0}), "betas has none for"),
            (dict(spot_shocks=[-1.0]), "zero or below"),
            (dict(spot_shocks=[0.1, 0.1]), "repeats"),
        ],
    )
    def test_the_library_names_what_is_missing(self, kwargs, phrase):
        arguments = dict(spots={"XYZ": 100.0}, rates={"XYZ": 0.03})
        arguments.update(kwargs)
        with pytest.raises(ValidationError, match=phrase):
            aggregate_option_book([_line()], **arguments)

    def test_an_expiry_needs_a_valuation_date_and_must_be_ahead_of_it(self):
        line = _line(time_to_expiry=None, expiry="2026-11-20")
        del line["time_to_expiry"]
        with pytest.raises(ValidationError, match="valuation_date"):
            aggregate_option_book([line], {"XYZ": 100.0}, {"XYZ": 0.03})
        with pytest.raises(ValidationError, match="at or past expiry"):
            aggregate_option_book(
                [line], {"XYZ": 100.0}, {"XYZ": 0.03}, valuation_date="2026-11-20"
            )

    @pytest.mark.parametrize(
        "overrides, phrase",
        [
            ({"quantity": 0}, "not a position"),
            ({"expiry": "2027-01-15"}, "exactly one of time_to_expiry"),
            ({"multiplier": 0}, "greater than 0"),
        ],
    )
    def test_the_input_model_refuses_a_line_by_name(
        self, derivatives, overrides, phrase
    ):
        with pytest.raises(pydantic.ValidationError, match=phrase):
            derivatives.dispatch(
                "aggregate_option_book",
                {
                    "options": [_line(**overrides)],
                    "spots": {"XYZ": 100.0},
                    "rates": {"XYZ": 0.03},
                },
            )

    def test_a_rate_in_percent_is_refused(self, derivatives):
        with pytest.raises(pydantic.ValidationError, match="decimal"):
            derivatives.dispatch(
                "aggregate_option_book",
                {"options": [_line()], "spots": {"XYZ": 100.0}, "rates": {"XYZ": 25}},
            )
