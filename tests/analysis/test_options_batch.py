"""
The chain functions answer what the single-contract functions answer.

`analysis.options_batch` exists so an option chain is one call rather than
one per contract. Its whole value rests on being the SAME computation: an
implied volatility, an iteration count, a converged flag and a refusal that
differ from the scalar's would make the batch a second model to reconcile
rather than a faster route to the first. So every test here compares three
things -- the compiled kernel, the numpy fallback, and the scalar functions
in `analysis.options` / `analysis.derivatives` -- on randomized chains that
reach every branch: deep in and out of the money, hours from expiry, zero
and negative rates, dividends of either sign, puts and calls, quotes at
intrinsic, and quotes no volatility can reproduce.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from standard_quant_tools.analysis import options_batch as ob
from standard_quant_tools.analysis.derivatives import option_greeks
from standard_quant_tools.analysis.options import (
    black_scholes_greeks,
    black_scholes_price,
    implied_volatility,
)
from standard_quant_tools.analysis.pricing import price_option
from standard_quant_tools.error import ValidationError

NATIVE = ob._native("implied_volatility_batch") is not None and (
    ob._native("black_scholes_greeks_batch") is not None
)

BACKENDS = [
    pytest.param(
        "native", marks=pytest.mark.skipif(not NATIVE, reason="extension not built")
    ),
    "python",
]


@pytest.fixture(params=BACKENDS)
def backend(request, monkeypatch):
    """Run a test once on the compiled kernel and once on the fallback."""
    if request.param == "python":
        monkeypatch.setattr(ob, "HAS_CPP", False)
    return request.param


def _path(backend: str) -> str:
    return "C++" if backend == "native" else "python"


SPOT = 100.0


def _chain(seed: int, n: int = 500):
    """A chain that reaches every branch of the solver.

    Strikes are log-normal around the spot with a wide spread (deep in and
    out of the money on both sides); expiries run from six hours to five
    years; rates and yields include zero and negative values. Quotes are
    the model price perturbed by 2% -- so some fall outside the bounds --
    with exact model prices (deep in the money, bit-for-bit at intrinsic),
    zeros, NaNs, prices past the upper bound and prices only a volatility
    past 500% reaches mixed in.
    """
    rng = np.random.default_rng(seed)
    strike = SPOT * np.exp(rng.normal(0.0, 0.7, n))
    t = np.exp(rng.uniform(math.log(0.25 / 365.0), math.log(5.0), n))
    vol = rng.uniform(0.03, 2.5, n)
    rate = rng.choice([0.0, -0.015, 0.02, 0.07], n)
    q = rng.choice([0.0, -0.01, 0.018, 0.05], n)
    call = rng.random(n) < 0.5
    fair = np.array(
        [
            black_scholes_price(SPOT, k, tt, r, v, "call" if c else "put", qq)
            for k, tt, r, v, c, qq in zip(strike, t, rate, vol, call, q)
        ]
    )
    price = fair * np.exp(rng.normal(0.0, 0.02, n))
    price[::9] = fair[::9]
    upper = np.where(call, SPOT * np.exp(-q * t), strike * np.exp(-rate * t))
    price[3::41] = 0.0
    price[5::53] = np.nan
    price[7::67] = upper[7::67] * 1.1
    price[11::71] = upper[11::71] * 0.999
    return dict(price=price, strike=strike, t=t, rate=rate, q=q, call=call, vol=vol)


def _scalar(price, k, t, r, q, call):
    try:
        return (
            implied_volatility(price, SPOT, k, t, r, "call" if call else "put", q),
            None,
        )
    except ValidationError as err:
        return None, str(err)


#: The scalar's refusal message, and the batch reasons that correspond to it.
#: The scalar says "outside the no-arbitrage range" for a NaN quote as well
#: as for one past a bound; the batch tells the three apart.
_REFUSALS = {
    "must be > 0": {"price_not_positive"},
    "outside the no-arbitrage range": {
        "price_not_finite",
        "below_lower_bound",
        "above_upper_bound",
    },
    "no root was found": {"no_root_in_bracket"},
}


def _batch(chain, **kwargs):
    return ob.implied_volatility_batch(
        chain["price"],
        SPOT,
        chain["strike"],
        chain["t"],
        chain["rate"],
        chain["q"],
        chain["call"],
        **kwargs,
    )


class TestImpliedVolatilityIsTheScalarsContractForContract:
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_every_contract_matches_the_scalar(self, backend, seed):
        chain = _chain(seed)
        out = _batch(chain)
        assert out["path"] == _path(backend)
        for i in range(chain["price"].size):
            ref, refusal = _scalar(
                chain["price"][i],
                chain["strike"][i],
                chain["t"][i],
                chain["rate"][i],
                chain["q"][i],
                chain["call"][i],
            )
            if ref is None:
                expected = next(v for k, v in _REFUSALS.items() if k in refusal)
                assert out["reason"][i] in expected, (i, refusal)
                assert math.isnan(out["implied_volatility"][i])
                assert not out["converged"][i]
                assert out["method"][i] == "none"
                continue
            assert out["reason"][i] == "solved", (i, ref)
            assert bool(out["converged"][i]) == ref["converged"]
            assert int(out["iterations"][i]) == ref["iterations"]
            assert out["method"][i] == ref["method"]
            assert bool(out["at_bound"][i]) == ref["at_bound"]
            assert out["implied_volatility"][i] == pytest.approx(
                ref["implied_volatility"], rel=1e-12, abs=0.0
            )
            assert out["price_error"][i] == pytest.approx(
                ref["price_error"], rel=1e-12, abs=1e-15
            )

    def test_the_chain_reaches_every_branch(self):
        """The differential test above is only as good as the chain: this
        pins that it exercises Newton, both bisections and every refusal."""
        out = _batch(_chain(0))
        reasons = set(out["reason"].tolist())
        assert reasons == set(ob.REASONS) - {"not_priceable", "invalid_input"}
        solved = out["reason"] == "solved"
        assert {"newton", "bisection"} <= set(out["method"][solved].tolist())
        assert out["at_bound"].any()

    @pytest.mark.skipif(not NATIVE, reason="extension not built")
    @pytest.mark.parametrize("seed", [3, 4])
    def test_the_kernel_and_the_fallback_agree(self, monkeypatch, seed):
        chain = _chain(seed, n=800)
        native = _batch(chain)
        monkeypatch.setattr(ob, "HAS_CPP", False)
        python = _batch(chain)
        assert (native["path"], python["path"]) == ("C++", "python")
        for key in ("reason", "converged", "iterations", "method", "at_bound"):
            np.testing.assert_array_equal(native[key], python[key], err_msg=key)
        for key in ("implied_volatility", "price_error"):
            np.testing.assert_allclose(
                native[key], python[key], rtol=1e-12, atol=1e-15, err_msg=key
            )
        assert native["refusals"] == python["refusals"]

    def test_solver_settings_reach_every_contract(self, backend):
        """The scalar's keyword arguments mean the same thing per contract."""
        chain = _chain(5, n=120)
        settings = dict(initial_guess=0.9, tol=1e-9, max_iterations=3, tol_sigma=1e-11)
        out = _batch(chain, **settings)
        for i in range(chain["price"].size):
            try:
                ref = implied_volatility(
                    chain["price"][i],
                    SPOT,
                    chain["strike"][i],
                    chain["t"][i],
                    chain["rate"][i],
                    "call" if chain["call"][i] else "put",
                    chain["q"][i],
                    **settings,
                )
            except ValidationError:
                assert out["reason"][i] != "solved"
                continue
            assert int(out["iterations"][i]) == ref["iterations"]
            assert out["method"][i] == ref["method"]
            assert out["implied_volatility"][i] == pytest.approx(
                ref["implied_volatility"], rel=1e-12
            )


class TestAQuoteIsReportedAndAnInputIsRefused:
    def test_one_bad_quote_does_not_refuse_the_chain(self, backend):
        fair = black_scholes_price(SPOT, 100.0, 0.5, 0.03, 0.25, "call")
        prices = [fair, 0.0, float("nan"), 150.0, fair * 1.01]
        out = ob.implied_volatility_batch(prices, SPOT, 100.0, 0.5, 0.03)
        assert out["reason"].tolist() == [
            "solved",
            "price_not_positive",
            "price_not_finite",
            "above_upper_bound",
            "solved",
        ]
        assert out["implied_volatility"][0] == pytest.approx(0.25, abs=1e-10)
        assert np.isnan(out["implied_volatility"][1:4]).all()
        assert out["n_solved"] == 2
        assert out["refusals"] == {
            "price_not_positive": 1,
            "price_not_finite": 1,
            "above_upper_bound": 1,
        }

    @pytest.mark.parametrize(
        "price,reason",
        [
            (0.0, "price_not_positive"),
            (-1.0, "price_not_positive"),
            (float("nan"), "price_not_finite"),
            (float("inf"), "price_not_finite"),
            (101.0, "above_upper_bound"),
            (99.0, "no_root_in_bracket"),
        ],
    )
    def test_each_reason_is_one_the_scalar_refuses(self, backend, price, reason):
        """Null case for the codes: every quote reported as unsolved is one
        the scalar function raises on."""
        out = ob.implied_volatility_batch(price, SPOT, 100.0, 1.0, 0.0)
        assert out["reason"].item() == reason
        with pytest.raises(ValidationError):
            implied_volatility(price, SPOT, 100.0, 1.0, 0.0)

    def test_below_the_lower_bound(self, backend):
        out = ob.implied_volatility_batch(9.0, 110.0, 100.0, 1.0, 0.0)
        assert out["reason"].item() == "below_lower_bound"

    @pytest.mark.parametrize(
        "field,value,words",
        [
            ("spot", 0.0, "spot must be > 0"),
            ("strike", -5.0, "strike must be > 0"),
            ("time_to_expiry", 0.0, "time_to_expiry must be > 0"),
            ("time_to_expiry", 150.0, "time_to_expiry=150"),
            ("risk_free_rate", 11.0, "risk_free_rate=11"),
            ("dividend_yield", float("nan"), "dividend_yield=nan"),
            ("strike", 2e12, "strike=2e+12"),
        ],
    )
    def test_a_unit_error_refuses_the_batch_naming_the_contract(
        self, backend, field, value, words
    ):
        args = dict(
            option_price=np.full(5, 5.0),
            spot=np.full(5, SPOT),
            strike=np.full(5, 100.0),
            time_to_expiry=np.full(5, 0.5),
            risk_free_rate=np.full(5, 0.02),
            dividend_yield=np.zeros(5),
        )
        args[field] = args[field].copy()
        args[field][3] = value
        with pytest.raises(ValidationError, match="contract 3") as err:
            ob.implied_volatility_batch(**args)
        assert words in str(err.value)

    def test_rate_times_time_past_exp_is_refused(self, backend):
        with pytest.raises(ValidationError, match="risk_free_rate x time_to_expiry"):
            ob.implied_volatility_batch(5.0, SPOT, 100.0, 100.0, -9.0)

    def test_a_discounted_strike_past_a_double_is_refused(self, backend):
        """Each factor inside its bound, the product 1e316: the scalar finds
        this as a non-finite price mid-solve; the batch before any solve."""
        with pytest.raises(ValidationError, match="Check the rate units"):
            ob.implied_volatility_batch(5.0, SPOT, 1e12, 77.7, -9.0)

    def test_a_two_dimensional_chain_names_the_contract_by_position(self, backend):
        strikes = np.full((2, 3), 100.0)
        strikes[1, 2] = 0.0
        with pytest.raises(ValidationError, match=r"contract \(1, 2\)"):
            ob.implied_volatility_batch(5.0, SPOT, strikes, 0.5, 0.0)

    def test_option_type_words_are_refused_with_the_remedy(self):
        with pytest.raises(ValidationError, match="== 'call'"):
            ob.implied_volatility_batch(5.0, SPOT, 100.0, 0.5, 0.0, 0.0, ["call"])

    def test_arrays_that_do_not_broadcast_are_refused(self):
        with pytest.raises(ValidationError, match="do not broadcast"):
            ob.implied_volatility_batch(
                [5.0, 6.0], SPOT, [90.0, 100.0, 110.0], 0.5, 0.0
            )

    @pytest.mark.parametrize(
        "setting",
        [
            dict(initial_guess=0.0),
            dict(initial_guess=150.0),
            dict(tol=-1.0),
            dict(tol_sigma=float("nan")),
            dict(max_iterations=-1),
            dict(max_iterations=2.5),
            dict(max_iterations=True),
        ],
    )
    def test_solver_settings_are_bounded(self, setting):
        with pytest.raises(ValidationError):
            ob.implied_volatility_batch(5.0, SPOT, 100.0, 0.5, 0.0, **setting)


class TestShapes:
    def test_scalars_broadcast_and_shapes_are_kept(self, backend):
        strikes = np.array([[90.0, 100.0], [110.0, 120.0]])
        prices = np.array(
            [
                [black_scholes_price(SPOT, k, 0.5, 0.01, 0.3, "put") for k in row]
                for row in strikes
            ]
        )
        out = ob.implied_volatility_batch(prices, SPOT, strikes, 0.5, 0.01, 0.0, False)
        assert out["implied_volatility"].shape == (2, 2)
        np.testing.assert_allclose(out["implied_volatility"], 0.3, atol=1e-9)

    def test_all_scalars_is_one_contract(self, backend):
        out = ob.implied_volatility_batch(4.759422392871528, 42.0, 40.0, 0.5, 0.10)
        ref = implied_volatility(4.759422392871528, 42.0, 40.0, 0.5, 0.10)
        assert out["implied_volatility"].shape == ()
        assert out["implied_volatility"].item() == ref["implied_volatility"]
        assert out["iterations"].item() == ref["iterations"] == 1

    def test_an_empty_chain_is_an_empty_answer(self, backend):
        """Null case: no contracts, no refusal, nothing solved."""
        out = ob.implied_volatility_batch([], SPOT, [], 0.5, 0.0)
        assert out["implied_volatility"].shape == (0,)
        assert out["n_contracts"] == 0 and out["n_solved"] == 0
        assert out["refusals"] == {}
        greeks = ob.black_scholes_greeks_batch(SPOT, [], 0.5, 0.2, 0.0)
        assert greeks["gamma"].shape == (0,)


# ── greeks ──────────────────────────────────────────────────────────────


def _greek_inputs(seed: int, n: int = 400):
    rng = np.random.default_rng(seed)
    return dict(
        spot=SPOT * np.exp(rng.normal(0.0, 0.3, n)),
        strike=SPOT * np.exp(rng.normal(0.0, 0.7, n)),
        t=np.exp(rng.uniform(math.log(0.25 / 365.0), math.log(5.0), n)),
        vol=rng.uniform(0.03, 2.5, n),
        rate=rng.choice([0.0, -0.015, 0.02, 0.07], n),
        q=rng.choice([0.0, -0.01, 0.018, 0.05], n),
        call=rng.random(n) < 0.5,
    )


def _greeks(g, **kwargs):
    return ob.black_scholes_greeks_batch(
        g["spot"], g["strike"], g["t"], g["vol"], g["rate"], g["q"], g["call"], **kwargs
    )


class TestGreeksAreOptionGreeks:
    @pytest.mark.parametrize("seed", [0, 1])
    def test_every_greek_matches_the_scalar(self, backend, seed):
        g = _greek_inputs(seed)
        out = _greeks(g)
        assert out["path"] == _path(backend)
        for i in range(g["spot"].size):
            ref = option_greeks(
                spot=g["spot"][i],
                strike=g["strike"][i],
                time_to_expiry=g["t"][i],
                volatility=g["vol"][i],
                risk_free_rate=g["rate"][i],
                option_type="call" if g["call"][i] else "put",
                dividend_yield=g["q"][i],
            )
            for key in ob.GREEKS:
                assert out[key][i] == pytest.approx(ref[key], rel=1e-12, abs=1e-12), (
                    i,
                    key,
                )

    def test_the_price_is_price_options(self, backend):
        g = _greek_inputs(2, n=150)
        out = _greeks(g)
        for i in range(g["spot"].size):
            ref = price_option(
                spot=g["spot"][i],
                strike=g["strike"][i],
                time_to_expiry=g["t"][i],
                volatility=g["vol"][i],
                risk_free_rate=g["rate"][i],
                option_type="call" if g["call"][i] else "put",
                dividend_yield=g["q"][i],
            )
            for key in ("price", "delta", "gamma", "vega", "theta", "rho"):
                assert out[key][i] == pytest.approx(ref[key], rel=1e-12, abs=1e-12)

    def test_the_raw_greeks_agree_once_scaled(self, backend):
        """`black_scholes_greeks` reports vega per 1.0 of vol, theta per year
        and rho per 1.0 of rate; scaled, they are the batch's numbers. Its
        put delta is formed as N(d1) - 1 rather than -N(-d1), so the two
        agree to rounding rather than to the bit."""
        g = _greek_inputs(3, n=150)
        out = _greeks(g)
        for i in range(g["spot"].size):
            raw = black_scholes_greeks(
                g["spot"][i],
                g["strike"][i],
                g["t"][i],
                g["rate"][i],
                g["vol"][i],
                "call" if g["call"][i] else "put",
                g["q"][i],
            )
            assert out["delta"][i] == pytest.approx(raw["delta"], abs=1e-12)
            assert out["gamma"][i] == pytest.approx(raw["gamma"], rel=1e-12, abs=1e-15)
            assert out["vega"][i] == pytest.approx(raw["vega"] / 100.0, rel=1e-12)
            assert out["theta"][i] == pytest.approx(
                raw["theta"] / 365.0, rel=1e-9, abs=1e-12
            )
            assert out["rho"][i] == pytest.approx(
                raw["rho"] / 100.0, rel=1e-9, abs=1e-12
            )

    @pytest.mark.skipif(not NATIVE, reason="extension not built")
    def test_the_kernel_and_the_fallback_agree(self, monkeypatch):
        g = _greek_inputs(4, n=1000)
        native = _greeks(g)
        monkeypatch.setattr(ob, "HAS_CPP", False)
        python = _greeks(g)
        for key in ob.GREEKS:
            np.testing.assert_allclose(
                native[key], python[key], rtol=1e-12, atol=1e-12, err_msg=key
            )


class TestTheSpotGrid:
    def test_every_contract_at_every_spot(self, backend):
        g = _greek_inputs(5, n=40)
        spots = np.linspace(60.0, 140.0, 17)
        out = _greeks(dict(g, spot=spots), grid=True)
        assert out["gamma"].shape == (40, 17)
        for j, s in enumerate(spots):
            column = _greeks(dict(g, spot=s))
            for key in ob.GREEKS:
                np.testing.assert_array_equal(out[key][:, j], column[key], err_msg=key)

    @pytest.mark.skipif(not NATIVE, reason="extension not built")
    def test_the_kernel_and_the_fallback_agree_on_a_grid(self, monkeypatch):
        g = _greek_inputs(6, n=476)
        spots = np.linspace(70.0, 130.0, 61)
        native = _greeks(dict(g, spot=spots), grid=True)
        monkeypatch.setattr(ob, "HAS_CPP", False)
        python = _greeks(dict(g, spot=spots), grid=True)
        for key in ob.GREEKS:
            np.testing.assert_allclose(
                native[key], python[key], rtol=1e-12, atol=1e-12, err_msg=key
            )

    def test_a_grid_is_one_dimensional(self):
        with pytest.raises(ValidationError, match="1-D grid"):
            ob.black_scholes_greeks_batch(
                np.ones((2, 2)) * SPOT, 100.0, 0.5, 0.2, 0.0, grid=True
            )

    def test_a_bad_spot_on_the_grid_is_named(self, backend):
        with pytest.raises(ValidationError, match="spot 2: spot must be > 0"):
            ob.black_scholes_greeks_batch(
                [90.0, 100.0, -1.0], 100.0, 0.5, 0.2, 0.0, grid=True
            )

    def test_a_bad_contract_is_named(self, backend):
        with pytest.raises(ValidationError, match="contract 1.*volatility"):
            ob.black_scholes_greeks_batch(
                SPOT, [90.0, 100.0], 0.5, [0.2, 0.0], 0.0, grid=False
            )

    def test_the_units_are_stated(self):
        out = ob.black_scholes_greeks_batch(SPOT, 100.0, 0.5, 0.2, 0.0)
        assert out["units"]["theta"] == "change in price per calendar day"
        assert out["units"]["vega"] == "change in price per 1 volatility point (0.01)"


# ── zero gamma ──────────────────────────────────────────────────────────


class TestZeroGammaSpot:
    def test_a_call_spread_crosses_where_the_closed_form_says(self, backend):
        """Long the 90 call, short the 110, one volatility and expiry, no
        carry: the two gammas are equal where d1(90) = -d1(110), which is
        S = sqrt(90 x 110) x exp(-vol^2 T / 2)."""
        vol, t = 0.25, 0.5
        out = ob.zero_gamma_spot(
            [90.0, 110.0], t, vol, [1.0, -1.0], spot_low=60.0, spot_high=150.0
        )
        expected = math.sqrt(90.0 * 110.0) * math.exp(-vol * vol * t / 2.0)
        assert out["crossing_found"] is True
        assert out["n_crossings"] == 1
        assert out["zero_gamma_spot"] == pytest.approx(expected, rel=1e-9)
        assert out["path"] == _path(backend)
        assert out["reason"] is None
        # Brent, not bisection: from a grid step of 0.45 to 1e-10 relative
        # is about 30 halvings, and each evaluation prices the whole book.
        assert out["refinement_evaluations"] <= 12

    def test_net_gamma_changes_sign_across_the_answer(self, backend):
        rng = np.random.default_rng(8)
        k = SPOT * np.exp(rng.normal(0.0, 0.15, 476))
        t = rng.uniform(0.02, 1.0, 476)
        v = rng.uniform(0.15, 0.6, 476)
        qty = np.where(k > SPOT, -1.0, 1.0) * rng.uniform(1.0, 50.0, 476)
        out = ob.zero_gamma_spot(k, t, v, qty, spot_low=50.0, spot_high=150.0)
        assert out["crossing_found"]
        s0 = out["zero_gamma_spot"]

        def net(s):
            gamma = ob.black_scholes_greeks_batch(s, k, t, v, 0.0)["gamma"]
            return float((qty * gamma).sum())

        assert net(s0 * (1 - 1e-6)) * net(s0 * (1 + 1e-6)) < 0

    @pytest.mark.skipif(not NATIVE, reason="extension not built")
    def test_the_kernel_and_the_fallback_find_the_same_spot(self, monkeypatch):
        rng = np.random.default_rng(9)
        k = SPOT * np.exp(rng.normal(0.0, 0.2, 200))
        t = rng.uniform(0.05, 1.0, 200)
        v = rng.uniform(0.15, 0.6, 200)
        qty = np.where(rng.random(200) < 0.5, -1.0, 1.0)
        kw = dict(spot_low=40.0, spot_high=200.0, risk_free_rate=0.03)
        native = ob.zero_gamma_spot(k, t, v, qty, **kw)
        monkeypatch.setattr(ob, "HAS_CPP", False)
        python = ob.zero_gamma_spot(k, t, v, qty, **kw)
        assert native["crossings"] == pytest.approx(python["crossings"], rel=1e-12)
        assert native["n_crossings"] == python["n_crossings"]

    def test_a_long_book_has_no_crossing_and_none_is_invented(self, backend):
        """The null case: every leg long, so net gamma is positive at every
        spot. No spot is returned, and the reason says which side of zero
        the book sits on."""
        out = ob.zero_gamma_spot(
            [90.0, 100.0, 110.0],
            0.5,
            0.25,
            [1.0, 2.0, 1.0],
            spot_low=50.0,
            spot_high=150.0,
        )
        assert out["zero_gamma_spot"] is None
        assert out["crossing_found"] is False
        assert out["crossings"] == []
        assert "positive (long gamma)" in out["reason"]
        assert out["net_gamma_at_low"] > 0 and out["net_gamma_at_high"] > 0

    def test_a_short_book_says_short(self, backend):
        out = ob.zero_gamma_spot(
            [95.0, 105.0], 0.25, 0.3, [-1.0, -1.0], spot_low=50.0, spot_high=150.0
        )
        assert out["zero_gamma_spot"] is None
        assert "negative (short gamma)" in out["reason"]

    def test_a_bracket_without_gamma_is_not_a_crossing(self, backend):
        """Far from every strike and an hour from expiry, every gamma
        underflows to exactly zero. That is gamma being absent, not gamma
        changing sign."""
        out = ob.zero_gamma_spot(
            [100.0, 105.0],
            1e-4,
            0.2,
            [1.0, -1.0],
            spot_low=1000.0,
            spot_high=2000.0,
        )
        assert out["zero_gamma_spot"] is None
        assert "exactly zero at every scanned spot" in out["reason"]

    def test_several_crossings_are_all_reported(self, backend):
        """A short butterfly body between long wings is long gamma in the
        wings and short at the body: two crossings."""
        out = ob.zero_gamma_spot(
            [80.0, 100.0, 120.0],
            0.25,
            0.2,
            [1.0, -2.0, 1.0],
            spot_low=60.0,
            spot_high=140.0,
            reference_spot=105.0,
        )
        assert out["n_crossings"] == 2
        low, high = out["crossings"]
        assert low < 100.0 < high
        assert out["zero_gamma_spot"] == high  # the one nearer 105
        assert any("changes sign 2 times" in w for w in out["warnings"])

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            (dict(spot_low=150.0, spot_high=50.0), "bracket"),
            (dict(spot_low=0.0, spot_high=50.0), "bracket"),
            (dict(spot_low=50.0, spot_high=150.0, n_grid=2), "n_grid"),
            (dict(spot_low=50.0, spot_high=150.0, xtol=0.0), "xtol"),
        ],
    )
    def test_the_search_settings_are_bounded(self, kwargs, match):
        with pytest.raises(ValidationError, match=match):
            ob.zero_gamma_spot([100.0], 0.5, 0.2, [1.0], **kwargs)

    def test_a_book_of_zero_positions_is_refused(self):
        with pytest.raises(ValidationError, match="every quantity is zero"):
            ob.zero_gamma_spot(
                [90.0, 110.0], 0.5, 0.2, [0.0, 0.0], spot_low=50.0, spot_high=150.0
            )

    def test_a_non_finite_position_is_refused(self):
        with pytest.raises(ValidationError, match="contract 1: quantity"):
            ob.zero_gamma_spot(
                [90.0, 110.0],
                0.5,
                0.2,
                [1.0, float("nan")],
                spot_low=50.0,
                spot_high=150.0,
            )


class TestTheBackendChoice:
    def test_the_fallback_is_taken_when_the_extension_is_off(self, monkeypatch):
        monkeypatch.setattr(ob, "HAS_CPP", False)
        assert ob._native("implied_volatility_batch") is None
        out = ob.implied_volatility_batch(4.759422392871528, 42.0, 40.0, 0.5, 0.10)
        assert out["path"] == "python"

    def test_an_extension_without_the_kernel_falls_back_for_that_kernel(
        self, monkeypatch
    ):
        """Per symbol, the way the other modules probe: an extension that
        predates a kernel answers through the fallback for that kernel."""

        class _Old:
            pass

        monkeypatch.setattr(ob, "HAS_CPP", True)
        monkeypatch.setattr(ob, "_cpp_core", _Old())
        out = ob.black_scholes_greeks_batch(SPOT, 100.0, 0.5, 0.2, 0.0)
        assert out["path"] == "python"
