"""
`binomial_lattice` is the numpy backward induction of
`analysis.pricing._binomial`, bit for bit on x86.

The induction was 95-98% of a binomial `price_option` call: one numpy
expression per level, 0.95 ms at the 200 steps the pricing tool defaults
to and 15 ms at 2,000. The kernel runs the same operations in the same
order over one buffer, with the powers still formed by numpy, so price,
delta and gamma are the numpy loop's doubles. The loop stays in the module
as the fallback (`_binomial_levels`) and is the reference here. See the
CHANGELOG entry of 2026-10-04.

The one place the two may part is the sign of a zero: numpy's maximum
returns its second argument on a +0.0/-0.0 tie on x86 and +0.0 on Arm,
and a put's payoff at a node priced exactly at the strike is -0.0. The
kernel follows x86, so on other machines zeros are compared by value.
"""

from __future__ import annotations

import math
import platform
import threading
from typing import Any

import numpy as np
import pytest

from standard_quant_tools.analysis import pricing

_cpp: Any = None
try:
    from standard_quant_tools import _sqt_core as _cpp  # type: ignore[attr-defined]

    HAS_CPP = hasattr(_cpp, "binomial_lattice")
except ImportError:
    HAS_CPP = False

pytestmark = pytest.mark.skipif(not HAS_CPP, reason="binomial_lattice not built")

_X86 = platform.machine().lower() in ("x86_64", "amd64", "i386", "i686", "x86")


def _same_double(got: float, want: float) -> bool:
    if got == 0.0 and want == 0.0 and not _X86:
        return True
    return np.float64(got).tobytes() == np.float64(want).tobytes()


def _price_both(**kw):
    native = pricing.price_option(**kw)
    try:
        pricing.HAS_CPP = False
        python = pricing.price_option(**kw)
    finally:
        pricing.HAS_CPP = True
    return native, python


def _assert_same(native, python):
    assert native.keys() == python.keys()
    for key in ("price", "delta", "gamma"):
        assert _same_double(native[key], python[key]), (key, native[key], python[key])
    for key in native:
        if key not in ("price", "delta", "gamma"):
            assert native[key] == python[key], key


def _lattice_inputs(t, vol, rate, q, steps):
    """The powers, probability and discount, formed as `_binomial` forms them."""
    dt = t / steps
    up = math.exp(vol * math.sqrt(dt))
    down = 1.0 / up
    growth = math.exp((rate - q) * dt)
    probability = (growth - down) / (up - down)
    discount = math.exp(-rate * dt)
    exponents = np.arange(steps + 1, dtype=float)
    return np.power(up, exponents), np.power(down, exponents), probability, discount


class TestTheKernelIsTheLoop:
    @pytest.mark.parametrize("steps", [10, 11, 57, 200, 1001, 2000])
    @pytest.mark.parametrize("option_type", ["call", "put"])
    @pytest.mark.parametrize("american", [False, True])
    def test_price_delta_and_gamma(self, steps, option_type, american):
        for strike in (70.0, 99.5, 100.0, 100.5, 140.0):
            for rate, q in ((0.04, 0.03), (0.0, 0.0), (-0.01, 0.02), (0.08, 0.0)):
                _assert_same(
                    *_price_both(
                        spot=100.0,
                        strike=strike,
                        time_to_expiry=0.75,
                        volatility=0.35,
                        risk_free_rate=rate,
                        option_type=option_type,
                        model="binomial",
                        dividend_yield=q,
                        american=american,
                        steps=steps,
                    )
                )

    @pytest.mark.parametrize("seed", range(4))
    def test_random_contracts(self, seed):
        rng = np.random.default_rng(seed)
        for _ in range(40):
            kw = dict(
                spot=float(rng.uniform(5, 500)),
                time_to_expiry=float(rng.uniform(0.01, 3.0)),
                volatility=float(rng.uniform(0.05, 1.2)),
                risk_free_rate=float(rng.uniform(-0.02, 0.1)),
                dividend_yield=float(rng.uniform(0.0, 0.08)),
                option_type=str(rng.choice(["call", "put"])),
                american=bool(rng.integers(0, 2)),
                steps=int(rng.integers(10, 700)),
                model="binomial",
            )
            kw["strike"] = float(kw["spot"] * rng.uniform(0.5, 1.6))
            try:
                native, python = _price_both(**kw)
            except Exception as exc:  # the same refusal on both paths
                pricing.HAS_CPP = False
                try:
                    with pytest.raises(type(exc)):
                        pricing.price_option(**kw)
                finally:
                    pricing.HAS_CPP = True
                continue
            _assert_same(native, python)

    def test_every_level_value_at_an_exact_strike_node(self):
        """Nodes priced exactly at the strike, where a put's payoff is -0.0:
        powers of 2 and 1/2 put the center of every even level on the
        strike exactly, and CRR powers at spot == strike now and then. The
        six level values are compared raw."""
        exact = 12
        cases = [
            (
                np.power(2.0, np.arange(exact + 1.0)),
                np.power(0.5, np.arange(exact + 1.0)),
                1 / 3,
                0.99,
            )
        ]
        cases += [
            _lattice_inputs(0.5, 0.3, 0.04, 0.01, steps) for steps in (10, 50, 200)
        ]
        for up, down, p, disc in cases:
            steps = len(up) - 1

            def nodes(level):
                return 100.0 * up[level::-1] * down[: level + 1]

            for strike in (25.0, 100.0, 400.0):
                for sign in (1.0, -1.0):
                    for american in (False, True):
                        got = _cpp.binomial_lattice(
                            up, down, 100.0, strike, sign, p, disc, american
                        )
                        price, (_, v1), (_, v2) = pricing._binomial_levels(
                            nodes, steps, strike, sign, p, disc, american
                        )
                        want = [price, *v1, *v2]
                        assert all(_same_double(g, w) for g, w in zip(got, want))

    def test_a_deep_out_of_the_money_option_prices_zero_on_both(self):
        native, python = _price_both(
            spot=100.0,
            strike=1e6,
            time_to_expiry=0.02,
            volatility=0.1,
            risk_free_rate=0.03,
            option_type="call",
            model="binomial",
            american=True,
            steps=60,
        )
        assert native["price"] == python["price"] == 0.0
        _assert_same(native, python)


class TestTheBinding:
    def test_the_power_arrays_must_agree(self):
        with pytest.raises(ValueError, match="same"):
            _cpp.binomial_lattice(
                np.ones(11), np.ones(10), 1.0, 1.0, 1.0, 0.5, 1.0, True
            )

    def test_three_steps_at_least(self):
        with pytest.raises(ValueError, match="at least 3 steps"):
            _cpp.binomial_lattice(np.ones(3), np.ones(3), 1.0, 1.0, 1.0, 0.5, 1.0, True)

    def test_two_dimensional_powers_are_refused(self):
        with pytest.raises(ValueError, match="1-D"):
            _cpp.binomial_lattice(
                np.ones((4, 4)), np.ones(16), 1.0, 1.0, 1.0, 0.5, 1.0, True
            )

    def test_concurrent_calls_keep_their_own_trees(self):
        """The induction runs with the GIL released."""
        inputs = [
            _lattice_inputs(1.0, 0.3 + 0.01 * i, 0.04, 0.02, 800) for i in range(8)
        ]
        want = [
            _cpp.binomial_lattice(u, d, 100.0, 80.0 + 5 * i, -1.0, p, disc, True)
            for i, (u, d, p, disc) in enumerate(inputs)
        ]
        got: list = [None] * 8

        def work(i):
            u, d, p, disc = inputs[i]
            for _ in range(10):
                got[i] = _cpp.binomial_lattice(
                    u, d, 100.0, 80.0 + 5 * i, -1.0, p, disc, True
                )

        threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
        for g, w in zip(got, want):
            assert g.tobytes() == w.tobytes()

    def test_the_docstring_names_the_contract(self):
        doc = _cpp.binomial_lattice.__doc__
        assert "bit for bit" in doc and "ValueError" in doc
