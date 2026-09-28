"""
The option-chain bindings, called directly.

`analysis.options_batch` validates before it calls these, so two of the
kernel's reason codes -- an input outside the pricing domain, and a price
that is not finite -- are unreachable through the library. A direct caller
can still reach them, and must get a code with NaN outputs rather than
undefined behaviour or an answer. The bindings also refuse shapes and solver
settings the kernel has no meaning for, and release the GIL around the work.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

_cpp = pytest.importorskip(
    "standard_quant_tools._sqt_core", reason="native extension not built"
)
if not hasattr(_cpp, "implied_volatility_batch"):  # pragma: no cover
    pytest.skip("extension predates the option-chain kernels", allow_module_level=True)

INVALID_INPUT, NOT_PRICEABLE = 7, 6


def _iv(price, spot, strike, t, rate, q, call, **kw):
    arrays = [
        np.atleast_1d(np.asarray(x, dtype=float))
        for x in (price, spot, strike, t, rate, q)
    ]
    n = max(a.size for a in arrays)
    arrays = [np.broadcast_to(a, n).copy() for a in arrays]
    calls = np.broadcast_to(np.asarray(call, dtype=np.uint8), n).copy()
    return _cpp.implied_volatility_batch(*arrays, calls, **kw)


class TestReasonCodesADirectCallerCanReach:
    @pytest.mark.parametrize(
        "spot,strike,t,rate,q",
        [
            (0.0, 100.0, 1.0, 0.0, 0.0),
            (100.0, np.nan, 1.0, 0.0, 0.0),
            (100.0, 100.0, -1.0, 0.0, 0.0),
            (100.0, 100.0, 1.0, 11.0, 0.0),
            (100.0, 100.0, 100.0, -9.0, 0.0),
            (2e12, 100.0, 1.0, 0.0, 0.0),
        ],
    )
    def test_an_input_outside_the_domain_is_a_code_not_an_answer(
        self, spot, strike, t, rate, q
    ):
        out = _iv(5.0, spot, strike, t, rate, q, 1)
        assert out["reason"][0] == INVALID_INPUT
        assert np.isnan(out["implied_volatility"][0])
        assert out["converged"][0] == 0

    def test_a_discounted_strike_past_a_double_is_not_priceable(self):
        out = _iv(5.0, 100.0, 1e12, 77.7, -9.0, 0.0, 1)
        assert out["reason"][0] == NOT_PRICEABLE
        assert np.isnan(out["implied_volatility"][0])

    def test_a_good_contract_beside_a_bad_one_is_still_solved(self):
        out = _iv([4.759422392871528, 5.0], [42.0, 0.0], 40.0, 0.5, 0.10, 0.0, 1)
        assert out["reason"].tolist() == [0, INVALID_INPUT]
        assert out["implied_volatility"][0] == 0.2

    def test_a_greek_outside_the_domain_is_nan(self):
        out = _cpp.black_scholes_greeks_batch(
            np.array([100.0, 0.0]),
            np.array([100.0, 100.0]),
            np.array([0.5, 0.5]),
            np.array([0.2, 0.2]),
            np.zeros(2),
            np.zeros(2),
            np.ones(2, dtype=np.uint8),
        )
        assert np.isfinite(out["price"][0])
        assert all(np.isnan(out[k][1]) for k in out)


class TestTheBindingRefusesWhatTheKernelCannotMean:
    def test_a_two_dimensional_array_is_refused(self):
        a = np.ones((2, 2))
        with pytest.raises(ValueError, match="1-D"):
            _cpp.implied_volatility_batch(a, a, a, a, a, a, np.ones((2, 2), np.uint8))

    def test_mismatched_lengths_are_refused(self):
        with pytest.raises(ValueError, match="one entry per contract"):
            _cpp.implied_volatility_batch(
                np.ones(3),
                np.ones(2),
                np.ones(3),
                np.ones(3),
                np.ones(3),
                np.ones(3),
                np.ones(3, np.uint8),
            )

    @pytest.mark.parametrize(
        "kw,match",
        [
            (dict(initial_guess=0.0), "initial_guess"),
            (dict(initial_guess=float("nan")), "initial_guess"),
            (dict(tol=-1.0), "tol"),
            (dict(tol_sigma=float("inf")), "tol_sigma"),
            (dict(max_iterations=-1), "max_iterations"),
        ],
    )
    def test_solver_settings_are_bounded(self, kw, match):
        with pytest.raises(ValueError, match=match):
            _iv(5.0, 100.0, 100.0, 1.0, 0.0, 0.0, 1, **kw)

    def test_without_grid_spot_is_one_per_contract(self):
        with pytest.raises(ValueError, match="grid=True"):
            _cpp.black_scholes_greeks_batch(
                np.ones(3),
                np.ones(2),
                np.ones(2),
                np.ones(2),
                np.ones(2),
                np.ones(2),
                np.ones(2, np.uint8),
            )

    def test_the_grid_is_contracts_by_spots(self):
        out = _cpp.black_scholes_greeks_batch(
            np.linspace(80.0, 120.0, 5),
            np.array([90.0, 100.0, 110.0]),
            np.full(3, 0.5),
            np.full(3, 0.2),
            np.zeros(3),
            np.zeros(3),
            np.ones(3, np.uint8),
            True,
        )
        assert out["gamma"].shape == (3, 5)


class TestTheGilIsReleased:
    def test_concurrent_chains_get_their_own_answers(self):
        """Eight threads solving eight different chains at once must each get
        their own chain's volatilities back."""
        n = 2000
        rng = np.random.default_rng(0)
        # Near the money, where every quote solves by Newton to the vol it
        # was priced at (deep in the money a quote sits at intrinsic and
        # the answer is a ceiling, not the vol).
        strikes = 100.0 * np.exp(rng.normal(0.0, 0.05, n))
        ones = np.ones(n)
        calls = np.ones(n, np.uint8)
        results = {}
        errors = []

        def work(i):
            try:
                vol = 0.1 + 0.05 * i
                prices = _cpp.black_scholes_greeks_batch(
                    100.0 * ones,
                    strikes,
                    0.5 * ones,
                    vol * ones,
                    0.0 * ones,
                    0.0 * ones,
                    calls,
                )["price"]
                out = _cpp.implied_volatility_batch(
                    prices,
                    100.0 * ones,
                    strikes,
                    0.5 * ones,
                    0.0 * ones,
                    0.0 * ones,
                    calls,
                )
                results[i] = (vol, out)
            except Exception as exc:  # noqa: BLE001 - re-raised below
                errors.append(exc)

        threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        assert not errors, errors
        assert len(results) == 8
        for vol, out in results.values():
            assert (out["reason"] == 0).all()
            np.testing.assert_allclose(out["implied_volatility"], vol, atol=1e-8)
