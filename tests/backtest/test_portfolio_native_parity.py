"""
The native portfolio-simulation fast path must agree with the Python loop.

`run_portfolio_simulation` has two implementations of its per-bar loop: a
native kernel for the configuration `_apply_rebalance` already treats as its
vectorized fast path (percentage commission, no impact model, no ADV
constraint), and the original Python loop for everything else. The kernel
exists to be faster, not different.

NOT bit-identical, and deliberately so. The Python accumulates its per-bar
sums with `np.sum`, which uses pairwise summation; the kernel accumulates
sequentially. That is a reassociation of the same additions, so the results
differ in the last few bits. Measured across horizons and universe sizes the
gap is 4.5-20 ULPs on the equity curve, growing like the square root of the
number of accumulation steps -- the signature of rounding, not of a bug. The
same convention (and the same reasoning) applies to `rolling_beta`'s AVX2
path, which is likewise tolerance-gated rather than assumed equal.

Run:
    pytest tests/backtest/test_portfolio_native_parity.py -v
"""

import numpy as np
import pandas as pd
import pytest

import standard_quant_tools.backtest.portfolio_engine as pe
from standard_quant_tools.error import ValidationError

requires_cpp = pytest.mark.skipif(not pe.HAS_CPP, reason="_sqt_core not built")

# 20 ULPs of headroom over the measured worst case, on a relative scale.
REL_TOL = 1e-13


def _universe(n_tickers=6, n_bars=250, seed=0, ragged=False, freq="B"):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-01", periods=n_bars, freq=freq)
    data = {}
    for i in range(n_tickers):
        cl = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, n_bars)))
        frame = pd.DataFrame(
            {
                "Open": cl * 0.999,
                "High": cl * 1.01,
                "Low": cl * 0.99,
                "Close": cl,
                "Volume": np.full(n_bars, 1e7),
            },
            index=idx,
        )
        data[f"T{i:03d}"] = frame.iloc[2:] if (ragged and i % 3 == 0) else frame
    reb = idx[5::5]
    w = rng.normal(0, 1, (len(reb), n_tickers))
    w /= np.abs(w).sum(axis=1, keepdims=True)
    return data, pd.DataFrame(w, index=reb, columns=list(data))


def _both(data, weights, **kwargs):
    """Run once natively and once through the Python loop."""
    native = pe.run_portfolio_simulation(data, weights, **kwargs)
    saved = pe.HAS_CPP
    pe.HAS_CPP = False
    try:
        looped = pe.run_portfolio_simulation(data, weights, **kwargs)
    finally:
        pe.HAS_CPP = saved
    return native, looped


def _helper_kwargs(data, weights, **kwargs):
    """The arguments `run_portfolio_simulation` would hand the native helper.

    Built by running the real engine once and capturing them, so this cannot
    drift from the production call site the way a hand-written copy would.
    """
    captured = {}
    real = pe._native_portfolio_sim

    def capture(**kw):
        captured.update(kw)
        return real(**kw)

    original = pe._native_portfolio_sim
    pe._native_portfolio_sim = capture
    try:
        pe.run_portfolio_simulation(data, weights, **kwargs)
    finally:
        pe._native_portfolio_sim = original
    return captured


def _assert_agrees(native, looped):
    # Rounding error is proportional to the magnitude of the terms that were
    # summed, not to the magnitude of the answer -- so the yardstick is the
    # GROSS exposure, the scale of the position values going into every one of
    # these curves. That matters for net_exposure in particular: it is a
    # signed sum that can cancel to near zero out of position values in the
    # hundreds, and judging a near-zero result by its own size would demand
    # far better than double precision can deliver. Same pure-ratio-against-
    # the-right-scale reasoning as numerics::is_negligible_pivot.
    scale = float(np.nanmax(np.abs(looped["gross_exposure_curve"].to_numpy())))
    scale = max(scale, float(np.nanmax(np.abs(looped["equity_curve"].to_numpy()))), 1.0)
    for key in (
        "equity_curve",
        "cash_curve",
        "gross_exposure_curve",
        "net_exposure_curve",
        "leverage_curve",
    ):
        a = native[key].to_numpy()
        b = looped[key].to_numpy()
        atol = REL_TOL * scale
        if key == "leverage_curve":
            atol = REL_TOL * 10.0  # a ratio, so its own scale is O(1)
        np.testing.assert_allclose(
            a, b, rtol=REL_TOL, atol=atol, equal_nan=True, err_msg=key
        )
        assert native[key].index.equals(looped[key].index), key
    assert native["final_equity"] == pytest.approx(looped["final_equity"], rel=REL_TOL)
    assert native["final_cash"] == pytest.approx(looped["final_cash"], rel=REL_TOL)
    assert native["warnings"] == looped["warnings"]

    la, lb = native["rebalance_log"], looped["rebalance_log"]
    assert list(la.columns) == list(lb.columns)
    assert len(la) == len(lb), "different number of rebalances executed"
    assert list(la["date"]) == list(lb["date"])
    assert list(la["n_positions"]) == list(lb["n_positions"])
    for col in ("turnover_pct", "gross_leverage_after"):
        np.testing.assert_allclose(
            la[col].to_numpy(dtype=float),
            lb[col].to_numpy(dtype=float),
            atol=1e-6,  # both sides round to 6dp before logging
            err_msg=col,
        )


@requires_cpp
class TestNativeMatchesPythonLoop:
    @pytest.mark.parametrize("fill", ["close", "next_open", "hl2_exploratory"])
    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_every_fill_mode(self, fill, seed):
        data, w = _universe(seed=seed)
        _assert_agrees(*_both(data, w, fill_price=fill))

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"commission_pct": 0.0, "slippage_pct": 0.0},
            {"commission_pct": 0.005, "slippage_pct": 0.002},
            {"initial_capital": 1_000_000.0},
            {"max_gross_leverage": 2.0},
            {"borrow_fee_bps": 50.0},
            {"margin_interest_rate": 0.06},
            {"borrow_fee_bps": 50.0, "margin_interest_rate": 0.06},
        ],
    )
    def test_cost_configurations(self, kwargs):
        data, w = _universe(seed=5)
        _assert_agrees(*_both(data, w, **kwargs))

    def test_financing_uses_calendar_days_not_bars(self):
        """A Friday->Monday gap must accrue three days, on both paths."""
        data, w = _universe(n_bars=120, seed=11)
        _assert_agrees(*_both(data, w, borrow_fee_bps=200.0, margin_interest_rate=0.10))

    def test_ragged_indexes(self):
        """Tickers with different histories: the master calendar is the
        intersection, and positional take must respect it."""
        data, w = _universe(n_tickers=9, ragged=True, seed=13)
        _assert_agrees(*_both(data, w))

    def test_long_short_book(self):
        data, w = _universe(n_tickers=8, n_bars=400, seed=21)
        # Force a genuinely two-sided book so the short-borrow path is live.
        w.iloc[:, ::2] = -w.iloc[:, ::2].abs()
        w.iloc[:, 1::2] = w.iloc[:, 1::2].abs()
        w = w.div(w.abs().sum(axis=1), axis=0)
        _assert_agrees(*_both(data, w, borrow_fee_bps=75.0))

    def test_single_ticker_and_single_rebalance(self):
        data, w = _universe(n_tickers=1, n_bars=60, seed=31)
        _assert_agrees(*_both(data, w.iloc[:1]))

    def test_large_universe(self):
        data, w = _universe(n_tickers=60, n_bars=300, seed=41)
        _assert_agrees(*_both(data, w))


@requires_cpp
class TestTheThreeFormerlyUnsupportedConfigurations:
    """per_share commission, the impact model and the ADV cap all run NATIVELY
    now, and must still agree with the loop.

    These three used to fall through to Python by design, on the reasoning
    that each is a per-element decision that would have to be restated in
    C++. Measured, that reasoning cost more than it saved: the kernel bought
    1.3x on the one configuration it covered while these three ran 6-21x
    slower with no acceleration at all. The kernel's rebalance loop was
    already per-ticker with a per-ticker error path, so supporting them was
    arguments rather than a different shape.

    The agreement assertion below is unchanged from when it guarded the
    fallback -- it was always the right test, and now it is testing the
    thing that actually runs.
    """

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"commission_model": "per_share", "per_share_rate": 0.005},
            {"use_impact_model": True},
            {"max_adv_participation": 0.1},
            # All three together, which is the configuration that measured
            # slowest of all before the kernel learned them.
            {
                "commission_model": "per_share",
                "per_share_rate": 0.005,
                "min_commission": 1.0,
                "use_impact_model": True,
                "max_adv_participation": 0.5,
            },
        ],
    )
    def test_native_and_looped_agree(self, kwargs):
        data, w = _universe(seed=7)
        _assert_agrees(*_both(data, w, **kwargs))

    @requires_cpp
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"commission_model": "per_share", "per_share_rate": 0.005},
            {"use_impact_model": True},
            {"max_adv_participation": 0.1},
        ],
    )
    def test_the_native_path_is_actually_taken(self, kwargs, monkeypatch):
        """Otherwise the agreement above would pass vacuously.

        A parity test that compares the Python loop against itself is the
        failure mode this guards: it stays green forever and proves nothing
        about the kernel.
        """
        called = {"n": 0}
        real = pe._native_portfolio_sim

        def counting(**kw):
            called["n"] += 1
            return real(**kw)

        monkeypatch.setattr(pe, "_native_portfolio_sim", counting)
        data, w = _universe(seed=7)
        pe.run_portfolio_simulation(data, w, **kwargs)
        assert called["n"] == 1, "the helper was never consulted"
        assert (
            real(**_helper_kwargs(data, w, **kwargs)) is not None
        ), "the helper declined a configuration the kernel now supports"

    def test_the_helper_still_declines_what_it_cannot_serve(self):
        """Returning None rather than raising is the contract: an
        unsupported option is not an error, it just means the loop runs.

        The impact model needs a volatility panel and the ADV cap needs a
        volume panel. Without them the helper must decline rather than
        dereference a missing array -- a caller who asked for a
        liquidity-aware run cannot have it satisfied by absent data.
        """
        close = np.ones((60, 3))
        base = dict(
            close_mat=close,
            open_mat=close,
            hl2_mat=close,
            weights_mat=np.zeros((1, 3)),
            rebalance_index=pd.RangeIndex(1),
            master_index=pd.RangeIndex(60),
            tickers=["A", "B", "C"],
            fill_price="close",
            commission_model="pct",
            use_impact_model=False,
            max_adv_participation=None,
            initial_capital=1000.0,
            commission_pct=0.0,
            # Required, not defaulted: there is exactly one production
            # call site and a default would let a future one silently
            # get symmetric commission on a spec that asked for two
            # rates.
            sell_commission_rate=0.0,
            slippage_pct=0.0,
            max_gross_leverage=1.0,
            max_position_pct=1.0,
            borrow_fee_bps=0.0,
            margin_interest_rate=0.0,
        )
        assert pe._native_portfolio_sim(**{**base, "fill_price": "nonsense"}) is None
        assert (
            pe._native_portfolio_sim(**{**base, "commission_model": "nonsense"}) is None
        )
        # Asked for the impact model, given no panels.
        assert pe._native_portfolio_sim(**{**base, "use_impact_model": True}) is None
        # Asked for an ADV cap, given no volume panel.
        assert (
            pe._native_portfolio_sim(**{**base, "max_adv_participation": 0.1}) is None
        )


@requires_cpp
class TestNativeErrorsMatchPython:
    """A kernel status code must surface as the message the loop always raised."""

    def test_insolvency_raises_the_same_way(self):
        data, w = _universe(n_tickers=3, n_bars=80, seed=3)
        # Enormous costs drive equity through zero.
        for engine_has_cpp in (True, False):
            saved = pe.HAS_CPP
            pe.HAS_CPP = saved and engine_has_cpp
            try:
                with pytest.raises(ValidationError, match="insolvent"):
                    pe.run_portfolio_simulation(
                        data, w, commission_pct=0.9, slippage_pct=0.9
                    )
            finally:
                pe.HAS_CPP = saved

    def test_leverage_breach_raises_the_same_way(self):
        data, w = _universe(n_tickers=4, n_bars=80, seed=9)
        w = w * 5.0  # far beyond the default max_gross_leverage of 1.0
        for engine_has_cpp in (True, False):
            saved = pe.HAS_CPP
            pe.HAS_CPP = saved and engine_has_cpp
            try:
                with pytest.raises(ValidationError, match="leverage"):
                    pe.run_portfolio_simulation(data, w)
            finally:
                pe.HAS_CPP = saved
