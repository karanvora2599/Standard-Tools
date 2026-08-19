"""
Regression tests for the vectorized portfolio simulator.

run_portfolio_simulation was rewritten to hold prices, weights and liquidity
baselines as dense (n_bars x n_tickers) matrices and to execute the default
cost configuration with array arithmetic instead of a per-ticker Python loop.
That rewrite created two things worth pinning down permanently:

  1. TWO ROUTES THROUGH THE SAME REBALANCE. The vectorized fast path handles
     the default configuration; per-share commission, the impact model and
     the ADV constraint each keep the explicit loop. Two implementations of
     one calculation can drift apart silently, so the tests below drive the
     SAME economics down both routes and require them to agree.

  2. ERROR MESSAGES THAT NAME A SPECIFIC OFFENDER. Screening a whole matrix
     at once finds every violation simultaneously, whereas the loops these
     replaced stopped at the first. Which ticker, column or date gets named
     is observable, so the search ORDER is asserted rather than left to the
     shape of whichever vectorized expression happened to be convenient.
"""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.backtest.portfolio_engine import run_portfolio_simulation
from standard_quant_tools.error import ValidationError

CURVES = (
    "equity_curve",
    "cash_curve",
    "gross_exposure_curve",
    "net_exposure_curve",
)


def _universe(n_tickers=6, n_bars=120, seed=0, volume=1e9):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2020-01-01", periods=n_bars, freq="B")
    data = {}
    for i in range(n_tickers):
        close = 100.0 * np.cumprod(1.0 + rng.normal(0.0004, 0.011, n_bars))
        data[f"T{i}"] = pd.DataFrame(
            {
                "Open": close * 0.998,
                "High": close * 1.006,
                "Low": close * 0.994,
                "Close": close,
                "Volume": np.full(n_bars, volume),
            },
            index=index,
        )
    return data, index


def _weights(index, tickers, every=5, seed=1, long_short=False):
    rng = np.random.default_rng(seed)
    dates = index[::every][:-1]
    if long_short:
        raw = rng.normal(0.0, 1.0, (len(dates), len(tickers)))
        raw /= np.abs(raw).sum(axis=1, keepdims=True)
    else:
        raw = rng.dirichlet(np.ones(len(tickers)), len(dates))
    return pd.DataFrame(raw, index=dates, columns=list(tickers))


def _assert_same_simulation(a, b, rtol=1e-12):
    # Tolerances are anchored to the size of the BOOK, not to each series'
    # own magnitude. A fully invested portfolio holds a cash balance of a few
    # times 1e-13 -- residue from sizing shares against equity -- and a
    # relative tolerance there is comparing two representations of zero.
    # 1e-9 of peak equity is a fraction of a cent on a $100k book: far above
    # summation-reassociation noise, far below any real divergence.
    scale = float(a["equity_curve"].abs().max())
    for key in CURVES:
        np.testing.assert_allclose(
            a[key].to_numpy(),
            b[key].to_numpy(),
            rtol=rtol,
            atol=1e-9 * scale,
        )
        assert a[key].index.equals(b[key].index)
    pd.testing.assert_frame_equal(a["rebalance_log"], b["rebalance_log"])


# ── The two rebalance routes must compute the same thing ────────────────────


@pytest.mark.parametrize("long_short", [False, True])
def test_vectorized_and_loop_paths_agree(long_short):
    """The ADV constraint forces the per-ticker loop; set it wide enough to
    bind on nothing, and the loop must reproduce the fast path exactly.

    This is the load-bearing test of the rewrite. Both routes see identical
    prices, weights and cost rates, so any difference in the result is a
    difference between the two implementations rather than a difference in
    the scenario. They are not bit-identical by construction -- the fast
    path reduces cash with np.sum's pairwise summation while the loop
    accumulates sequentially -- so the tolerance is loose enough to permit
    reassociation and far too tight to permit an actual formula divergence.
    """
    price_data, index = _universe()
    weights = _weights(index, price_data, long_short=long_short)

    fast = run_portfolio_simulation(
        price_data, weights, commission_pct=0.001, slippage_pct=0.0005
    )
    looped = run_portfolio_simulation(
        price_data,
        weights,
        commission_pct=0.001,
        slippage_pct=0.0005,
        # Participation is a fraction of average dollar volume; the universe
        # trades $1e9 a day against a $100k book, so this can never bind.
        max_adv_participation=1e6,
    )

    _assert_same_simulation(fast, looped)


def test_vectorized_and_loop_paths_agree_on_zero_cost():
    """per_share with a zero rate and no minimum is economically identical to
    pct with a zero rate -- different commission MODELS, same commission. The
    fast path declines per_share, so this pairs the two routes again with the
    cost term removed entirely, isolating the share-sizing and cash arithmetic
    from the cost arithmetic."""
    price_data, index = _universe(seed=4)
    weights = _weights(index, price_data, seed=5)

    fast = run_portfolio_simulation(
        price_data, weights, commission_pct=0.0, slippage_pct=0.0
    )
    looped = run_portfolio_simulation(
        price_data,
        weights,
        commission_model="per_share",
        per_share_rate=0.0,
        min_commission=0.0,
        slippage_pct=0.0,
    )

    _assert_same_simulation(fast, looped)


@pytest.mark.parametrize("fill_price", ["close", "next_open", "hl2_exploratory"])
def test_paths_agree_under_every_fill_model(fill_price):
    """Each fill model reads a different price matrix (Close, Open, or the
    High/Low midpoint). The fast path receives whichever row the caller
    selected, so the route and the fill model have to be independent."""
    price_data, index = _universe(seed=6)
    weights = _weights(index, price_data, seed=7)

    fast = run_portfolio_simulation(price_data, weights, fill_price=fill_price)
    looped = run_portfolio_simulation(
        price_data, weights, fill_price=fill_price, max_adv_participation=1e6
    )

    _assert_same_simulation(fast, looped)


def test_fast_path_charges_exactly_the_stated_cost():
    """Guard against the vectorized cost term drifting: on flat prices with a
    single rebalance from a flat book, the total cost is computable by hand.

    100,000 sized into 50% and 30% at a price of 100 buys 500 and 300 shares,
    so turnover is 50,000 + 30,000 = 80,000. Prices never move afterwards, so
    final equity is exactly the starting capital less the cost on that
    turnover -- nothing compounds and nothing else can explain a difference.
    """
    n_bars = 10
    index = pd.date_range("2021-01-04", periods=n_bars, freq="B")
    flat = np.full(n_bars, 100.0)
    price_data = {
        t: pd.DataFrame(
            {
                "Open": flat,
                "High": flat,
                "Low": flat,
                "Close": flat,
                "Volume": np.full(n_bars, 1e9),
            },
            index=index,
        )
        for t in ("A", "B")
    }
    weights = pd.DataFrame([[0.5, 0.3]], index=[index[0]], columns=["A", "B"])

    charged = run_portfolio_simulation(
        price_data,
        weights,
        initial_capital=100_000.0,
        commission_pct=0.002,
        slippage_pct=0.001,
    )

    expected_cost = 80_000.0 * (0.002 + 0.001)
    assert float(charged["equity_curve"].iloc[-1]) == pytest.approx(
        100_000.0 - expected_cost, rel=1e-12
    )


# ── Post-trade invariants still fire ────────────────────────────────────────


def test_position_limit_breach_is_still_detected_after_trading():
    """The realized-position check was a per-ticker generator and is now a
    single max over the position-value vector. Dividing every element by the
    same positive equity is monotonic, so the max must still identify the
    same offending position."""
    price_data, index = _universe(n_tickers=3, n_bars=40, seed=10)
    dates = index[::10][:-1]
    weights = pd.DataFrame(
        np.tile([0.8, 0.1, 0.05], (len(dates), 1)),
        index=dates,
        columns=list(price_data),
    )

    with pytest.raises(ValidationError, match="max_position_pct"):
        run_portfolio_simulation(price_data, weights, max_position_pct=0.5)


def test_empty_portfolio_does_not_trip_the_position_check():
    """All-zero weights leave every position value at zero. The vectorized
    max replaced a `max(..., default=0.0)`, and ndarray.max() raises on an
    empty selection -- so a fully-flat book must still simulate cleanly."""
    price_data, index = _universe(n_tickers=3, n_bars=40, seed=11)
    dates = index[::10][:-1]
    weights = pd.DataFrame(0.0, index=dates, columns=list(price_data))

    result = run_portfolio_simulation(price_data, weights, initial_capital=50_000.0)

    assert np.allclose(result["equity_curve"].to_numpy(), 50_000.0)
    assert np.allclose(result["gross_exposure_curve"].to_numpy(), 0.0)
    assert (result["rebalance_log"]["n_positions"] == 0).all()


# ── Vectorized screening must name the same offender the loops did ──────────


def test_bad_price_names_the_first_ticker_in_universe_order():
    """Price validation now screens each whole column at once, then walks
    ticker-major to build the message. With two bad tickers the EARLIER one
    in universe order must be reported, as the original nested loop did."""
    price_data, index = _universe(n_tickers=4, n_bars=30, seed=12)
    price_data["T1"].iloc[7, price_data["T1"].columns.get_loc("Close")] = -1.0
    price_data["T3"].iloc[9, price_data["T3"].columns.get_loc("Close")] = np.nan
    weights = _weights(index, price_data, every=10, seed=13)

    with pytest.raises(ValidationError) as excinfo:
        run_portfolio_simulation(price_data, weights)

    assert "'T1'" in str(excinfo.value)
    assert "'T3'" not in str(excinfo.value)


def test_bad_price_names_the_first_required_column_for_that_ticker():
    """Within one ticker the column order is Close, then Open (for
    next_open). Screening column-major would report Open here because T0's
    Open is bad and T2's Close is bad; ticker-major reports T2's Close."""
    price_data, index = _universe(n_tickers=4, n_bars=30, seed=14)
    price_data["T0"].iloc[5, price_data["T0"].columns.get_loc("Open")] = 0.0
    price_data["T2"].iloc[6, price_data["T2"].columns.get_loc("Close")] = -2.0
    weights = _weights(index, price_data, every=10, seed=15)

    with pytest.raises(ValidationError) as excinfo:
        run_portfolio_simulation(price_data, weights, fill_price="next_open")

    message = str(excinfo.value)
    assert "'T0'" in message and "'Open'" in message
    assert "'T2'" not in message


def test_weight_validation_reports_the_earliest_offending_date():
    """Gross-leverage and position-size screening now happen across all dates
    in one pass. The message must still name the first invalid date in index
    order, not whichever violation the vectorized scan noticed first."""
    price_data, index = _universe(n_tickers=3, n_bars=60, seed=16)
    dates = index[::10][:-1]
    weights = pd.DataFrame(0.1, index=dates, columns=list(price_data))
    # Gross at date 1 is 1.1, comfortably under the 5.0 ceiling, so ONLY the
    # position rule can fire there. Date 3 breaches both. If the scan
    # reported by rule rather than by date, the gross message from date 3
    # would surface instead.
    weights.iloc[1, 0] = 0.9  # position breach, earlier date
    weights.iloc[3] = [2.0, 2.0, 2.0]  # gross breach, later date

    with pytest.raises(ValidationError) as excinfo:
        run_portfolio_simulation(
            price_data, weights, max_gross_leverage=5.0, max_position_pct=0.5
        )

    message = str(excinfo.value)
    assert str(dates[1]) in message
    assert "max_position_pct" in message


def test_gross_breach_wins_over_position_breach_on_the_same_date():
    """When one date violates both rules, gross leverage is reported --
    the order the original per-date loop checked them in."""
    price_data, index = _universe(n_tickers=3, n_bars=60, seed=17)
    dates = index[::10][:-1]
    weights = pd.DataFrame(0.1, index=dates, columns=list(price_data))
    weights.iloc[2] = [0.9, 0.9, 0.9]

    with pytest.raises(ValidationError, match="gross leverage"):
        run_portfolio_simulation(
            price_data, weights, max_gross_leverage=1.0, max_position_pct=0.5
        )


# ── Financing accrual ───────────────────────────────────────────────────────


def test_short_borrow_fee_matches_a_hand_computed_accrual():
    """The borrow fee was a per-ticker loop and is now one masked sum, on the
    grounds that the fee is linear in notional. This checks the linearity
    claim against an explicit reconstruction rather than assuming it."""
    n_bars = 30
    index = pd.date_range("2021-01-04", periods=n_bars, freq="B")
    flat = np.full(n_bars, 100.0)
    price_data = {
        t: pd.DataFrame(
            {
                "Open": flat,
                "High": flat,
                "Low": flat,
                "Close": flat,
                "Volume": np.full(n_bars, 1e9),
            },
            index=index,
        )
        for t in ("A", "B")
    }
    weights = pd.DataFrame([[-0.5, 0.5]], index=[index[0]], columns=["A", "B"])

    borrowed = run_portfolio_simulation(
        price_data,
        weights,
        initial_capital=100_000.0,
        borrow_fee_bps=100.0,
        commission_pct=0.0,
        slippage_pct=0.0,
    )
    unborrowed = run_portfolio_simulation(
        price_data,
        weights,
        initial_capital=100_000.0,
        commission_pct=0.0,
        slippage_pct=0.0,
    )

    # Prices are flat, so the short book is a constant $50,000 from the bar
    # after the rebalance onward. The fee accrues on calendar days elapsed.
    short_notional = 50_000.0
    # The rebalance executes at bar 0's Close, after that bar's financing
    # step, so the short is carried into bars 1..n_bars-1 and each of those
    # accrues over the calendar days since the previous bar.
    elapsed_days = sum(float((index[i] - index[i - 1]).days) for i in range(1, n_bars))
    expected = short_notional * (100.0 / 10_000.0) * (elapsed_days / 365.0)

    actual = float(
        unborrowed["equity_curve"].iloc[-1] - borrowed["equity_curve"].iloc[-1]
    )
    assert actual == pytest.approx(expected, rel=1e-9)


def test_long_only_book_accrues_no_borrow_fee():
    """The masked sum must select on shares being negative. A long-only book
    has to be untouched by a nonzero borrow rate."""
    price_data, index = _universe(n_tickers=4, n_bars=60, seed=18)
    weights = _weights(index, price_data, every=10, seed=19)

    charged = run_portfolio_simulation(price_data, weights, borrow_fee_bps=250.0)
    free = run_portfolio_simulation(price_data, weights, borrow_fee_bps=0.0)

    _assert_same_simulation(charged, free)


# ── Liquidity baselines are read at the trigger bar, not the execution bar ──


def test_liquidity_baseline_still_reads_the_trigger_bar_under_next_open():
    """dollar_volume/volatility moved from pandas Series to matrices indexed
    by (bar, ticker). The lookahead-free contract of next_open depends on
    that index being the TRIGGER bar, so a volume collapse on the execution
    bar alone must not change the simulation."""
    n_bars = 40
    index = pd.date_range("2021-01-04", periods=n_bars, freq="B")
    close = 100.0 + np.arange(n_bars, dtype=float)
    volume = np.full(n_bars, 1e7)

    def build(vol):
        return {
            t: pd.DataFrame(
                {
                    "Open": close * 0.999,
                    "High": close * 1.002,
                    "Low": close * 0.998,
                    "Close": close,
                    "Volume": vol,
                },
                index=index,
            )
            for t in ("A", "B")
        }

    trigger_bar = 10
    weights = pd.DataFrame([[0.5, 0.5]], index=[index[trigger_bar]], columns=["A", "B"])

    collapsed = volume.copy()
    collapsed[trigger_bar + 1] = 1.0  # execution bar only

    baseline = run_portfolio_simulation(
        build(volume),
        weights,
        fill_price="next_open",
        use_impact_model=True,
        impact_lookback=5,
    )
    perturbed = run_portfolio_simulation(
        build(collapsed),
        weights,
        fill_price="next_open",
        use_impact_model=True,
        impact_lookback=5,
    )

    _assert_same_simulation(baseline, perturbed)


# ── Cost-rate validation happens once, at entry, for every rate ─────────────


@pytest.mark.parametrize(
    "param",
    [
        "commission_pct",
        "slippage_pct",
        "per_share_rate",
        "min_commission",
        "borrow_fee_bps",
        "margin_interest_rate",
        "impact_coefficient",
    ],
)
@pytest.mark.parametrize("bad_value", [True, "0.001", float("nan"), -1.0])
def test_every_cost_rate_is_validated_at_entry(param, bad_value):
    """Cost rates are function parameters and cannot change between
    rebalances, so they are validated once at entry rather than re-checked
    inside the cost functions on every trade.

    Doing it at entry made the check reach further than it used to. The
    hand-rolled `math.isfinite(value) or value < 0` guard it replaces
    accepted `True` -- `isfinite(True)` is True and `True < 0` is False, so a
    boolean silently became a 100% rate -- and raised a bare TypeError rather
    than a ValidationError on a string. And because validation used to happen
    inside whichever cost function ran, a bad rate belonging to the OTHER
    commission model was never examined at all: `per_share_rate=True` ran a
    complete simulation to a plausible-looking equity under the pct model.
    """
    price_data, index = _universe(n_tickers=3, n_bars=30, seed=20)
    weights = _weights(index, price_data, every=10, seed=21)

    with pytest.raises(ValidationError, match=param):
        run_portfolio_simulation(price_data, weights, **{param: bad_value})


def test_valid_rates_for_the_unused_commission_model_still_pass():
    """Validation reaching every rate must not mean rejecting rates the
    active model does not read -- only invalid ones."""
    price_data, index = _universe(n_tickers=3, n_bars=30, seed=22)
    weights = _weights(index, price_data, every=10, seed=23)

    result = run_portfolio_simulation(
        price_data,
        weights,
        commission_model="pct",
        per_share_rate=0.005,
        min_commission=1.0,
    )

    assert float(result["final_equity"]) > 0.0
