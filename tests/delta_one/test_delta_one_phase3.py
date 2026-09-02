"""
The two Phase III tools: a hedged book carried through time, and a scan.

Both are COMPOSITION. Neither computes an economic quantity of its own --
sizing is `delta_one.hedging.futures_hedge`, the futures account is
`backtest.futures_engine.run_futures_simulation`, the basis statistics are
`delta_one.basis`. What they add is the loop and the ordering, which is
exactly the deterministic work a model should not be asked to do by hand.

So these tests check the things composition can get wrong: signs, the
identity of the quantity being passed between two functions, and whether
the ranking is on the axis the tool claims.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.backtest.futures_hedge_backtest import (
    run_futures_hedge_backtest,
)
from standard_quant_tools.delta_one.scan import basis_scan
from standard_quant_tools.error import ValidationError


def _book_and_future(n=252, beta=1.12, idio=0.003, seed=0):
    """A book that is GENUINELY beta to the future.

    Built from ONE market draw used by both legs. My first version drew the
    book's market factor from a second `rng.normal` call, which makes the
    two independent -- the hedge then cannot work by construction, and the
    test reported a -6.4% volatility reduction that looked like a bug in the
    code rather than in the fixture.
    """
    rng = np.random.default_rng(seed)
    market = rng.normal(0.0003, 0.011, n)
    dates = [str(d.date()) for d in pd.bdate_range("2024-01-02", periods=n)]
    future = 6200.0 * np.cumprod(1 + market)
    book = 250e6 * np.cumprod(1 + beta * market + rng.normal(0, idio, n))
    return dates, book, future


class TestTheHedgeActuallyRemovesTheExposure:
    def test_residual_beta_collapses(self):
        """`beta_before` 1.1124 -> `beta_after` near zero. The number that
        says whether the hedge worked at all."""
        dates, book, future = _book_and_future()
        got = run_futures_hedge_backtest(
            portfolio_values=dict(zip(dates, book)),
            future_prices=dict(zip(dates, future)),
            multiplier=50.0,
            portfolio_beta=1.12,
            rehedge="daily",
        )
        assert got["hedge_effectiveness"]["beta_before"] == pytest.approx(
            1.11, abs=0.05
        )
        assert abs(got["residual_beta"]) < 0.05

    def test_the_hedge_ratio_is_negative_for_a_short_hedge(self):
        """`hedge_effectiveness` computes `portfolio + ratio * hedge`, so a
        short hedge must carry a NEGATIVE ratio -- it warns when one does
        not. Passing the positive figure made `beta_after` come back 2.2324
        against a `beta_before` of 1.1124: the hedge was ADDED to the book,
        exactly doubling the exposure it was meant to cancel."""
        dates, book, future = _book_and_future()
        got = run_futures_hedge_backtest(
            portfolio_values=dict(zip(dates, book)),
            future_prices=dict(zip(dates, future)),
            multiplier=50.0,
            portfolio_beta=1.12,
        )
        assert got["effective_hedge_ratio"] < 0
        assert got["effective_hedge_ratio"] == pytest.approx(-1.12, abs=0.05)

    def test_volatility_falls_to_the_idiosyncratic_component(self):
        """The book is 1.12 * market + 0.3% idio. Hedged, only the idio
        should be left -- so the residual volatility is the thing that was
        never hedgeable, not an arbitrary reduction."""
        dates, book, future = _book_and_future(idio=0.003)
        got = run_futures_hedge_backtest(
            portfolio_values=dict(zip(dates, book)),
            future_prices=dict(zip(dates, future)),
            multiplier=50.0,
            portfolio_beta=1.12,
            rehedge="daily",
        )
        assert got["hedged_volatility"] == pytest.approx(0.003, abs=0.0015)
        assert got["volatility_reduction"] > 0.6

    @pytest.mark.parametrize(
        "rule,expected_rehedges",
        [("daily", 252), ("weekly", 51), ("monthly", 12), ("drift", 1)],
    )
    def test_each_schedule_rehedges_when_it_says_it_does(self, rule, expected_rehedges):
        dates, book, future = _book_and_future()
        got = run_futures_hedge_backtest(
            portfolio_values=dict(zip(dates, book)),
            future_prices=dict(zip(dates, future)),
            multiplier=50.0,
            portfolio_beta=1.12,
            rehedge=rule,
        )
        assert got["n_rehedges"] == expected_rehedges

    def test_a_staler_hedge_leaves_more_residual(self):
        """The whole reason the schedule is a parameter. Monotone, because
        a hedge that is not re-sized drifts as the book's value moves."""
        dates, book, future = _book_and_future()
        residuals = {}
        for rule in ("daily", "monthly", "drift"):
            got = run_futures_hedge_backtest(
                portfolio_values=dict(zip(dates, book)),
                future_prices=dict(zip(dates, future)),
                multiplier=50.0,
                portfolio_beta=1.12,
                rehedge=rule,
            )
            residuals[rule] = abs(got["residual_beta"])
        assert residuals["daily"] <= residuals["monthly"] <= residuals["drift"]

    def test_the_two_legs_are_reported_separately(self):
        """A net number cannot distinguish a hedge that worked from a cash
        leg that happened to fall less."""
        dates, book, future = _book_and_future()
        got = run_futures_hedge_backtest(
            portfolio_values=dict(zip(dates, book)),
            future_prices=dict(zip(dates, future)),
            multiplier=50.0,
            portfolio_beta=1.12,
        )
        assert got["cash_pnl"] + got["hedge_pnl"] == pytest.approx(
            got["combined_pnl"], rel=1e-9
        )
        assert got["cash_pnl"] != got["combined_pnl"]

    def test_margin_never_binds(self):
        """Collateral is sized to the position on purpose: an arbitrary
        figure lets the maintenance logic liquidate the hedge mid-run, and
        the thing measured becomes a margin call rather than a hedge."""
        dates, book, future = _book_and_future()
        got = run_futures_hedge_backtest(
            portfolio_values=dict(zip(dates, book)),
            future_prices=dict(zip(dates, future)),
            multiplier=50.0,
            portfolio_beta=1.12,
            initial_margin=15_000.0,
        )
        assert got["hedge_margin_calls"] == 0
        assert got["peak_hedge_notional"] > 0

    def test_a_bad_schedule_is_refused(self):
        dates, book, future = _book_and_future(n=30)
        with pytest.raises(ValidationError, match="rehedge"):
            run_futures_hedge_backtest(
                portfolio_values=dict(zip(dates, book)),
                future_prices=dict(zip(dates, future)),
                multiplier=50.0,
                rehedge="whenever",
            )

    def test_a_mismatched_beta_path_is_refused(self):
        dates, book, future = _book_and_future(n=30)
        with pytest.raises(ValidationError, match="portfolio_beta"):
            run_futures_hedge_backtest(
                portfolio_values=dict(zip(dates, book)),
                future_prices=dict(zip(dates, future)),
                multiplier=50.0,
                portfolio_beta=[1.0, 1.1],
            )

    def test_a_beta_path_is_accepted(self):
        """A rolling or factor-model beta is supplied, never estimated -- the
        lookback is the most consequential choice in the simulation."""
        dates, book, future = _book_and_future(n=60)
        path = np.linspace(1.0, 1.3, 60).tolist()
        got = run_futures_hedge_backtest(
            portfolio_values=dict(zip(dates, book)),
            future_prices=dict(zip(dates, future)),
            multiplier=50.0,
            portfolio_beta=path,
            rehedge="daily",
        )
        assert got["n_bars"] == 60


class TestTheScanRanksOnTheAxisItClaims:
    @staticmethod
    def _pairs(seed=3, n=300):
        rng = np.random.default_rng(seed)
        out = []
        # WIDE_BUT_NORMAL must end up with the LARGEST current basis while
        # SPIKED has the largest z-score -- otherwise the two axes coincide
        # and the test passes without distinguishing them. My first values
        # gave SPIKED 9 + 38 = 47 bps against WIDE's 40, so the widest and
        # the top-ranked were the same pair and the assertion was vacuous.
        for label, mean_bp, spike in (
            ("ES", 12.0, 0.0),
            ("WIDE_BUT_NORMAL", 90.0, 0.0),
            ("SPIKED", 9.0, 38.0),
        ):
            spot = 6000 * np.cumprod(1 + rng.normal(0, 0.01, n))
            basis = mean_bp + rng.normal(0, 4, n)
            basis[-1] += spike
            out.append(
                {
                    "label": label,
                    "spot": spot.tolist(),
                    "futures": (spot * (1 + basis / 1e4)).tolist(),
                }
            )
        return out

    def test_the_widest_basis_is_not_the_top_rank(self):
        """The design point. A name that always trades 40 bps wide is not
        news at 40 bps; the one sitting away from its OWN history is."""
        got = basis_scan(self._pairs(), detect_shifts=False)
        ranked = got["ranked"]
        widest = max(ranked, key=lambda r: abs(r["current_basis_bps"]))
        assert widest["label"] == "WIDE_BUT_NORMAL"
        assert ranked[0]["label"] == "SPIKED"
        assert ranked[0]["label"] != widest["label"]

    def test_it_is_ordered_by_absolute_zscore(self):
        got = basis_scan(self._pairs(), detect_shifts=False)
        scores = [abs(r["zscore"]) for r in got["ranked"] if r["zscore"] is not None]
        assert scores == sorted(scores, reverse=True)

    def test_a_pair_it_cannot_evaluate_is_reported_not_dropped(self):
        """A misaligned series is a data problem, and a scan that quietly
        returned fewer rows than it was given would hide it."""
        pairs = self._pairs() + [
            {"label": "ragged", "spot": [1.0] * 50, "futures": [1.0] * 40},
            {"label": "tiny", "spot": [1.0] * 5, "futures": [1.0] * 5},
        ]
        got = basis_scan(pairs, detect_shifts=False)
        assert got["n_pairs"] == 5
        assert got["n_evaluated"] + got["n_skipped"] == 5
        labels = {s["label"] for s in got["skipped"]}
        assert labels == {"ragged", "tiny"}
        for entry in got["skipped"]:
            assert entry["reason"], "every skip must say why"

    def test_a_structural_shift_is_a_separate_finding_from_a_level(self):
        """A basis can sit 2 sigma wide for months (a level) or have moved
        there last week (an event). Without the detector they rank the
        same."""
        rng = np.random.default_rng(11)
        n = 300
        spot = 6000 * np.cumprod(1 + rng.normal(0, 0.01, n))
        basis = 10.0 + rng.normal(0, 3, n)
        basis[200:] += 45.0
        got = basis_scan(
            [
                {
                    "label": "shifted",
                    "spot": spot.tolist(),
                    "futures": (spot * (1 + basis / 1e4)).tolist(),
                }
            ]
        )
        assert got["ranked"][0]["shift_detected"] is True
        assert got["ranked"][0]["shift_at"] is not None

    def test_an_empty_set_is_refused(self):
        with pytest.raises(ValidationError, match="no pairs"):
            basis_scan([])

    def test_top_n_limits_the_rows_but_not_the_counts(self):
        got = basis_scan(self._pairs(), detect_shifts=False, top_n=1)
        assert len(got["ranked"]) == 1
        assert got["n_evaluated"] == 3, "the count is of what was evaluated"
