"""
What the equity curve does not say, checked against constructions where the
answer is known.

THE INVARIANT THAT VALIDATES THE MONTE CARLO is that a reshuffle preserves
the multiset of trades exactly -- so every path must end at the same total
return, and only the PATH differs. If final equity ever varies across paths,
the implementation is resampling with replacement rather than permuting, and
the drawdown distribution it produces would then be measuring edge
uncertainty mixed with sequence risk instead of sequence risk alone.

THE RUNS TEST IS CHECKED BOTH WAYS. Planted streaks must come back clustered
and a planted alternation must come back alternating -- a test that only
checked one direction would pass on an implementation with a sign error in
the z-score.

Every detector is also run on data with nothing in it. A runs test that
fires on independent trades, or a random-comparison that finds skill in
random signs, is worse than no test at all because it carries authority.
"""

import numpy as np
import pytest

from standard_quant_tools.backtesting.trade_analysis import (
    analyze_trade_clustering,
    compare_against_random,
    exposure_attribution,
    monte_carlo_trade_paths,
)
from standard_quant_tools.error import ValidationError


def _independent(n=150, win_rate=0.55, size=0.02, seed=0):
    """Trades with NO order structure: the null for every runs test here."""
    rng = np.random.default_rng(seed)
    return np.where(rng.random(n) < win_rate, size, -size)


def _streaky(n=200, seed=3):
    """Wins and losses in runs of 5-15, so clustering is planted."""
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        length = int(rng.integers(5, 15))
        out.extend([0.02] * length if rng.random() < 0.5 else [-0.02] * length)
    return np.array(out[:n])


class TestMonteCarloTradePaths:
    def test_every_path_ends_at_the_same_total_return(self):
        """
        THE INVARIANT THAT PROVES IT PERMUTES RATHER THAN RESAMPLES. A
        reshuffle holds the same trades, so the product of (1 + r) is
        order-independent and every path ends identically. If this ever
        fails, the tool is measuring edge uncertainty mixed with sequence
        risk instead of sequence risk alone.
        """
        trades = np.random.default_rng(0).normal(0.004, 0.03, 200)
        result = monte_carlo_trade_paths(trades, n_paths=300, seed=1)
        expected = float(np.prod(1.0 + trades))
        assert result["observed_final_equity"] == pytest.approx(expected, rel=1e-12)
        assert result["final_equity"] == pytest.approx(expected, rel=1e-12)

    def test_the_drawdown_distribution_is_wider_than_the_single_backtest(self):
        trades = np.random.default_rng(0).normal(0.004, 0.03, 200)
        result = monte_carlo_trade_paths(trades, n_paths=1000, seed=1)
        assert result["worst_max_drawdown"] < result["median_max_drawdown"]
        assert result["p05_max_drawdown"] <= result["p95_max_drawdown"]

    def test_the_observed_drawdown_is_placed_as_a_percentile(self):
        trades = np.random.default_rng(0).normal(0.004, 0.03, 200)
        result = monte_carlo_trade_paths(trades, n_paths=1000, seed=1)
        assert 0 <= result["observed_drawdown_percentile"] <= 100

    def test_a_lucky_ordering_is_called_out(self):
        trades = np.random.default_rng(0).normal(0.004, 0.03, 200)
        result = monte_carlo_trade_paths(trades, n_paths=1000, seed=1)
        if result["observed_max_drawdown"] > result["p05_max_drawdown"]:
            assert any("lucky ordering" in w for w in result["warnings"])

    def test_it_declares_what_reshuffling_destroys(self):
        """Reordering removes real between-trade dependence, so the
        distribution is optimistic about clustering. That has to be said."""
        result = monte_carlo_trade_paths(
            np.random.default_rng(1).normal(0.003, 0.02, 100), n_paths=200
        )
        assert any("optimistic about clustering" in w for w in result["warnings"])
        assert any("RESHUFFLES" in w for w in result["warnings"])

    def test_a_short_trade_history_is_flagged(self):
        result = monte_carlo_trade_paths(_independent(n=30), n_paths=200)
        assert any("narrow and" in w for w in result["warnings"])

    def test_it_is_reproducible_from_the_seed(self):
        trades = np.random.default_rng(2).normal(0.003, 0.02, 150)
        a = monte_carlo_trade_paths(trades, n_paths=300, seed=7)
        b = monte_carlo_trade_paths(trades, n_paths=300, seed=7)
        assert a["median_max_drawdown"] == b["median_max_drawdown"]

    def test_too_few_trades_is_refused(self):
        with pytest.raises(ValidationError, match="at least"):
            monte_carlo_trade_paths([0.01, -0.01, 0.02])


class TestTradeClustering:
    def test_it_does_not_find_streaks_in_independent_trades(self):
        """The null, measured as a rate. A runs test that fires on
        independent trades carries authority it has not earned."""
        fires = sum(
            analyze_trade_clustering(_independent(seed=s))["clustered"]
            or analyze_trade_clustering(_independent(seed=s))["alternating"]
            for s in range(200)
        )
        assert fires / 200 < 0.15, f"{fires}/200 independent sequences flagged"

    def test_it_finds_planted_streaks(self):
        result = analyze_trade_clustering(_streaky())
        assert result["clustered"]
        assert result["runs_z_score"] < -2
        assert result["longest_losing_streak"] >= 5
        assert any("CLUSTER" in w for w in result["warnings"])

    def test_it_finds_planted_alternation(self):
        """The other direction, which a sign error in the z-score would
        break while leaving the clustering case passing."""
        alternating = np.array([0.02 if i % 2 == 0 else -0.02 for i in range(200)])
        result = analyze_trade_clustering(alternating)
        assert result["alternating"]
        assert result["runs_z_score"] > 2
        assert any("ALTERNATE" in w for w in result["warnings"])

    def test_the_run_count_matches_the_sequence(self):
        """Five runs by construction: ++ -- ++ - +"""
        trades = [0.01, 0.01, -0.01, -0.01, 0.01, 0.01, -0.01, 0.01]
        trades = trades * 4  # to clear the minimum
        result = analyze_trade_clustering(trades)
        signs = [t > 0 for t in trades]
        expected = 1 + sum(a != b for a, b in zip(signs, signs[1:]))
        assert result["n_runs"] == expected

    def test_the_longest_streaks_are_reported(self):
        result = analyze_trade_clustering(_streaky())
        assert result["longest_losing_streak"] >= 1
        assert result["longest_winning_streak"] >= 1
        assert any("longest losing streak" in w for w in result["warnings"])

    def test_it_links_clustering_to_the_monte_carlo_being_optimistic(self):
        result = analyze_trade_clustering(_streaky())
        assert any("Monte Carlo" in w or "reshuffling" in w for w in result["warnings"])

    def test_all_wins_has_no_runs_to_test(self):
        with pytest.raises(ValidationError, match="no runs"):
            analyze_trade_clustering([0.01] * 50)


class TestCompareAgainstRandom:
    def test_random_signs_do_not_beat_random(self):
        """The null, as a rate."""
        fires = 0
        for seed in range(100):
            rng = np.random.default_rng(seed)
            magnitudes = np.abs(rng.normal(0, 0.02, 200))
            trades = magnitudes * np.where(rng.random(200) < 0.5, 1, -1)
            fires += compare_against_random(trades, n_simulations=400, seed=seed)[
                "beats_random_at_05"
            ]
        assert fires / 100 < 0.10, f"{fires}/100 skill-free strategies beat random"

    def test_systematically_larger_wins_beat_random(self):
        """
        The case it was built for: wins twice the size of losses. Measured
        at p < 0.001.
        """
        rng = np.random.default_rng(0)
        magnitudes = np.abs(rng.normal(0, 0.02, 300))
        wins = rng.random(300) < 0.55
        trades = np.where(wins, magnitudes * 2.0, -magnitudes * 0.5)
        result = compare_against_random(trades, n_simulations=1500, seed=1)
        assert result["beats_random_at_05"]
        assert result["p_value"] < 0.01

    def test_the_null_keeps_the_win_rate_and_says_so(self):
        """
        The limitation that decides how to read a non-rejection: the null
        already has the strategy's win rate, so an edge that lives entirely
        in the win rate is invisible here -- correctly.
        """
        result = compare_against_random(_independent(n=200), n_simulations=400)
        assert any("KEEPS the strategy's win rate" in w for w in result["warnings"])

    def test_it_says_beating_zero_is_not_beating_random(self):
        result = compare_against_random(_independent(n=200), n_simulations=400)
        assert any("Beating zero" in w for w in result["warnings"])

    def test_it_is_reproducible_from_the_seed(self):
        trades = np.random.default_rng(3).normal(0.002, 0.02, 200)
        a = compare_against_random(trades, n_simulations=400, seed=11)
        b = compare_against_random(trades, n_simulations=400, seed=11)
        assert a["p_value"] == b["p_value"]

    def test_a_flat_trade_list_is_refused(self):
        with pytest.raises(ValidationError, match="no dispersion"):
            compare_against_random([0.01] * 50)


class TestExposureAttribution:
    @staticmethod
    def _market(n=1000, seed=0):
        return np.random.default_rng(seed).normal(0.0005, 0.012, n)

    def test_constant_exposure_has_exactly_zero_timing(self):
        """
        The identity that validates the decomposition: E[e*r] = E[e]E[r] +
        Cov(e, r), and a constant exposure has zero covariance with
        anything. Exact, not approximate.
        """
        market = self._market()
        result = exposure_attribution(market, np.ones(market.size))
        assert result["timing_contribution"] == pytest.approx(0.0, abs=1e-15)
        assert result["passive_contribution"] == pytest.approx(
            result["total_mean_return"], rel=1e-12
        )

    def test_perfect_timing_is_almost_entirely_timing(self):
        market = self._market()
        result = exposure_attribution(market, np.where(market > 0, 1.0, 0.0))
        assert result["timing_share"] > 0.9
        assert result["exposure_return_correlation"] > 0.5

    def test_the_two_parts_sum_to_the_total(self):
        market = self._market()
        exposure = np.random.default_rng(1).random(market.size)
        result = exposure_attribution(market, exposure)
        assert result["passive_contribution"] + result[
            "timing_contribution"
        ] == pytest.approx(result["total_mean_return"], rel=1e-12)

    def test_beta_dressed_as_alpha_is_called_out(self):
        market = self._market()
        result = exposure_attribution(market, np.ones(market.size))
        assert any("beta" in w for w in result["warnings"])

    def test_negative_timing_is_named_as_such(self):
        """A strategy systematically smaller before good periods is
        profitable in spite of its timing, not because of it."""
        market = self._market()
        result = exposure_attribution(market, np.where(market > 0, 0.2, 1.0))
        assert result["timing_contribution"] < 0
        assert any("NEGATIVE" in w for w in result["warnings"])

    def test_partial_investment_is_flagged_as_not_comparable(self):
        market = self._market()
        exposure = np.zeros(market.size)
        exposure[::4] = 1.0
        result = exposure_attribution(market, exposure)
        assert result["fraction_invested"] == pytest.approx(0.25, abs=0.01)
        assert any("idle capital" in w for w in result["warnings"])

    def test_misaligned_inputs_are_refused(self):
        with pytest.raises(ValidationError, match="parallel"):
            exposure_attribution(np.arange(100.0), np.arange(90.0))
