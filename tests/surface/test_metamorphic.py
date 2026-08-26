"""
Relations that must hold between outputs when the input is transformed.

THE GAP THIS FILLS. The correctness tests check one input against one known
answer. The fuzzer checks that nothing crashes. Neither catches a function
that is *consistently* wrong — one that returns a plausible number for every
input, including the ones where a known relation says it must return the
same number as some other input.

Metamorphic testing asks a different question: not "is this answer right"
but "do these two answers stand in the relation they must". It needs no
known answer at all, which is why it reaches code where the right answer is
hard to compute independently.

THE RELATIONS USED HERE:

- **Scale invariance.** Multiplying every return by a positive constant
  leaves the Sharpe ratio unchanged; scaling a covariance matrix leaves
  every risk-parity weight unchanged. A bug that adds an absolute
  threshold anywhere breaks this and nothing else would notice.
- **Order invariance.** A statistic over an unordered bag — total return,
  win rate, the moments — cannot depend on the order it was given. A
  statistic that DOES depend on order (drawdown, runs) must change, and
  that direction is tested too.
- **Permutation equivariance.** Relabelling the assets permutes the weights
  and changes nothing else. A bug that indexes by position rather than by
  name breaks this silently.
- **Translation.** Adding a constant to a price series moves the mean by
  that constant and leaves the standard deviation alone.

WHERE A RELATION HOLDS ONLY APPROXIMATELY, the tolerance is stated and is
about floating point rather than about the method.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.analysis.inference import (
    bootstrap_statistic,
    decompose_returns,
)
from standard_quant_tools.backtesting.trade_analysis import (
    analyze_trade_clustering,
    monte_carlo_trade_paths,
)
from standard_quant_tools.portfolio.construction import (
    concentration_analysis,
    marginal_risk_contribution,
    max_diversification,
    risk_parity,
)

RNG = np.random.default_rng(20260826)
VOLS = np.array([0.10, 0.20, 0.40, 0.15])
CORR = np.array(
    [
        [1.0, 0.3, 0.2, 0.5],
        [0.3, 1.0, 0.4, 0.3],
        [0.2, 0.4, 1.0, 0.1],
        [0.5, 0.3, 0.1, 1.0],
    ]
)
NAMES = list("ABCD")


def _cov(names=NAMES):
    return pd.DataFrame(np.outer(VOLS, VOLS) * CORR, index=names, columns=names)


def _returns(n=500, seed=1):
    return [float(x) for x in np.random.default_rng(seed).normal(0.0006, 0.012, n)]


class TestScaleInvariance:
    """
    A relation that holds for every allocator here: the weights depend on
    the covariance matrix's SHAPE, not its scale. Doubling every variance
    doubles the portfolio's volatility and moves no weight.
    """

    @pytest.mark.parametrize("factor", [0.01, 4.0, 1e6])
    def test_risk_parity_weights_are_invariant_to_covariance_scale(self, factor):
        base = risk_parity(_cov())["weights"]
        scaled = risk_parity(_cov() * factor)["weights"]
        for name in NAMES:
            assert scaled[name] == pytest.approx(base[name], rel=1e-8), (
                f"scaling the covariance by {factor} moved a weight; an "
                "absolute threshold has crept into the solver"
            )

    @pytest.mark.parametrize("factor", [0.01, 4.0, 1e6])
    def test_max_diversification_weights_are_invariant_to_scale(self, factor):
        base = max_diversification(_cov())["weights"]
        scaled = max_diversification(_cov() * factor)["weights"]
        for name in NAMES:
            assert scaled[name] == pytest.approx(base[name], rel=1e-6)

    @pytest.mark.parametrize("factor", [0.01, 4.0, 100.0])
    def test_the_diversification_ratio_is_invariant_to_scale(self, factor):
        """The ratio is volatility over volatility, so the units cancel."""
        base = max_diversification(_cov())["diversification_ratio"]
        scaled = max_diversification(_cov() * factor)["diversification_ratio"]
        assert scaled == pytest.approx(base, rel=1e-6)

    @pytest.mark.parametrize("factor", [0.5, 3.0, 250.0])
    def test_the_sharpe_ratio_is_invariant_to_return_scale(self, factor):
        """
        Sharpe(c·r) = Sharpe(r) for any positive c, because the mean and the
        standard deviation scale together. Checked through the bootstrap so
        the whole path is covered, not just the statistic.
        """
        returns = _returns()
        base = bootstrap_statistic(returns, statistic="sharpe", n_bootstrap=200, seed=3)
        scaled = bootstrap_statistic(
            [r * factor for r in returns], statistic="sharpe", n_bootstrap=200, seed=3
        )
        assert scaled["point_estimate"] == pytest.approx(
            base["point_estimate"], rel=1e-9
        )

    @pytest.mark.parametrize("factor", [0.25, 7.0])
    def test_concentration_is_invariant_to_weight_scale(self, factor):
        """
        Effective N is a property of the SHARES. Scaling every weight scales
        the gross exposure and changes no share, so a leveraged book has the
        same concentration as an unleveraged one holding the same mix.
        """
        weights = {f"s{i}": w for i, w in enumerate([0.4, 0.3, 0.2, 0.1])}
        base = concentration_analysis(weights)
        scaled = concentration_analysis({k: v * factor for k, v in weights.items()})
        assert scaled["effective_n"] == pytest.approx(base["effective_n"], rel=1e-9)
        assert scaled["herfindahl"] == pytest.approx(base["herfindahl"], rel=1e-9)
        assert scaled["gross_exposure"] == pytest.approx(
            base["gross_exposure"] * factor, rel=1e-9
        )


class TestOrderInvariance:
    """
    A statistic over an unordered bag cannot depend on the order it was
    handed. Both directions matter: order-INDEPENDENT statistics must not
    move, and order-DEPENDENT ones must.
    """

    def test_total_return_does_not_depend_on_the_order_of_returns(self):
        """Multiplication is commutative, so the compounded total is the
        same however the returns are shuffled."""
        returns = _returns(300, seed=4)
        shuffled = list(np.random.default_rng(9).permutation(returns))
        assert decompose_returns(shuffled)["total_return"] == pytest.approx(
            decompose_returns(returns)["total_return"], rel=1e-9
        )

    def test_the_moments_do_not_depend_on_order(self):
        returns = _returns(300, seed=5)
        shuffled = list(np.random.default_rng(10).permutation(returns))
        base = decompose_returns(returns)
        moved = decompose_returns(shuffled)
        for field in ("arithmetic_mean", "geometric_mean", "win_rate", "variance"):
            assert moved[field] == pytest.approx(base[field], rel=1e-9), field

    def test_the_runs_test_DOES_depend_on_order(self):
        """
        The other direction, and it is the one that would catch a runs test
        that silently sorted its input. A statistic about ordering that is
        invariant to ordering is measuring nothing.
        """
        streaky = np.array([0.02] * 40 + [-0.02] * 40 + [0.02] * 40)
        alternating = np.array([0.02 if i % 2 == 0 else -0.02 for i in range(120)])
        assert (
            analyze_trade_clustering(streaky)["n_runs"]
            < analyze_trade_clustering(alternating)["n_runs"]
        )

    def test_the_drawdown_DOES_depend_on_order(self):
        """
        Same principle. Losses front-loaded and losses spread out give the
        same total return and very different drawdowns — which is the entire
        premise of run_monte_carlo_trade_paths.
        """
        trades = np.array([0.03] * 30 + [-0.04] * 30)
        clustered = monte_carlo_trade_paths(trades, n_paths=200, seed=1)
        assert clustered["worst_max_drawdown"] < clustered["median_max_drawdown"], (
            "reshuffling produced no dispersion in drawdown, which cannot be "
            "true for a series with both gains and losses"
        )

    def test_every_reshuffled_path_ends_at_the_same_total(self):
        """The invariant that separates a permutation from a resample."""
        trades = np.array(_returns(120, seed=6))
        result = monte_carlo_trade_paths(trades, n_paths=300, seed=2)
        assert result["final_equity"] == pytest.approx(
            float(np.prod(1.0 + trades)), rel=1e-12
        )


class TestPermutationEquivariance:
    """
    Relabelling the assets permutes the answer and changes nothing else. A
    bug that indexes by POSITION where it should index by NAME breaks this
    and produces a plausible number every time.
    """

    def test_risk_parity_follows_a_relabelling(self):
        order = [2, 0, 3, 1]
        names = [NAMES[i] for i in order]
        matrix = _cov().to_numpy()[np.ix_(order, order)]
        permuted = pd.DataFrame(matrix, index=names, columns=names)

        base = risk_parity(_cov())["weights"]
        moved = risk_parity(permuted)["weights"]
        for name in NAMES:
            assert moved[name] == pytest.approx(base[name], rel=1e-8), (
                f"weight for {name} changed when the assets were reordered; "
                "something is indexing by position rather than by name"
            )

    def test_marginal_risk_follows_a_relabelling(self):
        weights = {"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.1}
        order = [3, 1, 2, 0]
        names = [NAMES[i] for i in order]
        matrix = _cov().to_numpy()[np.ix_(order, order)]
        permuted = pd.DataFrame(matrix, index=names, columns=names)

        base = {
            row["asset"]: row["risk_share"]
            for row in marginal_risk_contribution(weights, _cov())["by_asset"]
        }
        moved = {
            row["asset"]: row["risk_share"]
            for row in marginal_risk_contribution(weights, permuted)["by_asset"]
        }
        for name in NAMES:
            assert moved[name] == pytest.approx(base[name], rel=1e-9), name

    def test_concentration_is_indifferent_to_asset_names(self):
        base = concentration_analysis({"a": 0.5, "b": 0.3, "c": 0.2})
        renamed = concentration_analysis({"z": 0.2, "y": 0.5, "x": 0.3})
        assert renamed["effective_n"] == pytest.approx(base["effective_n"], rel=1e-12)


class TestMonotonicity:
    """
    Relations of the form "more of this means more of that". They pin the
    DIRECTION of a response, which a single-point test cannot.
    """

    def test_more_observations_narrow_the_interval(self):
        widths = []
        for n in (250, 1000, 4000):
            result = bootstrap_statistic(_returns(n, seed=11), n_bootstrap=250, seed=4)
            widths.append(result["interval_width"])
        assert widths == sorted(
            widths, reverse=True
        ), f"the confidence interval did not narrow with more data: {widths}"

    def test_a_wider_confidence_never_narrows_the_interval(self):
        returns = _returns(600, seed=12)
        widths = [
            bootstrap_statistic(returns, confidence=c, n_bootstrap=250, seed=5)[
                "interval_width"
            ]
            for c in (0.5, 0.8, 0.95, 0.99)
        ]
        assert widths == sorted(widths)

    def test_falling_correlation_raises_the_diversification_ratio(self):
        ratios = []
        for rho in (0.95, 0.6, 0.2, 0.0):
            corr = np.full((4, 4), rho)
            np.fill_diagonal(corr, 1.0)
            frame = pd.DataFrame(
                np.outer(VOLS, VOLS) * corr, index=NAMES, columns=NAMES
            )
            ratios.append(max_diversification(frame)["diversification_ratio"])
        assert ratios == sorted(
            ratios
        ), f"the diversification ratio did not rise as correlation fell: {ratios}"

    def test_more_volatility_widens_the_expected_move(self):
        from standard_quant_tools.analysis.derivatives import expected_move

        moves = [
            expected_move(spot=100.0, implied_vol=v, days=30)["one_sd_move"]
            for v in (0.1, 0.2, 0.4, 0.8)
        ]
        assert moves == sorted(moves)


class TestTranslation:
    def test_adding_a_constant_moves_the_mean_and_not_the_spread(self):
        """
        A basic sanity relation that catches an off-by-one in a rolling
        window or a mean subtracted twice.
        """
        returns = _returns(400, seed=13)
        shift = 0.001
        base = decompose_returns(returns)
        moved = decompose_returns([r + shift for r in returns])
        assert moved["arithmetic_mean"] == pytest.approx(
            base["arithmetic_mean"] + shift, rel=1e-9
        )
        assert moved["variance"] == pytest.approx(base["variance"], rel=1e-9)

    def test_negating_returns_negates_the_mean(self):
        returns = _returns(400, seed=14)
        base = decompose_returns(returns)
        flipped = decompose_returns([-r for r in returns])
        assert flipped["arithmetic_mean"] == pytest.approx(
            -base["arithmetic_mean"], rel=1e-9
        )
        assert flipped["win_rate"] == pytest.approx(
            1.0 - base["win_rate"], abs=0.01
        ), "a return of exactly zero is neither a win nor a loss either way"
