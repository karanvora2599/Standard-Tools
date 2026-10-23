"""
The efficient frontier, from the closed form that was already in the file.

`portfolio/optimize.py` has carried the Merton constants and the exact
frontier weights for any target return since it was written, and the string
`frontier` appeared nowhere under `agent/` or `mcp/`. An agent that wanted a
curve had one route: call `run_portfolio_optimization(method='target_return')`
once per point -- one tool call and one agent turn each -- and it had to
know in advance which target returns the universe could support, because
nothing reported the span.

THE TEST THAT KEEPS THEM HONEST is the first class below: three points of
the curve against the tool an agent would otherwise have called forty
times, at the same target returns, with `allow_short=True` and
`max_weight=None` -- which is the setting under which the constants
describe the frontier at all, and under which the optimizer takes the same
closed-form branch. They agree to the last place either of them publishes,
and the arithmetic behind them is checked separately against the library
constants at full precision, because both tools round their outputs to six
decimals and a tool-to-tool comparison cannot see past that grid.

WHAT THE CURVE DOES NOT CLAIM. sum(w) = 1 and nothing else: a weight may be
negative and may exceed 1. Bounds make the feasible set compact and destroy
the closed form, so a long-only or capped frontier is a different curve and
remains one optimizer run per point. The tangency portfolio is the one
place the rate enters, and it is reported as absent-with-a-reason rather
than approximated when the closed form has no solution on the efficient
branch.

See the CHANGELOG entry of 2026-09-22.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError as PydanticValidationError

from standard_quant_tools.agent.models import (
    EfficientFrontierInput,
    PortfolioOptimizationInput,
)
from standard_quant_tools.agent.tools import (
    get_efficient_frontier,
    run_portfolio_optimization,
)
from standard_quant_tools.error import ValidationError

START, END = "2023-01-02", "2024-12-31"
N_BARS = 600

#: Both tools round every published number to six decimals, so no
#: tool-to-tool comparison can be tighter than one unit in that last place
#: -- two values a hair apart can land either side of the same boundary.
#: The 1e-9 claim is made against the library constants instead, where
#: nothing has been rounded yet.
WEIGHT_ROUNDING = 2e-6


def _returns(tickers, n=N_BARS, seed=4, spread=True):
    """Genuinely different paths per name: a provider answering every
    symbol with one frame makes the covariance singular and would let a
    frontier test pass on a universe that has no frontier."""
    rng = np.random.default_rng(seed)
    index = pd.date_range(START, periods=n, freq="B")
    return pd.DataFrame(
        {
            t: rng.normal(
                0.0004 + (0.00012 * i if spread else 0.0),
                0.011 + (0.0018 * i if spread else 0.0),
                n,
            )
            for i, t in enumerate(tickers)
        },
        index=index,
    )


def _patched(frame_for=None):
    def _fake(tickers, start, end, interval="1d"):
        return _returns(tickers) if frame_for is None else frame_for(tickers)

    return patch(
        "standard_quant_tools.agent.runtimes.portfolio.tools.fetch_returns_sync", _fake
    )


def _frontier(**kwargs):
    payload = {
        "tickers": ["AAA", "BBB", "CCC"],
        "start_date": START,
        "end_date": END,
    }
    payload.update(kwargs)
    with _patched():
        return get_efficient_frontier(EfficientFrontierInput(**payload))


class TestTheCurveAgreesWithTheToolItReplaces:
    @staticmethod
    def _solve_one(target_return):
        with _patched():
            return run_portfolio_optimization(
                PortfolioOptimizationInput(
                    tickers=["AAA", "BBB", "CCC"],
                    start_date=START,
                    end_date=END,
                    method="target_return",
                    target_return=target_return,
                    allow_short=True,
                    max_weight=None,
                )
            )

    def test_three_points_match_the_optimizer_at_the_published_precision(self):
        """The same target return, solved both ways. `allow_short=True`
        with `max_weight=None` is the unconstrained problem the closed form
        describes; any bound would make them different questions. Both
        tools publish six decimals, so this is agreement to the last place
        either of them has."""
        result = _frontier(n_points=7)
        for point in result.points[2:5]:
            solved = self._solve_one(point.expected_return)
            assert solved.expected_return == pytest.approx(
                point.expected_return, abs=1e-9
            )
            assert solved.expected_volatility == pytest.approx(
                point.volatility, abs=WEIGHT_ROUNDING
            )
            for ticker, weight in point.weights.items():
                assert solved.weights[ticker] == pytest.approx(
                    weight, abs=WEIGHT_ROUNDING
                )

    def test_the_optimizer_took_the_same_closed_form_branch(self):
        """Which is why the agreement above is exact rather than
        approximate: the frontier tool is a SHAPE over the algebra the
        optimizer already runs here, not a second implementation of it."""
        result = _frontier(n_points=5)
        solved = self._solve_one(result.points[2].expected_return)
        assert solved.solver is not None
        assert solved.solver.method == "closed_form"

    def test_each_point_is_the_frontier_algebra_to_1e_9(self):
        """Full precision, which a tool-to-tool comparison cannot reach
        because both round. The weights are re-derived from the library's
        own Merton constants and the point's own numbers must be the
        six-decimal rounding of that derivation, exactly -- any error above
        half a unit in the last place would break this."""
        from standard_quant_tools.portfolio.optimize import (
            annualized_mean_cov,
            frontier_stats,
            frontier_weights,
        )

        frame = _returns(["AAA", "BBB", "CCC"])
        mu, cov = annualized_mean_cov(frame, 252)
        sigma_inv, ones, A, B, C, D = frontier_stats(mu, cov)

        # An explicit span whose points are exactly representable, so the
        # target return the tool solved at is the one re-derived here
        # rather than a six-decimal rounding of it.
        result = _frontier(n_points=5, return_range=(0.10, 0.30))
        for target, point in zip([0.10, 0.15, 0.20, 0.25, 0.30], result.points):
            assert point.expected_return == pytest.approx(target, abs=1e-9)
            weights = frontier_weights(sigma_inv, ones, mu, A, B, C, D, target)
            assert round(float(np.sqrt(weights @ cov @ weights)), 6) == pytest.approx(
                point.volatility, abs=1e-9
            )
            for ticker, weight in zip(result.tickers, weights):
                assert round(float(weight), 6) == pytest.approx(
                    point.weights[ticker], abs=1e-9
                )

    def test_the_minimum_variance_point_matches_the_optimizer(self):
        result = _frontier()
        with _patched():
            solved = run_portfolio_optimization(
                PortfolioOptimizationInput(
                    tickers=["AAA", "BBB", "CCC"],
                    start_date=START,
                    end_date=END,
                    method="min_volatility",
                    allow_short=True,
                    max_weight=None,
                )
            )
        assert solved.expected_volatility == pytest.approx(
            result.min_variance.volatility, abs=1e-9
        )

    def test_the_tangency_point_matches_the_max_sharpe_optimizer(self):
        result = _frontier(risk_free_rate=0.02)
        with _patched():
            solved = run_portfolio_optimization(
                PortfolioOptimizationInput(
                    tickers=["AAA", "BBB", "CCC"],
                    start_date=START,
                    end_date=END,
                    method="max_sharpe",
                    risk_free_rate=0.02,
                    allow_short=True,
                    max_weight=None,
                )
            )
        assert result.tangency is not None
        assert solved.expected_return == pytest.approx(
            result.tangency.expected_return, abs=1e-9
        )
        assert solved.expected_volatility == pytest.approx(
            result.tangency.volatility, abs=1e-9
        )


class TestTheShapeOfTheCurve:
    def test_the_requested_number_of_points_comes_back(self):
        assert len(_frontier(n_points=31).points) == 31

    def test_every_weight_vector_sums_to_one(self):
        """The one constraint the closed form imposes. It holds
        algebraically for every target return, so a point that missed it
        would mean the constants were computed from the wrong matrix."""
        for point in _frontier(n_points=15).points:
            assert sum(point.weights.values()) == pytest.approx(1.0, abs=1e-5)

    def test_the_points_run_from_low_return_to_high(self):
        returns = [p.expected_return for p in _frontier(n_points=9).points]
        assert returns == sorted(returns)

    def test_the_minimum_variance_point_is_the_least_volatile_of_all(self):
        """It is the leftmost point of the curve by definition, so nothing
        on the traced span may sit below it."""
        result = _frontier(n_points=21)
        floor = result.min_variance.volatility
        assert min(p.volatility for p in result.points) >= floor - 1e-9

    def test_volatility_is_convex_around_the_minimum(self):
        """A frontier is a parabola in (variance, return). Sampling it
        above the minimum-variance return must give volatilities that only
        increase."""
        result = _frontier(n_points=15)
        above = [
            p.volatility
            for p in result.points
            if p.expected_return >= result.min_variance.expected_return
        ]
        assert above == sorted(above)

    def test_the_default_span_starts_at_the_minimum_variance_return(self):
        result = _frontier(n_points=5)
        assert result.points[0].expected_return == pytest.approx(
            result.min_variance.expected_return, abs=1e-9
        )

    def test_an_explicit_return_range_is_honoured(self):
        result = _frontier(n_points=5, return_range=(0.05, 0.25))
        assert result.points[0].expected_return == pytest.approx(0.05, abs=1e-6)
        assert result.points[-1].expected_return == pytest.approx(0.25, abs=1e-6)

    def test_the_universe_is_named_from_the_solved_columns(self):
        result = _frontier()
        assert result.tickers == ["AAA", "BBB", "CCC"]
        assert set(result.min_variance.weights) == {"AAA", "BBB", "CCC"}
        assert result.n_observations == N_BARS


class TestTheSchemaRefusesWhatHasNoFrontier:
    def test_a_one_asset_universe_is_refused(self):
        """With one asset there is no trade-off to draw: every fully
        invested portfolio is the same portfolio."""
        with pytest.raises(PydanticValidationError):
            EfficientFrontierInput(tickers=["AAA"], start_date=START, end_date=END)

    def test_an_empty_universe_is_refused(self):
        with pytest.raises(PydanticValidationError):
            EfficientFrontierInput(tickers=[], start_date=START, end_date=END)

    def test_a_duplicated_ticker_is_refused_by_name(self):
        """Two identical columns make the covariance singular by
        construction, which has no frontier rather than a poor one."""
        with pytest.raises(PydanticValidationError, match="duplicate"):
            EfficientFrontierInput(
                tickers=["AAA", "AAA"], start_date=START, end_date=END
            )

    def test_a_reversed_return_range_is_refused(self):
        with pytest.raises(PydanticValidationError, match="high"):
            EfficientFrontierInput(
                tickers=["AAA", "BBB"],
                start_date=START,
                end_date=END,
                return_range=(0.20, 0.05),
            )

    @pytest.mark.parametrize("n_points", [1, 0, 201])
    def test_an_unusable_point_count_is_refused(self, n_points):
        with pytest.raises(PydanticValidationError):
            EfficientFrontierInput(
                tickers=["AAA", "BBB"],
                start_date=START,
                end_date=END,
                n_points=n_points,
            )

    def test_an_unknown_argument_is_refused_rather_than_ignored(self):
        """`method` is run_portfolio_optimization's argument, not this
        tool's; a caller who passes it gets told rather than ignored."""
        with pytest.raises(PydanticValidationError):
            EfficientFrontierInput(
                tickers=["AAA", "BBB"],
                start_date=START,
                end_date=END,
                method="target_return",
            )

    def test_more_assets_than_observations_is_refused_with_the_remedy(self):
        """A sample covariance of N assets from fewer than N observations
        is singular by construction, and an optimizer can find a
        zero-variance portfolio in its null space that carries real risk."""
        tickers = [f"T{i}" for i in range(8)]
        with patch(
            "standard_quant_tools.agent.runtimes.portfolio.tools.fetch_returns_sync",
            lambda t, s, e, interval="1d": _returns(t, n=5),
        ):
            with pytest.raises(ValidationError, match="cannot estimate a covariance"):
                get_efficient_frontier(
                    EfficientFrontierInput(
                        tickers=tickers, start_date=START, end_date=END
                    )
                )


class TestANearlyCollinearUniverseSaysSo:
    """Rank alone does not detect a covariance that is EFFECTIVELY
    singular, and the frontier inverts that covariance once for every point
    it returns -- so the whole curve is the amplification of a difference
    close to noise, and nothing about the weights says so on their own."""

    @staticmethod
    def _almost_identical(tickers, n=N_BARS, seed=9):
        rng = np.random.default_rng(seed)
        index = pd.date_range(START, periods=n, freq="B")
        base = rng.normal(0.0004, 0.012, n)
        columns = {tickers[0]: base}
        for i, ticker in enumerate(tickers[1:], start=1):
            columns[ticker] = base + rng.normal(0.0, 1e-9, n) * i
        return pd.DataFrame(columns, index=index)

    def _collinear_frontier(self, **kwargs):
        with patch(
            "standard_quant_tools.agent.runtimes.portfolio.tools.fetch_returns_sync",
            lambda t, s, e, interval="1d": self._almost_identical(t),
        ):
            return get_efficient_frontier(
                EfficientFrontierInput(
                    tickers=["AAA", "BBB", "CCC"],
                    start_date=START,
                    end_date=END,
                    **kwargs,
                )
            )

    def test_the_ill_conditioned_warning_is_carried(self):
        result = self._collinear_frontier()
        assert any("ill-conditioned" in w for w in result.warnings)

    def test_the_condition_number_is_reported_at_every_level(self):
        """Reported rather than only warned about, so two universes that
        both sit under the threshold can still be compared."""
        healthy = _frontier()
        sick = self._collinear_frontier()
        assert healthy.condition_number is not None
        assert healthy.warnings == [] or not any(
            "ill-conditioned" in w for w in healthy.warnings
        )
        assert sick.condition_number > healthy.condition_number

    def test_the_curve_is_still_returned(self):
        """Reported, not refused: an ill-conditioned covariance is still
        the caller's data, and there are legitimate reasons to optimize
        over near-duplicates."""
        assert len(self._collinear_frontier(n_points=5).points) == 5


class TestTheTangencyPortfolioIsAbsentWithAReason:
    def test_a_rate_above_the_minimum_variance_return_yields_null(self):
        """At or above the minimum-variance portfolio's own return, every
        fully invested portfolio on the efficient branch has negative
        excess return and the Sharpe supremum is not attained. The same
        algebra normalized anyway would land on the INEFFICIENT branch and
        read as an ordinary answer."""
        healthy = _frontier()
        above = healthy.min_variance.expected_return + 0.05

        result = _frontier(risk_free_rate=above)
        assert result.tangency is None
        assert any("tangency" in w for w in result.warnings)

    def test_the_warning_names_a_rate_that_would_work(self):
        healthy = _frontier()
        result = _frontier(risk_free_rate=healthy.min_variance.expected_return + 0.05)
        assert any("risk_free_rate below" in w for w in result.warnings)

    def test_the_rest_of_the_curve_is_unaffected(self):
        """The frontier itself does not depend on the rate, so losing the
        tangency point must not cost the caller the answer they asked
        for."""
        healthy = _frontier(n_points=9)
        result = _frontier(
            n_points=9, risk_free_rate=healthy.min_variance.expected_return + 0.05
        )
        assert len(result.points) == 9
        assert result.min_variance.volatility == pytest.approx(
            healthy.min_variance.volatility, abs=1e-12
        )

    def test_an_ordinary_rate_still_produces_one(self):
        """The null case: without it, "tangency is null" would be satisfied
        by a tool that never produces one."""
        result = _frontier(risk_free_rate=0.01)
        assert result.tangency is not None
        assert sum(result.tangency.weights.values()) == pytest.approx(1.0, abs=1e-5)


class TestTheBoundedFrontier:
    """allow_short=False or a max_weight: the critical line algorithm's
    corner portfolios, and every point a mix of the two around it. Checked
    against the optimizer it replaces, point by point, under the same
    bounds -- each of those an exact solve of its own."""

    TICKERS = ["AAA", "BBB", "CCC", "DDD"]

    def _bounded(self, **kwargs):
        payload = {"tickers": self.TICKERS, "start_date": START, "end_date": END}
        payload.update(kwargs)
        with _patched():
            return get_efficient_frontier(EfficientFrontierInput(**payload))

    def _optimize(self, **kwargs):
        with _patched():
            return run_portfolio_optimization(
                PortfolioOptimizationInput(
                    tickers=self.TICKERS, start_date=START, end_date=END, **kwargs
                )
            )

    def test_the_default_is_still_the_closed_form(self):
        result = _frontier()
        assert result.method == "closed_form"
        assert result.corners is None
        assert result.certified is None
        assert result.largest_kkt_residual is None
        assert not any("largest_kkt_residual" in w for w in result.warnings)

    @pytest.mark.parametrize(
        "bounds",
        [dict(allow_short=False), dict(allow_short=False, max_weight=0.4)],
        ids=["long_only", "capped"],
    )
    def test_corners_are_certified_and_inside_the_bounds(self, bounds):
        result = self._bounded(**bounds)
        assert result.method == "critical_line"
        assert result.certified is True
        assert result.largest_kkt_residual <= 1e-12
        cap = bounds.get("max_weight", 1.0)
        for point in [*result.corners, *result.points]:
            assert sum(point.weights.values()) == pytest.approx(1.0, abs=1e-5)
            assert min(point.weights.values()) >= 0.0
            assert max(point.weights.values()) <= cap
        tolerances = [c.risk_tolerance for c in result.corners]
        assert tolerances == sorted(tolerances, reverse=True)
        assert tolerances[-1] == 0.0 or "minimum-variance end" in (
            result.corners[-1].event
        )

    @pytest.mark.parametrize(
        "bounds",
        [
            dict(allow_short=False),
            dict(allow_short=False, max_weight=0.4),
            dict(allow_short=True, max_weight=0.6),
        ],
        ids=["long_only", "capped", "shorting_capped"],
    )
    def test_every_point_is_the_optimizers_target_return_portfolio(self, bounds):
        """The same bounds, the same return, the optimizer's exact
        target_return solve: the same portfolio to the six decimals both
        publish. The returns are round numbers inside the attainable span,
        so both tools solve at the same one rather than one at the other's
        rounding (with shorts a weight moves ten times the return)."""
        span = self._bounded(n_points=2, **bounds).points
        low, high = span[0].expected_return, span[-1].expected_return
        targets = [round(low + share * (high - low), 4) for share in (0.2, 0.5, 0.8)]
        for target in targets:
            point = self._bounded(
                n_points=2, return_range=(target, targets[2] + 1e-4), **bounds
            ).points[0]
            assert point.expected_return == pytest.approx(target, abs=1e-9)
            solved = self._optimize(
                method="target_return", target_return=target, **bounds
            )
            assert solved.solver.certified is True
            assert solved.expected_volatility == pytest.approx(
                point.volatility, abs=WEIGHT_ROUNDING
            )
            for ticker, weight in point.weights.items():
                assert solved.weights[ticker] == pytest.approx(
                    weight, abs=WEIGHT_ROUNDING
                )

    def test_the_ends_and_the_tangency_match_the_optimizer(self):
        bounds = dict(allow_short=False, max_weight=0.4)
        result = self._bounded(risk_free_rate=0.02, **bounds)
        least = self._optimize(method="min_volatility", **bounds)
        assert least.expected_volatility == pytest.approx(
            result.min_variance.volatility, abs=WEIGHT_ROUNDING
        )
        best = self._optimize(method="max_sharpe", risk_free_rate=0.02, **bounds)
        assert best.solver.certified is True
        assert result.tangency is not None
        for ticker, weight in result.tangency.weights.items():
            assert best.weights[ticker] == pytest.approx(weight, abs=WEIGHT_ROUNDING)

    def test_the_default_span_is_the_attainable_one(self):
        """From the minimum-variance portfolio's return to the highest any
        portfolio inside the bounds earns: all in the best asset,
        long-only."""
        result = self._bounded(allow_short=False, n_points=5)
        means = _returns(self.TICKERS).mean() * 252
        assert result.points[-1].expected_return == pytest.approx(
            float(means.max()), abs=1e-6
        )
        assert result.points[0].expected_return == pytest.approx(
            result.min_variance.expected_return, abs=1e-9
        )
        assert result.corners[0].weights[str(means.idxmax())] == 1.0

    def test_target_volatility_lands_on_the_frontier(self):
        """A point's volatility asked of the optimizer's target_volatility
        returns that point: the optimizer now solves on the same corners."""
        bounds = dict(allow_short=False, max_weight=0.4)
        result = self._bounded(n_points=5, **bounds)
        point = result.points[2]
        solved = self._optimize(
            method="target_volatility", target_volatility=point.volatility, **bounds
        )
        assert solved.solver.method == "critical_line"
        assert solved.expected_return == pytest.approx(
            point.expected_return, abs=WEIGHT_ROUNDING
        )

    def test_a_return_range_outside_the_attainable_span_is_refused(self):
        with pytest.raises(ValidationError, match="returns of the efficient"):
            self._bounded(allow_short=False, return_range=(0.0, 5.0))

    def test_an_infeasible_cap_is_refused(self):
        with pytest.raises(PydanticValidationError, match="infeasible"):
            EfficientFrontierInput(
                tickers=self.TICKERS, start_date=START, end_date=END, max_weight=0.2
            )

    def test_no_tangency_when_nothing_inside_the_bounds_earns_the_rate(self):
        result = self._bounded(allow_short=False, risk_free_rate=5.0)
        assert result.tangency is None
        assert any("no maximum-Sharpe" in w for w in result.warnings)


class TestTheToolIsReachable:
    def test_it_is_in_the_portfolio_runtime_dispatch_table(self):
        from standard_quant_tools.agent.runtimes.portfolio import (
            TOOL_CATEGORY,
            TOOL_DEFS,
            TOOL_DISPATCH,
        )

        assert "get_efficient_frontier" in TOOL_DISPATCH
        assert TOOL_CATEGORY["get_efficient_frontier"] == "portfolio_risk"
        assert "get_efficient_frontier" in {name for name, _d, _m in TOOL_DEFS}

    def test_the_advertised_schema_is_the_dispatched_one(self):
        from standard_quant_tools.agent.runtimes.portfolio import TOOL_DISPATCH

        handler, model = TOOL_DISPATCH["get_efficient_frontier"]
        assert model is EfficientFrontierInput
        assert handler is get_efficient_frontier

    def test_it_is_exported_from_the_facade(self):
        import standard_quant_tools.agent as package

        assert "get_efficient_frontier" in package.__all__
        assert package.get_efficient_frontier is get_efficient_frontier

    def test_its_models_are_exported_too(self):
        import standard_quant_tools.agent as package

        for name in (
            "EfficientFrontierInput",
            "EfficientFrontierResult",
            "FrontierPoint",
        ):
            assert name in package.__all__
            assert hasattr(package, name)

    def test_it_dispatches_by_name_through_the_runtime(self):
        from standard_quant_tools.agent.runtimes import all_runtimes

        runtime = all_runtimes()["portfolio"]
        with _patched():
            payload = runtime.dispatch(
                "get_efficient_frontier",
                {
                    "tickers": ["AAA", "BBB", "CCC"],
                    "start_date": START,
                    "end_date": END,
                    "n_points": 4,
                },
            )
        assert len(payload["points"]) == 4
        assert payload["min_variance"]["weights"]
