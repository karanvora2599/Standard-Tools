"""
The reason behind a set of portfolio weights, not only the weights.

WHAT THESE ARE FOR. Two optimizers in this runtime computed a great deal
about their own answer and returned none of it:

    Black-Litterman     computed the implied equilibrium returns, the
                        posterior returns and the posterior covariance,
                        used two of them for two scalars, and returned
                        weights. So a caller could state a view, get
                        plausible weights back, and have no way to tell a
                        view the posterior TOOK from one that tau and the
                        confidence damped to nothing.
    mean-variance       discarded SLSQP's iteration count, exit status and
                        message, objective value and Lagrange multipliers,
                        keeping one boolean; and computed the covariance's
                        condition number on every call while mentioning it
                        only inside a prose warning above 1e10, so two
                        universes under the threshold could not be
                        compared.

See the CHANGELOG entry of 2026-09-22.

THE NUMBERS ARE PLANTED. The equilibrium prior is computed in the test from
`delta * cov @ w_market` -- the same definition the blend uses -- so a view
stated exactly at it has a known answer (no distance to cover, so no
fraction of it covered), and a view stated away from it has a known
direction to move in.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.agent.models import PortfolioOptimizationInput
from standard_quant_tools.agent.runtimes.portfolio.tools import (
    run_portfolio_optimization,
)
from standard_quant_tools.portfolio.optimize import (
    annualized_mean_cov,
    black_litterman,
    mean_variance_optimize,
    view_absorption,
)

START = "2022-01-03"
END = "2023-12-29"
N_BARS = 500

TICKERS = ["AAA", "BBB", "CCC"]

#: A two-asset covariance with a clean, readable structure: the tests that
#: drive `black_litterman` directly use it so the equilibrium is arithmetic.
COV = np.array([[0.04, 0.01], [0.01, 0.05]])
MARKET = np.array([0.5, 0.5])
RELATIVE_VIEW = np.array([[1.0, -1.0]])


def _bars(seed: int, drift: float, vol: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range(START, periods=N_BARS, freq="B")
    close = 100.0 * np.cumprod(1.0 + rng.normal(drift, vol, N_BARS))
    return pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.004,
            "Low": close * 0.996,
            "Close": close,
            "Volume": np.full(N_BARS, 2e6),
        },
        index=index,
    )


#: Three assets with clearly different means and volatilities, so a
#: long-only target return between the minimum-variance portfolio's return
#: and the best asset's is genuinely feasible.
_UNIVERSE = {
    "AAA": _bars(1, 0.0008, 0.011),
    "BBB": _bars(2, 0.0004, 0.014),
    "CCC": _bars(3, 0.0001, 0.009),
}


@pytest.fixture
def provider(monkeypatch):
    from unittest.mock import AsyncMock, MagicMock

    from standard_quant_tools.data.factory import DataFactory

    stub = MagicMock()
    stub.get_ohlcv.side_effect = lambda symbol, *a, **kw: _UNIVERSE[symbol]
    stub.get_ohlcv_async = AsyncMock(
        side_effect=lambda symbol, *a, **kw: _UNIVERSE[symbol]
    )
    monkeypatch.setattr(DataFactory, "get_provider", lambda *a, **kw: stub)
    return stub


def _returns_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {t: _UNIVERSE[t]["Close"].pct_change(fill_method=None) for t in TICKERS}
    ).dropna()


def _optimize(**overrides):
    base = dict(
        tickers=list(TICKERS),
        start_date=START,
        end_date=END,
        method="max_sharpe",
    )
    base.update(overrides)
    return run_portfolio_optimization(PortfolioOptimizationInput(**base))


# ── how much of a view the posterior took ───────────────────────────────


class TestViewAbsorption:
    def test_a_view_held_with_near_certainty_is_absorbed_almost_entirely(self):
        """
        Certainty is a near-zero omega. The posterior then has no reason to
        stay near the equilibrium and moves essentially the whole way.
        """
        stated = np.array([0.05])
        result = black_litterman(
            COV, MARKET, RELATIVE_VIEW, stated, omega=np.array([[1e-10]])
        )
        rows = view_absorption(
            RELATIVE_VIEW,
            stated,
            result["implied_equilibrium_returns"],
            result["posterior_returns"],
        )
        assert rows[0]["absorbed_fraction"] == pytest.approx(1.0, abs=1e-6)
        assert rows[0]["posterior_spread"] == pytest.approx(0.05, abs=1e-6)

    def test_a_view_held_loosely_moves_the_posterior_far_less(self):
        near_certain = black_litterman(
            COV, MARKET, RELATIVE_VIEW, np.array([0.05]), omega=np.array([[1e-10]])
        )
        loose = black_litterman(
            COV, MARKET, RELATIVE_VIEW, np.array([0.05]), omega=np.array([[1.0]])
        )
        tight_fraction = view_absorption(
            RELATIVE_VIEW,
            np.array([0.05]),
            near_certain["implied_equilibrium_returns"],
            near_certain["posterior_returns"],
        )[0]["absorbed_fraction"]
        loose_fraction = view_absorption(
            RELATIVE_VIEW,
            np.array([0.05]),
            loose["implied_equilibrium_returns"],
            loose["posterior_returns"],
        )[0]["absorbed_fraction"]
        assert 0.0 < loose_fraction < 0.05
        assert loose_fraction < tight_fraction

    def test_a_view_stating_the_equilibrium_has_no_fraction_to_report(self):
        """
        0/0, not 0. There is no distance to cover, so reporting zero would
        read as "the view was ignored" -- which is the opposite of what
        agreeing with the prior means.
        """
        pi = 2.5 * COV @ MARKET
        stated = np.array([float(RELATIVE_VIEW[0] @ pi)])
        result = black_litterman(COV, MARKET, RELATIVE_VIEW, stated)
        row = view_absorption(
            RELATIVE_VIEW,
            stated,
            result["implied_equilibrium_returns"],
            result["posterior_returns"],
        )[0]
        assert math.isnan(row["absorbed_fraction"])
        assert row["posterior_spread"] == pytest.approx(row["prior_spread"], abs=1e-12)


class TestBlackLittermanThroughTheTool:
    def _view(self, confidence: float, view_return: float = 0.30):
        return self._run(
            [
                {
                    "assets": {"AAA": 1.0},
                    "view_return": view_return,
                    "confidence": confidence,
                }
            ]
        )

    def _run(self, views):
        return _optimize(method="black_litterman", views=views)

    def test_the_equilibrium_prior_is_the_one_the_blend_used(self, provider):
        result = self._view(1.0)
        _, cov = annualized_mean_cov(_returns_frame(), 252)
        expected = 2.5 * cov @ np.full(len(TICKERS), 1.0 / len(TICKERS))
        assert [
            result.implied_equilibrium_returns[t] for t in TICKERS
        ] == pytest.approx(list(expected), abs=1e-6)

    def test_the_posterior_volatilities_are_the_roots_of_the_diagonal(self, provider):
        from standard_quant_tools.portfolio.optimize import build_bl_views

        result = self._view(1.0)
        _, cov = annualized_mean_cov(_returns_frame(), 252)
        P, Q, omega = build_bl_views(
            TICKERS,
            [{"assets": {"AAA": 1.0}, "view_return": 0.30, "confidence": 1.0}],
            cov,
            tau=0.05,
        )
        blend = black_litterman(
            cov,
            np.full(len(TICKERS), 1.0 / len(TICKERS)),
            P,
            Q,
            risk_aversion=2.5,
            tau=0.05,
            omega=omega,
        )
        expected = np.sqrt(np.diag(blend["posterior_cov"]))
        assert [result.posterior_volatilities[t] for t in TICKERS] == pytest.approx(
            list(expected), abs=1e-6
        )
        # The posterior carries the blend's own estimation error on top of
        # the prior, so it is never tighter than the sample covariance.
        for ticker, sample in zip(TICKERS, np.sqrt(np.diag(cov))):
            assert result.posterior_volatilities[ticker] >= sample - 1e-12

    def test_confidence_orders_the_absorbed_fraction(self, provider):
        """
        Both views are stated identically and differ only in confidence, so
        the absorbed fraction is the only thing that can tell them apart --
        and before this it was not reported at all.
        """
        confident = self._view(1.0).view_absorption[0]
        hesitant = self._view(0.05).view_absorption[0]
        assert confident.stated == hesitant.stated == 0.30
        assert 0.0 < hesitant.absorbed_fraction < confident.absorbed_fraction < 1.0

    def test_a_confident_view_still_enters_at_half_strength_by_default(self, provider):
        """
        The measured consequence of the He-Litterman default uncertainty,
        `tau * P @ cov @ P.T`: at confidence 1.0 the view's variance and the
        prior's are the same size, so a single view is absorbed at exactly
        one half however certain the caller says they are. Reading the
        weights alone could never have shown that.
        """
        row = self._view(1.0).view_absorption[0]
        assert row.absorbed_fraction == pytest.approx(0.5, abs=1e-6)
        assert row.posterior_spread == pytest.approx(
            (row.stated + row.prior_spread) / 2.0, abs=1e-5
        )

    def test_a_view_at_the_equilibrium_reports_no_fraction(self, provider):
        _, cov = annualized_mean_cov(_returns_frame(), 252)
        pi = 2.5 * cov @ np.full(len(TICKERS), 1.0 / len(TICKERS))
        result = self._run(
            [{"assets": {"AAA": 1.0}, "view_return": float(pi[0]), "confidence": 1.0}]
        )
        row = result.view_absorption[0]
        assert row.absorbed_fraction is None
        assert row.stated == pytest.approx(row.prior_spread, abs=1e-6)

    def test_one_row_per_view_in_the_order_they_were_given(self, provider):
        result = self._run(
            [
                {"assets": {"AAA": 1.0}, "view_return": 0.30, "confidence": 1.0},
                {
                    "assets": {"BBB": 1.0, "CCC": -1.0},
                    "view_return": 0.05,
                    "confidence": 0.5,
                },
            ]
        )
        assert [row.view_index for row in result.view_absorption] == [0, 1]
        assert [row.stated for row in result.view_absorption] == [0.30, 0.05]

    def test_the_mean_variance_only_fields_stay_absent(self, provider):
        """`solver` is the mean-variance path's; Black-Litterman does not go
        through it, and a null says so rather than a fabricated report."""
        result = self._view(1.0)
        assert result.solver is None
        assert result.risk_contributions is None
        assert math.isfinite(result.condition_number)


# ── what the solver said about its own run ──────────────────────────────


class TestSolverReport:
    def _feasible_target(self) -> float:
        """Between the minimum-variance portfolio's return and the best
        asset's, so a long-only solve genuinely has somewhere to land."""
        mu, _ = annualized_mean_cov(_returns_frame(), 252)
        return float((min(mu) + max(mu)) / 2.0)

    def test_a_feasible_long_only_target_return_reports_a_successful_solve(
        self, provider
    ):
        target = self._feasible_target()
        result = _optimize(
            method="target_return", target_return=target, allow_short=False
        )
        assert result.converged is True
        assert result.solver is not None
        assert result.solver.method == "SLSQP"
        assert result.solver.iterations >= 1
        assert result.solver.status == 0
        assert math.isfinite(result.solver.objective)
        assert result.solver.n_function_evals >= 1
        assert result.expected_return == pytest.approx(target, abs=1e-4)

    def test_the_reported_objective_is_the_variance_at_the_returned_weights(
        self, provider
    ):
        """`target_return` minimizes variance, so the objective is the
        square of the volatility beside it."""
        result = _optimize(
            method="target_return",
            target_return=self._feasible_target(),
            allow_short=False,
        )
        assert result.solver.objective == pytest.approx(
            result.expected_volatility**2, rel=1e-4
        )

    def test_the_multipliers_are_reported_where_scipy_supplies_them(self, provider):
        result = _optimize(
            method="target_return",
            target_return=self._feasible_target(),
            allow_short=False,
        )
        # Guarded rather than required: `multipliers` is a recent scipy
        # field, and an older one is a null here, not a crash.
        if result.solver.multipliers is not None:
            assert all(math.isfinite(m) for m in result.solver.multipliers)

    def test_the_unconstrained_path_says_it_ran_no_solver(self, provider):
        result = _optimize(method="min_volatility", allow_short=True, max_weight=None)
        assert result.solver.method == "closed_form"
        assert result.solver.iterations == 0
        assert result.solver.status is None
        assert result.solver.n_function_evals is None
        assert result.solver.multipliers is None
        assert math.isfinite(result.solver.objective)

    def test_both_paths_score_the_same_objective(self, provider):
        """One definition, used as SLSQP's objective function and to score
        the closed form, so the two numbers mean the same thing."""
        closed = _optimize(method="min_volatility", allow_short=True, max_weight=None)
        # `expected_volatility` crosses the boundary rounded to six
        # decimals; the objective does not, so they agree to the rounding.
        assert closed.solver.objective == pytest.approx(
            closed.expected_volatility**2, rel=1e-5
        )

    def test_an_infeasible_target_is_visible_in_the_solver_report(self, provider):
        """The independent verification already called it non-convergent;
        the solver report now says what the solver did about it."""
        result = _optimize(
            method="target_return", target_return=99.0, allow_short=False
        )
        assert result.converged is False
        assert result.solver.status not in (0, None)
        assert result.solver.message


class TestConditionNumberIsAlwaysReported:
    def test_a_well_conditioned_universe_reports_a_finite_number_and_no_warning(
        self, provider
    ):
        result = _optimize(method="min_volatility", allow_short=False)
        assert math.isfinite(result.condition_number)
        assert result.condition_number < 1e10
        assert not any("ill-conditioned" in w for w in result.warnings)

    def test_the_number_matches_the_covariance_that_was_inverted(self, provider):
        result = _optimize(method="min_volatility", allow_short=False)
        _, cov = annualized_mean_cov(_returns_frame(), 252)
        assert result.condition_number == pytest.approx(
            float(np.linalg.cond(cov)), rel=1e-9
        )

    def test_a_near_collinear_universe_still_carries_the_prose_warning(self):
        """The threshold behaviour is unchanged: the number is now readable
        BELOW it as well, not instead of the warning above it."""
        rng = np.random.default_rng(4)
        base = rng.normal(0.0004, 0.012, 400)
        frame = pd.DataFrame(
            {
                "A": base,
                "B": base + rng.normal(0.0, 1e-9, 400),
                "C": rng.normal(0.0004, 0.012, 400),
            }
        )
        result = mean_variance_optimize(
            frame, "min_volatility", allow_short=True, max_weight=None
        )
        assert any("ill-conditioned" in w for w in result["warnings"])
        assert result["condition_number"] > 1e10

    def test_every_method_reports_it(self, provider):
        for method in ("max_sharpe", "min_volatility", "risk_parity"):
            result = _optimize(method=method)
            assert math.isfinite(
                result.condition_number
            ), f"{method} reported no condition number"
