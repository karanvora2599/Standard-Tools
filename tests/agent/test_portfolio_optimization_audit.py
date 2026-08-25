"""
Regression tests for the agent-tool half of the portfolio/screener audit:
duplicate tickers desynchronizing a result's own fields, the optimizer's
caveats being dropped at the tool boundary, and the risk_parity /
black_litterman branches bypassing the covariance gate.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError as PydanticValidationError

from standard_quant_tools.agent.models import PortfolioOptimizationInput
from standard_quant_tools.agent.tools import run_portfolio_optimization
from standard_quant_tools.error import ValidationError


def _returns(tickers, n=600, seed=4):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-02", periods=n, freq="B")
    # dict comprehension mirrors fetch_returns_async: duplicate keys collapse
    return pd.DataFrame({t: rng.normal(0.0005, 0.012, n) for t in tickers}, index=idx)


def _patched(tickers, n=600):
    def _fake(req_tickers, start, end, interval="1d"):
        return _returns(req_tickers, n)

    return patch(
        "standard_quant_tools.agent.runtimes.portfolio.tools.fetch_returns_sync", _fake
    )


class TestDuplicateTickers:
    """
    The returns frame is built as {ticker: close}, so a repeated ticker
    collapses to one column and the optimizer silently sees fewer assets than
    were requested. The response then echoed the full requested list as
    `tickers` while `weights` had one entry per SURVIVING column -- so two
    fields of a single result disagreed about the size of the universe.
    """

    def test_duplicate_tickers_rejected_at_the_boundary(self):
        with pytest.raises(PydanticValidationError, match="duplicate symbols"):
            PortfolioOptimizationInput(
                tickers=["AAA", "BBB", "AAA"],
                start_date="2023-01-02",
                end_date="2024-01-01",
                method="risk_parity",
            )

    def test_rejection_explains_why_dedup_is_not_done_silently(self):
        with pytest.raises(PydanticValidationError, match="appear once"):
            PortfolioOptimizationInput(
                tickers=["AAA", "AAA"],
                start_date="2023-01-02",
                end_date="2024-01-01",
                method="max_sharpe",
            )

    def test_unique_tickers_still_accepted(self):
        spec = PortfolioOptimizationInput(
            tickers=["AAA", "BBB"],
            start_date="2023-01-02",
            end_date="2024-01-01",
            method="risk_parity",
        )
        assert spec.tickers == ["AAA", "BBB"]

    def test_weights_are_labelled_from_the_solved_columns(self):
        """
        Defense in depth: even with the validator in place, a weight must be
        named by the series it was actually computed from rather than by the
        requested list, so the two cannot drift apart again.
        """
        inp = PortfolioOptimizationInput(
            tickers=["AAA", "BBB", "CCC"],
            start_date="2023-01-02",
            end_date="2024-01-01",
            method="risk_parity",
        )
        with _patched(inp.tickers):
            result = run_portfolio_optimization(inp)
        assert set(result.weights) == {"AAA", "BBB", "CCC"}
        assert len(result.weights) == len(result.tickers)


class TestOptimizerCaveatsReachTheCaller:
    def test_small_sample_warning_is_surfaced_by_the_tool(self):
        """
        mean_variance_optimize gained a warnings list; dropping it at this
        boundary would leave an agent with a 22x volatility understatement
        and nothing indicating it.
        """
        inp = PortfolioOptimizationInput(
            tickers=["AAA", "BBB"],
            start_date="2023-01-02",
            end_date="2023-02-01",
            method="min_volatility",
            allow_short=True,
        )
        with _patched(inp.tickers, n=12):
            result = run_portfolio_optimization(inp)
        assert any("observations" in w for w in result.warnings)

    def test_ample_history_produces_no_small_sample_warning(self):
        inp = PortfolioOptimizationInput(
            tickers=["AAA", "BBB"],
            start_date="2023-01-02",
            end_date="2024-01-01",
            method="min_volatility",
            allow_short=True,
        )
        with _patched(inp.tickers, n=600):
            result = run_portfolio_optimization(inp)
        assert not any("observations" in w for w in result.warnings)


class TestNonMeanVarianceBranchesShareTheCovarianceGate:
    """
    risk_parity and black_litterman bypass mean_variance_optimize entirely,
    so the estimability gate added there would not have covered them -- the
    same singular-by-construction covariance would still reach them.
    """

    @pytest.mark.parametrize("method", ["risk_parity", "black_litterman"])
    def test_singular_by_construction_covariance_rejected(self, method):
        tickers = [f"A{i}" for i in range(6)]
        kwargs = dict(
            tickers=tickers,
            start_date="2023-01-02",
            end_date="2023-01-10",
            method=method,
        )
        if method == "black_litterman":
            kwargs["views"] = [{"assets": {"A0": 1.0}, "view_return": 0.1}]
        inp = PortfolioOptimizationInput(**kwargs)
        with _patched(tickers, n=5):
            with pytest.raises(ValidationError, match="observations for"):
                run_portfolio_optimization(inp)

    def test_sufficient_observations_still_solve(self):
        tickers = [f"A{i}" for i in range(6)]
        inp = PortfolioOptimizationInput(
            tickers=tickers,
            start_date="2023-01-02",
            end_date="2024-01-01",
            method="risk_parity",
        )
        with _patched(tickers, n=600):
            result = run_portfolio_optimization(inp)
        assert len(result.weights) == 6
        assert result.expected_volatility > 0
