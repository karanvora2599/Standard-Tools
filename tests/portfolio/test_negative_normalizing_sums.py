"""
A holding with a non-positive sum is refused, not flipped into a portfolio.

Black-Litterman and maximum diversification both solve for an unconstrained
optimal holding and rescale it to sum to 1. When the holding's sum is
negative the rescaling divides every position by a negative number, which
flips it: the result is the WORST portfolio along the optimal direction,
reported as the best. Black-Litterman with bearish views on every asset
returned a 100%-long book at a posterior Sharpe of -1.16 with
converged=True; maximum diversification returned a diversification ratio of
-5.32 on the three-asset matrix below. The max_sharpe tangency portfolio
already refused exactly this case. Both now do the same, with the reason.
See the CHANGELOG entry of 2026-10-22.

THE PLANTED ANSWERS. For Black-Litterman the posterior and the raw holding
are recomputed here from He and Litterman's formulas, so the refused sum is
known. For maximum diversification the holding is Sigma^-1 sigma, whose sum
and ratio sqrt(sigma' Sigma^-1 sigma) are one linear solve.
"""

from __future__ import annotations

import math
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.agent.models import PortfolioOptimizationInput
from standard_quant_tools.agent.runtimes import resolve
from standard_quant_tools.agent.runtimes.portfolio.construction_tools import (
    MaxDiversificationInput,
    optimize_max_diversification,
)
from standard_quant_tools.agent.runtimes.portfolio.tools import (
    run_portfolio_optimization,
)
from standard_quant_tools.data.factory import DataFactory
from standard_quant_tools.error import ValidationError
from standard_quant_tools.portfolio.construction import max_diversification
from standard_quant_tools.portfolio.optimize import black_litterman

# ── Black-Litterman ─────────────────────────────────────────────────────

BL_COV = np.array([[0.04, 0.006], [0.006, 0.09]])
BL_MARKET = np.array([0.6, 0.4])
BL_P = np.eye(2)
BL_Q = np.array([-0.30, -0.20])
BL_OMEGA = np.diag([1e-6, 1e-6])
DELTA, TAU = 2.5, 0.05


def _raw_holding(q):
    """He-Litterman by hand: the posterior mean and covariance, and the
    unconstrained optimal holding (delta * posterior_cov)^-1 posterior."""
    pi = DELTA * BL_COV @ BL_MARKET
    prior_precision = np.linalg.inv(TAU * BL_COV)
    view_precision = BL_P.T @ np.linalg.inv(BL_OMEGA) @ BL_P
    middle = np.linalg.inv(prior_precision + view_precision)
    posterior = middle @ (prior_precision @ pi + BL_P.T @ np.linalg.inv(BL_OMEGA) @ q)
    return np.linalg.solve(DELTA * (BL_COV + middle), posterior), posterior


class TestBlackLittermanRefusesANetShortHolding:
    def test_bearish_views_on_every_asset_are_refused_with_the_sum(self):
        raw, posterior = _raw_holding(BL_Q)
        assert raw.sum() == pytest.approx(-3.5895394, abs=1e-6)
        # What the old normalization returned: every position flipped onto
        # a long book whose posterior expected return is negative.
        flipped = raw / raw.sum()
        assert (flipped > 0).all()
        assert flipped @ posterior < 0 < raw @ posterior
        with pytest.raises(ValidationError, match=r"sums to -3\.589539") as caught:
            black_litterman(
                BL_COV,
                BL_MARKET,
                BL_P,
                BL_Q,
                risk_aversion=DELTA,
                tau=TAU,
                omega=BL_OMEGA,
            )
        message = str(caught.value)
        assert "net SHORT" in message
        assert "relative" in message

    def test_bullish_views_are_answered_as_before(self):
        """The null case: a positive sum is rescaled exactly as it was."""
        q = np.array([0.10, 0.06])
        raw, _ = _raw_holding(q)
        assert raw.sum() > 0
        result = black_litterman(
            BL_COV, BL_MARKET, BL_P, q, risk_aversion=DELTA, tau=TAU, omega=BL_OMEGA
        )
        np.testing.assert_allclose(
            result["implied_weights"], raw / raw.sum(), rtol=1e-9
        )


def _bars(seed, drift, vol, n=500):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-03", periods=n, freq="B")
    close = 100 * np.cumprod(1 + rng.normal(drift, vol, n))
    return pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.004,
            "Low": close * 0.996,
            "Close": close,
            "Volume": np.full(n, 2e6),
        },
        index=idx,
    )


@pytest.fixture
def three_assets(monkeypatch):
    universe = {
        "AAA": _bars(1, 0.0008, 0.011),
        "BBB": _bars(2, 0.0004, 0.014),
        "CCC": _bars(3, 0.0001, 0.009),
    }
    provider = MagicMock()
    provider.get_ohlcv.side_effect = lambda s, *a, **k: universe[s]
    provider.get_ohlcv_async = AsyncMock(side_effect=lambda s, *a, **k: universe[s])
    monkeypatch.setattr(DataFactory, "get_provider", lambda *a, **k: provider)


class TestTheToolDoor:
    def _input(self, view_return):
        return PortfolioOptimizationInput(
            tickers=["AAA", "BBB", "CCC"],
            start_date="2022-01-03",
            end_date="2024-01-01",
            method="black_litterman",
            views=[
                {"assets": {t: 1.0}, "view_return": view_return, "confidence": 1.0}
                for t in ("AAA", "BBB", "CCC")
            ],
        )

    def test_the_bearish_book_is_refused_not_reported_as_converged(self, three_assets):
        """It came back as weights 0.36/0.18/0.47, expected return -11.6%,
        Sharpe -1.16 and converged=True -- the literal True the tool wrote
        for every Black-Litterman answer."""
        with pytest.raises(ValidationError, match="net SHORT"):
            run_portfolio_optimization(self._input(-0.25))

    def test_a_bullish_book_reports_converged_from_its_weights(self, three_assets):
        result = run_portfolio_optimization(self._input(0.25))
        assert result.converged is True
        assert sum(result.weights.values()) == pytest.approx(1.0, abs=1e-5)
        assert result.sharpe_ratio > 0


# ── maximum diversification ─────────────────────────────────────────────

VOLS = np.array([3.1731706, 4.58591887, 0.80118955])
CORR = np.array([[1.0, -0.474, 0.792], [-0.474, 1.0, 0.144], [0.792, 0.144, 1.0]])
NAMES = ["A", "B", "C"]
SIGMA = CORR * np.outer(VOLS, VOLS)


class TestMaxDiversificationRefusesTheFlippedRatio:
    def test_the_planted_holding_sums_negative(self):
        """The oracle: a valid, moderately conditioned covariance whose
        maximum-diversification holding is net short."""
        assert np.linalg.eigvalsh(CORR).min() > 0.009
        assert np.linalg.cond(CORR) == pytest.approx(203.875, abs=1e-3)
        holding = np.linalg.solve(SIGMA, VOLS)
        assert holding.sum() < 0
        best = math.sqrt(VOLS @ holding)
        assert best == pytest.approx(5.3224562, abs=1e-6)
        # What the old normalization reported: the minimum, with its sign.
        flipped = holding / holding.sum()
        ratio = (flipped @ VOLS) / math.sqrt(flipped @ SIGMA @ flipped)
        assert ratio == pytest.approx(-best, rel=1e-9)

    def test_the_library_refuses_it_with_the_ratio_it_would_have_flipped(self):
        frame = pd.DataFrame(SIGMA, index=NAMES, columns=NAMES)
        with pytest.raises(ValidationError, match=r"its minimum \(-5\.3225\)"):
            max_diversification(frame)

    def test_the_tool_refuses_it_too(self):
        with pytest.raises(ValidationError, match="no fully-invested portfolio"):
            optimize_max_diversification(
                MaxDiversificationInput(assets=NAMES, covariance=SIGMA.tolist())
            )
        with pytest.raises(ValidationError, match="no fully-invested portfolio"):
            resolve("portfolio").dispatch(
                "optimize_max_diversification",
                {"assets": NAMES, "covariance": SIGMA.tolist()},
            )

    def test_a_positive_sum_is_answered_as_before(self):
        """The null case: three names at 20% volatility and correlation 0.3
        hold equal weights at a ratio of 1 / sqrt((1 + 2 * 0.3) / 3)."""
        corr = np.full((3, 3), 0.3) + 0.7 * np.eye(3)
        frame = pd.DataFrame(corr * 0.04, index=NAMES, columns=NAMES)
        result = max_diversification(frame)
        for weight in result["weights"].values():
            assert weight == pytest.approx(1 / 3, abs=1e-12)
        assert result["diversification_ratio"] == pytest.approx(
            1 / math.sqrt(1.6 / 3), abs=1e-12
        )
