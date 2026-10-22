"""
Four portfolio tools that answered with a number they did not have.

- `run_portfolio_optimization` took `max_weight` and `allow_short` with
  method="black_litterman" or "risk_parity" and ignored both: with
  max_weight=0.2 and allow_short=False, Black-Litterman returned 1.787 and
  -1.120. Neither method reads them, so they are now refused, naming the
  method.
- `get_portfolio_risk_attribution` reported `portfolio_pc_exposures` as raw
  weights times eigenvectors of the CORRELATION matrix -- a number in no
  unit. It is now each component's share of the portfolio's variance.
- The construction tools' results were plain models, so a null number
  arrived with nothing saying why. They now explain each null.
- `get_liquidity_metrics` reported 0.0 -- the most liquid reading either
  proxy can give -- when it could not estimate one, and could then name
  that ticker the most liquid.

See the CHANGELOG entry of 2026-10-22.
"""

from __future__ import annotations

import math
import typing
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pydantic
import pytest

from standard_quant_tools.agent.models import (
    LiquidityAnalysisInput,
    PortfolioOptimizationInput,
    RiskAttributionInput,
)
from standard_quant_tools.agent.runtimes import resolve
from standard_quant_tools.agent.runtimes._json_safe import ExplainsNulls
from standard_quant_tools.agent.runtimes.portfolio import construction_tools as CT
from standard_quant_tools.agent.runtimes.portfolio.tools import (
    get_liquidity_metrics,
    get_portfolio_risk_attribution,
)
from standard_quant_tools.data.factory import DataFactory

BASE = dict(
    tickers=["AAA", "BBB", "CCC"], start_date="2022-01-03", end_date="2024-01-01"
)
VIEWS = [{"assets": {"AAA": 1.0, "CCC": -1.0}, "view_return": 0.30}]


def _patch_provider(monkeypatch, frames):
    provider = MagicMock()
    provider.get_ohlcv.side_effect = lambda s, *a, **k: frames[s]
    provider.get_ohlcv_async = AsyncMock(side_effect=lambda s, *a, **k: frames[s])
    monkeypatch.setattr(DataFactory, "get_provider", lambda *a, **k: provider)
    return provider


def _ohlcv(returns, start="2022-01-03"):
    close = 100.0 * np.cumprod(np.r_[1.0, 1.0 + np.asarray(returns, dtype=float)])
    idx = pd.date_range(start, periods=len(close), freq="B")
    return pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.002,
            "Low": close * 0.998,
            "Close": close,
            "Volume": np.full(len(close), 1e6),
        },
        index=idx,
    )


# ── bounds a method does not read ───────────────────────────────────────


class TestBoundsAreRefusedWhereTheyAreIgnored:
    @pytest.mark.parametrize("method", ["black_litterman", "risk_parity"])
    @pytest.mark.parametrize(
        "bound",
        [dict(max_weight=0.2), dict(allow_short=False), dict(allow_short=True)],
        ids=["max_weight", "allow_short_false", "allow_short_true"],
    )
    def test_the_schema_refuses_them_naming_the_method(self, method, bound):
        extra = dict(views=VIEWS) if method == "black_litterman" else {}
        with pytest.raises(pydantic.ValidationError, match=f"method='{method}'"):
            PortfolioOptimizationInput(**BASE, method=method, **bound, **extra)

    def test_dispatch_refuses_the_probe_that_returned_1787(self):
        """The call that came back with weights of 1.787 and -1.120 is
        refused before any data is fetched."""
        with pytest.raises(pydantic.ValidationError, match="max_weight=0.2") as caught:
            resolve("portfolio").dispatch(
                "run_portfolio_optimization",
                {
                    **BASE,
                    "method": "black_litterman",
                    "views": VIEWS,
                    "max_weight": 0.2,
                    "allow_short": False,
                },
            )
        assert "allow_short=False" in str(caught.value)

    @pytest.mark.parametrize("method", ["black_litterman", "risk_parity"])
    def test_without_them_the_methods_are_accepted(self, method):
        extra = dict(views=VIEWS) if method == "black_litterman" else {}
        PortfolioOptimizationInput(**BASE, method=method, **extra)

    @pytest.mark.parametrize("method", ["min_volatility", "max_sharpe"])
    def test_the_mean_variance_methods_still_take_them(self, method):
        model = PortfolioOptimizationInput(
            **BASE, method=method, max_weight=0.5, allow_short=False
        )
        assert model.max_weight == 0.5


# ── principal-component shares of the portfolio's variance ──────────────


def _exact_daily(vols, rho, n=2000, seed=5):
    """Two daily return series with exactly these sample volatilities and
    this sample correlation (ddof=1)."""
    z = np.random.default_rng(seed).normal(size=(n, 2))
    z -= z.mean(axis=0)
    z = z @ np.linalg.inv(np.linalg.cholesky(np.cov(z, rowvar=False))).T
    corr = np.array([[1.0, rho], [rho, 1.0]])
    return (z @ np.linalg.cholesky(corr).T) * np.asarray(vols)


class TestPcExposuresAreSharesOfThePortfolioVariance:
    """THE PLANTED BOOK: two names at 1% and 10% daily volatility,
    correlated 0.5, held 50/50. The correlation matrix's components are
    (1, 1)/sqrt(2) with eigenvalue 1.5 and (1, -1)/sqrt(2) with 0.5. In
    those units the book is x = (0.005, 0.05), its variance x'Rx = 0.002775,
    and

        PC1  (0.055^2 / 2) * 1.5 / 0.002775 = 0.817568
        PC2  (0.045^2 / 2) * 0.5 / 0.002775 = 0.182432

    The old figure for PC2 was w'v_2 = (0.5 - 0.5)/sqrt(2) = 0.0: a
    component carrying 18% of the book's variance reported as no exposure.
    """

    def test_the_planted_shares(self, monkeypatch):
        returns = _exact_daily([0.01, 0.10], 0.5)
        _patch_provider(
            monkeypatch,
            {
                "LO": _ohlcv(returns[:, 0]),
                "HI": _ohlcv(returns[:, 1]),
                "SPY": _ohlcv(np.random.default_rng(1).normal(0, 0.01, 2000)),
            },
        )
        result = get_portfolio_risk_attribution(
            RiskAttributionInput(
                tickers=["LO", "HI"],
                weights=[0.5, 0.5],
                start_date="2022-01-03",
                end_date="2030-01-01",
                benchmark="SPY",
                n_components=2,
            )
        )
        shares = result.portfolio_pc_exposures
        assert shares["PC2"] == pytest.approx(0.182432, abs=1e-4)
        assert shares["PC1"] == pytest.approx(0.817568, abs=1e-4)
        assert shares["PC1"] + shares["PC2"] == pytest.approx(1.0, abs=2e-4)
        # The universe's own split, beside it, is a different number.
        assert result.pca_variance_explained["PC2"] == pytest.approx(0.25, abs=1e-4)


# ── construction results say why a number is null ──────────────────────


def _returned_model(fn):
    return typing.get_type_hints(fn)["return"]


class TestConstructionNullsCarryReasons:
    @pytest.mark.parametrize("tool", sorted(CT.CONSTRUCTION_TOOL_DISPATCH))
    def test_every_construction_result_explains_its_nulls(self, tool):
        fn, _ = CT.CONSTRUCTION_TOOL_DISPATCH[tool]
        assert issubclass(_returned_model(fn), ExplainsNulls), tool

    def test_a_factor_variance_with_no_factor_covariance(self):
        result = CT.get_factor_exposure_budget(
            CT.FactorExposureInput(
                weights={"A": 0.5, "B": 0.5},
                factor_loadings={"A": {"mkt": 1.0}, "B": {"mkt": 0.8}},
            )
        )
        assert result.factor_variance is None
        assert any(
            w.startswith("factor_variance is null: no factor_covariance")
            for w in result.warnings
        )

    def test_a_top_share_of_a_book_too_small_for_it(self):
        result = CT.analyze_concentration(
            CT.ConcentrationInput(weights={"A": 0.6, "B": 0.4})
        )
        assert result.top_1_share == pytest.approx(0.6)
        assert result.top_3_share is None
        assert "top_3_share is null: the book holds fewer than 3 positions." in (
            result.warnings
        )
        assert "top_10_share is null: the book holds fewer than 10 positions." in (
            result.warnings
        )

    def test_a_scenario_with_no_covariance_to_size_it(self):
        result = CT.run_portfolio_scenarios(
            CT.PortfolioScenariosInput(
                weights={"A": 0.5, "B": 0.5},
                scenarios={"crash": {"A": -0.2, "B": -0.1}},
            )
        )
        assert result.portfolio_volatility is None
        assert result.by_scenario[0].sigma_move is None
        text = " ".join(result.warnings)
        assert "portfolio_volatility is null: no covariance was supplied" in text
        assert "by_scenario[0].sigma_move is null: no covariance" in text

    def test_a_zero_position_in_a_liquidity_var(self):
        result = CT.get_liquidity_adjusted_var(
            CT.LiquidityVarInput(
                positions={"A": 0.0, "B": 1e6},
                volatilities={"A": 0.2, "B": 0.3},
                daily_volumes={"A": 1e7, "B": 1e7},
            )
        )
        (row,) = [r for r in result.by_position if r.asset == "A"]
        assert row.adjustment_multiple is None
        assert any(
            "adjustment_multiple is null: the position is zero" in w
            for w in result.warnings
        )


# ── liquidity proxies that cannot be estimated ──────────────────────────


class TestLiquidityProxiesAreNullNotZero:
    def _frames(self):
        rng = np.random.default_rng(3)
        return {
            "DEEP": _ohlcv(rng.normal(0, 0.01, 120)),
            "NEW": _ohlcv(rng.normal(0, 0.03, 8)),
        }

    def test_a_ticker_with_too_few_bars_is_null_and_not_the_most_liquid(
        self, monkeypatch
    ):
        """NEW has 9 bars for a 20-bar window. It used to read 0.0 on both
        proxies and come back as `most_liquid_ticker`."""
        _patch_provider(monkeypatch, self._frames())
        result = get_liquidity_metrics(
            LiquidityAnalysisInput(
                tickers=["DEEP", "NEW"], start_date="2022-01-03", end_date="2023-01-01"
            )
        )
        assert result.per_ticker["NEW"]["amihud_illiquidity"] is None
        assert result.per_ticker["NEW"]["corwin_schultz_spread_bps"] is None
        assert result.per_ticker["DEEP"]["amihud_illiquidity"] > 0
        assert result.most_liquid_ticker == "DEEP"
        assert result.least_liquid_ticker == "DEEP"
        text = " ".join(result.warnings)
        assert "per_ticker.NEW.amihud_illiquidity is null" in text
        assert "per_ticker.NEW.corwin_schultz_spread_bps is null" in text
        assert "['NEW'] could not be ranked" in text

    def test_with_no_estimate_at_all_the_ranking_is_null(self, monkeypatch):
        _patch_provider(monkeypatch, self._frames())
        out = resolve("portfolio").dispatch(
            "get_liquidity_metrics",
            {"tickers": ["NEW"], "start_date": "2022-01-03", "end_date": "2023-01-01"},
        )
        assert out["most_liquid_ticker"] is None
        assert out["least_liquid_ticker"] is None
        assert out["per_ticker"]["NEW"]["amihud_illiquidity"] is None
        assert math.isfinite(out["per_ticker"]["NEW"]["avg_dollar_volume"])
