"""
The `optimize_portfolio` tool: registered where the portfolio tools are,
reading the covariance in each of its three forms with the same answer to
the bit, refusing what it would otherwise ignore, and saying why a number
is null.

The arithmetic is held in `tests/portfolio/test_general_optimizer.py`
against closed forms and duality; here the door is held to the library: the
tool's weights are the library function's, to the bit, for the same
numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pydantic
import pytest

import standard_quant_tools.agent as package
from standard_quant_tools.agent.runtimes import resolve
from standard_quant_tools.agent.runtimes.portfolio import (
    TOOL_CATEGORY,
    TOOL_DEFS,
    TOOL_DISPATCH,
)
from standard_quant_tools.agent.runtimes.portfolio import tools as portfolio_tools
from standard_quant_tools.agent.runtimes.portfolio.optimizer_tools import (
    OptimizePortfolioInput,
    optimize_portfolio,
)
from standard_quant_tools.error import ValidationError
from standard_quant_tools.mcp.catalog import THIN_DESCRIPTION_LIMIT, first_sentence
from standard_quant_tools.portfolio import general_optimizer

ASSETS = ["AAA", "BBB", "CCC", "DDD"]


@pytest.fixture(scope="module")
def portfolio():
    return resolve("portfolio")


def _covariance():
    rng = np.random.default_rng(31)
    X = rng.normal(0, 0.01, (300, 4)) + rng.normal(0, 0.008, (300, 1))
    return np.cov(X, rowvar=False) * 252


MU = {"AAA": 0.06, "BBB": 0.09, "CCC": 0.04, "DDD": 0.07}


class TestRegistration:
    def test_it_is_a_portfolio_risk_tool_and_dispatches(self):
        assert TOOL_CATEGORY["optimize_portfolio"] == "portfolio_risk"
        handler, model = TOOL_DISPATCH["optimize_portfolio"]
        assert handler is optimize_portfolio and model is OptimizePortfolioInput
        assert "optimize_portfolio" in {name for name, _d, _m in TOOL_DEFS}
        assert "optimize_portfolio" in package.__all__
        assert package.optimize_portfolio is optimize_portfolio

    def test_its_first_sentence_stands_alone_when_thinned(self):
        """Under `auto` the portfolio runtime lists its most expensive tools
        by their first sentence alone, and this is one of them."""
        (description,) = [
            d for name, d, _m in TOOL_DEFS if name == "optimize_portfolio"
        ]
        sentence = first_sentence(description)
        assert len(sentence) <= THIN_DESCRIPTION_LIMIT
        for word in (
            "constrained",
            "expected returns",
            "covariance",
            "Sharpe",
            "tracking error",
        ):
            assert word in sentence


class TestTheDoorIsTheLibrary:
    def test_rows_and_the_map_give_the_librarys_bits(self, portfolio):
        cov = _covariance()
        frame = pd.DataFrame(cov, index=ASSETS, columns=ASSETS)
        expected = general_optimizer.optimize_portfolio(
            frame,
            expected_returns=MU,
            objective="max_utility",
            risk_aversion=3.0,
            upper_bounds=0.45,
            groups={"g": ["AAA", "BBB"]},
            group_max={"g": 0.5},
        )
        common = {
            "expected_returns": MU,
            "objective": "max_utility",
            "risk_aversion": 3.0,
            "max_weight": 0.45,
            "groups": {"g": ["AAA", "BBB"]},
            "group_max": {"g": 0.5},
        }
        rows = portfolio.dispatch(
            "optimize_portfolio",
            {"assets": ASSETS, "covariance": cov.tolist(), **common},
        )
        mapped = portfolio.dispatch(
            "optimize_portfolio",
            {
                "assets": ASSETS,
                "covariance": {
                    a: dict(zip(ASSETS, row)) for a, row in zip(ASSETS, cov.tolist())
                },
                **common,
            },
        )
        for out in (rows, mapped):
            assert out["certified"] is True
            assert out["weights"] == expected["weights"]
            assert out["binding_constraints"] == expected["binding_constraints"]

    def test_the_covariance_reference_gives_the_inline_answer(
        self, portfolio, tmp_path, monkeypatch
    ):
        """estimate_covariance's published matrix, passed as covariance_ref,
        is the same optimization as that matrix inline."""
        monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path / "runs"))
        rng = np.random.default_rng(11)
        index = pd.bdate_range("2024-01-01", periods=250)
        frame = pd.DataFrame(rng.normal(0, 0.01, (250, 4)), index=index, columns=ASSETS)
        monkeypatch.setattr(
            portfolio_tools, "fetch_returns_sync", lambda *a, **k: frame
        )
        est = portfolio.dispatch(
            "estimate_covariance",
            {
                "tickers": ASSETS,
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "run_id": "opt",
            },
        )
        matrix = est["matrix"]
        names = est["assets"]
        via_ref = portfolio.dispatch(
            "optimize_portfolio",
            {"covariance_ref": est["covariance_ref"], "max_weight": 0.4},
        )
        inline = portfolio.dispatch(
            "optimize_portfolio",
            {"assets": names, "covariance": matrix, "max_weight": 0.4},
        )
        assert via_ref["certified"] is True
        assert via_ref["weights"] == inline["weights"]


class TestRefusals:
    def test_risk_aversion_is_refused_where_the_objective_ignores_it(self):
        with pytest.raises(
            pydantic.ValidationError, match="risk_aversion is read only"
        ):
            OptimizePortfolioInput(
                assets=ASSETS, covariance=_covariance().tolist(), risk_aversion=2.0
            )

    def test_a_dumped_input_validates_again(self):
        """The refusal is of a value given, not of a default: an input
        written back out with its defaults is accepted as it was."""
        first = OptimizePortfolioInput(assets=ASSETS, covariance=_covariance().tolist())
        again = OptimizePortfolioInput(**first.model_dump(exclude_none=True))
        assert again.risk_aversion is None and again.objective == "min_variance"

    def test_group_limits_without_groups_are_refused(self):
        with pytest.raises(pydantic.ValidationError, match="`groups`"):
            OptimizePortfolioInput(
                assets=ASSETS, covariance=_covariance().tolist(), group_max={"g": 0.3}
            )

    def test_a_non_finite_forecast_is_refused_by_name(self):
        with pytest.raises(
            pydantic.ValidationError, match="expected_returns is not finite"
        ):
            OptimizePortfolioInput(
                assets=ASSETS,
                covariance=_covariance().tolist(),
                expected_returns={**MU, "CCC": float("nan")},
                objective="max_utility",
            )

    def test_a_bound_for_an_asset_not_in_the_covariance(self, portfolio):
        with pytest.raises(ValidationError, match="ZZZ"):
            portfolio.dispatch(
                "optimize_portfolio",
                {
                    "assets": ASSETS,
                    "covariance": _covariance().tolist(),
                    "upper_bounds": {"ZZZ": 0.1},
                },
            )

    def test_infeasible_limits_name_the_conflict(self, portfolio):
        with pytest.raises(ValidationError, match="cannot all hold"):
            portfolio.dispatch(
                "optimize_portfolio",
                {
                    "assets": ASSETS,
                    "covariance": _covariance().tolist(),
                    "max_weight": 0.2,
                },
            )


class TestNullsSayWhy:
    def test_numbers_that_need_an_input_not_given(self, portfolio):
        out = portfolio.dispatch(
            "optimize_portfolio",
            {"assets": ASSETS, "covariance": _covariance().tolist()},
        )
        assert out["certified"] is True
        assert out["tracking_error"] is None and out["active_weights"] is None
        assert out["expected_return"] is None and out["sharpe_ratio"] is None
        joined = " ".join(out["warnings"])
        assert "tracking_error is null: no benchmark_weights were given" in joined
        assert "expected_return is null: no expected_returns were given" in joined
        # min_variance takes no risk aversion, and says nothing about one.
        assert out["risk_aversion"] is None
        assert "risk_aversion" not in joined

    def test_a_volatility_cap_on_a_utility_reports_the_risk_aversion_it_implies(
        self, portfolio
    ):
        cov = _covariance()
        free = portfolio.dispatch(
            "optimize_portfolio",
            {
                "assets": ASSETS,
                "covariance": cov.tolist(),
                "expected_returns": MU,
                "objective": "max_utility",
                "risk_aversion": 0.5,
            },
        )
        capped = portfolio.dispatch(
            "optimize_portfolio",
            {
                "assets": ASSETS,
                "covariance": cov.tolist(),
                "expected_returns": MU,
                "objective": "max_utility",
                "risk_aversion": 0.5,
                "max_volatility": 0.9 * free["volatility"],
            },
        )
        assert capped["certified"] is True
        assert capped["volatility"] == pytest.approx(0.9 * free["volatility"], rel=1e-9)
        assert capped["effective_risk_aversion"] > capped["risk_aversion"] == 0.5
        assert capped["binding_constraints"][0]["constraint"] == "max_volatility"
