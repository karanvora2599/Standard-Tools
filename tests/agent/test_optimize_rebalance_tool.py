"""
The `optimize_rebalance` tool: registered where the portfolio tools are,
its answer the library function's to the bit, its frictionless target
optimize_portfolio's answer, and its refusals made before anything runs.

The arithmetic is held in `tests/portfolio/test_rebalance_optimizer.py`
against a brute force, closed forms and finite differences; here the door
is held to the library, and the target fields to what they claim to
measure.
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
from standard_quant_tools.agent.runtimes.portfolio.rebalance_optimizer_tools import (
    OptimizeRebalanceInput,
    optimize_rebalance,
)
from standard_quant_tools.mcp.catalog import THIN_DESCRIPTION_LIMIT, first_sentence
from standard_quant_tools.portfolio import general_optimizer, rebalance_optimizer

ASSETS = ["AAA", "BBB", "CCC", "DDD"]
MU = {"AAA": 0.06, "BBB": 0.09, "CCC": 0.04, "DDD": 0.07}
HELD = {"AAA": 0.25, "BBB": 0.25, "CCC": 0.25, "DDD": 0.25}


@pytest.fixture(scope="module")
def portfolio():
    return resolve("portfolio")


def _covariance():
    rng = np.random.default_rng(31)
    X = rng.normal(0, 0.01, (300, 4)) + rng.normal(0, 0.008, (300, 1))
    return np.cov(X, rowvar=False) * 252


def _call(portfolio, **over):
    arguments = {
        "assets": ASSETS,
        "covariance": _covariance().tolist(),
        "expected_returns": MU,
        "current_weights": HELD,
        "risk_aversion": 3.0,
        "linear_cost": 0.01,
        "max_weight": 0.6,
    }
    arguments.update(over)
    return portfolio.dispatch("optimize_rebalance", arguments)


class TestRegistration:
    def test_it_is_a_portfolio_risk_tool_and_dispatches(self):
        assert TOOL_CATEGORY["optimize_rebalance"] == "portfolio_risk"
        handler, model = TOOL_DISPATCH["optimize_rebalance"]
        assert handler is optimize_rebalance and model is OptimizeRebalanceInput
        assert "optimize_rebalance" in {name for name, _d, _m in TOOL_DEFS}
        assert "optimize_rebalance" in package.__all__
        assert package.optimize_rebalance is optimize_rebalance

    def test_its_first_sentence_names_what_it_returns(self):
        """A thinned listing keeps the first sentence alone: it must say
        that this is the trade from the weights held, net of costs."""
        (description,) = [
            d for name, d, _m in TOOL_DEFS if name == "optimize_rebalance"
        ]
        sentence = first_sentence(description)
        assert len(sentence) <= THIN_DESCRIPTION_LIMIT
        for word in ("trade", "weights you hold", "trading costs", "turnover"):
            assert word in sentence


class TestTheDoorIsTheLibrary:
    def test_rows_and_the_map_give_the_librarys_bits(self, portfolio):
        cov = _covariance()
        expected = rebalance_optimizer.optimize_rebalance(
            pd.DataFrame(cov, index=ASSETS, columns=ASSETS),
            expected_returns=MU,
            current_weights=HELD,
            risk_aversion=3.0,
            linear_cost=0.01,
            upper_bounds=0.6,
            max_turnover=0.2,
        )
        rows = _call(portfolio, max_turnover=0.2)
        mapped = _call(
            portfolio,
            max_turnover=0.2,
            covariance={
                a: dict(zip(ASSETS, row)) for a, row in zip(ASSETS, cov.tolist())
            },
        )
        for out in (rows, mapped):
            assert out["certified"] is True
            assert out["weights"] == expected["weights"]
            assert out["binding_constraints"] == expected["binding_constraints"]
            assert out["no_trade"] == expected["no_trade"]

    def test_the_covariance_reference_gives_the_inline_answer(
        self, portfolio, tmp_path, monkeypatch
    ):
        """estimate_covariance's published matrix, passed as covariance_ref,
        is the same rebalance as that matrix inline."""
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
                "run_id": "rebalance",
            },
        )
        common = {
            "expected_returns": MU,
            "current_weights": HELD,
            "linear_cost": 0.002,
            "max_weight": 0.5,
        }
        via_ref = portfolio.dispatch(
            "optimize_rebalance", {"covariance_ref": est["covariance_ref"], **common}
        )
        inline = portfolio.dispatch(
            "optimize_rebalance",
            {"assets": est["assets"], "covariance": est["matrix"], **common},
        )
        assert via_ref["certified"] is True
        assert via_ref["weights"] == inline["weights"]


class TestTheTarget:
    def test_the_target_is_the_constrained_optimizers_answer(self, portfolio):
        """The frictionless target -- no costs, no turnover cap -- is
        optimize_portfolio's max_utility under the same limits, and the
        distance, turnover and cost beside it are what they say."""
        out = _call(portfolio, max_turnover=0.2)
        reference = general_optimizer.optimize_portfolio(
            pd.DataFrame(_covariance(), index=ASSETS, columns=ASSETS),
            objective="max_utility",
            expected_returns=MU,
            risk_aversion=3.0,
            upper_bounds=0.6,
        )
        target = np.array([out["target_weights"][a] for a in ASSETS])
        assert (
            np.max(np.abs(target - [reference["weights"][a] for a in ASSETS])) < 1e-12
        )
        w = np.array([out["weights"][a] for a in ASSETS])
        w0 = np.array([HELD[a] for a in ASSETS])
        assert out["distance_to_target"] == pytest.approx(np.sum(np.abs(w - target)))
        assert out["target_turnover"] == pytest.approx(np.sum(np.abs(target - w0)))
        assert out["target_trading_cost"] == pytest.approx(
            0.01 * np.sum(np.abs(target - w0))
        )
        assert out["target_utility"] == pytest.approx(
            reference["objective_value"] - out["target_trading_cost"], rel=1e-12
        )
        gap = w - target
        assert out["tracking_to_target"] == pytest.approx(
            float(np.sqrt(gap @ _covariance() @ gap)), rel=1e-12
        )

    def test_the_answer_is_worth_at_least_the_target_after_costs(self, portfolio):
        """Without a turnover cap the target is a feasible answer of the
        costed problem, so its utility after the cost of reaching it is at
        most the answer's; with a cap that forbids it, it can be more --
        which is what the cap, not the costs, holds back."""
        free = _call(portfolio)
        assert free["target_utility"] <= free["utility"] + 1e-15
        capped = _call(portfolio, max_turnover=0.2)
        assert capped["target_turnover"] > 0.2
        assert capped["target_utility"] > capped["utility"]

    def test_without_costs_the_answer_is_the_target(self, portfolio):
        out = _call(portfolio, linear_cost=0.0)
        assert out["distance_to_target"] < 1e-12
        assert out["target_utility"] == pytest.approx(out["utility"], rel=1e-12)


class TestRefusals:
    @pytest.mark.parametrize(
        "over, words",
        [
            ({"linear_cost": -0.001}, "linear_cost must be between 0 and 1"),
            ({"linear_cost": {"AAA": 2.0}}, "linear_cost must be between 0 and 1"),
            ({"quadratic_cost": 1e9}, "quadratic_cost must be between 0 and 100"),
            (
                {"current_weights": {**HELD, "AAA": float("nan")}},
                "current_weights is not finite",
            ),
            ({"max_turnover": 0.0}, "max_turnover"),
            ({"group_max": {"g": 0.3}}, "`groups`"),
        ],
    )
    def test_the_input_model_refuses_before_anything_runs(self, over, words):
        arguments = {
            "assets": ASSETS,
            "covariance": _covariance().tolist(),
            "expected_returns": MU,
            **over,
        }
        with pytest.raises(pydantic.ValidationError, match=words.replace("(", r"\(")):
            OptimizeRebalanceInput(**arguments)

    def test_a_cost_map_missing_an_asset_is_refused_by_the_library(self, portfolio):
        from standard_quant_tools.error import ValidationError

        with pytest.raises(ValidationError, match="a cost left out is not zero"):
            _call(portfolio, linear_cost={"AAA": 0.001})
