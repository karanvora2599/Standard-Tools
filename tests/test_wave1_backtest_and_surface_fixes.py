"""
The rest of the first wave, each planted.

A callable strategy's grid dropped `risk_free_rate`, so its grid Sharpe
disagreed with its own single run; the feature lab's records could be
neither replayed nor pre-validated; the factor tools reported p-values
without saying what they assume; and two dead names are gone.
"""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.agent.models import (
    FactorRegressionInput,
    ValidateToolCallInput,
)
from standard_quant_tools.agent.runtimes.meta.tools import validate_tool_call
from standard_quant_tools.agent.tools import run_factor_regression
from standard_quant_tools.audit.replay import _resolve_tool
from standard_quant_tools.backtest.engine import backtest_grid, run_strategy
from standard_quant_tools.error import ValidationError


def _prices(n=300, seed=0):
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2022-01-03", periods=n)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, size=n)))
    return pd.DataFrame(
        {
            "Open": close * (1 + rng.normal(0, 0.001, size=n)),
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": np.full(n, 1e6),
        },
        index=index,
    )


class TestTheCallableGridHonoursTheRiskFreeRate:
    def test_grid_and_single_run_agree(self):
        prices = _prices()

        def alternate(df, period):
            signal = pd.Series(0.0, index=df.index)
            signal.iloc[::period] = 1.0
            return signal.ffill()

        grid = backtest_grid(
            prices, alternate, {"period": [7]}, risk_free_rate=0.05, n_workers=1
        )
        single = run_strategy(
            prices,
            alternate(prices, 7),
            initial_capital=10_000.0,
            commission_pct=0.001,
            slippage_pct=0.0005,
            risk_free_rate=0.05,
        )
        assert grid.iloc[0]["sharpe_ratio"] == pytest.approx(single["sharpe_ratio"])
        zero = backtest_grid(prices, alternate, {"period": [7]}, n_workers=1)
        assert zero.iloc[0]["sharpe_ratio"] != pytest.approx(single["sharpe_ratio"])


class TestTheFeatureLabIsASurface:
    def test_its_tools_resolve_for_replay(self):
        for name in ("profile_feature", "run_feature_ablation", "select_features"):
            _fn, model_cls, surface = _resolve_tool(name)
            assert surface == "feature_lab"
            assert hasattr(model_cls, "model_fields")

    def test_its_calls_pre_validate(self):
        result = validate_tool_call(
            ValidateToolCallInput(
                tool_name="profile_feature",
                arguments={"dataset_id": "ds_nope", "feature": "technical.rsi"},
            )
        )
        assert "Unknown tool" not in " ".join(str(p) for p in result.problems)
        # A near miss is refused with the feature-lab name suggested: the
        # third surface is in the catalog the suggestions come from.
        with pytest.raises(ValidationError, match="profile_feature"):
            validate_tool_call(
                ValidateToolCallInput(tool_name="profile_featur", arguments={})
            )


class TestTheFactorToolsSayWhatTheyAssume:
    def test_the_regression_warns_about_its_standard_errors(self, patched_factory):
        result = run_factor_regression(
            FactorRegressionInput(
                symbol="AAPL",
                factor_tickers=["SPY"],
                start_date="2023-01-01",
                end_date="2024-01-01",
            )
        )
        assert any("HAC" in warning for warning in result.warnings)


class TestDeadNamesAreGone:
    def test_they_no_longer_import(self):
        import standard_quant_tools.modeling.validation as validation
        import standard_quant_tools.portfolio.construction as construction

        assert not hasattr(validation, "holdout_split")
        assert not hasattr(construction, "MIN_OBS_PER_PARAMETER")
