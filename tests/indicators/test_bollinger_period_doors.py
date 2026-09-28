"""
A Bollinger period below 2 is refused at every door that admits one.

The bands are a sample standard deviation either side of the mean, and the
sample standard deviation of one bar is 0/0. The native kernel answered
all-NaN for period 1 and pandas a middle band equal to the price, so the
same call had two answers depending on the build and neither was a band.
The library function and the panel refuse it (tests/indicators/
test_indicators_volatility.py and tests/cpp_bindings/
test_missing_bar_parity.py); this pins the tool inputs and the strategy
parameter, and that the strategy catalogue reports the bound it enforces.
"""

import pandas as pd
import pytest
from pydantic import ValidationError as PydanticValidationError

from standard_quant_tools.agent.models import ListStrategiesInput, TechnicalPanelInput
from standard_quant_tools.agent.runtimes.meta.tools import list_strategies
from standard_quant_tools.agent.runtimes.research.reference_tools import (
    IndicatorPanelInput,
)
from standard_quant_tools.backtest.strategies import STRATEGY_REGISTRY
from standard_quant_tools.backtest.strategy_params import resolve_strategy_params
from standard_quant_tools.error import ValidationError

_PANEL = dict(tickers=["AAA", "BBB"], start_date="2024-01-01", end_date="2024-06-28")
_REF = dict(_PANEL, indicators=["bollinger_bands"], run_id="r1", name="bands")


class TestToolInputs:
    @pytest.mark.parametrize(
        "model,base", [(TechnicalPanelInput, _PANEL), (IndicatorPanelInput, _REF)]
    )
    def test_period_one_is_refused_at_the_schema(self, model, base):
        with pytest.raises(PydanticValidationError, match="bollinger_period"):
            model(**base, bollinger_period=1)

    @pytest.mark.parametrize(
        "model,base", [(TechnicalPanelInput, _PANEL), (IndicatorPanelInput, _REF)]
    )
    def test_period_two_and_the_default_are_accepted(self, model, base):
        assert model(**base, bollinger_period=2).bollinger_period == 2
        assert model(**base).bollinger_period == 20


class TestStrategyParameter:
    def test_bollinger_reversion_period_one_is_refused(self):
        with pytest.raises(ValidationError, match="at least 2 bars"):
            resolve_strategy_params("bollinger_reversion", {"period": 1})

    def test_the_registry_refuses_it_too(self):
        frame = pd.DataFrame({"Close": [100.0 + i for i in range(40)]})
        with pytest.raises(ValidationError, match="at least 2"):
            STRATEGY_REGISTRY["bollinger_reversion"](frame, period=1, num_std=2.0)

    def test_null_case_period_two_resolves(self):
        assert (
            resolve_strategy_params("bollinger_reversion", {"period": 2})["period"] == 2
        )

    def test_other_windows_still_start_at_one(self):
        assert (
            resolve_strategy_params("sma_crossover", {"fast_period": 1})["fast_period"]
            == 1
        )

    def test_the_catalogue_reports_the_bound_it_enforces(self):
        result = list_strategies(
            ListStrategiesInput(strategy_type="bollinger_reversion")
        )
        (descriptor,) = result.strategies
        period = next(p for p in descriptor.parameters if p.name == "period")
        assert period.minimum == 2.0
