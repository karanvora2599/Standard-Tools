"""
Three numbers the portfolio layer could not stand behind, and now says so.

A FREE SHORT. `borrow_fee_bps` defaults to 0 and the financing accrual is
gated on it being positive, so a market-neutral book shorted for nothing
unless the caller happened to set a rate. At 50 bps a year, 1.0x short is
about 0.5% of capital annually — the whole edge of many such books.

A RISK SCORE SIZED AS A RETURN FORECAST. `validation/survival.py` sets the
convention: a survival model emits a risk, "higher means sooner".
`predictions_to_score_panel` passes it through unchanged, like a
regressor's, so the largest long goes to the name whose event is expected
soonest. Classification is recentred so that "positive score = the model is
bullish" holds for every task; survival was left out of that, and only the
caller knows whether the event is good or bad.

A REFUSAL THAT POINTED NOWHERE. `uncertainty_scaled` on a classifier said
"train the model with ModelSpec.intervals", and ModelSpec refuses exactly
that — "intervals are calibrated for task='regression' only". The advice
was a closed loop.
"""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.portfolio_eval import (
    predictions_to_score_panel,
    scale_by_uncertainty,
)


class TestAFreeShortIsNamed:
    @staticmethod
    def _run(borrow_fee_bps, short=True):
        from standard_quant_tools.backtest.portfolio_engine import (
            run_portfolio_simulation,
        )

        dates = pd.date_range("2024-01-01", periods=40, freq="B")
        rng = np.random.default_rng(0)
        prices = pd.DataFrame(
            {
                t: 100 * np.cumprod(1 + rng.normal(0, 0.01, len(dates)))
                for t in ("AAA", "BBB")
            },
            index=dates,
        )
        data = {
            t: pd.DataFrame(
                {
                    "Open": prices[t],
                    "High": prices[t] * 1.01,
                    "Low": prices[t] * 0.99,
                    "Close": prices[t],
                    "Volume": 1e7,
                },
                index=dates,
            )
            for t in prices.columns
        }
        rebalances = dates[::10]
        weights = pd.DataFrame(
            {"AAA": 0.5, "BBB": -0.5 if short else 0.5},
            index=rebalances,
        )
        return run_portfolio_simulation(
            data,
            weights,
            initial_capital=100_000.0,
            borrow_fee_bps=borrow_fee_bps,
            fill_price="next_open",
        )

    def test_a_short_book_at_zero_borrow_is_warned_about(self):
        result = self._run(borrow_fee_bps=0.0)
        hits = [w for w in result["warnings"] if "borrow_fee_bps=0" in w]
        assert hits, result["warnings"]
        # The size of the thing being financed for free, not just its name.
        assert "0.50x capital" in hits[0], hits[0]

    def test_a_priced_short_book_is_not_warned_about(self):
        result = self._run(borrow_fee_bps=50.0)
        assert not [w for w in result["warnings"] if "borrow_fee_bps=0" in w]

    def test_a_long_only_book_is_not_warned_about(self):
        """The warning is about shorts, not about the rate being zero."""
        result = self._run(borrow_fee_bps=0.0, short=False)
        assert not [w for w in result["warnings"] if "borrow_fee_bps=0" in w]

    def test_the_free_short_actually_costs_nothing(self):
        """The warning exists because the money is real. Same book, same
        prices, only the borrow rate differs."""
        free = self._run(borrow_fee_bps=0.0)
        priced = self._run(borrow_fee_bps=500.0)
        assert free["equity_curve"].iloc[-1] > priced["equity_curve"].iloc[-1]


class TestTheSurvivalSignIsStated:
    def test_a_risk_score_passes_through_unchanged(self):
        """The mechanism behind the warning, so the warning is not the only
        record of it. Higher risk stays the bigger score, hence the bigger
        long."""
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01"] * 3),
                "entity": ["SOON", "MID", "LATE"],
                "prediction": [0.9, 0.5, 0.1],
            }
        )
        panel = predictions_to_score_panel(frame, "survival")
        assert panel.iloc[0]["SOON"] > panel.iloc[0]["LATE"]

    def test_classification_is_recentred_and_survival_is_not(self):
        """The contrast that makes this a gap rather than a choice: the
        recentring exists so a positive score means bullish for every
        task."""
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01"] * 2),
                "entity": ["A", "B"],
                "prediction": [0.9, 0.1],
            }
        )
        classified = predictions_to_score_panel(frame, "classification")
        survived = predictions_to_score_panel(frame, "survival")
        assert classified.iloc[0]["B"] < 0 < classified.iloc[0]["A"]
        assert (survived.iloc[0] > 0).all()


class TestUncertaintyScaledRefusesByTask:
    def test_the_old_refusal_pointed_at_something_the_spec_forbids(self):
        """Pinned as the reason the task-level refusal exists. The column
        check still fires for a regressor without intervals, where its
        advice IS actionable."""
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01"]),
                "entity": ["A"],
                "prediction": [0.7],
            }
        )
        with pytest.raises(ValidationError, match="ModelSpec.intervals"):
            scale_by_uncertainty(frame, "<regressor without intervals>")

    def test_a_classifier_cannot_be_given_intervals(self):
        """The other half of the closed loop."""
        from standard_quant_tools.modeling.specs import (
            ConformalSpec,
            EstimatorSpec,
            ModelSpec,
            ValidationSpec,
        )
        from pydantic import ValidationError as PydanticValidationError

        with pytest.raises(PydanticValidationError, match="task='regression' only"):
            ModelSpec(
                task="classification",
                estimator=EstimatorSpec(type="random_forest"),
                validation=ValidationSpec(train_window=50, test_window=10),
                intervals=ConformalSpec(),
            )
