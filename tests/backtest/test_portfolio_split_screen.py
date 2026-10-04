"""
The portfolio engine screens for unadjusted splits, as run_strategy does.

It compounds every held ticker's bar return, so an unadjusted split is a real
-50% (or -90%) bar to it too. A 10:1 split injected into one of two held
tickers moved final equity by -43.9%, and the only caveat in the result was
"cash went negative". Both engines now share one screen.
"""

import numpy as np
import pandas as pd
import pytest

import standard_quant_tools.backtest.portfolio_engine as portfolio_engine
from standard_quant_tools.backtest.portfolio_engine import run_portfolio_simulation


@pytest.fixture(params=["native", "python"])
def engine_path(request, monkeypatch):
    if request.param == "python":
        monkeypatch.setattr(portfolio_engine, "_native_portfolio_sim", lambda **_: None)
    return request.param


DATES = pd.date_range("2014-01-02", periods=300, freq="B")


def _bars(seed: int, drop_at: int = None, factor: float = 1.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0003, 0.01, len(DATES)))
    if drop_at is not None:
        close[drop_at:] = close[drop_at:] * factor
    return pd.DataFrame(
        {"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1e6},
        index=DATES,
    )


def _monthly(weights) -> pd.DataFrame:
    dates = DATES[::21]
    return pd.DataFrame([weights] * len(dates), index=dates)


def _screen(result):
    return [w for w in result["warnings"] if "SPLIT SCREEN" in w]


class TestTheSplitScreenRunsInThePortfolioEngine:
    def test_a_split_in_a_held_ticker_is_named(self, engine_path):
        prices = {"A": _bars(1, drop_at=250, factor=0.1), "B": _bars(2)}
        result = run_portfolio_simulation(
            prices, _monthly({"A": 0.5, "B": 0.5}), commission_pct=0.0, slippage_pct=0.0
        )
        (warning,) = _screen(result)
        assert warning.startswith("A: SPLIT SCREEN")
        assert str(DATES[250].date()) in warning
        assert "not known" in warning

    def test_the_provider_flag_phrases_it(self, engine_path):
        split = _bars(1, drop_at=250, factor=0.5)
        split.attrs["adjusted"] = False
        prices = {"A": split, "B": _bars(2)}
        (warning,) = _screen(
            run_portfolio_simulation(prices, _monthly({"A": 0.5, "B": 0.5}))
        )
        assert "adjusted=False" in warning
        (warning,) = _screen(
            run_portfolio_simulation(
                prices, _monthly({"A": 0.5, "B": 0.5}), adjusted=True
            )
        )
        assert "adjusted=True" in warning

    def test_an_ordinary_large_move_is_not_screened(self, engine_path):
        """Null case: -22% is under the threshold and smaller than the
        26% to 35% fall a 3:2 split makes. (The -28% this case used is now
        named as 3:2-sized; see the CHANGELOG entry of 2026-10-04.)"""
        prices = {"A": _bars(1, drop_at=250, factor=0.78), "B": _bars(2)}
        assert not _screen(
            run_portfolio_simulation(prices, _monthly({"A": 0.5, "B": 0.5}))
        )

    def test_a_fall_the_size_of_a_3_2_split_is_screened(self, engine_path):
        """-28% is under the threshold but within 10% (log scale) of a 3:2
        split's -33%: named, prefixed with the ticker."""
        prices = {"A": _bars(1, drop_at=250, factor=0.72), "B": _bars(2)}
        (warning,) = _screen(
            run_portfolio_simulation(prices, _monthly({"A": 0.5, "B": 0.5}))
        )
        assert warning.startswith("A: SPLIT SCREEN: 1 bar(s)")
        assert "near a 3:2 split" in warning

    def test_a_split_in_a_ticker_never_held_is_not_screened(self, engine_path):
        """Null case: a name that never carries a weight cannot move equity."""
        prices = {"A": _bars(1, drop_at=250, factor=0.1), "B": _bars(2)}
        assert not _screen(
            run_portfolio_simulation(prices, _monthly({"A": 0.0, "B": 1.0}))
        )

    def test_both_engines_word_it_identically(self):
        """One screen: the portfolio warning is run_strategy's, prefixed."""
        from standard_quant_tools.backtest.engine import run_strategy

        split = _bars(1, drop_at=250, factor=0.1)
        single = [
            w
            for w in run_strategy(split, pd.Series(1.0, index=DATES))["warnings"]
            if "SPLIT SCREEN" in w
        ]
        (portfolio,) = _screen(
            run_portfolio_simulation(
                {"A": split, "B": _bars(2)}, _monthly({"A": 0.5, "B": 0.5})
            )
        )
        assert portfolio == f"A: {single[0]}"
