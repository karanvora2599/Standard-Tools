"""
Every consumer that compounds bar returns says which returns they were.

A total-return series and a price-return series of the same stock differ by
its dividends, and nothing in a backtest's numbers says which one it
compounded: yfinance serves total-return bars, Polygon and Databento
price-return bars, and the old single `adjusted=True` hid the difference
(see the CHANGELOG entry of 2026-10-22). Each door below -- the two engines,
the two backtest tools on them, and the dataset build -- now echoes a
`return_basis` read off the bars' own stamps, and the ones that take several
inputs say so when the inputs disagree.

The bars are stamped by the real providers' own `get_metadata`, the way a
provider stamps what it serves.
"""

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.agent.models import BuyAndHoldInput, PortfolioSimulationInput
from standard_quant_tools.agent.runtimes.backtest.tools import (
    run_buy_and_hold,
)
from standard_quant_tools.agent.runtimes.backtest.tools import (
    run_portfolio_simulation as run_portfolio_simulation_tool,
)
from standard_quant_tools.backtest.engine import run_strategy
from standard_quant_tools.backtest.portfolio_engine import run_portfolio_simulation
from standard_quant_tools.data.adjustment import stamp_adjustment
from standard_quant_tools.data.factory import DataFactory
from standard_quant_tools.data.polygon_provider import PolygonProvider
from standard_quant_tools.data.yfinance_provider import YFinanceProvider
from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.dataset.builder import build_dataset
from standard_quant_tools.modeling.specs import DatasetSpec, FeatureSpec, TargetSpec

N_BARS = 400
START, END = "2022-01-03", "2023-07-31"
EX_DATE = "2022-06-01"
DIVIDEND = 0.02


def _total_return_bars(symbol: str) -> pd.DataFrame:
    seed = sum(ord(c) * 31**i for i, c in enumerate(symbol)) % (2**32)
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0.0004, 0.01, N_BARS))
    return pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.002,
            "Low": close * 0.998,
            "Close": close,
            "Volume": np.full(N_BARS, 1_000_000.0),
        },
        index=pd.bdate_range(START, periods=N_BARS),
    )


def _price_return_bars(symbol: str) -> pd.DataFrame:
    """The same stock as a split-only feed serves it: a 2% dividend stays
    in the price as the ex-date drop."""
    out = _total_return_bars(symbol)
    after = out.index >= pd.Timestamp(EX_DATE)
    for column in ("Open", "High", "Low", "Close"):
        out.loc[after, column] = out.loc[after, column] * (1 - DIVIDEND)
    return out


def _yahoo(frame: pd.DataFrame) -> pd.DataFrame:
    return stamp_adjustment(frame, YFinanceProvider().get_metadata("X"))


def _polygon(frame: pd.DataFrame) -> pd.DataFrame:
    return stamp_adjustment(frame, PolygonProvider(api_key="k").get_metadata("X"))


class _Stub:
    """A provider serving the frames given, sliced to the window, attrs
    and all. `metadata` is what get_metadata answers (None: it fails)."""

    def __init__(self, frames: Dict[str, pd.DataFrame], metadata=None) -> None:
        self.frames = frames
        self.metadata = metadata

    def get_ohlcv(self, symbol, start, end, interval="1d"):
        frame = self.frames[symbol]
        window = (frame.index >= pd.Timestamp(start)) & (
            frame.index <= pd.Timestamp(end)
        )
        return frame.loc[window].copy()

    async def get_ohlcv_async(self, symbol, start, end, interval="1d"):
        return self.get_ohlcv(symbol, start, end, interval)

    def get_metadata(self, symbol, interval="1d"):
        if self.metadata is None:
            raise RuntimeError("metadata unavailable")
        return self.metadata


@pytest.fixture
def serve(monkeypatch):
    def _serve(frames, metadata=None):
        stub = _Stub(frames, metadata)
        monkeypatch.setattr(DataFactory, "get_provider", lambda *a, **k: stub)
        return stub

    return _serve


def _weights(tickers) -> pd.DataFrame:
    first = pd.bdate_range(START, periods=N_BARS)[0]
    return pd.DataFrame({t: [1.0 / len(tickers)] for t in tickers}, index=[first])


# ── The doors ───────────────────────────────────────────────────────────
# Each takes {ticker: frame} and the provider metadata, and answers
# (return_basis, warnings).


def _engine(frames, metadata, serve) -> Tuple[str, Any]:
    ((_, frame),) = frames.items()
    out = run_strategy(
        frame, pd.Series(1.0, index=frame.index), commission_pct=0.0, slippage_pct=0.0
    )
    return out["return_basis"], out["warnings"]


def _portfolio_engine(frames, metadata, serve):
    out = run_portfolio_simulation(frames, _weights(list(frames)))
    return out["return_basis"], out["warnings"]


def _buy_and_hold_tool(frames, metadata, serve):
    ((symbol, _),) = frames.items()
    serve(frames, metadata)
    out = run_buy_and_hold(
        BuyAndHoldInput(symbol=symbol, start_date=START, end_date=END)
    )
    return out.return_basis, out.warnings


def _portfolio_tool(frames, metadata, serve):
    serve(frames, metadata)
    first = str(pd.bdate_range(START, periods=1)[0].date())
    out = run_portfolio_simulation_tool(
        PortfolioSimulationInput(
            tickers=list(frames),
            start_date=START,
            end_date=END,
            target_weights={t: {first: 1.0 / len(frames)} for t in frames},
        )
    )
    return out.return_basis, out.warnings


def _dataset_build(frames, metadata, serve):
    entities = dict(frames)
    entities.setdefault("SPY", next(iter(frames.values())))
    serve(entities, metadata)
    spec = DatasetSpec(
        universe=list(frames),
        start=START,
        end=END,
        features=[FeatureSpec(id="market.momentum", params={"lookback": 20})],
        target=TargetSpec(horizon=5),
        benchmark="SPY",
        provider="polygon",
    )
    built = build_dataset(spec)
    return built["return_basis"], built["warnings"]


SINGLE_INPUT_DOORS = {
    "run_strategy": _engine,
    "run_buy_and_hold": _buy_and_hold_tool,
}
MULTI_INPUT_DOORS = {
    "run_portfolio_simulation (engine)": _portfolio_engine,
    "run_portfolio_simulation (tool)": _portfolio_tool,
    "build_dataset": _dataset_build,
}
EVERY_DOOR = {**SINGLE_INPUT_DOORS, **MULTI_INPUT_DOORS}


class TestEveryDoorEchoesTheBasis:
    @pytest.mark.parametrize("door", sorted(EVERY_DOOR))
    def test_price_return_bars_are_said_to_be_price_returns(self, door, serve):
        """Bars from a split-only feed: every door says price_return, where
        none of them said anything."""
        polygon_meta = PolygonProvider(api_key="k").get_metadata("X")
        frames = {"AAA": _polygon(_price_return_bars("AAA"))}
        if door in MULTI_INPUT_DOORS:
            frames["BBB"] = _polygon(_price_return_bars("BBB"))
        basis, warnings = EVERY_DOOR[door](frames, polygon_meta, serve)
        assert basis == "price_return"
        assert not any("RETURN BASIS MIXED" in w for w in warnings)

    @pytest.mark.parametrize("door", sorted(EVERY_DOOR))
    def test_total_return_bars_are_said_to_be_total_returns(self, door, serve):
        yahoo_meta = YFinanceProvider().get_metadata("X")
        frames = {"AAA": _yahoo(_total_return_bars("AAA"))}
        if door in MULTI_INPUT_DOORS:
            frames["BBB"] = _yahoo(_total_return_bars("BBB"))
        basis, _ = EVERY_DOOR[door](frames, yahoo_meta, serve)
        assert basis == "total_return"

    @pytest.mark.parametrize("door", sorted(MULTI_INPUT_DOORS))
    def test_inputs_on_two_bases_are_mixed_and_named(self, door, serve):
        """One name from a total-return feed beside one from a split-only
        feed: the result's return is partly dividend policy, and the door
        says which name is which. The build reads the frames' stamps when
        the provider's metadata is unavailable."""
        frames = {
            "AAA": _yahoo(_total_return_bars("AAA")),
            "BBB": _polygon(_price_return_bars("BBB")),
        }
        basis, warnings = MULTI_INPUT_DOORS[door](frames, None, serve)
        assert basis == "mixed"
        (warning,) = [w for w in warnings if "RETURN BASIS MIXED" in w]
        assert "total_return (dividends in the return): AAA" in warning
        assert "price_return (dividends left out): BBB" in warning


class TestWhyTheBasisMatters:
    def test_the_two_bases_differ_by_exactly_the_dividend(self):
        """The same stock, held through a 2% dividend at zero cost: the
        price-return run ends 2% below the total-return run, and only
        return_basis tells the two results apart."""
        total = run_strategy(
            _yahoo(_total_return_bars("AAA")),
            pd.Series(1.0, index=pd.bdate_range(START, periods=N_BARS)),
            commission_pct=0.0,
            slippage_pct=0.0,
        )
        price = run_strategy(
            _polygon(_price_return_bars("AAA")),
            pd.Series(1.0, index=pd.bdate_range(START, periods=N_BARS)),
            commission_pct=0.0,
            slippage_pct=0.0,
        )
        assert (1 + price["total_return"]) / (1 + total["total_return"]) == (
            pytest.approx(1 - DIVIDEND, abs=2e-6)
        )
        assert (total["return_basis"], price["return_basis"]) == (
            "total_return",
            "price_return",
        )


class TestSplitOnlyBarsAreSplitAdjusted:
    """`adjusted` is now the AND of the two flags, so a split-only feed reads
    adjusted=False. The split logic reads `split_adjusted`; reading the AND
    would call Polygon's bars unadjusted and divide a declared split into
    them a second time."""

    def test_the_build_resolves_split_only_bars_as_split_adjusted(self, serve):
        polygon_meta = PolygonProvider(api_key="k").get_metadata("X")
        frames = {"AAA": _polygon(_price_return_bars("AAA"))}
        frames["SPY"] = frames["AAA"]
        serve(frames, polygon_meta)
        spec = DatasetSpec(
            universe=["AAA"],
            start=START,
            end=END,
            features=[FeatureSpec(id="market.momentum", params={"lookback": 20})],
            target=TargetSpec(horizon=5),
            benchmark="SPY",
            provider="polygon",
        )
        built = build_dataset(spec)
        assert built["bars_adjusted"] is True
        assert any("NOT dividend-adjusted" in w for w in built["warnings"])
        assert not any("serves unadjusted bars" in w for w in built["warnings"])

    def test_a_declared_split_on_split_only_bars_is_refused(self, serve):
        polygon_meta = PolygonProvider(api_key="k").get_metadata("X")
        frames = {"AAA": _polygon(_price_return_bars("AAA"))}
        frames["SPY"] = frames["AAA"]
        serve(frames, polygon_meta)
        spec = DatasetSpec(
            universe=["AAA"],
            start=START,
            end=END,
            features=[FeatureSpec(id="market.momentum", params={"lookback": 20})],
            target=TargetSpec(horizon=5),
            benchmark="SPY",
            provider="polygon",
            corporate_actions=[
                {"entity": "AAA", "ex_date": "2022-09-01", "split_ratio": 2.0}
            ],
        )
        with pytest.raises(ValidationError, match="split_adjusted=True"):
            build_dataset(spec)
