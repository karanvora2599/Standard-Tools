"""
The three microstructure tools.

No environment in CI has a tick feed, so these tests supply a stub provider
that does. That is the right shape anyway: the tools' hardest contract is
what they do when the feed is ABSENT, and the most important test in this
file is the one asserting they refuse rather than approximating ticks from
bars.

The stub's tape is built so the answers are arithmetic. The book is a
steady 99.95 / 100.05 -- mid exactly 100.00, spread exactly 10 bps -- so a
fill at the ask costs exactly 10 bps and a sweep to 100.15 costs exactly 30.
"""

import pandas as pd
import pytest

from standard_quant_tools.agent.tools import dispatch
from standard_quant_tools.data.base import DataProvider
from standard_quant_tools.error import ValidationError

BASE = pd.Timestamp("2024-03-01 14:30:00")


def _tape(trade_rows, quote_rows):
    trades = pd.DataFrame(
        {"price": [r[1] for r in trade_rows], "size": [r[2] for r in trade_rows]},
        index=pd.DatetimeIndex([BASE + pd.Timedelta(seconds=r[0]) for r in trade_rows]),
    )
    quotes = pd.DataFrame(
        {
            "bid_price": [r[1] for r in quote_rows],
            "ask_price": [r[2] for r in quote_rows],
            "bid_size": [r[3] for r in quote_rows],
            "ask_size": [r[4] for r in quote_rows],
        },
        index=pd.DatetimeIndex([BASE + pd.Timedelta(seconds=r[0]) for r in quote_rows]),
    )
    return trades, quotes


class _TickProvider(DataProvider):
    """A provider that serves ticks — which is what makes it usable here.
    Capability is probed by override, so this class must genuinely define
    get_trades/get_quotes rather than inheriting the base's refusal."""

    def __init__(self, trades, quotes, bars=None):
        self._trades, self._quotes, self._bars = trades, quotes, bars

    def get_ohlcv(self, symbol, start_date, end_date, interval="1d"):
        if self._bars is None:
            raise AssertionError("this test did not expect a bar fetch")
        return self._bars

    async def get_ohlcv_async(self, symbol, start_date, end_date, interval="1d"):
        return self.get_ohlcv(symbol, start_date, end_date, interval)

    def get_ticker_info(self, symbol):
        raise NotImplementedError

    def get_financial_ratios(self, symbol):
        raise NotImplementedError

    def get_metadata(self, symbol, interval="1d"):
        raise NotImplementedError

    def get_trades(self, symbol, start_date, end_date, limit=None):
        return self._trades if limit is None else self._trades.head(limit)

    def get_quotes(self, symbol, start_date, end_date, limit=None):
        return self._quotes if limit is None else self._quotes.head(limit)


@pytest.fixture
def tick_provider(monkeypatch):
    def _install(trades, quotes, bars=None):
        provider = _TickProvider(trades, quotes, bars)
        monkeypatch.setattr(
            "standard_quant_tools.agent.runtimes.portfolio.tools.DataFactory.get_provider",
            staticmethod(lambda *a, **k: provider),
        )
        return provider

    return _install


@pytest.fixture
def steady_book():
    quotes = [(s, 99.95, 100.05, 500, 500) for s in range(0, 400, 10)]
    trades = [
        (5, 100.05, 100),
        (15, 99.95, 100),
        (25, 100.05, 200),
        (35, 100.15, 50),
    ]
    return _tape(trades, quotes)


class TestFeedIsRequired:
    """The contract that matters most: no feed means no number."""

    def test_a_bar_only_provider_is_refused_by_name(self):
        with pytest.raises(ValidationError) as exc:
            dispatch(
                "get_microstructure_metrics",
                {
                    "symbol": "AAPL",
                    "start": "2024-03-01 14:30:00",
                    "end": "2024-03-01 15:00:00",
                    "source": "yfinance",
                },
            )
        message = str(exc.value)
        assert "no tick feed" in message
        assert "not a substitute" in message
        assert "get_liquidity_metrics" in message

    def test_the_refusal_points_at_the_capability_tool(self):
        for tool, extra in (
            ("get_trade_profile", {}),
            (
                "check_spread_proxy",
                {"bar_start_date": "2024-01-01", "bar_end_date": "2024-03-01"},
            ),
        ):
            with pytest.raises(ValidationError) as exc:
                dispatch(
                    tool,
                    {
                        "symbol": "AAPL",
                        "start": "2024-03-01 14:30:00",
                        "end": "2024-03-01 15:00:00",
                        "source": "yfinance",
                        **extra,
                    },
                )
            assert "describe_data_capabilities" in str(exc.value)

    def test_no_tool_here_ever_falls_back_to_bars(self, monkeypatch):
        """A silent bar fallback would be the worst possible behaviour: a
        fabricated spread that every downstream measure treats as
        measured."""
        called = []
        monkeypatch.setattr(
            "standard_quant_tools.data.yfinance_provider.YFinanceProvider.get_ohlcv",
            lambda self, *a, **k: called.append(1),
        )
        with pytest.raises(ValidationError):
            dispatch(
                "get_microstructure_metrics",
                {
                    "symbol": "AAPL",
                    "start": "2024-03-01 14:30:00",
                    "end": "2024-03-01 15:00:00",
                    "source": "yfinance",
                },
            )
        assert not called, "the tool fetched bars after refusing the feed"


class TestMicrostructureMetrics:
    def _call(self, **overrides):
        return dispatch(
            "get_microstructure_metrics",
            {
                "symbol": "TEST",
                "start": "2024-03-01 14:30:00",
                "end": "2024-03-01 14:40:00",
                "realized_horizon_seconds": None,
                **overrides,
            },
        )

    def test_measures_the_quoted_spread_exactly(self, tick_provider, steady_book):
        tick_provider(*steady_book)
        result = self._call()
        assert result["quoted_spread_bps_mean"] == pytest.approx(10.0)

    def test_the_effective_spread_reflects_the_sweep(self, tick_provider, steady_book):
        """Three fills at the touch and one through it: the size-weighted
        effective spread must sit above the quoted 10 bps."""
        tick_provider(*steady_book)
        result = self._call()
        assert result["effective_spread_bps_size_weighted"] > 10.0

    def test_directional_flow_shows_in_the_buy_fraction(
        self, tick_provider, steady_book
    ):
        tick_provider(*steady_book)
        result = self._call()
        assert 0.0 <= result["buy_volume_fraction"] <= 1.0

    def test_the_realized_split_is_reported_when_asked(self, tick_provider):
        quotes = [(0, 99.95, 100.05, 100, 100), (65, 99.97, 100.07, 100, 100)]
        trades = [(5, 100.05, 100)]
        tick_provider(*_tape(trades, quotes))
        result = self._call(realized_horizon_seconds=60)
        assert result["effective_spread_bps_size_weighted"] == pytest.approx(10.0)
        assert result["realized_spread_bps_size_weighted"] == pytest.approx(
            6.0, abs=1e-3
        )
        assert result["price_impact_bps_size_weighted"] == pytest.approx(4.0, abs=1e-3)

    def test_hitting_the_limit_is_reported_as_one_page(self, tick_provider):
        quotes = [(s, 99.95, 100.05, 100, 100) for s in range(0, 200, 10)]
        trades = [(s, 100.05, 10) for s in range(1, 21)]
        tick_provider(*_tape(trades, quotes))
        result = self._call(limit=20)
        assert any("one page" in note for note in result["notes"])

    def test_dropped_trades_are_counted_and_explained(self, tick_provider):
        """An unclassifiable trade must be visible in the output, not
        quietly missing from the denominator."""
        quotes = [(10, 99.95, 100.05, 100, 100)]
        trades = [(5, 100.00, 100), (15, 100.05, 100)]
        tick_provider(*_tape(trades, quotes))
        result = self._call()
        assert result["n_trades"] == 2
        assert result["n_signed"] < result["n_trades"]
        assert any("could not be classified" in note for note in result["notes"])

    def test_an_empty_window_is_an_error_with_a_reason(self, tick_provider):
        tick_provider(
            pd.DataFrame(columns=["price", "size"], index=pd.DatetimeIndex([])),
            pd.DataFrame(
                columns=["bid_price", "ask_price"], index=pd.DatetimeIndex([])
            ),
        )
        with pytest.raises(ValidationError) as exc:
            self._call()
        assert "market hours" in str(exc.value)

    def test_the_result_is_json_safe(self, tick_provider, steady_book):
        import json

        tick_provider(*steady_book)
        json.dumps(self._call(), allow_nan=False)


class TestTradeProfile:
    def test_block_heavy_volume_is_flagged(self, tick_provider):
        quotes = [(0, 99.95, 100.05, 100, 100)]
        trades = [(i, 100.0, 1) for i in range(1, 91)] + [(95, 100.0, 5_000_000)]
        tick_provider(*_tape(trades, quotes))
        result = dispatch(
            "get_trade_profile",
            {
                "symbol": "TEST",
                "start": "2024-03-01 14:30:00",
                "end": "2024-03-01 14:40:00",
            },
        )
        assert result["largest_bucket_volume_fraction"] > 0.99
        assert any("trades in blocks" in note for note in result["notes"])

    def test_the_intraday_profile_finds_the_busy_bucket(self, tick_provider):
        quotes = [(0, 99.95, 100.05, 100, 100)]
        trades = [(0, 100.0, 10), (60, 100.0, 10), (3_600, 100.0, 10_000)]
        tick_provider(*_tape(trades, quotes))
        result = dispatch(
            "get_trade_profile",
            {
                "symbol": "TEST",
                "start": "2024-03-01 14:30:00",
                "end": "2024-03-01 16:00:00",
                "intraday_freq": "30min",
            },
        )
        assert result["peak_volume_fraction"] > 0.9
        assert result["peak_time"] == "15:30:00"

    def test_size_buckets_are_quantiles_not_a_fixed_grid(self, tick_provider):
        quotes = [(0, 99.95, 100.05, 100, 100)]
        trades = [
            (i, 100.0, size) for i, size in enumerate([1, 5, 20, 100, 500, 9_000])
        ]
        tick_provider(*_tape(trades, quotes))
        result = dispatch(
            "get_trade_profile",
            {
                "symbol": "TEST",
                "start": "2024-03-01 14:30:00",
                "end": "2024-03-01 14:40:00",
                "size_buckets": 3,
            },
        )
        assert len(result["size_buckets"]) == 3
        assert result["size_buckets"][0]["lower"] <= result["median_size"]


class TestSpreadProxyCheck:
    def _bars(self, high_mult=1.001, n=60):
        import numpy as np

        close = pd.Series(
            100.0 + np.linspace(0, 5, n), index=pd.bdate_range("2024-01-02", periods=n)
        )
        return pd.DataFrame(
            {
                "Open": close,
                "High": close * high_mult,
                "Low": close / high_mult,
                "Close": close,
                "Volume": pd.Series(1_000_000.0, index=close.index),
            }
        )

    def test_reports_which_way_the_proxy_errs(self, tick_provider):
        quotes = [(s, 99.95, 100.05, 100, 100) for s in range(0, 200, 10)]
        trades = [(s, 100.05, 100) for s in range(5, 195, 10)]
        tick_provider(*_tape(trades, quotes), bars=self._bars())
        result = dispatch(
            "check_spread_proxy",
            {
                "symbol": "TEST",
                "start": "2024-03-01 14:30:00",
                "end": "2024-03-01 14:40:00",
                "bar_start_date": "2024-01-01",
                "bar_end_date": "2024-03-01",
            },
        )
        assert result["measured_effective_spread_bps"] == pytest.approx(10.0, abs=0.5)
        assert result["verdict"] in {
            "proxy_close",
            "proxy_overstates",
            "proxy_understates",
        }
        assert result["proxy_ratio"] == pytest.approx(
            result["corwin_schultz_spread_bps"]
            / result["measured_effective_spread_bps"],
            abs=1e-3,
        )

    def test_an_understating_proxy_says_backtests_were_optimistic(self, tick_provider):
        """The finding that changes a decision: costs charged from the
        proxy were too low, so reported returns are too good."""
        # A very wide measured spread against near-zero bar ranges.
        quotes = [(s, 99.00, 101.00, 100, 100) for s in range(0, 200, 10)]
        trades = [(s, 101.00, 100) for s in range(5, 195, 10)]
        tick_provider(*_tape(trades, quotes), bars=self._bars(high_mult=1.00001))
        result = dispatch(
            "check_spread_proxy",
            {
                "symbol": "TEST",
                "start": "2024-03-01 14:30:00",
                "end": "2024-03-01 14:40:00",
                "bar_start_date": "2024-01-01",
                "bar_end_date": "2024-03-01",
            },
        )
        assert result["verdict"] == "proxy_understates"
        assert any("charging too little" in note for note in result["notes"])

    def test_it_always_says_the_windows_differ(self, tick_provider):
        """A bar estimator and a tick measurement cannot cover identical
        periods, and pretending otherwise would overstate the comparison."""
        quotes = [(s, 99.95, 100.05, 100, 100) for s in range(0, 200, 10)]
        trades = [(s, 100.05, 100) for s in range(5, 195, 10)]
        tick_provider(*_tape(trades, quotes), bars=self._bars())
        result = dispatch(
            "check_spread_proxy",
            {
                "symbol": "TEST",
                "start": "2024-03-01 14:30:00",
                "end": "2024-03-01 14:40:00",
                "bar_start_date": "2024-01-01",
                "bar_end_date": "2024-03-01",
            },
        )
        assert any("overlapping but not identical" in n for n in result["notes"])

    def test_missing_quotes_means_no_ground_truth(self, tick_provider):
        trades = [(5, 100.05, 100)]
        empty_quotes = pd.DataFrame(
            columns=["bid_price", "ask_price"], index=pd.DatetimeIndex([])
        )
        frames = _tape(trades, [(0, 99.95, 100.05, 1, 1)])
        tick_provider(frames[0], empty_quotes, bars=self._bars())
        with pytest.raises(ValidationError) as exc:
            dispatch(
                "check_spread_proxy",
                {
                    "symbol": "TEST",
                    "start": "2024-03-01 14:30:00",
                    "end": "2024-03-01 14:40:00",
                    "bar_start_date": "2024-01-01",
                    "bar_end_date": "2024-03-01",
                },
            )
        assert "ground truth" in str(exc.value)


class TestADeadMarketIsReportedNotRaised:
    """`detect_liquidity_events` crashed on a flat market.

    `cusum`'s absolutely-constant branch returned `triggered`, `reason`,
    `n_observations` and `n_reference`. `ChannelResult` requires `severity`
    and `peak_statistic`, so building the result raised a validation error
    and the tool returned nothing at all — on the input a liquidity
    detector is most likely to be pointed at.

    `reason` was not carried either: `ChannelResult` has no such field and
    ignores extras, so the explanation was dropped on the way out even when
    it was produced.
    """

    @staticmethod
    def _flat(n=2400, *, blow_out=False):
        """One price, one size, and a spread pinned to the cent."""
        half = [0.25 if (blow_out and i >= int(n * 0.6)) else 0.01 for i in range(n)]
        quotes = [(i, 100.0 - half[i], 100.0 + half[i], 500, 500) for i in range(n)]
        trades = [(i, 100.0, 100) for i in range(n)]
        return _tape(trades, quotes)

    def _run(self, tick_provider, channels, **kwargs):
        trades, quotes = self._flat(**kwargs)
        tick_provider(trades, quotes)
        return dispatch(
            "detect_liquidity_events",
            {
                "symbol": "X",
                "start_date": "2024-03-01",
                "end_date": "2024-03-02",
                "channels": channels,
                "freq": "30s",
            },
        )

    def test_a_flat_market_returns_a_report(self, tick_provider):
        report = self._run(tick_provider, ["spread", "mid_return"])
        assert report["n_triggered"] == 0
        assert report["results"], "no channel reported at all"

    def test_every_channel_that_ran_is_labelled_degenerate(self, tick_provider):
        report = self._run(tick_provider, ["spread", "mid_return"])
        for result in report["results"]:
            assert result["degenerate_baseline"] is True, result["channel"]
            assert result["severity"] == "none"

    def test_the_explanation_reaches_the_caller(self, tick_provider):
        """It lived in a key the result model does not have."""
        report = self._run(tick_provider, ["spread"])
        note = " ".join(report["results"][0]["notes"])
        assert "CONSTANT" in note
        assert "not because nothing moved" in note

    def test_a_blowout_off_a_frozen_baseline_reports_its_shift(self, tick_provider):
        """No statistic is available, so `shift` is the only measure of a
        spread that went from two cents to fifty. It used to be an
        exception."""
        report = self._run(tick_provider, ["spread"], blow_out=True)
        result = report["results"][0]
        assert result["triggered"] is False
        assert result["degenerate_baseline"] is True
        assert result["shift"] > 0.1
        assert result["mean_after_reference"] > result["baseline_mean"]
