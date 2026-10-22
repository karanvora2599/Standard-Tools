"""
Split and dividend adjustment, stated separately by every provider from the
request it actually sends.

One `adjusted=True` used to cover three different feeds: yfinance asks for
total-return bars (`auto_adjust=True`), Polygon's `adjusted=true` takes out
splits only, and Bloomberg set `adjustmentSplit` alone, leaving dividends to
whatever the answering terminal's DPDF settings said. A return computed from
one is not a return computed from another, by the dividend yield. Each test
here pins a provider's two flags to the parameter it sends, so a change to
the request cannot leave the statement behind (see the CHANGELOG entry of
2026-10-22).

No network: each provider's single network seam is replaced, as the
provider suites do.
"""

import datetime as dt
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError as PydanticValidationError

from standard_quant_tools.data import _cache as cache_module
from standard_quant_tools.data import (
    bloomberg_provider,
    polygon_provider,
    yfinance_provider,
)
from standard_quant_tools.data.adjustment import (
    MIXED_RETURN_BASIS,
    combine_return_bases,
    return_basis_of,
    split_adjusted_of,
    stamp_adjustment,
)
from standard_quant_tools.data.bloomberg_provider import BloombergProvider
from standard_quant_tools.data.databento_provider import DatabentoProvider
from standard_quant_tools.data.metadata import DataSetMetadata
from standard_quant_tools.data.polygon_provider import PolygonProvider
from standard_quant_tools.data.yfinance_provider import YFinanceProvider

WINDOW = ("2024-01-02", "2024-01-31")


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_module, "_CACHE_ROOT", tmp_path / "cache")
    with cache_module._session_cache_lock:
        cache_module._session_cache.clear()


def _meta(**flags) -> DataSetMetadata:
    return DataSetMetadata(
        provider="test",
        survivorship_free=False,
        point_in_time=False,
        frequency="1d",
        timezone="UTC",
        **flags,
    )


# ── Each provider, from its own request ─────────────────────────────────


def _yahoo_frame() -> pd.DataFrame:
    index = pd.bdate_range("2024-01-02", "2024-02-01", inclusive="left", tz="UTC")
    close = 100.0 + np.arange(len(index))
    return pd.DataFrame(
        {
            "Open": close,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Volume": np.full(len(index), 1_000_000),
        },
        index=index,
    )


class TestYFinance:
    def _fetch(self, monkeypatch):
        calls = []

        class FakeTicker:
            def __init__(self, symbol, *_a, **_k):
                self.ticker = symbol.upper()

            def history(self, **kwargs):
                calls.append(kwargs)
                return _yahoo_frame()

        monkeypatch.setattr(yfinance_provider.yf, "Ticker", FakeTicker)
        return YFinanceProvider().get_ohlcv("AAPL", *WINDOW), calls

    def test_daily_bars_are_requested_and_stated_as_total_return(self, monkeypatch):
        """auto_adjust=True rescales OHLC by Yahoo's dividend- and
        split-adjusted close, so both flags are True for daily bars."""
        frame, calls = self._fetch(monkeypatch)
        assert calls[0]["auto_adjust"] is True
        meta = YFinanceProvider().get_metadata("AAPL", "1d")
        assert (meta.split_adjusted, meta.dividend_adjusted) == (True, True)
        assert meta.adjusted is True
        assert meta.return_basis == "total_return"
        assert "auto_adjust=True" in meta.adjustment_note
        assert frame.attrs["return_basis"] == "total_return"
        assert frame.attrs["split_adjusted"] is True

    @pytest.mark.parametrize("interval", ["1m", "5m", "60m", "1h"])
    def test_intraday_dividends_are_not_known_from_the_request(self, interval):
        """Yahoo sends an adjusted close only with daily and coarser bars,
        and yfinance substitutes Close when it is absent; whether an
        intraday answer carried one is not in the request, so the flag is
        None with the reason, not a guess."""
        meta = YFinanceProvider().get_metadata("AAPL", interval)
        assert meta.split_adjusted is True
        assert meta.dividend_adjusted is None
        assert meta.adjusted is None
        assert meta.return_basis == "unknown"
        assert "cannot be read from the request" in meta.adjustment_note

    @pytest.mark.parametrize("interval", ["1d", "5d", "1wk", "1mo", "3mo"])
    def test_daily_and_coarser_are_total_return(self, interval):
        meta = YFinanceProvider().get_metadata("AAPL", interval)
        assert meta.return_basis == "total_return"


class TestPolygon:
    def test_adjusted_true_is_split_only(self, monkeypatch):
        """Polygon documents the aggregates `adjusted` parameter as split
        adjustment, so the bars are price-return bars and `adjusted` -- the
        AND -- is False, where it was reported True."""
        calls = []

        def fake_get(path, params, api_key):
            calls.append((path, dict(params)))
            bar = {"o": 100.0, "h": 105.0, "l": 99.0, "c": 103.0, "v": 1000}
            return {"status": "OK", "results": [{**bar, "t": 1704171600000}]}

        monkeypatch.setattr(polygon_provider, "_polygon_get", fake_get)
        provider = PolygonProvider(api_key="test-key")
        frame = provider.get_ohlcv("AAPL", *WINDOW)

        (path, params), *_ = calls
        assert path.startswith("/v2/aggs/")
        assert params["adjusted"] == "true"
        meta = provider.get_metadata("AAPL")
        assert (meta.split_adjusted, meta.dividend_adjusted) == (True, False)
        assert meta.adjusted is False
        assert meta.return_basis == "price_return"
        assert frame.attrs["return_basis"] == "price_return"
        assert frame.attrs["split_adjusted"] is True


class TestBloomberg:
    def _fetch(self, monkeypatch):
        provider = BloombergProvider.__new__(BloombergProvider)
        session = MagicMock(name="session")
        monkeypatch.setattr(provider, "_open_session", lambda: session, raising=False)
        bars = [
            {
                "date": dt.date(2023, 1, 3),
                "PX_OPEN": 100.0,
                "PX_HIGH": 105.0,
                "PX_LOW": 99.0,
                "PX_LAST": 103.0,
                "PX_VOLUME": 1_000_000.0,
            },
            {
                "date": dt.date(2023, 1, 4),
                "PX_OPEN": 103.0,
                "PX_HIGH": 108.0,
                "PX_LOW": 102.0,
                "PX_LAST": 107.0,
                "PX_VOLUME": 1_200_000.0,
            },
        ]
        monkeypatch.setattr(
            bloomberg_provider, "_drain_historical_response", lambda s, t: (bars, None)
        )
        frame = provider.get_ohlcv("AAPL", "2023-01-03", "2023-01-04")
        request = session.getService.return_value.createRequest.return_value
        sent = {c.args[0]: c.args[1] for c in request.set.call_args_list}
        return provider, frame, sent

    def test_every_adjustment_switch_is_set_and_dpdf_is_not_followed(self, monkeypatch):
        """Only adjustmentSplit was set, so the dividend treatment was the
        answering terminal's DPDF default -- unknowable here. All four
        switches are explicit now, dividends adjusted."""
        _, _, sent = self._fetch(monkeypatch)
        assert sent["adjustmentFollowDPDF"] is False
        assert sent["adjustmentSplit"] is True
        assert sent["adjustmentNormal"] is True
        assert sent["adjustmentAbnormal"] is True

    def test_the_statement_follows_the_request(self, monkeypatch):
        provider, frame, _ = self._fetch(monkeypatch)
        meta = provider.get_metadata("AAPL")
        assert (meta.split_adjusted, meta.dividend_adjusted) == (True, True)
        assert meta.return_basis == "total_return"
        assert "adjustmentFollowDPDF False" in meta.adjustment_note
        assert frame.attrs["return_basis"] == "total_return"


class TestDatabento:
    def test_venue_prices_are_neither(self):
        meta = DatabentoProvider(api_key="not-used", client=MagicMock()).get_metadata(
            "NVDA"
        )
        assert (meta.split_adjusted, meta.dividend_adjusted) == (False, False)
        assert meta.adjusted is False
        assert meta.return_basis == "price_return"


# ── The metadata model ──────────────────────────────────────────────────


class TestAdjustedIsTheAnd:
    @pytest.mark.parametrize(
        "split,dividend,expected",
        [
            (True, True, True),
            (True, False, False),
            (False, True, False),
            (False, None, False),
            (True, None, None),
            (None, None, None),
        ],
    )
    def test_three_valued_and(self, split, dividend, expected):
        meta = _meta(split_adjusted=split, dividend_adjusted=dividend)
        assert meta.adjusted is expected

    def test_a_contradicting_adjusted_is_refused(self):
        with pytest.raises(PydanticValidationError, match="contradicts"):
            _meta(split_adjusted=True, dividend_adjusted=False, adjusted=True)

    @pytest.mark.parametrize("legacy", [True, False])
    def test_adjusted_alone_is_read_at_its_documented_word(self, legacy):
        """`adjusted` was documented as "split/dividend-adjusted", so a
        caller that sends it alone states both or neither."""
        meta = _meta(adjusted=legacy)
        assert meta.split_adjusted is legacy
        assert meta.dividend_adjusted is legacy
        assert meta.adjusted is legacy

    def test_the_basis_is_in_the_dump(self):
        dumped = _meta(split_adjusted=True, dividend_adjusted=False).model_dump()
        assert dumped["return_basis"] == "price_return"


# ── A frame's own stamp ─────────────────────────────────────────────────


class TestFrameStamps:
    def test_split_flag_is_read_before_the_and(self):
        """A split-only frame is split-adjusted; reading its AND would call
        it unadjusted and invite a declared split to divide it twice."""
        frame = stamp_adjustment(
            pd.DataFrame({"Close": [1.0]}),
            _meta(split_adjusted=True, dividend_adjusted=False),
        )
        assert split_adjusted_of(frame) is True
        assert return_basis_of(frame) == "price_return"

    def test_the_older_adjusted_stamp_is_read_for_both(self):
        frame = pd.DataFrame({"Close": [1.0]})
        frame.attrs["adjusted"] = False
        assert split_adjusted_of(frame) is False
        assert return_basis_of(frame) == "price_return"

    def test_an_unstamped_frame_is_unknown(self):
        frame = pd.DataFrame({"Close": [1.0]})
        assert split_adjusted_of(frame) is None
        assert return_basis_of(frame) == "unknown"

    def test_disagreeing_inputs_are_mixed_and_named(self):
        basis, warnings = combine_return_bases(
            {"AAA": "total_return", "BBB": "price_return"}
        )
        assert basis == MIXED_RETURN_BASIS
        (warning,) = warnings
        assert "RETURN BASIS MIXED" in warning
        assert "AAA" in warning and "BBB" in warning

    def test_one_known_basis_beside_silence_is_unknown_and_said(self):
        basis, warnings = combine_return_bases(
            {"AAA": "total_return", "BBB": "unknown"}
        )
        assert basis == "unknown"
        assert "BBB" in warnings[0]

    def test_agreeing_inputs_are_that_basis_silently(self):
        assert combine_return_bases({"A": "price_return", "B": "price_return"}) == (
            "price_return",
            [],
        )
        assert combine_return_bases({"A": "unknown"}) == ("unknown", [])
