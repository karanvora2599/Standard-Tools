"""
Phase 7 of the Databento live fix plan (CHANGELOG, 2026-09-20): plumbing.

The live findings (CHANGELOG, 2026-09-20, the plumbing)
named each of these. The tests here pin the fix:

  gc          `sqt cache gc` removes a dead format generation and nothing else
  unsettled   the session cache does not serve a still-forming bar as final
  utc         the disk guard compares against the UTC date
  containment one check, prefix handled once, used by every root
  source      the data runtime's tools take a source, and the tick error
              names the providers that serve ticks
  configured  an unconfigured Databento reports available=False and why
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from standard_quant_tools._containment import is_within, require_within
from standard_quant_tools.agent.runtimes.data.models import (
    FetchOhlcvInput,
    FetchTickTapeInput,
)
from standard_quant_tools.agent.runtimes.data.tools import fetch_ohlcv, fetch_tick_tape
from standard_quant_tools.cli import cmd_cache_gc
from standard_quant_tools.cli import main as cli_main
from standard_quant_tools.data import _cache as cache_module
from standard_quant_tools.data._cache import (
    _is_historical,
    _session_cache_get,
    _session_cache_set,
    dead_generations,
)
from standard_quant_tools.data.base import DataProvider
from standard_quant_tools.data.factory import DataFactory
from standard_quant_tools.error import ValidationError


@pytest.fixture(autouse=True)
def _own_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_module, "_CACHE_ROOT", tmp_path / "ohlcv")
    cache_module._session_cache.clear()
    monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path / "runs"))


# ── gc ───────────────────────────────────────────────────────────────────


class TestTheDeadGenerationIsCollected:
    """The live cache held 1,574 files, 501 of them a generation that is
    never read again."""

    def _populate(self):
        root = cache_module._CACHE_ROOT
        root.mkdir(parents=True)
        for name in (
            "v1_yfinance_AAPL_a_b_1d.parquet",
            "v2_polygon_MSFT_a_b_1d.parquet",
        ):
            (root / name).write_bytes(b"old")
        (root / "v3_yfinance_AAPL_a_b_1d.parquet").write_bytes(b"current")
        (root / "notes.txt").write_text("not ours")
        (root / "unversioned.parquet").write_bytes(b"not ours either")
        return root

    def test_a_dry_run_lists_only_the_dead_generation(self):
        root = self._populate()
        dead = dead_generations(dry_run=True)
        assert [p.name for p in dead] == [
            "v1_yfinance_AAPL_a_b_1d.parquet",
            "v2_polygon_MSFT_a_b_1d.parquet",
        ]
        assert len(list(root.iterdir())) == 5  # nothing removed

    def test_confirm_removes_them_and_nothing_else(self):
        root = self._populate()
        removed = cmd_cache_gc(confirm=True)
        assert len(removed) == 2
        assert sorted(p.name for p in root.iterdir()) == [
            "notes.txt",
            "unversioned.parquet",
            "v3_yfinance_AAPL_a_b_1d.parquet",
        ]

    def test_the_cli_command_is_wired(self, capsys):
        self._populate()
        assert cli_main(["cache", "gc"]) == 0
        out = capsys.readouterr().out
        assert "Dead generation (dry-run): 2 file(s)" in out
        assert cli_main(["cache", "gc", "--confirm"]) == 0
        assert "Deleted: 2 file(s)" in capsys.readouterr().out
        assert cmd_cache_gc() == []

    def test_a_missing_cache_directory_is_nothing_to_collect(self):
        assert dead_generations() == []


# ── the session cache and the guard ──────────────────────────────────────


class TestAnUnsettledBarIsNotServedAsFinal:
    """The session cache's hour-long TTL served a still-forming bar as
    final for up to an hour."""

    def test_a_historical_window_is_kept_for_the_session(self, monkeypatch):
        frame = pd.DataFrame({"Close": [1.0]})
        _session_cache_set(("k", "hist"), frame, end="2020-01-01")
        clock = cache_module.time.monotonic()
        monkeypatch.setattr(cache_module.time, "monotonic", lambda: clock + 3000.0)
        assert _session_cache_get(("k", "hist")) is frame

    def test_an_unsettled_window_expires_within_a_minute(self, monkeypatch):
        frame = pd.DataFrame({"Close": [1.0]})
        today = datetime.now(timezone.utc).date().isoformat()
        _session_cache_set(("k", "live"), frame, end=today)
        assert _session_cache_get(("k", "live")) is frame
        clock = cache_module.time.monotonic()
        monkeypatch.setattr(cache_module.time, "monotonic", lambda: clock + 61.0)
        assert _session_cache_get(("k", "live")) is None

    def test_without_an_end_the_entry_is_kept_as_before(self):
        _session_cache_set(("k", "bare"), 1)
        assert _session_cache_get(("k", "bare")) == 1

    def test_the_guard_uses_the_utc_date(self, monkeypatch):
        class _Clock(datetime):
            @classmethod
            def now(cls, tz=None):
                # 01:00 UTC on the 2nd: a local clock west of UTC still says the 1st.
                return datetime(2026, 9, 2, 1, 0, tzinfo=timezone.utc)

        monkeypatch.setattr(cache_module, "datetime", _Clock)
        assert _is_historical("2026-09-01") is True
        assert _is_historical("2026-09-02") is False


# ── containment ──────────────────────────────────────────────────────────


class TestOneContainmentCheck:
    def test_inside_and_outside(self, tmp_path):
        assert is_within(tmp_path / "a" / "b", tmp_path)
        assert not is_within(tmp_path.parent, tmp_path)

    def test_the_extended_length_prefix_is_ignored(self, tmp_path):
        prefixed = Path("\\\\?\\" + str(tmp_path / "child"))
        assert is_within(prefixed, tmp_path)
        assert is_within(tmp_path / "child", Path("\\\\?\\" + str(tmp_path)))

    def test_the_message_is_the_callers(self, tmp_path):
        with pytest.raises(ValidationError, match="escapes here"):
            require_within(tmp_path.parent, tmp_path, "it escapes here")
        assert require_within(tmp_path / "x", tmp_path, "unused") == tmp_path / "x"

    def test_every_root_uses_it(self):
        import inspect

        from standard_quant_tools import _runspath, artifact_store
        from standard_quant_tools.agent.runtimes.meta import tools as meta_tools

        for module in (cache_module, _runspath, artifact_store, meta_tools):
            source = inspect.getsource(module)
            assert "require_within(" in source, module.__name__
            assert (
                "is_relative_to(" not in source.replace("# ", "")
                or "require_within(" in source
            )


# ── the data runtime ─────────────────────────────────────────────────────


def _bars():
    index = pd.date_range("2024-01-02", periods=5, freq="B")
    return pd.DataFrame(
        {"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Volume": 100.0},
        index=index,
    )


class TestTheDataRuntimeTakesASource:
    def test_the_source_reaches_the_factory(self, monkeypatch):
        seen = []
        provider = MagicMock()
        provider.get_ohlcv.return_value = _bars()

        def get_provider(source="yfinance", *args, **kwargs):
            seen.append(source)
            return provider

        monkeypatch.setattr(DataFactory, "get_provider", get_provider)
        fetch_ohlcv(
            FetchOhlcvInput(
                symbol="AAPL",
                start_date="2024-01-02",
                end_date="2024-01-08",
                run_id="run_src",
                name="bars",
                source="databento",
            )
        )
        assert seen == ["databento"]
        fetch_ohlcv(
            FetchOhlcvInput(
                symbol="AAPL",
                start_date="2024-01-02",
                end_date="2024-01-08",
                run_id="run_default",
                name="bars",
            )
        )
        assert seen[-1] == "yfinance"

    def test_the_tick_refusal_names_the_providers_that_serve_ticks(self, monkeypatch):
        class Bars(DataProvider):
            def get_ohlcv(self, *a, **k):
                return _bars()

            def get_ohlcv_async(self, *a, **k):
                raise NotImplementedError

            def get_ticker_info(self, *a, **k):
                raise NotImplementedError

            def get_financial_ratios(self, *a, **k):
                raise NotImplementedError

            def get_metadata(self, *a, **k):
                raise NotImplementedError

        monkeypatch.setattr(DataFactory, "get_provider", lambda *a, **k: Bars())
        with pytest.raises(ValidationError) as excinfo:
            fetch_tick_tape(
                FetchTickTapeInput(
                    symbol="AAPL",
                    start_date="2024-01-02",
                    end_date="2024-01-03",
                    run_id="run_ticks",
                    name="tape",
                )
            )
        message = str(excinfo.value)
        assert "DatabentoProvider" in message
        assert "source='databento'" in message
        assert "Only PolygonProvider" not in message


class TestAnUnconfiguredDatabentoIsNotAvailable:
    def test_available_false_with_the_reason(self, monkeypatch):
        from standard_quant_tools.agent.models import DataCapabilitiesInput
        from standard_quant_tools.agent.runtimes.meta.tools import (
            describe_data_capabilities,
        )

        monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
        result = describe_data_capabilities(DataCapabilitiesInput(source="databento"))
        assert result.available is False
        assert "DATABENTO_API_KEY" in (result.unavailable_reason or "")
        # The class is still described: ticks would be reachable once configured.
        assert result.trades is True

    def test_a_key_makes_it_available(self, monkeypatch):
        from standard_quant_tools.agent.models import DataCapabilitiesInput
        from standard_quant_tools.agent.runtimes.meta.tools import (
            describe_data_capabilities,
        )

        monkeypatch.setenv("DATABENTO_API_KEY", "db-test-key")
        result = describe_data_capabilities(DataCapabilitiesInput(source="databento"))
        assert result.available is True
