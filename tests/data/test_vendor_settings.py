"""
The vendor settings the data providers read from the environment.

Every one is read the library's one way (`_env.env_str` / `env_int`): a
blank value is unset, a local `.env` is loaded first, and a refusal names
the variable without echoing the value. Before, Polygon sent a
whitespace-only key to the vendor, Bloomberg connected to the host `""`
and refused an empty port, and each reader decided for itself whether to
load the `.env` at all.
"""

from __future__ import annotations

import pytest

from standard_quant_tools.data.bloomberg_provider import _resolve_bloomberg_config
from standard_quant_tools.data.databento import (
    DATASET_DEPTH,
    DATASET_FUTURES,
    DATASET_NASDAQ_BASIC,
    DATASET_OPTIONS,
)
from standard_quant_tools.data.databento_provider import DatabentoProvider
from standard_quant_tools.data.polygon_provider import _resolve_polygon_api_key
from standard_quant_tools.error import APIError, ValidationError

_DATABENTO_SETTINGS = (
    "DATABENTO_API_KEY",
    "DATABENTO_DATASET",
    "DATABENTO_DEPTH_DATASET",
    "DATABENTO_FUTURES_DATASET",
    "DATABENTO_OPTIONS_DATASET",
    "DATABENTO_OHLCV_DATASET",
)


class TestPolygonKey:
    @pytest.mark.parametrize("blank", ["", "   ", "\t"])
    def test_a_blank_key_is_no_key(self, monkeypatch, blank):
        """A whitespace-only key used to be returned as the key and sent
        to the vendor, which answered with a 401 about a key nobody set."""
        monkeypatch.setenv("SQT_POLYGON_API_KEY", blank)
        with pytest.raises(APIError, match="No Polygon.io API key found"):
            _resolve_polygon_api_key()

    def test_a_padded_key_is_read_without_its_padding(self, monkeypatch):
        monkeypatch.setenv("SQT_POLYGON_API_KEY", "  env-key  ")
        assert _resolve_polygon_api_key() == "env-key"

    def test_an_explicit_key_still_wins(self, monkeypatch):
        monkeypatch.setenv("SQT_POLYGON_API_KEY", "env-key")
        assert _resolve_polygon_api_key("explicit-key") == "explicit-key"


class TestBloombergAddress:
    def test_a_blank_host_is_the_default_host(self, monkeypatch):
        monkeypatch.setenv("SQT_BLOOMBERG_HOST", "  ")
        monkeypatch.delenv("SQT_BLOOMBERG_PORT", raising=False)
        assert _resolve_bloomberg_config() == ("localhost", 8194)

    def test_a_blank_port_is_the_default_port(self, monkeypatch):
        """An empty port used to be refused as not an integer, while an
        unset one was the default: the same intent, two answers."""
        monkeypatch.delenv("SQT_BLOOMBERG_HOST", raising=False)
        monkeypatch.setenv("SQT_BLOOMBERG_PORT", "")
        assert _resolve_bloomberg_config() == ("localhost", 8194)

    @pytest.mark.parametrize("port", ["0", "70000", "-1"])
    def test_a_port_no_socket_can_use_is_refused_by_name(self, monkeypatch, port):
        monkeypatch.setenv("SQT_BLOOMBERG_PORT", port)
        with pytest.raises(ValidationError, match="SQT_BLOOMBERG_PORT") as exc:
            _resolve_bloomberg_config()
        assert port not in str(exc.value)

    def test_a_padded_port_is_read(self, monkeypatch):
        monkeypatch.setenv("SQT_BLOOMBERG_HOST", " bbg.internal ")
        monkeypatch.setenv("SQT_BLOOMBERG_PORT", " 8195 ")
        assert _resolve_bloomberg_config() == ("bbg.internal", 8195)

    def test_an_explicit_port_is_not_second_guessed_by_the_environment(
        self, monkeypatch
    ):
        monkeypatch.setenv("SQT_BLOOMBERG_PORT", "not-a-port")
        assert _resolve_bloomberg_config(host="h", port=1234) == ("h", 1234)


class TestDatabentoSettings:
    @pytest.fixture(autouse=True)
    def _clean(self, monkeypatch):
        for name in _DATABENTO_SETTINGS:
            monkeypatch.delenv(name, raising=False)

    def test_blank_settings_are_the_defaults(self, monkeypatch):
        for name in _DATABENTO_SETTINGS:
            monkeypatch.setenv(name, "   ")
        provider = DatabentoProvider(client=object())
        assert provider.is_configured is False
        assert provider._dataset == DATASET_NASDAQ_BASIC
        assert provider._depth_dataset == DATASET_DEPTH
        assert provider._futures_dataset == DATASET_FUTURES
        assert provider._options_dataset == DATASET_OPTIONS
        assert "   " not in provider._bar_datasets("ohlcv-1m")

    def test_set_values_are_read_without_their_padding(self, monkeypatch):
        monkeypatch.setenv("DATABENTO_API_KEY", " db-key ")
        monkeypatch.setenv("DATABENTO_DATASET", " XNAS.ITCH ")
        monkeypatch.setenv("DATABENTO_OHLCV_DATASET", " EQUS.MINI ")
        provider = DatabentoProvider(client=object())
        assert provider.is_configured is True
        assert provider._api_key == "db-key"
        assert provider._dataset == "XNAS.ITCH"
        assert provider._bar_datasets("ohlcv-1d")[0] == "EQUS.MINI"

    def test_arguments_win_over_the_environment(self, monkeypatch):
        monkeypatch.setenv("DATABENTO_DATASET", "XNAS.ITCH")
        provider = DatabentoProvider(
            api_key="arg-key", client=object(), dataset="XNAS.BASIC"
        )
        assert provider._api_key == "arg-key"
        assert provider._dataset == "XNAS.BASIC"
