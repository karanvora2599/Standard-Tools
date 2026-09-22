"""
The three things this library enforced and reported nowhere.

A rule that is checked on every call and stated by no tool is discoverable
only by being wrong. Three of them shared that shape:

  THE NUMERICAL CONTRACT ran at every public boundary -- infinity refused,
  all-NaN refused, prices strictly positive, an annualization ceiling, a
  covariance symmetric to 1e-9 -- and an agent learned each rule by
  triggering it, after paying for a fetch and a run. `validate_tool_call`
  ran the schema layer only, so an all-NaN series came back `valid: true`
  and then failed at execution.

  THE CONFIGURATION decided whether a decision was recorded, where an
  artifact landed and which provider could be reached, across twenty
  environment variables that no tool mentioned.

  THE ARTIFACT STORE had a complete listing API that the tool surface
  touched zero times, and `describe_artifact` refused the store's own key
  format because it resolved a relative key against the process working
  directory.

The strongest test here is the round trip: every rule the contract
DESCRIBES is triggered and its described excerpt matched against what was
actually raised. A table of rules nobody checks against the code is
documentation with a schema, and it goes stale the same way.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools import numeric_contract as nc
from standard_quant_tools.agent.tools import dispatch
from standard_quant_tools.backtest.artifacts import save_artifact
from standard_quant_tools.error import ValidationError

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SRC = REPO_ROOT / "src" / "standard_quant_tools"

#: The five tools this file and its sibling add. Their inputs must stay
#: free of the property names the MCP catalog derives `reads_market_data`
#: from, or an offline door would advertise itself as a market fetch.
NEW_DOORS = (
    "describe_audit_log",
    "find_decisions",
    "describe_numeric_contract",
    "describe_effective_config",
    "list_artifacts",
)

#: rule id -> a call that must raise it. One per row of the described
#: table, so a row with no trigger here is a row nothing checks.
TRIGGERS = {
    "series_rejects_infinity": lambda: nc.require_finite_series(
        pd.Series([1.0, float("inf"), 2.0]), "returns", "a_tool"
    ),
    "series_rejects_all_nan": lambda: nc.require_finite_series(
        pd.Series([float("nan")] * 3), "returns", "a_tool"
    ),
    "series_allows_partial_nan": lambda: nc.require_finite_series(
        pd.Series([1.0, float("nan"), 2.0]), "returns", "a_tool", allow_nan=False
    ),
    "series_rejects_empty": lambda: nc.require_finite_series(
        pd.Series([], dtype="float64"), "returns", "a_tool"
    ),
    "price_series_strictly_positive": lambda: nc.require_positive_price_series(
        pd.Series([10.0, -5.0, 11.0]), "close", "a_tool"
    ),
    "level_series_starts_positive": lambda: nc.require_positive_start_level(
        pd.Series([0.0, 5.0, 7.0]), "equity_curve", "a_tool"
    ),
    "paired_series_share_a_length": lambda: nc.require_aligned(
        pd.Series([1.0, 2.0]), pd.Series([1.0]), "strategy", "benchmark", "a_tool"
    ),
    "paired_series_share_an_index": lambda: nc.require_aligned(
        pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2024-01-01", periods=3)),
        pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2024-01-02", periods=3)),
        "strategy",
        "benchmark",
        "a_tool",
    ),
    "count_rejects_a_bool": lambda: nc.require_positive_int(True, "window", "a_tool"),
    "count_is_a_whole_number": lambda: nc.require_positive_int(2.5, "window", "a_tool"),
    "count_is_at_least_one": lambda: nc.require_positive_int(0, "window", "a_tool"),
    "periods_per_year_ceiling": lambda: nc.require_periods_per_year(
        31_536_001, "a_tool"
    ),
    "scalar_is_finite_before_any_range_check": lambda: nc.require_finite_scalar(
        float("nan"), "risk_free_rate", "a_tool", minimum=0.0
    ),
    "scalar_within_its_declared_range": lambda: nc.require_finite_scalar(
        0.5, "risk_free_rate", "a_tool", minimum=1.0
    ),
    "covariance_is_square": lambda: nc.require_finite_covariance(
        np.zeros((2, 3)), "covariance", "a_tool"
    ),
    "covariance_is_finite": lambda: nc.require_finite_covariance(
        np.array([[1.0, np.nan], [np.nan, 1.0]]), "covariance", "a_tool"
    ),
    "covariance_is_symmetric": lambda: nc.require_finite_covariance(
        np.array([[1.0, 0.2], [0.9, 1.0]]), "covariance", "a_tool"
    ),
    "frame_rejects_infinity": lambda: nc.require_finite_series_frame(
        pd.DataFrame({"AAA": [1.0, np.inf], "BBB": [1.0, 2.0]}), "panel", "a_tool"
    ),
    "frame_rejects_all_nan": lambda: nc.require_finite_series_frame(
        pd.DataFrame({"AAA": [np.nan, np.nan]}), "panel", "a_tool"
    ),
}

#: The same calls with a value the rule permits. Each must return rather
#: than raise, because a contract that refused everything would pass the
#: trigger tests above and be useless.
LEGAL = (
    lambda: nc.require_finite_series(pd.Series([0.01, -0.02]), "returns", "a_tool"),
    lambda: nc.require_finite_series(
        pd.Series([0.01, float("nan"), -0.02]), "returns", "a_tool"
    ),
    lambda: nc.require_finite_series(
        pd.Series([], dtype="float64"), "returns", "a_tool", allow_empty=True
    ),
    lambda: nc.require_positive_price_series(
        pd.Series([10.0, 11.0]), "close", "a_tool"
    ),
    lambda: nc.require_positive_start_level(
        pd.Series([100.0, 0.0, -5.0]), "equity_curve", "a_tool"
    ),
    lambda: nc.require_aligned(
        pd.Series([1.0, 2.0], index=pd.date_range("2024-01-01", periods=2)),
        pd.Series([3.0, 4.0], index=pd.date_range("2024-01-01", periods=2)),
        "strategy",
        "benchmark",
        "a_tool",
    ),
    lambda: nc.require_positive_int(20, "window", "a_tool"),
    lambda: nc.require_periods_per_year(252, "a_tool"),
    lambda: nc.require_finite_scalar(0.03, "risk_free_rate", "a_tool", minimum=0.0),
    lambda: nc.require_finite_covariance(
        np.array([[1.0, 0.2], [0.2, 1.0]]), "covariance", "a_tool"
    ),
    lambda: nc.require_finite_series_frame(
        pd.DataFrame({"AAA": [1.0, 2.0]}), "panel", "a_tool"
    ),
)


@pytest.fixture(autouse=True)
def runs_dir(tmp_path, monkeypatch):
    """Every artifact test writes into a throwaway store, so a listing is
    a statement about this test and not about the machine."""
    monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("SQT_AUDIT_DIR", str(tmp_path / "audit"))
    return tmp_path / "runs"


def _ohlcv(seed: int, n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(rng.normal(0.0005, 0.012, n).cumsum())
    return pd.DataFrame(
        {
            "Open": close * 0.998,
            "High": close * 1.012,
            "Low": close * 0.988,
            "Close": close,
            "Volume": np.full(n, 1_000_000.0),
        },
        index=pd.bdate_range("2023-01-02", periods=n),
    )


@pytest.fixture
def universe(monkeypatch):
    panel = {"AAA": _ohlcv(1)}
    monkeypatch.setattr(
        "standard_quant_tools.agent.runtimes.research.tools.fetch_ohlcv_panel_sync",
        lambda tickers, start, end, interval="1d": {t: panel[t] for t in tickers},
    )
    return panel


def _described_rules():
    return {
        row["rule"]: row for row in dispatch("describe_numeric_contract", {})["rules"]
    }


class TestTheContractSaysWhatItEnforces:
    def test_every_described_rule_has_a_trigger_and_the_reverse(self):
        """A row nothing triggers is a row that can go stale unnoticed."""
        described = set(_described_rules())
        assert described == set(TRIGGERS), (
            f"described but never triggered: {sorted(described - set(TRIGGERS))}; "
            f"triggered but never described: {sorted(set(TRIGGERS) - described)}"
        )

    @pytest.mark.parametrize("rule", sorted(TRIGGERS))
    def test_the_described_excerpt_is_in_the_refusal_it_raises(self, rule):
        row = _described_rules()[rule]
        with pytest.raises(ValidationError) as exc:
            TRIGGERS[rule]()
        assert row["message_excerpt"] in str(exc.value), (
            f"{rule}: the table says the refusal contains "
            f"{row['message_excerpt']!r} and it raised {str(exc.value)!r}"
        )

    @pytest.mark.parametrize("index", range(len(LEGAL)))
    def test_a_legal_value_passes(self, index):
        LEGAL[index]()

    def test_every_rule_carries_its_reason_and_what_it_applies_to(self):
        for rule, row in _described_rules().items():
            assert row["applies_to"], rule
            assert len(row["why"]) > 40, rule
            assert row["message_excerpt"], rule

    def test_the_table_covers_every_function_the_contract_exports(self):
        """The contract module's own `__all__` is the definition of the
        surface; a described table that covered eight of nine would read as
        complete."""
        exercised = {
            name
            for trigger in TRIGGERS.values()
            for name in trigger.__code__.co_names
            if name.startswith("require_")
        }
        assert set(nc.__all__) <= exercised, sorted(set(nc.__all__) - exercised)

    def test_the_named_thresholds_are_the_ones_in_force(self):
        rules = _described_rules()
        assert "31536000" in rules["periods_per_year_ceiling"]["threshold"]
        assert "1e-9" in rules["covariance_is_symmetric"]["threshold"]
        assert nc.require_periods_per_year(31_536_000, "a_tool") == 31_536_000


class TestValidateToolCallRunsTheContractWithoutRunningAnything:
    def _validate(self, tool, arguments):
        return dispatch(
            "validate_tool_call", {"tool_name": tool, "arguments": arguments}
        )

    def test_an_all_nan_series_is_invalid_before_anything_runs(self, monkeypatch):
        """The schema layer alone called this valid and execution then
        refused it -- see the CHANGELOG entry of 2026-09-22."""
        fetched = []
        monkeypatch.setattr(
            "standard_quant_tools.data.yfinance_provider.YFinanceProvider.get_ohlcv",
            lambda self, *a, **k: fetched.append(1),
        )

        result = self._validate(
            "calculate_series_metrics",
            {
                "series": {"values": [float("nan")] * 3},
                "metrics": ["sharpe_ratio"],
            },
        )

        assert result["valid"] is False
        assert result["checked_numeric_contract"] is True
        assert any("no observations" in p["problem"] for p in result["problems"])
        assert not fetched

    def test_a_finite_series_is_valid(self):
        result = self._validate(
            "calculate_series_metrics",
            {
                "series": {"values": [0.01, -0.004, 0.006]},
                "metrics": ["sharpe_ratio"],
            },
        )
        assert result["valid"] is True
        assert result["checked_numeric_contract"] is True
        assert result["problems"] == []

    def test_an_infinity_in_the_values_is_caught(self):
        result = self._validate(
            "calculate_series_metrics",
            {"series": {"values": [0.01, float("inf")]}, "metrics": ["max_drawdown"]},
        )
        assert result["valid"] is False
        assert any("infinite" in p["problem"] for p in result["problems"])

    def test_an_annualization_factor_over_the_ceiling_is_caught(self):
        """The schema bounds this one below and not above, so the ceiling
        is the contract's to enforce."""
        result = self._validate(
            "calculate_series_metrics",
            {
                "series": {"values": [0.01, -0.004]},
                "metrics": ["sharpe_ratio"],
                "periods_per_year": 31_536_001,
            },
        )
        assert result["valid"] is False
        assert any("31536000" in p["problem"] for p in result["problems"])
        assert result["checked_numeric_contract"] is True

    def test_a_source_naming_a_symbol_is_not_resolved(self):
        """Checking the numbers behind a symbol would mean fetching them,
        and a validator that fetched would defeat its own purpose."""
        result = self._validate(
            "calculate_series_metrics",
            {"series": {"symbol": "AAPL"}, "metrics": ["sharpe_ratio"]},
        )
        assert result["valid"] is True
        assert result["problems"] == []
        assert any("not checked" in note for note in result["notes"])

    def test_a_tool_with_no_inline_numbers_reports_that_nothing_was_checked(self):
        result = self._validate("list_strategies", {"strategy_type": "sma_crossover"})
        assert result["valid"] is True
        assert result["checked_numeric_contract"] is False

    def test_the_strategy_layer_still_runs(self):
        result = self._validate(
            "run_backtest_compact",
            {
                "symbol": "AAPL",
                "start_date": "2022-01-01",
                "end_date": "2023-01-01",
                "strategy_type": "sma_crossover",
                "parameters": {"fast_period": 30, "slow_period": 10},
            },
        )
        assert result["valid"] is False
        assert result["checked_strategy_parameters"] is True


class TestArtifactsCanBeListedAndNamedTwoWays:
    def _write(self, run_id: str, name: str, values) -> str:
        return save_artifact(pd.Series(values, name="equity"), run_id, name)

    def test_a_dispatch_that_wrote_an_artifact_is_listed(self, universe):
        result = dispatch(
            "get_technical_panel",
            {
                "tickers": ["AAA"],
                "start_date": "2023-01-02",
                "end_date": "2023-07-01",
                "indicators": ["rsi"],
                "persist_run_id": "panelrun",
            },
        )
        written = Path(result["artifact_uris"]["rsi"])
        expected = f"panelrun/{written.name}"

        listed = dispatch("list_artifacts", {"run_id": "panelrun"})
        assert listed["total"] >= 1
        assert listed["n_runs"] == 1
        keys = {entry["key"] for entry in listed["artifacts"]}
        assert expected in keys
        assert all(entry["size_bytes"] > 0 for entry in listed["artifacts"])
        assert all(entry["modified_utc"] for entry in listed["artifacts"])

        described = dispatch("describe_artifact", {"uri": expected})
        assert described["rows"] == result["n_bars"]

    def test_a_run_id_narrows_the_listing(self):
        self._write("runone", "curve", [1.0, 2.0])
        self._write("runtwo", "curve", [3.0, 4.0])

        everything = dispatch("list_artifacts", {})
        assert everything["total"] == 2
        assert everything["n_runs"] == 2

        narrowed = dispatch("list_artifacts", {"run_id": "runone"})
        assert narrowed["total"] == 1
        assert narrowed["artifacts"][0]["key"] == "runone/curve.parquet"

    def test_the_hash_is_computed_only_under_the_flag(self):
        self._write("hashrun", "curve", [1.0, 2.0, 3.0])

        without = dispatch("list_artifacts", {"run_id": "hashrun"})
        assert without["artifacts"][0]["content_hash"] is None

        with_hash = dispatch(
            "list_artifacts", {"run_id": "hashrun", "include_hash": True}
        )
        digest = with_hash["artifacts"][0]["content_hash"]
        assert digest and len(digest) == 16

    def test_the_listing_and_the_description_agree_on_the_hash(self):
        """Two hashes of one file that never compare equal are worse than
        one -- see the CHANGELOG entry of 2026-09-22."""
        uri = self._write("agreerun", "curve", [1.0, 2.0, 3.0])
        listed = dispatch(
            "list_artifacts", {"run_id": "agreerun", "include_hash": True}
        )
        described = dispatch("describe_artifact", {"uri": uri})
        assert listed["artifacts"][0]["content_hash"] == described["content_hash"]

    def test_the_store_key_describes_the_same_file_as_the_uri(self):
        uri = self._write("keyrun", "curve", [1.0, 2.0, 3.0])
        by_uri = dispatch("describe_artifact", {"uri": uri})
        by_key = dispatch("describe_artifact", {"uri": "keyrun/curve.parquet"})
        assert by_key["rows"] == by_uri["rows"] == 3
        assert by_key["content_hash"] == by_uri["content_hash"]

    def test_a_traversing_key_is_still_refused(self):
        self._write("keyrun", "curve", [1.0, 2.0])
        for spelling in (
            "../keyrun/curve.parquet",
            "keyrun/../curve.parquet",
            "..\\keyrun\\curve.parquet",
            "/etc/passwd",
        ):
            with pytest.raises(ValidationError):
                dispatch("describe_artifact", {"uri": spelling})

    def test_an_unusable_run_id_is_refused_rather_than_listing_everything(self):
        self._write("runone", "curve", [1.0, 2.0])
        with pytest.raises(ValidationError):
            dispatch("list_artifacts", {"run_id": "../elsewhere"})

    def test_an_empty_store_says_where_it_looked(self, runs_dir):
        result = dispatch("list_artifacts", {})
        assert result["artifacts"] == []
        assert result["runs_dir"] == str(runs_dir)
        assert any("No artifacts" in w for w in result["warnings"])

    def test_the_limit_truncates(self):
        for index in range(3):
            self._write("manyrun", f"curve{index}", [1.0, 2.0])
        result = dispatch("list_artifacts", {"limit": 2})
        assert result["n_artifacts"] == 2
        assert result["total"] == 3
        assert result["truncated"] is True


class TestTheConfigurationIsReadable:
    def test_a_secret_reports_set_only_while_a_plain_setting_reports_its_value(
        self, monkeypatch
    ):
        monkeypatch.setenv("SQT_MODEL_SIGNING_KEY_PATH", "/keys/private.ed25519")
        monkeypatch.setenv("SQT_MODEL_FORMAT", "skops")

        result = dispatch("describe_effective_config", {})
        by_name = {row["name"]: row for row in result["settings"]}

        secret = by_name["SQT_MODEL_SIGNING_KEY_PATH"]
        assert secret["is_secret"] is True
        assert secret["set"] is True
        assert secret["value"] is None
        assert "private.ed25519" not in json.dumps(result)

        plain = by_name["SQT_MODEL_FORMAT"]
        assert plain["is_secret"] is False
        assert plain["value"] == "skops"
        assert plain["default"] == "joblib"

    def test_every_secret_is_withheld(self, monkeypatch):
        secrets = {
            "SQT_AUDIT_REDACT_SALT": "salt-abcdef",
            "SQT_AUDIT_SIGNING_KEY_PATH": "/keys/audit-abcdef",
            "SQT_MODEL_SIGNING_KEY_PATH": "/keys/model-abcdef",
            "SQT_MCP_TOKEN": "token-abcdef",
            "SQT_POLYGON_API_KEY": "key-abcdef",
            "SQT_MODEL_MIRROR_URL": "s3://user:pass-abcdef@bucket/models",
        }
        for name, value in secrets.items():
            monkeypatch.setenv(name, value)

        result = dispatch("describe_effective_config", {})
        dumped = json.dumps(result)
        by_name = {row["name"]: row for row in result["settings"]}

        for name, value in secrets.items():
            assert by_name[name]["is_secret"] is True
            assert by_name[name]["set"] is True
            assert by_name[name]["value"] is None
            assert value not in dumped
        assert result["n_secrets_set"] == len(secrets)

    def test_an_unset_variable_still_reports_the_value_in_force(self, monkeypatch):
        monkeypatch.delenv("SQT_MODEL_FORMAT", raising=False)
        monkeypatch.delenv("SQT_BLOOMBERG_HOST", raising=False)

        by_name = {
            row["name"]: row
            for row in dispatch("describe_effective_config", {})["settings"]
        }
        assert by_name["SQT_MODEL_FORMAT"]["set"] is False
        assert by_name["SQT_MODEL_FORMAT"]["value"] == "joblib"
        assert by_name["SQT_BLOOMBERG_HOST"]["value"] == "localhost"
        assert by_name["SQT_RUNS_DIR"]["value"]

    def test_the_effective_audit_directory_is_the_one_in_force(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("SQT_AUDIT_DIR", str(tmp_path / "elsewhere"))
        by_name = {
            row["name"]: row
            for row in dispatch("describe_effective_config", {})["settings"]
        }
        assert by_name["SQT_AUDIT_DIR"]["value"] == str(tmp_path / "elsewhere")

    def test_the_twenty_names_are_exactly_the_set_the_library_reads(self):
        """Grep-backed: a variable added to the library and not to this
        table is a setting nothing reports, which is the state this tool
        exists to end."""
        pattern = re.compile(r"SQT_[A-Z][A-Z0-9_]*")
        in_source = set()
        for path in SRC.rglob("*.py"):
            in_source |= set(pattern.findall(path.read_text(encoding="utf-8")))

        reported = {
            row["name"]
            for row in dispatch("describe_effective_config", {})["settings"]
            if row["name"].startswith("SQT_")
        }
        assert reported == in_source, (
            f"read by the library and not reported: {sorted(in_source - reported)}; "
            f"reported and not read: {sorted(reported - in_source)}"
        )
        assert len(reported) == 20

    def test_the_platform_directories_are_behind_the_flag(self):
        with_paths = {
            row["name"] for row in dispatch("describe_effective_config", {})["settings"]
        }
        assert {"LOCALAPPDATA", "XDG_STATE_HOME"} <= with_paths

        without = dispatch("describe_effective_config", {"include_paths": False})
        names = {row["name"] for row in without["settings"]}
        assert not {"LOCALAPPDATA", "XDG_STATE_HOME"} & names
        assert len(names) == 20

    def test_every_setting_names_its_reader_and_its_effect(self):
        for row in dispatch("describe_effective_config", {})["settings"]:
            assert row["reader"], row["name"]
            assert len(row["effect"]) > 20, row["name"]
            assert row["category"], row["name"]

    def test_a_value_the_library_refuses_is_a_warning_not_a_crash(self, monkeypatch):
        """A broken setting must not make the report that would explain it
        unavailable."""
        monkeypatch.setenv("SQT_MODEL_FORMAT", "not-a-format")
        result = dispatch("describe_effective_config", {})
        by_name = {row["name"]: row for row in result["settings"]}
        assert by_name["SQT_MODEL_FORMAT"]["value"] is None
        assert by_name["SQT_MODEL_FORMAT"]["set"] is True
        assert any("SQT_MODEL_FORMAT" in w for w in result["warnings"])

    def test_it_changes_nothing_it_reports(self, monkeypatch):
        import os

        monkeypatch.setenv("SQT_MODEL_FORMAT", "skops")
        before = dict(os.environ)
        dispatch("describe_effective_config", {})
        assert dict(os.environ) == before


class TestTheseDoorsStayOffline:
    def test_no_new_input_carries_a_market_data_property(self):
        """The catalog derives `reads_market_data` from the property names,
        so a meta tool that named a symbol would advertise itself as a
        fetch and be scoped out of every offline session."""
        from standard_quant_tools.mcp.catalog import build_catalog

        catalog = build_catalog()
        for name in NEW_DOORS:
            entry = catalog[name]
            assert entry.reads_market_data is False, name
            properties = (entry.input_schema or {}).get("properties", {}) or {}
            for field in properties:
                assert not any(
                    marker in field.lower()
                    for marker in ("symbol", "ticker", "universe")
                ), f"{name}.{field}"

    def test_every_new_input_forbids_an_argument_it_does_not_take(self):
        from standard_quant_tools.agent.runtimes import resolve

        meta = resolve("meta")
        for name in NEW_DOORS:
            _fn, model = meta.dispatch_table[name]
            assert model.model_config.get("extra") == "forbid", name
            with pytest.raises(Exception):
                dispatch(name, {"not_an_argument_of_this_tool": 1})

    def test_every_new_result_carries_a_warnings_channel(self):
        for name in NEW_DOORS:
            result = dispatch(name, {})
            assert isinstance(result.get("warnings"), list), name
            json.dumps(result, allow_nan=False)
