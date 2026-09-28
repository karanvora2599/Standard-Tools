"""
The audit settings are read the way every other setting is.

`SQT_AUDIT_ENABLED=""` -- the usual way of writing "leave the default" --
switched the decision log OFF, while a padded " 0 " or "off" left it on.
`SQT_AUDIT_FAIL_CLOSED` read "on" as off. The native switch had a third
vocabulary. `SQT_AUDIT_DIR="  "` named a relative directory of spaces, and
`SQT_AUDIT_RETENTION_DAYS=-5` put the deletion cutoff in the future, so
`gc --confirm` deleted every day file including today's. Each now goes
through the library's one reader: blank is the default, 1/true/yes/on and
0/false/no/off in any case and padding, and anything else refused by name.
See the CHANGELOG entry of 2026-09-28.
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from standard_quant_tools import audit, cli, native_disabled
from standard_quant_tools.agent.tools import dispatch
from standard_quant_tools.audit.dispatch import _audit_fail_closed, _run_and_record
from standard_quant_tools.audit.paths import _audit_dir, _audit_enabled
from standard_quant_tools.audit.retention import (
    _retention_days_from_env,
    gc,
    gc_candidates,
)
from standard_quant_tools.error import ValidationError

OFF = ["0", " 0 ", "off", "no", "FALSE ", "\tOff\n"]
ON = ["1", "on", " yes ", "TRUE", "On\n"]
BLANK = ["", " ", "\t\n"]


@pytest.fixture
def audit_dir(tmp_path, monkeypatch) -> Path:
    directory = tmp_path / "audit"
    monkeypatch.setenv("SQT_AUDIT_DIR", str(directory))
    monkeypatch.delenv("SQT_AUDIT_FAIL_CLOSED", raising=False)
    return directory


def _records(directory: Path) -> int:
    return sum(
        1
        for day in audit._iter_day_files(directory)
        for line in day.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


class TestTheRecordingSwitch:
    @pytest.mark.parametrize("value", BLANK)
    def test_blank_is_the_default_which_is_on(self, monkeypatch, value):
        monkeypatch.setenv("SQT_AUDIT_ENABLED", value)
        assert _audit_enabled() is True

    @pytest.mark.parametrize("value", OFF)
    def test_every_spelling_of_off_is_off(self, monkeypatch, value):
        monkeypatch.setenv("SQT_AUDIT_ENABLED", value)
        assert _audit_enabled() is False

    @pytest.mark.parametrize("value", ON)
    def test_every_spelling_of_on_is_on(self, monkeypatch, value):
        monkeypatch.setenv("SQT_AUDIT_ENABLED", value)
        assert _audit_enabled() is True

    def test_an_unknown_word_is_refused_by_name_not_value(self, monkeypatch):
        monkeypatch.setenv("SQT_AUDIT_ENABLED", "enabled-please")
        with pytest.raises(ValidationError, match="SQT_AUDIT_ENABLED") as refused:
            _audit_enabled()
        assert "enabled-please" not in str(refused.value)

    def test_an_empty_setting_still_records_the_call(self, audit_dir, monkeypatch):
        """It recorded nothing: the empty string was in the OFF set."""
        monkeypatch.setenv("SQT_AUDIT_ENABLED", "")
        dispatch("list_strategies", {"strategy_type": "sma_crossover"})
        assert _records(audit_dir) == 1

    @pytest.mark.parametrize("value", [" off ", "0"])
    def test_off_records_nothing(self, audit_dir, monkeypatch, value):
        """Null case, and a padded "off" -- which used to leave it on."""
        monkeypatch.setenv("SQT_AUDIT_ENABLED", value)
        dispatch("list_strategies", {"strategy_type": "sma_crossover"})
        assert _records(audit_dir) == 0

    def test_an_unknown_word_refuses_the_call_before_the_tool_runs(
        self, audit_dir, monkeypatch
    ):
        """Whether an action is recorded cannot be guessed, so the call is
        refused -- before it acts, not after."""
        monkeypatch.setenv("SQT_AUDIT_ENABLED", "sometimes")
        ran = []

        class _Input:
            def model_dump(self):
                return {}

        with pytest.raises(ValidationError, match="SQT_AUDIT_ENABLED"):
            _run_and_record("probe", lambda _: ran.append(1) or {}, _Input())
        assert ran == []
        assert _records(audit_dir) == 0


class TestTheFailClosedSwitch:
    @pytest.mark.parametrize("value", BLANK + OFF)
    def test_blank_and_off_are_fail_open(self, monkeypatch, value):
        monkeypatch.setenv("SQT_AUDIT_FAIL_CLOSED", value)
        assert _audit_fail_closed() is False

    @pytest.mark.parametrize("value", ON)
    def test_on_is_fail_closed(self, monkeypatch, value):
        """ "on" read as OFF here, unlike the other two switches."""
        monkeypatch.setenv("SQT_AUDIT_FAIL_CLOSED", value)
        assert _audit_fail_closed() is True

    def test_an_unknown_word_is_refused(self, monkeypatch):
        monkeypatch.setenv("SQT_AUDIT_FAIL_CLOSED", "strict")
        with pytest.raises(ValidationError, match="SQT_AUDIT_FAIL_CLOSED"):
            _audit_fail_closed()


class TestTheNativeSwitchSpeaksTheSameWords:
    @pytest.mark.parametrize("value", BLANK + OFF)
    def test_blank_and_off_keep_the_extension(self, monkeypatch, value):
        monkeypatch.setenv("SQT_DISABLE_NATIVE", value)
        assert native_disabled() is False

    @pytest.mark.parametrize("value", ON)
    def test_on_disables_it(self, monkeypatch, value):
        monkeypatch.setenv("SQT_DISABLE_NATIVE", value)
        assert native_disabled() is True

    def test_an_unknown_word_is_refused(self, monkeypatch):
        monkeypatch.setenv("SQT_DISABLE_NATIVE", "maybe")
        with pytest.raises(ValidationError, match="SQT_DISABLE_NATIVE"):
            native_disabled()


class TestTheAuditDirectory:
    @pytest.mark.parametrize("value", [" ", "\t"])
    def test_whitespace_is_unset_not_a_directory_of_spaces(
        self, monkeypatch, tmp_path, value
    ):
        """`Path("  ")` -- a relative directory named by whitespace, in
        whatever the working directory was. Unset, the platform default
        applies; it is pointed at a scratch directory here so resolving it
        never looks at this machine's own trail."""
        from standard_quant_tools.audit import paths

        monkeypatch.setattr(paths, "_legacy_cache_audit_dir", lambda: tmp_path / "x")
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
        monkeypatch.setenv("SQT_AUDIT_DIR", value)

        resolved = _audit_dir()

        assert resolved.is_absolute()
        assert resolved == tmp_path / "standard_quant_tools" / "audit"

    def test_a_relative_path_is_refused_by_name(self, monkeypatch):
        monkeypatch.setenv("SQT_AUDIT_DIR", "relative/audit-trail")
        with pytest.raises(ValidationError, match="SQT_AUDIT_DIR") as refused:
            _audit_dir()
        assert "relative/audit-trail" not in str(refused.value)

    def test_a_home_relative_path_is_expanded(self, monkeypatch):
        monkeypatch.setenv("SQT_AUDIT_DIR", "~/sqt-audit-somewhere")
        assert _audit_dir() == Path.home() / "sqt-audit-somewhere"

    def test_an_absolute_path_is_used_as_given(self, monkeypatch, tmp_path):
        """Null case."""
        monkeypatch.setenv("SQT_AUDIT_DIR", str(tmp_path))
        assert _audit_dir() == tmp_path


class TestTheSigningKeyPath:
    def test_a_relative_key_path_is_refused_without_repeating_it(self, monkeypatch):
        if not audit.HAS_CRYPTOGRAPHY:
            pytest.skip("cryptography is not installed")
        from standard_quant_tools.audit.signing import _load_signer

        monkeypatch.setenv("SQT_AUDIT_SIGNING_KEY_PATH", "keys/secret.private")
        with pytest.raises(ValidationError, match="SQT_AUDIT_SIGNING_KEY_PATH") as r:
            _load_signer(None)
        assert "secret.private" not in str(r.value)

    def test_a_blank_key_path_is_unset(self, monkeypatch):
        if not audit.HAS_CRYPTOGRAPHY:
            pytest.skip("cryptography is not installed")
        from standard_quant_tools.audit.signing import _load_signer

        monkeypatch.setenv("SQT_AUDIT_SIGNING_KEY_PATH", "  ")
        with pytest.raises(FileNotFoundError, match="No signing key found"):
            _load_signer(None)


def _day(directory: Path, days_ago: int) -> str:
    date = (datetime.now(timezone.utc) - timedelta(days=days_ago)).strftime("%Y-%m-%d")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{date}.jsonl").write_text(
        json.dumps({"request_id": date}) + "\n", encoding="utf-8"
    )
    return date


class TestTheRetentionWindow:
    def test_a_negative_window_from_the_environment_is_refused(
        self, tmp_path, monkeypatch
    ):
        """It returned every day file, today's included."""
        _day(tmp_path, 0)
        monkeypatch.setenv("SQT_AUDIT_RETENTION_DAYS", "-5")
        with pytest.raises(ValidationError, match="SQT_AUDIT_RETENTION_DAYS"):
            gc_candidates(tmp_path)

    def test_a_negative_window_argument_is_refused_and_nothing_is_deleted(
        self, tmp_path
    ):
        today = _day(tmp_path, 0)
        old = _day(tmp_path, 10)
        with pytest.raises(ValidationError, match="negative"):
            gc(tmp_path, retention_days=-5, dry_run=False)
        assert (tmp_path / f"{today}.jsonl").exists()
        assert (tmp_path / f"{old}.jsonl").exists()

    def test_a_window_that_is_not_a_number_is_refused(self, monkeypatch):
        """It was logged to a handler nobody attaches and read as unset."""
        monkeypatch.setenv("SQT_AUDIT_RETENTION_DAYS", "thirty")
        with pytest.raises(ValidationError, match="SQT_AUDIT_RETENTION_DAYS"):
            _retention_days_from_env()

    def test_zero_never_offers_today(self, tmp_path):
        today = _day(tmp_path, 0)
        yesterday = _day(tmp_path, 1)
        candidates = gc_candidates(tmp_path, retention_days=0)
        assert yesterday in candidates and today not in candidates

    def test_a_future_dated_file_is_never_offered(self, tmp_path):
        future = _day(tmp_path, -3)
        assert future not in gc_candidates(tmp_path, retention_days=0)

    @pytest.mark.parametrize("value", BLANK)
    def test_blank_is_no_window(self, monkeypatch, value):
        monkeypatch.setenv("SQT_AUDIT_RETENTION_DAYS", value)
        assert _retention_days_from_env() is None

    def test_an_ordinary_window_behaves_as_before(self, tmp_path, monkeypatch):
        """Null case."""
        recent = _day(tmp_path, 3)
        old = _day(tmp_path, 40)
        monkeypatch.setenv("SQT_AUDIT_RETENTION_DAYS", " 30 ")
        assert gc_candidates(tmp_path) == [old]
        assert recent not in gc_candidates(tmp_path)

    def test_sqt_gc_refuses_a_negative_window(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("SQT_AUDIT_DIR", str(tmp_path))
        today = _day(tmp_path, 0)
        assert cli.main(["gc", "--confirm", "--retention-days", "-5"]) == 1
        assert "negative" in capsys.readouterr().err
        assert (tmp_path / f"{today}.jsonl").exists()

    def test_describe_audit_log_reports_a_refused_window(self, audit_dir, monkeypatch):
        """The tool that explains the configuration does not fail on it."""
        monkeypatch.setenv("SQT_AUDIT_RETENTION_DAYS", "-5")
        result = dispatch("describe_audit_log", {})
        assert result["retention_days"] is None
        assert result["gc_candidate_dates"] == []
        assert any("SQT_AUDIT_RETENTION_DAYS" in w for w in result["warnings"])
