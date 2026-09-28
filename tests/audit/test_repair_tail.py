"""
`sqt audit repair-tail` cuts a torn final record off the newest day, and
nothing else.

A crash or a full disk during a write leaves the newest day ending in a
fragment the writer refuses to append after. The remedy was a manual
truncation at a byte offset. The command shows the fragment by default and,
with --confirm, cuts exactly those bytes under the writer's lock, keeping
them in a side file beside the day -- after which the trail verifies as it
did before the interrupted write. It refuses a complete final line, damage
anywhere but the end, and any day but the newest; and like every function
that changes the trail it is reachable from no dispatch table.
"""

import json
import os
import stat
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest

from standard_quant_tools import _filelock, audit, cli
from standard_quant_tools.audit.storage import lock_path_for
from standard_quant_tools.audit.writer import AuditWriter
from standard_quant_tools.error import AuditIntegrityError, ValidationError

FRAGMENT = b'{"request_id": "r3", "record_ha'


def _record(i: int) -> "audit.DecisionRecord":
    return audit.DecisionRecord(
        request_id=f"r{i}",
        timestamp_utc="2024-01-01T00:00:00+00:00",
        tool_name=f"t{i}",
        input={"i": i, "name": "Zoë"},
        cpp_available=False,
        duration_ms=1.0,
        status="ok",
    )


def _verified(directory: Path):
    head: Dict[str, Any] = {}
    notes: List[str] = []
    problems = audit.verify_audit_trail_integrity(directory, notes=notes, head=head)
    return problems, notes, head


def _side_files(day: Path) -> List[Path]:
    return sorted(day.parent.glob(f"{day.name}.torn-*"))


@pytest.fixture
def trail(tmp_path):
    """Two whole records, and what the trail verified as before the tear."""
    directory = tmp_path / "audit"
    writer = AuditWriter(audit_dir=directory)
    writer.write(_record(1))
    day = Path(writer.write(_record(2)))
    return directory, day, day.read_bytes(), _verified(directory)


def _tear(day: Path, fragment: bytes = FRAGMENT) -> None:
    with open(day, "ab") as handle:
        handle.write(fragment)


class TestTheWriterNamesTheCommand:
    def test_a_torn_day_is_refused_with_the_repair_command(self, trail):
        directory, day, _whole, _before = trail
        _tear(day)
        with pytest.raises(AuditIntegrityError) as refused:
            AuditWriter(audit_dir=directory).write(_record(3))
        assert f"sqt audit repair-tail {day.stem}" in str(refused.value)
        assert "--confirm" in str(refused.value)

    def test_a_torn_chain_index_names_only_the_manual_cut(self, trail):
        """Null case: the command repairs day files, so a torn index is not
        pointed at it."""
        directory, _day, _whole, _before = trail
        index = directory / audit._INDEX_FILENAME
        _tear(index, b'{"date": "2026')
        with pytest.raises(AuditIntegrityError) as refused:
            AuditWriter(audit_dir=directory)._read_tail(index, "entry")
        assert "repair-tail" not in str(refused.value)
        assert "truncate" in str(refused.value)


class TestTheCutIsShownThenMade:
    def test_by_default_nothing_is_touched(self, trail):
        _directory, day, whole, _before = trail
        _tear(day)
        torn_bytes = day.read_bytes()

        torn = cli.cmd_repair_tail(audit_dir=day.parent)

        assert not torn.cut and torn.side_file is None
        assert torn.offset == len(whole) and torn.fragment == FRAGMENT
        assert day.read_bytes() == torn_bytes
        assert _side_files(day) == []

    def test_with_confirm_exactly_the_torn_bytes_go_to_a_side_file(self, trail):
        directory, day, whole, before = trail
        _tear(day)

        torn = cli.cmd_repair_tail(day.stem, confirm=True, audit_dir=directory)

        assert torn.cut
        assert day.read_bytes() == whole
        assert _side_files(day) == [torn.side_file]
        assert torn.side_file.read_bytes() == FRAGMENT
        # The day verifies as it did before the interrupted write.
        assert _verified(directory) == before
        assert before[0] == []

    def test_the_writer_carries_on_afterwards(self, trail):
        directory, day, _whole, _before = trail
        _tear(day)
        cli.cmd_repair_tail(confirm=True, audit_dir=directory)

        AuditWriter(audit_dir=directory).write(_record(3))

        assert audit.verify_audit_trail_integrity(directory) == []
        assert len(day.read_bytes().splitlines()) == 3

    def test_a_write_cut_inside_a_character_is_torn_too(self, trail):
        """A multi-byte character split by the crash: not UTF-8 and not a
        record, and the writer's refusal points here as well."""
        directory, day, whole, before = trail
        _tear(day, '{"request_id": "r3", "name": "Zoë'.encode("utf-8")[:-1])
        with pytest.raises(AuditIntegrityError, match="sqt audit repair-tail"):
            AuditWriter(audit_dir=directory).write(_record(3))

        cli.cmd_repair_tail(confirm=True, audit_dir=directory)

        assert day.read_bytes() == whole
        assert _verified(directory) == before

    def test_the_cut_waits_for_the_writers_lock(self, trail):
        directory, day, whole, _before = trail
        _tear(day)
        held = _filelock.acquire_lock(lock_path_for(day))
        assert held is not None
        done = threading.Event()

        def _repair():
            cli.cmd_repair_tail(confirm=True, audit_dir=directory)
            done.set()

        worker = threading.Thread(target=_repair)
        try:
            worker.start()
            time.sleep(0.3)
            assert not done.is_set()
            assert day.read_bytes().endswith(FRAGMENT)
        finally:
            _filelock.release_lock(held)
            worker.join(timeout=10)
        assert done.is_set() and day.read_bytes() == whole

    def test_no_lock_no_cut(self, trail, monkeypatch):
        directory, day, _whole, _before = trail
        _tear(day)
        torn_bytes = day.read_bytes()
        monkeypatch.setattr(_filelock, "acquire_lock", lambda path: None)

        with pytest.raises(ValidationError, match="could not be created or locked"):
            cli.cmd_repair_tail(confirm=True, audit_dir=directory)
        assert day.read_bytes() == torn_bytes and _side_files(day) == []

    def test_a_sealed_day_is_refused_and_left_whole(self, trail):
        directory, day, _whole, _before = trail
        _tear(day)
        torn_bytes = day.read_bytes()
        audit.seal_day(day.stem, audit_dir=directory)
        try:
            with pytest.raises(ValidationError, match="cannot be opened for writing"):
                cli.cmd_repair_tail(confirm=True, audit_dir=directory)
        finally:
            os.chmod(day, stat.S_IREAD | stat.S_IWRITE)
        assert day.read_bytes() == torn_bytes and _side_files(day) == []


class TestEverythingElseIsRefused:
    @pytest.mark.parametrize(
        "tail, reason",
        [
            (b"", "complete record"),
            (b'{"extra": 1}', "lost only its newline"),
            (b"{not json\n", "ends in a newline"),
            (b"42", "not a record"),
            (b'{"extra": 1}{"request_id": "r4"', "goes on past it"),
        ],
        ids=[
            "an intact day",
            "a complete record without its newline",
            "a damaged line with its newline",
            "a JSON value that is not a record",
            "a complete record glued to a fragment",
        ],
    )
    def test_a_final_line_that_is_not_a_torn_record(self, trail, tail, reason):
        """Null cases: none of these is what a cut-short write leaves."""
        directory, day, _whole, _before = trail
        _tear(day, tail)
        untouched = day.read_bytes()
        for confirm in (False, True):
            with pytest.raises(ValidationError, match=reason) as refused:
                cli.cmd_repair_tail(confirm=confirm, audit_dir=directory)
            assert "Nothing was cut" in str(refused.value)
        assert day.read_bytes() == untouched and _side_files(day) == []

    def test_damage_that_is_not_only_at_the_end(self, trail):
        directory, day, _whole, _before = trail
        lines = day.read_bytes().splitlines(keepends=True)
        day.write_bytes(lines[0] + b"{damaged\n" + lines[1] + FRAGMENT)
        untouched = day.read_bytes()

        with pytest.raises(ValidationError, match=r"line\(s\) \[2\]") as refused:
            cli.cmd_repair_tail(confirm=True, audit_dir=directory)
        assert "not only at the end" in str(refused.value)
        assert day.read_bytes() == untouched and _side_files(day) == []

    def test_a_day_that_is_not_the_newest(self, trail):
        directory, day, _whole, _before = trail
        older = directory / "2000-01-01.jsonl"
        older.write_bytes(day.read_bytes() + FRAGMENT)
        untouched = older.read_bytes()

        with pytest.raises(ValidationError, match="is not the newest day"):
            cli.cmd_repair_tail("2000-01-01", confirm=True, audit_dir=directory)
        assert older.read_bytes() == untouched and _side_files(older) == []

    @pytest.mark.parametrize("date", ["yesterday", "2026-13-45x", "../2026-01-01"])
    def test_a_date_that_is_not_one(self, trail, date):
        directory, _day, _whole, _before = trail
        with pytest.raises(ValidationError, match="YYYY-MM-DD"):
            cli.cmd_repair_tail(date, audit_dir=directory)

    def test_a_date_with_no_day_file(self, trail):
        directory, _day, _whole, _before = trail
        with pytest.raises(ValidationError, match="no day file for 1999-01-01"):
            cli.cmd_repair_tail("1999-01-01", audit_dir=directory)

    def test_an_empty_audit_directory(self, tmp_path):
        with pytest.raises(ValidationError, match="no day file"):
            cli.cmd_repair_tail(audit_dir=tmp_path)


class TestTheCommandLine:
    @pytest.fixture(autouse=True)
    def _audit_dir(self, trail, monkeypatch):
        monkeypatch.setenv("SQT_AUDIT_DIR", str(trail[0]))

    def test_a_dry_run_then_a_confirmed_cut(self, trail, capsys):
        _directory, day, whole, _before = trail
        _tear(day)

        assert cli.main(["audit", "repair-tail"]) == 0
        shown = capsys.readouterr().out
        assert "Would cut (dry-run; pass --confirm)" in shown
        assert f"from byte {len(whole)}" in shown and "record_ha" in shown
        assert day.read_bytes().endswith(FRAGMENT)

        assert cli.main(["audit", "repair-tail", day.stem, "--confirm"]) == 0
        done = capsys.readouterr().out
        assert "Cut:" in done and ".torn-" in done
        assert "OK — no integrity problems found." in done
        assert "Verified through" in done
        assert day.read_bytes() == whole

    def test_a_refusal_exits_one_and_says_why(self, trail, capsys):
        """Null case: an intact day has nothing to cut."""
        assert cli.main(["audit", "repair-tail", "--confirm"]) == 1
        assert "complete record" in capsys.readouterr().err


class TestItIsNeverATool:
    def test_no_dispatch_table_reaches_it(self):
        from standard_quant_tools.agent.runtimes import all_runtimes
        from standard_quant_tools.agent.tools import _TOOL_DISPATCH
        from standard_quant_tools.audit import repair

        tables = [_TOOL_DISPATCH] + [r.dispatch_table for r in all_runtimes().values()]
        reachable = {fn for table in tables for fn, _model in table.values()}
        for fn in (
            repair.repair_torn_tail,
            audit.repair_torn_tail,
            cli.cmd_repair_tail,
        ):
            assert fn not in reachable

        names = {name for table in tables for name in table}
        offenders = sorted(
            n for n in names if {"repair", "cut", "truncate"} & set(n.split("_"))
        )
        assert not offenders, offenders

    def test_the_mcp_catalog_serves_nothing_named_for_it(self):
        from standard_quant_tools.mcp.catalog import build_catalog

        served = set(build_catalog())
        assert served, "the catalog serves nothing at all"
        assert not {n for n in served if "repair" in n.split("_")}


def test_the_side_file_is_no_day_file(trail):
    """Readers of day files, the verifier among them, never take the side
    file for a day."""
    directory, day, _whole, _before = trail
    _tear(day)
    torn = cli.cmd_repair_tail(confirm=True, audit_dir=directory)
    assert audit._iter_day_files(directory) == [day]
    assert not audit._DAY_FILE_RE.match(torn.side_file.name)
    assert json.loads(day.read_text(encoding="utf-8").splitlines()[-1])["request_id"]
