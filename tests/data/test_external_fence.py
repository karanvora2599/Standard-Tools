"""
Where the agent surface may read and write data this library did not fetch.

`register_external_dataset`, `register_external_panel` and
`prepare_vendor_extract` take a path an agent chooses. Until the CHANGELOG
entry of 2026-09-27 the only bound on it was format -- Parquet or CSV with
the columns of a kind -- so any market-data-shaped file anywhere on the
machine could be registered and previewed row by row, the conversion could
write anywhere, and a `$NAME` in the path was expanded, putting that
variable's value into the refusal and so into the decision log.

What these pin:

    a path outside the fence            -> refused, naming SQT_EXTERNAL_DIRS,
                                           the same sentence whether or not
                                           it exists
    a link inside the fence to outside  -> refused after links are followed
    a directory holding such a link     -> refused, naming the entry
    a network path                      -> refused before any filesystem call
    `$NAME`, `%NAME%`, `${NAME}`        -> never expanded
    a conversion's output               -> only the extracts folder or a
                                           listed directory, never the cache,
                                           the audit directory or the rest of
                                           the runs directory
    a registration made under a wider
    fence                               -> stops resolving when it narrows

and, for each, that a path inside the fence is still accepted.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.agent.runtimes import handoff
from standard_quant_tools.agent.tools import dispatch
from standard_quant_tools.data import external
from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.agent.dispatch import modeling_dispatch

SECRET_NOTE = "DECOY-NOT-A-REAL-SECRET-0001"
PLANTED = "zz-planted-zz"


@pytest.fixture
def fence(tmp_path, monkeypatch):
    """Runs and audit under tmp_path; nothing listed beyond the runs
    directory; a decoy with a column a caller must not see, outside."""
    runs = tmp_path / "runs"
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.setenv("SQT_RUNS_DIR", str(runs))
    monkeypatch.setenv("SQT_AUDIT_DIR", str(tmp_path / "audit"))
    monkeypatch.delenv("SQT_EXTERNAL_DIRS", raising=False)
    decoy = outside / "positions.csv"
    pd.DataFrame(
        {
            # `timestamp` because a `tick_tape` requires one. These tests are
            # about the fence, not the schema, and a tape that cannot be read
            # would never reach the fence in the first place.
            "timestamp": [
                "2024-03-05T14:30:00Z",
                "2024-03-05T14:30:01Z",
                "2024-03-05T14:30:02Z",
            ],
            "price": [101.5, 101.6, 101.7],
            "size": [100, 200, 300],
            "secret_note": [SECRET_NOTE] * 3,
        }
    ).to_csv(decoy, index=False)
    return {"tmp": tmp_path, "runs": runs, "outside": outside, "decoy": decoy}


def _audit_text(tmp_path: Path) -> str:
    directory = tmp_path / "audit"
    if not directory.exists():
        return ""
    return "".join(
        p.read_text(encoding="utf-8") for p in sorted(directory.glob("*.jsonl"))
    )


def _register(path, run_id="probe", name="probe"):
    return dispatch(
        "register_external_dataset",
        {"path": str(path), "kind": "tick_tape", "run_id": run_id, "name": name},
    )


def _raw_trades(path: Path) -> Path:
    stamps = (
        (
            pd.Timestamp("2024-03-01 14:30")
            + pd.to_timedelta(np.arange(50) * 5, unit="ms")
        )
        .astype("datetime64[ns]")
        .astype("int64")
    )
    pd.DataFrame(
        {
            "ts_recv": stamps,
            "price": (np.full(50, 100.0) * 1e9).astype("int64"),
            "size": np.arange(1, 51),
        }
    ).to_parquet(path, index=False)
    return path


def _panel(path: Path) -> Path:
    rng = np.random.default_rng(0)
    rows = [
        {"date": day, "entity": entity, "f1": rng.normal(), "target": rng.normal()}
        for day in pd.bdate_range("2024-01-01", periods=60)
        for entity in ("AAA", "BBB", "CCC")
    ]
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


class TestAPathOutsideTheFenceIsRefused:
    def test_the_decoy_is_refused_and_nothing_of_it_is_disclosed(self, fence):
        with pytest.raises(ValidationError) as caught:
            _register(fence["decoy"])
        message = str(caught.value)
        assert "SQT_EXTERNAL_DIRS" in message
        assert SECRET_NOTE not in message
        assert "secret_note" not in message
        audit = _audit_text(fence["tmp"])
        assert audit, "the refused call left no decision record to inspect"
        assert SECRET_NOTE not in audit
        assert not (fence["runs"] / "probe").exists()

    def test_the_refusal_is_the_same_whether_or_not_the_file_exists(self, fence):
        """The fence answers no question about what lies outside it."""
        absent = fence["outside"] / "absent.csv"
        with pytest.raises(ValidationError) as present_error:
            external.resolve_path(str(fence["decoy"]))
        with pytest.raises(ValidationError) as absent_error:
            external.resolve_path(str(absent))
        assert str(present_error.value).replace(
            repr(str(fence["decoy"])), "<path>"
        ) == str(absent_error.value).replace(repr(str(absent)), "<path>")

    def test_a_walk_out_of_the_runs_directory_is_refused(self, fence):
        walk = fence["runs"] / ".." / "outside" / "positions.csv"
        with pytest.raises(ValidationError, match="SQT_EXTERNAL_DIRS"):
            external.resolve_path(str(walk))

    def test_a_file_under_the_runs_directory_is_accepted(self, fence):
        inbox = fence["runs"] / "inbox"
        inbox.mkdir(parents=True)
        copy = inbox / "positions.csv"
        copy.write_bytes(fence["decoy"].read_bytes())
        registered = _register(copy)
        assert registered["ref"] == "sqt://tick_tape/probe/probe"

    def test_listing_the_directory_admits_it(self, fence, monkeypatch):
        monkeypatch.setenv("SQT_EXTERNAL_DIRS", str(fence["outside"]))
        registered = _register(fence["decoy"])
        described = dispatch(
            "describe_external_dataset",
            {"ref": registered["ref"], "preview_rows": 1},
        )
        assert described["preview"][0]["price"] == pytest.approx(101.5)

    def test_every_listed_directory_counts(self, fence, monkeypatch):
        elsewhere = fence["tmp"] / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.setenv(
            "SQT_EXTERNAL_DIRS",
            os.pathsep.join([str(elsewhere), str(fence["outside"])]),
        )
        assert _register(fence["decoy"])["ref"]

    def test_a_relative_entry_is_refused_by_name(self, fence, monkeypatch):
        monkeypatch.setenv("SQT_EXTERNAL_DIRS", "relative/inbox")
        with pytest.raises(ValidationError, match="SQT_EXTERNAL_DIRS") as caught:
            external.resolve_path(str(fence["decoy"]))
        assert "relative/inbox" not in str(caught.value)


class TestEnvironmentVariablesAreNeverExpanded:
    """A model-chosen `$NAME` used to come back as that variable's value in
    the refusal, and the refusal is written verbatim into the decision log,
    which cannot be edited afterwards."""

    SPELLINGS = ("$SQT_PLANTED", "%SQT_PLANTED%", "${SQT_PLANTED}")

    @pytest.mark.parametrize("spelling", SPELLINGS)
    def test_register_external_dataset(self, fence, monkeypatch, spelling):
        monkeypatch.setenv("SQT_PLANTED", PLANTED)
        with pytest.raises(ValidationError) as caught:
            _register(spelling)
        assert PLANTED not in str(caught.value)
        assert PLANTED not in _audit_text(fence["tmp"])

    @pytest.mark.parametrize("spelling", SPELLINGS)
    def test_inside_the_fence_too(self, fence, monkeypatch, spelling):
        """Inside a listed directory the path reaches the existence check,
        whose message echoes it -- still unexpanded."""
        monkeypatch.setenv("SQT_PLANTED", PLANTED)
        fence["runs"].mkdir(parents=True, exist_ok=True)
        with pytest.raises(ValidationError, match="no file or directory") as caught:
            _register(str(fence["runs"] / f"{spelling}.csv"))
        assert PLANTED not in str(caught.value)
        assert PLANTED not in _audit_text(fence["tmp"])

    @pytest.mark.parametrize("spelling", SPELLINGS)
    def test_prepare_vendor_extract(self, fence, monkeypatch, spelling):
        monkeypatch.setenv("SQT_PLANTED", PLANTED)
        for arguments in (
            {"path": spelling, "out_path": "x.parquet"},
            {
                "path": str(fence["decoy"]),
                "out_path": str(fence["outside"] / f"{spelling}.parquet"),
            },
        ):
            with pytest.raises(ValidationError) as caught:
                dispatch(
                    "prepare_vendor_extract",
                    {"kind": "tick_tape", "dry_run": True, **arguments},
                )
            assert PLANTED not in str(caught.value)
        assert PLANTED not in _audit_text(fence["tmp"])

    @pytest.mark.parametrize("spelling", SPELLINGS)
    def test_register_external_panel(self, fence, monkeypatch, spelling):
        monkeypatch.setenv("SQT_PLANTED", PLANTED)
        with pytest.raises(ValidationError) as caught:
            modeling_dispatch(
                "register_external_panel",
                {"path": spelling, "horizon": 1, "target_column": "target"},
            )
        assert PLANTED not in str(caught.value)
        assert PLANTED not in _audit_text(fence["tmp"])


def _junction(target: Path, link: Path) -> str:
    """A directory link that needs no privilege: a junction on Windows, a
    symlink elsewhere. Skips the test when neither can be made."""
    try:
        if sys.platform == "win32":
            import _winapi

            _winapi.CreateJunction(str(target), str(link))
            return "junction"
        os.symlink(target, link, target_is_directory=True)
        return "symlink"
    except (OSError, ImportError) as exc:
        pytest.skip(f"a directory link cannot be created here: {exc}")


def _unlink_directory_link(link: Path) -> None:
    try:
        os.rmdir(link) if sys.platform == "win32" else os.unlink(link)
    except OSError:
        pass


class TestLinksAreFollowedBeforeTheyAreTrusted:
    def test_a_link_inside_the_runs_directory_pointing_out_is_refused(self, fence):
        fence["runs"].mkdir(parents=True)
        link = fence["runs"] / "link_to_outside"
        _junction(fence["outside"], link)
        try:
            with pytest.raises(ValidationError, match="SQT_EXTERNAL_DIRS") as caught:
                _register(link / "positions.csv")
            # Only the caller's own text is echoed: where the link points
            # is not disclosed, in either spelling of a path.
            message = str(caught.value)
            for spelled in (str(link), repr(str(link))[1:-1]):
                message = message.replace(spelled, "")
            for spelled in (str(fence["outside"]), repr(str(fence["outside"]))[1:-1]):
                assert spelled not in message
        finally:
            _unlink_directory_link(link)

    def test_a_link_to_a_directory_inside_the_fence_is_accepted(self, fence):
        inbox = fence["runs"] / "inbox"
        inbox.mkdir(parents=True)
        (inbox / "positions.csv").write_bytes(fence["decoy"].read_bytes())
        link = fence["runs"] / "alias"
        _junction(inbox, link)
        try:
            assert _register(link / "positions.csv")["ref"]
        finally:
            _unlink_directory_link(link)

    def test_a_directory_dataset_holding_a_link_out_is_refused(self, fence):
        dataset = fence["runs"] / "partitioned"
        dataset.mkdir(parents=True)
        pd.DataFrame({"price": [1.0], "size": [1]}).to_parquet(
            dataset / "part-0.parquet", index=False
        )
        pd.DataFrame({"price": [9.0], "size": [9]}).to_parquet(
            fence["outside"] / "part-9.parquet", index=False
        )
        link = dataset / "linked"
        _junction(fence["outside"], link)
        try:
            with pytest.raises(ValidationError, match="link") as caught:
                external.inspect(str(dataset), kind="tick_tape")
            assert "linked" in str(caught.value)
            assert "SQT_EXTERNAL_DIRS" in str(caught.value)
        finally:
            _unlink_directory_link(link)

    def test_a_file_link_inside_a_directory_dataset_is_refused(self, fence):
        dataset = fence["runs"] / "partitioned"
        dataset.mkdir(parents=True)
        pd.DataFrame({"price": [1.0], "size": [1]}).to_parquet(
            dataset / "part-0.parquet", index=False
        )
        secret = fence["outside"] / "secret.parquet"
        pd.DataFrame({"price": [9.0], "size": [9]}).to_parquet(secret, index=False)
        try:
            os.symlink(secret, dataset / "part-1.parquet")
        except OSError as exc:
            pytest.skip(f"a file symlink cannot be created here: {exc}")
        with pytest.raises(ValidationError, match="part-1.parquet"):
            external.inspect(str(dataset), kind="tick_tape")

    def test_a_clean_directory_dataset_reads_every_partition(self, fence):
        dataset = fence["runs"] / "partitioned"
        dataset.mkdir(parents=True)
        for index in range(3):
            pd.DataFrame({"price": [float(index)], "size": [1]}).to_parquet(
                dataset / f"part-{index}.parquet", index=False
            )
        # The reader's own discovery skips these, and so does the vetted
        # list it is now handed.
        (dataset / "_SUCCESS").write_text("", encoding="utf-8")
        handle = external.inspect(str(dataset), kind="tick_tape")
        assert handle.rows == 3
        assert sum(len(batch) for batch in handle.batches()) == 3


class TestANetworkPathIsNeverTouched:
    PATHS = (
        r"\\sqt-test-nonexistent\share\x.csv",
        "//sqt-test-nonexistent/share/x.csv",
        r"\\?\UNC\sqt-test-nonexistent\share\x.csv",
        r"\\.\sqt-test-nonexistent",
    )

    @pytest.fixture
    def no_filesystem_call(self, monkeypatch):
        """Fails the test if anything asks the filesystem about the host."""

        def guard(original):
            def guarded(*args, **kwargs):
                if any("sqt-test-nonexistent" in str(a) for a in args):
                    raise AssertionError(f"the filesystem was asked about {args[0]!r}")
                return original(*args, **kwargs)

            return guarded

        for owner, attribute in (
            (os, "stat"),
            (os, "lstat"),
            (os.path, "realpath"),
            (os.path, "exists"),
            (pathlib.Path, "resolve"),
            (pathlib.Path, "exists"),
            (pathlib.Path, "stat"),
            (pathlib.Path, "is_dir"),
            (pathlib.Path, "is_file"),
        ):
            monkeypatch.setattr(owner, attribute, guard(getattr(owner, attribute)))

    @pytest.mark.parametrize("path", PATHS)
    def test_a_read_is_refused_on_the_text(self, fence, no_filesystem_call, path):
        with pytest.raises(ValidationError, match="network or device path"):
            external.resolve_path(path)

    @pytest.mark.parametrize("path", PATHS)
    def test_a_write_is_refused_on_the_text(self, fence, no_filesystem_call, path):
        with pytest.raises(ValidationError, match="network or device path"):
            external.resolve_output_path(path)

    def test_an_extended_length_local_path_is_the_path_it_spells(self, fence):
        """`\\\\?\\C:\\...` is a local path spelled long, and what a resolve
        can hand back for one that exists; it is not a device path."""
        if sys.platform != "win32":
            pytest.skip("the extended-length prefix is a Windows spelling")
        inbox = fence["runs"] / "inbox"
        inbox.mkdir(parents=True)
        copy = inbox / "positions.csv"
        copy.write_bytes(fence["decoy"].read_bytes())
        assert external.resolve_path("\\\\?\\" + str(copy)) == copy.resolve()


class TestWhereAConversionMayWrite:
    def _prepare(self, fence, out_path, **extra):
        raw = _raw_trades(fence["runs"] / "inbox" / "raw.parquet")
        return dispatch(
            "prepare_vendor_extract",
            {"path": str(raw), "kind": "tick_tape", "out_path": out_path, **extra},
        )

    @pytest.fixture(autouse=True)
    def _inbox(self, fence):
        (fence["runs"] / "inbox").mkdir(parents=True)

    def test_an_absolute_path_outside_is_refused_and_nothing_is_written(self, fence):
        target = fence["outside"] / "written_by_agent.parquet"
        with pytest.raises(ValidationError, match="SQT_EXTERNAL_DIRS"):
            self._prepare(fence, str(target))
        assert not target.exists()

    def test_a_relative_name_lands_in_the_extracts_folder(
        self, fence, tmp_path, monkeypatch
    ):
        cwd = tmp_path / "cwd"
        cwd.mkdir()
        monkeypatch.chdir(cwd)
        result = self._prepare(fence, "converted.parquet")
        expected = (fence["runs"] / "extracts" / "converted.parquet").resolve()
        assert Path(result["out_path"]) == expected
        assert expected.exists()
        assert not (cwd / "converted.parquet").exists()
        # and the next step it names is a call the fence accepts
        assert _register(result["out_path"])["ref"]

    def test_a_relative_name_cannot_walk_out_of_the_extracts_folder(self, fence):
        with pytest.raises(ValidationError):
            self._prepare(fence, "../probe/equity.parquet")
        assert not (fence["runs"] / "probe").exists()

    def test_a_listed_directory_is_writable(self, fence, monkeypatch):
        monkeypatch.setenv("SQT_EXTERNAL_DIRS", str(fence["outside"]))
        target = fence["outside"] / "converted.parquet"
        result = self._prepare(fence, str(target))
        assert Path(result["out_path"]) == target.resolve()
        assert target.exists()

    def test_the_cache_is_never_a_target_even_inside_a_listed_directory(
        self, fence, monkeypatch
    ):
        """A Parquet named like a cache entry for a window nobody fetched
        would be served as a hit."""
        from standard_quant_tools.data import _cache

        cache = fence["tmp"] / "cache"
        monkeypatch.setattr(_cache, "_CACHE_ROOT", cache)
        monkeypatch.setenv("SQT_EXTERNAL_DIRS", str(fence["tmp"]))
        target = cache / "v3_yfinance_AAPL_2024-01-01_2024-02-01_1d.parquet"
        with pytest.raises(ValidationError, match="cache"):
            self._prepare(fence, str(target))
        assert not target.exists()

    def test_the_audit_directory_is_never_a_target(self, fence, monkeypatch):
        monkeypatch.setenv("SQT_EXTERNAL_DIRS", str(fence["tmp"]))
        target = fence["tmp"] / "audit" / "planted.parquet"
        with pytest.raises(ValidationError, match="audit directory"):
            self._prepare(fence, str(target))
        assert not target.exists()

    def test_the_artifact_store_is_never_a_target(self, fence, monkeypatch):
        """A Parquet at `runs/<run_id>/<name>.parquet` resolves as an artifact
        of whatever kind a reader asks for, with nothing having published it."""
        monkeypatch.setenv("SQT_EXTERNAL_DIRS", str(fence["tmp"]))
        target = fence["runs"] / "probe" / "equity.parquet"
        with pytest.raises(ValidationError, match="runs directory"):
            self._prepare(fence, str(target))
        assert not target.exists()

    def test_a_dry_run_refuses_what_the_real_run_would(self, fence):
        with pytest.raises(ValidationError, match="SQT_EXTERNAL_DIRS"):
            self._prepare(fence, str(fence["outside"] / "never.parquet"), dry_run=True)

    def test_the_input_is_fenced_too(self, fence):
        raw = _raw_trades(fence["outside"] / "raw.parquet")
        with pytest.raises(ValidationError, match="SQT_EXTERNAL_DIRS"):
            dispatch(
                "prepare_vendor_extract",
                {"path": str(raw), "kind": "tick_tape", "out_path": "x.parquet"},
            )
        assert not (fence["runs"] / "extracts" / "x.parquet").exists()


class TestNarrowingTheFenceRevokesAccess:
    def test_a_registration_stops_resolving_when_its_directory_is_unlisted(
        self, fence, monkeypatch
    ):
        monkeypatch.setenv("SQT_EXTERNAL_DIRS", str(fence["outside"]))
        ref = _register(fence["decoy"])["ref"]
        assert isinstance(handoff.resolve(ref), external.ExternalDataset)

        monkeypatch.delenv("SQT_EXTERNAL_DIRS")
        with pytest.raises(ValidationError, match="SQT_EXTERNAL_DIRS"):
            handoff.resolve(ref)
        with pytest.raises(ValidationError, match="SQT_EXTERNAL_DIRS"):
            dispatch("describe_external_dataset", {"ref": ref, "preview_rows": 1})

    def test_a_handle_already_held_is_fenced_when_it_reads(self, fence, monkeypatch):
        monkeypatch.setenv("SQT_EXTERNAL_DIRS", str(fence["outside"]))
        handle = external.inspect(str(fence["decoy"]), kind="tick_tape")
        monkeypatch.delenv("SQT_EXTERNAL_DIRS")
        with pytest.raises(ValidationError, match="SQT_EXTERNAL_DIRS"):
            handle.head(1)


class TestTheModelingDoor:
    def test_a_panel_outside_the_fence_is_refused(self, fence):
        panel = _panel(fence["outside"] / "panel.parquet")
        with pytest.raises(ValidationError, match="SQT_EXTERNAL_DIRS"):
            modeling_dispatch(
                "register_external_panel",
                {"path": str(panel), "horizon": 1, "target_column": "target"},
            )

    def test_a_panel_inside_is_accepted(self, fence):
        inbox = fence["runs"] / "inbox"
        inbox.mkdir(parents=True)
        panel = _panel(inbox / "panel.parquet")
        result = modeling_dispatch(
            "register_external_panel",
            {"path": str(panel), "horizon": 1, "target_column": "target"},
        )
        assert result["dataset_id"]


class TestTheFenceIsReported:
    def test_the_default_is_the_runs_directory_only(self, fence):
        rows = {
            row["name"]: row
            for row in dispatch("describe_effective_config", {})["settings"]
        }
        setting = rows["SQT_EXTERNAL_DIRS"]
        assert setting["set"] is False
        assert setting["is_secret"] is False
        assert setting["value"] == str(fence["runs"].resolve())

    def test_listed_directories_follow_the_runs_directory(self, fence, monkeypatch):
        monkeypatch.setenv("SQT_EXTERNAL_DIRS", str(fence["outside"]))
        rows = {
            row["name"]: row
            for row in dispatch("describe_effective_config", {})["settings"]
        }
        assert rows["SQT_EXTERNAL_DIRS"]["value"] == os.pathsep.join(
            [str(fence["runs"].resolve()), str(fence["outside"].resolve())]
        )

    def test_a_relative_entry_is_a_warning_not_a_crash(self, fence, monkeypatch):
        monkeypatch.setenv("SQT_EXTERNAL_DIRS", "rel/x")
        result = dispatch("describe_effective_config", {})
        rows = {row["name"]: row for row in result["settings"]}
        assert rows["SQT_EXTERNAL_DIRS"]["value"] is None
        assert rows["SQT_EXTERNAL_DIRS"]["set"] is True
        assert any("SQT_EXTERNAL_DIRS" in w for w in result["warnings"])
        assert "rel/x" not in json.dumps(result)
