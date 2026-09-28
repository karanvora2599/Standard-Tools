"""
Promotions under concurrency, and a log that survives an interrupted append.

A promotion reads the current stage, checks the move, and appends the
decision. Unserialised, two callers could both check against the same old
stage and both append: a model archived and live in staging at once, with
two records claiming the same `from_stage`. The invariant these pin is the
one that makes a stage history mean anything -- every record's
`from_stage` is the previous record's `to_stage` -- and it holds however
the threads interleave.

A crash part-way through an append leaves an unterminated fragment at the
end of the log. That is not an edit and must not make the stage
unreadable; a bad line anywhere else still is, and still does.

Most of these use a placeholder `manifest.json`, which is all the lifecycle
checks for, so no model is trained; the two that need a real package say so.
"""

from __future__ import annotations

import sys
import threading
from uuid import uuid4

import pytest

from standard_quant_tools import _filelock
from standard_quant_tools.artifact_store import LocalArtifactStore
from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling import artifacts as _artifacts
from standard_quant_tools.modeling.registry.lifecycle import (
    PROMOTIONS_FILE,
    PROMOTIONS_LOCK_FILE,
    current_stage,
    promote,
    promotions,
)
from standard_quant_tools.modeling.registry.mirror import MIRROR_URL_ENV

TRIALS = 20
REASON = "reviewed the walk-forward evidence"


@pytest.fixture(autouse=True)
def _own_runs_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.delenv(MIRROR_URL_ENV, raising=False)


def _placeholder_model() -> str:
    model_id = f"mdl_{uuid4().hex[:12]}"
    directory = _artifacts.run_dir(model_id)
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text("{}", encoding="utf-8")
    return model_id


def _walk_to(model_id: str, stage: str) -> None:
    for step in ("validated", "staging", "production"):
        promote(model_id, step, REASON)
        if step == stage:
            return


def _race(*calls):
    """Run each zero-argument call in its own thread, released together;
    what each returned or raised, in order."""
    barrier = threading.Barrier(len(calls))
    outcomes = [None] * len(calls)

    def run(index, call):
        barrier.wait()
        try:
            outcomes[index] = ("accepted", call())
        except Exception as exc:  # noqa: BLE001 - the outcome is the point
            outcomes[index] = ("refused", exc)

    threads = [
        threading.Thread(target=run, args=(i, call)) for i, call in enumerate(calls)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    return outcomes


def _assert_consistent_chain(model_id: str) -> None:
    history = promotions(model_id)
    expected_from = "candidate"
    for record in history:
        assert record.from_stage == expected_from, (
            f"a record claims {record.from_stage!r} -> {record.to_stage!r} "
            f"but the model was at {expected_from!r}: two promotions were "
            "decided against the same stage"
        )
        expected_from = record.to_stage
    assert current_stage(model_id) == expected_from


def _log(model_id: str):
    return _artifacts.run_dir(model_id) / PROMOTIONS_FILE


def _lock_held_elsewhere(lock_path) -> bool:
    """Whether another handle holds the promotion lock right now."""
    with open(lock_path, "a+b") as handle:
        handle.seek(0)
        if sys.platform == "win32":
            import msvcrt

            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                return True
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            return False
        import fcntl

        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return True
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return False


class TestConcurrentPromotionsAreSerialised:
    def test_archive_and_demote_racing_leave_a_consistent_history(self):
        """Planted: from production, one caller archives while another
        demotes to staging. Either order is a legitimate history; two
        records both starting from production is not."""
        for _ in range(TRIALS):
            model_id = _placeholder_model()
            _walk_to(model_id, "production")
            outcomes = _race(
                lambda: promote(model_id, "archived", REASON),
                lambda: promote(model_id, "staging", REASON),
            )
            for status, value in outcomes:
                if status == "refused":
                    assert isinstance(value, ValidationError), value
            _assert_consistent_chain(model_id)

    def test_the_same_promotion_twice_is_made_once(self):
        """Planted: two callers both promote a candidate to validated. One
        is recorded; the other finds it already made."""
        for _ in range(TRIALS):
            model_id = _placeholder_model()
            outcomes = _race(
                lambda: promote(model_id, "validated", REASON),
                lambda: promote(model_id, "validated", REASON),
            )
            accepted = [value for status, value in outcomes if status == "accepted"]
            refused = [value for status, value in outcomes if status == "refused"]
            assert len(accepted) == 1, outcomes
            assert len(refused) == 1 and "already at 'validated'" in str(refused[0])
            assert [p.to_stage for p in promotions(model_id)] == ["validated"]

    def test_a_promotion_is_refused_when_no_lock_can_be_taken(self, monkeypatch):
        model_id = _placeholder_model()
        monkeypatch.setattr(_filelock, "acquire_lock", lambda path: None)
        with pytest.raises(ValidationError, match="could not be created or locked"):
            promote(model_id, "validated", REASON)
        assert promotions(model_id) == []

    def test_sequential_promotions_are_unchanged(self):
        """Null: one caller at a time records exactly what it always did."""
        model_id = _placeholder_model()
        _walk_to(model_id, "production")
        promote(model_id, "archived", REASON)
        history = promotions(model_id)
        assert [(p.from_stage, p.to_stage) for p in history] == [
            ("candidate", "validated"),
            ("validated", "staging"),
            ("staging", "production"),
            ("production", "archived"),
        ]
        lines = _log(model_id).read_bytes().splitlines()
        assert len(lines) == 4
        assert _log(model_id).read_bytes().endswith(b"\n")


class TestATornFinalLineIsNotAnEdit:
    # What an append interrupted part-way leaves: the start of a record,
    # with no closing brace and no newline.
    FRAGMENT = b'{"actor": "reviewer-2", "evidence": ["backtest_ref=sqt://eq'

    def test_the_stage_survives_an_interrupted_append(self):
        """Planted: the fragment a crash leaves at the end of the log. The
        stage is the last complete record, and the next promotion goes
        through."""
        model_id = _placeholder_model()
        promote(model_id, "validated", REASON)
        with open(_log(model_id), "ab") as handle:
            handle.write(self.FRAGMENT)
        assert current_stage(model_id) == "validated"
        promote(model_id, "staging", REASON)
        assert [p.to_stage for p in promotions(model_id)] == ["validated", "staging"]
        assert self.FRAGMENT not in _log(model_id).read_bytes()

    def test_the_fragment_is_kept_as_evidence_and_cut_off(self):
        model_id = _placeholder_model()
        promote(model_id, "validated", REASON)
        with open(_log(model_id), "ab") as handle:
            handle.write(self.FRAGMENT)
        promote(model_id, "staging", REASON)
        directory = _artifacts.run_dir(model_id)
        kept = list(directory.glob(".promotions.torn-*"))
        assert len(kept) == 1
        assert kept[0].read_bytes() == self.FRAGMENT
        assert all(line.strip() for line in _log(model_id).read_bytes().splitlines())

    def test_reading_repairs_it_too(self):
        model_id = _placeholder_model()
        promote(model_id, "validated", REASON)
        with open(_log(model_id), "ab") as handle:
            handle.write(self.FRAGMENT)
        assert [p.to_stage for p in promotions(model_id)] == ["validated"]
        assert _log(model_id).read_bytes().endswith(b"\n")
        assert self.FRAGMENT not in _log(model_id).read_bytes()

    def test_a_final_record_that_lost_only_its_newline_still_counts(self):
        """Null: a complete record without its newline is a record. The
        next append must not glue itself onto it."""
        model_id = _placeholder_model()
        promote(model_id, "validated", REASON)
        log = _log(model_id)
        log.write_bytes(log.read_bytes().rstrip(b"\n"))
        assert current_stage(model_id) == "validated"
        promote(model_id, "staging", REASON)
        assert [p.to_stage for p in promotions(model_id)] == ["validated", "staging"]
        assert not list(_artifacts.run_dir(model_id).glob(".promotions.torn-*"))

    def test_a_bad_line_in_the_middle_is_still_an_edit(self):
        """Null: the torn-tail rule is for the END of the log. A line that
        does not parse with a record after it means the file was edited."""
        model_id = _placeholder_model()
        promote(model_id, "validated", REASON)
        with open(_log(model_id), "ab") as handle:
            handle.write(b"this line was typed in by hand\n")
        promote_blocked = None
        with pytest.raises(ValidationError, match="cannot be trusted"):
            current_stage(model_id)
        try:
            promote(model_id, "staging", REASON)
        except ValidationError as exc:
            promote_blocked = exc
        assert promote_blocked is not None and "cannot be trusted" in str(
            promote_blocked
        )


class TestTheLockStaysOutOfThePackage:
    def test_the_lock_file_is_not_a_package_file(self):
        """Null: the lock is dot-prefixed, so a store listing -- and with it
        verification, mirroring and pulling -- never sees it."""
        model_id = _placeholder_model()
        promote(model_id, "validated", REASON)
        directory = _artifacts.run_dir(model_id)
        assert (directory / PROMOTIONS_LOCK_FILE).exists()
        listed = LocalArtifactStore().list(model_id)
        assert f"{model_id}/{PROMOTIONS_LOCK_FILE}" not in listed
        assert listed == [f"{model_id}/manifest.json", f"{model_id}/{PROMOTIONS_FILE}"]

    def test_verification_does_not_report_the_lock(self, patched_multi_factory):
        from standard_quant_tools.modeling.registry.package import (
            verify_model_package,
        )

        from .test_scoring import _dataset_spec, _train_a_model_with_spec

        model_id = _train_a_model_with_spec(_dataset_spec(), dataset_id="ds_lock_null")
        promote(model_id, "validated", REASON)
        report = verify_model_package(model_id)
        assert report.ok
        assert not any(name.startswith(".") for name in report.unhashed)
        assert PROMOTIONS_FILE in report.unhashed


class TestAPullWritesTheLogUnderTheSameLock:
    def test_the_pulled_promotion_log_is_written_holding_the_lock(
        self, patched_multi_factory, monkeypatch
    ):
        """Planted: a pull that replaces the local promotion log while a
        promotion appends to it loses that promotion. The write has to be
        made holding the lock `promote` takes."""
        from standard_quant_tools.artifact_store import (
            fsspec_available,
            store_from_url,
        )
        from standard_quant_tools.modeling.registry.package import (
            mirror_model_package,
            pull_model_package,
        )

        from .test_scoring import _dataset_spec, _train_a_model_with_spec

        if not fsspec_available():
            pytest.skip("fsspec is not installed")
        model_id = _train_a_model_with_spec(_dataset_spec(), dataset_id="ds_pull_lock")
        promote(model_id, "validated", REASON)
        store = store_from_url(f"memory://sqt-lock/{uuid4().hex}")
        mirror_model_package(model_id, store)

        lock_path = _artifacts.run_dir(model_id) / PROMOTIONS_LOCK_FILE
        held_while_writing = []
        real_put = LocalArtifactStore.put

        def watching_put(self, key, data):
            if key.endswith(f"/{PROMOTIONS_FILE}"):
                held_while_writing.append(_lock_held_elsewhere(lock_path))
            return real_put(self, key, data)

        monkeypatch.setattr(LocalArtifactStore, "put", watching_put)
        report = pull_model_package(model_id, store, overwrite=True)
        assert report.ok
        assert held_while_writing == [True]
