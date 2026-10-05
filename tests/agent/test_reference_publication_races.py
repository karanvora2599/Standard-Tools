"""
One reference, one value -- also when two publishers race for it.

A reference promises that resolving it twice gives the same value, and the
collision rule that keeps the promise was an `exists()` check followed by a
write: two publishers to one `(run_id, name)` both passed the check, both
were told they had published, and the reference resolved to whichever
rename landed last. These race two threads through a barrier and require
exactly one winner every time, with the loser refused in the library's own
words -- never a raw OS error -- and the stored value the winner's.

The sidecar beside a value decides whether a reference names a stored
value or an external dataset. A damaged one used to read as "no sidecar",
which resolved an external registration as whatever Parquet shared its
name; it is refused now, by name.
"""

from __future__ import annotations

import threading

import pandas as pd
import pytest

from standard_quant_tools.agent.runtimes import handoff
from standard_quant_tools.backtest.artifacts import load_artifact, save_artifact
from standard_quant_tools.error import ValidationError

TRIALS = 30


@pytest.fixture
def runs(tmp_path, monkeypatch):
    root = tmp_path / "runs"
    monkeypatch.setenv("SQT_RUNS_DIR", str(root))
    return root


def _race(*calls):
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


def _one_winner(outcomes, refusal: str) -> int:
    """The index of the single accepted call; every other was refused
    with `refusal` as a ValidationError."""
    accepted = [i for i, (status, _v) in enumerate(outcomes) if status == "accepted"]
    assert len(accepted) == 1, f"{len(accepted)} publishers were accepted: {outcomes}"
    for status, value in outcomes:
        if status == "refused":
            assert isinstance(value, ValidationError), repr(value)
            assert refusal in str(value)
    return accepted[0]


def _series(value: float) -> pd.Series:
    return pd.Series(
        [value] * 50, index=pd.date_range("2025-01-01", periods=50), name="equity"
    )


def _no_temp_files(root) -> bool:
    return not list(root.rglob(".*.tmp"))


class TestSaveArtifactHasOneWinner:
    def test_two_saves_racing_to_one_name(self, runs):
        """Planted: two different series saved to the same (run_id, name)."""
        for trial in range(TRIALS):
            outcomes = _race(
                lambda: save_artifact(_series(1.0), f"race{trial}", "equity"),
                lambda: save_artifact(_series(2.0), f"race{trial}", "equity"),
            )
            winner = _one_winner(outcomes, "already exists")
            stored = load_artifact(outcomes[winner][1])["equity"].iloc[0]
            assert stored == (1.0, 2.0)[winner]
        assert _no_temp_files(runs)

    def test_overwrite_still_replaces(self, runs):
        """Null: overwrite=True is the deliberate replacement it always was."""
        uri = save_artifact(_series(1.0), "run1", "equity")
        save_artifact(_series(2.0), "run1", "equity", overwrite=True)
        assert load_artifact(uri)["equity"].iloc[0] == 2.0
        assert _no_temp_files(runs)

    def test_a_second_save_is_refused_and_leaves_the_first(self, runs):
        uri = save_artifact(_series(1.0), "run1", "equity")
        with pytest.raises(ValidationError, match="already exists"):
            save_artifact(_series(2.0), "run1", "equity")
        assert load_artifact(uri)["equity"].iloc[0] == 1.0


class TestPublishHasOneWinner:
    def test_two_publishes_racing_to_one_reference(self, runs):
        """Planted: both callers used to receive the same reference, which
        then resolved to whichever value landed last."""
        for trial in range(TRIALS):
            outcomes = _race(
                lambda: handoff.publish(
                    _series(1.0), "equity_curve", f"pub{trial}", "curve"
                ),
                lambda: handoff.publish(
                    _series(2.0), "equity_curve", f"pub{trial}", "curve"
                ),
            )
            winner = _one_winner(outcomes, "already published")
            resolved = handoff.resolve(outcomes[winner][1], expect="equity_curve")
            assert resolved.iloc[0] == (1.0, 2.0)[winner]
        assert _no_temp_files(runs)


class TestExternalRegistrationHasOneWinner:
    @staticmethod
    def _tape(path, price: float) -> str:
        pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-03-02 09:30", periods=200, freq="s"),
                "price": [price] * 200,
                "size": [100.0] * 200,
            }
        ).to_parquet(path, index=False)
        return str(path)

    def test_two_registrations_racing_to_one_reference(self, runs, tmp_path):
        """Planted: two different files registered under one (run_id, name).
        Exactly one registration lands, and the reference describes the
        winner's file."""
        first = self._tape(tmp_path / "tape_a.parquet", 10.0)
        second = self._tape(tmp_path / "tape_b.parquet", 20.0)
        for trial in range(TRIALS // 3):
            outcomes = _race(
                lambda: handoff.publish_external(
                    first, "tick_tape", f"ext{trial}", "t"
                ),
                lambda: handoff.publish_external(
                    second, "tick_tape", f"ext{trial}", "t"
                ),
            )
            winner = _one_winner(outcomes, "already registered")
            ref = outcomes[winner][1][0]
            assert handoff.describe(ref)["path"].endswith(
                ("tape_a.parquet", "tape_b.parquet")[winner]
            )
        assert _no_temp_files(runs)

    def test_a_publish_cannot_take_over_a_registered_reference(self, runs, tmp_path):
        """Planted: the sidecar IS an external registration, so a publish to
        the same pair used to replace it and silently repoint every holder
        at the published value."""
        tape = self._tape(tmp_path / "tape.parquet", 10.0)
        ref, _handle = handoff.publish_external(tape, "tick_tape", "run1", "tape")
        frame = pd.read_parquet(tape)
        with pytest.raises(ValidationError, match="already published"):
            handoff.publish(frame, "trade_log", "run1", "tape")
        assert handoff.describe(ref)["storage"] == "external"
        assert not (runs / "run1" / "tape.parquet").exists()


class TestADamagedSidecarIsRefused:
    @pytest.mark.parametrize(
        "damage",
        [b'{"kind": "tick_tape", "storage": "exter', b"{not json", b"[1, 2, 3]"],
        ids=["torn", "garbage", "not-an-object"],
    )
    def test_it_is_not_read_as_absent(self, runs, tmp_path, damage):
        """Planted: an external tape registered where a fetch would have put
        its Parquet (inside the runs directory, under the reference's own
        name). With the sidecar damaged, the reference used to resolve as
        a plain DataFrame -- the wrong storage, silently."""
        directory = runs / "run1"
        directory.mkdir(parents=True)
        tape = directory / "tape.parquet"
        pd.DataFrame({
            # A `tick_tape` requires a stamp. This test is about a damaged
            # sidecar, so the tape only has to be registrable.
            "timestamp": ["2024-03-05T14:30:00Z", "2024-03-05T14:30:01Z"],
            "price": [10.0, 10.5],
            "size": [100.0, 200.0],
        }).to_parquet(tape, index=False)
        ref, _handle = handoff.publish_external(str(tape), "tick_tape", "run1", "tape")
        sidecar = directory / "tape._handoff.json"
        sidecar.write_bytes(damage)
        for read in (handoff.resolve, handoff.describe):
            with pytest.raises(ValidationError, match="is damaged") as excinfo:
                read(ref)
            assert "tape._handoff.json" in str(excinfo.value)
            assert "fresh run_id" in str(excinfo.value)

    def test_a_missing_sidecar_still_resolves_a_published_value(self, runs):
        """Null: a value published before sidecars carried anything has
        none, and resolves as the stored value it is."""
        ref = handoff.publish(_series(3.0), "equity_curve", "run1", "curve")
        (runs / "run1" / "curve._handoff.json").unlink()
        assert handoff.resolve(ref, expect="equity_curve").iloc[0] == 3.0

    def test_a_healthy_sidecar_is_read(self, runs, tmp_path):
        ref = handoff.publish(_series(3.0), "equity_curve", "run1", "curve")
        assert handoff.describe(ref)["storage"] == "local"
