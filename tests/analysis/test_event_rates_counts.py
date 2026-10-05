"""
`event_rates` counts actions by object identity, in the order `value_counts` gave.

`event_rates` spent most of its time in `value_counts` on the action column
(about 0.1 s of 0.137 s at 2,000,000 events), and in copying the kept rows
of the two columns it reads. It now counts a column of one-letter codes by
the few objects the rows point at, reads datetime stamps as integers, and
selects rows from the arrays (the CHANGELOG entry of 2026-10-04).

The key order of `counts_by_action` is part of the result. `value_counts`
lists values by count, largest first; equal counts come in the order its
final sort leaves them: the order of first appearance under pandas 3, whose
sort is stable, and under pandas 2 whatever its default quicksort makes of
the counts in that same first-appearance order. The new count rebuilds that
first-appearance order and calls the same sort, so it is held here to the
code as it stood (`_before_event_rates`) on inputs built to have many ties,
missing values, other types, other dtypes and snapshot rows, under both
pandas versions: every key in the same order, every number to the bit, and
the same Python warnings.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.analysis.order_events import (
    _VALUE_COUNTS_SORT,
    ADD,
    CANCEL,
    FILL,
    TRADE,
    _counts_by_identity,
    _elapsed_seconds,
    _snapshot_mask,
    event_rates,
)

T0 = pd.Timestamp("2026-03-02 14:30:00", tz="UTC")


def _before_event_rates(events: pd.DataFrame) -> Dict[str, Any]:
    """`event_rates` as it stood, comments cut."""
    snapshot = _snapshot_mask(events)
    stamps, actions = events["timestamp"], events["action"]
    if snapshot.any():
        stamps, actions = stamps[~snapshot], actions[~snapshot]
    seconds = _elapsed_seconds(stamps)
    counts = actions.value_counts().to_dict()
    total = int(len(actions))
    per_action = {str(k): int(v) for k, v in counts.items()}
    rates = (
        {str(k): float(v) / seconds for k, v in per_action.items()} if seconds else {}
    )
    adds = per_action.get(ADD, 0)
    cancels = per_action.get(CANCEL, 0)
    trades = per_action.get(TRADE, 0) or per_action.get(FILL, 0)
    return {
        "n_events": total,
        "n_snapshot_events": int(snapshot.sum()),
        "elapsed_seconds": seconds,
        "events_per_second": (total / seconds) if seconds else None,
        "counts_by_action": per_action,
        "rates_by_action": rates,
        "cancel_to_add": (cancels / adds) if adds else None,
        "cancel_to_trade": (cancels / trades) if trades else None,
    }


def _outcome(fn: Any, *args: Any) -> Any:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result: Any = ("returned", fn(*args))
        except Exception as exc:  # the refusal is part of what is compared
            result = ("raised", type(exc), str(exc))
    return result, [(w.category, str(w.message)) for w in caught]


def _same(a: Any, b: Any, where: str = "result") -> None:
    """Equal to the bit, dicts in their key order."""
    if isinstance(a, float):
        assert type(b) is float, (where, type(b))
        assert np.float64(a).tobytes() == np.float64(b).tobytes(), (where, a, b)
    elif isinstance(a, dict):
        assert type(b) is dict, where
        assert list(a) == list(b), (where, list(a), list(b))
        for key in a:
            _same(a[key], b[key], f"{where}[{key!r}]")
    elif isinstance(a, (list, tuple)):
        assert type(b) is type(a) and len(a) == len(b), where
        for i, (x, y) in enumerate(zip(a, b)):
            _same(x, y, f"{where}[{i}]")
    else:
        assert type(b) is type(a), (where, type(a), type(b))
        assert a == b, (where, a, b)


def _both(events: pd.DataFrame) -> Dict[str, Any]:
    old = _outcome(_before_event_rates, events)
    new = _outcome(event_rates, events)
    _same(old, new)
    return new[0][1] if new[0][0] == "returned" else {}


def _frame(
    actions: List[Any],
    *,
    seconds: Optional[np.ndarray] = None,
    flags: Optional[np.ndarray] = None,
    snapshot: Optional[List[Any]] = None,
    dtype: Any = object,
) -> pd.DataFrame:
    n = len(actions)
    if seconds is None:
        seconds = np.arange(n, dtype=float) * 0.25
    if dtype is object:
        # Held as objects on both pandas versions: pandas 3 would read a
        # list of strings as its string dtype.
        values = np.empty(n, dtype=object)
        values[:] = actions
        column: Any = pd.Series(values, dtype=object)
    else:
        column = pd.array(actions, dtype=dtype)
    columns: Dict[str, Any] = {
        "timestamp": T0 + pd.to_timedelta(seconds, unit="s"),
        "order_id": np.arange(n, dtype=np.int64),
        "action": column,
    }
    if flags is not None:
        columns["flags"] = flags
    if snapshot is not None:
        columns["snapshot"] = pd.array(snapshot, dtype=object)
    return pd.DataFrame(columns)


def _ties(seed: int, k: int, per: int, letters: bool) -> List[Any]:
    """`k` codes, each `per` times, shuffled: every count ties."""
    rng = np.random.default_rng(seed)
    if letters:
        alphabet = [chr(c) for c in range(ord("A"), ord("A") + 26)] + [
            chr(c) for c in range(ord("a"), ord("a") + 26)
        ]
        codes = [alphabet[i % len(alphabet)] for i in range(k)]
        codes = [c if i < 52 else c + c for i, c in enumerate(codes)]
    else:
        codes = [f"code{i}" for i in range(k)]
    out = [codes[i] for i in range(k) for _ in range(per)]
    rng.shuffle(out)
    return out


# ── the count order ──────────────────────────────────────────────────────


@pytest.mark.parametrize("letters", [True, False])
@pytest.mark.parametrize("k", [2, 5, 8, 16, 17, 30, 64, 65])
@pytest.mark.parametrize("seed", range(4))
def test_equal_counts_come_in_the_same_order(seed, k, letters):
    """Every code ties with every other, in a shuffled first appearance:
    the order of the keys is the sort's alone. 16 and 17 straddle the size
    below which numpy's quicksort sorts by insertion; 64 and 65 the most
    objects counted by identity."""
    out = _both(_frame(_ties(seed, k, 3, letters)))
    assert len(out["counts_by_action"]) == k


@pytest.mark.parametrize("seed", range(40))
def test_a_few_count_levels_with_ties_inside_each(seed):
    """Three count levels over three to ten codes. Inside a level, pandas
    2's quicksort and a stable sort can order the ties differently (on an
    x86 build of numpy 2 they do for about a third of these seeds), so a
    count sorted the other way fails here."""
    rng = np.random.default_rng(seed)
    codes = list("ACFMRTNXYZ")[: int(rng.integers(3, 11))]
    levels = rng.integers(1, 4, len(codes)) * 5
    actions = [c for c, m in zip(codes, levels) for _ in range(int(m))]
    rng.shuffle(actions)
    _both(_frame(actions))


def test_under_pandas_3_equal_counts_keep_their_first_appearance():
    """The order `value_counts` gives, stated: largest first, and among
    equal counts the order each first appears in when its sort is
    stable."""
    if _VALUE_COUNTS_SORT.get("kind") != "stable":
        pytest.skip("pandas 2 orders equal counts by its quicksort")
    actions = ["T", "C", "A", "C", "M", "A", "T", "F", "F"]
    out = _both(_frame(actions))
    assert list(out["counts_by_action"]) == ["T", "C", "A", "F", "M"]


@pytest.mark.parametrize("seed", range(4))
def test_the_same_value_as_different_objects_is_one_key(seed):
    """Strings equal in value but built separately -- three objects that
    read 'Add', two that read 'Can' -- are one key each, wherever the
    copies fall; one object per row is too many to read by identity and
    `value_counts` counts it."""
    rng = np.random.default_rng(seed)
    adds = ["".join(["A", "dd"]) for _ in range(3)]
    cans = ["".join(["Ca", "n"]) for _ in range(2)]
    assert len({id(s) for s in adds + cans}) == 5
    pool = adds + cans
    actions = [pool[i] for i in rng.integers(0, 5, 3000)]
    frame = _frame(actions)
    out = _both(frame)
    assert list(out["counts_by_action"]) in (["Add", "Can"], ["Can", "Add"])
    assert _counts_by_identity(frame["action"].to_numpy()) is not None
    _both(_frame(["".join(["A", "dd"]) if i % 3 else "Can" for i in range(3000)]))


@pytest.mark.parametrize(
    "actions",
    [
        ["A", "C", None, "A", np.nan, "C", "T", None],
        ["A", "C", pd.NA, "A", "T"],
        [1, "1", "A", 1, "C", "1"],
        [1, True, 1.0, "A", "A", 2],
        [b"A", "A", b"A", "C"],
        ["A", "C", 2.5, "A"],
        [None, None, None],
        [np.nan, np.nan],
    ],
    ids=range(8),
)
def test_missing_values_and_other_types_count_as_they_did(actions):
    _both(_frame(actions * 40))


def test_a_string_subclass_counts_as_it_did():
    class Code(str):
        pass

    _both(_frame([Code("A"), "A", Code("C"), "C", "C"] * 30))


@pytest.mark.parametrize(
    "dtype",
    ["string[python]", "string[pyarrow]", "category", "str"],
)
def test_other_column_dtypes_count_as_they_did(dtype):
    if dtype == "string[pyarrow]":
        pytest.importorskip("pyarrow")
    if dtype == "str" and int(pd.__version__.split(".")[0]) < 3:
        pytest.skip("pandas 2 has no default string dtype")
    actions = _ties(3, 9, 4, True) + [None, "A"]
    _both(_frame(actions, dtype=dtype))


def test_many_distinct_objects_count_as_they_did():
    """More objects than are read by identity: `value_counts` counts."""
    actions = [f"{i % 300}x" for i in range(6000)]
    _both(_frame(actions))
    values = np.empty(len(actions), dtype=object)
    values[:] = actions
    assert _counts_by_identity(values) is None


# ── snapshot rows and the clock ──────────────────────────────────────────


@pytest.mark.parametrize("seed", range(6))
def test_snapshot_rows_are_neither_counted_nor_first(seed):
    """Snapshot rows open the window and are the first appearance of some
    codes; dropping them changes which code comes first among ties."""
    rng = np.random.default_rng(seed)
    body = _ties(seed, 6, 5, True)
    head = list(rng.choice(["F", "A", "M", "Q"], size=8))
    actions = head + body
    flags = np.zeros(len(actions), dtype=np.uint8)
    flags[: len(head)] = 32
    seconds = np.concatenate([np.zeros(len(head)), 1.0 + np.arange(len(body))])
    _both(_frame(actions, seconds=seconds, flags=flags))
    snapshot: List[Any] = [True] * len(head) + [False] * len(body)
    snapshot[-1] = None
    _both(_frame(actions, seconds=seconds, snapshot=snapshot))


def _with_stamps(stamps: Any, n: int, *, snapshot_rows: int = 0) -> pd.DataFrame:
    frame = _frame((["A", "C", "T", "F"] * n)[:n])
    frame["timestamp"] = stamps
    if snapshot_rows:
        flags = np.zeros(n, dtype=np.uint8)
        flags[:snapshot_rows] = 32
        frame["flags"] = flags
    return frame


@pytest.mark.parametrize(
    "make",
    [
        lambda n: pd.date_range("2026-03-02 14:30", periods=n, freq="137ms", tz="UTC"),
        lambda n: pd.date_range("2026-03-02 14:30", periods=n, freq="1s").as_unit("s"),
        lambda n: pd.date_range("2026-03-02 14:30", periods=n, freq="3us").as_unit(
            "us"
        ),
        lambda n: pd.date_range(
            "2026-03-02 09:30", periods=n, freq="1min", tz="America/New_York"
        ),
        lambda n: pd.Series(pd.date_range("2026-03-02", periods=n, freq="1s", tz="UTC"))
        .sample(frac=1.0, random_state=1)
        .to_numpy(),
        lambda n: [T0 + pd.Timedelta(seconds=i) if i % 7 else pd.NaT for i in range(n)],
        lambda n: [pd.NaT] * n,
        lambda n: [T0] * n,
        lambda n: [T0] + [pd.NaT] * (n - 1),
        lambda n: [str(T0 + pd.Timedelta(seconds=i)) for i in range(n)],
        lambda n: ["2026-03-02 14:30:00", "03/02/2026 14:31"] * (n // 2),
        lambda n: [T0 + pd.Timedelta(seconds=i) for i in range(n)],
    ],
    ids=range(12),
)
@pytest.mark.parametrize("snapshot_rows", [0, 3])
def test_the_clock_reads_the_same(make, snapshot_rows):
    n = 40
    try:
        stamps = make(n)
    except Exception:
        pytest.skip("this pandas cannot build these stamps")
    _both(_with_stamps(stamps, n, snapshot_rows=snapshot_rows))


def test_snapshot_rows_holding_the_extreme_stamps_do_not_set_the_clock():
    n = 50
    stamps = list(T0 + pd.to_timedelta(np.arange(n), unit="s"))
    stamps[0] = T0 - pd.Timedelta(days=3)
    stamps[1] = T0 + pd.Timedelta(days=3)
    out = _both(_with_stamps(stamps, n, snapshot_rows=2))
    assert out["elapsed_seconds"] == float(n - 3)


def test_an_empty_window_and_a_single_event():
    _both(_frame([]))
    _both(_frame(["A"]))
    flags = np.full(5, 32, dtype=np.uint8)
    _both(_frame(["A"] * 5, flags=flags))


# ── at size ──────────────────────────────────────────────────────────────


def test_a_session_of_two_hundred_thousand_events():
    """A realistic mix, shaped like a liquid US equity's day, with the
    opening snapshot flagged."""
    rng = np.random.default_rng(7)
    n = 200_000
    actions = rng.choice(
        ["A", "C", "F", "T", "M"], size=n, p=[0.45, 0.40, 0.06, 0.06, 0.03]
    ).astype(object)
    gaps = rng.exponential(23_400e9 / n, size=n).astype(np.int64) + 1
    stamps = pd.to_datetime(T0.value + np.cumsum(gaps), utc=True)
    flags = np.zeros(n, dtype=np.uint8)
    flags[:200] = 32
    frame = pd.DataFrame(
        {
            "timestamp": stamps,
            "order_id": np.arange(n),
            "action": actions,
            "flags": flags,
        }
    )
    out = _both(frame)
    assert out["n_events"] == n - 200
    values = frame["action"].to_numpy()[200:]
    assert _counts_by_identity(values) is not None
