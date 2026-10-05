"""
`order_queue_ahead` and `order_lifetimes` are the loops of
`analysis.order_events`, number for number.

The two loops were 94% of `order_event_metrics` on a 2,000,000-event
session (8.0 s). The kernels run the same state machine over coded
columns; the loops stay in the module as the fallback, and they are the
reference here. Every comparison is of the whole result, floats by their
bits: a queue-ahead figure feeds a mean and a median, and a lifetime a
percentile, so a last-bit difference would be a different answer. See the
CHANGELOG entry of 2026-10-04.

The sessions below plant what a session-level comparison has to reach: an
opening snapshot, CLEARs, cancels of orders the window never saw, partial
fills, repeated order ids, non-finite prices and sizes, unknown actions,
NaT timestamps, -0.0 prices, and order ids, sides and timestamps in each
dtype and unit the library reads.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.analysis import order_events as oe

_cpp: Any = None
try:
    from standard_quant_tools import _sqt_core as _cpp  # type: ignore[attr-defined]

    HAS_CPP = hasattr(_cpp, "order_queue_ahead") and hasattr(_cpp, "order_lifetimes")
except ImportError:
    HAS_CPP = False

pytestmark = pytest.mark.skipif(not HAS_CPP, reason="order-event kernels not built")


def _same(got, want, path="result"):
    """Equal everywhere, floats to the bit (NaN equal to NaN)."""
    if isinstance(want, dict):
        assert list(got) == list(want), path
        for key in want:
            _same(got[key], want[key], f"{path}[{key!r}]")
    elif isinstance(want, (list, tuple, np.ndarray)):
        assert len(got) == len(want), path
        for i, (g, w) in enumerate(zip(got, want)):
            _same(g, w, f"{path}[{i}]")
    elif isinstance(want, (float, np.floating)):
        assert isinstance(got, (float, np.floating)), path
        assert np.float64(got).tobytes() == np.float64(want).tobytes(), (
            path,
            got,
            want,
        )
    else:
        assert type(got) is type(want) and got == want, (path, got, want)


def session(n=6_000, seed=0, snapshot=40, clears=2, opened_mid=True):
    """A market-by-order window with every case the loops branch on."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2026-03-02 14:30", tz="UTC").value
    stamps = start + np.cumsum(rng.integers(1, 2_000_000, n))
    rows = []
    live = []
    next_id = 1_000
    if opened_mid:
        # Orders resting before the window opened, never seen added.
        live.extend(
            [(i, "B" if i % 2 else "A", 100.0 + i % 5, 300.0) for i in range(30)]
        )
    clear_at = set(rng.choice(np.arange(snapshot + 10, n), size=clears, replace=False))
    for i in range(n):
        flags = 0
        if i < snapshot:
            kind, flags = "A", 32
        elif i in clear_at:
            rows.append((stamps[i], 0, "R", "N", np.nan, np.nan, 0))
            live = []
            continue
        else:
            u = rng.random()
            kind = (
                "A"
                if u < 0.42 or not live
                else (
                    "C"
                    if u < 0.76
                    else (
                        "F"
                        if u < 0.84
                        else "T" if u < 0.9 else "M" if u < 0.95 else "X"
                    )
                )
            )
        if kind == "A":
            side = "B" if rng.random() < 0.5 else "A"
            price = 100.0 + float(rng.integers(-6, 7))
            if rng.random() < 0.01:
                price = -0.0 if rng.random() < 0.5 else 0.0
            size = float(rng.integers(1, 9) * 100)
            # Now and then an id is reused, the way a venue recycles one.
            oid = (
                int(rng.integers(1_000, next_id))
                if rng.random() < 0.02 and next_id > 1_000
                else next_id
            )
            next_id += 1
            live.append((oid, side, price, size))
            rows.append((stamps[i], oid, "A", side, price, size, flags))
        elif kind in ("C", "F", "M"):
            j = len(live) - 1 - int(min(len(live) - 1, rng.exponential(len(live) / 6)))
            oid, side, price, size = live[j]
            take = size if kind == "C" else float(min(size, rng.integers(1, 9) * 100))
            rows.append((stamps[i], oid, kind, side, price, take, 0))
            if kind == "C" or take >= size:
                live.pop(j)
            elif kind == "F":
                live[j] = (oid, side, price, size - take)
        else:
            rows.append((stamps[i], 0, kind, "N", 100.0, 100.0, 0))
    frame = pd.DataFrame(
        rows,
        columns=["timestamp", "order_id", "action", "side", "price", "size", "flags"],
    )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"].astype("int64"), utc=True)
    frame["order_id"] = frame["order_id"].astype(np.uint64)
    # Non-finite prices and sizes on a few live rows, and NaT stamps.
    bad = rng.choice(len(frame), size=max(len(frame) // 200, 1), replace=False)
    frame.loc[bad[: len(bad) // 2], "price"] = np.nan
    frame.loc[bad[len(bad) // 2 :], "size"] = np.inf
    frame.loc[rng.choice(len(frame), size=5, replace=False), "timestamp"] = pd.NaT
    return frame


def _native(frame, sides=True, stamps=True):
    """Whether the frame takes the native passes (asserted, so a test of
    them cannot pass by running the loops twice)."""
    codes = oe._native_codes(frame, sides=True, stamps=True)
    return (
        codes is not None
        and (codes["sides"] is not None or not sides)
        and (codes["stamps"] is not None or not stamps)
    )


def _both(frame):
    native = oe.order_event_metrics(frame)
    try:
        oe.HAS_CPP = False
        python = oe.order_event_metrics(frame)
    finally:
        oe.HAS_CPP = True
    return native, python


class TestTheKernelsAreTheLoops:
    @pytest.mark.parametrize("seed", range(6))
    def test_whole_sessions(self, seed):
        frame = session(seed=seed, opened_mid=seed % 2 == 0)
        assert _native(frame)
        native, python = _both(frame)
        _same(native, python)

    def test_the_passes_directly(self):
        """Below the size gate too: each pass against its loop."""
        for n in (1, 3, 20, 200):
            frame = session(n=n + 45, seed=n, snapshot=min(5, n)).iloc[:n]
            codes = oe._native_codes(frame, sides=True, stamps=True)
            _same(oe._queue_pass_native(frame, codes), oe._queue_pass_python(frame))
            _same(oe._lifetime_pass_native(codes), oe._lifetime_pass_python(frame))

    @pytest.mark.parametrize(
        "kind", ["int64", "object_int", "str", "float", "pandas_str"], ids=str
    )
    def test_order_ids_of_every_kind(self, kind):
        frame = session(seed=11)
        ids = frame["order_id"].astype(np.int64)
        frame["order_id"] = {
            "int64": ids,
            "object_int": ids.astype(object),
            "str": ids.map(lambda v: f"o{v}"),
            "float": ids.astype(np.float64),
            "pandas_str": ids.map(lambda v: f"o{v}").astype("string"),
        }[kind]
        assert _native(frame)
        _same(*_both(frame))

    def test_integer_sides(self):
        frame = session(seed=12)
        frame["side"] = frame["side"].map({"B": 1, "A": -1, "N": 0})
        assert _native(frame)
        _same(*_both(frame))

    @pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
    @pytest.mark.parametrize("tz", [None, "UTC", "America/Chicago"])
    def test_timestamps_in_every_unit(self, unit, tz):
        frame = session(seed=13)
        stamps = (
            frame["timestamp"].dt.tz_convert(tz)
            if tz
            else frame["timestamp"].dt.tz_localize(None)
        )
        frame["timestamp"] = stamps.astype(
            f"datetime64[{unit}, {tz}]" if tz else f"datetime64[{unit}]"
        )
        assert _native(frame)
        _same(*_both(frame))

    def test_timestamps_as_strings_are_parsed_as_the_loop_parses_them(self):
        frame = session(n=800, seed=14)
        frame["timestamp"] = (
            frame["timestamp"].astype(str).where(frame["timestamp"].notna(), None)
        )
        assert _native(frame)
        _same(*_both(frame))


class TestWhatTheCodesCannotRepresentRunsTheLoop:
    def test_a_missing_order_id_on_a_cancel(self):
        frame = session(seed=15)
        frame["order_id"] = frame["order_id"].astype(object)
        frame.loc[frame.index[frame["action"] == "C"][3], "order_id"] = None
        assert oe._native_codes(frame, sides=True, stamps=True) is None
        _same(*_both(frame))

    def test_a_missing_side_on_an_add(self):
        """The queue runs the loop; the lifetimes, which never read the
        side, stay native."""
        frame = session(seed=16)
        frame.loc[frame.index[frame["action"] == "A"][50], "side"] = np.nan
        codes = oe._native_codes(frame, sides=True, stamps=True)
        assert codes is not None and codes["sides"] is None
        _same(*_both(frame))

    def test_a_missing_side_on_a_trade_is_not_read(self):
        frame = session(seed=17)
        frame.loc[frame.index[frame["action"] == "T"][:5], "side"] = None
        assert _native(frame)
        _same(*_both(frame))

    def test_timestamps_that_parse_to_nat_are_skipped_as_the_loop_skips_them(self):
        """An unparseable string and a stamp in another zone in an object
        column: pd.to_datetime(errors="coerce") makes both NaT, and both
        paths skip those rows."""
        frame = session(n=300, seed=18)
        frame["timestamp"] = frame["timestamp"].astype(object)
        frame.loc[7, "timestamp"] = "not a time"
        frame.loc[9, "timestamp"] = pd.Timestamp(
            "2026-03-02 09:30", tz="America/New_York"
        )
        assert _native(frame)
        _same(*_both(frame))


class TestSecondsAsPandasCountsThem:
    @pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
    def test_total_seconds_matches_the_scalar(self, unit):
        """Timedelta.total_seconds() is days * 86400 + seconds plus
        microseconds / 1e6 in floating point, nanoseconds dropped by
        flooring -- not a division of the duration. Held to the scalar on
        sub-microsecond, negative, multi-day and 280-year durations."""
        rng = np.random.default_rng(1)
        per_second = {"s": 1, "ms": 10**3, "us": 10**6, "ns": 10**9}[unit]
        scales = [
            1,
            999,
            10**6,
            10**9,
            86_400 * per_second,
            280 * 365 * 86_400 * per_second,
        ]
        values = np.concatenate(
            [rng.integers(-s, s, 400) for s in scales if s < 2**62]
            + [np.array([0, 1, -1])]
        )
        got = oe._total_seconds(values, unit)
        want = np.array(
            [pd.Timedelta(int(v), unit=unit).total_seconds() for v in values]
        )
        assert got is not None
        assert got.tobytes() == want.tobytes()

    def test_a_sub_microsecond_lifetime_reads_zero_on_both(self):
        """500 ns is 0.0 seconds to total_seconds(), and 1,500 ns is 1e-06."""
        t0 = pd.Timestamp("2026-03-02 14:30", tz="UTC")
        frame = pd.DataFrame(
            {
                "timestamp": [
                    t0,
                    t0,
                    t0 + pd.Timedelta(500, "ns"),
                    t0 + pd.Timedelta(1_500, "ns"),
                ],
                "order_id": np.array([1, 2, 1, 2], dtype=np.uint64),
                "action": ["A", "A", "C", "F"],
                "side": ["B", "B", "B", "B"],
                "price": [100.0] * 4,
                "size": [10.0] * 4,
            }
        )
        codes = oe._native_codes(frame, stamps=True)
        native = oe._lifetime_pass_native(codes)
        _same(native, oe._lifetime_pass_python(frame))
        assert list(native[1]) == [0.0] and list(native[0]) == [1e-06]


class TestTheBindings:
    def _args(self, n=8):
        return (
            np.arange(n, dtype=np.int64),
            n,
            np.zeros(n, dtype=np.int64),
            np.zeros(n, dtype=np.int64),
            np.full(n, 100.0),
            np.full(n, 10.0),
            np.zeros(n, dtype=bool),
        )

    def test_an_action_code_outside_the_vocabulary_is_refused(self):
        args = list(self._args())
        args[2] = args[2].copy()
        args[2][3] = 7
        with pytest.raises(ValueError, match="action code"):
            _cpp.order_queue_ahead(*args)

    def test_an_order_code_outside_its_range_is_refused_on_an_add(self):
        args = list(self._args())
        args[0] = args[0].copy()
        args[0][2] = 8
        with pytest.raises(ValueError, match="outside"):
            _cpp.order_queue_ahead(*args)
        with pytest.raises(ValueError, match="outside"):
            _cpp.order_lifetimes(
                args[0], 8, args[2], args[6], np.zeros(8, dtype=np.int64)
            )

    def test_but_not_on_a_row_the_loops_do_not_key(self):
        """A trade's order id is never looked up, so any code passes."""
        args = list(self._args())
        args[0] = np.full(8, -1, dtype=np.int64)
        args[2] = np.full(8, 4, dtype=np.int64)
        ahead, *_ = _cpp.order_queue_ahead(*args)
        assert ahead.size == 0

    def test_lengths_must_agree(self):
        args = list(self._args())
        args[4] = np.full(7, 100.0)
        with pytest.raises(ValueError, match="same length"):
            _cpp.order_queue_ahead(*args)

    def test_a_float_code_array_is_refused_not_truncated(self):
        args = list(self._args())
        args[0] = args[0].astype(float)
        with pytest.raises(ValueError, match="integer"):
            _cpp.order_queue_ahead(*args)

    def test_concurrent_calls_keep_their_own_sessions(self):
        """Both kernels release the GIL; eight threads on eight sessions
        each get their own session's answer."""
        frames = [session(n=3_000, seed=40 + i) for i in range(8)]
        want = [oe.order_event_metrics(f) for f in frames]
        got: list = [None] * len(frames)

        def work(i):
            for _ in range(3):
                got[i] = oe.order_event_metrics(frames[i])

        threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=120)
        for g, w in zip(got, want):
            _same(g, w)

    def test_the_docstrings_name_the_contract(self):
        for fn in (_cpp.order_queue_ahead, _cpp.order_lifetimes):
            assert "ValueError" in fn.__doc__ and "analysis.order_events" in fn.__doc__
        assert "bit for bit" in _cpp.order_queue_ahead.__doc__
