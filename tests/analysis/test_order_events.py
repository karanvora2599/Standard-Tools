"""
Order-level measures, against feeds built so the right answer is known.

WHY EACH TEST PLANTS A NUMBER. "It returned a float" is not evidence for a
queue statistic. Every fixture below has a hand-computable answer -- a queue
of exactly 300 ahead, a lifetime of exactly 5 seconds, two cancels per add
-- so a measure that silently counted the wrong events would come back
plausible and wrong rather than obviously broken.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.analysis import order_events as order_events_module
from standard_quant_tools.analysis.order_events import (
    ORDER_EVENT_COLUMNS,
    event_rates,
    order_event_metrics,
    order_lifetimes,
    queue_positions,
)
from standard_quant_tools.error import ValidationError

T0 = pd.Timestamp("2026-03-02 14:30:00", tz="UTC")


def _events(rows) -> pd.DataFrame:
    """rows: (seconds, order_id, action, side, price, size)."""
    return pd.DataFrame(
        [
            {
                "timestamp": T0 + pd.Timedelta(seconds=s),
                "order_id": oid,
                "action": action,
                "side": side,
                "price": price,
                "size": size,
            }
            for s, oid, action, side, price, size in rows
        ]
    )


class TestQueuePosition:
    """The number aggregated depth cannot produce: how much is AHEAD."""

    def test_each_add_sees_what_is_already_resting(self) -> None:
        events = _events(
            [
                (0, 1, "A", "B", 100.0, 100),
                (1, 2, "A", "B", 100.0, 200),
                (2, 3, "A", "B", 100.0, 50),
            ]
        )
        queue = queue_positions(events)
        assert queue["n_adds"] == 3
        # 0 ahead, then 100, then 300.
        assert queue["mean_queue_ahead"] == pytest.approx((0 + 100 + 300) / 3)
        assert queue["share_joining_empty"] == pytest.approx(1 / 3)

    def test_a_different_price_is_a_different_queue(self) -> None:
        """The level is (side, price). An order at 99 does not queue behind
        one at 100."""
        events = _events(
            [
                (0, 1, "A", "B", 100.0, 500),
                (1, 2, "A", "B", 99.0, 100),
            ]
        )
        assert queue_positions(events)["mean_queue_ahead"] == 0.0

    def test_a_different_side_is_a_different_queue(self) -> None:
        events = _events(
            [
                (0, 1, "A", "B", 100.0, 500),
                (1, 2, "A", "A", 100.0, 100),
            ]
        )
        assert queue_positions(events)["mean_queue_ahead"] == 0.0

    def test_cancels_and_fills_shorten_the_queue(self) -> None:
        events = _events(
            [
                (0, 1, "A", "B", 100.0, 400),
                (1, 1, "C", "B", 100.0, 400),
                (2, 2, "A", "B", 100.0, 100),
            ]
        )
        # The 400 was withdrawn, so order 2 joins an empty level.
        assert queue_positions(events)["mean_queue_ahead"] == 0.0

    def test_a_clear_resets_every_level(self) -> None:
        """A CLEAR wipes the book. Carrying depth across one would report a
        queue accumulated over a boundary where none existed."""
        events = _events(
            [
                (0, 1, "A", "B", 100.0, 900),
                (1, 0, "R", "N", np.nan, np.nan),
                (2, 2, "A", "B", 100.0, 100),
            ]
        )
        assert queue_positions(events)["mean_queue_ahead"] == 0.0

    def test_an_empty_feed_reports_null_not_zero(self) -> None:
        events = _events([(0, 1, "C", "B", 100.0, 10)])
        queue = queue_positions(events)
        assert queue["n_adds"] == 0
        assert queue["mean_queue_ahead"] is None


def _opened_mid_session():
    """
    A window that opens with 1,000 shares already resting at 99.99 and no
    snapshot of them. Five 100-share orders join; 400 shares of an order
    placed before the window are cancelled; one more 100-share order joins.

    The real queue ahead of the six arrivals was 1,000, 1,100, 1,200,
    1,300, 1,400 and 1,100. What the window can see is only its own adds:
    0, 100, 200, 300, 400, and then the same 500 -- the cancelled order was
    never in the running total, so it takes nothing out of it.
    """
    rows = [(k, 1000 + k, "A", "B", 99.99, 100.0) for k in range(5)]
    rows.append((10, 1, "C", "B", 99.99, 400.0))
    rows.append((11, 1100, "A", "B", 99.99, 100.0))
    return _events(rows)


class TestAWindowThatOpensMidSession:
    """Every level starts empty when nothing told the window what was
    resting, so each queue figure counts only what arrived inside it. That
    was reported as market structure: no count, no warning, and a cancel of
    an order the window never saw was taken out of the orders it had."""

    def test_the_queue_figures_are_marked_as_lower_bounds(self) -> None:
        queue = queue_positions(_opened_mid_session())
        assert queue["queue_is_lower_bound"] is True
        assert queue["n_unseeded_adds"] == 6
        assert queue["n_adds"] == 6

    def test_the_unseen_cancel_is_counted_not_subtracted(self) -> None:
        queue = queue_positions(_opened_mid_session())
        assert queue["n_unseen_decrements"] == 1
        assert queue["unseen_size"] == 400.0
        # 0, 100, 200, 300, 400, 500: the last arrival still sees the five
        # orders the window watched join. Taking the unseen 400 out of them
        # left it 100 ahead, which was nobody's queue.
        assert queue["mean_queue_ahead"] == pytest.approx(1500 / 6)

    def test_the_report_says_so(self) -> None:
        report = order_event_metrics(_opened_mid_session())
        warning = [w for w in report["warnings"] if "LOWER BOUNDS" in w]
        assert len(warning) == 1
        assert "6 of 6" in warning[0]
        assert "400 shares" in warning[0]

    def test_a_partial_fill_takes_only_what_it_filled(self) -> None:
        events = _events(
            [
                (0, 0, "R", "N", np.nan, np.nan),
                (1, 1, "A", "B", 100.0, 300),
                (2, 1, "F", "B", 100.0, 100),
                (3, 2, "A", "B", 100.0, 50),
                (4, 1, "F", "B", 100.0, 500),  # more than the 200 it had left
                (5, 3, "A", "B", 100.0, 50),
            ]
        )
        queue = queue_positions(events)
        # Order 1 joins an empty level; order 2 sees the 200 left of order
        # 1; order 3 sees only order 2, because order 1 took out its own 200
        # and no more.
        assert queue["mean_queue_ahead"] == pytest.approx((0 + 200 + 50) / 3)
        assert queue["n_unseen_decrements"] == 0

    def test_a_window_opened_by_its_snapshot_is_not_a_lower_bound(self) -> None:
        """Null case: the snapshot seeds the levels, so the arrival's 800
        ahead is the queue, as it always was."""
        events = _events(
            [
                (0, 1, "A", "B", 100.0, 500),
                (0, 2, "A", "B", 100.0, 300),
                (1, 3, "A", "B", 100.0, 100),
            ]
        )
        events["snapshot"] = [True, True, False]
        queue = queue_positions(events)
        assert queue.get("queue_is_lower_bound", False) is False
        assert queue["n_snapshot_orders"] == 2
        assert queue["mean_queue_ahead"] == 800.0
        assert queue.get("n_unseen_decrements", 0) == 0

    def test_a_window_opened_by_a_clear_is_exact(self) -> None:
        """Null case: a CLEAR empties the book, so the window knows it
        started from nothing and its queue figures are the queue."""
        events = _events(
            [
                (0, 0, "R", "N", np.nan, np.nan),
                (1, 1, "A", "B", 100.0, 100),
                (2, 2, "A", "B", 100.0, 200),
            ]
        )
        queue = queue_positions(events)
        assert queue.get("queue_is_lower_bound", False) is False
        assert queue.get("n_unseeded_adds", 0) == 0
        assert queue["mean_queue_ahead"] == pytest.approx(50.0)
        assert not any(
            "LOWER BOUNDS" in w for w in order_event_metrics(events)["warnings"]
        )


class TestOrderLifetime:
    def test_a_filled_order_is_measured_from_its_add(self) -> None:
        events = _events([(0, 1, "A", "B", 100.0, 100), (5, 1, "F", "B", 100.0, 100)])
        summary = order_lifetimes(events)
        assert summary["filled"]["n"] == 1
        assert summary["filled"]["mean_seconds"] == pytest.approx(5.0)
        assert summary["cancelled"]["n"] == 0

    def test_fills_and_cancels_are_reported_apart(self) -> None:
        """They mean opposite things about who wanted to trade, so an
        average over both would answer neither question."""
        events = _events(
            [
                (0, 1, "A", "B", 100.0, 100),
                (2, 1, "C", "B", 100.0, 100),
                (0, 2, "A", "B", 100.0, 100),
                (10, 2, "F", "B", 100.0, 100),
            ]
        )
        summary = order_lifetimes(events)
        assert summary["cancelled"]["mean_seconds"] == pytest.approx(2.0)
        assert summary["filled"]["mean_seconds"] == pytest.approx(10.0)

    def test_an_order_resting_before_the_window_is_counted_not_measured(
        self,
    ) -> None:
        """
        THE BIAS THIS AVOIDS. Order 9 has no add here, so its true lifetime
        is longer than anything the window can see. Folding it in as the
        time since the window opened would drag every average down, worst
        for exactly the long-resting orders a queue study is about.
        """
        events = _events(
            [
                (1, 9, "C", "B", 100.0, 100),
                (0, 1, "A", "B", 100.0, 100),
                (8, 1, "C", "B", 100.0, 100),
            ]
        )
        summary = order_lifetimes(events)
        assert summary["terminated_without_an_add"] == 1
        assert summary["cancelled"]["n"] == 1
        assert summary["cancelled"]["mean_seconds"] == pytest.approx(8.0)

    def test_an_order_still_open_is_right_censored(self) -> None:
        events = _events([(0, 1, "A", "B", 100.0, 100)])
        assert order_lifetimes(events)["still_resting"] == 1


class TestRatesAndRatios:
    def test_cancel_to_add_is_exact(self) -> None:
        """A book snapshot cannot compute this at all: it sees size vanish
        and cannot say whether it was cancelled or filled."""
        events = _events(
            [
                (0, 1, "A", "B", 100.0, 10),
                (1, 1, "C", "B", 100.0, 10),
                (2, 2, "A", "B", 100.0, 10),
                (3, 2, "C", "B", 100.0, 10),
                (4, 3, "A", "B", 100.0, 10),
                (5, 3, "C", "B", 100.0, 10),
                (6, 4, "A", "B", 100.0, 10),
            ]
        )
        rates = event_rates(events)
        assert rates["counts_by_action"]["A"] == 4
        assert rates["counts_by_action"]["C"] == 3
        assert rates["cancel_to_add"] == pytest.approx(3 / 4)

    def test_intensity_uses_the_real_clock(self) -> None:
        events = _events([(i, i, "A", "B", 100.0, 10) for i in range(11)])
        rates = event_rates(events)
        assert rates["elapsed_seconds"] == pytest.approx(10.0)
        assert rates["events_per_second"] == pytest.approx(1.1)

    def test_a_zero_length_window_reports_null_not_zero(self) -> None:
        """Zero would read as a quiet market. Null says no rate is defined."""
        events = _events([(0, 1, "A", "B", 100.0, 10), (0, 2, "A", "B", 100.0, 10)])
        rates = event_rates(events)
        assert rates["elapsed_seconds"] is None
        assert rates["events_per_second"] is None

    def test_an_absent_denominator_reports_null(self) -> None:
        events = _events([(0, 1, "C", "B", 100.0, 10)])
        rates = event_rates(events)
        assert rates["cancel_to_add"] is None
        assert rates["cancel_to_trade"] is None


class TestTheReport:
    def test_it_refuses_a_frame_that_is_not_an_order_feed(self) -> None:
        book = pd.DataFrame(
            {"timestamp": [T0], "bid_price_0": [100.0], "bid_size_0": [10.0]}
        )
        with pytest.raises(ValidationError, match="order_id"):
            order_event_metrics(book)

    def test_it_refuses_an_empty_frame(self) -> None:
        with pytest.raises(ValidationError, match="non-empty"):
            order_event_metrics(pd.DataFrame(columns=list(ORDER_EVENT_COLUMNS)))

    def test_a_clear_is_reported_as_a_discontinuity(self) -> None:
        events = _events(
            [(0, 1, "A", "B", 100.0, 10), (1, 0, "R", "N", np.nan, np.nan)]
        )
        report = order_event_metrics(events)
        assert any("CLEAR" in w for w in report["warnings"])

    def test_modify_is_counted_but_not_requeued(self) -> None:
        """A modify that raises size or changes price loses priority at the
        venue, and the rule differs by venue -- so it is reported rather
        than guessed."""
        events = _events([(0, 1, "A", "B", 100.0, 10), (1, 1, "M", "B", 100.0, 20)])
        report = order_event_metrics(events)
        assert any("MODIFY" in w for w in report["warnings"])

    def test_an_unknown_action_is_named(self) -> None:
        events = _events([(0, 1, "A", "B", 100.0, 10), (1, 2, "Z", "B", 100.0, 10)])
        report = order_event_metrics(events)
        assert any("'Z'" in w or "Z" in w for w in report["warnings"])

    def test_a_realistic_session_produces_every_measure(self) -> None:
        rng = np.random.default_rng(4)
        rows = []
        oid = 0
        for step in range(400):
            oid += 1
            rows.append((step * 0.1, oid, "A", "B", 100.0, float(rng.integers(1, 50))))
            if step % 3 == 0:
                rows.append((step * 0.1 + 0.05, oid, "C", "B", 100.0, 10.0))
            elif step % 7 == 0:
                rows.append((step * 0.1 + 0.05, oid, "F", "B", 100.0, 10.0))
        report = order_event_metrics(_events(rows))
        assert report["queue"]["n_adds"] == 400
        assert report["queue"]["mean_queue_ahead"] > 0
        assert report["rates"]["cancel_to_add"] > 0
        assert report["lifetimes"]["cancelled"]["n"] > 0
        assert report["lifetimes"]["still_resting"] > 0


@pytest.mark.skipif(
    not order_events_module.HAS_CPP, reason="native extension not built"
)
class TestBothPathsGiveTheSameReport:
    """
    The queue and lifetime passes are the extension's kernels when it is
    built, the loops above otherwise (see the CHANGELOG entry of
    2026-10-04): 8.0 s of a 2,000,000-event session's report in the loops.
    The report is the same on both paths, every float to the bit, on the
    realistic session above and on one opened by a snapshot and cut by a
    CLEAR, with cancels of orders the window never saw.
    """

    @staticmethod
    def _both(events):
        native = order_event_metrics(events)
        order_events_module.HAS_CPP = False
        try:
            python = order_event_metrics(events)
        finally:
            order_events_module.HAS_CPP = True
        return native, python

    @staticmethod
    def _session(seed):
        rng = np.random.default_rng(seed)
        rows = [(0.0, 900 + i, "A", "B", 99.0 + i % 3, 100.0) for i in range(5)]
        oid = 0
        for step in range(600):
            oid += 1
            t = 1.0 + step * 0.01
            side = "B" if step % 2 else "A"
            price = 100.0 + float(rng.integers(-3, 4))
            rows.append((t, oid, "A", side, price, float(rng.integers(1, 9) * 100)))
            if step % 3 == 0:
                rows.append((t + 0.004, oid, "C", side, price, 100.0))
            elif step % 5 == 0:
                rows.append((t + 0.003, oid, "F", side, price, 50.0))
            if step % 50 == 7:
                rows.append((t + 0.002, 900 + step % 5, "C", "B", 99.0, 100.0))
            if step == 300:
                rows.append((t + 0.001, 0, "R", "N", float("nan"), float("nan")))
        frame = _events(rows)
        flags = np.zeros(len(frame), dtype=np.int64)
        flags[:5] = 32
        frame["flags"] = flags
        return frame

    @pytest.mark.parametrize("seed", [0, 1])
    def test_the_same_report(self, seed):
        native, python = self._both(self._session(seed))
        assert native == python
        for kind in ("filled", "cancelled"):
            for key, value in native["lifetimes"][kind].items():
                want = python["lifetimes"][kind][key]
                assert np.float64(value).tobytes() == np.float64(want).tobytes(), key

    def test_the_realistic_session(self):
        rng = np.random.default_rng(4)
        rows = []
        oid = 0
        for step in range(400):
            oid += 1
            rows.append((step * 0.1, oid, "A", "B", 100.0, float(rng.integers(1, 50))))
            if step % 3 == 0:
                rows.append((step * 0.1 + 0.05, oid, "C", "B", 100.0, 10.0))
        native, python = self._both(_events(rows))
        assert native == python

    def test_a_small_frame_on_its_own_runs_the_loop(self):
        """Below 64 events a standalone pass codes nothing: the three
        factorize calls cost more than the loop (0.82-0.92x at 25-50)."""
        events = self._session(0).iloc[:40]
        assert not order_events_module._sized_for_native(events)
        assert queue_positions(events) == order_events_module._queue_positions(
            events, None
        )
