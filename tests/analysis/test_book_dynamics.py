"""
Order-flow imbalance from the book, against hand-computable transitions.

WHY THESE CASES. The Cont-Kukanov-Stoikov definition reduces to a size
DIFFERENCE only when the price is unchanged, and a naive implementation
gets exactly that case right and the other two wrong. Each test below fixes
one transition where the answer can be worked out on paper:

    bid price rises   the whole NEW size is demand          +q_b(n)
    bid price falls   the whole OLD size left               -q_b(n-1)
    bid price equal   the size difference                   q_b(n) - q_b(n-1)

A wrong implementation passes the third and fails the first two, and the
first two are the interesting book states.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.analysis.order_book import book_dynamics
from standard_quant_tools.error import ValidationError

T0 = "2026-03-02T14:30:00Z"
T1 = "2026-03-02T14:30:01Z"


def _pair(bid, ask, *, stamps=(T0, T1)) -> pd.DataFrame:
    """bid/ask are ((price, size), (price, size)) for the two snapshots."""
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(list(stamps)),
            "bid_price_0": [bid[0][0], bid[1][0]],
            "bid_size_0": [bid[0][1], bid[1][1]],
            "ask_price_0": [ask[0][0], ask[1][0]],
            "ask_size_0": [ask[0][1], ask[1][1]],
        }
    )


STEADY_ASK = ((100.02, 400.0), (100.02, 400.0))


class TestTheThreeBidTransitions:
    def test_a_rising_bid_contributes_its_whole_new_size(self) -> None:
        book = _pair(((100.00, 300.0), (100.01, 500.0)), STEADY_ASK)
        assert book_dynamics(book)["ofi"] == pytest.approx(500.0)

    def test_a_falling_bid_removes_the_whole_old_size(self) -> None:
        """The old bid was pulled or hit. What left is what was there, not
        the difference between the two."""
        book = _pair(((100.00, 300.0), (99.99, 700.0)), STEADY_ASK)
        assert book_dynamics(book)["ofi"] == pytest.approx(-300.0)

    def test_an_unchanged_bid_reduces_to_the_size_difference(self) -> None:
        book = _pair(((100.00, 300.0), (100.00, 450.0)), STEADY_ASK)
        assert book_dynamics(book)["ofi"] == pytest.approx(150.0)


class TestTheAskSideIsMirrored:
    def test_a_falling_ask_is_negative_pressure(self) -> None:
        """A more aggressive offer pushes the other way, so its whole new
        size enters with a minus."""
        book = _pair(
            ((100.00, 300.0), (100.00, 300.0)),
            ((100.02, 400.0), (100.01, 600.0)),
        )
        assert book_dynamics(book)["ofi"] == pytest.approx(-600.0)

    def test_a_rising_ask_removes_the_whole_old_size(self) -> None:
        book = _pair(
            ((100.00, 300.0), (100.00, 300.0)),
            ((100.02, 400.0), (100.03, 900.0)),
        )
        assert book_dynamics(book)["ofi"] == pytest.approx(400.0)

    def test_a_still_book_has_zero_flow(self) -> None:
        book = _pair(((100.00, 300.0), (100.00, 300.0)), STEADY_ASK)
        assert book_dynamics(book)["ofi"] == pytest.approx(0.0)


class TestWhatIsRefusedRatherThanInterpolated:
    def test_one_snapshot_cannot_be_compared_with_anything(self) -> None:
        book = _pair(((100.0, 1.0), (100.0, 1.0)), STEADY_ASK).head(1)
        with pytest.raises(ValidationError, match="at least two snapshots"):
            book_dynamics(book)

    def test_a_crossed_snapshot_drops_its_pairs_and_says_so(self) -> None:
        """Comparing across a crossed book describes a transition nobody
        observed."""
        # Four snapshots, the second crossed. It poisons the pairs on
        # either side of it and leaves the last pair usable, so the drop is
        # observable rather than fatal.
        book = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2026-03-02 14:30", periods=4, freq="1s", tz="UTC"
                ),
                "bid_price_0": [100.00, 100.05, 100.00, 100.01],
                "bid_size_0": [300.0, 300.0, 300.0, 500.0],
                "ask_price_0": [100.02, 100.02, 100.02, 100.02],
                "ask_size_0": [400.0, 400.0, 400.0, 400.0],
            }
        )
        report = book_dynamics(book)
        assert report["n_pairs"] == 1
        assert report["n_pairs_dropped"] == 2
        assert any("dropped" in w for w in report["warnings"])
        # The one surviving pair is the rising bid: its whole new size.
        assert report["ofi"] == pytest.approx(500.0)

    def test_a_book_with_no_usable_pair_refuses(self) -> None:
        book = _pair(((100.05, 300.0), (100.06, 300.0)), STEADY_ASK)
        with pytest.raises(ValidationError, match="no usable consecutive pair"):
            book_dynamics(book)

    def test_a_missing_touch_column_is_named(self) -> None:
        book = _pair(((100.0, 1.0), (100.0, 1.0)), STEADY_ASK).drop(
            columns=["ask_size_0"]
        )
        with pytest.raises(ValidationError, match="ask_size_0"):
            book_dynamics(book)


class TestRates:
    def test_rates_use_the_real_clock(self) -> None:
        stamps = pd.date_range("2026-03-02 14:30", periods=11, freq="1s", tz="UTC")
        book = pd.DataFrame(
            {
                "timestamp": stamps,
                "bid_price_0": np.full(11, 100.00),
                "bid_size_0": np.full(11, 300.0),
                "ask_price_0": np.full(11, 100.02),
                "ask_size_0": np.full(11, 400.0),
            }
        )
        report = book_dynamics(book)
        assert report["updates_per_second"] == pytest.approx(1.1)
        assert report["mid_changes"] == 0

    def test_a_timestampless_book_reports_null_rates(self) -> None:
        """Zero would read as a still book."""
        book = _pair(((100.0, 300.0), (100.0, 300.0)), STEADY_ASK).drop(
            columns=["timestamp"]
        )
        report = book_dynamics(book)
        assert report["ofi_per_second"] is None
        assert report["updates_per_second"] is None
        assert any("null rather than zero" in w for w in report["warnings"])


class TestItIsNotTheBarProxy:
    def test_the_tool_exposes_it_behind_a_flag(self) -> None:
        """
        `get_order_flow_imbalance` computes signed return times volume from
        BARS. This is a different quantity from the same book, so it lands
        on the tool that already reads a book rather than as a second tool
        with a confusable name.
        """
        from standard_quant_tools.agent.runtimes.microstructure import (
            get_order_book_metrics,
        )
        from standard_quant_tools.agent.runtimes.microstructure.book_tools import (
            OrderBookInput,
        )

        book = _pair(((100.00, 300.0), (100.01, 500.0)), STEADY_ASK)
        plain = get_order_book_metrics(
            OrderBookInput(
                snapshots=book.drop(columns=["timestamp"]).to_dict("records")
            )
        )
        assert plain.ofi is None, "dynamics must be opt-in"

        with_flow = get_order_book_metrics(
            OrderBookInput(
                snapshots=book.drop(columns=["timestamp"]).to_dict("records"),
                include_dynamics=True,
            )
        )
        assert with_flow.ofi == pytest.approx(500.0)
        assert with_flow.mid_changes == 1
