"""
A book's touch, when it is not a touch.

WHAT THESE ARE FOR. Two summaries read the same book and handled a broken
snapshot differently, and neither said so:

    one infinite ask in a hundred      book_metrics returned null for the
                                       mean spread, mid and microprice, and
                                       n_crossed said the book was clean
    sixty missing asks in a hundred    averaged over the other forty with
                                       no count and no warning
    an all-crossed book                depth_profile reported the bid ABOVE
                                       the mid -- a negative distance -- with
                                       no count and no warning

The rule now shared by both: a snapshot whose level 0 is not four finite
numbers has no touch and is excluded from every statistic; a crossed one is
excluded from every statistic measured from the mid. Both are counted.

THE BOOKS ARE BUILT SO THE ANSWER IS ARITHMETIC. A steady 99.99 / 100.01
touch -- a mid of exactly 100.00, a 2-cent spread, 1 bp from the mid on
each side -- with 500 resting at the touch and 800 behind it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.analysis.order_book import book_metrics, depth_profile


def _steady_book(n: int = 100) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "bid_price_0": np.full(n, 99.99),
            "bid_size_0": np.full(n, 500.0),
            "ask_price_0": np.full(n, 100.01),
            "ask_size_0": np.full(n, 500.0),
            "bid_price_1": np.full(n, 99.98),
            "bid_size_1": np.full(n, 800.0),
            "ask_price_1": np.full(n, 100.02),
            "ask_size_1": np.full(n, 800.0),
        }
    )


class TestATouchThatIsNotANumber:
    def test_one_infinite_ask_no_longer_blanks_every_mean(self):
        book = _steady_book()
        book.loc[0, "ask_price_0"] = np.inf
        result = book_metrics(book)
        assert result["n_nonfinite_touch"] == 1
        assert result["n_crossed"] == 0
        # The 99 clean snapshots, exactly.
        assert result["mean_spread"] == pytest.approx(0.02)
        assert result["mean_mid"] == pytest.approx(100.0)
        assert result["mean_microprice"] == pytest.approx(100.0)
        assert result["mean_spread_bps"] == pytest.approx(2.0, rel=1e-6)
        assert any("NON-FINITE" in w for w in result["warnings"])

    def test_missing_asks_are_counted_and_said(self):
        book = _steady_book()
        book.loc[:59, "ask_price_0"] = np.nan
        result = book_metrics(book)
        assert result["n_nonfinite_touch"] == 60
        assert any("60 of 100" in w and "NON-FINITE" in w for w in result["warnings"])
        assert result["mean_spread_bps"] == pytest.approx(2.0, rel=1e-6)

    def test_an_infinite_size_is_not_a_size(self):
        """Sizes are touch too: an infinite one made the mean touch size
        infinite, which came back null."""
        book = _steady_book()
        book.loc[3, "bid_size_0"] = np.inf
        result = book_metrics(book)
        assert result["n_nonfinite_touch"] == 1
        assert result["mean_touch_size"] == pytest.approx(1000.0)
        assert result["mean_touch_imbalance"] == pytest.approx(0.0)

    def test_a_clean_book_is_unchanged(self):
        """Null case: nothing to exclude, nothing counted, no new warning."""
        result = book_metrics(_steady_book())
        assert result.get("n_nonfinite_touch", 0) == 0
        assert result["n_crossed"] == 0
        assert result["mean_spread"] == pytest.approx(0.02)
        assert result["mean_touch_size"] == pytest.approx(1000.0)
        assert not any("NON-FINITE" in w for w in result["warnings"])
        assert not any("crossed" in w for w in result["warnings"])


class TestACrossedBookHasNoDistanceFromItsMid:
    @staticmethod
    def _crossed(book: pd.DataFrame, rows) -> pd.DataFrame:
        book = book.copy()
        book.loc[rows, "ask_price_0"] = 99.97
        book.loc[rows, "ask_price_1"] = 99.96
        return book

    def test_an_all_crossed_book_reports_no_distance(self):
        book = self._crossed(_steady_book(), slice(None))
        result = depth_profile(book)
        assert result["n_crossed"] == 100
        for level in result["profile"]:
            assert level["mean_bid_distance_bps"] is None
            assert level["mean_ask_distance_bps"] is None
        assert any("crossed" in w for w in result["warnings"])

    def test_a_half_crossed_book_is_measured_on_its_clean_half(self):
        book = self._crossed(_steady_book(), slice(0, 49))
        result = depth_profile(book)
        assert result["n_crossed"] == 50
        touch = result["profile"][0]
        assert touch["mean_bid_distance_bps"] == pytest.approx(1.0, rel=1e-6)
        assert touch["mean_ask_distance_bps"] == pytest.approx(1.0, rel=1e-6)
        # Sizes are still sizes on a crossed snapshot.
        assert touch["mean_bid_size"] == pytest.approx(500.0)

    def test_a_non_finite_touch_is_counted_here_too(self):
        book = _steady_book()
        book.loc[0, "bid_price_0"] = np.nan
        result = depth_profile(book)
        assert result["n_nonfinite_touch"] == 1
        assert result["profile"][0]["mean_bid_distance_bps"] == pytest.approx(
            1.0, rel=1e-6
        )

    def test_an_empty_book_is_refused(self):
        from standard_quant_tools.error import ValidationError

        with pytest.raises(ValidationError, match="non-empty"):
            depth_profile(pd.DataFrame())

    def test_a_clean_book_is_unchanged(self):
        """Null case: 1 bp and 2 bp from the mid, every snapshot measured."""
        result = depth_profile(_steady_book())
        assert result.get("n_crossed", 0) == 0
        assert result.get("n_nonfinite_touch", 0) == 0
        distances = [r["mean_bid_distance_bps"] for r in result["profile"]]
        assert distances == pytest.approx([1.0, 2.0], rel=1e-6)
        assert not any("crossed" in w for w in result["warnings"])

    def test_the_book_metrics_depth_slope_skips_crossed_snapshots(self):
        """The slope is measured from the mid as well. A crossed snapshot's
        deeper levels sat at positive distances from a mid that was not one
        and were fitted as depth."""
        clean = book_metrics(_steady_book(50))["depth_slope"]
        # A stale bid above a stale ask around a mid of 100.00: the touch is
        # at negative distances, but level 1 sits 2 bp out as usual.
        crossed = _steady_book(50)
        crossed["bid_price_0"] = 100.01
        crossed["ask_price_0"] = 99.99
        mixed = pd.concat([_steady_book(50), crossed], ignore_index=True)
        result = book_metrics(mixed)
        assert result["n_crossed"] == 50
        assert result["depth_slope"] == pytest.approx(clean)


class TestTheToolReportsEachExclusionOnce:
    def test_a_crossed_book_is_one_warning_with_the_profile_on(self):
        """Both summaries run on one book under one rule, so the tool says
        each exclusion once rather than once per summary."""
        from standard_quant_tools.agent.tools import dispatch

        book = TestACrossedBookHasNoDistanceFromItsMid._crossed(
            _steady_book(20), slice(0, 4)
        )
        snapshots = book.to_dict("records")
        snapshots[10]["ask_size_0"] = None  # a vendor gap: no size at the ask
        result = dispatch(
            "get_order_book_metrics",
            {"snapshots": snapshots, "include_profile": True},
        )
        assert result["n_crossed"] == 5
        assert result["n_nonfinite_touch"] == 1
        assert sum("crossed or locked" in w for w in result["warnings"]) == 1
        assert sum("NON-FINITE" in w for w in result["warnings"]) == 1
        assert result["profile"][0]["mean_bid_distance_bps"] == pytest.approx(
            1.0, rel=1e-6
        )
