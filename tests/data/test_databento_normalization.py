"""
Databento's wire shape, and the four ways reading it naively goes wrong.

EVERY TEST HERE IS OFFLINE. The module under test is frame-in, frame-out on
purpose: a trap that can only be exercised against a live API, a key and an
entitlement is a trap that gets tested once, by hand, and then never again.

The four:

    fixed-point int64 prices   read as dollars, everything is 1e9x too big
    int64-max empty levels     read as prices, a $9.2 BILLION quote
    a trailing empty level     kept, and depth_slope regresses over nothing
    ts_event vs ts_recv        chosen silently, and a latency study is wrong
                               in a way its author cannot see

None of them raises. Each produces finite, ordered, plausible-looking
numbers, which is what makes them worth a test file.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.data import databento as db
from standard_quant_tools.data.external import book_levels
from standard_quant_tools.error import ValidationError

ROWS = 1_000
LEVELS = 10
LIVE_LEVELS = 6


def _raw_mbp10(
    *, rows: int = ROWS, levels: int = LEVELS, live: int = LIVE_LEVELS, seed: int = 5
) -> pd.DataFrame:
    """A frame shaped exactly like `.to_df()` on an mbp-10 store."""
    rng = np.random.default_rng(seed)
    mid = 100.0 + np.cumsum(rng.normal(0, 0.01, rows))
    base = pd.Timestamp("2026-03-02 14:30", tz="UTC").value
    data = {
        "ts_recv": (base + np.arange(rows) * 1_000_000).astype("int64"),
        "ts_event": (base + np.arange(rows) * 1_000_000 - 250_000).astype("int64"),
        "action": np.where(rng.random(rows) < 0.3, "C", "A"),
        "side": np.where(rng.random(rows) < 0.5, "B", "A"),
        "flags": np.zeros(rows, dtype="int64"),
        "sequence": np.arange(rows, dtype="int64"),
    }
    for level in range(levels):
        offset = 0.01 * level
        bid = np.round((mid - 0.01 - offset) * db.FIXED_PRICE_SCALE).astype("int64")
        ask = np.round((mid + 0.01 + offset) * db.FIXED_PRICE_SCALE).astype("int64")
        bid_size = rng.integers(100, 3000, rows).astype("int64")
        ask_size = rng.integers(100, 3000, rows).astype("int64")
        if level >= live:
            bid[:] = db.UNDEF_PRICE
            ask[:] = db.UNDEF_PRICE
            bid_size[:] = db.UNDEF_ORDER_SIZE
            ask_size[:] = db.UNDEF_ORDER_SIZE
        data[f"bid_px_{level:02d}"] = bid
        data[f"ask_px_{level:02d}"] = ask
        data[f"bid_sz_{level:02d}"] = bid_size
        data[f"ask_sz_{level:02d}"] = ask_size
        data[f"bid_ct_{level:02d}"] = rng.integers(1, 40, rows).astype("int64")
        data[f"ask_ct_{level:02d}"] = rng.integers(1, 40, rows).astype("int64")
    return pd.DataFrame(data)


@pytest.fixture
def raw() -> pd.DataFrame:
    return _raw_mbp10()


class TestTheConstantsMatchTheSdk:
    """
    Pinned against the installed SDK rather than trusted from a docstring.

    These four numbers are load-bearing: a wrong FIXED_PRICE_SCALE moves
    every price by three orders of magnitude, and a wrong UNDEF_PRICE means
    the sentinels are never masked at all.
    """

    def test_constants_agree_with_databento_dbn(self) -> None:
        dbn = pytest.importorskip("databento_dbn")
        assert db.FIXED_PRICE_SCALE == dbn.FIXED_PRICE_SCALE
        assert db.UNDEF_PRICE == dbn.UNDEF_PRICE
        assert db.UNDEF_ORDER_SIZE == dbn.UNDEF_ORDER_SIZE
        assert db.UNDEF_TIMESTAMP == dbn.UNDEF_TIMESTAMP

    def test_undef_price_is_int64_max(self) -> None:
        assert db.UNDEF_PRICE == 2**63 - 1

    def test_an_unmasked_sentinel_would_be_a_nine_figure_quote(self) -> None:
        """The number that makes masking non-optional."""
        assert db.UNDEF_PRICE / db.FIXED_PRICE_SCALE == pytest.approx(9.223372e9)

    def test_flag_bits_agree_with_the_sdk(self) -> None:
        dbn = pytest.importorskip("databento_dbn")
        assert db.F_MAYBE_BAD_BOOK == dbn.F_MAYBE_BAD_BOOK
        assert db.F_BAD_TS_RECV == dbn.F_BAD_TS_RECV
        assert db.F_SNAPSHOT == dbn.F_SNAPSHOT
        assert db.F_TOB == dbn.F_TOB


class TestDetection:
    def test_a_databento_frame_is_recognized(self, raw) -> None:
        assert db.looks_like_databento(raw.columns)

    def test_this_librarys_own_spelling_is_not(self) -> None:
        assert not db.looks_like_databento(
            ["timestamp", "bid_price_0", "bid_size_0", "ask_price_0", "ask_size_0"]
        )

    def test_depth_counts_two_digit_levels(self, raw) -> None:
        assert db.book_depth(raw.columns) == LEVELS


class TestFixedPointPrices:
    def test_integer_prices_are_scaled(self, raw) -> None:
        out, _ = db.normalize_book(raw)
        assert 90.0 < out["bid_price_0"].median() < 110.0

    def test_the_choice_is_reported(self, raw) -> None:
        _, notes = db.normalize_book(raw)
        assert any("fixed-point" in n for n in notes)

    def test_float_dollars_are_not_scaled_again(self, raw) -> None:
        """`.to_df()` already returns dollars. Dividing twice would put every
        price at 1e-7 -- still finite, still ordered, entirely wrong."""
        floats = raw.copy()
        for level in range(LEVELS):
            for side in ("bid", "ask"):
                column = f"{side}_px_{level:02d}"
                values = floats[column].astype("float64")
                floats[column] = (
                    values.mask(values == float(db.UNDEF_PRICE)) / db.FIXED_PRICE_SCALE
                )
        out, notes = db.normalize_book(floats)
        assert 90.0 < out["bid_price_0"].median() < 110.0
        assert any("already in whole units" in n for n in notes)

    def test_integer_dollars_are_not_divided_by_a_billion(self, raw) -> None:
        """
        The failure the other direction. An extract rounded to whole dollars
        is still an integer column, and the dtype test alone would scale it
        into oblivion -- so magnitude is checked as well.
        """
        whole = raw.copy()
        for level in range(LEVELS):
            for side in ("bid", "ask"):
                column = f"{side}_px_{level:02d}"
                values = whole[column].astype("float64")
                whole[column] = (
                    (
                        values.mask(values == float(db.UNDEF_PRICE))
                        / db.FIXED_PRICE_SCALE
                    )
                    .round(0)
                    .fillna(0)
                    .astype("int64")
                )
        out, notes = db.normalize_book(whole)
        assert out["bid_price_0"].median() < 1000
        assert any("plausible price already" in n for n in notes)

    def test_an_explicit_scale_overrides_the_guess(self, raw) -> None:
        out, notes = db.normalize_book(raw, price_scale="float")
        assert out["bid_price_0"].median() > 1e9
        assert any("as requested" in n for n in notes)

    def test_an_unknown_scale_is_refused(self, raw) -> None:
        with pytest.raises(ValidationError, match="price_scale"):
            db.normalize_book(raw, price_scale="nanodollars")


class TestSentinels:
    def test_empty_levels_become_null_not_prices(self) -> None:
        kept, _ = db.normalize_book(_raw_mbp10(), keep_empty_levels=True)
        for level in range(LIVE_LEVELS, LEVELS):
            assert kept[f"bid_price_{level}"].isna().all()
            assert kept[f"ask_size_{level}"].isna().all()

    def test_no_price_survives_as_a_nine_figure_quote(self) -> None:
        kept, _ = db.normalize_book(_raw_mbp10(), keep_empty_levels=True)
        prices = [c for c in kept.columns if "price" in c]
        assert kept[prices].max().max() < 1_000

    def test_the_masking_is_counted(self, raw) -> None:
        _, notes = db.normalize_book(raw, keep_empty_levels=True)
        empty_levels = LEVELS - LIVE_LEVELS
        expected = ROWS * empty_levels * 2 * 2  # two sides, price and size
        assert any(f"{expected} sentinel" in n for n in notes)


class TestEmptyLevelsAreNotLevels:
    def test_trailing_empty_levels_are_dropped(self, raw) -> None:
        """
        Left in, the dataset DECLARES ten levels while four hold nothing,
        and depth_slope regresses size against distance over the four.
        """
        out, notes = db.normalize_book(raw)
        assert book_levels(out.columns) == LIVE_LEVELS
        assert f"bid_price_{LIVE_LEVELS}" not in out.columns
        assert any("levels 6-9 are empty" in n for n in notes)

    def test_keeping_them_is_available_for_aligning_extracts(self, raw) -> None:
        out, _ = db.normalize_book(raw, keep_empty_levels=True)
        assert book_levels(out.columns) == LEVELS

    def test_a_gap_below_a_live_level_is_reported_not_renumbered(self) -> None:
        """
        A hole in the middle is a malformed book, not a thin one. Silently
        closing it would renumber level 5 into level 4 and hide the problem.
        """
        frame = _raw_mbp10(live=LEVELS)
        for side in ("bid", "ask"):
            frame[f"{side}_px_03"] = db.UNDEF_PRICE
            frame[f"{side}_sz_03"] = db.UNDEF_ORDER_SIZE
        out, notes = db.normalize_book(frame)
        assert f"bid_price_3" in out.columns
        assert any("malformed book" in n for n in notes)

    def test_a_fully_live_book_drops_nothing(self) -> None:
        out, notes = db.normalize_book(_raw_mbp10(live=LEVELS))
        assert book_levels(out.columns) == LEVELS
        assert not any("are empty in every" in n for n in notes)


class TestTheTimestampChoiceIsSaidOutLoud:
    def test_ts_recv_is_the_default_and_is_named(self, raw) -> None:
        out, notes = db.normalize_book(raw)
        assert any("ts_recv" in n for n in notes)
        expected = pd.to_datetime(raw["ts_recv"], unit="ns", utc=True)
        assert out["timestamp"].iloc[0] == expected.iloc[0]

    def test_ts_event_can_be_asked_for(self, raw) -> None:
        out, notes = db.normalize_book(raw, timestamp="ts_event")
        expected = pd.to_datetime(raw["ts_event"], unit="ns", utc=True)
        assert out["timestamp"].iloc[0] == expected.iloc[0]
        assert any("ts_event" in n for n in notes)

    def test_the_two_differ_by_the_network(self, raw) -> None:
        """250 microseconds here, and it is exactly what a latency study
        measures -- so which column was used cannot be left implicit."""
        recv, _ = db.normalize_book(raw)
        event, _ = db.normalize_book(raw, timestamp="ts_event")
        gap = (recv["timestamp"] - event["timestamp"]).dt.total_seconds()
        assert gap.iloc[0] == pytest.approx(0.00025)

    def test_asking_for_an_absent_column_is_refused(self, raw) -> None:
        with pytest.raises(ValidationError, match="ts_event"):
            db.normalize_book(raw.drop(columns=["ts_event"]), timestamp="ts_event")

    def test_an_unknown_source_is_refused(self, raw) -> None:
        with pytest.raises(ValidationError, match="timestamp="):
            db.normalize_book(raw, timestamp="whenever")


class TestWhatIsDeliberatelyKept:
    def test_order_counts_survive(self, raw) -> None:
        """The only queue-shaped quantity mbp-10 carries. A queue-position
        proxy cannot be built later from a frame this was dropped from."""
        out, _ = db.normalize_book(raw)
        assert "bid_count_0" in out.columns
        assert out["bid_count_0"].notna().all()

    def test_action_and_side_survive(self, raw) -> None:
        """What makes cancellation rate computable at all."""
        out, _ = db.normalize_book(raw)
        assert set(out["action"].unique()) <= {"A", "C"}
        assert "side" in out.columns

    def test_the_output_satisfies_this_librarys_contract(self, raw) -> None:
        out, _ = db.normalize_book(raw)
        for column in (
            "timestamp",
            "bid_price_0",
            "bid_size_0",
            "ask_price_0",
            "ask_size_0",
        ):
            assert column in out.columns


class TestTheVendorsOwnQualityFlags:
    """
    Read here, and read nowhere else in either codebase.

    `F_MAYBE_BAD_BOOK` is the venue reporting that it could not keep the
    book consistent. No downstream arithmetic recovers from that, and no
    check invented here would find it.
    """

    def test_a_bad_book_flag_warns(self, raw) -> None:
        flagged = raw.copy()
        flagged.loc[:49, "flags"] = db.F_MAYBE_BAD_BOOK
        _, notes = db.normalize_book(flagged)
        assert any("may be inconsistent" in n for n in notes)
        assert any("50 of 1000" in n for n in notes)

    def test_snapshots_warn_about_overstated_intensity(self, raw) -> None:
        flagged = raw.copy()
        flagged.loc[:9, "flags"] = db.F_SNAPSHOT
        _, notes = db.normalize_book(flagged)
        assert any("quote intensity" in n for n in notes)

    def test_a_clean_book_says_nothing(self, raw) -> None:
        _, notes = db.normalize_book(raw)
        assert not any("inconsistent" in n for n in notes)

    def test_combined_bits_are_read_independently(self, raw) -> None:
        flagged = raw.copy()
        flagged.loc[:9, "flags"] = db.F_MAYBE_BAD_BOOK | db.F_SNAPSHOT
        _, notes = db.normalize_book(flagged)
        assert any("may be inconsistent" in n for n in notes)
        assert any("snapshot" in n.lower() for n in notes)


class TestQuotesAndTrades:
    def test_quotes_take_the_quote_panel_names(self, raw) -> None:
        out, _ = db.normalize_quotes(raw)
        assert {"bid_price", "ask_price", "bid_size", "ask_size"} <= set(out.columns)
        assert 90.0 < out["bid_price"].median() < 110.0

    def test_quotes_carry_no_level_suffix(self, raw) -> None:
        out, _ = db.normalize_quotes(raw)
        assert "bid_price_0" not in out.columns

    def test_trades_are_scaled_too(self, raw) -> None:
        """The same fixed-point trap. A caller who normalized their book and
        hand-renamed their tape would hit it on the tape."""
        trades = pd.DataFrame(
            {
                "ts_recv": raw["ts_recv"],
                "price": raw["bid_px_00"],
                "size": np.full(len(raw), 100, dtype="int64"),
                "flags": raw["flags"],
            }
        )
        out, _ = db.normalize_trades(trades)
        assert 90.0 < out["price"].median() < 110.0
        assert {"timestamp", "price", "size"} <= set(out.columns)

    def test_a_frame_without_price_is_refused(self, raw) -> None:
        with pytest.raises(ValidationError, match="price"):
            db.normalize_trades(raw)


class TestRefusals:
    def test_an_empty_frame_is_refused(self, raw) -> None:
        """There is no dtype or magnitude to read the scale from, and
        guessing it is the error this module exists to prevent."""
        with pytest.raises(ValidationError, match="empty frame"):
            db.normalize_book(raw.iloc[:0])

    def test_a_frame_with_no_complete_level_is_refused(self, raw) -> None:
        broken = raw.drop(columns=[f"ask_sz_{i:02d}" for i in range(LEVELS)])
        with pytest.raises(ValidationError, match="no complete Databento book level"):
            db.normalize_book(broken)

    def test_asking_for_more_levels_than_exist_is_reported(self, raw) -> None:
        _, notes = db.normalize_book(raw, levels=20)
        assert any("20 levels asked for" in n for n in notes)
