"""
The conversion layer that had no door.

`data/databento.py` is 643 lines with 39 dedicated tests and had ZERO
references anywhere under `agent/` -- not even an export from
`data/__init__`. Four normalizers, mapping exactly onto the four external
kinds `register_external_dataset` mints, and no way for an agent to run any
of them.

`external.check_schema` made it sharper than a plain omission: when it
recognized a raw vendor export it already told the caller to run
`normalize_book()` by name. The remedy was named and could not be executed.

These tests cover the tool that closes it, and one thing the tool had to get
right that the library never had to: `normalize_book` drops a level that is
empty in EVERY snapshot, which a single batch cannot know. Streaming it means
deciding the depth in its own pass and forcing it for the writing pass, and
the test that matters is that this produces exactly what one in-memory call
would have.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.agent.runtimes import resolve
from standard_quant_tools.data.databento import normalize_book
from standard_quant_tools.error import ValidationError

UNDEF_PRICE = 9_223_372_036_854_775_807
ROWS = 2_000
LIVE_LEVELS = 4
VENDOR_LEVELS = 10


@pytest.fixture
def runtime():
    return resolve("data")


@pytest.fixture
def mid():
    return 100.0 + np.cumsum(np.random.default_rng(11).normal(0, 0.002, ROWS))


@pytest.fixture
def raw_book(mid):
    """An MBP-10 export in Databento's own spelling.

    Ten levels subscribed, four ever populated -- the case the empty-level
    rule exists for. Prices are int64 nanodollars and absent levels carry
    int64 max, which are the two traps that do not look like errors.
    """
    rng = np.random.default_rng(5)
    stamps = pd.Timestamp("2024-03-01 14:30") + pd.to_timedelta(
        np.arange(ROWS) * 7, unit="ms"
    )
    frame = {
        "ts_recv": stamps.astype("int64"),
        "ts_event": (stamps - pd.Timedelta("3ms")).astype("int64"),
    }
    for level in range(VENDOR_LEVELS):
        live = level < LIVE_LEVELS
        offset = (level + 1) * 0.01
        frame[f"bid_px_{level:02d}"] = np.where(
            live, ((mid - offset) * 1e9).astype("int64"), UNDEF_PRICE
        )
        frame[f"ask_px_{level:02d}"] = np.where(
            live, ((mid + offset) * 1e9).astype("int64"), UNDEF_PRICE
        )
        frame[f"bid_sz_{level:02d}"] = np.where(live, rng.integers(1, 500, ROWS), 0)
        frame[f"ask_sz_{level:02d}"] = np.where(live, rng.integers(1, 500, ROWS), 0)
    return pd.DataFrame(frame)


def _write(frame, tmp_path, name="raw"):
    path = tmp_path / f"{name}.parquet"
    frame.to_parquet(path, index=False)
    return str(path)


def _prepare(runtime, tmp_path, raw, **overrides):
    payload = {
        "path": _write(raw, tmp_path, overrides.pop("src_name", "raw")),
        "kind": overrides.pop("kind", "order_book_panel"),
        "out_path": str(tmp_path / overrides.pop("out_name", "out.parquet")),
    }
    payload.update(overrides)
    return runtime.dispatch("prepare_vendor_extract", payload)


class TestTheDoorExists:
    def test_the_tool_is_registered_and_reachable(self):
        from standard_quant_tools.agent.tools import _TOOL_DISPATCH

        assert "prepare_vendor_extract" in _TOOL_DISPATCH

    def test_every_kind_the_register_tool_mints_has_a_normalizer_or_a_reason(self):
        """`event_panel` is the one external kind with no normalizer, and
        deliberately so: it is a point-in-time contract, not a market-data
        shape. Every OTHER mintable kind must be preparable, or the door is
        ajar rather than open."""
        from typing import get_args

        from standard_quant_tools.agent.runtimes.data.models import (
            PrepareVendorExtractInput,
            RegisterExternalDatasetInput,
        )

        mintable = set(
            get_args(RegisterExternalDatasetInput.model_fields["kind"].annotation)
        )
        preparable = set(
            get_args(PrepareVendorExtractInput.model_fields["kind"].annotation)
        )
        assert mintable - preparable == {"event_panel"}

    def test_the_normalizer_map_names_functions_that_exist(self):
        from standard_quant_tools.agent.runtimes.data.tools import _NORMALIZERS
        from standard_quant_tools.data import databento

        for kind, name in _NORMALIZERS.items():
            assert hasattr(databento, name), f"{kind} names a missing {name}"


class TestTheTwoTrapsThatDoNotLookLikeErrors:
    def test_fixed_point_prices_are_de_scaled(self, runtime, tmp_path, raw_book, mid):
        """A factor of a billion does not look like an error, it looks like
        a market -- `check_schema` warns that registering unconverted would
        put $9.2 billion quotes in your book."""
        result = _prepare(runtime, tmp_path, raw_book)
        converted = pd.read_parquet(result["out_path"])
        assert converted["bid_price_0"].iloc[0] == pytest.approx(
            mid[0] - 0.01, abs=1e-9
        )
        assert converted["bid_price_0"].max() < 1_000

    def test_the_scale_decision_is_reported(self, runtime, tmp_path, raw_book):
        result = _prepare(runtime, tmp_path, raw_book)
        assert any("price_scale" in note for note in result["notes"])

    def test_sentinels_become_null_not_a_number(self, runtime, tmp_path, raw_book):
        result = _prepare(runtime, tmp_path, raw_book, keep_empty_levels=True)
        converted = pd.read_parquet(result["out_path"])
        assert converted["bid_price_9"].isna().all()
        assert (converted["bid_price_9"] == UNDEF_PRICE).sum() == 0
        assert any("sentinel" in note for note in result["notes"])

    def test_which_timestamp_was_taken_is_reported(self, runtime, tmp_path, raw_book):
        result = _prepare(runtime, tmp_path, raw_book)
        assert any("ts_recv" in note for note in result["notes"])

    def test_the_timestamp_can_be_overridden(self, runtime, tmp_path, raw_book):
        recv = pd.read_parquet(
            _prepare(runtime, tmp_path, raw_book, out_name="a.parquet")["out_path"]
        )
        event = pd.read_parquet(
            _prepare(
                runtime,
                tmp_path,
                raw_book,
                out_name="b.parquet",
                timestamp="ts_event",
            )["out_path"]
        )
        assert (recv["timestamp"] - event["timestamp"]).eq(pd.Timedelta("3ms")).all()


class TestStreamingMatchesOneCall:
    """The correctness risk the tool introduces and the library never had.

    `normalize_book` drops a trailing level empty in EVERY snapshot, a fact
    no single batch holds. So the depth is decided in its own pass and then
    FORCED for the writing pass. If those two disagree, a converted book is
    quietly a different object from a hand-normalized one.
    """

    @pytest.mark.parametrize("batch_rows", [ROWS * 2, 512, 97])
    def test_the_result_is_batch_size_independent(
        self, runtime, tmp_path, raw_book, batch_rows
    ):
        expected, _ = normalize_book(raw_book)
        result = _prepare(
            runtime,
            tmp_path,
            raw_book,
            out_name=f"b{batch_rows}.parquet",
            batch_rows=batch_rows,
        )
        got = pd.read_parquet(result["out_path"])
        assert list(got.columns) == list(expected.columns)
        pd.testing.assert_frame_equal(
            got.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=False,
        )

    def test_the_empty_trailing_levels_are_dropped(self, runtime, tmp_path, raw_book):
        result = _prepare(runtime, tmp_path, raw_book, batch_rows=256)
        assert result["levels_available"] == VENDOR_LEVELS
        assert result["levels_kept"] == LIVE_LEVELS
        # Same sentence the library emits, because it is the same rule.
        assert any(
            "empty in every one of" in note and "levels deep" in note
            for note in result["notes"]
        )

    def test_keeping_them_preserves_the_vendor_width(self, runtime, tmp_path, raw_book):
        result = _prepare(
            runtime, tmp_path, raw_book, keep_empty_levels=True, batch_rows=256
        )
        converted = pd.read_parquet(result["out_path"])
        assert "bid_price_9" in converted.columns

    def test_a_gap_in_the_middle_is_not_closed(self, runtime, tmp_path, mid):
        """Only a TRAILING run is dropped. Level 1 empty while level 2 has
        size is not a thin book, it is a broken extract, and closing the gap
        would hide it."""
        rng = np.random.default_rng(2)
        stamps = pd.Timestamp("2024-03-01") + pd.to_timedelta(
            np.arange(ROWS) * 7, unit="ms"
        )
        frame = {"ts_recv": stamps.astype("int64")}
        for level in range(3):
            hollow = level == 1
            frame[f"bid_px_{level:02d}"] = np.where(
                hollow, UNDEF_PRICE, (mid * 1e9).astype("int64")
            )
            frame[f"ask_px_{level:02d}"] = np.where(
                hollow, UNDEF_PRICE, (mid * 1e9).astype("int64")
            )
            frame[f"bid_sz_{level:02d}"] = np.where(hollow, 0, rng.integers(1, 9, ROWS))
            frame[f"ask_sz_{level:02d}"] = np.where(hollow, 0, rng.integers(1, 9, ROWS))
        result = _prepare(
            runtime, tmp_path, pd.DataFrame(frame), batch_rows=256, src_name="gap"
        )
        assert result["levels_kept"] == 3
        assert pd.read_parquet(result["out_path"])["bid_price_1"].isna().all()

    def test_a_one_sided_level_is_not_dropped(self, runtime, tmp_path, mid):
        """The bug the first version of this had. A level with no bid but a
        live ask is ONE-SIDED, which is a real state in a thin book -- the
        library requires BOTH sides absent before calling a level empty, and
        a bid-only test would have discarded the side that was there."""
        rng = np.random.default_rng(4)
        stamps = pd.Timestamp("2024-03-01") + pd.to_timedelta(
            np.arange(ROWS) * 7, unit="ms"
        )
        frame = {"ts_recv": stamps.astype("int64")}
        for level in range(3):
            ask_only = level == 2
            frame[f"bid_px_{level:02d}"] = np.where(
                ask_only, UNDEF_PRICE, (mid * 1e9).astype("int64")
            )
            frame[f"ask_px_{level:02d}"] = (mid * 1e9).astype("int64")
            frame[f"bid_sz_{level:02d}"] = np.where(
                ask_only, 0, rng.integers(1, 9, ROWS)
            )
            frame[f"ask_sz_{level:02d}"] = rng.integers(1, 9, ROWS)
        result = _prepare(
            runtime,
            tmp_path,
            pd.DataFrame(frame),
            batch_rows=256,
            src_name="onesided",
        )
        assert result["levels_kept"] == 3, "a live ask was thrown away"
        converted = pd.read_parquet(result["out_path"])
        assert converted["ask_price_2"].notna().any()
        # And it is what one in-memory call would have produced, which is
        # the contract the bid-only test quietly broke.
        expected, _ = normalize_book(pd.DataFrame(frame))
        pd.testing.assert_frame_equal(
            converted.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=False,
        )

    def test_the_malformed_book_warning_survives_streaming(
        self, runtime, tmp_path, mid
    ):
        """`normalize_book` warns when an empty level sits BELOW a live one.
        Both streamed passes run with keep_empty_levels=True, so the library
        never gets to say it -- the tool has to."""
        rng = np.random.default_rng(2)
        stamps = pd.Timestamp("2024-03-01") + pd.to_timedelta(
            np.arange(ROWS) * 7, unit="ms"
        )
        frame = {"ts_recv": stamps.astype("int64")}
        for level in range(3):
            hollow = level == 1
            for side in ("bid", "ask"):
                frame[f"{side}_px_{level:02d}"] = np.where(
                    hollow, UNDEF_PRICE, (mid * 1e9).astype("int64")
                )
                frame[f"{side}_sz_{level:02d}"] = np.where(
                    hollow, 0, rng.integers(1, 9, ROWS)
                )
        result = _prepare(
            runtime, tmp_path, pd.DataFrame(frame), batch_rows=256, src_name="hollow"
        )
        assert any("malformed book" in w for w in result["warnings"])

    def test_a_book_with_nothing_in_it_is_refused(self, runtime, tmp_path):
        """The library would drop every column and hand back a frame with no
        book in it. Writing that to disk and calling it a conversion is
        worse than saying so."""
        stamps = pd.Timestamp("2024-03-01") + pd.to_timedelta(
            np.arange(200) * 7, unit="ms"
        )
        frame = {"ts_recv": stamps.astype("int64")}
        for level in range(3):
            for side in ("bid", "ask"):
                frame[f"{side}_px_{level:02d}"] = np.full(200, UNDEF_PRICE)
                frame[f"{side}_sz_{level:02d}"] = np.zeros(200, dtype="int64")
        with pytest.raises(ValidationError, match="no book here"):
            _prepare(runtime, tmp_path, pd.DataFrame(frame), src_name="void")

    def test_notes_are_one_fact_not_one_per_batch(self, runtime, tmp_path, raw_book):
        few = _prepare(
            runtime, tmp_path, raw_book, batch_rows=ROWS * 2, out_name="few.parquet"
        )
        many = _prepare(
            runtime, tmp_path, raw_book, batch_rows=97, out_name="many.parquet"
        )
        assert len(many["notes"]) == len(set(many["notes"]))
        assert len(many["notes"]) <= len(few["notes"]) + 1


class TestTheOtherThreeKinds:
    @pytest.fixture
    def stamps(self):
        return (
            pd.Timestamp("2024-03-01 14:30")
            + pd.to_timedelta(np.arange(ROWS) * 5, unit="ms")
        ).astype("int64")

    def test_quotes(self, runtime, tmp_path, stamps, mid):
        rng = np.random.default_rng(1)
        raw = pd.DataFrame(
            {
                "ts_recv": stamps,
                "bid_px_00": ((mid - 0.01) * 1e9).astype("int64"),
                "ask_px_00": ((mid + 0.01) * 1e9).astype("int64"),
                "bid_sz_00": rng.integers(1, 500, ROWS),
                "ask_sz_00": rng.integers(1, 500, ROWS),
            }
        )
        result = _prepare(runtime, tmp_path, raw, kind="quote_panel", src_name="q")
        converted = pd.read_parquet(result["out_path"])
        assert {"bid_price", "ask_price"} <= set(converted.columns)
        assert converted["bid_price"].max() < 1_000

    def test_trades(self, runtime, tmp_path, stamps, mid):
        raw = pd.DataFrame(
            {
                "ts_recv": stamps,
                "price": (mid * 1e9).astype("int64"),
                "size": np.random.default_rng(1).integers(1, 900, ROWS),
            }
        )
        result = _prepare(runtime, tmp_path, raw, kind="tick_tape", src_name="t")
        converted = pd.read_parquet(result["out_path"])
        assert converted["price"].max() < 1_000

    def test_order_events(self, runtime, tmp_path, stamps, mid):
        rng = np.random.default_rng(1)
        raw = pd.DataFrame(
            {
                "ts_recv": stamps,
                "order_id": rng.integers(1, 400, ROWS),
                "action": rng.choice(list("ACMFT"), ROWS),
                "side": rng.choice(["B", "A"], ROWS),
                "price": (mid * 1e9).astype("int64"),
                "size": rng.integers(1, 500, ROWS),
            }
        )
        result = _prepare(
            runtime, tmp_path, raw, kind="order_event_panel", src_name="e"
        )
        converted = pd.read_parquet(result["out_path"])
        assert {"order_id", "action", "side"} <= set(converted.columns)
        assert converted["price"].max() < 1_000


class TestTheGuardsAroundWriting:
    def test_a_dry_run_writes_nothing_and_still_reports(
        self, runtime, tmp_path, raw_book
    ):
        result = _prepare(runtime, tmp_path, raw_book, dry_run=True)
        assert result["out_path"] == ""
        assert not (tmp_path / "out.parquet").exists()
        assert any("price_scale" in note for note in result["notes"])
        assert any("DRY RUN" in w for w in result["warnings"])

    def test_it_will_not_overwrite(self, runtime, tmp_path, raw_book):
        _prepare(runtime, tmp_path, raw_book)
        with pytest.raises(ValidationError, match="already exists"):
            _prepare(runtime, tmp_path, raw_book)

    def test_a_vendor_export_is_recognized(self, runtime, tmp_path, raw_book):
        result = _prepare(runtime, tmp_path, raw_book)
        assert result["looked_like_databento"] is True
        assert "bid_px_00" in result["source_columns"]


class TestTheWholeChain:
    """prepare -> register -> validate -> the tool at the end of it."""

    def test_a_raw_export_reaches_the_book_metrics(
        self, runtime, tmp_path, raw_book, monkeypatch
    ):
        monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path / "runs"))
        prepared = _prepare(runtime, tmp_path, raw_book)
        registered = runtime.dispatch(
            "register_external_dataset",
            {
                "path": prepared["out_path"],
                "kind": "order_book_panel",
                "run_id": "l2",
                "name": "book",
                "source": "databento",
            },
        )
        report = runtime.dispatch(
            "validate_external_dataset", {"ref": registered["ref"]}
        )
        assert report["usable"], report.get("blocking")
        metrics = resolve("microstructure").dispatch(
            "get_order_book_metrics", {"ref": registered["ref"]}
        )
        assert metrics["n_snapshots"] == ROWS
        assert metrics["levels_available"] == LIVE_LEVELS
        # The spread is a cent either side, and would be a billion times
        # that if the fixed-point scaling had not happened.
        assert metrics["mean_spread"] == pytest.approx(0.02, abs=1e-6)

    def test_the_next_step_it_names_is_the_call_that_works(
        self, runtime, tmp_path, raw_book
    ):
        result = _prepare(runtime, tmp_path, raw_book)
        assert "register_external_dataset" in result["next_step"]
        assert result["kind"] in result["next_step"]
