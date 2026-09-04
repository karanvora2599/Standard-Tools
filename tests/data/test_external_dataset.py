"""
Registering data too large to copy, and refusing the ways it goes wrong.

WHAT THESE PIN. Not that the code runs -- that the four things which would
each produce a confident wrong number are refused or reported:

    a book with bid and ask transposed        -> blocking, not a warning
    an event set with the timestamps swapped  -> the leak point_in_time
                                                 exists to prevent
    a missing column                          -> refused at registration,
                                                 before anything reads a row
    a file that changed after registration    -> reported, because nothing
                                                 was copied and it can

The last of those has no equivalent anywhere else in this library. Every
other artifact is written by this code and frozen; an external file belongs
to the caller.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.agent.runtimes import handoff
from standard_quant_tools.data import external
from standard_quant_tools.data.external_validation import validate_external
from standard_quant_tools.error import ValidationError

LEVELS = 4
ROWS = 5_000


@pytest.fixture
def runs_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path / "runs"))
    return tmp_path


@pytest.fixture
def book_frame() -> pd.DataFrame:
    rng = np.random.default_rng(19)
    mid = 100.0 + np.cumsum(rng.normal(0, 0.01, ROWS))
    half = 0.01 + np.abs(rng.normal(0, 0.002, ROWS))
    data = {"timestamp": pd.date_range("2026-03-02 09:30", periods=ROWS, freq="100ms")}
    for level in range(LEVELS):
        offset = 0.01 * level
        data[f"bid_price_{level}"] = np.round(mid - half - offset, 4)
        data[f"ask_price_{level}"] = np.round(mid + half + offset, 4)
        data[f"bid_size_{level}"] = rng.integers(100, 5000, ROWS).astype(float)
        data[f"ask_size_{level}"] = rng.integers(100, 5000, ROWS).astype(float)
    return pd.DataFrame(data)


@pytest.fixture
def book_path(tmp_path, book_frame) -> str:
    path = tmp_path / "book.parquet"
    book_frame.to_parquet(path, index=False)
    return str(path)


@pytest.fixture
def events_frame() -> pd.DataFrame:
    rng = np.random.default_rng(23)
    return pd.DataFrame(
        {
            "event_time": pd.date_range("2026-01-31", periods=300, freq="D"),
            "available_time": pd.date_range("2026-02-28", periods=300, freq="D"),
            "entity": ["AAPL"] * 300,
            "eps": rng.normal(2.0, 0.3, 300),
        }
    )


class TestNothingIsCopied:
    """The property the whole module exists for."""

    def test_registering_writes_a_sidecar_and_no_data(
        self, runs_dir, book_path
    ) -> None:
        ref, handle = handoff.publish_external(
            book_path, "order_book_panel", "run1", "book"
        )
        assert ref == "sqt://order_book_panel/run1/book"
        written = sorted(p.name for p in (runs_dir / "runs" / "run1").iterdir())
        assert written == ["book._handoff.json"], (
            f"registration wrote {written}; a copy of the data is exactly "
            "what this path exists to avoid"
        )
        assert handle.rows == ROWS

    def test_resolving_gives_a_handle_not_a_frame(self, runs_dir, book_path) -> None:
        """
        A DataFrame here would defeat the point: a consumer written for a
        fetched panel would pull the whole file through an `.iloc` without
        anyone deciding to.
        """
        ref, _ = handoff.publish_external(book_path, "order_book_panel", "run1", "book")
        resolved = handoff.resolve(ref)
        assert isinstance(resolved, external.ExternalDataset)
        assert not isinstance(resolved, pd.DataFrame)

    def test_batches_stream_every_row(self, runs_dir, book_path) -> None:
        ref, _ = handoff.publish_external(book_path, "order_book_panel", "run1", "book")
        handle = handoff.resolve(ref)
        assert sum(len(b) for b in handle.batches(batch_rows=1000)) == ROWS

    def test_batches_project_columns(self, runs_dir, book_path) -> None:
        """Reading four columns of a sixty-column book should read four."""
        ref, _ = handoff.publish_external(book_path, "order_book_panel", "run1", "book")
        handle = handoff.resolve(ref)
        first = next(handle.batches(columns=["timestamp", "bid_price_0"]))
        assert list(first.columns) == ["timestamp", "bid_price_0"]

    def test_an_unknown_column_is_refused_by_name(self, runs_dir, book_path) -> None:
        ref, _ = handoff.publish_external(book_path, "order_book_panel", "run1", "book")
        handle = handoff.resolve(ref)
        with pytest.raises(ValidationError, match="no column"):
            list(handle.batches(columns=["bid_price_0", "not_a_column"]))


class TestTheKindIsCheckedBeforeAnyRowIsRead:
    def test_a_missing_column_is_refused_at_registration(
        self, runs_dir, tmp_path, book_frame
    ) -> None:
        path = tmp_path / "short.parquet"
        book_frame.drop(columns=["ask_size_0"]).to_parquet(path, index=False)
        with pytest.raises(ValidationError, match="ask_size_0"):
            handoff.publish_external(str(path), "order_book_panel", "run1", "s")

    def test_a_raw_databento_export_is_told_which_normalizer_to_run(self) -> None:
        """
        The refusal that saves the most time. `bid_px_00` IS `bid_price_0`,
        and "missing column bid_price_0" sends someone hunting for data that
        is right there under another name.
        """
        raw = ["ts_recv", "bid_px_00", "bid_sz_00", "ask_px_00", "ask_sz_00"]
        problems = external.check_schema("order_book_panel", raw)
        assert problems
        joined = " ".join(problems)
        assert "DATABENTO" in joined
        assert "normalize_book" in joined
        assert "bid_px_00" in joined

    def test_an_external_only_kind_cannot_be_published_from_memory(
        self, runs_dir, book_frame
    ) -> None:
        with pytest.raises(ValidationError, match="cannot be published from memory"):
            handoff.publish(book_frame, "order_book_panel", "run1", "book")

    def test_an_in_memory_kind_cannot_be_registered_by_path(
        self, runs_dir, book_path
    ) -> None:
        with pytest.raises(ValidationError, match="cannot be registered by path"):
            handoff.publish_external(book_path, "price_panel", "run1", "p")

    def test_the_kind_check_still_bites_on_resolve(self, runs_dir, book_path) -> None:
        ref, _ = handoff.publish_external(book_path, "order_book_panel", "run1", "book")
        with pytest.raises(ValidationError, match="expected a 'tick_tape'"):
            handoff.resolve(ref, expect="tick_tape")

    def test_a_second_registration_under_one_name_is_refused(
        self, runs_dir, book_path
    ) -> None:
        """A reference names one dataset. Repointing it changes what every
        existing holder resolves."""
        handoff.publish_external(book_path, "order_book_panel", "run1", "book")
        with pytest.raises(ValidationError, match="already registered"):
            handoff.publish_external(book_path, "order_book_panel", "run1", "book")


class TestOnlyTabularFilesAreReadable:
    """
    The containment that replaces the SQT_RUNS_DIR bound.

    The path is caller-supplied and reaches this library from an agent, so a
    reader that accepts any file is a way to read any file on the machine
    into a tool result.
    """

    @pytest.mark.parametrize("name", ["secrets.env", "id_rsa", "notes.docx"])
    def test_a_non_tabular_file_is_refused(self, tmp_path, name) -> None:
        path = tmp_path / name
        path.write_text("API_KEY=hunter2\n", encoding="utf-8")
        with pytest.raises(ValidationError, match="does not read"):
            external.inspect(str(path), kind="tick_tape")

    def test_a_missing_path_is_refused_by_name(self, tmp_path) -> None:
        with pytest.raises(ValidationError, match="no file or directory"):
            external.inspect(str(tmp_path / "absent.parquet"), kind="tick_tape")

    def test_an_empty_path_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="needs a path"):
            external.resolve_path("   ")


class TestAChangedFileIsReported:
    def test_a_rewrite_is_detected(self, runs_dir, book_path, book_frame) -> None:
        """
        Nothing was copied, so the reference resolves to whatever is at the
        path NOW. That is the cost of not copying, and it is reported rather
        than left to be discovered.
        """
        ref, _ = handoff.publish_external(book_path, "order_book_panel", "run1", "book")
        assert handoff.describe(ref)["changed_since_registration"] is False
        time.sleep(0.01)
        book_frame.to_parquet(book_path, index=False)
        assert handoff.describe(ref)["changed_since_registration"] is True

    def test_a_deleted_file_refuses_with_the_reason(
        self, runs_dir, tmp_path, book_frame
    ) -> None:
        path = tmp_path / "transient.parquet"
        book_frame.to_parquet(path, index=False)
        ref, _ = handoff.publish_external(str(path), "order_book_panel", "run1", "book")
        path.unlink()
        with pytest.raises(ValidationError, match="only as good as the file"):
            handoff.resolve(ref)

    def test_the_fingerprint_is_not_called_a_content_hash(
        self, runs_dir, book_path
    ) -> None:
        """
        A weaker claim spelled with a different key. `content_hash` on a
        published artifact means the bytes; `fingerprint` here means name,
        size and mtime, and conflating them would overstate the guarantee.
        """
        ref, _ = handoff.publish_external(book_path, "order_book_panel", "run1", "book")
        described = handoff.describe(ref)
        assert "fingerprint" in described
        assert "content_hash" not in described
        assert described["storage"] == "external"


class TestValidationCatchesTheWrongNumbers:
    def _register(self, frame, tmp_path, kind, name="d"):
        path = tmp_path / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        return external.inspect(str(path), kind=kind)

    def test_a_clean_book_is_usable(self, tmp_path, book_frame) -> None:
        report = validate_external(
            self._register(book_frame, tmp_path, "order_book_panel")
        )
        assert report.usable, report.blocking
        assert report.stats["levels"] == LEVELS
        assert report.rows_scanned == ROWS

    def test_transposed_bid_and_ask_blocks(self, tmp_path, book_frame) -> None:
        """
        The failure a schema check cannot see: every column is present and
        every value is a real price. Only the ORDER is wrong, and a book
        that is 100% crossed is not a locked market.
        """
        swapped = book_frame.copy()
        for level in range(LEVELS):
            bid, ask = f"bid_price_{level}", f"ask_price_{level}"
            swapped[bid], swapped[ask] = (
                book_frame[ask].copy(),
                book_frame[bid].copy(),
            )
        report = validate_external(
            self._register(swapped, tmp_path, "order_book_panel")
        )
        assert not report.usable
        assert any("crossed" in b for b in report.blocking)
        assert any("wrong way round" in b for b in report.blocking)

    def test_a_few_crossed_snapshots_only_warn(self, tmp_path, book_frame) -> None:
        """Three crossed books in nine million rows is fine; a third of them
        is transposed columns. Only a count separates the two."""
        nearly = book_frame.copy()
        nearly.loc[:9, "bid_price_0"] = nearly.loc[:9, "ask_price_0"] + 0.01
        report = validate_external(self._register(nearly, tmp_path, "order_book_panel"))
        assert report.usable, report.blocking
        assert any("crossed" in w for w in report.warnings)

    def test_a_one_level_book_warns_that_depth_means_nothing(
        self, tmp_path, book_frame
    ) -> None:
        thin = book_frame[
            ["timestamp", "bid_price_0", "bid_size_0", "ask_price_0", "ask_size_0"]
        ]
        report = validate_external(self._register(thin, tmp_path, "order_book_panel"))
        assert report.usable
        assert any("ONE complete level" in w for w in report.warnings)

    def test_out_of_order_timestamps_warn(self, tmp_path, book_frame) -> None:
        shuffled = book_frame.sample(frac=1.0, random_state=5)
        report = validate_external(
            self._register(shuffled, tmp_path, "order_book_panel")
        )
        assert report.stats["rows_out_of_time_order"] > 0
        assert any("arrive earlier" in w for w in report.warnings)

    def test_swapped_event_and_available_time_blocks(
        self, tmp_path, events_frame
    ) -> None:
        """
        The check `point_in_time.py` exists for, reached through a file
        instead of through 5,000 inline rows. A record available BEFORE the
        period it describes makes every model built on it look prescient.
        """
        leaked = events_frame.rename(
            columns={"event_time": "available_time", "available_time": "event_time"}
        )
        report = validate_external(self._register(leaked, tmp_path, "event_panel"))
        assert not report.usable
        assert any("prescient" in b for b in report.blocking)

    def test_publication_lag_is_measured(self, tmp_path, events_frame) -> None:
        report = validate_external(
            self._register(events_frame, tmp_path, "event_panel")
        )
        assert report.usable, report.blocking
        assert report.stats["mean_publication_lag_days"] == pytest.approx(28.0)

    def test_zero_lag_warns_about_a_copied_column(self, tmp_path) -> None:
        same = pd.DataFrame(
            {
                "event_time": pd.date_range("2026-01-01", periods=100, freq="D"),
                "available_time": pd.date_range("2026-01-01", periods=100, freq="D"),
                "value": np.arange(100, dtype=float),
            }
        )
        report = validate_external(self._register(same, tmp_path, "event_panel"))
        assert any("no publication lag" in w for w in report.warnings)

    def test_non_positive_trade_prices_block(self, tmp_path) -> None:
        tape = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-03-02", periods=100, freq="s"),
                "price": [0.0] * 5 + [100.0] * 95,
                "size": [10.0] * 100,
            }
        )
        report = validate_external(self._register(tape, tmp_path, "tick_tape"))
        assert not report.usable
        assert any("non-positive price" in b for b in report.blocking)

    def test_an_empty_dataset_blocks(self, tmp_path, book_frame) -> None:
        report = validate_external(
            self._register(book_frame.iloc[:0], tmp_path, "order_book_panel")
        )
        assert not report.usable
        assert any("no rows" in b for b in report.blocking)


class TestTheScanIsBoundedAndSaysSo:
    def test_the_limit_truncates_and_reports(self, tmp_path, book_frame) -> None:
        path = tmp_path / "book.parquet"
        book_frame.to_parquet(path, index=False)
        handle = external.inspect(str(path), kind="order_book_panel")
        report = validate_external(handle, scan_limit=1000, batch_rows=500)
        assert report.truncated
        assert report.rows_scanned == 1000
        assert report.coverage() == pytest.approx(1000 / ROWS)
        assert any("stopped after" in w for w in report.warnings)

    def test_counts_are_reported_against_what_was_scanned(
        self, tmp_path, book_frame
    ) -> None:
        path = tmp_path / "book.parquet"
        book_frame.to_parquet(path, index=False)
        handle = external.inspect(str(path), kind="order_book_panel")
        report = validate_external(handle, scan_limit=1000)
        assert report.rows_total == ROWS
        assert report.rows_scanned == 1000


class TestBookLevelsCountsCompleteLevelsOnly:
    def test_a_level_missing_one_column_does_not_count(self) -> None:
        """
        `book_metrics` reading a price with no size would weight the touch
        against a missing size and report a microprice that leans on nothing.
        """
        columns = [
            "bid_price_0",
            "bid_size_0",
            "ask_price_0",
            "ask_size_0",
            "bid_price_1",
            "ask_price_1",
            "ask_size_1",  # no bid_size_1
        ]
        assert external.book_levels(columns) == 1

    def test_counting_stops_at_the_first_gap(self) -> None:
        columns = [
            "bid_price_0",
            "bid_size_0",
            "ask_price_0",
            "ask_size_0",
            "bid_price_7",
            "bid_size_7",
            "ask_price_7",
            "ask_size_7",
        ]
        assert external.book_levels(columns) == 1

    def test_no_levels_at_all(self) -> None:
        assert external.book_levels(["timestamp", "price", "size"]) == 0


class TestCsvIsReadableToo:
    """Vendors ship CSV, and the row count costs a scan rather than a footer
    read -- a real difference, reported rather than hidden."""

    def test_a_csv_registers_and_validates(self, runs_dir, tmp_path, book_frame):
        path = tmp_path / "book.csv"
        book_frame.to_csv(path, index=False)
        ref, handle = handoff.publish_external(
            str(path), "order_book_panel", "run1", "csvbook"
        )
        assert handle.fmt == "csv"
        assert handle.rows == ROWS
        report = validate_external(handoff.resolve(ref))
        assert report.usable, report.blocking


class TestADirectoryIsOneDataset:
    def test_partitions_are_read_together(self, runs_dir, tmp_path, book_frame):
        directory = tmp_path / "partitioned"
        directory.mkdir()
        for index, chunk in enumerate(np.array_split(book_frame, 4)):
            chunk.to_parquet(directory / f"part-{index}.parquet", index=False)
        ref, handle = handoff.publish_external(
            str(directory), "order_book_panel", "run1", "parts"
        )
        assert handle.n_files == 4
        assert handle.rows == ROWS
        assert sum(len(b) for b in handoff.resolve(ref).batches()) == ROWS

    def test_a_directory_with_nothing_readable_is_refused(self, tmp_path) -> None:
        directory = tmp_path / "empty"
        directory.mkdir()
        (directory / "README.md").write_text("nothing here", encoding="utf-8")
        with pytest.raises(ValidationError, match="no Parquet or CSV"):
            external.inspect(str(directory), kind="order_book_panel")


class TestTheBookToolReadsAReference:
    """
    The point of the phase, pinned.

    `get_order_book_metrics` and `analysis/order_book.py` have existed and
    been tested against synthetic books since before any source could feed
    them -- `DataProvider.get_order_book` raises `NotImplementedError` in
    every shipped provider, and `book_tools.py` says so in its own docstring.
    These tests are the first time the tool reads a book it did not receive
    as a literal argument.
    """

    @staticmethod
    def _register(book_frame, tmp_path, name="book"):
        path = tmp_path / f"{name}.parquet"
        book_frame.to_parquet(path, index=False)
        ref, _ = handoff.publish_external(str(path), "order_book_panel", "run1", name)
        return ref

    def test_metrics_come_back_finite_off_a_reference(
        self, runs_dir, tmp_path, book_frame
    ) -> None:
        from standard_quant_tools.agent.runtimes.microstructure import (
            get_order_book_metrics,
        )
        from standard_quant_tools.agent.runtimes.microstructure.book_tools import (
            OrderBookInput,
        )

        result = get_order_book_metrics(
            OrderBookInput(ref=self._register(book_frame, tmp_path))
        )
        assert result.n_snapshots == ROWS
        assert result.levels_read == LEVELS
        for value in (
            result.mean_microprice,
            result.mean_spread_bps,
            result.depth_slope,
            result.mean_touch_imbalance,
        ):
            assert value is not None and np.isfinite(value)
        assert 90.0 < result.mean_microprice < 110.0

    def test_inline_and_reference_agree(self, runs_dir, tmp_path, book_frame) -> None:
        """Two ways in, one answer. A divergence would mean the batched read
        is not reconstructing the same book."""
        from standard_quant_tools.agent.runtimes.microstructure import (
            get_order_book_metrics,
        )
        from standard_quant_tools.agent.runtimes.microstructure.book_tools import (
            OrderBookInput,
        )

        by_ref = get_order_book_metrics(
            OrderBookInput(ref=self._register(book_frame, tmp_path))
        )
        inline = get_order_book_metrics(
            OrderBookInput(
                snapshots=book_frame.drop(columns=["timestamp"]).to_dict("records")
            )
        )
        assert by_ref.mean_microprice == pytest.approx(inline.mean_microprice)
        assert by_ref.depth_slope == pytest.approx(inline.depth_slope)
        assert by_ref.mean_touch_imbalance == pytest.approx(inline.mean_touch_imbalance)

    def test_the_cap_says_it_read_a_prefix(
        self, runs_dir, tmp_path, book_frame
    ) -> None:
        """
        These statistics are MEANS, so a cap makes them a mean over the
        start of the session -- which is its least typical part. Saying so
        is the difference between a bounded answer and a wrong one.
        """
        from standard_quant_tools.agent.runtimes.microstructure import (
            get_order_book_metrics,
        )
        from standard_quant_tools.agent.runtimes.microstructure.book_tools import (
            OrderBookInput,
        )

        result = get_order_book_metrics(
            OrderBookInput(ref=self._register(book_frame, tmp_path), max_snapshots=500)
        )
        assert result.n_snapshots == 500
        assert any("PREFIX" in w for w in result.warnings)

    def test_a_wrong_kind_of_reference_is_refused(
        self, runs_dir, tmp_path, book_frame
    ) -> None:
        from standard_quant_tools.agent.runtimes.microstructure import (
            get_order_book_metrics,
        )
        from standard_quant_tools.agent.runtimes.microstructure.book_tools import (
            OrderBookInput,
        )

        path = tmp_path / "tape.parquet"
        pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-03-02", periods=50, freq="s"),
                "price": np.full(50, 100.0),
                "size": np.full(50, 10.0),
            }
        ).to_parquet(path, index=False)
        ref, _ = handoff.publish_external(str(path), "tick_tape", "run1", "tape")
        with pytest.raises(ValidationError, match="expected a 'order_book_panel'"):
            get_order_book_metrics(OrderBookInput(ref=ref))

    def test_exactly_one_source_is_required(self) -> None:
        import pydantic

        from standard_quant_tools.agent.runtimes.microstructure.book_tools import (
            OrderBookInput,
        )

        with pytest.raises(pydantic.ValidationError, match="exactly one"):
            OrderBookInput()
        with pytest.raises(pydantic.ValidationError, match="exactly one"):
            OrderBookInput(snapshots=[{"bid_price_0": 1.0}], ref="sqt://a/b/c")


class TestTheOrderEventPanelCanBeRegisteredAtAll:
    """One string, and a whole chain.

    `event_tools.py` resolves with `expect="order_event_panel"`, the kind
    has a column contract, a description and a place in `EXTERNAL_KINDS`,
    and the batched reader handles it. The ONLY tool that mints an external
    reference offered `event_panel` instead -- a different kind entirely,
    `(event_time, available_time)`, the point-in-time contract -- and never
    offered this one. So no reference of this kind could exist, and
    `get_order_event_metrics`' `ref` path was unreachable.

    The inline `events` path still worked, which is why nothing failed. But
    that field's own description says a session of market-by-order "is
    millions of records and cannot travel through a tool argument": what was
    stranded was the only path that scales.
    """

    @staticmethod
    def _events(n=200, seed=7):
        rng = np.random.default_rng(seed)
        stamps = pd.Timestamp("2024-03-01 14:30:00") + pd.to_timedelta(
            np.sort(rng.integers(0, 600_000, n)), unit="ms"
        )
        actions = rng.choice(list("ACMFT"), n, p=[0.42, 0.34, 0.09, 0.10, 0.05])
        return pd.DataFrame(
            {
                "timestamp": stamps,
                "order_id": rng.integers(1, 90, n),
                "action": actions,
                "side": rng.choice(["B", "A"], n),
                "price": (100 + rng.normal(0, 0.05, n)).round(2),
                "size": rng.integers(1, 500, n),
            }
        )

    def _register(self, frame, tmp_path, name="mbo"):
        path = tmp_path / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        return external.inspect(str(path), kind="order_event_panel")

    def test_the_kind_is_offered_by_the_only_tool_that_mints_one(self):
        from typing import get_args

        from standard_quant_tools.agent.runtimes.data.models import (
            RegisterExternalDatasetInput,
        )

        offered = set(
            get_args(RegisterExternalDatasetInput.model_fields["kind"].annotation)
        )
        assert "order_event_panel" in offered

    def test_every_external_kind_is_reachable(self):
        """The guard that stops this recurring. A kind in `EXTERNAL_KINDS`
        that the register tool does not offer is a kind no reference can
        ever have, and the two drifted apart silently for exactly one."""
        from typing import get_args

        from standard_quant_tools.agent.runtimes.data.models import (
            RegisterExternalDatasetInput,
        )
        from standard_quant_tools.agent.runtimes.handoff import EXTERNAL_KINDS

        offered = set(
            get_args(RegisterExternalDatasetInput.model_fields["kind"].annotation)
        )
        assert offered == set(EXTERNAL_KINDS), (
            f"unmintable kinds: {sorted(set(EXTERNAL_KINDS) - offered)}; "
            f"offered but not external: {sorted(offered - set(EXTERNAL_KINDS))}"
        )

    def test_the_column_contract_matches_what_will_read_it(self):
        """`price` was missing from the registration contract, so a panel
        could satisfy registration and then fail inside the metrics."""
        from standard_quant_tools.analysis.order_events import ORDER_EVENT_COLUMNS

        assert set(external.KIND_COLUMNS["order_event_panel"]) == set(
            ORDER_EVENT_COLUMNS
        )

    def test_a_clean_panel_registers_and_validates(self, tmp_path):
        report = validate_external(self._register(self._events(), tmp_path))
        assert report.usable, report.blocking
        assert report.rows_scanned == 200

    def test_a_panel_missing_price_is_refused(self, tmp_path):
        frame = self._events().drop(columns=["price"])
        report = validate_external(self._register(frame, tmp_path, name="noprice"))
        assert not report.usable
        assert any("price" in problem for problem in report.blocking)

    def test_a_raw_databento_export_is_told_which_normalizer_to_run(self):
        """`check_schema` names the fix when it recognizes the vendor, and
        the map it reads had no entry for this kind -- so an MBO export
        would have been told to run `normalize_book`, which produces a
        different kind entirely."""
        from standard_quant_tools.data.external import _DATABENTO_NORMALIZER

        assert _DATABENTO_NORMALIZER["order_event_panel"] == "normalize_mbo"

    @pytest.mark.parametrize(
        "column,values,stat,expected",
        [
            ("action", ["A", "Z", "C", "Q"], "unknown_action", 2),
            ("side", ["B", "X", "A", "B"], "unknown_side", 1),
            ("size", [10, 0, -5, 10], "non_positive_size", 2),
        ],
    )
    def test_the_validator_has_a_handler_at_all(
        self, tmp_path, column, values, stat, expected
    ):
        """`feed()` dispatches on `_check_{kind}` and no-ops when there is
        none. This kind had none, so a panel would have been scanned and
        pronounced on without even its timestamps being checked."""
        frame = self._events(n=4)
        frame[column] = values
        report = validate_external(self._register(frame, tmp_path, name=column))
        assert report.stats.get(stat) == expected

    def test_a_cancel_without_a_price_is_not_counted_against_it(self, tmp_path):
        """A cancel or a clear legitimately carries no price; an add or a
        fill without one is a row the metrics cannot use."""
        frame = self._events(n=4)
        frame["action"] = ["A", "C", "F", "C"]
        frame["price"] = [100.0, float("nan"), float("nan"), float("nan")]
        report = validate_external(self._register(frame, tmp_path, name="cxlprice"))
        assert report.stats.get("non_finite_price_on_priced_event") == 1

    def test_the_reference_reaches_the_tool_that_was_stranded(
        self, tmp_path, monkeypatch
    ):
        """The point of all of it."""
        from standard_quant_tools.agent.runtimes import resolve

        # The registry lives on disk and outlives a pytest session, so a
        # test that publishes must own its runs directory -- the convention
        # the rest of the suite already follows.
        monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path / "runs"))

        frame = self._events(n=500)
        path = tmp_path / "mbo.parquet"
        frame.to_parquet(path, index=False)
        registered = resolve("data").dispatch(
            "register_external_dataset",
            {
                "path": str(path),
                "kind": "order_event_panel",
                "run_id": "l2",
                "name": "mbo",
            },
        )
        assert registered["ref"].startswith("sqt://order_event_panel/")
        metrics = resolve("microstructure").dispatch(
            "get_order_event_metrics", {"ref": registered["ref"]}
        )
        assert metrics["n_events"] == 500
        assert set(metrics["counts_by_action"]) <= set("ACMFTR")
