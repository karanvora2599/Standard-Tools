"""
A consumer that needs a whole frame gets one from either storage.

`EXTERNAL_KINDS` states the invariant in its own comment: `tick_tape` and
`quote_panel` exist in both forms deliberately, "one kind, two storages, rather
than an `external_tick_tape` that would double the taxonomy and let a consumer
accept one and refuse the other". Four microstructure tools did accept one and
refuse the other, through two copies of the same helper, and reported a valid
registration as "resolved to nothing usable".

The reason the fix is a bounded read rather than a plain `pd.concat` is the
other half of the design: `ExternalDataset` is a distinct type so that nothing
materialises a registered file BY ACCIDENT. A ceiling with its size in the
refusal keeps that property while removing the asymmetry -- the decision is
still made in code, once, out loud.
"""

import pandas as pd
import pytest

from standard_quant_tools.agent.runtimes import handoff
from standard_quant_tools.agent.runtimes.handoff import resolve_frame
from standard_quant_tools.error import ValidationError


@pytest.fixture(autouse=True)
def runs_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("SQT_AUDIT_DIR", str(tmp_path / "audit"))
    return tmp_path


def _tape(rows: int = 40) -> pd.DataFrame:
    stamps = pd.date_range("2024-03-05 14:30:00", periods=rows, freq="1s", tz="UTC")
    return pd.DataFrame({
        "timestamp": stamps,
        "price": [100.0 + i * 0.01 for i in range(rows)],
        "size": [100] * rows,
    })


@pytest.fixture
def published_ref() -> str:
    """The fetched storage: `fetch_tick_tape` publishes a frame."""
    return handoff.publish(_tape(), kind="tick_tape", run_id="r1", name="fetched")


@pytest.fixture
def external_ref(tmp_path) -> str:
    """The registered storage: a tape bought from a vendor, left where it is."""
    path = tmp_path / "vendor" / "trades.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    _tape().to_parquet(path, index=False)
    ref, _handle = handoff.publish_external(
        str(path), kind="tick_tape", run_id="r1", name="bought"
    )
    return ref


class TestBothStoragesReachTheSameConsumer:
    def test_a_published_tape_resolves_to_a_frame(self, published_ref):
        frame = resolve_frame(published_ref, expect="tick_tape", who="t")
        assert isinstance(frame, pd.DataFrame) and len(frame) == 40

    def test_a_registered_tape_resolves_to_the_same_frame(
        self, published_ref, external_ref
    ):
        fetched = resolve_frame(published_ref, expect="tick_tape", who="t")
        bought = resolve_frame(external_ref, expect="tick_tape", who="t")
        assert isinstance(bought, pd.DataFrame)
        # Same content addressed two ways, which is what one kind and two
        # storages is supposed to mean.
        pd.testing.assert_frame_equal(
            bought[["price", "size"]].reset_index(drop=True),
            fetched[["price", "size"]].reset_index(drop=True),
        )

    def test_the_handle_itself_is_still_what_resolve_returns(self, external_ref):
        """The positive control on the premise.

        If `resolve` had started returning a frame for a registered dataset,
        the test above would pass for the wrong reason and `ExternalDataset`'s
        guarantee -- that nothing materialises forty gigabytes by accident --
        would be gone. `resolve_frame` is the opt-in; `resolve` is not.
        """
        from standard_quant_tools.data.external import ExternalDataset

        assert isinstance(handoff.resolve(external_ref, expect="tick_tape"), ExternalDataset)

    def test_a_rebuilt_frame_says_where_it_came_from(self, external_ref):
        """A registration records the vendor on the sidecar, not in the bytes,
        so a frame rebuilt from them would otherwise carry no provenance at
        all and every consumer reporting it would report none."""
        frame = resolve_frame(external_ref, expect="tick_tape", who="t")
        assert frame.attrs.get("external_path", "").endswith("trades.parquet")
        assert frame.attrs.get("source")


class TestTheCeilingIsNamedNotSilent:
    def test_a_dataset_over_the_ceiling_is_refused_with_its_size(self, external_ref):
        with pytest.raises(ValidationError) as caught:
            resolve_frame(external_ref, expect="tick_tape", who="t", max_rows=10)
        message = str(caught.value)
        assert "40" in message, "the size a caller needs in order to decide"
        assert "10" in message, "and the ceiling it exceeded"
        assert "nothing was computed" in message.lower()

    def test_an_unmeasured_dataset_is_refused_rather_than_clipped(
        self, external_ref, monkeypatch
    ):
        """A registration whose row count was never recorded.

        Streaming it has to stop somewhere, and stopping quietly would hand a
        microstructure estimator a clipped tape -- wrong, not approximate. So
        the stop is a refusal.
        """
        import standard_quant_tools.data.external as ext

        real = ext.inspect

        def unmeasured(*args, **kwargs):
            handle = real(*args, **kwargs)
            return type(handle)(**{**handle.__dict__, "rows": None})

        monkeypatch.setattr(ext, "inspect", unmeasured)
        with pytest.raises(ValidationError) as caught:
            resolve_frame(external_ref, expect="tick_tape", who="t", max_rows=5)
        assert "not recorded" in str(caught.value)
        assert "NOT truncated" in str(caught.value)

    def test_a_dataset_inside_the_ceiling_is_read_whole(self, external_ref):
        frame = resolve_frame(external_ref, expect="tick_tape", who="t", max_rows=40)
        assert len(frame) == 40, "the ceiling is inclusive, not off by one"


class TestRefusalsNameTheCause:
    def test_the_wrong_kind_still_names_both(self, published_ref):
        with pytest.raises(ValidationError) as caught:
            resolve_frame(published_ref, expect="quote_panel", who="t")
        assert "tick_tape" in str(caught.value) and "quote_panel" in str(caught.value)

    @pytest.mark.parametrize(
        "ref",
        [
            "sqt://tick_tape/gone/missing",   # a cleared runs directory
            "not-a-reference",               # a malformed string
            "sqt://tick_tape/r1/never",      # a name nothing published
        ],
    )
    def test_every_bad_reference_names_what_produces_a_good_one(self, ref):
        """The contract one of the two merged helpers had and the other did not.

        A caller holding a bad reference needs two things: what is wrong with
        the string, and where a good one comes from. The underlying messages
        give the first. Whether they happen to arrive as `ValidationError` or
        as something else is invisible to that caller, so wrapping only the
        non-`ValidationError` half -- which is what the merged helper first
        inherited -- dropped the producer hint from exactly the case the
        docstring calls out: a run directory that was cleared.
        """
        with pytest.raises(ValidationError) as caught:
            resolve_frame(
                ref, expect="tick_tape", who="t", produced_by="run fetch_tick_tape."
            )
        assert "fetch_tick_tape" in str(caught.value)

    def test_the_hint_is_the_callers_to_give(self):
        """`resolve_frame` is generic; only its caller knows the producer.

        Without one the refusal still says what is wrong, and says nothing it
        cannot know -- rather than naming tick-tape tools at a consumer of
        some other kind.
        """
        with pytest.raises(ValidationError) as caught:
            resolve_frame("sqt://tick_tape/gone/missing", expect="tick_tape", who="t")
        assert "fetch_tick_tape" not in str(caught.value)
        assert "could not be resolved" in str(caught.value)

    def test_an_empty_dataset_is_not_reported_as_an_unusable_one(self, tmp_path):
        """The message this change was really about.

        An empty result and an unreadable reference were one refusal, and
        "resolved to nothing usable" fit neither: the registration is fine and
        there is simply nothing in the file. It names the path, because the
        next thing a reader does is go and look at it.

        This goes through a registration rather than a publish because
        `publish` refuses an empty frame outright -- so the in-memory half of
        that refusal is unreachable, and is marked as such where it lives
        rather than tested here against a state no producer can create.
        """
        path = tmp_path / "vendor" / "nothing.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        _tape(0).to_parquet(path, index=False)
        ref, _ = handoff.publish_external(
            str(path), kind="tick_tape", run_id="r1", name="void"
        )
        with pytest.raises(ValidationError) as caught:
            resolve_frame(ref, expect="tick_tape", who="t")
        assert "no rows" in str(caught.value)
        assert "registration is good" in str(caught.value)
        assert "nothing.parquet" in str(caught.value)

    def test_publishing_an_empty_frame_is_refused_before_a_reference_exists(self):
        """Why the test above takes the external route."""
        with pytest.raises(ValidationError):
            handoff.publish(_tape(0), kind="tick_tape", run_id="r1", name="void")
class TestTheTimeIndexIsRestored:
    """The layer a column check could not see.

    `resolve_frame` returned a frame with `price`, `size` and the right row
    count, and `analysis/microstructure.py` refused it on its INDEX TYPE three
    frames further in -- a message that names what the estimator needs and
    cannot say that the storage is why. Both storages can arrive unindexed:
    Parquet carries an index as a column, and a hand-published frame need not
    have had one.
    """

    def test_a_registered_tape_comes_back_time_indexed(self, external_ref):
        frame = resolve_frame(external_ref, expect="tick_tape", who="t")
        assert isinstance(frame.index, pd.DatetimeIndex)
        assert frame.index.name == "timestamp"
        assert "timestamp" not in frame.columns, "on the index, not in both"

    def test_a_published_tape_stamped_in_a_column_is_indexed_too(self):
        """The same defect with the storages swapped.

        Fixing only the registered side left a published tape with its stamp
        in a column refused while the same bytes registered were accepted --
        so the normalisation belongs at the door both go through.
        """
        flat = _tape().reset_index(drop=True)
        assert "timestamp" in flat.columns
        ref = handoff.publish(flat, kind="tick_tape", run_id="r1", name="flat")
        frame = resolve_frame(ref, expect="tick_tape", who="t")
        assert isinstance(frame.index, pd.DatetimeIndex)

    def test_an_already_indexed_frame_is_left_alone(self):
        stamps = pd.date_range(
            "2024-03-05 14:30", periods=4, freq="1s", tz="UTC", name="timestamp"
        )
        indexed = pd.DataFrame({"price": [1.0, 2, 3, 4], "size": [1] * 4}, index=stamps)
        ref = handoff.publish(indexed, kind="tick_tape", run_id="r1", name="idx")
        frame = resolve_frame(ref, expect="tick_tape", who="t")
        pd.testing.assert_index_equal(frame.index, stamps)

    def test_a_pandas_written_index_column_is_recognised(self, tmp_path):
        """`__index_level_0__` is what pandas names an unnamed index in
        Parquet, and a dataset scanner does not apply the metadata that would
        restore it -- so a file written from an indexed tape arrives with that
        column and a RangeIndex."""
        stamps = pd.date_range("2024-03-05 14:30", periods=4, freq="1s", tz="UTC")
        indexed = pd.DataFrame(
            {"timestamp": stamps, "price": [1.0, 2, 3, 4], "size": [1] * 4}
        ).set_index(pd.DatetimeIndex(stamps))
        path = tmp_path / "vendor" / "unnamed.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        indexed.drop(columns=["timestamp"]).to_parquet(path)
        ref, _ = handoff.publish_external(
            str(path), kind="tick_tape", run_id="r1", name="unnamed"
        )
        frame = resolve_frame(ref, expect="tick_tape", who="t")
        assert isinstance(frame.index, pd.DatetimeIndex)

    def test_an_index_that_is_not_an_instant_is_refused_on_resolve(self, tmp_path):
        """Where the alias stops.

        `__index_level_0__` is accepted at registration by NAME, because
        `check_schema` is handed column names without their types. An unnamed
        index holding strings passes that and is refused here, by a message
        listing what was found rather than by a pandas error further in.
        """
        frame = pd.DataFrame(
            {"price": [1.0, 2.0], "size": [1, 2]}, index=["first", "second"]
        )
        path = tmp_path / "vendor" / "stringy.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path)
        ref, _ = handoff.publish_external(
            str(path), kind="tick_tape", run_id="r1", name="stringy"
        )
        with pytest.raises(ValidationError) as caught:
            resolve_frame(ref, expect="tick_tape", who="t")
        assert "no usable timestamp" in str(caught.value)
        assert "price" in str(caught.value), "the columns it did find"

    def test_a_kind_that_indexes_by_position_is_not_touched(self):
        """Not every kind wants this, and the difference is not cosmetic.

        A book update and an order event are not unique in time, so
        `order_book_panel` and `order_event_panel` carry `timestamp` as a
        column and index by position on purpose -- reading their index would
        report row 0 and row n-1 as the window.
        """
        assert "order_book_panel" not in handoff.TIME_INDEXED_KINDS
        assert "order_event_panel" not in handoff.TIME_INDEXED_KINDS
        assert handoff.TIME_INDEXED_KINDS == {"tick_tape", "quote_panel"}

    def test_a_stampless_tape_cannot_be_registered_at_all(self, tmp_path):
        """The earliest place to say so.

        A tape with no stamp used to register cleanly and fail inside the
        estimator -- exactly the bug `KIND_COLUMNS` records having fixed once
        for `order_event_panel`, whose comment says `price` was missing "so a
        panel could satisfy REGISTRATION and then fail inside
        order_event_metrics".
        """
        path = tmp_path / "vendor" / "stampless.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"price": [1.0, 2.0], "size": [1, 2]}).to_parquet(
            path, index=False
        )
        with pytest.raises(ValidationError) as caught:
            handoff.publish_external(
                str(path), kind="tick_tape", run_id="r1", name="stampless"
            )
        assert "timestamp" in str(caught.value)


class TestTheTwoStoragesGiveTheSameAnswer:
    """The invariant `EXTERNAL_KINDS` states, measured through the tools.

    "One kind, two storages, rather than an `external_tick_tape` that would
    double the taxonomy and let a consumer accept one and refuse the other."
    Four tools accepted one and refused the other, and the unit tests above
    would pass with the tools still broken -- so this runs the tools.
    """

    @staticmethod
    def _both(tmp_path):
        from standard_quant_tools.agent.runtimes.microstructure import (
            estimator_tools,
            series_tools,
        )

        n = 2400
        stamps = pd.date_range(
            "2024-03-05 14:30:00", periods=n, freq="1s", tz="UTC", name="timestamp"
        )
        mid, prices, bids, asks = 100.0, [], [], []
        for i in range(n):
            mid += 0.01 if i % 3 else -0.01
            bids.append(round(mid - 0.01, 4))
            asks.append(round(mid + 0.01, 4))
            prices.append(round(mid + (0.01 if i % 2 else -0.01), 4))
        trades = pd.DataFrame(
            {"timestamp": stamps, "price": prices, "size": [100] * n}
        )
        quotes = pd.DataFrame(
            {"timestamp": stamps, "bid_price": bids, "ask_price": asks}
        )

        vendor = tmp_path / "vendor"
        vendor.mkdir(parents=True, exist_ok=True)
        trades.to_parquet(vendor / "t.parquet", index=False)
        quotes.to_parquet(vendor / "q.parquet", index=False)
        ext_t, _ = handoff.publish_external(
            str(vendor / "t.parquet"), kind="tick_tape", run_id="e", name="t"
        )
        ext_q, _ = handoff.publish_external(
            str(vendor / "q.parquet"), kind="quote_panel", run_id="e", name="q"
        )
        pub_t = handoff.publish(trades, kind="tick_tape", run_id="p", name="t")
        pub_q = handoff.publish(quotes, kind="quote_panel", run_id="p", name="q")

        answers = []
        for run, tref, qref in (("e", ext_t, ext_q), ("p", pub_t, pub_q)):
            signed = series_tools.classify_trade_direction(
                series_tools.ClassifyTradesInput(
                    tick_tape_ref=tref, quote_panel_ref=qref, run_id=run, name="s"
                )
            )
            kyle = estimator_tools.estimate_kyle_lambda(
                estimator_tools.KyleLambdaInput(trades_ref=tref, quotes_ref=qref)
            )
            answers.append((
                signed.method,
                signed.n_trades,
                signed.n_buys,
                kyle.n_observations,
                kyle.kyle_lambda,
            ))
        return answers

    def test_a_registered_tape_and_a_published_one_agree_exactly(self, tmp_path):
        registered, published = self._both(tmp_path)
        assert registered == published, (
            "the same content addressed two ways must give one answer, not "
            "two close ones"
        )

    def test_and_it_is_lee_ready_rather_than_a_fallback(self, tmp_path):
        """The positive control on the agreement above.

        Two refusals also agree, and so do two tick-rule fallbacks. The test
        is only worth anything if both sides actually used the quotes.
        """
        registered, _ = self._both(tmp_path)
        assert registered[0] == "lee_ready"
        assert registered[1] == 2400 and registered[3] > 0
