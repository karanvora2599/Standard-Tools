"""Databento against the real market, and against a second vendor.

EVERY TEST HERE IS LIVE. That is the whole point, and it is the exact
complement of `test_databento_provider.py`, which is offline by design and
says so: dataset preference, the finalization walk-back and entitlement
memory are logic, and logic is tested with an injected client. What logic
cannot test is whether the bytes coming back are the market. A provider
that parses a fixture perfectly and returns three percent of the volume
passes every offline test ever written.

So what is checked here is agreement with reality, in three ways that do
not depend on Databento being right about itself:

  1. INTERNAL INVARIANTS the market itself guarantees. A low is not above
     an open. A book is not crossed for long. A trade prints inside the
     session it belongs to. These need no second source.

  2. A SECOND VENDOR. Closes and volumes are joined against yfinance, an
     independent consolidated source with different infrastructure and a
     different business. Two vendors agreeing on a price to a fraction of
     a basis point is evidence; one vendor agreeing with itself is not.

  3. CROSS-SCHEMA AGREEMENT. The depth feed and the daily bars are
     different products assembled by different pipelines. Every quote in
     the book has to sit inside that session's own high and low, and every
     trade has to as well. This is what catches a price-scaling error,
     which is the failure mode `data/databento.py` exists to prevent and
     the one an offline fixture can never demonstrate.

COST. Every request here is a few thousand records over one symbol and a
few seconds or a month. Priced through `metadata.get_cost` before this
file was written, a full pass is under a cent. The fixtures are module
scoped so a pass fetches each window once.

ENVIRONMENT. `DatabentoProvider` reads `DATABENTO_DATASET`,
`DATABENTO_DEPTH_DATASET` and `DATABENTO_OHLCV_DATASET` from the
environment, and a sibling project on the same machine sets all three. A
test whose answer depends on whose `.env` was loaded last is not a test,
so the fixture below clears them and each test that cares names its
dataset outright.
"""

from __future__ import annotations

import os
from datetime import date, timedelta

import pandas as pd
import pytest

from standard_quant_tools.analysis import order_book as ob
from standard_quant_tools.data.databento_provider import (
    DATASET_CONSOLIDATED,
    DATASET_DEPTH,
    DatabentoProvider,
)
from standard_quant_tools.error import ValidationError

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("DATABENTO_API_KEY", "").strip(),
        reason="live Databento tests need DATABENTO_API_KEY",
    ),
]

#: A settled regular session, comfortably behind the daily finalization
#: lag and not adjacent to a holiday. Fixed rather than computed: a window
#: that moves makes a failure unreproducible, which is the one thing a
#: live test cannot afford.
SESSION = "2026-09-16"

#: A month of sessions ending well before the coverage edge.
BAR_START, BAR_END = "2026-08-03", "2026-09-18"

#: Ten seconds of the open auction's aftermath, where the book is busy
#: enough that a quiet symbol's empty book cannot pass a test vacuously.
BOOK_START = f"{SESSION}T14:30:00"
BOOK_END = f"{SESSION}T14:30:10"

SYMBOL = "AAPL"

#: Liquid names on different venues and with different share structures,
#: so a conclusion is not a fact about one ticker.
CROSS_SYMBOLS = ("AAPL", "MSFT", "SPY", "TSLA")


# --- fixtures -----------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def _no_inherited_dataset_pins():
    """A sibling project's `.env` must not decide what this suite measures."""
    saved = {
        name: os.environ.pop(name, None)
        for name in (
            "DATABENTO_DATASET",
            "DATABENTO_DEPTH_DATASET",
            "DATABENTO_OHLCV_DATASET",
        )
    }
    yield
    for name, value in saved.items():
        if value is not None:
            os.environ[name] = value


@pytest.fixture(scope="module")
def provider() -> DatabentoProvider:
    return DatabentoProvider()


@pytest.fixture(scope="module")
def bars(provider: DatabentoProvider) -> pd.DataFrame:
    return provider.get_ohlcv(SYMBOL, BAR_START, BAR_END, interval="1d")


@pytest.fixture(scope="module")
def book(provider: DatabentoProvider) -> pd.DataFrame:
    return provider.get_order_book(SYMBOL, BOOK_START, BOOK_END, levels=5, limit=800)


@pytest.fixture(scope="module")
def session_range(provider: DatabentoProvider) -> tuple[float, float]:
    """That session's own high and low, for the cross-schema checks."""
    day = provider.get_ohlcv(SYMBOL, SESSION, SESSION, interval="1d")
    row = day.loc[day.index.normalize() == pd.Timestamp(SESSION, tz="UTC")]
    if row.empty:
        row = day.head(1)
    return float(row["Low"].iloc[0]), float(row["High"].iloc[0])


def _vendor_daily(symbol: str, start: str, end: str) -> pd.DataFrame:
    """The second vendor, over the same window and indexed to compare.

    Two adjustments, both of which silently break the comparison if left
    out. Databento indexes UTC-aware and yfinance indexes exchange-local
    naive, so a join without normalising returns nothing -- and a test
    written on an empty join passes by comparing nothing to nothing. And
    yfinance treats `end` as EXCLUSIVE while Databento treats it as
    inclusive, so the same two dates name windows that differ by a session
    and the last one reads as a session the other vendor never saw.
    """
    import yfinance as yf

    exclusive_end = (pd.Timestamp(end) + pd.Timedelta(days=1)).date().isoformat()
    hist = yf.Ticker(symbol).history(start=start, end=exclusive_end, auto_adjust=False)
    if hist.empty:
        pytest.skip(f"the second vendor returned nothing for {symbol}")
    hist = hist.copy()
    index = pd.to_datetime(hist.index)
    if index.tz is not None:
        index = index.tz_convert("UTC").tz_localize(None)
    hist.index = index.normalize()
    return hist


def _as_naive_days(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    index = pd.to_datetime(out.index)
    if index.tz is not None:
        index = index.tz_convert("UTC").tz_localize(None)
    out.index = index.normalize()
    return out


# --- 1. what the market itself guarantees --------------------------------------


class TestBarsObeyTheMarket:
    """Invariants no vendor is allowed to break, checked on real bars."""

    def test_the_month_is_sessions_not_calendar_days(self, bars):
        assert not bars.empty, "no bars came back for a month of regular sessions"
        assert bars.index.is_monotonic_increasing, "bars are out of order"
        assert bars.index.is_unique, "a session is reported twice"
        assert bars.index.tz is not None, "a naive timestamp cannot be placed in a session"
        weekdays = {ts.weekday() for ts in bars.index}
        assert weekdays <= {0, 1, 2, 3, 4}, f"a weekend session appeared: {sorted(weekdays)}"
        # Roughly 21 sessions a month; the window is about seven weeks.
        assert 25 <= len(bars) <= 40, f"{len(bars)} sessions in {BAR_START}..{BAR_END}"

    def test_every_bar_is_internally_consistent(self, bars):
        """low <= min(open, close) <= max(open, close) <= high, on every row.

        The check that catches a column transposed in normalisation, which
        a fixture built from the same normaliser cannot catch.
        """
        low, high = bars["Low"], bars["High"]
        body_low = bars[["Open", "Close"]].min(axis=1)
        body_high = bars[["Open", "Close"]].max(axis=1)
        assert (high >= low).all(), "a high below its own low"
        bad_low = bars[low > body_low + 1e-9]
        bad_high = bars[high < body_high - 1e-9]
        assert bad_low.empty, f"low above the body on {list(bad_low.index.date)}"
        assert bad_high.empty, f"high below the body on {list(bad_high.index.date)}"

    def test_prices_and_volumes_are_plausible_rather_than_merely_present(self, bars):
        for column in ("Open", "High", "Low", "Close"):
            series = bars[column]
            assert series.notna().all(), f"{column} has gaps"
            assert (series > 0).all(), f"{column} has a non-positive price"
            # A scaling error lands orders of magnitude away, not a tick.
            assert (series < 100_000).all(), f"{column} looks unscaled: max {series.max()}"
        assert (bars["Volume"] > 0).all(), "a regular session with no volume"

    def test_a_day_to_day_move_is_a_market_move_not_a_units_change(self, bars):
        """A scale flipping mid-frame shows up as an impossible return.

        The normaliser decides units from the dtype and cross-checks the
        magnitude; this is that decision checked against thirty real days
        rather than against the one row a magnitude test would read.
        """
        returns = bars["Close"].pct_change().dropna()
        assert not returns.empty
        worst = returns.abs().max()
        assert worst < 0.35, (
            f"a {worst:.1%} single-session move in a large-cap name is a units "
            "change or a split, not a price"
        )


# --- 2. a second vendor --------------------------------------------------------


class TestASecondVendorAgrees:
    """Two vendors, different infrastructure, same market."""

    def test_closes_agree_with_an_independent_consolidated_source(self, bars):
        theirs = _vendor_daily(SYMBOL, BAR_START, BAR_END)
        ours = _as_naive_days(bars)
        joined = ours[["Close"]].join(theirs[["Close"]], how="inner", lsuffix="_db", rsuffix="_yf")
        assert len(joined) >= 20, (
            f"only {len(joined)} sessions joined; the indices are not comparable, "
            "which would make every agreement test below pass vacuously"
        )
        relative = (joined["Close_db"] - joined["Close_yf"]).abs() / joined["Close_yf"]
        # A venue's last print is not the consolidated last print, so they
        # differ by a tick or two rather than by nothing.
        assert relative.median() < 0.005, (
            f"median close disagreement {relative.median():.4%} is too wide to be "
            "the last-print difference between two consolidated tapes"
        )
        assert relative.max() < 0.02, (
            f"worst close disagreement {relative.max():.4%} on "
            f"{relative.idxmax().date()} is a different number, not a different venue"
        )

    def test_the_two_vendors_see_the_same_sessions(self, bars):
        theirs = _vendor_daily(SYMBOL, BAR_START, BAR_END)
        ours = _as_naive_days(bars)
        missing = sorted(set(theirs.index) - set(ours.index))
        extra = sorted(set(ours.index) - set(theirs.index))
        assert not extra, f"Databento reports sessions the other vendor does not: {extra}"
        assert len(missing) <= 1, f"Databento is missing sessions: {missing}"

    def test_the_bars_are_unadjusted_and_say_so(self, provider, bars):
        """`adjusted=False` is a claim about the numbers, so the numbers are
        what checks it: a raw series and an adjusted one cannot both be
        right about a dividend, and the metadata is what tells a caller
        which one they hold."""
        meta = provider.get_metadata(SYMBOL, interval="1d")
        assert meta.adjusted is False, "raw venue prices must never be reported as adjusted"
        assert meta.provider == "databento"
        assert meta.timezone == "UTC"


# --- 3. which dataset is actually the tape -------------------------------------


class TestWhichDatasetIsTheConsolidatedTape:
    """Measured, not assumed.

    The provider prefers one dataset by name and calls it consolidated.
    Whether it *is* consolidated is a question about volume, and volume is
    the one field where being wrong is invisible: the prices stay right, so
    the frame looks correct while every liquidity number built on it is off
    by more than an order of magnitude.
    """

    @staticmethod
    def _volume_share(dataset: str, symbol: str) -> tuple[float, float]:
        """(median volume / consolidated volume, median relative close error)."""
        import databento as db

        client = db.Historical(os.environ["DATABENTO_API_KEY"])
        store = client.timeseries.get_range(
            dataset=dataset, schema="ohlcv-1d", symbols=[symbol],
            start=BAR_START, end=BAR_END, stype_in="raw_symbol",
        )
        frame = store.to_df()
        if frame.empty:
            pytest.skip(f"{dataset} returned nothing for {symbol}")
        frame = _as_naive_days(frame)
        # A feed that publishes per venue emits several rows per session;
        # summing is what makes its volume comparable at all.
        daily = frame.groupby(level=0).agg({"close": "last", "volume": "sum"})
        theirs = _vendor_daily(symbol, BAR_START, BAR_END)
        joined = daily.join(theirs[["Close", "Volume"]], how="inner")
        assert not joined.empty, f"{dataset} and the second vendor share no session"
        share = (joined["volume"].astype(float) / joined["Volume"].astype(float)).median()
        error = ((joined["close"] - joined["Close"]).abs() / joined["Close"]).median()
        return float(share), float(error)

    def test_equs_summary_is_the_consolidated_tape(self):
        """It matches an independent consolidated source share for share."""
        for symbol in CROSS_SYMBOLS:
            share, error = self._volume_share("EQUS.SUMMARY", symbol)
            assert 0.97 <= share <= 1.03, (
                f"EQUS.SUMMARY volume for {symbol} is {share:.4f} of the "
                "consolidated tape; it was the dataset that matched it exactly"
            )
            assert error < 0.001, f"EQUS.SUMMARY close for {symbol} is off by {error:.4%}"

    def test_equs_mini_is_a_sample_and_its_prices_are_still_right(self):
        """The trap, written down.

        EQUS.MINI carries a few percent of the tape's volume while its
        prices are correct to a few basis points. A frame from it looks
        entirely healthy and every volume-weighted number computed on it is
        wrong by more than thirty times.
        """
        for symbol in CROSS_SYMBOLS:
            share, error = self._volume_share("EQUS.MINI", symbol)
            assert share < 0.15, (
                f"EQUS.MINI volume for {symbol} is {share:.4f} of consolidated. "
                "If this now matches the tape the dataset changed, and "
                "DATASET_CONSOLIDATED may finally deserve the name."
            )
            assert error < 0.01, (
                f"EQUS.MINI close for {symbol} is off by {error:.4%}: the prices "
                "are supposed to be the sound part of this feed"
            )

    def test_the_depth_venue_reports_its_own_share_not_the_tape(self):
        """XNAS.ITCH is one venue, and its bars say so."""
        share, error = self._volume_share(DATASET_DEPTH, SYMBOL)
        assert 0.05 <= share <= 0.75, (
            f"{DATASET_DEPTH} volume is {share:.4f} of consolidated, which is "
            "neither a venue share nor the whole tape"
        )
        assert error < 0.01

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "DATASET_CONSOLIDATED is EQUS.MINI and it is first in the bar "
            "preference, so the default bars carry about 3% of the tape's "
            "volume with correct prices. EQUS.SUMMARY is the dataset that "
            "matches consolidated volume exactly. Fixing it is a behaviour "
            "change for every volume number this library produces, so it is "
            "recorded here rather than made silently; remove this xfail with "
            "the fix."
        ),
    )
    def test_the_default_bars_carry_consolidated_volume(self, bars):
        ours = _as_naive_days(bars)
        theirs = _vendor_daily(SYMBOL, BAR_START, BAR_END)
        joined = ours[["Volume"]].join(theirs[["Volume"]], how="inner", lsuffix="_db", rsuffix="_yf")
        share = (joined["Volume_db"].astype(float) / joined["Volume_yf"].astype(float)).median()
        assert 0.9 <= share <= 1.1, (
            f"the provider's default bars carry {share:.4f} of consolidated volume "
            f"(DATASET_CONSOLIDATED is {DATASET_CONSOLIDATED})"
        )


# --- 4. the depth feed, which is the headline claim ----------------------------


class TestTheRealOrderBook:
    """Every depth measure in this library was written against a synthetic
    book because nothing could serve a real one. This is a real one."""

    def test_the_book_arrives_in_the_canonical_shape(self, book):
        assert not book.empty, "no depth came back for ten seconds of a liquid open"
        assert len(book) >= 100, f"only {len(book)} snapshots in ten seconds of AAPL"
        for level in range(5):
            for side in ("bid", "ask"):
                for field in ("price", "size"):
                    column = f"{side}_{field}_{level}"
                    assert column in book.columns, f"{column} missing from the book"
        assert book.index.is_monotonic_increasing, "snapshots are out of order"

    def test_the_levels_are_ordered_the_way_a_book_is(self, book):
        bids = [f"bid_price_{i}" for i in range(5)]
        asks = [f"ask_price_{i}" for i in range(5)]
        descending = (book[bids].diff(axis=1).iloc[:, 1:] <= 1e-9).all(axis=1)
        ascending = (book[asks].diff(axis=1).iloc[:, 1:] >= -1e-9).all(axis=1)
        assert descending.all(), (
            f"{(~descending).sum()} snapshots have a bid level above the one "
            "in front of it, which is not a book"
        )
        assert ascending.all(), (
            f"{(~ascending).sum()} snapshots have an ask level below the one "
            "in front of it, which is not a book"
        )

    def test_the_book_is_not_crossed(self, book):
        """A crossed book is arbitrage. A few locked prints happen; a
        persistently crossed feed is a parsing error."""
        crossed = book["bid_price_0"] > book["ask_price_0"] + 1e-9
        assert crossed.mean() < 0.001, (
            f"{crossed.mean():.2%} of snapshots are crossed, which is a sign "
            "the two sides were read from the wrong columns"
        )

    def test_the_sentinels_were_masked_before_they_were_scaled(self, book):
        """An unmasked UNDEF_PRICE is `int64` max, and scaling it produces a
        number nine billion times the stock rather than a missing level."""
        prices = [c for c in book.columns if c.endswith(tuple(str(i) for i in range(10))) and "price" in c]
        values = book[prices].stack().dropna()
        assert not values.empty
        assert (values > 0).all(), "a non-positive price survived normalisation"
        assert (values < 100_000).all(), (
            f"a price of {values.max():g} is an unmasked sentinel, not a quote"
        )

    def test_sizes_are_counts_not_prices(self, book):
        sizes = book[[f"bid_size_{i}" for i in range(5)] + [f"ask_size_{i}" for i in range(5)]]
        stacked = sizes.stack().dropna()
        assert (stacked >= 0).all(), "a negative resting size"
        assert stacked.max() < 10_000_000, "a size that large is a price in the wrong column"

    def test_every_quote_sits_inside_that_session_s_own_range(self, book, session_range):
        """Cross-schema, and the check that would catch a scaling error.

        The depth feed and the daily bar are different products from
        different pipelines. A quote outside the session's own high and low
        cannot be explained by either being right.
        """
        low, high = session_range
        prices = [c for c in book.columns if "price" in c]
        values = book[prices].stack().dropna()
        outside = values[(values < low - 1e-6) | (values > high + 1e-6)]
        assert outside.empty, (
            f"{len(outside)} quotes fall outside the session's {low}-{high} range; "
            f"worst {outside.min() if outside.min() < low else outside.max()}"
        )


class TestTheDepthAnalyticsOnRealDepth:
    """The measures themselves, over a book the market actually made."""

    def test_book_metrics_describe_a_real_large_cap_touch(self, book):
        metrics = ob.book_metrics(book, levels=5)
        assert metrics["n_snapshots"] == len(book)
        assert metrics["levels_read"] == 5
        assert metrics["n_crossed"] == 0
        spread = metrics["mean_spread_bps"]
        assert 0 < spread < 25, (
            f"a mean touch spread of {spread:.2f} bps in a mega-cap is not a "
            "market; a penny on a $300 stock is about 0.3 bps"
        )
        assert metrics["mean_touch_size"] > 0
        assert -1.0 <= metrics["mean_touch_imbalance"] <= 1.0
        assert -1.0 <= metrics["mean_cumulative_imbalance"] <= 1.0

    def test_the_microprice_sits_between_the_touch_prices(self, book):
        """It weights each side by the opposite side's size, so it can lean
        but it cannot leave the spread."""
        value = ob.microprice(
            book["bid_price_0"], book["bid_size_0"],
            book["ask_price_0"], book["ask_size_0"],
        )
        series = pd.Series(pd.array(value, dtype="float64"), index=book.index).dropna()
        assert len(series) > 0.9 * len(book), "the microprice is undefined too often"
        bid = book["bid_price_0"].reindex(series.index)
        ask = book["ask_price_0"].reindex(series.index)
        assert (series >= bid - 1e-9).all(), "a microprice below the bid"
        assert (series <= ask + 1e-9).all(), "a microprice above the offer"

    def test_the_depth_profile_thickens_away_from_the_touch(self, book):
        """Real books rest more size further out. A profile that does not is
        either a synthetic book or a misread one."""
        profile = ob.depth_profile(book, levels=5)["profile"]
        assert len(profile) == 5
        distances = [row["mean_bid_distance_bps"] for row in profile]
        assert distances == sorted(distances), (
            f"levels are not ordered by distance from the touch: {distances}"
        )
        assert profile[0]["mean_bid_size"] > 0 and profile[-1]["mean_bid_size"] > 0
        near = profile[0]["mean_bid_size"] + profile[0]["mean_ask_size"]
        far = profile[-1]["mean_bid_size"] + profile[-1]["mean_ask_size"]
        assert far > near, f"the touch ({near:.0f}) rests more than the back ({far:.0f})"

    def test_book_dynamics_measure_a_plausible_update_rate(self, book):
        dynamics = ob.book_dynamics(book)
        assert dynamics["n_pairs"] == len(book) - 1
        assert dynamics["elapsed_seconds"] > 0
        rate = dynamics["updates_per_second"]
        assert 1 < rate < 100_000, f"{rate:.1f} book updates a second is not a real feed"
        assert dynamics["mid_changes"] >= 0
        assert dynamics["mid_changes"] <= dynamics["n_pairs"]


# --- 5. trades and quotes ------------------------------------------------------


class TestTradesAndQuotes:
    def test_trades_print_inside_the_session_they_belong_to(self, provider, session_range):
        low, high = session_range
        trades = provider.get_trades(SYMBOL, f"{SESSION}T14:30:00", f"{SESSION}T14:35:00")
        assert not trades.empty, "five minutes after the open with no prints"
        assert (trades["price"] > 0).all()
        outside = trades[(trades["price"] < low - 1e-6) | (trades["price"] > high + 1e-6)]
        assert outside.empty, f"{len(outside)} trades printed outside the session's range"
        assert (trades["size"] > 0).all(), "a trade of no size"
        assert trades.index.is_monotonic_increasing, "trades are out of time order"

    def test_the_quoted_spread_is_the_right_way_round(self, provider):
        quotes = provider.get_quotes(SYMBOL, f"{SESSION}T14:30:00", f"{SESSION}T14:31:00")
        assert not quotes.empty
        both = quotes.dropna(subset=["bid_price", "ask_price"])
        assert len(both) > 0.9 * len(quotes), "top of book is missing too often"
        inverted = both[both["bid_price"] > both["ask_price"] + 1e-9]
        assert len(inverted) / len(both) < 0.001, (
            f"{len(inverted)} of {len(both)} quotes are inverted"
        )

    def test_the_traded_volume_is_a_share_of_that_venue_s_day(self, provider, bars):
        """Five minutes of one venue cannot exceed the whole session.

        Loose on purpose: the point is the direction of the inequality, and
        a tighter bound would fail on a day with an opening auction print.
        """
        trades = provider.get_trades(SYMBOL, f"{SESSION}T14:30:00", f"{SESSION}T14:35:00")
        day = _as_naive_days(bars)
        row = day.loc[day.index == pd.Timestamp(SESSION)]
        if row.empty:
            pytest.skip("the session is not in the bar window")
        traded = float(trades["size"].sum())
        assert 0 < traded, "no size traded in five minutes"
        assert traded <= float(row["Volume"].iloc[0]) * 2, (
            "five minutes of one venue printed more than twice the whole "
            "session's bar volume, so the two are not the same instrument"
        )


# --- 6. the operational claims, against a live edge ----------------------------


class TestTheOperationalClaims:
    """The parts the offline suite tests as logic, confirmed against the
    vendor that motivated them."""

    def test_a_request_ending_today_returns_the_last_settled_session(self, provider):
        """The claim: `end` is anchored to the dataset's own edge rather
        than to wall-clock now, so a weekend or a pre-finalization request
        returns the tail instead of a 422."""
        today = date.today()
        frame = provider.get_ohlcv(SYMBOL, (today - timedelta(days=12)).isoformat(),
                                   today.isoformat(), interval="1d")
        assert not frame.empty, (
            "a request ending today returned nothing; the end is being sent "
            "as wall-clock now rather than as the dataset's edge"
        )
        last = frame.index[-1]
        assert last.date() <= today
        assert (today - last.date()).days <= 6, (
            f"the last bar is {last.date()}, {(today - last.date()).days} days "
            "behind today, which is more than a weekend and a finalization lag"
        )

    def test_a_far_future_window_is_declined_rather_than_invented(self, provider):
        future = date.today() + timedelta(days=365)
        with pytest.raises(Exception) as caught:
            provider.get_ohlcv(
                SYMBOL, future.isoformat(),
                (future + timedelta(days=5)).isoformat(), interval="1d",
            )
        assert caught.value is not None

    def test_the_share_class_mapping_is_the_one_the_feeds_use(self):
        assert DatabentoProvider.to_raw_symbol("AAPL") == "AAPL"
        assert DatabentoProvider.to_raw_symbol("BRK.B") == "BRKB"
        assert DatabentoProvider.to_raw_symbol("brk.b") == "BRKB"
        with pytest.raises(ValidationError):
            DatabentoProvider.to_raw_symbol("NOT A TICKER")

    def test_depth_is_refused_rather_than_served_one_level_deep(self, provider):
        """The base class refuses to substitute top-of-book for depth, and
        so does this provider: a one-level book has zero imbalance by
        construction, which reads as a balanced market."""
        with pytest.raises(ValidationError):
            provider.get_order_book(SYMBOL, BOOK_START, BOOK_END, levels=11)
        with pytest.raises(ValidationError):
            provider.get_order_book(SYMBOL, BOOK_START, BOOK_END, levels=0)

    def test_an_interval_the_vendor_does_not_publish_is_named_not_resampled(self, provider):
        with pytest.raises(ValidationError) as caught:
            provider.get_ohlcv(SYMBOL, BAR_START, BAR_END, interval="1w")
        assert "1d" in str(caught.value)


# --- 7. defects found by the second pass, recorded rather than fixed --------
#
# Each of these is reproduced against the live feed. They are strict xfails so
# the suite stays green while the defect stands and goes RED the moment someone
# fixes it, which is the signal to delete the xfail rather than the test.


class TestTheDailyWindowIsOffByOneSession:
    """The single most serious finding of the second pass.

    `_get_range` (databento_provider.py:347-348) adds a day to an `end` that
    `_to_utc(end_of_day=True)` has already pushed to the next midnight, and
    Databento's day-granular end is exclusive. So a request through date X
    returns X and X+1, and nothing trims it -- the other three providers in
    this library all call `trim_to_inclusive_end`; this one does not.

    It is a lookahead leak, not a row count. A caller who asks for bars
    "as of" a date is handed the next session's close.
    """

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "databento_provider.py:347 adds a day to an end that _to_utc has "
            "already rolled to next-midnight, and get_ohlcv never trims. Asking "
            "for one session returns two, the second being the future. Fix: "
            "round the end up to a whole day instead of adding one, and call "
            "trim_to_inclusive_end as polygon/yfinance/bloomberg do."
        ),
    )
    def test_one_session_asked_for_is_one_session_returned(self, provider):
        one = provider.get_ohlcv(SYMBOL, SESSION, SESSION, interval="1d")
        assert len(one) == 1, (
            f"asked for {SESSION} alone and got {len(one)} bars ending "
            f"{one.index[-1].date()}"
        )

    @pytest.mark.xfail(
        strict=True,
        reason="the same off-by-one, stated as the lookahead it actually is",
    )
    def test_the_last_bar_is_not_the_next_session_s_close(self, provider):
        """2026-09-16 closed at 332.85. 337.00 is the 17th."""
        frame = provider.get_ohlcv(SYMBOL, "2026-09-10", SESSION, interval="1d")
        last = frame.index[-1]
        assert str(last.date()) <= SESSION, (
            f"a window ending {SESSION} returned a bar dated {last.date()} "
            f"whose close is {float(frame['Close'].iloc[-1])}"
        )

    def test_the_intraday_schemas_do_not_share_the_defect(self, provider):
        """Passing, and it localises the bug: only the `ohlcv-1d` branch of
        `_get_range` adds the extra day."""
        for interval in ("1h", "1m"):
            frame = provider.get_ohlcv(SYMBOL, "2026-09-14", SESSION, interval=interval)
            days = {str(ts.date()) for ts in frame.index}
            assert max(days) == SESSION, f"{interval} reached {max(days)}"


class TestTheDefaultFeedsCloseIsNotTheOfficialClose:
    """The first pass reported that EQUS.MINI's prices were correct and only
    its volume was wrong. That was too kind, and this is the correction.

    EQUS.MINI's daily bar spans the UTC day and its `close` is the last print
    in that day -- which on a busy afternoon is an after-hours trade, not the
    official close. The level is usually within a tenth of a percent, so the
    error hides in a price series and surfaces in the returns computed from it.
    """

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "EQUS.MINI's daily close is the last print of the UTC day, which is "
            "frequently an after-hours trade. On 2025-04-02 it reports 208.00 "
            "against the tape's 223.89, a 7.1% error, because the tariff "
            "announcement moved the stock after the bell. Fix is F1's: prefer "
            "EQUS.SUMMARY for daily bars."
        ),
    )
    def test_a_news_afternoon_closes_where_the_tape_closed(self):
        import databento as db

        client = db.Historical(os.environ["DATABENTO_API_KEY"])

        def close_on(dataset: str, day: str) -> float:
            frame = client.timeseries.get_range(
                dataset=dataset, schema="ohlcv-1d", symbols=[SYMBOL],
                start=day, end=(pd.Timestamp(day) + pd.Timedelta(days=1)).date().isoformat(),
                stype_in="raw_symbol",
            ).to_df()
            return float(frame["close"].iloc[0])

        mini = close_on("EQUS.MINI", "2025-04-02")
        tape = close_on("EQUS.SUMMARY", "2025-04-02")
        assert abs(mini - tape) / tape < 0.01, (
            f"EQUS.MINI closed {SYMBOL} at {mini} and the consolidated tape at "
            f"{tape}, a {abs(mini - tape) / tape:.2%} difference"
        )


class TestABacktestCompoundsASplitAsAReturn:
    """`backtest/` never reads the `adjusted` flag the provider sets.

    The provider is honest -- `get_metadata` reports `adjusted=False` and the
    docstring says a split is a real -50% bar. Nothing under `backtest/` looks,
    so a 10-for-1 split is compounded as a -90% session.
    """

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "backtest/engine.py has no corporate-action awareness: grep for "
            "'adjusted' under backtest/ finds only a local variable in sizing.py. "
            "Buy-and-hold on LRCX across its 2024-10-03 ten-for-one split reports "
            "-62.4% against a true +276%. Fix: screen bar returns for |r| > 0.35 "
            "and warn -- the engine already walks every bar for its total-loss "
            "guard, so the pass is free."
        ),
    )
    def test_a_ten_for_one_split_does_not_read_as_a_ninety_percent_loss(self, provider):
        from standard_quant_tools.backtest.engine import run_strategy

        bars = provider.get_ohlcv("LRCX", "2024-09-03", "2026-09-18", interval="1d")
        worst = float(bars["Close"].pct_change().min())
        assert worst > -0.5, f"a {worst:.1%} session is a split, not a return"

        out = run_strategy(
            bars, pd.Series(1.0, index=bars.index),
            commission_pct=0.0, slippage_pct=0.0,
        )
        warned = " ".join(out.get("warnings") or [])
        assert "split" in warned.lower() or out["total_return"] > 0, (
            f"buy-and-hold reported {out['total_return']:.2%} across a split and "
            f"warned only about: {warned[:80]}"
        )
