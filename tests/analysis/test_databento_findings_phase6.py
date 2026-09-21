"""
Phase 6 of Development/databento_live_fix_plan.md: microstructure.

The live findings (Development/databento_live_findings.md, D7, D8 and
"Also in microstructure") measured each of these on real ticks. The tests
here reproduce each defect's shape offline and pin the fix:

  D7      three functions crashed, or fanned out, on repeated timestamps
  D8      Kyle's lambda regressed on the sign of its own dependent variable
  profile the volume profile buckets the regular session
  vpin    the residue bucket is not a bucket
  roll    the windowed branch judges significance
  cusum   the detector reports the channel's memory and false-alarm rate
  mbo     a snapshot seeds the queue and explains a cancel; a trade is
          one event; the clock excludes the snapshot
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.analysis.liquidity_events import (
    cusum,
    detect_liquidity_events,
)
from standard_quant_tools.analysis.microstructure import (
    effective_spread,
)
from standard_quant_tools.analysis.microstructure import (
    intraday_volume_profile as trade_volume_profile,
)
from standard_quant_tools.analysis.microstructure import (
    microstructure_summary,
    sign_trades,
)
from standard_quant_tools.analysis.microstructure_estimators import (
    estimate_vpin,
    intraday_volume_profile,
    kyle_lambda,
    roll_spread,
)
from standard_quant_tools.analysis.order_events import (
    event_rates,
    order_event_metrics,
    order_lifetimes,
    queue_positions,
)

BASE = pd.Timestamp("2024-03-04 14:30:00")


def _quotes(rows):
    index = [BASE + pd.Timedelta(seconds=s) for s, *_ in rows]
    return pd.DataFrame(
        {
            "bid_price": [r[1] for r in rows],
            "ask_price": [r[2] for r in rows],
            "bid_size": [r[3] for r in rows],
            "ask_size": [r[4] for r in rows],
        },
        index=pd.DatetimeIndex(index),
    )


def _trades(rows):
    index = [BASE + pd.Timedelta(seconds=s) for s, *_ in rows]
    return pd.DataFrame(
        {"price": [r[1] for r in rows], "size": [r[2] for r in rows]},
        index=pd.DatetimeIndex(index),
    )


def _real_tape(n: int = 600, seed: int = 0):
    """A tape where a third of the prints share a timestamp with the one
    before, as a live AAPL minute does."""
    rng = np.random.default_rng(seed)
    seconds = np.cumsum(rng.choice([0.0, 0.0, 1.0], n))
    mid = 100.0 + np.cumsum(rng.normal(0, 0.01, n))
    side = rng.choice([-1.0, 1.0], n)
    trades = _trades(
        [
            (s, round(m + 0.05 * sd, 2), int(v))
            for s, m, sd, v in zip(seconds, mid, side, rng.integers(10, 300, n))
        ]
    )
    quote_seconds = np.arange(-1, seconds[-1] + 2, 1.0)
    quotes = _quotes(
        [(s, 100.0 - 0.05 + 0.0 * s, 100.0 + 0.05, 500, 500) for s in quote_seconds]
    )
    return trades, quotes


# ── D7 ───────────────────────────────────────────────────────────────────


class TestRepeatedTimestampsAreJustRows:
    """32% of prints in a live AAPL minute share a timestamp; every
    label-based alignment either raised or fanned rows out."""

    def test_the_tape_really_repeats(self):
        trades, _ = _real_tape()
        assert trades.index.duplicated().mean() > 0.25

    def test_effective_spread_runs_and_keeps_one_row_per_trade(self):
        trades, quotes = _real_tape()
        result = effective_spread(trades, quotes)
        signed = sign_trades(trades, quotes)
        assert len(result) == len(signed)
        assert len(result) <= len(trades)
        assert result["effective_spread_bps"].notna().all()

    def test_the_summary_runs_with_and_without_quotes(self):
        trades, quotes = _real_tape()
        with_quotes = microstructure_summary(trades, quotes)
        without = microstructure_summary(trades)
        for summary in (with_quotes, without):
            assert 0.0 <= summary["buy_volume_fraction"] <= 1.0
            assert summary["n_signed"] <= len(trades)

    def test_signed_volume_cannot_exceed_what_traded(self):
        trades, quotes = _real_tape()
        report = detect_liquidity_events(
            channels=["signed_volume"],
            trades=trades.reset_index().rename(columns={"index": "timestamp"}),
            quotes=quotes.reset_index().rename(columns={"index": "timestamp"}),
            freq="10s",
        )
        assert report["unavailable"] == []
        from standard_quant_tools.analysis.liquidity_events import _signed_volume

        series = _signed_volume(
            trades.reset_index().rename(columns={"index": "timestamp"}),
            quotes.reset_index().rename(columns={"index": "timestamp"}),
            freq="10s",
        )
        assert series.abs().sum() <= trades["size"].sum() + 1e-9

    def test_one_channels_failure_does_not_kill_the_others(self, monkeypatch):
        import dataclasses

        from standard_quant_tools.analysis import liquidity_events as module

        trades, quotes = _real_tape()

        def broken(*args, **kwargs):
            raise ValueError("cannot reindex on an axis with duplicate labels")

        monkeypatch.setitem(
            module.CHANNELS,
            "signed_volume",
            dataclasses.replace(module.CHANNELS["signed_volume"], compute=broken),
        )
        report = detect_liquidity_events(
            channels=["signed_volume", "trade_intensity"],
            trades=trades.reset_index().rename(columns={"index": "timestamp"}),
            quotes=quotes.reset_index().rename(columns={"index": "timestamp"}),
            freq="10s",
        )
        (failed,) = report["unavailable"]
        assert failed["channel"] == "signed_volume"
        assert "ValueError" in failed["reason"]
        assert report["channels_run"] == ["trade_intensity"]


# ── D8 ───────────────────────────────────────────────────────────────────


def _bars_with_no_impact(n: int = 800, seed: int = 4):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "close": 100 + np.cumsum(rng.normal(0, 0.05, n)),
            "volume": rng.uniform(1e5, 5e5, n),
        }
    )


class TestKyleLambdaKnowsWhenItIsCircular:
    """From bars the flow was signed by the bar's own return, so lambda was
    positive by construction; shuffling the returns left 80% of it."""

    def test_bars_alone_are_declared_circular(self):
        result = kyle_lambda(_bars_with_no_impact())
        assert result["circular"] is True
        assert result["sign_source"] == "return_sign"
        assert any("positive by construction" in w for w in result["warnings"])
        assert any("TICK RULE" in w for w in result["warnings"])
        # And, as the findings measured, the circular estimate is positive
        # on a series with NO impact at all.
        assert result["kyle_lambda"] > 0

    def test_a_trade_tape_with_no_impact_gives_a_lambda_near_zero(self):
        rng = np.random.default_rng(11)
        n = 6000
        seconds = np.cumsum(rng.choice([0.0, 0.5, 1.0], n))
        mid = 100.0 + np.cumsum(rng.normal(0, 0.002, n))
        side = rng.choice([-1.0, 1.0], n)  # flow independent of the price path
        trades = _trades(
            [
                (s, float(m + 0.01 * sd), int(v))
                for s, m, sd, v in zip(seconds, mid, side, rng.integers(10, 200, n))
            ]
        )
        quotes = _quotes(
            [
                (s, float(m) - 0.01, float(m) + 0.01, 500, 500)
                for s, m in zip(seconds - 0.25, mid)
            ]
        )
        genuine = kyle_lambda(trades=trades, quotes=quotes, freq="10s")
        assert genuine["circular"] is False
        assert genuine["sign_source"] == "lee_ready"
        circular = kyle_lambda(
            pd.DataFrame(
                {
                    "close": pd.Series(trades["price"].to_numpy(), index=trades.index)
                    .resample("10s")
                    .last()
                    .ffill(),
                    "volume": pd.Series(trades["size"].to_numpy(), index=trades.index)
                    .resample("10s")
                    .sum(),
                }
            ).dropna()
        )
        # The genuine sign finds (nearly) nothing; the circular one finds impact.
        assert abs(genuine["kyle_lambda"]) < 0.2 * abs(circular["kyle_lambda"])
        assert genuine["r_squared"] < 0.1 < circular["r_squared"]

    def test_a_planted_impact_is_recovered_from_a_tape(self):
        rng = np.random.default_rng(5)
        n = 6000
        lam = 2e-4
        # Strictly increasing stamps: two trades on one timestamp share one
        # quote, and the earlier one's sign is then read against a midpoint
        # that already carries the later one's impact.
        seconds = np.cumsum(rng.choice([0.5, 1.0], n))
        side = rng.choice([-1.0, 1.0], n)
        size = rng.integers(50, 200, n)
        # The price moves WITH the signed flow: impact.
        mid = 100.0 + np.cumsum(lam * side * size + rng.normal(0, 0.003, n))
        trades = _trades(
            [
                (s, float(m + 0.01 * sd), int(v))
                for s, m, sd, v in zip(seconds, mid, side, size)
            ]
        )
        quotes = _quotes(
            [
                (s, float(m) - 0.01, float(m) + 0.01, 500, 500)
                for s, m in zip(seconds - 0.25, mid)
            ]
        )
        result = kyle_lambda(trades=trades, quotes=quotes, freq="5s")
        assert result["kyle_lambda"] == pytest.approx(lam, rel=0.35)
        assert result["r_squared"] > 0.3

    def test_neither_input_is_refused_by_name(self):
        from standard_quant_tools.error import ValidationError

        with pytest.raises(ValidationError, match="ohlcv"):
            kyle_lambda()


# ── the profile ──────────────────────────────────────────────────────────


def _extended_bars(days: int = 3):
    rows = []
    for day in range(days):
        stamps = pd.date_range(
            f"2024-01-0{day + 2} 04:00", f"2024-01-0{day + 2} 19:55", freq="5min"
        )
        for stamp in stamps:
            minute = stamp.hour * 60 + stamp.minute
            if 570 <= minute < 960:
                # Regular session: a U over 09:30-16:00.
                volume = 3.0 - 2.6 * np.sin(np.pi * (minute - 570) / 390)
            else:
                volume = 0.02  # a trickle before the open and after the close
            rows.append((stamp, volume * 1e5))
    index = pd.DatetimeIndex([r[0] for r in rows]).tz_localize("America/New_York")
    return pd.DataFrame({"volume": [r[1] for r in rows]}, index=index)


class TestTheProfileBucketsTheSession:
    """A live feed with extended hours gave open_share 0.00004, close_share
    0.0 and u_shaped False; restricted to the session, 0.234 / 0.153 and
    True."""

    def test_a_timezone_aware_index_is_profiled_over_the_session(self):
        frame = _extended_bars()
        naive = frame.tz_localize(None)
        whole_range = intraday_volume_profile(naive)
        session = intraday_volume_profile(frame)
        assert not whole_range["u_shaped"] and whole_range["open_share"] < 0.01
        assert session["u_shaped"]
        assert session["open_share"] > 0.1 and session["close_share"] > 0.1
        assert 0.0 < session["extended_hours_share"] < 0.05
        assert session["session"] == ["09:30", "16:00"]

    def test_a_utc_index_is_placed_with_index_timezone(self):
        frame = _extended_bars()
        utc_naive = frame.tz_convert("UTC").tz_localize(None)
        result = intraday_volume_profile(utc_naive, index_timezone="UTC")
        assert result["u_shaped"]
        assert result["extended_hours_share"] is not None

    def test_the_trade_profile_does_the_same(self):
        rows = []
        for hour, share in ((8, 1), (10, 30), (13, 10), (15, 40), (17, 1)):
            rows.append(
                (
                    pd.Timestamp(f"2024-01-02 {hour:02d}:00", tz="America/New_York"),
                    share,
                )
            )
        trades = pd.DataFrame(
            {"price": 100.0, "size": [float(r[1] * 100) for r in rows]},
            index=pd.DatetimeIndex([r[0] for r in rows]),
        )
        result = trade_volume_profile(trades, freq="60min")
        assert result["extended_hours_share"] == pytest.approx(2 / 82)
        assert all(
            "08:" not in b["time"] and "17:" not in b["time"] for b in result["buckets"]
        )


# ── vpin, roll ───────────────────────────────────────────────────────────


class TestTheResidueBucketIsGone:
    def test_exactly_the_requested_buckets(self):
        rng = np.random.default_rng(2)
        frame = pd.DataFrame(
            {
                "close": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 500))),
                "volume": rng.uniform(1e5, 9e5, 500),
            }
        )
        result = estimate_vpin(frame, n_buckets=50)
        assert result["n_buckets"] == 50
        assert 0.0 <= result["residual_volume"] < result["bucket_volume"]


class TestTheWindowedRollGuardRuns:
    def test_a_random_walk_is_judged_not_significant(self):
        rng = np.random.default_rng(3)
        prices = pd.Series(100 + np.cumsum(rng.normal(0, 0.05, 2000)))
        result = roll_spread(prices, window=200)
        assert result["significant"] is not None
        assert result["significant"] is False
        assert np.isfinite(result["serial_covariance"])

    def test_a_real_spread_is_judged_significant(self):
        rng = np.random.default_rng(4)
        mid = 100 + np.cumsum(rng.normal(0, 0.01, 4000))
        prices = pd.Series(mid + 0.10 * rng.choice([-1.0, 1.0], 4000))
        result = roll_spread(prices, window=400)
        assert result["significant"] is True


# ── cusum ────────────────────────────────────────────────────────────────


class TestTheDetectorReportsItsMemory:
    """The threshold is calibrated on i.i.d. noise; a spread channel with
    lag-1 autocorrelation +0.67 fired on 43% of quiet real windows."""

    @staticmethod
    def _ar1(phi: float, n: int = 400, seed: int = 0) -> pd.Series:
        rng = np.random.default_rng(seed)
        values = np.empty(n)
        values[0] = rng.normal()
        for t in range(1, n):
            values[t] = phi * values[t - 1] + rng.normal()
        return pd.Series(
            values, index=pd.date_range("2024-01-02 09:30", periods=n, freq="min")
        )

    def test_the_result_carries_the_autocorrelation_and_the_rate(self):
        result = cusum(self._ar1(0.7))
        assert result["lag1_autocorrelation"] > 0.5
        assert result["threshold"] == 9.0
        assert result["false_alarm_rate_at_threshold"] > 0.10
        assert any("false-alarm" in n or "crosses" in n for n in result["notes"])
        iid = cusum(self._ar1(0.0))
        assert iid["false_alarm_rate_at_threshold"] < 0.15

    def test_calibration_takes_the_threshold_from_the_null(self):
        result = cusum(self._ar1(0.7), calibrate_threshold=True)
        assert result["threshold_calibrated"] is True
        assert result["threshold"] > 9.0
        assert result["false_alarm_rate_at_threshold"] == pytest.approx(0.05, abs=0.03)

    def test_calibration_lowers_the_false_alarm_rate_on_quiet_memory(self):
        fired_default = fired_calibrated = 0
        for seed in range(20):
            series = self._ar1(0.7, seed=seed)
            fired_default += int(cusum(series)["triggered"])
            fired_calibrated += int(
                cusum(series, calibrate_threshold=True)["triggered"]
            )
        assert fired_calibrated < fired_default

    def test_the_tool_path_carries_the_switch(self):
        trades, quotes = _real_tape(n=3000)
        report = detect_liquidity_events(
            channels=["trade_intensity"],
            trades=trades.reset_index().rename(columns={"index": "timestamp"}),
            freq="10s",
            calibrate_threshold=True,
        )
        (channel,) = report["results"]
        assert channel["threshold_calibrated"] is True
        assert "lag1_autocorrelation" in channel


# ── mbo ──────────────────────────────────────────────────────────────────

T0 = pd.Timestamp("2024-03-04 14:30:00")


def _events(rows, snapshot_ids=()):
    frame = pd.DataFrame(
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
    frame["flags"] = [32 if oid in snapshot_ids else 0 for oid in frame["order_id"]]
    return frame


class TestASnapshotIsTheBookNotAnEvent:
    def test_snapshot_orders_seed_the_queue_without_counting_as_arrivals(self):
        events = _events(
            [
                (0, 1, "A", "B", 100.0, 500),  # in the snapshot: already resting
                (0, 2, "A", "B", 100.0, 300),  # in the snapshot
                (5, 3, "A", "B", 100.0, 100),  # a real arrival behind 800
            ],
            snapshot_ids={1, 2},
        )
        queue = queue_positions(events)
        assert queue["n_adds"] == 1
        assert queue["n_snapshot_orders"] == 2
        assert queue["mean_queue_ahead"] == 800.0
        # Without the snapshot the same arrival looks like it joined an empty level.
        naive = queue_positions(events.drop(columns=["flags"]).iloc[2:])
        assert naive["mean_queue_ahead"] == 0.0

    def test_a_cancel_of_a_snapshot_order_is_explained_not_censored(self):
        events = _events(
            [
                (0, 1, "A", "B", 100.0, 500),
                (5, 1, "C", "B", 100.0, 500),
                (6, 9, "C", "B", 100.0, 100),  # never seen: genuinely censored
            ],
            snapshot_ids={1},
        )
        lifetimes = order_lifetimes(events)
        assert lifetimes["terminated_from_snapshot"] == 1
        assert lifetimes["terminated_without_an_add"] == 1
        assert lifetimes["cancelled"]["n"] == 0

    def test_the_clock_and_the_counts_exclude_the_snapshot(self):
        rows = [(-3600, oid, "A", "B", 100.0, 10) for oid in range(1, 101)]
        rows += [(i, 200 + i, "A", "B", 100.0, 10) for i in range(11)]
        events = _events(rows, snapshot_ids=set(range(1, 101)))
        rates = event_rates(events)
        assert rates["n_snapshot_events"] == 100
        assert rates["n_events"] == 11
        assert rates["elapsed_seconds"] == pytest.approx(10.0)
        assert rates["events_per_second"] == pytest.approx(1.1)

    def test_a_trade_is_one_event_not_two(self):
        events = _events(
            [
                (0, 1, "A", "B", 100.0, 100),
                (1, 1, "F", "B", 100.0, 100),
                (1, 0, "T", "N", 100.0, 100),
                (2, 2, "A", "B", 100.0, 100),
                (3, 2, "C", "B", 100.0, 100),
            ]
        )
        rates = event_rates(events)
        assert rates["cancel_to_trade"] == pytest.approx(1.0)
        assert rates["cancel_to_add"] == pytest.approx(0.5)

    def test_the_metrics_note_the_snapshot(self):
        events = _events(
            [
                (0, 1, "A", "B", 100.0, 500),
                (5, 2, "A", "B", 100.0, 100),
                (6, 2, "C", "B", 100.0, 100),
            ],
            snapshot_ids={1},
        )
        metrics = order_event_metrics(events)
        assert metrics["rates"]["n_snapshot_events"] == 1
        assert any("snapshot" in w for w in metrics["warnings"])
