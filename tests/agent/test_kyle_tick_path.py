"""
`estimate_kyle_lambda` against a tape, beside the same tape as bars.

WHAT THIS FILE IS ABOUT. Kyle's lambda is a regression of a price change on
a SIGNED volume, and the sign is the whole question. From bars there is no
sign in the data: the only one available is the bar's own return, which is
the variable being explained, so `sign(y) * volume` is regressed on `y`.
The library has always known this and always said so -- `circular=True` and
a warning -- but the tool could be called only that way, because its input
model took `close` and `volume` and nothing else. Every lambda the surface
could produce was the circular one. See the CHANGELOG entry of 2026-09-22.

THE CIRCULAR NUMBER IS NOT BIASED IN A KNOWN DIRECTION, which is what makes
it unusable rather than merely imprecise. The two tapes below are the same
shape and differ only in whether the flow moves the price:

    flow that moves nothing   tape (Lee-Ready) ~0, bars 2.2e-05  -- too HIGH
    flow with planted 2e-04   tape (Lee-Ready) 1.9e-04, bars 6e-05 -- too LOW

and the circular r-squared is about 0.62 in BOTH cases, so it cannot tell
the two tapes apart. A caller reading r_squared to decide whether to trust
the estimate is reading a number that does not move with the truth.

THE TAPES ARE BUILT SO THE ANSWER IS ARITHMETIC. A mid-price random walk,
a side drawn independently of it, and (where impact is planted) a mid that
moves by `lambda * side * size` per print. Trades print one tick either
side of the mid; quotes are that mid plus and minus the same tick, stamped
a second earlier so Lee-Ready matches each print against the quote that
preceded it. Four hundred minutes at roughly ten prints a minute, so a
one-minute bucket holds about ten trades and a five-minute bucket five
times that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pydantic
import pytest

from standard_quant_tools.agent.runtimes import handoff
from standard_quant_tools.agent.tools import dispatch
from standard_quant_tools.error import ValidationError

BASE = pd.Timestamp("2026-03-02 14:30:00")

#: Half the quoted spread, in dollars on a $100 mid: 1 bp each side.
TICK = 0.01

#: The impact planted in the second tape, in dollars per share of signed
#: flow. A 100-share buy lifts the mid by two cents.
PLANTED_LAMBDA = 2e-4


def _tape(planted_lambda: float, seed: int, n: int = 4000):
    """
    A trade tape and the quote panel that was standing when it printed.

    `planted_lambda` is the truth: zero for a tape whose flow moves
    nothing, PLANTED_LAMBDA for one where every print pushes the mid by
    that much per share.
    """
    rng = np.random.default_rng(seed)
    seconds = np.cumsum(rng.choice([4.0, 6.0, 8.0], n))
    side = rng.choice([-1.0, 1.0], n)
    size = rng.integers(50, 200, n).astype(float)
    mid = 100.0 + np.cumsum(planted_lambda * side * size + rng.normal(0, 0.01, n))
    trades = pd.DataFrame(
        {"price": mid + TICK * side, "size": size},
        index=pd.DatetimeIndex(
            [BASE + pd.Timedelta(seconds=float(s)) for s in seconds]
        ),
    )
    quotes = pd.DataFrame(
        {
            "bid_price": mid - TICK,
            "ask_price": mid + TICK,
            "bid_size": 500.0,
            "ask_size": 500.0,
        },
        index=pd.DatetimeIndex(
            [BASE + pd.Timedelta(seconds=float(s) - 1.0) for s in seconds]
        ),
    )
    return trades, quotes


def _one_minute_bars(trades: pd.DataFrame) -> dict:
    """The same tape as the OHLCV an agent would otherwise have: last trade
    price and summed size per minute, as `close` and `volume` lists."""
    close = trades["price"].resample("1min").last().ffill()
    volume = trades["size"].resample("1min").sum()
    frame = pd.DataFrame({"close": close, "volume": volume}).dropna()
    frame = frame[frame["volume"] > 0]
    return {
        "close": [float(v) for v in frame["close"]],
        "volume": [float(v) for v in frame["volume"]],
    }


@pytest.fixture
def runs_dir(tmp_path, monkeypatch):
    """A private artifact store, so a published ref belongs to one test."""
    monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path / "runs"))
    return tmp_path


def _published(trades: pd.DataFrame, quotes: pd.DataFrame, run: str) -> tuple:
    handoff.publish(trades, "tick_tape", run, "tape")
    handoff.publish(quotes, "quote_panel", run, "quotes")
    return f"sqt://tick_tape/{run}/tape", f"sqt://quote_panel/{run}/quotes"


class TestFlowThatMovesNothing:
    """The tape's side is drawn independently of the price path, so the
    true lambda is zero. The signed estimate finds that; the circular one
    reports a confident-looking number with a respectable r-squared."""

    def _both_ways(self):
        trades, quotes = _tape(0.0, seed=11)
        tape_ref, quote_ref = _published(trades, quotes, "flat")
        signed = dispatch(
            "estimate_kyle_lambda",
            {"trades_ref": tape_ref, "quotes_ref": quote_ref, "freq": "1min"},
        )
        bars = dispatch("estimate_kyle_lambda", _one_minute_bars(trades))
        return signed, bars

    def test_the_signed_estimate_is_a_fraction_of_the_circular_one(self, runs_dir):
        signed, bars = self._both_ways()
        assert signed["circular"] is False
        assert bars["circular"] is True
        assert abs(signed["kyle_lambda"]) < 0.5 * abs(bars["kyle_lambda"]), (
            "on a tape whose flow moves nothing, the circular estimate is "
            "the one reporting impact"
        )
        assert bars["kyle_lambda"] > 0, "positive by construction, as advertised"

    def test_the_circular_estimate_is_the_better_fitted_of_the_two(self, runs_dir):
        """The trap. The wrong number also looks like the more trustworthy
        one, because its regressor is built from its own regressand."""
        signed, bars = self._both_ways()
        assert bars["r_squared"] > 0.5
        assert signed["r_squared"] < 0.05

    def test_each_result_says_which_sign_it_used(self, runs_dir):
        signed, bars = self._both_ways()
        assert signed["sign_source"] == "lee_ready"
        assert signed["freq"] == "1min"
        assert bars["sign_source"] == "return_sign"
        assert bars["freq"] is None, "bars are their own buckets; nothing to say"

    def test_the_bars_path_still_warns_that_it_is_circular(self, runs_dir):
        _, bars = self._both_ways()
        assert any("CIRCULAR" in w for w in bars["warnings"])
        assert any("Pass trades" in w for w in bars["warnings"])

    def test_the_tape_path_names_the_rule_and_the_bucket(self, runs_dir):
        signed, _ = self._both_ways()
        assert any(
            "lee_ready" in w and "1min" in w for w in signed["warnings"]
        ), "the caller should not have to infer how the flow was signed"

    def test_without_quotes_the_bid_ask_bounce_comes_back(self, runs_dir):
        """Why `quotes_ref` is worth fetching. With quotes the price that
        moves is the MIDPOINT; without them it is the last trade price,
        which carries a bounce whose sign is the last trade's side -- and
        that side is also in the signed volume. On a tape with no impact at
        all that produces a spurious positive lambda, well above even the
        circular one."""
        trades, quotes = _tape(0.0, seed=11)
        tape_ref, quote_ref = _published(trades, quotes, "flat")
        with_quotes = dispatch(
            "estimate_kyle_lambda", {"trades_ref": tape_ref, "quotes_ref": quote_ref}
        )
        without = dispatch("estimate_kyle_lambda", {"trades_ref": tape_ref})
        assert without["sign_source"] == "tick_rule"
        assert without["circular"] is False
        assert without["freq"] == "1min"
        assert abs(without["kyle_lambda"]) > 5 * abs(with_quotes["kyle_lambda"])


class TestFlowThatMovesThePrice:
    """The same tape shape with a lambda planted in it. The tape recovers
    the planted number; the bars path reports a third of it."""

    def _three_ways(self):
        trades, quotes = _tape(PLANTED_LAMBDA, seed=7)
        tape_ref, quote_ref = _published(trades, quotes, "impact")
        lee_ready = dispatch(
            "estimate_kyle_lambda", {"trades_ref": tape_ref, "quotes_ref": quote_ref}
        )
        tick_rule = dispatch("estimate_kyle_lambda", {"trades_ref": tape_ref})
        bars = dispatch("estimate_kyle_lambda", _one_minute_bars(trades))
        return lee_ready, tick_rule, bars

    def test_the_tape_recovers_the_planted_lambda(self, runs_dir):
        lee_ready, tick_rule, _ = self._three_ways()
        assert lee_ready["kyle_lambda"] == pytest.approx(PLANTED_LAMBDA, rel=0.35)
        assert lee_ready["r_squared"] > 0.3
        # Without quotes the tick rule is signing a tape whose prints DO
        # move the price, which is the case it was designed for, so it
        # lands in the same place by a worse route.
        assert tick_rule["sign_source"] == "tick_rule"
        assert tick_rule["kyle_lambda"] == pytest.approx(PLANTED_LAMBDA, rel=0.35)

    def test_the_bars_path_reports_a_fraction_of_the_planted_lambda(self, runs_dir):
        """The direction reverses. On the flat tape the circular estimate
        was too high; here it is too low, because the bar's single sign
        collapses a minute of opposing prints into one direction while the
        whole minute's volume stays on the other side of the regression."""
        _, _, bars = self._three_ways()
        assert bars["circular"] is True
        assert bars["kyle_lambda"] < 0.5 * PLANTED_LAMBDA

    def test_the_circular_r_squared_cannot_tell_the_two_tapes_apart(self, runs_dir):
        """The reason `circular` had to reach the caller. One tape has a
        planted impact and the other has none; the circular fit is about as
        good on both, so r_squared carries no information about whether the
        lambda beside it means anything."""
        _, _, with_impact = self._three_ways()
        flat_trades, flat_quotes = _tape(0.0, seed=11)
        flat = dispatch("estimate_kyle_lambda", _one_minute_bars(flat_trades))
        assert abs(with_impact["r_squared"] - flat["r_squared"]) < 0.15
        # ... while the signed fits are worlds apart, as they should be.
        tape_ref, quote_ref = _published(flat_trades, flat_quotes, "flat")
        signed_flat = dispatch(
            "estimate_kyle_lambda", {"trades_ref": tape_ref, "quotes_ref": quote_ref}
        )
        assert with_impact["r_squared"] - signed_flat["r_squared"] > 0.3


class TestTheBucketIsTheCallersChoice:
    def test_five_minute_buckets_leave_fewer_observations_than_one_minute(
        self, runs_dir
    ):
        """`freq` is the tape's only tuning knob and it changes the sample,
        not the presentation: a five-minute bucket nets opposing flow away
        inside itself and leaves a fifth of the observations."""
        trades, quotes = _tape(PLANTED_LAMBDA, seed=7)
        tape_ref, quote_ref = _published(trades, quotes, "impact")
        minute = dispatch(
            "estimate_kyle_lambda",
            {"trades_ref": tape_ref, "quotes_ref": quote_ref, "freq": "1min"},
        )
        five = dispatch(
            "estimate_kyle_lambda",
            {"trades_ref": tape_ref, "quotes_ref": quote_ref, "freq": "5min"},
        )
        assert five["freq"] == "5min"
        assert minute["freq"] == "1min"
        assert five["n_observations"] < minute["n_observations"] / 3
        # The slope survives the coarser bucket; it is the same impact.
        assert five["kyle_lambda"] == pytest.approx(PLANTED_LAMBDA, rel=0.35)

    def test_an_unlisted_bucket_is_refused_rather_than_resampled(self, runs_dir):
        trades, quotes = _tape(0.0, seed=11)
        tape_ref, _ = _published(trades, quotes, "flat")
        with pytest.raises(pydantic.ValidationError):
            dispatch("estimate_kyle_lambda", {"trades_ref": tape_ref, "freq": "1h"})


class TestTheRefusals:
    """Every way of asking for something the tool cannot measure, answered
    with the tool that produces what is missing."""

    def test_a_reference_that_points_at_nothing_names_the_fetch_tool(self, runs_dir):
        with pytest.raises(ValidationError) as exc:
            dispatch(
                "estimate_kyle_lambda",
                {"trades_ref": "sqt://tick_tape/never/published"},
            )
        assert "fetch_tick_tape" in str(exc.value)

    def test_a_string_that_is_not_a_reference_names_the_fetch_tool(self, runs_dir):
        with pytest.raises(ValidationError) as exc:
            dispatch("estimate_kyle_lambda", {"trades_ref": "yesterdays_tape.parquet"})
        assert "fetch_tick_tape" in str(exc.value)

    def test_a_quote_panel_handed_in_as_the_tape_is_refused(self, runs_dir):
        trades, quotes = _tape(0.0, seed=11)
        _, quote_ref = _published(trades, quotes, "flat")
        with pytest.raises(ValidationError) as exc:
            dispatch("estimate_kyle_lambda", {"trades_ref": quote_ref})
        assert "tick_tape" in str(exc.value)
        assert "fetch_tick_tape" in str(exc.value)

    def test_a_bad_quote_reference_names_the_quote_fetch_tool(self, runs_dir):
        trades, quotes = _tape(0.0, seed=11)
        tape_ref, _ = _published(trades, quotes, "flat")
        with pytest.raises(ValidationError) as exc:
            dispatch(
                "estimate_kyle_lambda",
                {"trades_ref": tape_ref, "quotes_ref": "sqt://quote_panel/never/book"},
            )
        assert "fetch_quote_panel" in str(exc.value)

    def test_bars_and_a_tape_together_are_refused(self, runs_dir):
        """They are different estimates of different things. Quietly
        preferring one would hide which was measured -- and which one was
        preferred is exactly what the caller needs to know."""
        trades, quotes = _tape(0.0, seed=11)
        tape_ref, _ = _published(trades, quotes, "flat")
        with pytest.raises(pydantic.ValidationError) as exc:
            dispatch(
                "estimate_kyle_lambda",
                dict(_one_minute_bars(trades), trades_ref=tape_ref),
            )
        assert "not both" in str(exc.value)

    def test_neither_bars_nor_a_tape_is_refused_by_name(self):
        with pytest.raises(pydantic.ValidationError) as exc:
            dispatch("estimate_kyle_lambda", {})
        message = str(exc.value)
        assert "trades_ref" in message and "fetch_tick_tape" in message

    def test_quotes_without_a_tape_are_refused(self, runs_dir):
        trades, quotes = _tape(0.0, seed=11)
        _, quote_ref = _published(trades, quotes, "flat")
        with pytest.raises(pydantic.ValidationError) as exc:
            dispatch(
                "estimate_kyle_lambda",
                dict(_one_minute_bars(trades), quotes_ref=quote_ref),
            )
        assert "quotes_ref" in str(exc.value)

    def test_half_the_bars_is_refused_by_the_missing_half(self):
        trades, _ = _tape(0.0, seed=11)
        with pytest.raises(pydantic.ValidationError) as exc:
            dispatch(
                "estimate_kyle_lambda", {"close": _one_minute_bars(trades)["close"]}
            )
        assert "volume" in str(exc.value)


class TestTheSurfaceSaysWhichEstimateItGave:
    """The three fields reached the caller as undeclared extras before, so
    a client reading the schema could not know they existed -- and a client
    that validated against the schema would have dropped them."""

    def _entry(self):
        from standard_quant_tools.mcp.catalog import build_catalog

        return build_catalog()["estimate_kyle_lambda"]

    @pytest.mark.parametrize("field", ["circular", "sign_source", "freq"])
    def test_the_output_schema_declares_it(self, field):
        properties = (self._entry().output_schema or {}).get("properties", {})
        assert field in properties, f"{field} is not in the declared result"
        assert properties[field].get("description"), f"{field} says nothing"

    @pytest.mark.parametrize("field", ["trades_ref", "quotes_ref", "freq"])
    def test_the_input_schema_offers_the_tape(self, field):
        properties = (self._entry().input_schema or {}).get("properties", {})
        assert field in properties

    def test_the_bucket_choices_are_in_the_schema(self):
        properties = (self._entry().input_schema or {}).get("properties", {})
        assert set(properties["freq"]["enum"]) == {
            "1s",
            "5s",
            "10s",
            "30s",
            "1min",
            "5min",
        }

    def test_the_description_says_which_path_to_prefer(self):
        description = self._entry().description
        assert "fetch_tick_tape" in description
        assert "fetch_quote_panel" in description
        assert "circular" in description.lower()
