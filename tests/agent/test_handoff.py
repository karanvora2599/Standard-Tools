"""
The handoff interconnect.

Three things make this usable as a general mechanism rather than a shared
folder, and each has a test class here:

  TYPE. A reference carries a content kind, checked on resolve. That is
  what turns a wrong handoff between two tool calls from plausible garbage
  into an error naming both kinds — which is the difference between an
  agent recovering and an agent confidently reporting a number computed
  from a trade log it thought was an equity curve.

  SCALE. Many agents publish concurrently. The tests below pin the two
  properties that fail quietly at that scale: publishing different names
  under one run_id must not lose either, and publishing over an existing
  reference must be refused rather than silently clobbering every holder
  of it.

  GENERALITY. Conversion between kinds is what replaces a bridge tool per
  producer/consumer pair. The last class walks the whole path — publish
  predictions, convert, consume — with no code that knows about both ends.
"""

import json

import pandas as pd
import pytest

from standard_quant_tools.agent.runtimes import handoff, resolve
from standard_quant_tools.error import ValidationError


@pytest.fixture(autouse=True)
def runs_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("SQT_AUDIT_DIR", str(tmp_path / "audit"))
    return tmp_path


@pytest.fixture
def panel():
    return {
        "AAA": {"2024-01-02": 1.0, "2024-01-03": -1.0},
        "BBB": {"2024-01-02": 0.0, "2024-01-03": 1.0},
    }


@pytest.fixture
def predictions():
    dates = pd.bdate_range("2024-01-02", periods=6)
    return pd.DataFrame(
        [
            {"date": d, "entity": entity, "prediction": value}
            for d in dates
            for entity, value in (("AAA", 0.03), ("BBB", -0.02))
        ]
    )


class TestTyping:
    def test_a_reference_round_trips_the_exact_shape(self, panel):
        ref = handoff.publish(panel, "signal_panel", "r1", "sig")
        assert handoff.resolve(ref, expect="signal_panel") == panel

    def test_a_wrong_kind_fails_naming_both(self, panel):
        """The property the whole interconnect rests on. Without it a
        mis-wired handoff surfaces as a missing column somewhere deep in
        pandas, which an agent cannot act on."""
        ref = handoff.publish(panel, "signal_panel", "r1", "sig")
        with pytest.raises(ValidationError) as exc:
            handoff.resolve(ref, expect="equity_curve")
        message = str(exc.value)
        assert "signal_panel" in message and "equity_curve" in message

    def test_an_unknown_kind_is_refused_at_publish(self, panel):
        with pytest.raises(ValidationError):
            handoff.publish(panel, "vibes", "r1", "sig")

    def test_a_malformed_reference_explains_the_shape(self):
        with pytest.raises(ValidationError) as exc:
            handoff.parse("not-a-reference")
        assert "sqt://<kind>/<run_id>/<name>" in str(exc.value)

    def test_a_raw_path_cannot_be_type_checked(self, panel):
        """Older tools return bare artifact paths. Those still load, but
        silently skipping the check for them would make the guarantee
        conditional on which tool happened to produce the value."""
        from standard_quant_tools.backtest.artifacts import save_artifact

        uri = save_artifact(pd.Series([1.0, 2.0], name="equity"), "r1", "curve")
        with pytest.raises(ValidationError) as exc:
            handoff.resolve(uri, expect="equity_curve")
        assert "carries no kind" in str(exc.value)

    def test_an_equity_curve_resolves_as_a_series(self):
        curve = pd.Series(
            [100.0, 101.0, 99.0],
            index=pd.bdate_range("2024-01-02", periods=3),
            name="equity",
        )
        ref = handoff.publish(curve, "equity_curve", "r1", "curve")
        assert isinstance(handoff.resolve(ref, expect="equity_curve"), pd.Series)


class TestConcurrentPublishers:
    def test_two_names_under_one_run_id_both_survive(self, panel):
        """A shared catalogue would race here and lose one, leaving a live
        reference that resolves to data of unknown kind."""
        first = handoff.publish(panel, "signal_panel", "shared", "a", producer="one")
        second = handoff.publish(panel, "score_panel", "shared", "b", producer="two")
        assert handoff.describe(first)["producer"] == "one"
        assert handoff.describe(second)["producer"] == "two"
        assert handoff.describe(first)["kind"] == "signal_panel"
        assert handoff.describe(second)["kind"] == "score_panel"

    def test_republishing_over_a_reference_is_refused(self, panel):
        """A reference promises the same value twice. Clobbering it breaks
        that for every holder, including ones already in an audit log."""
        handoff.publish(panel, "signal_panel", "r1", "sig")
        with pytest.raises(ValidationError) as exc:
            handoff.publish(panel, "signal_panel", "r1", "sig")
        assert "fresh run_id" in str(exc.value)

    def test_overwriting_is_possible_but_deliberate(self, panel):
        handoff.publish(panel, "signal_panel", "r1", "sig")
        handoff.publish(panel, "signal_panel", "r1", "sig", overwrite=True)

    def test_a_content_hash_lets_a_consumer_prove_what_it_read(self, panel):
        ref = handoff.publish(panel, "signal_panel", "r1", "sig")
        first = handoff.describe(ref)["content_hash"]
        assert len(first) == 64
        assert handoff.describe(ref)["content_hash"] == first

    def test_a_dangling_reference_says_so(self):
        with pytest.raises(ValidationError) as exc:
            handoff.resolve("sqt://signal_panel/never/published")
        assert "nothing to resolve" in str(exc.value)


class TestGenerality:
    def test_predictions_reach_a_backtest_with_no_bridge(self, predictions):
        """The path that used to need a bespoke tool knowing both ends:
        modeling publishes, meta converts, backtest consumes. No code in
        this test knows about more than one side at a time."""
        produced = handoff.publish(
            predictions, "predictions", "fleet", "oos", producer="modeling"
        )
        converted = resolve("meta").dispatch(
            "convert_reference",
            {
                "ref": produced,
                "to_kind": "signal_panel",
                "run_id": "fleet",
                "name": "sig",
                "task": "regression",
            },
        )
        panel = handoff.resolve(converted["ref"], expect="signal_panel")
        assert set(panel) == {"AAA", "BBB"}
        assert set(panel["AAA"].values()) == {1.0}
        assert set(panel["BBB"].values()) == {-1.0}

    def test_the_conversion_says_what_it_discarded(self, predictions):
        produced = handoff.publish(predictions, "predictions", "fleet", "oos")
        converted = resolve("meta").dispatch(
            "convert_reference",
            {
                "ref": produced,
                "to_kind": "signal_panel",
                "run_id": "fleet",
                "name": "sig",
                "task": "regression",
            },
        )
        assert any("Magnitude is discarded" in n for n in converted["notes"])

    def test_a_regression_frame_needs_its_task_named(self, predictions):
        """Thresholding a raw forward return as a probability produces a
        nonsensical but valid-looking panel — a wrong answer, not an
        error, which is exactly what must not happen silently."""
        produced = handoff.publish(predictions, "predictions", "fleet", "oos")
        with pytest.raises(Exception) as exc:
            resolve("meta").dispatch(
                "convert_reference",
                {
                    "ref": produced,
                    "to_kind": "signal_panel",
                    "run_id": "fleet",
                    "name": "sig",
                },
            )
        assert "task" in str(exc.value)

    def test_an_unsupported_conversion_refuses_rather_than_guessing(self, panel):
        produced = handoff.publish(panel, "signal_panel", "fleet", "sig")
        with pytest.raises(ValidationError) as exc:
            resolve("meta").dispatch(
                "convert_reference",
                {
                    "ref": produced,
                    "to_kind": "equity_curve",
                    "run_id": "fleet",
                    "name": "curve",
                },
            )
        assert "no best-effort path" in str(exc.value)

    def test_the_kind_map_lists_what_reaches_what(self):
        kinds = resolve("meta").dispatch("list_reference_kinds", {})["kinds"]
        by_kind = {k["kind"]: k for k in kinds}
        assert "signal_panel" in by_kind["predictions"]["convertible_to"]
        assert "weight_panel" in by_kind["score_panel"]["convertible_to"]

    def test_a_published_panel_drives_a_backtest_by_reference(self, panel, monkeypatch):
        """The consumer side: a tool takes the reference instead of the
        panel, so nothing is transcribed through the conversation."""
        import numpy as np

        n = 40
        prices = pd.DataFrame(
            {
                "Open": np.linspace(100, 110, n),
                "High": np.linspace(101, 111, n),
                "Low": np.linspace(99, 109, n),
                "Close": np.linspace(100, 110, n),
                "Volume": np.full(n, 1e6),
            },
            index=pd.bdate_range("2024-01-02", periods=n),
        )

        monkeypatch.setattr(
            "standard_quant_tools.agent.runtimes.backtest.tools."
            "fetch_ohlcv_panel_sync",
            lambda tickers, start, end, interval="1d": {t: prices for t in tickers},
        )
        ref = handoff.publish(panel, "signal_panel", "fleet", "sig")
        result = resolve("backtest").dispatch(
            "run_signal_panel_backtest",
            {
                "tickers": ["AAA", "BBB"],
                "start_date": "2024-01-02",
                "end_date": "2024-02-28",
                "signal_panel_ref": ref,
            },
        )
        # The panel arrived by reference and was backtested: per-ticker
        # results exist for exactly the tickers the reference carried.
        assert set(result["per_ticker"]) == {"AAA", "BBB"}
        assert "sharpe_ratio" in result["portfolio_metrics"]

    def test_supplying_both_a_panel_and_a_reference_is_refused(self, panel):
        with pytest.raises(Exception) as exc:
            resolve("backtest").dispatch(
                "run_signal_panel_backtest",
                {
                    "tickers": ["AAA"],
                    "start_date": "2024-01-02",
                    "end_date": "2024-02-28",
                    "signal_panel": panel,
                    "signal_panel_ref": "sqt://signal_panel/fleet/sig",
                },
            )
        assert "exactly one" in str(exc.value)

    def test_a_reference_is_json_and_therefore_crosses_a_process(self, panel):
        """The reason this is a string and not an object: two agents in the
        orchestrator are two processes."""
        ref = handoff.publish(panel, "signal_panel", "fleet", "sig")
        assert json.loads(json.dumps({"ref": ref}))["ref"] == ref
        assert handoff.resolve(json.loads(json.dumps(ref)), expect="signal_panel")
