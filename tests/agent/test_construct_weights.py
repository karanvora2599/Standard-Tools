"""
A tool that was broken on every one of its methods, committed, with no tests.

`construct_weights_from_scores` built a correct date-by-entity frame of
weights and then handed it to `publish` for a kind whose storage is
`mapping`, which refused anything that was not a dict. All four methods
failed identically, with

    kind 'weight_panel' expects a non-empty {ticker: {date: value}} mapping

which reads as a problem with the data and is a problem with its container:
the mapping is converted straight back into a frame on the next line.

The fix is at the handoff, not at the call site. `publish` takes either
container for a mapping kind now, because the frame IS the stored shape --
and because requiring the conversion meant every producer holding a frame
had to remember it, which is a rule that gets forgotten exactly once per
producer. There were three such call sites and two hand-rolled copies of
`_frame_to_mapping` between them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.agent.runtimes import handoff, resolve
from standard_quant_tools.error import ValidationError

DATES = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2024-01-02", periods=40)]
NAMES = [f"T{i}" for i in range(8)]


@pytest.fixture(autouse=True)
def runs_dir(tmp_path, monkeypatch):
    """The registry outlives a pytest session, so a publishing test owns
    its own runs directory or collides with the last run of itself."""
    monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path / "runs"))


@pytest.fixture
def scores_ref():
    rng = np.random.default_rng(0)
    panel = {t: {d: float(rng.normal()) for d in DATES} for t in NAMES}
    return handoff.publish(panel, "score_panel", "r", "scores", producer="test")


@pytest.fixture
def returns_ref():
    rng = np.random.default_rng(1)
    panel = {t: {d: float(rng.normal(0, 0.01)) for d in DATES} for t in NAMES}
    return handoff.publish(panel, "score_panel", "r", "rets", producer="test")


@pytest.fixture
def portfolio():
    return resolve("portfolio")


def _call(portfolio, scores_ref, **kwargs):
    payload = {
        "scores_ref": scores_ref,
        "run_id": "r",
        "name": kwargs.pop("name", "w"),
    }
    payload.update(kwargs)
    return portfolio.dispatch("construct_weights_from_scores", payload)


def _extra(method, returns_ref):
    if method == "top_bottom":
        return {"n_long": 3, "n_short": 3}
    if method == "vol_scaled":
        return {"returns_ref": returns_ref}
    return {}


METHODS = ("rank", "zscore", "top_bottom", "vol_scaled")


class TestEveryMethodProducesAWeightPanel:
    """The regression. Every one of these raised before."""

    @pytest.mark.parametrize("method", METHODS)
    def test_it_publishes_something_resolvable(
        self, portfolio, scores_ref, returns_ref, method
    ):
        result = _call(
            portfolio,
            scores_ref,
            method=method,
            name=f"w_{method}",
            **_extra(method, returns_ref),
        )
        assert result["ref"].startswith("sqt://weight_panel/")
        panel = handoff.resolve(result["ref"], expect="weight_panel")
        assert set(panel) == set(NAMES)

    @pytest.mark.parametrize("method", METHODS)
    def test_the_dates_come_back_as_the_keys_they_went_in_as(
        self, portfolio, scores_ref, returns_ref, method
    ):
        """A frame published to a mapping kind goes through the mapping on
        the way in, so a datetime index cannot come back as
        '2024-01-02 00:00:00' where the dict path gives '2024-01-02'."""
        result = _call(
            portfolio,
            scores_ref,
            method=method,
            name=f"k_{method}",
            **_extra(method, returns_ref),
        )
        panel = handoff.resolve(result["ref"], expect="weight_panel")
        assert set(panel[NAMES[0]]) <= set(DATES)

    @pytest.mark.parametrize("method", METHODS)
    def test_the_reported_stats_describe_the_panel_it_published(
        self, portfolio, scores_ref, returns_ref, method
    ):
        result = _call(
            portfolio,
            scores_ref,
            method=method,
            name=f"s_{method}",
            **_extra(method, returns_ref),
        )
        panel = handoff.resolve(result["ref"], expect="weight_panel")
        frame = pd.DataFrame(panel)
        last = frame.loc[max(frame.index)].dropna()
        assert result["n_entities"] == len(NAMES)
        assert result["n_long"] == int((last > 0).sum())
        assert result["n_short"] == int((last < 0).sum())
        assert result["gross_leverage"] == pytest.approx(
            float(last.abs().sum()), rel=1e-9
        )

    @pytest.mark.parametrize("method", ["rank", "zscore", "top_bottom"])
    def test_the_requested_leverage_is_what_comes_out(
        self, portfolio, scores_ref, returns_ref, method
    ):
        result = _call(
            portfolio,
            scores_ref,
            method=method,
            gross_leverage=2.5,
            name=f"g_{method}",
            **_extra(method, returns_ref),
        )
        assert result["gross_leverage"] == pytest.approx(2.5, rel=1e-6)

    def test_top_bottom_holds_exactly_what_was_asked_for(self, portfolio, scores_ref):
        result = _call(
            portfolio, scores_ref, method="top_bottom", n_long=2, n_short=5, name="tb"
        )
        assert result["n_long"] == 2
        assert result["n_short"] == 5

    def test_dollar_neutral_removes_the_net(self, portfolio, scores_ref):
        result = _call(
            portfolio, scores_ref, method="zscore", dollar_neutral=True, name="dn"
        )
        assert result["net_exposure"] == pytest.approx(0.0, abs=1e-9)
        assert any("neutralised" in w.lower() for w in result["warnings"])


class TestItStillRefusesWhatItCannotDo:
    def test_top_bottom_without_n_long(self, portfolio, scores_ref):
        with pytest.raises(ValidationError, match="n_long"):
            _call(portfolio, scores_ref, method="top_bottom", name="x")

    def test_vol_scaled_without_returns(self, portfolio, scores_ref):
        with pytest.raises(ValidationError, match="returns_ref"):
            _call(portfolio, scores_ref, method="vol_scaled", name="x")

    def test_an_unknown_method_is_refused_by_the_schema(self, portfolio, scores_ref):
        """It was a bare `str`, so the four valid names lived only in the
        description and a wrong one failed in the body. A Literal puts them
        in the schema an agent reads."""
        with pytest.raises(Exception):
            _call(portfolio, scores_ref, method="Rank", name="x")

    def test_the_schema_values_are_the_ones_the_body_accepts(self):
        from typing import get_args

        from standard_quant_tools.agent.runtimes.portfolio.weight_tools import (
            METHODS as BODY_METHODS,
        )
        from standard_quant_tools.agent.runtimes.portfolio.weight_tools import (
            ConstructWeightsInput,
        )

        declared = get_args(ConstructWeightsInput.model_fields["method"].annotation)
        assert set(declared) == set(BODY_METHODS)

    def test_an_unresolvable_scores_ref(self, portfolio):
        with pytest.raises(ValidationError):
            _call(portfolio, "sqt://score_panel/nope/nothing", name="x")


class TestPublishTakesEitherContainer:
    """The root fix, tested where it lives rather than only through the tool
    that tripped over it."""

    def _mapping(self):
        rng = np.random.default_rng(7)
        return {t: {d: float(rng.normal()) for d in DATES[:5]} for t in NAMES[:3]}

    def test_a_frame_and_its_mapping_publish_to_the_same_value(self):
        mapping = self._mapping()
        frame = pd.DataFrame(mapping)
        from_mapping = handoff.resolve(
            handoff.publish(mapping, "weight_panel", "r", "m"), expect="weight_panel"
        )
        from_frame = handoff.resolve(
            handoff.publish(frame, "weight_panel", "r", "f"), expect="weight_panel"
        )
        assert from_frame == from_mapping

    def test_a_datetime_index_keys_the_way_resolve_has_always_keyed(self):
        """Pinning what is true rather than what would be tidy.

        A frame goes in through the same `_frame_to_mapping` that `resolve`
        uses coming out, so the two entry points agree -- but that keys a
        DatetimeIndex as `str(Timestamp)`, which carries the midnight, and
        does NOT normalize it to the date-only strings the dict path has.
        Left alone deliberately: changing it would rewrite the keys of every
        mapping already published, and every panel here arrives through
        `_mapping_to_frame` with the caller's own strings anyway."""
        frame = pd.DataFrame(self._mapping())
        frame.index = pd.to_datetime(frame.index)
        published = handoff.resolve(
            handoff.publish(frame, "weight_panel", "r", "dt"), expect="weight_panel"
        )
        keys = set(published[NAMES[0]])
        assert keys == {f"{d} 00:00:00" for d in DATES[:5]}
        assert keys.isdisjoint(DATES), "date-only keys would mean silent renaming"

    def test_an_empty_frame_is_still_refused(self):
        with pytest.raises(ValidationError, match="non-empty"):
            handoff.publish(pd.DataFrame(), "weight_panel", "r", "empty")

    def test_something_that_is_neither_is_still_refused(self):
        with pytest.raises(ValidationError, match="non-empty"):
            handoff.publish([1, 2, 3], "weight_panel", "r", "list")

    def test_the_refusal_names_both_accepted_containers(self):
        with pytest.raises(ValidationError) as exc:
            handoff.publish({}, "weight_panel", "r", "blank")
        assert "DataFrame" in str(exc.value)
