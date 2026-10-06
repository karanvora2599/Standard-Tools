"""
The fused native pair is matched wherever it sits, not only alone.

`_is_default_pooled` compared the WHOLE step list against
`[winsorize(0.01, 0.99), zscore]`, so one extra step before or after
dropped the entire pipeline to pandas. The repo's own table measures what
that costs: the same ablation is 4.37 s on the default pipeline and 43.8 s
with `winsorize + quantile_transform + zscore`. The pair is still there in
that pipeline; it is simply no longer the whole of it.

The state does not change, which is what makes this safe to do at all:
`_fused_state` already wrote the pair as two ORDINARY step entries and
`_fused_stats` reconstructs the kernel's input from them, so a fused span
and two hand-run steps are the same bytes on disk. STATE_VERSION does not
move — bumping it would refuse every `preprocessing_state.json` already
written for a registered model.

The first test is the one that matters: identical output, because a faster
path that answers differently is not the same path.
"""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.modeling.preprocessing import pipeline as pipeline_module
from standard_quant_tools.modeling.preprocessing.base import FoldContext
from standard_quant_tools.modeling.preprocessing.pipeline import (
    _fused_span,
    _normalize,
    apply_pipeline,
    fit_pipeline,
)

WINSOR = ("winsorize", {"lower": 0.01, "upper": 0.99})
ZSCORE = ("zscore", {})
QUANTILE = ("quantile_transform", {})
ROBUST = ("robust_scale", {})


@pytest.fixture
def frame():
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {name: rng.normal(size=400) for name in ("a", "b", "c")}
    )


@pytest.fixture
def ctx(frame):
    dates = np.repeat(
        np.arange(40).astype("datetime64[D]").astype("datetime64[ns]"), 10
    )
    return FoldContext(dates=dates)


class TestWhereThePairIsFound:
    @pytest.mark.parametrize(
        "steps,expected",
        [
            ([WINSOR, ZSCORE], 0),
            ([WINSOR, ZSCORE, QUANTILE], 0),
            ([QUANTILE, WINSOR, ZSCORE], 1),
            ([ROBUST, WINSOR, ZSCORE, QUANTILE], 1),
        ],
    )
    def test_a_contiguous_pair_is_located(self, steps, expected):
        assert _fused_span(_normalize(steps)) == expected

    @pytest.mark.parametrize(
        "steps",
        [
            [WINSOR, QUANTILE, ZSCORE],  # separated — not the fused pair
            [ZSCORE, WINSOR],  # wrong order
            [WINSOR],
            [ZSCORE],
            [("winsorize", {"lower": 0.05, "upper": 0.95}), ZSCORE],
        ],
    )
    def test_anything_else_is_not(self, steps):
        assert _fused_span(_normalize(steps)) is None


class TestTheAnswerIsUnchanged:
    @staticmethod
    def _without_fusion(monkeypatch):
        """Force the step-by-step path, to compare against."""
        monkeypatch.setattr(pipeline_module, "_fused_span", lambda steps: None)

    @pytest.mark.parametrize(
        "steps",
        [
            [WINSOR, ZSCORE, QUANTILE],
            [QUANTILE, WINSOR, ZSCORE],
            [ROBUST, WINSOR, ZSCORE, QUANTILE],
        ],
    )
    def test_fit_gives_the_same_matrix(self, steps, frame, ctx, monkeypatch):
        """A faster path that answers differently is not the same path."""
        _state, fused = fit_pipeline(steps, frame, ctx)
        self._without_fusion(monkeypatch)
        _state2, plain = fit_pipeline(steps, frame, ctx)
        assert np.allclose(fused.to_numpy(), plain.to_numpy(), equal_nan=True)

    @pytest.mark.parametrize(
        "steps", [[WINSOR, ZSCORE, QUANTILE], [QUANTILE, WINSOR, ZSCORE]]
    )
    def test_apply_gives_the_same_matrix(self, steps, frame, ctx, monkeypatch):
        state, _ = fit_pipeline(steps, frame, ctx)
        fused = apply_pipeline(state, frame, ctx)
        self._without_fusion(monkeypatch)
        plain = apply_pipeline(state, frame, ctx)
        assert np.allclose(fused.to_numpy(), plain.to_numpy(), equal_nan=True)

    def test_the_whole_pipeline_case_is_untouched(self, frame, ctx):
        """The exact-match path existed and must behave exactly as before."""
        state, out = fit_pipeline([WINSOR, ZSCORE], frame, ctx)
        assert [e["type"] for e in state["steps"]] == ["winsorize", "zscore"]
        assert np.allclose(
            apply_pipeline(state, frame, ctx).to_numpy(),
            out.to_numpy(),
            equal_nan=True,
        )


class TestTheStateIsTheSameShape:
    def test_a_fused_span_writes_the_ordinary_two_entries(self, frame, ctx):
        """Which is why STATE_VERSION does not move and every registered
        model keeps loading."""
        state, _ = fit_pipeline([QUANTILE, WINSOR, ZSCORE], frame, ctx)
        assert state["version"] == pipeline_module.STATE_VERSION
        assert [e["type"] for e in state["steps"]] == [
            "quantile_transform",
            "winsorize",
            "zscore",
        ]
        winsorize = next(e for e in state["steps"] if e["type"] == "winsorize")
        assert set(winsorize["state"]) == {"lo", "hi"}
        zscore = next(e for e in state["steps"] if e["type"] == "zscore")
        assert set(zscore["state"]) == {"mean", "std"}

    def test_a_state_written_without_fusion_still_applies(
        self, frame, ctx, monkeypatch
    ):
        """The two paths produce interchangeable states in both
        directions, which is what keeps old files readable."""
        monkeypatch.setattr(pipeline_module, "_fused_span", lambda steps: None)
        state, expected = fit_pipeline([QUANTILE, WINSOR, ZSCORE], frame, ctx)
        monkeypatch.undo()
        assert np.allclose(
            apply_pipeline(state, frame, ctx).to_numpy(),
            expected.to_numpy(),
            equal_nan=True,
        )
