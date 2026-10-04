"""
The `winsorize` and `zscore` registry steps refuse an infinite value the
way the default fit does (see the CHANGELOG entry of 2026-10-04).

The default pipeline's fit refuses +/-inf in the training rows: a winsorize
bound read through an infinity is infinite or NaN, and the clipped mean and
scale with it. The same two steps named in a non-default pipeline -- other
quantiles, or either step alone -- fitted the infinity and passed it on:
`zscore` turned it into an infinite feature value and `winsorize` clipped
it to a bound as though it were an ordinary extreme. They now refuse it at
fit, with the default fit's own message, and at transform, worded the same
way for the rows being transformed. The default pipeline is unchanged.
"""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.features.transforms import (
    apply_preprocessing,
    fit_preprocessing,
)
from standard_quant_tools.modeling.preprocessing import (
    FoldContext,
    apply_pipeline,
    build_step,
    fit_and_apply_pipeline,
    fit_pipeline,
)

CTX = FoldContext(dates=np.array([]), entities=None)
DEFAULT = [("winsorize", {"lower": 0.01, "upper": 0.99}), ("zscore", {})]


def _frame(n=200, seed=0, inf_at=()):
    """Three normal columns; `inf_at` plants (row, column, sign) infinities."""
    rng = np.random.default_rng(seed)
    values = rng.normal(0, 1, (n, 3))
    for row, column, sign in inf_at:
        values[row, column] = sign * np.inf
    return pd.DataFrame(values, columns=["a", "b", "c"])


def _message(call):
    with pytest.raises(ValidationError) as caught:
        call()
    return str(caught.value)


@pytest.mark.parametrize(
    "step_type, params",
    [("winsorize", {"lower": 0.05, "upper": 0.95}), ("zscore", {})],
)
class TestTheStepsRefuseAnInfinity:
    def test_at_fit_with_the_default_fits_own_words(self, step_type, params):
        """Everything after the function name is the default fit's message,
        word for word: the columns, their counts, and the remedy."""
        train = _frame(inf_at=[(7, 0, 1), (11, 1, -1), (12, 1, 1)])
        step = build_step(step_type, params)
        refused = _message(lambda: step.fit(train, CTX))
        default = _message(lambda: fit_preprocessing(train))
        assert refused.startswith(f"{step_type}: the training rows hold infinite")
        assert "'a' (1), 'b' (2)" in refused
        assert refused.split(": ", 1)[1] == default.split(": ", 1)[1]

    def test_at_transform_in_the_rows_being_transformed(self, step_type, params):
        """Fitted on finite rows, the state is applied to rows holding an
        infinity: refused, naming the column, rather than passed on as an
        infinite value or clipped to a bound."""
        step = build_step(step_type, params)
        state = step.fit(_frame(), CTX)
        test = _frame(n=20, seed=1, inf_at=[(3, 2, -1)])
        refused = _message(lambda: step.transform(test, state, CTX))
        assert refused.startswith(
            f"{step_type}: the rows to transform hold infinite values in "
            "column(s) 'c' (1)."
        )
        assert "Mark the value missing with NaN" in refused

    def test_a_missing_value_is_still_skipped_and_kept(self, step_type, params):
        """NaN is not an infinity: the fit skips it and the transform keeps
        it, as before."""
        train = _frame()
        train.iloc[5, 0] = np.nan
        step = build_step(step_type, params)
        state = step.fit(train, CTX)
        out = step.transform(train, state, CTX)
        assert np.isnan(out.iloc[5, 0])
        assert np.isfinite(out.drop(index=5).to_numpy()).all()

    def test_a_pipeline_naming_the_step_refuses_on_either_side(self, step_type, params):
        """Through the pipeline the engine's folds call: an infinity among
        the training rows is refused at fit, one among the test rows at
        transform."""
        steps = [(step_type, params)]
        bad = _frame(inf_at=[(0, 0, 1)])
        with pytest.raises(ValidationError, match="the training rows hold infinite"):
            fit_and_apply_pipeline(steps, bad, _frame(seed=1), CTX, CTX)
        with pytest.raises(ValidationError, match="the rows to transform hold"):
            fit_and_apply_pipeline(steps, _frame(seed=1), bad, CTX, CTX)


class TestTheDefaultPipelineIsUnchanged:
    def test_it_refuses_an_infinity_in_the_training_rows_as_before(self):
        """The default pair runs the fused fit, whose refusal names its own
        function, not a step."""
        bad = _frame(inf_at=[(7, 0, 1)])
        with pytest.raises(ValidationError, match=r"^fit_preprocessing: the training"):
            fit_pipeline(DEFAULT, bad, CTX)
        with pytest.raises(
            ValidationError, match=r"^fit_and_apply_preprocessing: the training"
        ):
            fit_and_apply_pipeline(DEFAULT, bad, _frame(seed=1), CTX, CTX)

    def test_it_clips_an_infinity_in_the_rows_it_applies_to_as_before(self):
        """The fused apply clips a test-row infinity to the fitted bound and
        z-scores it; the same numbers `apply_preprocessing` gives, to the
        bit, on both the separate and the fused call."""
        train = _frame()
        test = _frame(n=20, seed=1, inf_at=[(3, 2, -1), (4, 0, 1)])
        state, _ = fit_pipeline(DEFAULT, train, CTX)
        expected = apply_preprocessing(test, fit_preprocessing(train))
        applied = apply_pipeline(state, test, CTX)
        _, _, fused = fit_and_apply_pipeline(DEFAULT, train, test, CTX, CTX)
        assert np.isfinite(applied.to_numpy()).all()
        for out in (applied, fused):
            assert out.to_numpy().tobytes() == expected.to_numpy().tobytes()
