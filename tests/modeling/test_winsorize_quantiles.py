"""
The winsorize step's fitted bounds are the per-column quantiles to the bit
(see the CHANGELOG entry of 2026-10-02).

`Winsorize.fit` took two `Series.quantile` calls per column and now takes
one `DataFrame.quantile([lower, upper])` for the whole frame. pandas
computes each column's quantiles with one numpy percentile call, and with
both bounds in the call the column is partitioned at both positions at
once: the same values, but -0.0 and +0.0 (or two NaNs) can land in a
different order, and a bound can then differ in its bits where it is zero
or NaN. Those bounds are recomputed per column. The reference below is the
fit as it was, kept verbatim, and every state is required to be identical
-- the same keys in the same order, the same doubles, the sign of zero
included -- not close.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.preprocessing import FoldContext, build_step

CTX = FoldContext(dates=np.array([]), entities=None)


def _reference_fit(X: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
    """`Winsorize.fit` as it was."""
    lower = float(params["lower"])
    upper = float(params["upper"])
    if not lower < upper:
        raise ValidationError(f"winsorize: lower={lower} must be below upper={upper}.")
    return {
        "lo": {c: float(X[c].quantile(lower)) for c in X.columns},
        "hi": {c: float(X[c].quantile(upper)) for c in X.columns},
    }


def _identical(a, b) -> bool:
    """Equal and of the same type, all the way down; dict keys in the same
    order; floats to the bit."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return list(a) == list(b) and all(_identical(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_identical(x, y) for x, y in zip(a, b))
    if isinstance(a, float):
        if math.isnan(a) or math.isnan(b):
            return math.isnan(a) and math.isnan(b)
        return a == b and math.copysign(1.0, a) == math.copysign(1.0, b)
    return a == b


def _outcome(fn, *args):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            return fn(*args)
        except Exception as error:  # noqa: BLE001 -- the refusal is compared too
            return (type(error).__name__, str(error))


def _assert_same_fit(X, lower=0.01, upper=0.99):
    params = {"lower": lower, "upper": upper}
    step = build_step("winsorize", params)
    expected = _outcome(_reference_fit, X, params)
    actual = _outcome(step.fit, X, CTX)
    assert _identical(actual, expected), (X.shape, lower, upper)
    return actual


def _signed_zero_frame(rng, rows, cols):
    """Columns mostly of -0.0 and +0.0, so a bound lands on a tie between
    them -- the case the per-column recomputation exists for."""
    values = np.where(rng.random((rows, cols)) < 0.5, -0.0, 0.0)
    spread = rng.random((rows, cols)) < 0.1
    values[spread] = rng.normal(0, 1, spread.sum())
    return pd.DataFrame(values, columns=[f"z{i}" for i in range(cols)])


QUANTILE_PAIRS = [
    (0.01, 0.99),
    (0.05, 0.95),
    (0.0, 1.0),
    (0.25, 0.75),
    (0.1, 0.1000001),
    (0.0, 0.3),
]


class TestTheStateIsThePerColumnStateToTheBit:
    @pytest.mark.parametrize("seed", range(4))
    @pytest.mark.parametrize("lower, upper", QUANTILE_PAIRS)
    def test_seeded_frames(self, seed, lower, upper):
        rng = np.random.default_rng(seed)
        for rows, cols in [(1, 1), (2, 3), (7, 5), (60, 12), (1000, 40)]:
            X = pd.DataFrame(
                rng.standard_t(3, (rows, cols)) * rng.uniform(0.01, 5, cols),
                columns=[f"f{i}" for i in range(cols)],
            )
            _assert_same_fit(X, lower, upper)

    @pytest.mark.parametrize("seed", range(8))
    @pytest.mark.parametrize(
        "lower, upper", [(0.1, 0.9), (0.25, 0.75), (0.3, 0.6), (0.15, 0.85)]
    )
    def test_planted_signed_zeros(self, seed, lower, upper):
        """Nine values in ten are a signed zero, so both bounds land on one."""
        rng = np.random.default_rng(seed)
        state = _assert_same_fit(_signed_zero_frame(rng, 200, 20), lower, upper)
        bounds = np.array(list(state["lo"].values()) + list(state["hi"].values()))
        assert np.all(bounds == 0.0) and np.any(np.signbit(bounds))

    @pytest.mark.parametrize("seed", range(6))
    @pytest.mark.parametrize("shape", [(20, 60), (101, 13), (1000, 60)])
    def test_planted_ties_around_zero_with_nan_holes(self, seed, shape):
        """Rounded normals put -0.0 and +0.0 at the middle quantiles, and a
        NaN anywhere sends pandas down its per-column NaN-skipping path. On
        pandas 2.3 and 3.0 the frame call alone disagrees with the per-column
        calls in the sign of a zero bound on several of these frames."""
        rng = np.random.default_rng(seed)
        values = np.round(rng.normal(0, 1, shape))
        values[rng.random(shape) < 0.1] = np.nan
        X = pd.DataFrame(values, columns=[f"c{i}" for i in range(shape[1])])
        _assert_same_fit(X, 0.25, 0.75)
        state = _assert_same_fit(X, 0.4, 0.6)
        assert 0.0 in state["lo"].values() and 0.0 in state["hi"].values()

    @pytest.mark.parametrize("seed", range(4))
    def test_nan_gaps_and_an_all_nan_column(self, seed):
        rng = np.random.default_rng(seed)
        values = rng.normal(0, 1, (300, 8))
        values[rng.random((300, 8)) < 0.1] = np.nan
        values[:, 3] = np.nan
        values[rng.random(300) < 0.5, 5] = -0.0
        X = pd.DataFrame(values, columns=list("abcdefgh"))
        for lower, upper in QUANTILE_PAIRS:
            state = _assert_same_fit(X, lower, upper)
            assert math.isnan(state["lo"]["d"]) and math.isnan(state["hi"]["d"])

    @pytest.mark.parametrize("seed", range(4))
    def test_infinities_are_refused_not_fitted(self, seed):
        """These frames were fitted, to the bit of the per-column fit, until
        the step took the default fit's refusal of an infinity (see the
        CHANGELOG entry of 2026-10-04): a bound read through an infinity is
        infinite or NaN, and the clip it sets is not usable."""
        rng = np.random.default_rng(seed)
        values = rng.normal(0, 1, (300, 8))
        values[rng.random((300, 8)) < 0.1] = np.nan
        values[rng.random((300, 8)) < 0.02] = np.inf
        values[rng.random((300, 8)) < 0.02] = -np.inf
        X = pd.DataFrame(values, columns=list("abcdefgh"))
        for lower, upper in QUANTILE_PAIRS:
            step = build_step("winsorize", {"lower": lower, "upper": upper})
            with pytest.raises(
                ValidationError, match="winsorize: the training rows hold infinite"
            ):
                step.fit(X, CTX)

    def test_constant_columns_and_ties(self):
        rng = np.random.default_rng(3)
        X = pd.DataFrame(
            {
                "flat": np.full(100, 2.5),
                "zero": np.zeros(100),
                "negzero": np.full(100, -0.0),
                "ties": np.round(rng.normal(0, 2, 100)),
            }
        )
        for lower, upper in QUANTILE_PAIRS:
            _assert_same_fit(X, lower, upper)

    def test_frames_built_column_by_column(self):
        """Several blocks rather than one consolidated array."""
        rng = np.random.default_rng(4)
        X = pd.DataFrame(index=range(250))
        for i in range(12):
            X[f"c{i}"] = rng.normal(0, 1, 250)
        X["holes"] = np.where(rng.random(250) < 0.2, np.nan, rng.normal(0, 1, 250))
        _assert_same_fit(X, 0.05, 0.95)

    def test_labels_that_are_not_strings(self):
        rng = np.random.default_rng(5)
        X = pd.DataFrame(
            rng.normal(0, 1, (80, 5)), columns=[10, 2.5, ("a", 1), None, True]
        )
        _assert_same_fit(X, 0.05, 0.95)

    @pytest.mark.parametrize(
        "X",
        [
            pd.DataFrame({"a": np.arange(20)}),
            pd.DataFrame({"a": np.arange(20.0), "b": np.arange(20)}),
            pd.DataFrame({"a": np.arange(20.0), "b": np.arange(20.0) > 9}),
            pd.DataFrame(np.ones((10, 2)), columns=["a", "a"]),
            pd.DataFrame({"a": pd.array(np.arange(10.0), dtype="Float64")}),
            pd.DataFrame({"a": np.arange(10.0, dtype=np.float32)}),
            pd.DataFrame(np.empty((0, 3)), columns=list("abc")),
            pd.DataFrame(index=range(5)),
        ],
        ids=[
            "integers",
            "mixed_float_int",
            "a_bool_column",
            "repeated_labels",
            "nullable_float",
            "float32",
            "no_rows",
            "no_columns",
        ],
    )
    def test_frames_the_one_call_does_not_take(self, X):
        """Every one of these is the per-column calls, refusals included."""
        _assert_same_fit(X, 0.05, 0.95)

    @pytest.mark.parametrize(
        "lower, upper", [(0.9, 0.1), (0.5, 0.5), (-0.1, 0.5), (0.2, 1.5)]
    )
    def test_refusals_are_the_same(self, lower, upper):
        X = pd.DataFrame(np.random.default_rng(6).normal(0, 1, (50, 3)))
        result = _assert_same_fit(X, lower, upper)
        assert isinstance(result, tuple)


class TestKnownAnswers:
    def test_planted_bounds(self):
        """0..100 in steps of 1: the linear 5th and 95th percentiles are 5
        and 95 exactly; a NaN is skipped."""
        X = pd.DataFrame(
            {
                "a": np.arange(101.0),
                "b": np.concatenate([np.arange(101.0), [np.nan]])[:101],
            }
        )
        state = build_step("winsorize", {"lower": 0.05, "upper": 0.95}).fit(X, CTX)
        assert state == {"lo": {"a": 5.0, "b": 5.0}, "hi": {"a": 95.0, "b": 95.0}}

    def test_noise_bounds_are_near_the_normal_quantiles(self):
        """The null case: on standard normal draws the 1% and 99% bounds sit
        near -2.326 and 2.326."""
        X = pd.DataFrame(np.random.default_rng(7).normal(0, 1, (200_000, 3)))
        state = build_step("winsorize", {"lower": 0.01, "upper": 0.99}).fit(X, CTX)
        for c in X.columns:
            assert state["lo"][c] == pytest.approx(-2.326, abs=0.03)
            assert state["hi"][c] == pytest.approx(2.326, abs=0.03)

    def test_a_frame_call_differs_only_where_the_recomputation_looks(self):
        """The premise of the recomputation: wherever the frame call's bound
        disagrees in its bits with the per-column call, it is zero or NaN."""
        for seed in range(8):
            rng = np.random.default_rng(seed)
            X = _signed_zero_frame(rng, 200, 20)
            frame = X.quantile([0.25, 0.75]).to_numpy()
            for k, c in enumerate(X.columns):
                for row, q in enumerate((0.25, 0.75)):
                    single = float(X[c].quantile(q))
                    got = float(frame[row, k])
                    if math.copysign(1.0, single) != math.copysign(1.0, got) or (
                        single != got
                    ):
                        assert got == 0.0 or math.isnan(got)
