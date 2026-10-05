"""
`volatility_scale` composes with the sizing method instead of replacing it.

`PredictionTransformSpec.volatility_scale` says what it does: divide each
raw prediction by that entity's trailing realized volatility "before
weighting". `_raw_weights_for_group` instead returned `vol_scaled(...)`,
which does its own cross-sectional gross-leverage normalization and
therefore never ran the chosen method -- a `cross_sectional_zscore` spec
lost its z-scoring, a `cross_sectional_rank` spec lost its ranking, and
nothing reported the substitution. The promise was "and then weight it".

The two membership methods are a documented exception, not an oversight:
`sign` and `top_bottom_quantile` ignore the flag because "scaling a score
cannot change an equal weight". Note the flag would change quantile
MEMBERSHIP if it were applied before selection, which is why it is not.
"""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.backtest.sizing import (
    rank_weighted,
    vol_adjusted_scores,
    vol_scaled,
    zscore_normalized,
)
from standard_quant_tools.modeling.portfolio_eval import (
    transform_predictions_to_weights,
)
from standard_quant_tools.modeling.specs import PredictionTransformSpec

LOOKBACK = 20
N_DATES = 60


@pytest.fixture(scope="module")
def panel():
    """One calm name, one wild one, two ordinary — scored identically, so
    any weight difference between them comes from volatility alone."""
    dates = pd.date_range("2024-01-01", periods=N_DATES, freq="B")
    names = ["CALM", "WILD", "C", "D"]
    rng = np.random.default_rng(9)
    returns = pd.DataFrame(
        {
            "CALM": rng.normal(0, 0.001, N_DATES),
            "WILD": rng.normal(0, 0.05, N_DATES),
            "C": rng.normal(0, 0.01, N_DATES),
            "D": rng.normal(0, 0.01, N_DATES),
        },
        index=dates,
    )
    scores = pd.DataFrame(
        rng.normal(0, 1, (N_DATES, 4)), index=dates, columns=names
    )
    return scores, returns


def _spec(method, **kw):
    return PredictionTransformSpec(
        method=method,
        volatility_scale=True,
        volatility_lookback=LOOKBACK,
        max_position_weight=1.0,
        **kw,
    )


class TestTheMethodStillRuns:
    def test_zscore_is_applied_to_the_adjusted_scores(self, panel):
        """The composition, stated as an equality against the two
        functions run in order."""
        scores, returns = panel
        weights, _ = transform_predictions_to_weights(scores, _spec("cross_sectional_zscore"), returns)
        expected = zscore_normalized(
            vol_adjusted_scores(scores, returns, lookback=LOOKBACK),
            gross_leverage=1.0,
        )
        # The transform renormalizes to its own exposure targets, so compare
        # the SHAPE of the book: the ordering of weights within each date.
        last = weights.iloc[-1]
        assert list(last.sort_values().index) == list(
            expected.iloc[-1].sort_values().index
        )

    def test_rank_is_applied_to_the_adjusted_scores(self, panel):
        scores, returns = panel
        weights, _ = transform_predictions_to_weights(scores, _spec("cross_sectional_rank"), returns)
        expected = rank_weighted(
            vol_adjusted_scores(scores, returns, lookback=LOOKBACK),
            gross_leverage=1.0,
        )
        last = weights.iloc[-1]
        assert list(last.sort_values().index) == list(
            expected.iloc[-1].sort_values().index
        )

    def test_the_composed_book_differs_from_the_replacement_it_used_to_be(
        self, panel
    ):
        """The regression has to be observable, or neither behaviour is
        pinned. `rank_weighted` discretises the scores and the uncentred
        divide-by-volatility does not, so the two cannot agree."""
        scores, returns = panel
        weights, _ = transform_predictions_to_weights(scores, _spec("cross_sectional_rank"), returns)
        replaced = vol_scaled(
            scores, returns_df=returns, lookback=LOOKBACK, gross_leverage=1.0
        )
        assert not np.allclose(
            weights.iloc[-1].to_numpy(),
            replaced.iloc[-1].to_numpy(),
        ), "the composed book matches the old replacement, so nothing changed"

    def test_a_calm_name_still_outweighs_a_wild_one(self, panel):
        """The property the flag exists for, which composing must not
        cost. Scored alike, CALM takes the larger position."""
        scores, returns = panel
        alike = pd.DataFrame(
            1.0, index=scores.index, columns=scores.columns
        )
        for method in ("cross_sectional_zscore", "cross_sectional_rank"):
            weights, _ = transform_predictions_to_weights(
                alike, _spec(method), returns
            )
            last = weights.iloc[-1]
            assert last["CALM"] > last["WILD"], method


class TestTheDocumentedExceptions:
    @pytest.mark.parametrize("method", ["sign", "top_bottom_quantile"])
    def test_a_membership_method_ignores_the_flag(self, panel, method):
        """Documented on the field: scaling a score cannot change an equal
        weight. Asserted because applying the adjustment before selection
        WOULD change quantile membership, which the flag does not claim."""
        scores, returns = panel
        on, _ = transform_predictions_to_weights(scores, _spec(method), returns)
        off, _ = transform_predictions_to_weights(
            scores,
            PredictionTransformSpec(
                method=method, volatility_scale=False, max_position_weight=1.0
            ),
            returns,
        )
        pd.testing.assert_frame_equal(on, off)


class TestTheDivisionOnItsOwn:
    def test_vol_scaled_is_the_division_then_the_normalization(self, panel):
        """`vol_scaled` is public API that backtest callers use directly,
        so extracting the division must not have changed it."""
        scores, returns = panel
        adjusted = vol_adjusted_scores(scores, returns, lookback=LOOKBACK)
        gross = adjusted.abs().sum(axis=1)
        gross_safe = gross.where(gross > 1e-12, other=1.0)
        pd.testing.assert_frame_equal(
            vol_scaled(scores, returns_df=returns, lookback=LOOKBACK, gross_leverage=1.0),
            adjusted.div(gross_safe, axis=0) * 1.0,
        )

    def test_a_name_with_no_volatility_yet_gets_zero_not_a_blowup(self, panel):
        """Before `lookback` observations the rolling std is NaN."""
        scores, returns = panel
        adjusted = vol_adjusted_scores(scores, returns, lookback=LOOKBACK)
        assert (adjusted.iloc[: LOOKBACK - 1] == 0.0).all().all()
        assert np.isfinite(adjusted.to_numpy()).all()

    def test_a_missing_returns_column_is_refused_by_name(self, panel):
        from standard_quant_tools.error import ValidationError

        scores, returns = panel
        with pytest.raises(ValidationError, match="WILD"):
            vol_adjusted_scores(scores, returns.drop(columns=["WILD"]), lookback=LOOKBACK)
