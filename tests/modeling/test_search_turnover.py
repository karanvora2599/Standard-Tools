"""
The inner economic objective: rank IC net of the turnover it costs.

Selecting on IC alone can pick the candidate whose edge is traded away
between rebalances. The score is the mean per-date rank IC minus a
penalty times the candidate's rank turnover. Planted: two candidates,
one with the higher IC and a reshuffled ordering every date, one a
little lower and perfectly stable. At penalty zero the first wins; at a
penalty that prices the reshuffling, the second does.
"""

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError as PydanticValidationError

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.specs import (
    EstimatorSpec,
    ModelSpec,
    SearchSpec,
    ValidationSpec,
)
from standard_quant_tools.modeling.validation.search import (
    _score,
    rank_turnover,
    search_best_params,
)


def _frame(n_entities=20, n_dates=60, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=n_entities)
    dates = np.repeat(pd.bdate_range("2022-01-03", periods=n_dates), n_entities)
    entities = np.tile([f"E{i:02d}" for i in range(n_entities)], n_dates)
    noise = rng.normal(size=n_entities * n_dates)
    target = np.tile(base, n_dates) + noise
    return pd.DataFrame(
        {
            "date": dates,
            "entity": entities,
            "target": target,
            "stable": np.tile(base, n_dates),
        }
    )


class TestRankTurnover:
    def test_it_is_zero_for_a_fixed_ordering_and_half_for_a_reversal(self):
        dates = np.repeat(pd.to_datetime(["2022-01-03", "2022-01-04"]), 4)
        entities = np.tile(["A", "B", "C", "D"], 2)
        same = np.array([1, 2, 3, 4, 1, 2, 3, 4], dtype=float)
        assert rank_turnover(same, dates, entities) == 0.0
        reversed_ = np.array([1, 2, 3, 4, 4, 3, 2, 1], dtype=float)
        # Percentile ranks .25 .5 .75 1 become 1 .75 .5 .25: mean |change| = 0.5
        assert rank_turnover(reversed_, dates, entities) == pytest.approx(0.5)
        # An entity missing on one date contributes nothing to that pair.
        partial = np.array([1, 2, 3, 3, 2, 1], dtype=float)
        partial_dates = np.repeat(pd.to_datetime(["2022-01-03", "2022-01-04"]), 3)
        partial_entities = np.array(["A", "B", "C", "A", "B", "D"])
        turnover = rank_turnover(partial, partial_dates, partial_entities)
        assert 0.0 < turnover < 1.0
        assert rank_turnover(same[:4], dates[:4], entities[:4]) == 0.0  # one date

    def test_the_net_score_is_ic_minus_the_penalty_times_turnover(self):
        frame = _frame()
        y = frame["target"].to_numpy()
        predictions = frame["stable"].to_numpy() + 0.1 * np.arange(len(frame)) % 7
        dates = frame["date"].to_numpy()
        entities = frame["entity"].to_numpy()
        plain = _score("regression", "cs_rank_ic", y, predictions, None, dates)
        net = _score(
            "regression",
            "cs_rank_ic_net_of_turnover",
            y,
            predictions,
            None,
            dates,
            entities=entities,
            turnover_penalty=0.7,
        )
        assert net == pytest.approx(
            plain - 0.7 * rank_turnover(predictions, dates, entities)
        )
        with pytest.raises(ValidationError, match="entities"):
            _score(
                "regression", "cs_rank_ic_net_of_turnover", y, predictions, None, dates
            )


class TestTheSpec:
    def test_the_penalty_and_the_scoring_go_together(self):
        with pytest.raises(PydanticValidationError, match="turnover_penalty > 0"):
            SearchSpec(
                param_grid={"alpha": [1.0]}, scoring="cs_rank_ic_net_of_turnover"
            )
        with pytest.raises(PydanticValidationError, match="read by scoring"):
            SearchSpec(
                param_grid={"alpha": [1.0]}, scoring="cs_rank_ic", turnover_penalty=0.5
            )
        spec = SearchSpec(
            param_grid={"alpha": [1.0]},
            scoring="cs_rank_ic_net_of_turnover",
            turnover_penalty=0.5,
        )
        assert spec.turnover_penalty == 0.5
        with pytest.raises(PydanticValidationError, match="concordance"):
            ModelSpec(
                task="survival",
                estimator=EstimatorSpec(type="cox_ph", params={}),
                validation=ValidationSpec(train_window=100, test_window=20),
                search=spec,
            )


class TestTheSelection:
    def _search(self, penalty: float):
        frame = _frame()
        scoring = "cs_rank_ic_net_of_turnover" if penalty > 0 else "cs_rank_ic"
        spec = SearchSpec(
            param_grid={"kind": ["exact", "stable"]},
            scoring=scoring,
            turnover_penalty=penalty,
            inner_splits=2,
        )

        def fit_predict(params, inner_train, inner_test, fold_index):
            # "exact" predicts the noisy target itself: IC of one, an ordering
            # reshuffled every date. "stable" predicts the persistent part:
            # a lower IC and no turnover at all.
            column = "target" if params["kind"] == "exact" else "stable"
            return inner_test[column].to_numpy(dtype=float), None

        return search_best_params(
            task="regression",
            search_spec=spec,
            base_params={},
            train_frame=frame,
            feature_ids=["stable"],
            random_seed=0,
            fit_predict=fit_predict,
        )

    def test_the_penalty_changes_which_candidate_wins(self):
        best_plain, report_plain = self._search(0.0)
        assert best_plain == {"kind": "exact"}
        assert report_plain["turnover_penalty"] == 0.0
        best_net, report_net = self._search(2.0)
        assert best_net == {"kind": "stable"}
        assert report_net["turnover_penalty"] == 2.0
        scores = {c["params"]["kind"]: c["score"] for c in report_net["candidates"]}
        assert scores["stable"] > scores["exact"]
