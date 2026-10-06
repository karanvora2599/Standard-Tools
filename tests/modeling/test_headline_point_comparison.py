"""
A point comparison is not a test.

`_headline_report` runs a t on the per-date series for regression and
ranking. Classification (AUC) and survival (concordance) have no such
series, and that branch read `beats = bool(value > null)` — no sample size,
no standard error, no p-value. An AUC of 0.5000001 "beat its null" exactly
as 0.75 did.

THAT True BECAME LOAD-BEARING, which is why this is a fix and not a note:
Carbon's promotion gate refuses `staging`/`production` on
`beats_null=False` and allows on True, so the weakest possible evidence
opened the strongest door. On this install's own registry,
mdl_f7a88c30e0ca reports auc=0.5049 with beats_null=True while its accuracy
is 0.4929 against a majority-class baseline of 0.6167 — worse than always
predicting the majority class, and holding a True a deployment stage rests
on.

AUC has a standard error (Hanley–McNeil) and both class counts are already
in `oos_metrics`, so this is a real test at no cost. A concordance mean has
none recorded, and inventing one would repeat the original mistake in a new
place — it reports None, "no test was made", which refuses a stage rather
than opening one.
"""

import math

import pytest

from standard_quant_tools.modeling.engine import auc_vs_chance


def _metrics(rate=0.5, rows=420):
    return {"positive_rate": rate, "n_oos_rows": rows}


class TestTheAucTest:
    def test_a_near_chance_auc_is_not_established(self):
        """The live case. 0.5049 on 420 rows used to read as a win."""
        z, p, _pos, _neg = auc_vs_chance(0.5049, _metrics(0.5452, 420))
        assert p > 0.05
        assert abs(z) < 1.0

    def test_the_same_auc_on_far_more_rows_is(self):
        """Which is the whole point of having a sample size in the
        verdict: the effect is identical and the evidence is not."""
        _z, p, _pos, _neg = auc_vs_chance(0.5049, _metrics(0.5452, 100_000))
        assert p < 0.05

    def test_a_strong_auc_is_established_on_few_rows(self):
        z, p, _pos, _neg = auc_vs_chance(0.75, _metrics(0.5, 420))
        assert p < 0.001
        assert z > 5

    def test_an_auc_below_chance_gives_a_negative_z(self):
        z, _p, _pos, _neg = auc_vs_chance(0.40, _metrics(0.5, 420))
        assert z < 0

    def test_the_counts_come_from_the_rate_and_the_rows(self):
        _z, _p, positive, negative = auc_vs_chance(0.6, _metrics(0.25, 400))
        assert positive == pytest.approx(100.0)
        assert negative == pytest.approx(300.0)

    def test_the_p_value_is_two_sided(self):
        z, p, _pos, _neg = auc_vs_chance(0.75, _metrics(0.5, 420))
        assert p == pytest.approx(math.erfc(abs(z) / math.sqrt(2.0)))


class TestItRefusesToInventOne:
    def test_without_the_counts_there_is_no_test(self):
        assert auc_vs_chance(0.6, {}) is None

    def test_a_single_class_is_not_an_auc(self):
        assert auc_vs_chance(0.6, _metrics(1.0, 400)) is None
        assert auc_vs_chance(0.6, _metrics(0.0, 400)) is None

    def test_too_few_rows_for_either_class(self):
        assert auc_vs_chance(0.6, _metrics(0.5, 1)) is None


class TestThroughARun:
    def test_a_classification_run_carries_a_p_value_now(
        self, patched_multi_factory
    ):
        """It carried a bare True before, with nothing behind it."""
        from standard_quant_tools.modeling.dataset.builder import build_dataset
        from standard_quant_tools.modeling.engine import run_experiment
        from standard_quant_tools.modeling.specs import (
            EstimatorSpec,
            ModelSpec,
            TargetSpec,
            ValidationSpec,
        )

        from .test_scoring import _dataset_spec

        dataset = build_dataset(
            _dataset_spec(target=TargetSpec(type="forward_direction", horizon=5))
        )
        result = run_experiment(
            dataset,
            ModelSpec(
                task="classification",
                estimator=EstimatorSpec(type="random_forest"),
                validation=ValidationSpec(
                    method="walk_forward",
                    train_window=120,
                    test_window=40,
                    min_folds=1,
                ),
                random_seed=0,
            ),
            "ds_auc_test",
            register=False,
        )
        headline = result["validation_report"]["headline"]
        assert headline["metric"] == "auc"
        # The verdict now rests on something.
        assert headline["p_value"] is not None
        assert headline["n_dates"] is not None
        if headline["beats_null"] is True:
            assert headline["p_value"] < 0.05

    def test_a_weak_classifier_is_no_longer_called_a_winner(
        self, patched_multi_factory
    ):
        """The regression this closes: a near-chance AUC reporting True.
        Whatever this fixture's model scores, a True must now be backed by
        a p-value under 5% — which is the assertion above — and a near-
        chance AUC must not produce one."""
        from standard_quant_tools.modeling.engine import auc_vs_chance

        _z, p, _pos, _neg = auc_vs_chance(0.501, _metrics(0.5, 500))
        assert p > 0.05
