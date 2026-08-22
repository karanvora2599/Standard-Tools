"""
The native correlation kernel must be invisible except in the clock.

Ranking and correlation are 93% of `regression_metrics` at universe scale --
the POOLED rank IC alone is 41-51% of it, because it sorts the whole test
fold through scipy -- so both the pooled and the per-date forms are served by
one kernel. That is deliberate: they are the same computation over different
segmentations, and two implementations would be two things to keep in step.

`TestCrossSectionalICVectorization` in test_validation_statistics.py already
compares `cross_sectional_ic` against a per-date pandas oracle, and with the
extension present those tests exercise the kernel. What is added here is the
explicit native-vs-Python toggle, the pooled path (which that file does not
reach), and the boundary conditions where the two implementations could
plausibly diverge.
"""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.modeling.features import transforms
from standard_quant_tools.modeling.features.transforms import (
    standardize_cross_sectional,
)
from standard_quant_tools.modeling.validation import metrics
from standard_quant_tools.modeling.validation.metrics import (
    _safe_corr,
    cross_sectional_ic,
)

pytestmark = pytest.mark.skipif(
    not metrics.HAS_CPP, reason="native extension not built"
)

TOL = 1e-12


def _both_ic(y, p, dates, method, monkeypatch):
    native = cross_sectional_ic(y, p, dates, method)
    monkeypatch.setattr(metrics, "HAS_CPP", False)
    python = cross_sectional_ic(y, p, dates, method)
    monkeypatch.undo()
    return native, python


def _assert_ic_agree(y, p, dates, monkeypatch):
    for method in ("spearman", "pearson"):
        native, python = _both_ic(y, p, dates, method, monkeypatch)
        assert list(native.index) == list(python.index), method
        assert len(native) == len(python), method
        if len(python):
            np.testing.assert_allclose(
                native.to_numpy(), python.to_numpy(), rtol=0, atol=TOL
            )


class TestCrossSectionalCorrelation:
    @pytest.mark.parametrize("seed", range(6))
    def test_matches_the_python_path(self, monkeypatch, seed):
        rng = np.random.default_rng(seed)
        n_dates, n_entities = 40, 15
        dates = np.repeat(pd.date_range("2020-01-01", periods=n_dates), n_entities)
        y = rng.normal(0, 1, n_dates * n_entities)
        p = 0.3 * y + rng.normal(0, 1, n_dates * n_entities)
        _assert_ic_agree(y, p, dates, monkeypatch)

    def test_ties(self, monkeypatch):
        """The rank kernel assigns the MEAN ordinal over a tied run; getting
        that wrong is quiet, not loud."""
        rng = np.random.default_rng(11)
        dates = np.repeat(pd.date_range("2020-01-01", periods=25), 20)
        y = np.round(rng.normal(0, 1, 500), 1)
        p = np.round(0.4 * y + rng.normal(0, 1, 500), 1)
        assert len(np.unique(y)) < 100, "test data is not actually tied"
        _assert_ic_agree(y, p, dates, monkeypatch)

    def test_return_scale(self, monkeypatch):
        rng = np.random.default_rng(3)
        dates = np.repeat(pd.date_range("2020-01-01", periods=40), 60)
        y = rng.normal(0.0004, 0.012, 2400)
        p = 0.25 * y + rng.normal(0.0004, 0.012, 2400)
        _assert_ic_agree(y, p, dates, monkeypatch)

    def test_constant_cross_section(self, monkeypatch):
        rng = np.random.default_rng(5)
        dates = np.repeat(pd.date_range("2020-01-01", periods=4), 6)
        y = rng.normal(0, 1, 24)
        p = rng.normal(0, 1, 24)
        p[:6] = 7.0
        native, _ = _both_ic(y, p, dates, "pearson", monkeypatch)
        assert native.iloc[0] == 0.0
        _assert_ic_agree(y, p, dates, monkeypatch)

    def test_nan_pairs_are_dropped(self, monkeypatch):
        rng = np.random.default_rng(29)
        dates = np.repeat(pd.date_range("2022-02-01", periods=25), 12)
        y = rng.normal(0, 1, 300)
        p = 0.4 * y + rng.normal(0, 1, 300)
        y[rng.random(300) < 0.25] = np.nan
        p[rng.random(300) < 0.15] = np.nan
        _assert_ic_agree(y, p, dates, monkeypatch)

    def test_infinities_flow_through(self, monkeypatch):
        rng = np.random.default_rng(31)
        dates = np.repeat(pd.date_range("2023-01-02", periods=5), 8)
        y = rng.normal(0, 1, 40)
        p = rng.normal(0, 1, 40)
        p[3] = np.inf
        y[20] = -np.inf
        _assert_ic_agree(y, p, dates, monkeypatch)

    def test_ragged_panel(self, monkeypatch):
        """The kernel counting-sorts rather than assuming equal-width dates."""
        rng = np.random.default_rng(17)
        rows = []
        for date in pd.date_range("2021-01-04", periods=45):
            for _ in range(int(rng.integers(1, 22))):
                rows.append((date, rng.normal(), rng.normal()))
        frame = pd.DataFrame(rows, columns=["date", "y", "p"])
        assert frame.groupby("date").size().nunique() > 1
        _assert_ic_agree(
            frame["y"].to_numpy(),
            frame["p"].to_numpy(),
            frame["date"].to_numpy(),
            monkeypatch,
        )

    def test_unsorted_rows(self, monkeypatch):
        """
        The kernel buckets by date code and does not require the caller to
        have sorted. Shuffling the rows must not change any date's answer.
        """
        rng = np.random.default_rng(41)
        n_dates, n_entities = 30, 12
        dates = np.repeat(pd.date_range("2020-01-01", periods=n_dates), n_entities)
        y = rng.normal(0, 1, n_dates * n_entities)
        p = 0.3 * y + rng.normal(0, 1, n_dates * n_entities)
        ordered = cross_sectional_ic(y, p, dates, "spearman")
        shuffle = rng.permutation(y.size)
        shuffled = cross_sectional_ic(
            y[shuffle], p[shuffle], dates[shuffle], "spearman"
        )
        assert list(ordered.index) == list(shuffled.index)
        np.testing.assert_allclose(
            ordered.to_numpy(), shuffled.to_numpy(), rtol=0, atol=TOL
        )

    def test_all_nan_date_still_emitted_as_zero(self, monkeypatch):
        """The raw-count gate is applied on the Python side, so the kernel
        must not change which dates appear."""
        dates = np.repeat(pd.date_range("2020-05-05", periods=2), 4)
        y = np.array([np.nan] * 4 + [1.0, 2.0, 3.0, 4.0])
        p = np.array([np.nan] * 4 + [2.0, 1.0, 4.0, 3.0])
        native, python = _both_ic(y, p, dates, "pearson", monkeypatch)
        assert len(native) == 2
        assert native.iloc[0] == 0.0
        np.testing.assert_allclose(native.to_numpy(), python.to_numpy(), atol=TOL)


class TestPooledCorrelation:
    """
    `_safe_corr` routes pooled SPEARMAN through the kernel above a size
    threshold, as one segment covering every row. Pearson stays in pandas:
    it is already a couple of passes and the kernel does not beat it by
    enough to pay for the conversion.
    """

    @pytest.mark.parametrize("n", [6_000, 30_000])
    def test_pooled_spearman_matches_pandas(self, n):
        rng = np.random.default_rng(2)
        y = pd.Series(rng.normal(0, 0.02, n))
        p = pd.Series(0.3 * y.to_numpy() + rng.normal(0, 0.02, n))
        expected = float(y.corr(p, method="spearman"))
        assert _safe_corr(y, p, "spearman") == pytest.approx(expected, abs=TOL)

    def test_pooled_spearman_with_ties(self):
        rng = np.random.default_rng(4)
        y = pd.Series(np.round(rng.normal(0, 1, 20_000), 1))
        p = pd.Series(np.round(0.4 * y.to_numpy() + rng.normal(0, 1, 20_000), 1))
        expected = float(y.corr(p, method="spearman"))
        assert _safe_corr(y, p, "spearman") == pytest.approx(expected, abs=TOL)

    def test_below_the_threshold_uses_pandas(self):
        """Small inputs must still be correct, whichever path they take."""
        rng = np.random.default_rng(6)
        y = pd.Series(rng.normal(0, 1, 200))
        p = pd.Series(rng.normal(0, 1, 200))
        expected = float(y.corr(p, method="spearman"))
        assert _safe_corr(y, p, "spearman") == pytest.approx(expected, abs=TOL)

    def test_constant_input_is_zero_not_nan(self):
        """The 0.0-not-NaN contract, on the pooled path as well."""
        y = pd.Series(np.arange(10_000, dtype=float))
        p = pd.Series(np.full(10_000, 3.0))
        assert _safe_corr(y, p, "spearman") == 0.0

    def test_pooled_pearson_still_matches(self):
        rng = np.random.default_rng(8)
        y = pd.Series(rng.normal(0, 0.02, 20_000))
        p = pd.Series(0.3 * y.to_numpy() + rng.normal(0, 0.02, 20_000))
        assert _safe_corr(y, p, "pearson") == pytest.approx(
            float(y.corr(p, method="pearson")), abs=TOL
        )


class TestStandardizeByDate:
    def _agree(self, frame, dates, monkeypatch, clip=3.0):
        native = standardize_cross_sectional(frame, dates, clip)
        monkeypatch.setattr(transforms, "HAS_CPP", False)
        python = standardize_cross_sectional(frame, dates, clip)
        monkeypatch.undo()
        np.testing.assert_allclose(
            native.to_numpy(), python.to_numpy(), rtol=0, atol=TOL
        )
        assert list(native.columns) == list(python.columns)
        assert native.index.equals(python.index)

    @pytest.mark.parametrize("seed", range(4))
    def test_matches_the_python_path(self, monkeypatch, seed):
        rng = np.random.default_rng(seed)
        dates = np.repeat(pd.date_range("2020-01-01", periods=30), 8)
        frame = pd.DataFrame(
            rng.normal(0, 1, (240, 4)), columns=[f"f{i}" for i in range(4)]
        )
        self._agree(frame, dates, monkeypatch)

    def test_constant_date_becomes_zero(self, monkeypatch):
        dates = np.repeat(pd.date_range("2020-01-01", periods=3), 4)
        frame = pd.DataFrame({"f": [7.0] * 4 + list(range(4)) + [2.0] * 4})
        self._agree(frame, dates, monkeypatch)
        out = standardize_cross_sectional(frame, dates)
        assert out["f"].iloc[:4].eq(0.0).all()

    def test_nan_poisons_its_whole_date(self, monkeypatch):
        """
        A quirk preserved deliberately. The Python path reduces with
        np.add.reduceat, which propagates a NaN into the date's mean, and
        then maps every non-finite result to 0.0 -- so one missing value
        zeroes that date's ENTIRE column, not only its own row. The kernel
        reproduces it exactly, because this is a speed change. In practice
        it never fires: alignment drops NaN rows before the engine sees the
        panel.
        """
        dates = np.repeat(pd.date_range("2020-01-01", periods=2), 4)
        frame = pd.DataFrame({"x": [1.0, 2.0, np.nan, 4.0, 1.0, 2.0, 3.0, 4.0]})
        self._agree(frame, dates, monkeypatch)
        out = standardize_cross_sectional(frame, dates)
        assert out["x"].iloc[:4].eq(0.0).all(), "the whole poisoned date is 0.0"
        assert not out["x"].iloc[4:].eq(0.0).any(), "the clean date is unaffected"

    def test_ragged_dates(self, monkeypatch):
        rng = np.random.default_rng(12)
        rows = []
        for i, date in enumerate(pd.date_range("2020-01-01", periods=15)):
            for _ in range(2 + i % 5):
                rows.append((date, rng.normal(), rng.normal()))
        frame = pd.DataFrame(rows, columns=["date", "a", "b"])
        self._agree(frame[["a", "b"]], frame["date"].to_numpy(), monkeypatch)

    def test_clipping_disabled(self, monkeypatch):
        rng = np.random.default_rng(14)
        dates = np.repeat(pd.date_range("2020-01-01", periods=10), 20)
        frame = pd.DataFrame({"f": rng.standard_t(2.0, 200)})
        self._agree(frame, dates, monkeypatch, clip=0.0)

    def test_unsorted_rows(self):
        """Rows need not arrive grouped by date."""
        rng = np.random.default_rng(16)
        dates = np.repeat(pd.date_range("2020-01-01", periods=20), 10)
        frame = pd.DataFrame(rng.normal(0, 1, (200, 3)), columns=["a", "b", "c"])
        ordered = standardize_cross_sectional(frame, dates).to_numpy()
        shuffle = rng.permutation(200)
        shuffled = standardize_cross_sectional(
            frame.iloc[shuffle].reset_index(drop=True), dates[shuffle]
        ).to_numpy()
        np.testing.assert_allclose(ordered[shuffle], shuffled, rtol=0, atol=TOL)
