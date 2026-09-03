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

    @staticmethod
    def _contract(values, dates, clip_sigma=3.0):
        """
        An INDEPENDENT oracle, written from panel_stats.hpp rather than from
        either implementation.

        `_agree` compares the two backends to each other, which cannot catch
        a misunderstanding they share -- and one they shared survived here
        for exactly that reason: both zeroed a whole date's cross-section
        when any single entity was missing, while the header promised the
        opposite. Backend parity and contract conformance are two different
        questions and need two different oracles.
        """
        values = np.asarray(values, dtype=float)
        dates = np.asarray(dates)
        out = np.full(values.shape, np.nan)
        for date in pd.unique(dates):
            rows = dates == date
            segment = values[rows]
            present = ~np.isnan(segment)
            if not present.any():
                continue
            spread = float(np.nanstd(segment, ddof=1)) if present.sum() > 1 else 0.0
            if spread > 0.0:
                z = (segment[present] - np.nanmean(segment)) / spread
            else:
                # No dispersion: every PRESENT entity sits exactly at the mean.
                z = np.zeros(int(present.sum()))
            here = np.full(segment.shape, np.nan)
            here[present] = np.clip(z, -clip_sigma, clip_sigma)
            out[rows] = here
        return out

    def test_a_missing_entity_does_not_move_the_others(self, monkeypatch):
        """
        THE DEFECT THIS EXISTS TO CATCH, and it was shipped, deliberately,
        with a comment calling it a wart.

        One NaN used to poison the date's mean, and the non-finite sweep
        then wrote 0.0 for every entity in that date -- reporting each
        PRESENT name as sitting exactly at the cross-sectional mean, which
        panel_stats.hpp names as the specific thing that must not happen.
        Its justification was that alignment drops NaN rows before the
        engine sees a panel; `load_external_panel` retired that premise,
        because an externally computed panel keeps its warm-up NaNs.
        """
        dates = np.repeat(pd.date_range("2020-01-01", periods=2), 4)
        frame = pd.DataFrame({"x": [1.0, 2.0, np.nan, 4.0, 1.0, 2.0, 3.0, 4.0]})
        self._agree(frame, dates, monkeypatch)

        out = standardize_cross_sectional(frame, dates)["x"].to_numpy()
        expected = self._contract(frame["x"].to_numpy(), dates)
        assert np.allclose(out, expected, equal_nan=True)

        # Stated concretely, so the regression is unmistakable: the three
        # present names keep their ordering and spread, and only the absent
        # one is absent.
        assert np.isnan(out[2]), "the missing entity stays missing"
        assert not np.isclose(out[0], 0.0), "a present entity is not zeroed"
        assert out[0] < out[1] < out[3], "their ordering survives"

    @pytest.mark.parametrize(
        "values",
        [
            [1.0, 2.0, np.nan, 4.0],
            [np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, 2.0, np.nan],
            [2.0, 2.0, 2.0, 2.0],
            [2.0, 2.0, np.nan, 2.0],
            [-5.0, 0.0, 5.0, np.nan],
        ],
    )
    def test_the_contract_holds_on_every_degenerate_cross_section(
        self, values, monkeypatch
    ):
        """Checked against the header, not against the other backend."""
        dates = np.repeat(pd.date_range("2020-01-01", periods=1), 4)
        frame = pd.DataFrame({"x": values})
        self._agree(frame, dates, monkeypatch)
        assert np.allclose(
            standardize_cross_sectional(frame, dates)["x"].to_numpy(),
            self._contract(np.asarray(values, dtype=float), dates),
            equal_nan=True,
        )

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


class TestLabelUniqueness:
    """
    The worst per-row cost in the module before it got a kernel: 5.7
    microseconds per row at 2,000,000 rows, because the Python loops once
    per entity. The kernel is gated by size, since below the crossover the
    argument conversion costs more than the loop saves -- a fast path that
    is slower is a bug, not a trade-off.
    """

    @staticmethod
    def _panel(n_entities, n_bars, horizon=5, shuffle=False, seed=0):
        rng = np.random.default_rng(seed)
        rows = []
        for entity in range(n_entities):
            index = pd.date_range("2015-01-02", periods=n_bars, freq="B")
            ends = np.r_[
                index[horizon:].to_numpy(),
                np.repeat(np.datetime64("NaT"), min(horizon, n_bars)),
            ][:n_bars]
            for i in range(n_bars):
                rows.append((index[i], ends[i], f"E{entity}"))
        if shuffle:
            rng.shuffle(rows)
        frame = pd.DataFrame(rows, columns=["date", "end", "entity"])
        return (
            frame["date"].to_numpy(),
            frame["end"].to_numpy(),
            frame["entity"].to_numpy(),
        )

    def _agree(self, dates, ends, entities, monkeypatch):
        from standard_quant_tools.modeling.validation import weights as weights_module

        native = weights_module.label_uniqueness_weights(dates, ends, entities)
        monkeypatch.setattr(weights_module, "HAS_CPP", False)
        python = weights_module.label_uniqueness_weights(dates, ends, entities)
        monkeypatch.undo()
        np.testing.assert_allclose(native, python, rtol=0, atol=1e-12)
        assert np.isclose(native.mean(), 1.0), "weights are normalized to mean 1"

    def test_matches_python_above_the_threshold(self, monkeypatch):
        """Big enough that the native path actually engages."""
        from standard_quant_tools.modeling.validation import weights as weights_module

        dates, ends, entities = self._panel(250, 252)
        assert dates.size >= weights_module._NATIVE_MIN_ROWS
        self._agree(dates, ends, entities, monkeypatch)

    def test_matches_python_below_the_threshold(self, monkeypatch):
        """Small panels take the Python path; they must still be right."""
        dates, ends, entities = self._panel(5, 60)
        self._agree(dates, ends, entities, monkeypatch)

    def test_unsorted_rows(self, monkeypatch):
        """The kernel buckets by entity and orders each by its own dates, so
        the caller need not have sorted anything."""
        dates, ends, entities = self._panel(250, 252, shuffle=True, seed=3)
        self._agree(dates, ends, entities, monkeypatch)

    def test_entities_on_different_calendars(self, monkeypatch):
        """
        The reason label ends are timestamps and not integer offsets: with
        entities on different bar calendars, t+horizon of one entity is not
        t+horizon of another, and an offset would purge the wrong rows.
        """
        rows = []
        for entity in range(200):
            freq = "B" if entity % 2 else "2B"
            index = pd.date_range("2015-01-02", periods=252, freq=freq)
            ends = np.r_[index[5:].to_numpy(), np.repeat(np.datetime64("NaT"), 5)]
            for i in range(252):
                rows.append((index[i], ends[i], f"E{entity}"))
        frame = pd.DataFrame(rows, columns=["date", "end", "entity"])
        self._agree(
            frame["date"].to_numpy(),
            frame["end"].to_numpy(),
            frame["entity"].to_numpy(),
            monkeypatch,
        )

    def test_all_labels_unresolved(self, monkeypatch):
        """Every label NaT: each row spans only its own bar, so every weight
        is identical and normalization makes them all 1.0."""
        from standard_quant_tools.modeling.validation import weights as weights_module

        n = 60_000
        dates = np.tile(
            pd.date_range("2015-01-02", periods=300, freq="B").to_numpy(), 200
        )
        ends = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
        entities = np.repeat([f"E{i}" for i in range(200)], 300)
        out = weights_module.label_uniqueness_weights(dates, ends, entities)
        np.testing.assert_allclose(out, np.ones(n), rtol=0, atol=1e-12)
