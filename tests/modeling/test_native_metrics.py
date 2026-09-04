import gc
import time

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

from standard_quant_tools.modeling.analysis import feature_stability
from standard_quant_tools.modeling.analysis.feature_stability import (
    _null_distribution,
)
from standard_quant_tools.modeling.features import transforms
from standard_quant_tools.modeling.features.transforms import (
    cross_sectional_counts,
    rank_within_date,
)
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


class TestRankWithinDate:
    """
    The one kernel in this layer that exists because VECTORISATION LOST.
    pandas' `groupby.rank` is already good -- a numpy rewrite measured 395 ms
    against its 407 ms at three columns and worse at five -- so the parity
    oracle here is pandas itself, and the contract oracle is what
    panel_stats.hpp promises about NaN.
    """

    @staticmethod
    def _panel(kind, n_dates=40, per_date=25, n_cols=3, seed=0):
        rng = np.random.default_rng(seed)
        n = n_dates * per_date
        dates = np.repeat(np.arange(n_dates), per_date)
        values = rng.normal(0, 1, (n, n_cols))
        if kind == "nan":
            values[rng.random(values.shape) < 0.2] = np.nan
        elif kind == "ties":
            values = rng.integers(0, 3, (n, n_cols)).astype(float)
        elif kind == "constant":
            values = np.full((n, n_cols), 4.0)
        elif kind == "dead_column":
            values[:, 1] = np.nan
        elif kind == "dead_date":
            values[dates == 7] = np.nan
        elif kind == "negatives":
            values = rng.integers(-3, 2, (n, n_cols)).astype(float)
        frame = pd.DataFrame(values, columns=[f"m{i}" for i in range(n_cols)])
        return frame, dates

    @pytest.mark.parametrize(
        "kind",
        ["clean", "nan", "ties", "constant", "dead_column", "dead_date", "negatives"],
    )
    def test_both_backends_match_pandas(self, kind, monkeypatch) -> None:
        frame, dates = self._panel(kind)
        expected = frame.groupby(dates, sort=False).rank(method="average")

        native = rank_within_date(frame, dates).to_numpy()
        monkeypatch.setattr(transforms, "HAS_CPP", False)
        python = rank_within_date(frame, dates).to_numpy()
        monkeypatch.undo()

        assert np.array_equal(native, expected.to_numpy(), equal_nan=True)
        assert np.array_equal(python, expected.to_numpy(), equal_nan=True)

    def test_a_missing_name_does_not_shift_the_others(self) -> None:
        """
        The contract panel_stats.hpp states: NaN is skipped by the ranking
        and preserved, and the values that ARE present rank 1..n-1.
        """
        frame = pd.DataFrame({"m0": [10.0, np.nan, 30.0, 20.0]})
        dates = np.zeros(4, dtype=int)
        ranked = rank_within_date(frame, dates)["m0"].to_numpy()
        assert np.isnan(ranked[1])
        assert ranked[0] == 1.0 and ranked[3] == 2.0 and ranked[2] == 3.0

    def test_ties_take_the_mean_of_their_ordinals(self) -> None:
        frame = pd.DataFrame({"m0": [5.0, 5.0, 5.0, 9.0]})
        dates = np.zeros(4, dtype=int)
        ranked = rank_within_date(frame, dates)["m0"].to_numpy()
        # Ordinals 1,2,3 average to 2.0; the odd one out is 4.
        assert ranked.tolist() == [2.0, 2.0, 2.0, 4.0]

    def test_dates_need_not_be_sorted(self) -> None:
        frame = pd.DataFrame({"m0": [1.0, 9.0, 2.0, 8.0]})
        dates = np.array([1, 0, 1, 0])
        ranked = rank_within_date(frame, dates)["m0"].to_numpy()
        assert ranked.tolist() == [1.0, 2.0, 2.0, 1.0]

    def test_counts_match_pandas(self, kind="nan") -> None:
        frame, dates = self._panel(kind)
        expected = frame.groupby(dates, sort=False).transform("count")
        got = cross_sectional_counts(frame, dates)
        assert np.array_equal(got.to_numpy(), expected.to_numpy().astype(float))

    @pytest.mark.benchmark
    def test_the_kernel_beats_pandas(self) -> None:
        """
        The plan this kernel came from found that `HAS_CPP` appeared nowhere
        in `tests/bench/`, so the previous native work's speedup figures
        could not be re-derived from committed code. This one ships with its
        own gate.
        """
        frame, dates = self._panel("clean", n_dates=400, per_date=250, n_cols=3)

        def _timed(fn, reps=3):
            fn()
            best = float("inf")
            gc.disable()
            try:
                for _ in range(reps):
                    start = time.perf_counter()
                    fn()
                    best = min(best, time.perf_counter() - start)
            finally:
                gc.enable()
            return best * 1000.0

        pandas_ms = _timed(
            lambda: frame.groupby(dates, sort=False).rank(method="average")
        )
        kernel_ms = _timed(lambda: rank_within_date(frame, dates))
        speedup = pandas_ms / kernel_ms
        print(
            f"\n  rank_by_date 100,000 x 3: pandas {pandas_ms:7.1f} ms  "
            f"kernel {kernel_ms:6.1f} ms  speedup {speedup:.1f}x"
        )
        assert kernel_ms < pandas_ms
        assert speedup >= 3.0, f"expected >= 3x, got {speedup:.1f}x"


class TestPermutationNull:
    """
    The kernel the two backends CANNOT agree bit for bit, and why that is
    still correct.

    A permutation test is a Monte Carlo estimate. The kernel uses its own
    generator rather than a reimplementation of numpy's PCG64 stream, so the
    same seed produces different DRAWS with and without the extension --
    exactly the contract `simulate_forward_paths` states. What must agree is
    the null they are drawn FROM, so that is what these assert, against an
    analytic value rather than against each other.
    """

    N_DATES = 200
    PER_DATE = 50

    @classmethod
    def _noise_panel(cls, seed=0):
        rng = np.random.default_rng(seed)
        dates = np.repeat(np.arange(cls.N_DATES), cls.PER_DATE)
        n = dates.size
        # Pure noise: the null hypothesis is TRUE here, so the null
        # distribution is the whole answer.
        return dates, rng.normal(0, 1, n), rng.normal(0, 1, n)

    @classmethod
    def _analytic_sd(cls):
        # Each date's spearman IC has variance about 1/(m-1); the mean of
        # n_dates independent ones has that over n_dates.
        return np.sqrt(1.0 / ((cls.PER_DATE - 1) * cls.N_DATES))

    def test_the_null_has_the_width_theory_says(self) -> None:
        dates, values, target = self._noise_panel()
        null = _null_distribution(target, values, dates, 4000, "spearman", 1)
        assert abs(null.mean()) < 3 * self._analytic_sd()
        # A null that is too NARROW is the dangerous direction: it makes
        # ordinary noise look significant.
        assert 0.9 < null.std() / self._analytic_sd() < 1.1

    def test_the_python_fallback_has_the_same_width(self, monkeypatch) -> None:
        dates, values, target = self._noise_panel()
        monkeypatch.setattr(feature_stability, "HAS_CPP", False)
        null = _null_distribution(target, values, dates, 600, "spearman", 1)
        monkeypatch.undo()
        assert 0.85 < null.std() / self._analytic_sd() < 1.15

    def test_a_seed_reproduces_a_run_within_the_backend(self) -> None:
        dates, values, target = self._noise_panel()
        first = _null_distribution(target, values, dates, 100, "spearman", 5)
        second = _null_distribution(target, values, dates, 100, "spearman", 5)
        assert np.array_equal(first, second)

    def test_a_different_seed_draws_differently(self) -> None:
        dates, values, target = self._noise_panel()
        assert not np.array_equal(
            _null_distribution(target, values, dates, 100, "spearman", 5),
            _null_distribution(target, values, dates, 100, "spearman", 6),
        )

    def test_both_backends_reject_a_real_signal(self, monkeypatch) -> None:
        """The p-value is what callers act on, so agree on THAT."""
        rng = np.random.default_rng(3)
        dates = np.repeat(np.arange(self.N_DATES), self.PER_DATE)
        values = rng.normal(0, 1, dates.size)
        target = 0.25 * values + rng.normal(0, 1, dates.size)
        observed = float(
            cross_sectional_ic(target, values, dates, method="spearman").mean()
        )

        def _p(null):
            return (np.sum(np.abs(null) >= abs(observed)) + 1) / (null.size + 1)

        native = _p(_null_distribution(target, values, dates, 400, "spearman", 2))
        monkeypatch.setattr(feature_stability, "HAS_CPP", False)
        python = _p(_null_distribution(target, values, dates, 400, "spearman", 2))
        monkeypatch.undo()
        assert native == python  # both bottom out at 1/(n+1) on a real signal
        assert native < 0.01

    def test_pearson_is_wired_through_too(self) -> None:
        dates, values, target = self._noise_panel()
        null = _null_distribution(target, values, dates, 500, "pearson", 1)
        assert np.isfinite(null).all()
        assert 0.85 < null.std() / self._analytic_sd() < 1.15

    @pytest.mark.benchmark
    def test_the_kernel_beats_the_python_loop(self, monkeypatch) -> None:
        dates, values, target = self._noise_panel()
        reps = 200

        start = time.perf_counter()
        _null_distribution(target, values, dates, reps, "spearman", 1)
        kernel_s = time.perf_counter() - start

        monkeypatch.setattr(feature_stability, "HAS_CPP", False)
        start = time.perf_counter()
        _null_distribution(target, values, dates, reps, "spearman", 1)
        python_s = time.perf_counter() - start
        monkeypatch.undo()

        speedup = python_s / kernel_s
        print(
            f"\n  permutation_null_ic {dates.size:,} rows x {reps} draws: "
            f"python {python_s:6.2f} s  kernel {kernel_s:5.3f} s  "
            f"speedup {speedup:.1f}x"
        )
        assert kernel_s < python_s
        assert speedup >= 5.0, f"expected >= 5x, got {speedup:.1f}x"
