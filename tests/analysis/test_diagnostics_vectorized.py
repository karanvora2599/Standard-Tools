"""
`entropy_measures` and `lead_lag_matrix` do their per-element work as array
passes and return what their loops returned.

`entropy_measures` sorted every window of the series in its own `argsort`
call and counted the rank patterns in a dictionary; `lead_lag_matrix`
visited every (leader, follower, lag) triple in Python to keep the few that
clear `min_correlation`. Both now run those steps over whole arrays (see
the CHANGELOG entry of 2026-10-01). The references below are the functions
as they were, kept verbatim, and every result is required to be identical
-- the same doubles, the same rows in the same order, the same types --
not close.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.analysis import diagnostics
from standard_quant_tools.analysis.diagnostics import (
    _clean,
    entropy_measures,
    lead_lag_matrix,
)
from standard_quant_tools.error import ValidationError


def _reference_entropy_measures(
    series: pd.Series,
    *,
    n_bins: int = 8,
    embedding: int = 3,
) -> Dict[str, Any]:
    """`entropy_measures` as it was, verbatim but for this docstring."""
    values = _clean(series, "entropy_measures", minimum=50)
    array = values.to_numpy()
    n = array.size
    n_bins = max(2, int(n_bins))
    embedding = int(embedding)
    if not 2 <= embedding <= 7:
        raise ValidationError(
            f"entropy_measures: embedding={embedding} must be between 2 and "
            "7. Below 2 there is no ordering to read; above 7 there are "
            "5040 patterns and no sample fills them."
        )
    if n < math.factorial(embedding) * 5:
        raise ValidationError(
            f"entropy_measures: {n} observations cannot populate the "
            f"{math.factorial(embedding)} rank patterns of an embedding of "
            f"{embedding}. Use a smaller embedding or more data."
        )

    counts, _ = np.histogram(array, bins=n_bins)
    probabilities = counts[counts > 0] / counts.sum()
    shannon = float(-(probabilities * np.log(probabilities)).sum())
    shannon_normalized = shannon / math.log(n_bins)

    # Permutation entropy: count each ordinal pattern of length `embedding`.
    patterns: Dict[tuple, int] = {}
    for i in range(n - embedding + 1):
        window = array[i : i + embedding]
        key = tuple(np.argsort(window))
        patterns[key] = patterns.get(key, 0) + 1
    total = sum(patterns.values())
    pattern_probabilities = np.array([c / total for c in patterns.values()])
    permutation = float(-(pattern_probabilities * np.log(pattern_probabilities)).sum())
    permutation_normalized = permutation / math.log(math.factorial(embedding))

    per_bin = n / n_bins
    warnings: List[str] = []
    if permutation_normalized > 0.98:
        warnings.append(
            f"Permutation entropy is {permutation_normalized:.4f} of its "
            "maximum -- this series is indistinguishable from random in its "
            "ordering. No nonlinear structure is detectable at this "
            "embedding."
        )
    elif permutation_normalized < 0.9:
        warnings.append(
            f"Permutation entropy is {permutation_normalized:.4f}, "
            "materially below random. There is ordering structure here that "
            "a linear test would miss -- though structure is not an edge, "
            "and the commonest cause is a trend."
        )
    if per_bin < 20:
        warnings.append(
            f"{per_bin:.0f} observations per bin. The Shannon figure is "
            "sensitive to the bin count and this is thin; too few bins make "
            "everything look uniform and too many do the same."
        )
    warnings.append(
        "Permutation entropy reads RANKS only, so it is invariant to any "
        "monotone transformation and robust to outliers. That is a strength "
        "for detection and a limitation for interpretation: it will not "
        "tell you how large the structure is, only that it is there."
    )

    return {
        "n_observations": int(n),
        "n_bins": n_bins,
        "embedding": embedding,
        "observations_per_bin": float(per_bin),
        "shannon_entropy": shannon,
        "shannon_normalized": float(shannon_normalized),
        "permutation_entropy": permutation,
        "permutation_normalized": float(permutation_normalized),
        "n_patterns_observed": len(patterns),
        "n_patterns_possible": math.factorial(embedding),
        "warnings": warnings,
    }


def _reference_lead_lag_matrix(
    returns: pd.DataFrame,
    *,
    max_lag: int = 3,
    min_correlation: float = 0.1,
) -> Dict[str, Any]:
    """`lead_lag_matrix` as it was, verbatim but for this docstring and for
    reaching `_f_sf` through its module, so a test can watch the calls."""
    frame = pd.DataFrame(returns).astype(float).dropna()
    n_assets = frame.shape[1]
    if n_assets < 2:
        raise ValidationError("lead_lag_matrix: needs at least two series.")
    max_lag = int(max_lag)
    if max_lag < 1:
        raise ValidationError("lead_lag_matrix: max_lag must be at least 1.")
    n = len(frame)
    if n < 50:
        raise ValidationError(
            f"lead_lag_matrix: {n} observations. A correlation search this "
            "wide needs far more data to have any power after correction."
        )

    n_tests = n_assets * (n_assets - 1) * max_lag
    pairs: List[Dict[str, Any]] = []
    columns = list(frame.columns)

    values = frame.to_numpy(dtype=float)
    correlations: Dict[int, np.ndarray] = {}
    for lag in range(1, max_lag + 1):
        if n - lag < 30:
            continue
        lead = values[:-lag]
        follow = values[lag:]
        lead_c = lead - lead.mean(axis=0)
        follow_c = follow - follow.mean(axis=0)
        denominator = np.sqrt(
            np.outer((lead_c**2).sum(axis=0), (follow_c**2).sum(axis=0))
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            correlations[lag] = (lead_c.T @ follow_c) / denominator

    for i, leader in enumerate(columns):
        for j, follower in enumerate(columns):
            if i == j:
                continue
            for lag in range(1, max_lag + 1):
                matrix = correlations.get(lag)
                if matrix is None:
                    continue
                rho = float(matrix[i, j])
                if not math.isfinite(rho) or abs(rho) < min_correlation:
                    continue
                effective = n - lag
                t = rho * math.sqrt(max(effective - 2, 1) / max(1 - rho * rho, 1e-12))
                raw_p = diagnostics._f_sf(t * t, 1, max(effective - 2, 1))
                pairs.append(
                    {
                        "leader": str(leader),
                        "follower": str(follower),
                        "lag": lag,
                        "correlation": rho,
                        "p_value_raw": float(raw_p),
                        "p_value_corrected": float(min(raw_p * n_tests, 1.0)),
                        "survives_correction": bool(raw_p * n_tests < 0.05),
                    }
                )
    pairs.sort(key=lambda p: abs(p["correlation"]), reverse=True)
    survivors = [p for p in pairs if p["survives_correction"]]

    expected_false = n_tests * 0.05
    warnings: List[str] = []
    if not survivors:
        warnings.append(
            f"NOTHING SURVIVED the correction. {n_tests} correlations were "
            f"tested and about {expected_false:.0f} would clear an "
            "uncorrected 5% bar on data with no lead-lag structure at all. "
            "The top of the ranked list below is what noise looks like, not "
            "a finding."
        )
    else:
        warnings.append(
            f"{len(survivors)} of {n_tests} tested relationships survive "
            "Bonferroni correction. Before trading one, rule out different "
            "closing times across exchanges -- that produces exactly this "
            "pattern and is the commonest cause by a wide margin."
        )
    warnings.append(
        "Temporal precedence is not causality and is not a trade. A common "
        "driver produces it, and so does a faster-updating proxy for the "
        "same information."
    )

    return {
        "n_assets": int(n_assets),
        "n_observations": int(n),
        "max_lag": max_lag,
        "n_tests": int(n_tests),
        "expected_false_positives_uncorrected": float(expected_false),
        "n_surviving": len(survivors),
        "surviving_pairs": survivors[:20],
        "strongest_pairs": pairs[:10],
        "warnings": warnings,
    }


def _identical(a, b) -> bool:
    """Equal and of the same type, all the way down; floats to the bit."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_identical(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_identical(x, y) for x, y in zip(a, b))
    if isinstance(a, float):
        if math.isnan(a) or math.isnan(b):
            return math.isnan(a) and math.isnan(b)
        return a == b and math.copysign(1.0, a) == math.copysign(1.0, b)
    return a == b


def _outcome(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except ValidationError as error:
        return ("ValidationError", str(error))


# ── entropy ─────────────────────────────────────────────────────────────


def _assert_same_entropy(series, **kwargs):
    expected = _outcome(_reference_entropy_measures, series, **kwargs)
    actual = _outcome(entropy_measures, series, **kwargs)
    assert _identical(actual, expected), (kwargs, actual, expected)
    return actual


class TestEntropyIsTheLoopToTheBit:
    @pytest.mark.parametrize("embedding", [2, 3, 4, 5, 6])
    @pytest.mark.parametrize("seed", range(3))
    def test_seeded_series(self, embedding, seed):
        rng = np.random.default_rng(seed)
        n = max(math.factorial(embedding) * 5, int(rng.integers(50, 4000)))
        series = pd.Series(np.cumsum(rng.normal(0, 1, n)) * 0.3 + rng.normal(0, 1, n))
        _assert_same_entropy(series, embedding=embedding)

    def test_the_largest_embedding(self):
        """7 is the bound, and 5,040 possible patterns is where the order of
        first appearance has the most terms to get wrong."""
        rng = np.random.default_rng(4)
        _assert_same_entropy(
            pd.Series(rng.normal(0, 1, math.factorial(7) * 5)), embedding=7
        )

    @pytest.mark.parametrize("decimals", [0, 1, 2])
    def test_ties_rank_the_same_way(self, decimals):
        """Rounding makes most windows contain equal values. Which of two
        equal values `argsort` ranks first decides the pattern, so this is
        the case a different sort would get wrong."""
        rng = np.random.default_rng(decimals)
        series = pd.Series(np.round(rng.normal(0, 1, 3000), decimals))
        for embedding in (3, 5):
            _assert_same_entropy(series, embedding=embedding)

    @pytest.mark.parametrize(
        "case", ["constant", "alternating", "period_three", "minimum_length"]
    )
    def test_degenerate_series(self, case):
        series = pd.Series(
            {
                "constant": np.full(200, 0.5),
                "alternating": np.tile([0.0, 1.0], 100),
                "period_three": np.tile([3.0, 1.0, 2.0], 70),
                "minimum_length": np.random.default_rng(1).normal(0, 1, 50),
            }[case]
        )
        _assert_same_entropy(series)
        _assert_same_entropy(series, embedding=2)

    @pytest.mark.parametrize("n_bins", [-3, 1, 2, 64, 5000])
    def test_bin_counts_at_and_past_their_bounds(self, n_bins):
        series = pd.Series(np.random.default_rng(9).normal(0, 1, 600))
        _assert_same_entropy(series, n_bins=n_bins)

    def test_nan_gaps_are_dropped_the_same_way(self):
        values = np.random.default_rng(2).normal(0, 1, 400)
        values[[3, 4, 5, 100, 399]] = np.nan
        _assert_same_entropy(pd.Series(values))

    @pytest.mark.parametrize(
        "values, kwargs",
        [
            ([], {}),
            ([1.0], {}),
            ([np.nan] * 100, {}),
            ([0.1] * 60 + [np.inf] + [0.2] * 60, {}),
            (list(range(100)), {"embedding": 1}),
            (list(range(100)), {"embedding": 8}),
            (list(range(100)), {"embedding": 5}),
        ],
        ids=[
            "empty",
            "one_row",
            "all_nan",
            "inf",
            "embedding_1",
            "embedding_8",
            "short",
        ],
    )
    def test_refusals_are_the_same(self, values, kwargs):
        result = _assert_same_entropy(pd.Series(values, dtype=float), **kwargs)
        assert isinstance(result, tuple)

    def test_a_trend_has_one_pattern_and_no_entropy(self):
        result = _assert_same_entropy(pd.Series(np.arange(300.0)))
        assert result["n_patterns_observed"] == 1
        assert result["permutation_entropy"] == 0.0

    def test_an_alternating_series_has_two_patterns_in_equal_measure(self):
        result = _assert_same_entropy(pd.Series(np.tile([0.0, 1.0], 150)))
        assert result["n_patterns_observed"] == 2
        assert result["permutation_entropy"] == pytest.approx(math.log(2), rel=1e-6)

    def test_noise_is_at_maximum_permutation_entropy(self):
        """The null case."""
        series = pd.Series(np.random.default_rng(0).normal(0, 1, 5000))
        assert _assert_same_entropy(series)["permutation_normalized"] > 0.99


# ── lead-lag ────────────────────────────────────────────────────────────


def _universe(n_names, n_bars, seed, lagged=0.3):
    """A market factor that some names take a bar or two late, so pairs
    clear the correlation floor the way real equities do. Random returns
    alone give almost none, which would leave the per-pair work untested."""
    rng = np.random.default_rng(seed)
    market = rng.normal(0, 0.011, n_bars + 2)
    beta = rng.uniform(0.5, 1.5, n_names)
    late_1 = np.where(rng.random(n_names) < lagged, rng.uniform(0.1, 0.4, n_names), 0)
    late_2 = np.where(
        rng.random(n_names) < lagged / 2, rng.uniform(0.1, 0.3, n_names), 0
    )
    noise = rng.normal(0, 1, (n_bars + 2, n_names)) * rng.uniform(0.006, 0.02, n_names)
    returns = (
        market[:, None] * beta
        + np.roll(market, 1)[:, None] * beta * late_1
        + np.roll(market, 2)[:, None] * beta * late_2
        + noise
    )[2:]
    return pd.DataFrame(returns, columns=[f"S{i:02d}" for i in range(n_names)])


def _assert_same_lead_lag(frame, monkeypatch, **kwargs):
    """Both results, and the p-value calls each made: one per pair that
    clears the floor, in row order, before the sort. Equal call lists mean
    the same rows in the same order with the same t-statistics, not only
    the same top twenty."""
    calls: Dict[str, list] = {"reference": [], "new": []}
    f_sf = diagnostics._f_sf

    def watch(key):
        def recorded(statistic, d1, d2):
            calls[key].append((statistic, d1, d2))
            return f_sf(statistic, d1, d2)

        return recorded

    monkeypatch.setattr(diagnostics, "_f_sf", watch("reference"))
    expected = _outcome(_reference_lead_lag_matrix, frame, **kwargs)
    monkeypatch.setattr(diagnostics, "_f_sf", watch("new"))
    actual = _outcome(lead_lag_matrix, frame, **kwargs)
    monkeypatch.setattr(diagnostics, "_f_sf", f_sf)

    assert _identical(actual, expected), kwargs
    assert _identical(calls["new"], calls["reference"]), kwargs
    for statistic, d1, d2 in calls["new"]:
        assert type(statistic) is float and type(d1) is int and type(d2) is int
    return actual, len(calls["new"])


class TestLeadLagIsTheLoopToTheBit:
    @pytest.mark.parametrize("seed", range(3))
    @pytest.mark.parametrize("max_lag", [1, 3, 5])
    def test_factor_universes(self, seed, max_lag, monkeypatch):
        frame = _universe(25 + 10 * seed, 300 + 400 * seed, seed)
        _, n_pairs = _assert_same_lead_lag(frame, monkeypatch, max_lag=max_lag)
        assert n_pairs > 10, "the universe should put pairs past the floor"

    @pytest.mark.parametrize("min_correlation", [0.0, 0.05, 0.2, 0.9999, 1.0, -1.0])
    def test_the_floor_at_and_past_its_bounds(self, min_correlation, monkeypatch):
        """0 and below keep every finite pair; 1 keeps only a perfect one."""
        frame = _universe(12, 200, 7)
        _assert_same_lead_lag(frame, monkeypatch, min_correlation=min_correlation)

    def test_a_nan_floor_keeps_every_finite_pair(self, monkeypatch):
        _assert_same_lead_lag(_universe(8, 120, 3), monkeypatch, min_correlation=np.nan)

    def test_lags_too_long_for_the_sample_are_skipped(self, monkeypatch):
        """At 60 rows, lags past 30 leave fewer than 30 overlapping points
        and have no matrix; the vectorized pass must skip the same ones."""
        frame = _universe(6, 60, 1)
        _, n_pairs = _assert_same_lead_lag(
            frame, monkeypatch, max_lag=40, min_correlation=0.0
        )
        assert n_pairs == 6 * 5 * 30

    def test_a_constant_column_and_nan_rows(self, monkeypatch):
        """A flat column correlates as NaN and is filtered; NaN rows are
        dropped before anything is computed."""
        frame = _universe(10, 400, 5)
        frame["FLAT"] = 0.01
        frame.iloc[[3, 50, 51, 399], [0, 4]] = np.nan
        _assert_same_lead_lag(frame, monkeypatch, min_correlation=0.0)

    def test_labels_that_are_not_strings(self, monkeypatch):
        frame = _universe(6, 150, 2)
        frame.columns = [10, 2.5, ("a", 1), None, True, "x"]
        _assert_same_lead_lag(frame, monkeypatch, min_correlation=0.0)

    def test_negated_copies_tie_on_absolute_correlation(self, monkeypatch):
        """A column and its negation correlate with everything to the same
        magnitude, so the sort meets ties and keeps whatever order the rows
        were built in."""
        frame = _universe(6, 300, 4)
        frame["NEG"] = -frame["S00"]
        frame["COPY"] = frame["S00"]
        _assert_same_lead_lag(frame, monkeypatch, min_correlation=0.0)

    @pytest.mark.parametrize(
        "frame, kwargs",
        [
            (pd.DataFrame({"a": np.arange(100.0)}), {}),
            (_universe(3, 49, 0), {}),
            (_universe(3, 100, 0), {"max_lag": 0}),
        ],
        ids=["one_series", "too_short", "no_lag"],
    )
    def test_refusals_are_the_same(self, frame, kwargs, monkeypatch):
        result, _ = _assert_same_lead_lag(frame, monkeypatch, **kwargs)
        assert isinstance(result, tuple)

    def test_a_planted_leader_is_found_and_survives(self, monkeypatch):
        rng = np.random.default_rng(0)
        leader = rng.normal(0, 0.01, 803)
        frame = pd.DataFrame(
            {
                "A": leader[3:],
                "B": 0.8 * leader[1:-2] + rng.normal(0, 0.006, 800),
                "C": rng.normal(0, 0.01, 800),
            }
        )
        result, _ = _assert_same_lead_lag(frame, monkeypatch)
        top = result["strongest_pairs"][0]
        assert (top["leader"], top["follower"], top["lag"]) == ("A", "B", 2)
        assert top["survives_correction"] is True

    def test_noise_has_no_survivors(self, monkeypatch):
        """The null case."""
        noise = pd.DataFrame(np.random.default_rng(3).normal(0, 0.01, (500, 15)))
        result, _ = _assert_same_lead_lag(noise, monkeypatch)
        assert result["n_surviving"] == 0
        assert result["warnings"][0].startswith("NOTHING SURVIVED")
