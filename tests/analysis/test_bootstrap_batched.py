"""
`bootstrap_statistic` evaluates its resamples together and returns what its
per-resample loop returned.

The loop drew one moving-block resample at a time and handed each to
`_statistic`, which for a Sharpe, a Sortino or a VaR built a pandas Series
per draw. The resamples are now drawn in batches and every named statistic
is computed along the rows of a batch (see the CHANGELOG entry of
2026-10-01). The reference below is the loop, kept verbatim, and every
result is required to be the same double -- the same random draws, in the
same order, reduced in the same order. A seeded interval that moved in the
last bit would be a different interval under the same seed.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pytest

from standard_quant_tools._resampling import block_indices
from standard_quant_tools.analysis import inference
from standard_quant_tools.analysis.inference import (
    STATISTICS,
    TRADING_DAYS,
    _block_indices,
    _clean,
    _statistic,
    bootstrap_statistic,
)
from standard_quant_tools.error import ValidationError


def _reference_bootstrap_statistic(
    values: Sequence[float],
    *,
    statistic: str = "sharpe",
    n_bootstrap: int = 2000,
    block_size: Optional[int] = None,
    confidence: float = 0.95,
    periods_per_year: int = TRADING_DAYS,
    seed: int = 0,
) -> Dict[str, Any]:
    """`bootstrap_statistic` as it was before the resamples were batched,
    verbatim but for this docstring."""
    array = _clean(values, "bootstrap_statistic")
    if statistic not in STATISTICS:
        raise ValidationError(
            f"bootstrap_statistic: unknown statistic {statistic!r}. "
            f"Available: {', '.join(STATISTICS)}."
        )
    if not 0 < confidence < 1:
        raise ValidationError(f"confidence must be in (0, 1), got {confidence!r}")
    n = array.size
    if block_size is None:
        block_size = max(1, int(round(n ** (1.0 / 3.0))))
    block_size = max(1, min(int(block_size), n // 2))
    n_bootstrap = max(100, int(n_bootstrap))

    observed = _statistic(array, statistic, periods_per_year)
    rng = np.random.default_rng(int(seed))
    draws = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        draws[i] = _statistic(
            array[_block_indices(n, block_size, rng)], statistic, periods_per_year
        )
    usable = draws[np.isfinite(draws)]
    if usable.size < n_bootstrap // 2:
        raise ValidationError(
            f"bootstrap_statistic: {statistic!r} was undefined on "
            f"{n_bootstrap - usable.size} of {n_bootstrap} resamples. The "
            "statistic is not estimable on this series."
        )

    alpha = (1.0 - confidence) / 2.0
    lower = float(np.percentile(usable, alpha * 100))
    upper = float(np.percentile(usable, (1 - alpha) * 100))
    bias = float(usable.mean() - observed) if math.isfinite(observed) else None

    warnings: List[str] = []
    if block_size == 1:
        warnings.append(
            "block_size=1 is an IID bootstrap, which destroys the serial "
            "correlation in the series. The interval below is too NARROW: "
            "measured on AR(1) returns at phi = 0.8 it understates the "
            "Sharpe's interval by 2.24x and the maximum drawdown's by "
            "1.63x, while looking entirely plausible either way."
        )
    if (
        math.isfinite(observed)
        and lower <= 0 <= upper
        and statistic
        in (
            "sharpe",
            "sortino",
            "mean",
        )
    ):
        warnings.append(
            f"The {confidence:.0%} interval [{lower:.3f}, {upper:.3f}] "
            "CONTAINS ZERO. This sample does not distinguish the strategy "
            "from no edge at all, whatever the point estimate says."
        )
    if bias is not None and abs(bias) > abs(observed) * 0.15:
        warnings.append(
            f"The bootstrap mean sits {bias:+.4f} from the point estimate, "
            + (
                f"which is {abs(bias / observed):.0%} of it. "
                if observed != 0
                else "which is exactly zero. "
            )
            + "That is estimator "
            "bias, and it is large for exactly the statistics people quote: "
            "maximum drawdown is a minimum over the sample and is biased "
            "toward zero in short ones."
        )
    warnings.append(
        f"Block bootstrap with blocks of {block_size} observations, which "
        "preserves local serial correlation. The interval is a statement "
        "about THIS sample's distribution, not about a regime the sample "
        "does not contain."
    )

    return {
        "statistic": statistic,
        "n_observations": int(n),
        "n_bootstrap": int(n_bootstrap),
        "block_size": int(block_size),
        "confidence": float(confidence),
        "point_estimate": float(observed) if math.isfinite(observed) else None,
        "lower": lower,
        "upper": upper,
        "interval_width": float(upper - lower),
        "bootstrap_mean": float(usable.mean()),
        "bootstrap_std": float(usable.std(ddof=1)),
        "estimated_bias": bias,
        "contains_zero": bool(lower <= 0 <= upper),
        "warnings": warnings,
    }


def _reference_draws(values, statistic, n_bootstrap, block_size, seed):
    """The reference's resample loop on its own, with its argument handling,
    so the draws can be compared and not only what is summarised from them."""
    array = _clean(values, "bootstrap_statistic")
    n = array.size
    if block_size is None:
        block_size = max(1, int(round(n ** (1.0 / 3.0))))
    block_size = max(1, min(int(block_size), n // 2))
    rng = np.random.default_rng(int(seed))
    draws = np.empty(max(100, int(n_bootstrap)))
    for i in range(draws.size):
        draws[i] = _statistic(
            array[_block_indices(n, block_size, rng)], statistic, TRADING_DAYS
        )
    return draws


def _batched_draws(monkeypatch, values, statistic, n_bootstrap, block_size, seed):
    """Every draw `bootstrap_statistic` computes, in order, captured from the
    batches as they are reduced."""
    captured = []
    batched = inference._statistic_rows

    def record(sample, name, periods):
        out = batched(sample, name, periods)
        captured.append(out.copy())
        return out

    monkeypatch.setattr(inference, "_statistic_rows", record)
    try:
        bootstrap_statistic(
            values,
            statistic=statistic,
            n_bootstrap=n_bootstrap,
            block_size=block_size,
            seed=seed,
        )
    except ValidationError:
        pass
    monkeypatch.setattr(inference, "_statistic_rows", batched)
    return np.concatenate(captured)


def _identical(a, b) -> bool:
    """Equal and of the same type, all the way down; floats to the bit,
    which `==` alone does not see for a signed zero."""
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


def _outcome(fn, values, **kwargs):
    try:
        return fn(values, **kwargs)
    except ValidationError as error:
        return (type(error).__name__, str(error))


def _assert_same(values, **kwargs):
    expected = _outcome(_reference_bootstrap_statistic, values, **kwargs)
    actual = _outcome(bootstrap_statistic, values, **kwargs)
    assert _identical(actual, expected), (kwargs, actual, expected)
    return actual


def _ar1(n, phi=0.3, seed=0, mu=0.0005, sigma=0.012):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, n)
    out = np.empty(n)
    out[0] = noise[0]
    for t in range(1, n):
        out[t] = phi * out[t - 1] + noise[t]
    return out + mu


@pytest.fixture(autouse=True)
def _quiet_flat_resample_warnings(caplog):
    """`sharpe_ratio` logs a warning per flat resample on the reference
    path; it says nothing these tests read."""
    caplog.set_level("ERROR")


class TestEveryStatisticIsTheLoopToTheBit:
    @pytest.mark.parametrize("statistic", STATISTICS)
    @pytest.mark.parametrize("seed", [0, 1])
    def test_seeded_series_give_the_same_result(self, statistic, seed):
        n = (97, 503)[seed]
        _assert_same(
            _ar1(n, seed=seed), statistic=statistic, n_bootstrap=150, seed=seed
        )

    @pytest.mark.parametrize("statistic", STATISTICS)
    def test_the_draws_themselves_are_the_same(self, statistic, monkeypatch):
        values = _ar1(311, seed=5)
        expected = _reference_draws(values, statistic, 120, None, 9)
        actual = _batched_draws(monkeypatch, values, statistic, 120, None, 9)
        assert np.array_equal(actual, expected, equal_nan=True)
        assert np.array_equal(np.signbit(actual), np.signbit(expected))

    @pytest.mark.parametrize("statistic", STATISTICS)
    @pytest.mark.parametrize("block_size", [1, 2, 7, 10**6])
    def test_block_sizes_at_and_past_their_bounds(self, statistic, block_size):
        """1 is the IID draw, and a block larger than half the series is
        clamped to n // 2 -- the bounds the batch has to reproduce."""
        _assert_same(
            _ar1(64, seed=3),
            statistic=statistic,
            n_bootstrap=100,
            block_size=block_size,
            seed=4,
        )

    @pytest.mark.parametrize("statistic", ["sharpe", "cvar_95"])
    def test_a_series_longer_than_one_pass(self, statistic, monkeypatch):
        """Past `_PASS_ELEMENTS` a pass holds a single resample, so the
        batching degenerates to the loop's own shape."""
        n = inference._PASS_ELEMENTS + 1_001
        values = _ar1(n, seed=8)
        expected = _reference_draws(values, statistic, 100, None, 2)
        actual = _batched_draws(monkeypatch, values, statistic, 100, None, 2)
        assert np.array_equal(actual, expected)

    @pytest.mark.parametrize("statistic", STATISTICS)
    def test_the_minimum_length_and_the_minimum_draw_count(self, statistic):
        """30 observations is the floor `_clean` enforces, and fewer than
        100 resamples are raised to 100."""
        _assert_same(_ar1(30, seed=6), statistic=statistic, n_bootstrap=5, seed=1)

    def test_the_resample_draw_is_the_shared_helpers_draw(self):
        """The batch takes one `rng.integers` call for many resamples where
        the loop took one each. numpy draws them from the same stream one
        after another either way, so every row matches `block_indices` and
        the generator is left in the same state."""
        array = _ar1(401, seed=2)
        for block_size, count in ((1, 7), (3, 33), (7, 50), (200, 9)):
            rng_loop = np.random.default_rng(13)
            loop = np.stack(
                [array[block_indices(401, block_size, rng_loop)] for _ in range(count)]
            )
            rng_batch = np.random.default_rng(13)
            batch = inference._draw_resamples(array, block_size, count, rng_batch)
            assert np.array_equal(batch, loop)
            assert batch.flags["C_CONTIGUOUS"]
            assert rng_batch.integers(0, 2**31, 4).tolist() == (
                rng_loop.integers(0, 2**31, 4).tolist()
            )


class TestEdgeCasesAreTheLoops:
    @pytest.mark.parametrize("statistic", STATISTICS)
    @pytest.mark.parametrize(
        "case",
        [
            "constant",
            "zeros",
            "mostly_constant",
            "all_positive",
            "zero_or_positive",
            "ties",
            "nan_gaps",
            "tiny",
            "huge",
            "signed_zeros",
        ],
    )
    def test_degenerate_series(self, statistic, case):
        """Flat resamples are the rows the batch hands back to `_statistic`;
        a series that is all positive has no downside for a Sortino; ties,
        NaN gaps and extreme magnitudes stress the reductions."""
        rng = np.random.default_rng(17)
        values = {
            "constant": np.full(60, 0.001),
            "zeros": np.zeros(60),
            "mostly_constant": np.r_[np.full(80, 0.002), [0.01, -0.01]],
            "all_positive": np.abs(rng.normal(0.01, 0.01, 80)) + 1e-4,
            "zero_or_positive": np.r_[np.zeros(30), np.abs(rng.normal(0, 0.01, 30))],
            "ties": np.round(rng.normal(0, 0.01, 200), 3),
            "nan_gaps": np.r_[
                rng.normal(0, 0.01, 40), [np.nan] * 5, rng.normal(0, 0.01, 40)
            ],
            "tiny": rng.normal(0, 1, 100) * 1e-170,
            "huge": rng.normal(0, 1, 100) * 1e200,
            "signed_zeros": np.r_[np.full(40, -0.0), np.full(40, 0.0), [0.01]],
        }[case]
        with np.errstate(all="ignore"):
            _assert_same(values, statistic=statistic, n_bootstrap=100, seed=3)

    @pytest.mark.parametrize(
        "values",
        [[], [0.01], [np.nan] * 40, [0.01] * 20 + [np.inf] + [0.02] * 20],
        ids=["empty", "one_row", "all_nan", "inf"],
    )
    def test_unusable_input_is_refused_identically(self, values):
        expected = _outcome(_reference_bootstrap_statistic, values)
        actual = _outcome(bootstrap_statistic, values)
        assert isinstance(expected, tuple)
        assert actual == expected

    @pytest.mark.parametrize("statistic", ["sharpe", "skew"])
    def test_a_sharpe_pandas_cannot_be_matched_on_is_left_to_pandas(
        self, statistic, monkeypatch
    ):
        """With bottleneck active, pandas' `Series.std` is not numpy's sum,
        and the batch hands every Sharpe row back to `_statistic`. Forced
        here, since the suite runs without bottleneck."""
        monkeypatch.setattr(inference, "_pandas_std_is_numpy", lambda: False)
        _assert_same(_ar1(120, seed=4), statistic=statistic, n_bootstrap=100)


class TestKnownAnswers:
    def test_a_series_that_only_rises_has_a_win_rate_of_one_on_every_draw(self):
        values = np.abs(_ar1(90, seed=1)) + 1e-4
        result = _assert_same(values, statistic="win_rate", n_bootstrap=100)
        assert result["lower"] == result["upper"] == 1.0
        assert result["bootstrap_std"] == 0.0

    def test_a_driftless_series_has_a_mean_interval_containing_zero(self):
        """The null case: no drift, so the interval must not exclude zero."""
        rng = np.random.default_rng(21)
        values = rng.normal(0.0, 0.01, 400)
        values -= values.mean()
        result = _assert_same(values, statistic="mean", n_bootstrap=300)
        assert result["contains_zero"] is True


class TestABiasAgainstAZeroEstimateIsReported:
    """The bias warning divided the bias by the point estimate, so a point
    estimate of exactly zero with any bias raised ZeroDivisionError instead
    of returning the interval (see the CHANGELOG entry of 2026-10-01)."""

    def test_a_zero_point_estimate_returns_the_interval_and_says_so(self):
        values = np.concatenate([np.zeros(30), np.full(30, 0.01)])
        result = bootstrap_statistic(values, statistic="var_95", block_size=10000)
        assert result["point_estimate"] == 0.0
        assert result["estimated_bias"] != 0.0
        assert any("which is exactly zero" in w for w in result["warnings"])

    def test_a_nonzero_point_estimate_still_reports_the_share(self):
        """The null case: the message is unchanged when the share exists."""
        values = _ar1(60, seed=3)
        result = bootstrap_statistic(values, statistic="max_drawdown", block_size=10)
        bias_lines = [w for w in result["warnings"] if "bootstrap mean sits" in w]
        assert all("% of it." in w for w in bias_lines)
        assert not any("exactly zero" in w for w in result["warnings"])
