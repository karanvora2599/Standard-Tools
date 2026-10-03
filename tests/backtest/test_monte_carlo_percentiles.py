"""
`simulate_forward_paths` and `simulate_forward_paths_terminal` return the
doubles they returned when every percentile was its own `np.percentile`
call (see the CHANGELOG entry of 2026-10-02).

The three per-day equity bands are now one call on a transposed copy, and
the terminal 5th and 95th percentiles one call. A call with several
percentiles partitions once at every position any of them needs; that puts
the same value at each position, but values that compare equal -- -0.0 and
+0.0, or two NaNs -- can land in a different order, and the interpolated
result can then differ in its bits, though only when it is zero or NaN.
Those results are recomputed as their own call. The references below are
the functions as they were, kept verbatim, and every result is required to
be identical -- the same doubles, the sign of zero included -- not close.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.backtest import monte_carlo
from standard_quant_tools.backtest.monte_carlo import (
    _band_percentiles,
    _terminal_statistics,
    simulate_forward_paths,
    simulate_forward_paths_terminal,
)
from standard_quant_tools.error import ValidationError
from standard_quant_tools.validation import require_finite_array


def _reference_paths(values, horizon_days, n_simulations, block_size, capital, seed):
    """The path matrix, from whichever backend the module is using."""
    if monte_carlo.HAS_CPP and monte_carlo._cpp_core is not None:
        return monte_carlo._cpp_core.simulate_forward_paths(
            values, horizon_days, n_simulations, block_size, capital, seed
        )
    rng = np.random.default_rng(seed)
    n = len(values)
    paths = np.empty((n_simulations, horizon_days), dtype=float)
    for i in range(n_simulations):
        resampled = values[monte_carlo.block_indices(n, block_size, rng, horizon_days)]
        paths[i, :] = capital * np.cumprod(1.0 + resampled)
    return paths


def _reference_terminal(values, horizon_days, n_simulations, block_size, capital, seed):
    if monte_carlo.HAS_CPP and monte_carlo._cpp_core is not None:
        return monte_carlo._cpp_core.simulate_forward_paths_terminal(
            values, horizon_days, n_simulations, block_size, capital, seed
        )
    rng = np.random.default_rng(seed)
    n = len(values)
    terminal = np.empty(n_simulations, dtype=float)
    for i in range(n_simulations):
        resampled = values[monte_carlo.block_indices(n, block_size, rng, horizon_days)]
        terminal[i] = capital * np.prod(1.0 + resampled)
    return terminal


def _validate(returns, horizon_days, n_simulations, block_size, capital, who):
    n = len(returns)
    if n == 0:
        raise ValidationError("returns is empty")
    if horizon_days <= 0:
        raise ValidationError(f"horizon_days must be > 0, got {horizon_days}")
    if n_simulations <= 0:
        raise ValidationError(f"n_simulations must be > 0, got {n_simulations}")
    if capital <= 0:
        raise ValidationError(f"initial_capital must be > 0, got {capital}")
    if block_size <= 0 or block_size > n:
        raise ValidationError(f"block_size must be in (0, {n}], got {block_size}")
    values = returns.to_numpy(dtype=float)
    require_finite_array(values, "returns", who)
    return values


def _reference_simulate_forward_paths(
    returns: pd.Series,
    horizon_days: int,
    n_simulations: int = 1000,
    block_size: int = 20,
    initial_capital: float = 10_000.0,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """`simulate_forward_paths` as it was: validation and the path matrix
    as above, then its statistics verbatim."""
    values = _validate(
        returns,
        horizon_days,
        n_simulations,
        block_size,
        initial_capital,
        "simulate_forward_paths",
    )
    paths = _reference_paths(
        values, horizon_days, n_simulations, block_size, initial_capital, seed
    )

    terminal = paths[:, -1]
    terminal_returns = terminal / initial_capital - 1.0

    terminal_median = float(np.median(terminal))
    terminal_p5 = float(np.percentile(terminal, 5.0))
    terminal_p95 = float(np.percentile(terminal, 95.0))
    prob_loss = float(np.mean(terminal < initial_capital))

    var_95 = float(-np.percentile(terminal_returns, 5.0))
    tail = terminal_returns[terminal_returns <= np.percentile(terminal_returns, 5.0)]
    cvar_95 = float(-tail.mean()) if len(tail) > 0 else var_95

    equity_band_p5 = np.percentile(paths, 5.0, axis=0).tolist()
    equity_band_p50 = np.percentile(paths, 50.0, axis=0).tolist()
    equity_band_p95 = np.percentile(paths, 95.0, axis=0).tolist()

    return {
        "terminal_median": terminal_median,
        "terminal_p5": terminal_p5,
        "terminal_p95": terminal_p95,
        "prob_loss": prob_loss,
        "terminal_var_95": var_95,
        "terminal_cvar_95": cvar_95,
        "equity_band_p5": equity_band_p5,
        "equity_band_p50": equity_band_p50,
        "equity_band_p95": equity_band_p95,
    }


def _reference_simulate_forward_paths_terminal(
    returns: pd.Series,
    horizon_days: int,
    n_simulations: int = 1000,
    block_size: int = 20,
    initial_capital: float = 10_000.0,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """`simulate_forward_paths_terminal` as it was."""
    values = _validate(
        returns,
        horizon_days,
        n_simulations,
        block_size,
        initial_capital,
        "simulate_forward_paths_terminal",
    )
    terminal = _reference_terminal(
        values, horizon_days, n_simulations, block_size, initial_capital, seed
    )

    terminal_returns = terminal / initial_capital - 1.0

    terminal_median = float(np.median(terminal))
    terminal_p5 = float(np.percentile(terminal, 5.0))
    terminal_p95 = float(np.percentile(terminal, 95.0))
    prob_loss = float(np.mean(terminal < initial_capital))

    var_95 = float(-np.percentile(terminal_returns, 5.0))
    tail = terminal_returns[terminal_returns <= np.percentile(terminal_returns, 5.0)]
    cvar_95 = float(-tail.mean()) if len(tail) > 0 else var_95

    return {
        "terminal_median": terminal_median,
        "terminal_p5": terminal_p5,
        "terminal_p95": terminal_p95,
        "prob_loss": prob_loss,
        "terminal_var_95": var_95,
        "terminal_cvar_95": cvar_95,
    }


def _identical(a, b) -> bool:
    """Equal and of the same type, all the way down; floats to the bit."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return list(a) == list(b) and all(_identical(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_identical(x, y) for x, y in zip(a, b))
    if isinstance(a, float):
        if math.isnan(a) or math.isnan(b):
            return math.isnan(a) and math.isnan(b)
        return a == b and math.copysign(1.0, a) == math.copysign(1.0, b)
    return a == b


def _bits(x) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(x, dtype=np.float64)).view(np.uint64)


def _bit(x) -> int:
    return int(np.array(x, dtype=np.float64).view(np.uint64))


def _outcome(fn, *args, **kwargs):
    with warnings.catch_warnings():
        # inf - inf inside numpy's interpolation warns; both versions do it.
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            return fn(*args, **kwargs)
        except ValidationError as error:
            return ("ValidationError", str(error))


def _assert_both_identical(returns, *args, **kwargs):
    expected = _outcome(_reference_simulate_forward_paths, returns, *args, **kwargs)
    actual = _outcome(simulate_forward_paths, returns, *args, **kwargs)
    assert _identical(actual, expected), (args, kwargs)
    expected_t = _outcome(
        _reference_simulate_forward_paths_terminal, returns, *args, **kwargs
    )
    actual_t = _outcome(simulate_forward_paths_terminal, returns, *args, **kwargs)
    assert _identical(actual_t, expected_t), (args, kwargs)
    return actual, actual_t


@pytest.fixture(params=["native", "python"])
def backend(request, monkeypatch):
    """Both backends, the reference and the function always on the same one."""
    if request.param == "native":
        if not monte_carlo.HAS_CPP:
            pytest.skip("the compiled extension is not built")
    else:
        monkeypatch.setattr(monte_carlo, "HAS_CPP", False)
    return request.param


def _signed_zero_returns(n=40, seed=0):
    """-1 takes an equity path to zero; a later return below -1 flips the
    sign of that zero. The bands then sit on columns of mixed -0.0 and
    +0.0, which is exactly where tie order shows."""
    rng = np.random.default_rng(seed)
    pool = np.array([-1.0, -2.0, -1.5, -2.0, 0.01, -0.02])
    return pd.Series(pool[rng.integers(0, len(pool), n)])


class TestTheResultIsTheSeparateCallsToTheBit:
    @pytest.mark.parametrize("seed", range(6))
    @pytest.mark.parametrize(
        "horizon, n_sims, block",
        [(1, 1, 1), (1, 7, 3), (5, 2, 2), (30, 200, 20), (60, 1000, 20), (252, 333, 5)],
    )
    def test_seeded_ordinary_returns(self, backend, seed, horizon, n_sims, block):
        rng = np.random.default_rng(100 + seed)
        returns = pd.Series(rng.normal(0.0004, 0.012, 300))
        if backend == "python" and n_sims * horizon > 20_000:
            n_sims = 20_000 // horizon
        _assert_both_identical(returns, horizon, n_sims, block, 10_000.0, seed)

    @pytest.mark.parametrize("seed", range(8))
    def test_paths_at_the_default_size(self, seed):
        rng = np.random.default_rng(seed)
        returns = pd.Series(rng.standard_t(3, 750) * 0.01)
        _assert_both_identical(returns, 60, 1000, 20, 25_000.0, seed)

    def test_a_large_simulation(self):
        if not monte_carlo.HAS_CPP:
            pytest.skip("the compiled extension is not built")
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0.0004, 0.012, 2106))
        _assert_both_identical(returns, 60, 50_000, 20, 10_000.0, 7)

    @pytest.mark.parametrize("seed", range(10))
    def test_planted_signed_zeros(self, backend, seed):
        """The case the fallback exists for. The reference's bands and
        terminal percentiles must actually contain -0.0 for this to test
        anything, so that is asserted too."""
        returns = _signed_zero_returns(seed=seed)
        full, _ = _assert_both_identical(returns, 24, 400, 1, 10_000.0, seed)
        values = np.concatenate(
            [
                full["equity_band_p5"],
                full["equity_band_p50"],
                full["equity_band_p95"],
                [full["terminal_p5"], full["terminal_median"], full["terminal_p95"]],
            ]
        )
        assert np.any((values == 0.0) & np.signbit(values))

    @pytest.mark.parametrize("seed", range(6))
    def test_planted_nan_and_infinity(self, backend, seed):
        """A finite 1e300 return overflows an equity path to inf; a -1 after
        it is inf * 0, NaN, which the path then carries."""
        rng = np.random.default_rng(seed)
        pool = np.array([1e300, -1.0, 0.01, -2.0, 0.02])
        returns = pd.Series(pool[rng.integers(0, len(pool), 30)])
        full, terminal = _assert_both_identical(returns, 10, 300, 1, 10_000.0, seed)
        bands = np.array(full["equity_band_p50"] + full["equity_band_p95"])
        assert np.isnan(bands).any() or np.isinf(bands).any()

    def test_flat_paths(self, backend):
        """Every path the same: each band is the capital, the losses zero."""
        full, terminal = _assert_both_identical(
            pd.Series(np.zeros(50)), 20, 100, 5, 10_000.0, 1
        )
        assert full["equity_band_p50"] == [10_000.0] * 20
        assert full["terminal_var_95"] == 0.0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"horizon_days": 0},
            {"n_simulations": 0},
            {"initial_capital": -1.0},
            {"block_size": 0},
            {"block_size": 51},
        ],
    )
    def test_refusals_are_the_same(self, kwargs):
        args = {
            "horizon_days": 10,
            "n_simulations": 10,
            "block_size": 5,
            "initial_capital": 1.0,
            "seed": 0,
        }
        args.update(kwargs)
        full, terminal = _assert_both_identical(pd.Series(np.zeros(50)), **args)
        assert full[0] == "ValidationError" and terminal[0] == "ValidationError"


def _matrices(rng):
    """Synthetic (paths x days) matrices: ties, both signed zeros, values
    spanning the exponent range, NaN, infinities, one and two rows."""
    for shape in [(1, 3), (2, 5), (3, 1), (7, 4), (20, 60), (101, 13), (1000, 60)]:
        yield rng.normal(1e4, 300.0, shape)
        yield np.round(rng.normal(0, 3, shape))
        wide = rng.normal(0, 1, shape) * 10.0 ** rng.integers(-300, 300, shape)
        wide[rng.random(shape) < 0.1] = -0.0
        wide[rng.random(shape) < 0.1] = 0.0
        yield wide
        zeros = np.where(rng.random(shape) < 0.5, -0.0, 0.0)
        zeros[rng.random(shape) < 0.3] = 1.0
        yield zeros
        yield np.full(shape, 12345.678)
        holes = rng.normal(1e4, 300.0, shape)
        holes[rng.random(shape) < 0.05] = np.nan
        holes[rng.random(shape) < 0.05] = np.inf
        holes[rng.random(shape) < 0.05] = -np.inf
        yield holes


class TestTheHelpersAgainstTheSeparateCalls:
    """The helpers on matrices the kernel would rarely produce, where tie
    order between -0.0 and +0.0 is common rather than rare."""

    @pytest.mark.parametrize("seed", range(4))
    def test_bands(self, seed):
        rng = np.random.default_rng(seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for paths in _matrices(rng):
                original = paths.copy()
                expected = np.stack(
                    [np.percentile(paths, q, axis=0) for q in (5.0, 50.0, 95.0)]
                )
                got = _band_percentiles(paths)
                assert np.array_equal(_bits(got), _bits(expected)), paths.shape
                # The caller's matrix is not the one partitioned in place.
                assert np.array_equal(_bits(paths), _bits(original))

    @pytest.mark.parametrize("seed", range(4))
    def test_terminal_statistics(self, seed):
        rng = np.random.default_rng(seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for paths in _matrices(rng):
                terminal = paths[:, -1]
                returns = terminal / 10_000.0 - 1.0
                got = _terminal_statistics(terminal, 10_000.0)
                assert _bit(got["terminal_median"]) == _bit(np.median(terminal))
                for key, q in (("terminal_p5", 5.0), ("terminal_p95", 95.0)):
                    assert _bit(got[key]) == _bit(np.percentile(terminal, q))
                assert _bit(got["terminal_var_95"]) == _bit(
                    -np.percentile(returns, 5.0)
                )

    def test_a_consolidated_call_differs_only_where_the_fallback_looks(self):
        """The premise the fallback rests on. Wherever one call with three
        percentiles disagrees in its bits with three calls, the value is
        zero or NaN. On numpy 2.0 and 2.4 the unguarded call disagrees on
        dozens of these matrices; the guarded one, `test_bands`, on none."""
        rng = np.random.default_rng(11)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for _ in range(3):
                for paths in _matrices(rng):
                    separate = np.stack(
                        [np.percentile(paths, q, axis=0) for q in (5.0, 50.0, 95.0)]
                    )
                    combined = np.percentile(
                        np.ascontiguousarray(paths.T),
                        [5.0, 50.0, 95.0],
                        axis=1,
                        overwrite_input=True,
                    )
                    disagree = _bits(separate) != _bits(combined)
                    assert np.all(
                        (combined[disagree] == 0.0) | np.isnan(combined[disagree])
                    )
