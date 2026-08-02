#pragma once

#include "sqt/platform.hpp"

#include <cstddef>
#include <vector>

namespace sqt {

/**
 * Moving-block bootstrap Monte Carlo forward simulation.
 *
 * For each of n_simulations independent paths: draw
 * ceil(horizon_days/block_size) block-start indices uniformly from
 * [0, hist_n - block_size], concatenate the resulting blocks of `values`,
 * truncate to horizon_days, and compute
 * initial_capital * cumprod(1 + resampled_returns) for that path.
 *
 * Matches simulate_forward_paths's resampling loop in
 * backtest/monte_carlo.py exactly (same block-draw/concatenate/cumprod
 * logic) — the terminal-distribution and percentile-band statistics built
 * from the returned paths stay in NumPy on the Python side, since those
 * are already vectorized and not part of this port.
 *
 * RNG: each path is seeded independently (derived from `seed`/path index),
 * so paths have no shared mutable state — safe to compute in parallel with
 * no locking. This does NOT reproduce NumPy's PCG64 bit stream: the same
 * seed produces different concrete numbers than the pure-Python fallback.
 * Reproducibility is guaranteed only WITHIN this C++ path — same seed and
 * inputs give bit-identical output across repeat calls.
 *
 * @param values          Historical per-bar returns, length hist_n.
 * @param hist_n          Number of historical observations.
 * @param horizon_days    Forward bars per path.
 * @param n_simulations   Number of independent paths.
 * @param block_size      Bootstrap block length in bars; must be in
 *                        (0, hist_n].
 * @param initial_capital Starting capital for every path.
 * @param seed            RNG seed; ignored if has_seed is false.
 * @param has_seed        If false, a nondeterministic seed is drawn (matches
 *                        np.random.default_rng(None) behavior).
 * @returns  Flat row-major array, length n_simulations * horizon_days:
 *           paths_flat[i*horizon_days + t] = equity of path i at bar t.
 *           Returns an empty vector if horizon_days<=0, n_simulations<=0,
 *           hist_n==0, block_size<=0, block_size>hist_n, or
 *           initial_capital is non-positive/non-finite — the Python caller
 *           already validates all of these before calling, so this is
 *           defense-in-depth for a direct binding call, not the primary
 *           validation path.
 */
std::vector<double> simulate_forward_paths(
    const double* values,
    std::size_t   hist_n,
    int           horizon_days,
    int           n_simulations,
    int           block_size,
    double        initial_capital,
    unsigned long long seed,
    bool          has_seed);

/**
 * Buffer-writing form of simulate_forward_paths(). `out` must have length
 * n_simulations*horizon_days -- unlike the vector-returning form (which
 * signals "invalid input" via an empty vector), this returns false and
 * leaves `out`'s contents undefined/untouched when the input is invalid,
 * since a pre-allocated fixed-size buffer can't itself signal "empty".
 * Returns true iff `out` was actually written. SQT_RESTRICT: `out` is
 * always freshly allocated at the one call site (bindings.cpp), never
 * aliased with `values`.
 */
bool simulate_forward_paths_into(
    const double* SQT_RESTRICT values,
    std::size_t   hist_n,
    int           horizon_days,
    int           n_simulations,
    int           block_size,
    double        initial_capital,
    unsigned long long seed,
    bool          has_seed,
    double* SQT_RESTRICT       out);

/**
 * Terminal-only variant of simulate_forward_paths(): identical RNG/
 * block-bootstrap core, but returns only each path's TERMINAL equity
 * (length n_simulations) instead of the full (n_simulations x
 * horizon_days) path matrix. For memory-constrained large-simulation use
 * where only the terminal distribution is needed -- e.g. n_simulations=
 * 1,000,000 x horizon_days=252 would be a ~2GB full path matrix that this
 * variant avoids allocating entirely.
 *
 * For identical (seed, inputs), result[i] == simulate_forward_paths(...)
 * [i*horizon_days + (horizon_days-1)] exactly (both draw from the
 * identical RNG sequence and bootstrap logic; only the write target
 * differs).
 *
 * @returns  Length n_simulations. Empty vector under the same invalid-
 *           input conditions as simulate_forward_paths().
 */
std::vector<double> simulate_forward_paths_terminal(
    const double* values,
    std::size_t   hist_n,
    int           horizon_days,
    int           n_simulations,
    int           block_size,
    double        initial_capital,
    unsigned long long seed,
    bool          has_seed);

/**
 * Buffer-writing form of simulate_forward_paths_terminal(). `out_terminal`
 * must have length n_simulations. Same invalid-input/return-value
 * conventions as simulate_forward_paths_into().
 */
bool simulate_forward_paths_terminal_into(
    const double* SQT_RESTRICT values,
    std::size_t   hist_n,
    int           horizon_days,
    int           n_simulations,
    int           block_size,
    double        initial_capital,
    unsigned long long seed,
    bool          has_seed,
    double* SQT_RESTRICT       out_terminal);

}  // namespace sqt
