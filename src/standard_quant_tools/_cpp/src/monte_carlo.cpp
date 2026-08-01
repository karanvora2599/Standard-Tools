#include "sqt/monte_carlo.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <random>

#ifdef SQT_HAS_OPENMP
#include <omp.h>
#endif

namespace sqt {

namespace {

// splitmix64 (Vigna) — cheap, well-distributed mixing function used to
// derive an independent RNG seed per simulated path from a single base
// seed, so paths need no shared mutable state and can run in parallel
// (see performance_insights.md's OpenMP follow-on) with no locking.
std::uint64_t splitmix64(std::uint64_t& state) {
    std::uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

}  // namespace

std::vector<double> simulate_forward_paths(
    const double* values,
    std::size_t   hist_n,
    int           horizon_days,
    int           n_simulations,
    int           block_size,
    double        initial_capital,
    unsigned long long seed,
    bool          has_seed)
{
    if (horizon_days <= 0 || n_simulations <= 0) return {};

    const std::size_t out_size =
        static_cast<std::size_t>(n_simulations) * static_cast<std::size_t>(horizon_days);
    std::vector<double> result;

    if (hist_n == 0 || block_size <= 0 ||
        static_cast<std::size_t>(block_size) > hist_n ||
        !(initial_capital > 0.0) || !std::isfinite(initial_capital)) {
        return result;  // empty — signals "invalid input" to the caller
    }

    result.assign(out_size, 0.0);

    const std::size_t block = static_cast<std::size_t>(block_size);
    const std::size_t horizon = static_cast<std::size_t>(horizon_days);
    const std::size_t max_start = hist_n - block;

    std::uint64_t base_seed = has_seed
        ? static_cast<std::uint64_t>(seed)
        : static_cast<std::uint64_t>(
              std::chrono::steady_clock::now().time_since_epoch().count());

    // `gen`/`dist` are declared once *per thread* (inside the `#pragma omp
    // parallel` block below, not inside the `for` loop it wraps) rather
    // than freshly constructed on every single path iteration — each
    // thread's own `gen` is fully reseeded (`gen.seed(path_seed)`) at the
    // start of every path it handles, so this is still exactly as
    // reproducible as constructing a fresh generator each time (seeding
    // fully reinitializes a Mersenne Twister's state either way) and no
    // two threads ever touch the same `gen` instance -- no data race, just
    // one seed operation per path instead of one full object construction.
    // `resampled` is gone entirely: each block's sampled values are
    // consumed directly into `row[]` as they're drawn (bounded by
    // `t < horizon`, the same truncation semantics the old `pos <
    // resampled.size()` guard gave the last, possibly-partial block) —
    // no intermediate heap buffer, no second pass reading it back. With
    // 200,000 paths, this is 200,000 fewer heap allocations and frees.
#ifdef SQT_HAS_OPENMP
    #pragma omp parallel if(n_simulations > 1)
#endif
    {
        std::mt19937_64 gen;
        std::uniform_int_distribution<std::size_t> dist(0, max_start);

#ifdef SQT_HAS_OPENMP
        #pragma omp for schedule(static)
#endif
        for (int i = 0; i < n_simulations; ++i) {
            // Derive this path's own seed from the base seed and its
            // index — independent of every other path's RNG state, so no
            // shared mutable RNG state and no locking is needed across
            // threads regardless of which thread ends up running path i.
            std::uint64_t mix_state = base_seed ^ (static_cast<std::uint64_t>(i) * 0x9E3779B97F4A7C15ULL + 1);
            const std::uint64_t path_seed = splitmix64(mix_state);
            gen.seed(path_seed);

            double equity = initial_capital;
            double* row = result.data() + static_cast<std::size_t>(i) * horizon;
            std::size_t t = 0;
            while (t < horizon) {
                const std::size_t start = dist(gen);
                for (std::size_t k = 0; k < block && t < horizon; ++k, ++t) {
                    equity *= (1.0 + values[start + k]);
                    row[t] = equity;
                }
            }
        }
    }

    return result;
}

}  // namespace sqt
