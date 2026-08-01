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
    const std::size_t n_blocks = (horizon + block - 1) / block;  // ceil(horizon/block)

    std::uint64_t base_seed = has_seed
        ? static_cast<std::uint64_t>(seed)
        : static_cast<std::uint64_t>(
              std::chrono::steady_clock::now().time_since_epoch().count());

    // `resampled` is declared *inside* the loop body (not hoisted above it)
    // so each iteration gets its own local buffer — required for
    // correctness once the loop runs under #pragma omp parallel for below:
    // a buffer shared across iterations would be a data race the moment
    // more than one thread is active. Same reasoning for `gen`/`dist`,
    // which are already loop-local.
#ifdef SQT_HAS_OPENMP
    #pragma omp parallel for schedule(static) if(n_simulations > 1)
#endif
    for (int i = 0; i < n_simulations; ++i) {
        // Derive this path's own seed from the base seed and its index —
        // independent of every other path's RNG state, so no shared
        // mutable RNG and no locking is needed across threads.
        std::uint64_t mix_state = base_seed ^ (static_cast<std::uint64_t>(i) * 0x9E3779B97F4A7C15ULL + 1);
        const std::uint64_t path_seed = splitmix64(mix_state);
        std::mt19937_64 gen(path_seed);
        std::uniform_int_distribution<std::size_t> dist(0, max_start);

        std::vector<double> resampled(n_blocks * block);
        std::size_t pos = 0;
        for (std::size_t b = 0; b < n_blocks; ++b) {
            const std::size_t start = dist(gen);
            for (std::size_t k = 0; k < block && pos < resampled.size(); ++k, ++pos) {
                resampled[pos] = values[start + k];
            }
        }

        double equity = initial_capital;
        double* row = result.data() + static_cast<std::size_t>(i) * horizon;
        for (std::size_t t = 0; t < horizon; ++t) {
            equity *= (1.0 + resampled[t]);
            row[t] = equity;
        }
    }

    return result;
}

}  // namespace sqt
