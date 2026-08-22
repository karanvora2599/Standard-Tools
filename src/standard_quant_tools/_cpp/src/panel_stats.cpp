#include "sqt/panel_stats.hpp"

#include "sqt/omp_policy.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <new>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace sqt {
namespace {

/**
 * pandas' linear-interpolated quantile over an already-sorted, finite range.
 *
 * `Series.quantile(q)` places the result at h = (n-1)*q and interpolates
 * between the two neighbouring order statistics. Rounding h to an index --
 * which is what a bare nth_element gives -- disagrees with pandas on almost
 * every column that is not exactly (n-1)*q integral, so the interpolation
 * is the point of this helper rather than a refinement of it.
 */
double interpolated_quantile(std::vector<double>& sorted_scratch,
                             std::size_t n,
                             double q) {
    if (n == 0) return std::nan("");
    if (n == 1) return sorted_scratch[0];

    const double h = static_cast<double>(n - 1) * q;
    double lower_pos = std::floor(h);
    const double frac = h - lower_pos;
    auto lower_index = static_cast<std::size_t>(lower_pos);
    if (lower_index >= n - 1) return sorted_scratch[n - 1];

    // Only the two order statistics that bracket h are needed, so a full
    // sort is wasted work: two nth_element passes are O(n) each. The second
    // searches only the tail the first left above the pivot, which is where
    // the upper neighbour must be.
    std::nth_element(sorted_scratch.begin(),
                     sorted_scratch.begin() + static_cast<std::ptrdiff_t>(lower_index),
                     sorted_scratch.begin() + static_cast<std::ptrdiff_t>(n));
    const double low_value = sorted_scratch[lower_index];
    if (frac == 0.0) return low_value;

    const double high_value =
        *std::min_element(sorted_scratch.begin() +
                              static_cast<std::ptrdiff_t>(lower_index + 1),
                          sorted_scratch.begin() + static_cast<std::ptrdiff_t>(n));
    return low_value + frac * (high_value - low_value);
}

/**
 * Pairwise summation, matching what numpy's reduction actually does.
 *
 * This is not a refinement, it is a correctness requirement for agreeing
 * with the Python path. `Series.mean()` dispatches to numpy, which sums
 * pairwise: error grows as O(log n) rather than the O(n) of a sequential
 * accumulator. Measured on 50,000 return-scale values, the sequential form
 * disagreed with numpy in the 12th significant digit, which propagated to
 * 3.8e-14 on the standardized output -- small in absolute terms and still
 * far larger than this kernel is allowed to be wrong by.
 *
 * The base case uses eight independent accumulators, which is both numpy's
 * layout and what lets the compiler keep the chain in vector registers
 * instead of serializing on one dependency.
 *
 * `value_at(i)` supplies the i-th term, so the same routine serves the
 * plain sum and the sum of squared deviations without materializing either.
 */
template <typename F>
double pairwise_sum(const F& value_at, std::size_t begin, std::size_t n) {
    constexpr std::size_t kBlock = 128;
    if (n == 0) return 0.0;
    if (n <= kBlock) {
        double acc[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        std::size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            for (int k = 0; k < 8; ++k) acc[k] += value_at(begin + i + k);
        }
        double tail = 0.0;
        for (; i < n; ++i) tail += value_at(begin + i);
        return ((acc[0] + acc[1]) + (acc[2] + acc[3])) +
               ((acc[4] + acc[5]) + (acc[6] + acc[7])) + tail;
    }
    // Split on an 8-aligned boundary so each half keeps the same unrolled
    // base case numpy would have used.
    std::size_t half = (n / 2) & ~static_cast<std::size_t>(7);
    if (half == 0) half = n / 2;
    return pairwise_sum(value_at, begin, half) +
           pairwise_sum(value_at, begin + half, n - half);
}

}  // namespace

bool fit_preprocess_stats(const double* values,
                          std::size_t n_rows,
                          std::size_t n_cols,
                          double q_low,
                          double q_high,
                          PreprocessStats out) {
    if (n_cols == 0) return true;
    // A zero-row panel still has columns, and each of them is "entirely
    // missing" rather than absent -- so the outputs must be filled with the
    // all-NaN rule, not left at whatever the caller allocated. Returning
    // early on a null pointer got this wrong: an empty std::vector's data()
    // is null, so a legitimate (0, n_cols) panel silently kept uninitialized
    // stats and the caller then divided by them.
    const bool readable = (values != nullptr) && n_rows > 0;

    // A per-column failure flag rather than an exception: an exception
    // escaping an OpenMP structured block is undefined behaviour, and this
    // file is compiled with OpenMP on. Same containment the other parallel
    // kernels here use.
    bool alloc_error = false;

    #pragma omp parallel for schedule(guided) reduction(|| : alloc_error) \
        if (sqt::omp_policy::worth_parallel(n_cols, n_rows)) \
        num_threads(sqt::omp_policy::max_threads() > 0 \
                        ? sqt::omp_policy::max_threads() : omp_get_max_threads())
    for (std::ptrdiff_t col = 0; col < static_cast<std::ptrdiff_t>(n_cols); ++col) {
        const auto c = static_cast<std::size_t>(col);

        // Gather the column into a contiguous, thread-local buffer. The
        // panel is row-major, so a column has stride n_cols; nth_element
        // needs to permute it anyway, and it must not touch the caller's
        // data. Copying once and working contiguously beats striding
        // through the original twice.
        std::vector<double> scratch;
        try {
            scratch.resize(n_rows);
        } catch (const std::bad_alloc&) {
            alloc_error = true;
            continue;
        }

        std::size_t finite_count = 0;
        if (readable) {
            for (std::size_t row = 0; row < n_rows; ++row) {
                const double v = values[row * n_cols + c];
                // NaN is skipped, matching Series.quantile/std. Infinities
                // are NOT: pandas treats only NaN as missing, so an inf is a
                // real (if pathological) order statistic and must sort as
                // one.
                if (!std::isnan(v)) scratch[finite_count++] = v;
            }
        }

        if (finite_count == 0) {
            out.lo[c] = std::nan("");
            out.hi[c] = std::nan("");
            out.mean[c] = std::nan("");
            // std=1.0 keeps the caller's division defined; fit_preprocessing
            // makes the same substitution in Python.
            out.stdev[c] = 1.0;
            continue;
        }

        const double lo = interpolated_quantile(scratch, finite_count, q_low);
        const double hi = interpolated_quantile(scratch, finite_count, q_high);
        out.lo[c] = lo;
        out.hi[c] = hi;

        // Mean and ddof=1 standard deviation of the CLIPPED column. Two
        // passes rather than one from sum and sum-of-squares: on
        // return-scale data the one-pass form differences two nearly equal
        // large numbers and loses most of its significant digits, the same
        // trap the cross-sectional IC had to avoid. Both passes sum
        // pairwise, which is what numpy does and therefore what agreeing
        // with pandas requires.
        const double* buffer = scratch.data();
        const double mean =
            pairwise_sum(
                [&](std::size_t i) {
                    return std::min(std::max(buffer[i], lo), hi);
                },
                0, finite_count) /
            static_cast<double>(finite_count);
        out.mean[c] = mean;

        if (finite_count < 2) {
            out.stdev[c] = 1.0;
            continue;
        }
        const double sum_sq = pairwise_sum(
            [&](std::size_t i) {
                const double d = std::min(std::max(buffer[i], lo), hi) - mean;
                return d * d;
            },
            0, finite_count);
        const double variance = sum_sq / static_cast<double>(finite_count - 1);
        const double stdev = std::sqrt(variance);
        // A constant column has no dispersion to divide by. 1.0 leaves it at
        // its centered value (zero) rather than producing inf or NaN, which
        // is what fit_preprocessing does.
        out.stdev[c] = (stdev > 0.0 && std::isfinite(stdev)) ? stdev : 1.0;
    }

    return !alloc_error;
}

void apply_preprocess_stats(const double* values,
                            std::size_t n_rows,
                            std::size_t n_cols,
                            const PreprocessStats& stats,
                            double* out) {
    if (values == nullptr || out == nullptr || n_cols == 0) return;

    // Parallel over ROW BLOCKS, with the inner loop over columns. Row-major
    // means the inner loop walks contiguous memory, and the four per-column
    // parameter arrays are small enough to stay in L1 for the whole sweep --
    // so each row costs one streaming read, one streaming write, and no
    // cache pressure from the parameters.
    //
    // schedule(static), against this codebase's guided default, and the
    // exception is earned rather than assumed. omp_policy's case for guided
    // rests on work per iteration varying; here it provably does not --
    // every row is the same n_cols operations, and the only branch is a NaN
    // check whose cost does not depend on the answer. What this loop IS, is
    // memory-bandwidth bound, and guided's shrinking, non-contiguous chunks
    // work against the hardware prefetcher exactly when sequential
    // streaming is the only thing that matters. Measured at 504,000 x 8,
    // static was the better of the two at every thread count above one.
    //
    // Scaling flattens near the physical core count (19.2 ms at one thread,
    // ~6-7 ms from eight upward) because the memory system, not the ALUs,
    // is the limit. That is a property of the kernel on any machine and
    // needs no thread cap tuned to this one; the point-to-point wobble
    // beyond that is measurement noise on a workstation with other work on
    // it, not a scheduling defect worth engineering against.
    #pragma omp parallel for schedule(static) \
        if (sqt::omp_policy::worth_parallel(n_rows, n_cols)) \
        num_threads(sqt::omp_policy::max_threads() > 0 \
                        ? sqt::omp_policy::max_threads() : omp_get_max_threads())
    for (std::ptrdiff_t row = 0; row < static_cast<std::ptrdiff_t>(n_rows); ++row) {
        const std::size_t base = static_cast<std::size_t>(row) * n_cols;
        for (std::size_t c = 0; c < n_cols; ++c) {
            const double v = values[base + c];
            // NaN passes through untouched: Series.clip leaves missing
            // values missing rather than pinning them to a bound.
            if (std::isnan(v)) {
                out[base + c] = v;
                continue;
            }
            const double clipped = std::min(std::max(v, stats.lo[c]), stats.hi[c]);
            out[base + c] = (clipped - stats.mean[c]) / stats.stdev[c];
        }
    }
}

}  // namespace sqt
