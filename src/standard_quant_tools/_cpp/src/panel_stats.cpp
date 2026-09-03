#include "sqt/panel_stats.hpp"

#include "sqt/omp_policy.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
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


namespace {

/**
 * Bucket rows by date in O(n_rows), replacing the caller's argsort.
 *
 * The Python path sorted the whole panel by date code (O(n log n)) and then
 * gathered both columns through the permutation. A counting sort does the
 * same job in one counting pass and one scatter pass, and it is stable, so
 * rows keep their original relative order inside a date -- which matters
 * because a tie-break that depended on the sort would otherwise differ from
 * the reference implementation.
 *
 * `keep(i)` decides whether row i participates; excluded rows land nowhere,
 * so the buckets contain exactly the usable rows.
 */
template <typename Keep>
bool bucket_by_date(const long long* date_codes,
                    std::size_t n_rows,
                    std::size_t n_dates,
                    const Keep& keep,
                    std::vector<std::size_t>& offsets,
                    std::vector<std::size_t>& counts,
                    std::vector<std::size_t>& order) {
    try {
        offsets.assign(n_dates + 1, 0);
        counts.assign(n_dates, 0);
        order.resize(n_rows);
    } catch (const std::bad_alloc&) {
        return false;
    }

    for (std::size_t i = 0; i < n_rows; ++i) {
        const long long code = date_codes[i];
        // A code outside range would corrupt neighbouring buckets, so it is
        // dropped rather than trusted. The caller builds these from
        // pd.factorize and cannot produce one, but a silent out-of-bounds
        // write is not a failure mode worth leaving open.
        if (code < 0 || static_cast<std::size_t>(code) >= n_dates) continue;
        if (!keep(i)) continue;
        ++counts[static_cast<std::size_t>(code)];
    }
    std::size_t running = 0;
    for (std::size_t d = 0; d < n_dates; ++d) {
        offsets[d] = running;
        running += counts[d];
    }
    offsets[n_dates] = running;

    std::vector<std::size_t> cursor(offsets.begin(), offsets.begin() +
                                                         static_cast<std::ptrdiff_t>(n_dates));
    for (std::size_t i = 0; i < n_rows; ++i) {
        const long long code = date_codes[i];
        if (code < 0 || static_cast<std::size_t>(code) >= n_dates) continue;
        if (!keep(i)) continue;
        order[cursor[static_cast<std::size_t>(code)]++] = i;
    }
    return true;
}

/**
 * Average ranks over one contiguous block, matching Series.rank()'s default.
 *
 * Ties take the MEAN of the ordinals they span. Getting this wrong does not
 * fail loudly -- it produces a correlation that is quietly a little
 * different from pandas' -- which is why it is factored out and tested
 * directly rather than inlined twice.
 *
 * `scratch` is the caller's per-thread index buffer, reused across dates so
 * a 2,000-date panel does not allocate 2,000 times.
 */
void average_ranks(const double* values,
                   std::size_t n,
                   std::vector<std::size_t>& scratch,
                   double* out,
                   bool allow_parallel_sort = false) {
    scratch.resize(n);
    for (std::size_t i = 0; i < n; ++i) scratch[i] = i;
    const auto by_value = [values](std::size_t a, std::size_t b) {
        return values[a] < values[b];
    };

    // The POOLED correlation is one enormous segment, so the per-date
    // parallelism that carries the cross-sectional case does nothing for it
    // and the whole cost is this one sort. Splitting it into per-thread
    // runs and merging them recovers the parallelism; measured on 2,000,000
    // rows the single-threaded version was barely faster than numpy's,
    // which is not a reason to have written a kernel.
    //
    // Only for a segment big enough to pay for the merges. Every date in a
    // cross-sectional call is far below this, so they take the plain sort
    // and never enter the parallel region -- which matters, because this is
    // itself called from inside a parallel loop there.
    constexpr std::size_t kParallelSortMin = 50000;
#ifdef _OPENMP
    const int configured = sqt::omp_policy::max_threads();
    const int usable = configured > 0 ? configured : omp_get_max_threads();
    if (allow_parallel_sort && n >= kParallelSortMin && usable > 1) {
        const auto chunks = static_cast<std::size_t>(usable);
        std::vector<std::size_t> bounds(chunks + 1);
        for (std::size_t c = 0; c <= chunks; ++c) {
            bounds[c] = n * c / chunks;
        }
        #pragma omp parallel for schedule(static) num_threads(usable)
        for (std::ptrdiff_t c = 0; c < static_cast<std::ptrdiff_t>(chunks); ++c) {
            const auto i = static_cast<std::size_t>(c);
            std::sort(scratch.begin() + static_cast<std::ptrdiff_t>(bounds[i]),
                      scratch.begin() + static_cast<std::ptrdiff_t>(bounds[i + 1]),
                      by_value);
        }
        // Pairwise merge, halving the run count each round. The rounds are
        // sequential but each round's merges are independent.
        for (std::size_t width = 1; width < chunks; width *= 2) {
            const std::size_t stride = width * 2;
            const auto pairs =
                static_cast<std::ptrdiff_t>((chunks + stride - 1) / stride);
            #pragma omp parallel for schedule(static) num_threads(usable)
            for (std::ptrdiff_t pair = 0; pair < pairs; ++pair) {
                const std::size_t left = static_cast<std::size_t>(pair) * stride;
                const std::size_t mid = left + width;
                if (mid >= chunks) continue;
                const std::size_t right = std::min(left + stride, chunks);
                std::inplace_merge(
                    scratch.begin() + static_cast<std::ptrdiff_t>(bounds[left]),
                    scratch.begin() + static_cast<std::ptrdiff_t>(bounds[mid]),
                    scratch.begin() + static_cast<std::ptrdiff_t>(bounds[right]),
                    by_value);
            }
        }
    } else {
        std::sort(scratch.begin(), scratch.end(), by_value);
    }
#else
    (void)allow_parallel_sort;
    std::sort(scratch.begin(), scratch.end(), by_value);
#endif
    std::size_t i = 0;
    while (i < n) {
        std::size_t j = i + 1;
        while (j < n && values[scratch[j]] == values[scratch[i]]) ++j;
        // Ordinals are 1-based; the mean of i+1 .. j is (i + j + 1) / 2.
        const double mean_rank =
            (static_cast<double>(i) + static_cast<double>(j) + 1.0) * 0.5;
        for (std::size_t k = i; k < j; ++k) out[scratch[k]] = mean_rank;
        i = j;
    }
}

/**
 * Pearson correlation of two equal-length blocks, centered form.
 *
 * The n*Sxy - Sx*Sy shortcut differences two nearly equal large numbers on
 * return-scale data and loses most of its significant digits; this is the
 * same two-pass form numpy's corrcoef uses, and the reason the Python
 * implementation was written that way too.
 */
double centered_correlation(const double* x, const double* y, std::size_t n) {
    if (n < 2) return 0.0;
    const double inv_n = 1.0 / static_cast<double>(n);
    const double mean_x = pairwise_sum([x](std::size_t i) { return x[i]; }, 0, n) * inv_n;
    const double mean_y = pairwise_sum([y](std::size_t i) { return y[i]; }, 0, n) * inv_n;

    const double cov = pairwise_sum(
        [&](std::size_t i) { return (x[i] - mean_x) * (y[i] - mean_y); }, 0, n);
    const double var_x = pairwise_sum(
        [&](std::size_t i) { const double d = x[i] - mean_x; return d * d; }, 0, n);
    const double var_y = pairwise_sum(
        [&](std::size_t i) { const double d = y[i] - mean_y; return d * d; }, 0, n);

    const double denom = std::sqrt(var_x * var_y);
    if (!(denom > 0.0)) return 0.0;  // constant cross-section: undefined -> 0.0
    const double r = cov / denom;
    return std::isfinite(r) ? r : 0.0;
}

}  // namespace

bool cross_sectional_correlation(const double* y_true,
                                 const double* y_pred,
                                 const long long* date_codes,
                                 std::size_t n_rows,
                                 std::size_t n_dates,
                                 bool spearman,
                                 double* out_ic) {
    if (out_ic == nullptr || n_dates == 0) return true;
    for (std::size_t d = 0; d < n_dates; ++d) out_ic[d] = 0.0;
    if (n_rows == 0 || y_true == nullptr || y_pred == nullptr ||
        date_codes == nullptr)
        return true;

    // Drop NaN PAIRS, which is what Series.corr does before it correlates
    // and, for spearman, before it ranks. An infinity is kept: pandas treats
    // only NaN as missing.
    const auto keep = [y_true, y_pred](std::size_t i) {
        return !std::isnan(y_true[i]) && !std::isnan(y_pred[i]);
    };

    std::vector<std::size_t> offsets, counts, order;
    if (!bucket_by_date(date_codes, n_rows, n_dates, keep, offsets, counts, order))
        return false;

    bool alloc_error = false;
    // The pooled case (one segment) gets its parallelism from inside the
    // ranking sort instead of from this loop, and the two must not be
    // nested: OpenMP disables nested regions by default, so entering a
    // one-iteration parallel region here would silently serialize the sort
    // that was supposed to be parallel. Hence the explicit n_dates > 1.
    const bool parallel_over_dates =
        n_dates > 1 &&
        sqt::omp_policy::worth_parallel(n_dates, n_rows / n_dates);
    const bool parallel_within_segment = (n_dates == 1);

    #pragma omp parallel for schedule(guided) reduction(|| : alloc_error) \
        if (parallel_over_dates) \
        num_threads(sqt::omp_policy::max_threads() > 0 \
                        ? sqt::omp_policy::max_threads() : omp_get_max_threads())
    for (std::ptrdiff_t date = 0; date < static_cast<std::ptrdiff_t>(n_dates); ++date) {
        const auto d = static_cast<std::size_t>(date);
        const std::size_t n = counts[d];
        if (n < 2) continue;  // undefined; already 0.0

        // Thread-local buffers, declared inside the loop body so each thread
        // owns its own, and reused across the dates that thread handles.
        std::vector<double> xs, ys;
        std::vector<std::size_t> rank_scratch;
        try {
            xs.resize(n);
            ys.resize(n);
        } catch (const std::bad_alloc&) {
            alloc_error = true;
            continue;
        }

        const std::size_t base = offsets[d];
        for (std::size_t i = 0; i < n; ++i) {
            const std::size_t row = order[base + i];
            xs[i] = y_true[row];
            ys[i] = y_pred[row];
        }

        if (spearman) {
            std::vector<double> rx, ry;
            try {
                rx.resize(n);
                ry.resize(n);
            } catch (const std::bad_alloc&) {
                alloc_error = true;
                continue;
            }
            average_ranks(xs.data(), n, rank_scratch, rx.data(),
                          parallel_within_segment);
            average_ranks(ys.data(), n, rank_scratch, ry.data(),
                          parallel_within_segment);
            out_ic[d] = centered_correlation(rx.data(), ry.data(), n);
        } else {
            out_ic[d] = centered_correlation(xs.data(), ys.data(), n);
        }
    }

    return !alloc_error;
}

bool standardize_by_date(const double* values,
                         std::size_t n_rows,
                         std::size_t n_cols,
                         const long long* date_codes,
                         std::size_t n_dates,
                         double clip_sigma,
                         double* out) {
    if (out == nullptr || n_cols == 0) return true;
    if (n_rows == 0 || values == nullptr || date_codes == nullptr) return true;

    // Every row participates in the bucketing here -- unlike the correlation,
    // where a NaN pair is dropped outright. A row with a NaN in one column
    // still has usable values in the others, so missingness is handled per
    // column inside the date rather than per row.
    const auto keep_all = [](std::size_t) { return true; };
    std::vector<std::size_t> offsets, counts, order;
    if (!bucket_by_date(date_codes, n_rows, n_dates, keep_all, offsets, counts,
                        order))
        return false;

    bool alloc_error = false;

    #pragma omp parallel for schedule(guided) reduction(|| : alloc_error) \
        if (sqt::omp_policy::worth_parallel(n_dates, \
                                            (n_rows / (n_dates ? n_dates : 1)) * n_cols)) \
        num_threads(sqt::omp_policy::max_threads() > 0 \
                        ? sqt::omp_policy::max_threads() : omp_get_max_threads())
    for (std::ptrdiff_t date = 0; date < static_cast<std::ptrdiff_t>(n_dates); ++date) {
        const auto d = static_cast<std::size_t>(date);
        const std::size_t n = counts[d];
        if (n == 0) continue;
        const std::size_t base = offsets[d];

        std::vector<double> column;
        try {
            column.resize(n);
        } catch (const std::bad_alloc&) {
            alloc_error = true;
            continue;
        }

        for (std::size_t c = 0; c < n_cols; ++c) {
            // NaN IS SKIPPED BY THE MOMENTS, which is what panel_stats.hpp
            // has always promised. It used to be propagated instead, to
            // match a wart on the Python side: one NaN poisoned the date's
            // mean and the non-finite sweep then wrote 0.0 for EVERY entity
            // in that date -- reporting each present name as sitting exactly
            // at the cross-sectional mean, the one fabricated observation
            // the header says must not happen.
            //
            // That was deliberate, and its stated justification was "in
            // practice it never fires: alignment drops NaN rows before the
            // panel reaches the engine." `load_external_panel` retired that
            // premise -- an externally computed panel keeps its warm-up
            // NaNs -- so the wart became reachable. Changed on the Python
            // side first, then here, with the tests, as that note asked.
            //
            // Only the finite values are compacted into `column`, so the
            // pairwise summation still sees a dense run and keeps agreeing
            // with pandas in the last bits.
            std::size_t n_valid = 0;
            for (std::size_t i = 0; i < n; ++i) {
                const double v = values[order[base + i] * n_cols + c];
                if (!std::isnan(v)) column[n_valid++] = v;
            }
            const double* buffer = column.data();

            const double mean =
                n_valid ? pairwise_sum([buffer](std::size_t i) { return buffer[i]; },
                                       0, n_valid) /
                              static_cast<double>(n_valid)
                        : std::numeric_limits<double>::quiet_NaN();
            // max(n-1, 1) rather than n-1, matching the Python's
            // np.maximum(n_valid - 1.0, 1.0): a single usable entity divides
            // by one instead of zero, and is then caught by the std > 0 test.
            const double denom = static_cast<double>(n_valid > 1 ? n_valid - 1 : 1);
            const double sum_sq = pairwise_sum(
                [buffer, mean](std::size_t i) {
                    const double delta = buffer[i] - mean;
                    return delta * delta;
                },
                0, n_valid);
            const double stdev = std::sqrt(sum_sq / denom);
            // No dispersion means every PRESENT entity sits exactly at the
            // mean, so its standardized value is 0.0 by definition -- not
            // NaN, which would drop the whole date downstream.
            const bool usable = (stdev > 0.0) && std::isfinite(stdev);

            for (std::size_t i = 0; i < n; ++i) {
                const std::size_t index = order[base + i] * n_cols + c;
                const double raw = values[index];
                if (std::isnan(raw)) {
                    // Absent stays absent.
                    out[index] = raw;
                    continue;
                }
                double z = usable ? (raw - mean) / stdev : 0.0;
                if (!std::isfinite(z)) z = 0.0;
                if (clip_sigma > 0.0) {
                    z = std::min(std::max(z, -clip_sigma), clip_sigma);
                }
                out[index] = z;
            }
        }
    }

    return !alloc_error;
}


bool label_uniqueness(const long long* dates,
                      const long long* label_end,
                      const long long* entity_codes,
                      std::size_t n_rows,
                      std::size_t n_entities,
                      double* out_weights) {
    if (out_weights == nullptr) return true;
    for (std::size_t i = 0; i < n_rows; ++i) out_weights[i] = 1.0;
    if (n_rows == 0 || n_entities == 0 || dates == nullptr ||
        label_end == nullptr || entity_codes == nullptr)
        return true;

    const auto keep_all = [](std::size_t) { return true; };
    std::vector<std::size_t> offsets, counts, order;
    if (!bucket_by_date(entity_codes, n_rows, n_entities, keep_all, offsets,
                        counts, order))
        return false;

    bool alloc_error = false;

    #pragma omp parallel for schedule(guided) reduction(|| : alloc_error) \
        if (sqt::omp_policy::worth_parallel(n_entities, n_rows / n_entities)) \
        num_threads(sqt::omp_policy::max_threads() > 0 \
                        ? sqt::omp_policy::max_threads() : omp_get_max_threads())
    for (std::ptrdiff_t entity = 0;
         entity < static_cast<std::ptrdiff_t>(n_entities); ++entity) {
        const auto e = static_cast<std::size_t>(entity);
        const std::size_t n = counts[e];
        if (n == 0) continue;
        const std::size_t base = offsets[e];

        std::vector<std::size_t> rows;
        std::vector<long long> axis;
        std::vector<double> delta, cumulative;
        try {
            rows.assign(order.begin() + static_cast<std::ptrdiff_t>(base),
                        order.begin() + static_cast<std::ptrdiff_t>(base + n));
            axis.resize(n);
            delta.assign(n + 1, 0.0);
            cumulative.resize(n + 1);
        } catch (const std::bad_alloc&) {
            alloc_error = true;
            continue;
        }

        // Sort this entity's rows by its OWN date axis. Positions are then
        // bar indices for this entity specifically, which is the whole
        // reason label ends are carried as timestamps rather than integer
        // offsets: with entities on different calendars, t+horizon of one
        // entity's bars is not t+horizon of the global panel's dates.
        std::sort(rows.begin(), rows.end(),
                  [dates](std::size_t a, std::size_t b) {
                      return dates[a] < dates[b];
                  });
        for (std::size_t i = 0; i < n; ++i) axis[i] = dates[rows[i]];

        // Concurrency by difference array: +1 where a label starts, -1 just
        // past where it ends, then a running sum. O(n) for what would
        // otherwise be an O(n * horizon) sweep over every label's span.
        for (std::size_t i = 0; i < n; ++i) {
            const long long end = label_end[rows[i]];
            std::size_t end_pos = i;
            // NaT marks a label that never resolves (the final `horizon`
            // rows). numpy stores it as INT64_MIN; such a row spans only
            // itself rather than being dropped, matching the Python.
            if (end != std::numeric_limits<long long>::min()) {
                const auto upper =
                    std::upper_bound(axis.begin(), axis.end(), end);
                const auto distance = upper - axis.begin();
                if (distance > 0) {
                    const auto candidate = static_cast<std::size_t>(distance - 1);
                    if (candidate > end_pos) end_pos = candidate;
                }
            }
            delta[i] += 1.0;
            delta[end_pos + 1] -= 1.0;
        }

        double running = 0.0;
        cumulative[0] = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            running += delta[i];
            // A bar covered by no label cannot lie inside any label's span,
            // so the guard only protects the division, never a real value.
            cumulative[i + 1] = cumulative[i] + 1.0 / std::max(running, 1.0);
        }

        for (std::size_t i = 0; i < n; ++i) {
            const long long end = label_end[rows[i]];
            std::size_t end_pos = i;
            if (end != std::numeric_limits<long long>::min()) {
                const auto upper =
                    std::upper_bound(axis.begin(), axis.end(), end);
                const auto distance = upper - axis.begin();
                if (distance > 0) {
                    const auto candidate = static_cast<std::size_t>(distance - 1);
                    if (candidate > end_pos) end_pos = candidate;
                }
            }
            const double span = static_cast<double>(end_pos - i + 1);
            out_weights[rows[i]] =
                (cumulative[end_pos + 1] - cumulative[i]) / span;
        }
    }

    if (alloc_error) return false;

    // Normalized to mean 1 over EVERY row, so switching weighting on does
    // not also rescale the effective regularization strength. Summed
    // pairwise for the same reason the other kernels are: this has to agree
    // with numpy's mean.
    const double* weights = out_weights;
    const double total =
        pairwise_sum([weights](std::size_t i) { return weights[i]; }, 0, n_rows);
    const double mean = total / static_cast<double>(n_rows);
    if (mean > 0.0) {
        for (std::size_t i = 0; i < n_rows; ++i) out_weights[i] /= mean;
    }
    return true;
}

}  // namespace sqt
