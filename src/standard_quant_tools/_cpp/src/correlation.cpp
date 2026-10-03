#include "sqt/fp_contract.hpp"  // first: no contraction in this unit
#include "sqt/correlation.hpp"

#include "sqt/numerics.hpp"
#include "sqt/omp_policy.hpp"
#include "sqt/platform.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace sqt {
namespace {

// Complete columns handled together: pass 1 writes a block's values to one
// cache line of each row of D and E instead of one line per column, and
// advances that many independent Welford recursions at once (each one is a
// chain of dependent adds); pass 2 reuses each row of E for that many
// columns of x. Neither changes any operation inside a pair.
constexpr std::size_t kColumnBlock = 8;

inline double element(const double* values, std::ptrdiff_t row_stride,
                      std::ptrdiff_t col_stride, std::size_t i, std::size_t k) {
    return values[static_cast<std::ptrdiff_t>(i) * row_stride +
                  static_cast<std::ptrdiff_t>(k) * col_stride];
}

// pandas' cell, from a pair's three sums: NaN below min_periods, otherwise
// covxy / sqrt(ssqdmx * ssqdmy), and NaN when that divisor is 0 (a NaN
// divisor is not 0, so it gives covxy / NaN, as in pandas).
inline double finish(double covxy, double ssqdmx, double ssqdmy, long long nobs,
                     long long min_periods) {
    if (nobs < min_periods) return std::numeric_limits<double>::quiet_NaN();
    const double divisor = std::sqrt(ssqdmx * ssqdmy);
    if (divisor != 0.0) return covxy / divisor;
    return std::numeric_limits<double>::quiet_NaN();
}

// pandas' loop for one pair, as written in pandas/_libs/algos.pyx, over the
// rows where both values are finite. x is the column with the larger index.
double nancorr_pair(const double* values, std::ptrdiff_t row_stride,
                    std::ptrdiff_t col_stride, std::size_t n_rows, std::size_t x,
                    std::size_t y, long long min_periods) {
    long long nobs = 0;
    double meanx = 0.0, meany = 0.0, ssqdmx = 0.0, ssqdmy = 0.0, covxy = 0.0;
    for (std::size_t i = 0; i < n_rows; ++i) {
        const double vx = element(values, row_stride, col_stride, i, x);
        const double vy = element(values, row_stride, col_stride, i, y);
        if (!(numerics::is_finite(vx) && numerics::is_finite(vy))) continue;
        nobs += 1;
        const double dx = vx - meanx;
        const double dy = vy - meany;
        // `meanx += 1. / nobs * dx`: the reciprocal, then the product.
        const double inv = 1.0 / static_cast<double>(nobs);
        meanx += inv * dx;
        meany += inv * dy;
        const double rx = vx - meanx;
        ssqdmx += rx * dx;
        ssqdmy += (vy - meany) * dy;
        covxy += rx * dy;
    }
    return finish(covxy, ssqdmx, ssqdmy, nobs, min_periods);
}

// acc[b] += da * e[b] for b < n: one row's products added to n pairs' sums,
// each sum its own lane -- a vectorized loop reorders nothing inside a sum.
inline void accumulate_row(double* SQT_RESTRICT acc, const double* SQT_RESTRICT e,
                           double da, std::size_t n) {
    for (std::size_t b = 0; b < n; ++b) acc[b] += da * e[b];
}

#ifdef _OPENMP
inline int thread_budget() {
    return omp_policy::max_threads() > 0 ? omp_policy::max_threads()
                                         : omp_get_max_threads();
}
#endif

}  // namespace

void pearson_correlation_into(const double* values,
                              std::size_t n_rows,
                              std::size_t n_cols,
                              std::ptrdiff_t row_stride,
                              std::ptrdiff_t col_stride,
                              long long min_periods,
                              double* out) {
    if (n_cols == 0) return;
    const char* fn = "pearson_correlation";
    const long long n_cols_ll = numerics::checked_narrow_to_ll(n_cols, fn);
    const long long nobs_complete = numerics::checked_narrow_to_ll(n_rows, fn);

    // Which columns are finite on every row. A pair of two such columns sees
    // every row, which is what lets its Welford sequences be shared.
    std::vector<std::size_t> complete;
    std::vector<unsigned char> is_complete(n_cols, 0);
    complete.reserve(n_cols);
    for (std::size_t k = 0; k < n_cols; ++k) {
        bool finite = true;
        for (std::size_t i = 0; i < n_rows && finite; ++i)
            finite = numerics::is_finite(element(values, row_stride, col_stride, i, k));
        if (finite) {
            is_complete[k] = 1;
            complete.push_back(k);
        }
    }
    const std::size_t kc = complete.size();

    if (kc > 0) {
        // D[i, a] = v - mean after row i's update and E[i, a] = v - mean
        // before it, for complete column a; row-major, so a row of E is
        // contiguous for pass 2. acc holds the lower triangle of the pair
        // sums, row a contiguous.
        const std::size_t panel = numerics::checked_mul(n_rows, kc, fn);
        std::vector<double> D(panel), E(panel), ssq(kc, 0.0);
        std::vector<double> acc(numerics::checked_mul(kc, kc, fn), 0.0);
        double* const d_data = D.data();
        double* const e_data = E.data();
        double* const ssq_data = ssq.data();
        double* const acc_data = acc.data();
        const std::size_t* const complete_data = complete.data();
        const std::size_t n_blocks = (kc + kColumnBlock - 1) / kColumnBlock;
        const long long n_blocks_ll = numerics::checked_narrow_to_ll(n_blocks, fn);

        // Pass 1: each complete column's Welford recursion, once. Every row
        // is complete, so nobs on row i is i + 1 in every column, and its
        // reciprocal is computed once per row -- the same division of the
        // same operands pandas performs per pair.
        const omp_policy::parallel_call columns_call(
            n_blocks, n_rows * kColumnBlock, omp_policy::cost::correlation_columns);
#ifdef _OPENMP
        #pragma omp parallel for schedule(guided) \
            if (columns_call.parallel()) \
            num_threads(thread_budget())
#endif
        for (long long blk = 0; blk < n_blocks_ll; ++blk) {
            const std::size_t a0 = static_cast<std::size_t>(blk) * kColumnBlock;
            const std::size_t width = std::min(kColumnBlock, kc - a0);
            double mean[kColumnBlock];
            double sum_sq[kColumnBlock];
            std::ptrdiff_t offset[kColumnBlock];
            for (std::size_t j = 0; j < width; ++j) {
                mean[j] = 0.0;
                sum_sq[j] = 0.0;
                offset[j] = static_cast<std::ptrdiff_t>(complete_data[a0 + j]) * col_stride;
            }
            for (std::size_t i = 0; i < n_rows; ++i) {
                const double inv = 1.0 / static_cast<double>(i + 1);
                const double* row = values + static_cast<std::ptrdiff_t>(i) * row_stride;
                double* d_row = d_data + i * kc + a0;
                double* e_row = e_data + i * kc + a0;
                for (std::size_t j = 0; j < width; ++j) {
                    const double v = row[offset[j]];
                    const double before = v - mean[j];
                    e_row[j] = before;
                    mean[j] += inv * before;
                    const double after = v - mean[j];
                    d_row[j] = after;
                    sum_sq[j] += after * before;
                }
            }
            for (std::size_t j = 0; j < width; ++j) ssq_data[a0 + j] = sum_sq[j];
        }

        // Pass 2: covxy for every complete pair (a, b <= a), summed over the
        // rows in order. A block of x columns shares each row of E. The
        // triangle's rows grow with a, so the widest blocks go first and are
        // handed out one at a time; the answer does not depend on which
        // thread takes which block.
        const omp_policy::parallel_call pairs_call(
            n_blocks, n_rows * kc * kColumnBlock / 2, omp_policy::cost::correlation_pairs);
#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 1) \
            if (pairs_call.parallel()) \
            num_threads(thread_budget())
#endif
        for (long long t = 0; t < n_blocks_ll; ++t) {
            const std::size_t blk = n_blocks - 1 - static_cast<std::size_t>(t);
            const std::size_t a0 = blk * kColumnBlock;
            const std::size_t a1 = std::min(a0 + kColumnBlock, kc);
            for (std::size_t i = 0; i < n_rows; ++i) {
                const double* d_row = d_data + i * kc;
                const double* e_row = e_data + i * kc;
                for (std::size_t a = a0; a < a1; ++a)
                    accumulate_row(acc_data + a * kc, e_row, d_row[a], a + 1);
            }
        }

        // Pass 3: the cells. ssqdmx is x's (the larger index), as in pandas.
        for (std::size_t a = 0; a < kc; ++a) {
            const std::size_t x = complete_data[a];
            for (std::size_t b = 0; b <= a; ++b) {
                const std::size_t y = complete_data[b];
                const double r = finish(acc_data[a * kc + b], ssq_data[a], ssq_data[b],
                                        nobs_complete, min_periods);
                out[x * n_cols + y] = r;
                out[y * n_cols + x] = r;
            }
        }
    }

    // Every pair with a missing value in either column: pandas' loop as
    // written, over that pair's own complete rows.
    if (kc < n_cols) {
        const unsigned char* const complete_flag = is_complete.data();
        const omp_policy::parallel_call gapped_call(
            n_cols, n_rows * n_cols / 2, omp_policy::cost::correlation_gapped_pairs);
#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 1) \
            if (gapped_call.parallel()) \
            num_threads(thread_budget())
#endif
        for (long long xl = 0; xl < n_cols_ll; ++xl) {
            const auto x = static_cast<std::size_t>(xl);
            for (std::size_t y = 0; y <= x; ++y) {
                if (complete_flag[x] && complete_flag[y]) continue;
                const double r =
                    nancorr_pair(values, row_stride, col_stride, n_rows, x, y, min_periods);
                out[x * n_cols + y] = r;
                out[y * n_cols + x] = r;
            }
        }
    }
}

}  // namespace sqt
