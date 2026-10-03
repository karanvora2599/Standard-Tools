#pragma once

#include <cstddef>

namespace sqt {

/**
 * The Pearson correlation matrix of a panel's columns, exactly as pandas'
 * `DataFrame.corr()` computes it -- `pandas._libs.algos.nancorr(mat,
 * cov=False, minp=min_periods)` -- before the clip to [-1, 1] that pandas 3
 * added (the caller applies that; see below).
 *
 * WHY THIS IS NATIVE. portfolio.construction.hierarchical_risk_parity spends
 * 78-86% of its time at 235 assets in `frame.corr()`: nancorr walks every
 * row once per PAIR of columns, a serial Welford recursion per pair, so
 * 27,730 pairs x 2,106 rows is 58 million dependent updates on one thread.
 * A faster formula (np.corrcoef, a centred matrix product) differs in the
 * last bits, and HRP's single-linkage tree breaks ties on those bits, so the
 * answer must be pandas' own, not a better one.
 *
 * THE SAME ARITHMETIC. For each pair (xi, yi) with yi <= xi -- x is the
 * column with the LARGER index, as in pandas' loop -- over the rows i, in
 * order, where both values are finite (NaN and +/-inf are both missing:
 * pandas masks with np.isfinite):
 *
 *     nobs += 1
 *     dx = vx - meanx;          dy = vy - meany
 *     meanx += 1.0 / nobs * dx; meany += 1.0 / nobs * dy
 *     ssqdmx += (vx - meanx) * dx
 *     ssqdmy += (vy - meany) * dy
 *     covxy  += (vx - meanx) * dy
 *
 * all starting at 0.0; then NaN when nobs < min_periods, otherwise
 * covxy / sqrt(ssqdmx * ssqdmy), or NaN when that divisor is 0. The result
 * is written to [xi, yi] and [yi, xi] alike, so the matrix is exactly
 * symmetric. Nothing is fused (the unit compiles without contraction) and
 * nothing is reassociated, so every cell is pandas' to the bit.
 *
 * HOW IT IS FASTER WITHOUT CHANGING A BIT. When BOTH columns of a pair are
 * finite on every row -- every pair, in HRP, which drops incomplete rows
 * first -- the rows a pair sees are all of them, so x's Welford sequence
 * (meanx, ssqdmx and each vx - meanx) is the same in every pair x is in:
 * it is computed once per column. Each pair then reduces to
 * covxy = sum over i of D[i, x] * E[i, y], with D = vx - meanx after the
 * update and E = vy - meany before it, accumulated in row order -- the
 * same products added in the same order. 1.0 / nobs, the same for every
 * column on a row, is divided once per row. The pair sums run many pairs
 * side by side, which reorders nothing within a pair. A pair with a missing
 * value in either column runs pandas' loop exactly as written.
 *
 * THREADS. Pairs are independent and each cell is computed by one thread
 * from its own inputs in a fixed order, so the result is the same bits on
 * any number of threads (OpenMP over blocks of columns, gated by
 * omp_policy like the other kernels).
 *
 * THE CLIP IS THE CALLER'S. pandas 3 clips each coefficient to [-1, 1];
 * pandas 1.5-2.3 do not. The arithmetic above is identical in all of them,
 * so the kernel returns the unclipped value and the Python caller clips
 * exactly when the installed pandas does.
 *
 * @param values       The panel, element (i, k) at values[i * row_stride +
 *                     k * col_stride] (strides in elements, so a C- or
 *                     Fortran-ordered array is read in place). May be null
 *                     when n_rows or n_cols is 0.
 * @param n_rows       Observations.
 * @param n_cols       Columns (assets).
 * @param row_stride   Distance between consecutive rows, in doubles.
 * @param col_stride   Distance between consecutive columns, in doubles.
 * @param min_periods  pandas' minp: a pair with fewer complete rows is NaN.
 * @param out          Output, row-major (n_cols, n_cols). Never aliased with
 *                     values (the binding allocates it fresh).
 *
 * Throws std::bad_alloc when its scratch (two n_rows x n_cols panels and
 * an n_cols x n_cols accumulator for the complete columns) cannot be
 * allocated.
 */
void pearson_correlation_into(const double* values,
                              std::size_t n_rows,
                              std::size_t n_cols,
                              std::ptrdiff_t row_stride,
                              std::ptrdiff_t col_stride,
                              long long min_periods,
                              double* out);

}  // namespace sqt
