#pragma once

#include "sqt/platform.hpp"

#include <cstddef>

namespace sqt {

/**
 * The peak two-sided CUSUM statistic of every row of a standardized panel.
 *
 * WHY THIS IS NATIVE. analysis/liquidity_events.py calibrates its threshold
 * against simulated AR(1) null paths -- 200 paths of the channel's length --
 * and scans each with the same CUSUM it runs on the data. The recursion
 *
 *     up   = max(0, up   + z - slack)
 *     down = max(0, down - z - slack)
 *
 * is nonlinear, so it has no filter form and no prefix-scan form. The exact
 * closed form, up_t = S_t - min(0, min_{j<=t} S_j), was measured 1.5-1.7x
 * SLOWER than the numpy loop it would replace: it materialises four
 * panel-sized temporaries and goes memory-bound where the loop keeps one
 * column in L1. The loop itself costs about 20 ns per element in Python,
 * against a few in C++ over contiguous rows.
 *
 * THE SAME ARITHMETIC, NOT A SIMILAR ONE. Per row, and per column t from 1
 * (column 0 is never read, as in the Python loop), this evaluates exactly
 *
 *     up   = maximum(0.0, (up + z[t]) - slack)
 *     down = maximum(0.0, (down - z[t]) - slack)
 *     if t >= n_reference:
 *         peak = maximum(peak, maximum(up, down))
 *
 * with up, down and peak starting at 0.0, and `maximum` as numpy's
 * float64 maximum: a NaN in either argument propagates, otherwise the
 * larger. The association is the Python expression's, left to right;
 * nothing is hoisted (z - slack, say, would round differently). There is no
 * multiply, so no contraction into a fused multiply-add is possible, and
 * the translation unit is compiled without floating-point contraction
 * regardless. The result is the Python loop's, bit for bit.
 *
 * One limit, on bits no caller reads: a NaN's payload. Where a row's NaNs
 * all come from one source the peak is the same NaN on both paths. Where a
 * row mixes two -- a NaN in z and one made by inf - inf, say -- the peak is
 * NaN on both, but which of the two NaNs it carries can differ: numpy's
 * own add keeps the second operand's NaN when both are NaN, and a compiled
 * add is free to keep either. The simulated paths the library scans are
 * finite, so it never arises there.
 *
 * ROWS ARE INDEPENDENT. Each row's recursion reads only its own row, so
 * rows are advanced four at a time to overlap their dependency chains;
 * that changes no operation inside a row, so the answer does not depend on
 * which rows shared a block. The kernel is serial on purpose: a parallel
 * version was measured to make its callers slower end to end (see
 * cusum.cpp).
 *
 * @param z            Row-major (n_rows, n_cols) standardized values; row r
 *                     is one path, column t one time step.
 * @param n_rows       Number of rows (paths).
 * @param n_cols       Number of columns (time steps).
 * @param n_reference  First column the peak reads. Columns before it still
 *                     advance the recursion; they define normal and are not
 *                     scanned. Any value works: 0 and 1 both scan from
 *                     column 1, and n_reference >= n_cols scans nothing.
 * @param slack        Drift absorbed per step, in the units of z. Any
 *                     double: NaN or an infinity follows IEEE arithmetic
 *                     exactly as the Python loop does.
 * @param peaks        Output, length n_rows. Never aliased with z (the
 *                     binding allocates it fresh).
 */
void cusum_peaks_into(const double* SQT_RESTRICT z,
                      std::size_t n_rows,
                      std::size_t n_cols,
                      std::size_t n_reference,
                      double slack,
                      double* SQT_RESTRICT peaks);

}  // namespace sqt
