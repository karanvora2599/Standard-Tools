#include "sqt/fp_contract.hpp"  // first: no contraction in this unit
#include "sqt/cusum.hpp"

#include "sqt/platform.hpp"

#include <algorithm>
#include <cstddef>

namespace sqt {
namespace {

// numpy's float64 `maximum`: a NaN in either argument comes back -- the
// first argument's when it is NaN, otherwise the second's -- and otherwise
// the larger of the two.
//
// Written as two selects with one comparison each: `larger` is b whenever
// either is NaN, and the second select puts a back when a is the NaN.
// Measured, the one-line `(a >= b || a != a) ? a : b` compiled to two
// conditional jumps on MSVC, and `up >= down` is a coin toss on noise, so
// its mispredictions were most of the kernel's time; this form compiles to
// compare-and-blend with no branch.
//
// WHICH ARGUMENT A TIE RETURNS CANNOT SHOW. Two equal doubles differ in
// bits only as +0.0 and -0.0, and no -0.0 ever reaches a maximum here. up,
// down and peak start at +0.0, and in round-to-nearest a sum is -0.0 only
// when both operands are and a difference only when its left operand is --
// so (up + z) - slack and (down - z) - slack are never -0.0 whatever z and
// slack hold, and by induction neither is anything a maximum below sees.
// So the tie rule is free to differ, and it does: the source returns a,
// MSVC compiles the first select to vmaxsd, which returns b, and numpy
// returns b (np.maximum(0.0, -0.0) is -0.0).
//
// `a != a` rather than std::isnan: the two agree under /fp:precise and
// without -ffast-math, which is how this file is compiled, and MSVC's
// std::isnan is a library classification rather than one compare.
inline double np_maximum(double a, double b) {
    const double larger = (a >= b) ? a : b;
    return (a != a) ? a : larger;
}

// Rows advanced together. A single row is one dependency chain -- each
// step's up reads the previous step's -- so its speed is the latency of
// add, subtract and select, not their throughput. Independent rows
// interleaved column by column fill those gaps.
constexpr std::size_t kRowsPerBlock = 4;

// Rows [0, K) of `z` (already offset to the block's first row), with the
// recursion split at the first scanned column so the scan test is not
// evaluated per element. The split moves no arithmetic: every column still
// runs both updates in the Python order, and only columns >= first_scanned
// touch the peak.
template <std::size_t K>
inline void cusum_block(const double* SQT_RESTRICT z,
                        std::size_t n_cols,
                        std::size_t first_scanned,
                        double slack,
                        double* SQT_RESTRICT peaks) {
    double up[K];
    double down[K];
    double peak[K];
    const double* row[K];
    for (std::size_t k = 0; k < K; ++k) {
        up[k] = 0.0;
        down[k] = 0.0;
        peak[k] = 0.0;
        row[k] = z + k * n_cols;
    }
    // Column 0 is never read: the Python loop starts at t = 1.
    const std::size_t warm_end = std::min(first_scanned, n_cols);
    for (std::size_t t = 1; t < warm_end; ++t) {
        for (std::size_t k = 0; k < K; ++k) {
            const double v = row[k][t];
            const double u = (up[k] + v) - slack;
            up[k] = np_maximum(0.0, u);
            const double d = (down[k] - v) - slack;
            down[k] = np_maximum(0.0, d);
        }
    }
    for (std::size_t t = std::max<std::size_t>(warm_end, 1); t < n_cols; ++t) {
        for (std::size_t k = 0; k < K; ++k) {
            const double v = row[k][t];
            const double u = (up[k] + v) - slack;
            up[k] = np_maximum(0.0, u);
            const double d = (down[k] - v) - slack;
            down[k] = np_maximum(0.0, d);
            peak[k] = np_maximum(peak[k], np_maximum(up[k], down[k]));
        }
    }
    for (std::size_t k = 0; k < K; ++k) peaks[k] = peak[k];
}

}  // namespace

void cusum_peaks_into(const double* SQT_RESTRICT z,
                      std::size_t n_rows,
                      std::size_t n_cols,
                      std::size_t n_reference,
                      double slack,
                      double* SQT_RESTRICT peaks) {
    // `t >= n_reference` for t from 1: n_reference 0 and 1 scan the same
    // columns, and the split in cusum_block needs no other special case.
    const std::size_t first_scanned = std::max<std::size_t>(n_reference, 1);
    const std::size_t n_blocks = n_rows / kRowsPerBlock;

    // SERIAL, DELIBERATELY. Rows are independent, so the blocks could run
    // in parallel without changing a bit, and measured on 200 x 2,105 the
    // kernel alone went from 0.6 ms to 0.1 ms on 16 threads. The CALLERS
    // got slower: vcomp's workers keep spinning for a while after a region
    // and compete with the Python that runs next, so `_ar1_null_peaks` went
    // from 7.5-10 ms to 13-15 ms and detect_basis_dislocation from 19-25 ms
    // to 36-41 ms. Four threads still lost end to end. Serially the kernel
    // is 20x the numpy loop and about 5% of what is left of the null
    // simulation, which is drawing the normals, filtering and standardizing
    // them -- so threads could save little here even without the spinning.
    for (std::size_t b = 0; b < n_blocks; ++b) {
        const std::size_t first = b * kRowsPerBlock;
        cusum_block<kRowsPerBlock>(z + first * n_cols, n_cols, first_scanned, slack,
                                   peaks + first);
    }
    // The rows left over after whole blocks, one at a time: the same
    // per-row arithmetic, so which rows shared a block changes nothing.
    for (std::size_t r = n_blocks * kRowsPerBlock; r < n_rows; ++r) {
        cusum_block<1>(z + r * n_cols, n_cols, first_scanned, slack, peaks + r);
    }
}

}  // namespace sqt
