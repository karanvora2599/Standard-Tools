#include "sqt/rolling_regression.hpp"

#include "sqt/isa_dispatch.hpp"
#include "sqt/numerics.hpp"
#include "sqt/qr.hpp"
#include "sqt/rolling_beta_avx2.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace sqt {

namespace {
    constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

}  // namespace

// ── rolling_factor_loadings ───────────────────────────────────────────────────

void rolling_factor_loadings_into(
    const double* SQT_RESTRICT y,
    const double* SQT_RESTRICT factors,
    std::size_t   n,
    std::size_t   k,
    int           window,
    double* SQT_RESTRICT       out)
{
    const int p = static_cast<int>(k) + 1;  // intercept + k factors
    std::fill(out, out + n * static_cast<std::size_t>(p), kNaN);

    if (window < p + 1 || n < static_cast<std::size_t>(window)) return;

    // ── Why this is a per-window QR and no longer a rank-1 XtX update ───────
    //
    // The old path maintained XtX/Xty incrementally (O(p^2) per bar) and
    // solved by Cholesky. It was faster, and it was wrong in two ways that a
    // faster wrong answer does not justify:
    //
    //   1. Its pivot test compared every column against the SINGLE largest
    //      diagonal of XtX. That diagonal belongs to the intercept column and
    //      equals the window length, so with factor values around 1e-6 the
    //      threshold (1e-12 * window) landed right on top of the factor
    //      columns' own magnitudes and the whole window was declared singular.
    //      Measured: all-NaN from this kernel where the NumPy fallback
    //      returned 1.516061 for the same input. Column-pivoted QR ranks each
    //      column by its OWN remaining norm, so the answer is now invariant to
    //      the factors' scale (verified from 1e-12 through 1e12).
    //
    //   2. Forming XtX squares the condition number. A design QR handles to
    //      full precision could be numerically singular by the time Cholesky
    //      saw it -- and normal equations built by rank-1 updates also
    //      accumulate drift, which is why the old code needed a periodic
    //      full recompute to paper over it. Neither problem exists here.
    //
    // COST: this is a real regression, measured, not hand-waved. Per-window QR
    // is O(n * window * p^2) against the rank-1 update's O(n * p^2):
    //
    //     bars   factors  window     old      QR    ratio
    //     5000      4       60     0.51ms  8.93ms   17.7x
    //     5000      4      252     0.57ms 35.04ms   62.1x
    //    20000      4       60     2.44ms 38.26ms   15.7x
    //   100000      4       60    12.30ms 232.2ms   18.9x
    //
    // That is the price of a correct answer here, and it is paid knowingly: the
    // old path was 17-62x faster at returning all-NaN for any window whose
    // factors happened to be smaller than about 1e-6, and at disagreeing with
    // the NumPy fallback about whether a rank-deficient window has an answer.
    // Absolute cost stays in the single-digit-milliseconds range for typical
    // series, but a wide screener will notice, and the obvious next step if it
    // does is QR update/downdate across the sliding window (which restores
    // O(p^2) per bar without giving back the conditioning or the rank policy).
    // Deliberately not attempted here -- correctness first, then profile.
    const std::size_t window_sz = static_cast<std::size_t>(window);
    const std::size_t p_sz      = static_cast<std::size_t>(p);
    const int         W         = window;  // rows per solve; == window

    // Allocated once for the whole sweep, refilled per window.
    std::vector<double> A(numerics::checked_mul(window_sz, p_sz,
        "rolling_factor_loadings: design matrix size"));
    std::vector<double> b(window_sz);
    std::vector<int>    perm(p_sz);
    std::vector<double> qr_scratch;

    for (std::size_t i = window_sz - 1; i < n; ++i) {
        const std::size_t start = i - window_sz + 1;

        for (std::size_t r = 0; r < window_sz; ++r) {
            const double* fi = factors + (start + r) * k;
            double*       rp = A.data() + r * p_sz;
            rp[0] = 1.0;
            for (std::size_t c = 0; c < k; ++c) rp[c + 1] = fi[c];
            b[r] = y[start + r];
        }

        const auto sol = qr::lstsq(A.data(), b.data(), W, p, perm.data(), qr_scratch);

        // Rank-deficient window (duplicated or perfectly collinear factors):
        // leave the row at NaN. This is the ONE rank policy, shared with the
        // NumPy fallback in analysis/multi_factor.py, which now checks the
        // rank lstsq reports and emits NaN too. The backends used to disagree
        // outright here -- C++ produced NaN while NumPy produced its
        // minimum-norm solution -- so the same call returned numbers or blanks
        // depending only on whether the extension had been built. A
        // minimum-norm coefficient vector is an arbitrary member of an
        // infinite solution set, not an estimated factor loading, so NaN is
        // the honest answer and is now the answer both paths give.
        if (!sol.full_rank) continue;

        for (int j = 0; j < p; ++j)
            out[i * p_sz + static_cast<std::size_t>(j)] =
                sol.beta[static_cast<std::size_t>(j)];
    }
}

std::vector<double> rolling_factor_loadings(
    const double* y,
    const double* factors,
    std::size_t   n,
    std::size_t   k,
    int           window)
{
    const int p = static_cast<int>(k) + 1;
    std::vector<double> result(n * static_cast<std::size_t>(p));
    rolling_factor_loadings_into(y, factors, n, k, window, result.data());
    return result;
}

// ── rolling_beta ─────────────────────────────────────────────────────────────

void rolling_beta_into(
    const double* SQT_RESTRICT y,
    const double* SQT_RESTRICT x,
    std::size_t   n,
    int           window,
    double* SQT_RESTRICT       out)
{
    std::fill(out, out + n, kNaN);
    if (window < 2 || n < static_cast<std::size_t>(window)) return;

    // beta = cov(x,y) / var(x)
    //      = [W*Sxy - Sx*Sy] / [W*Sxx - Sx^2]
    // where W = window (constant), maintained with O(1) sliding updates.
    //
    // Raw-moment sums on the *unshifted* x/y suffer catastrophic
    // cancellation for a large-baseline series -- e.g. a ~1e9-level x with
    // a genuine beta near 1.5 previously came out as -0.003, and the
    // denominator collapsed to exactly zero at a ~1e12 offset (verified by
    // hand). Shifting both x and y by per-window reference points (their
    // own first values, so x/y - c stays close to the window's actual
    // *variation*, not its absolute level) before accumulating fixes this
    // the same way as bollinger_bands' fix in indicators.cpp -- periodic
    // full recompute every `window` bars both re-centers the shift and
    // bounds floating-point drift, matching rolling_factor_loadings'
    // existing periodic-refresh idiom elsewhere in this file.
    const double W = static_cast<double>(window);
    double cx = 0.0, cy = 0.0;
    double Sx = 0.0, Sy = 0.0, Sxy = 0.0, Sxx = 0.0;
    std::size_t since_refresh = 0;

    // Runtime ISA dispatch (item L): use the AVX2+FMA reduction when the
    // actual CPU supports it, otherwise the portable scalar path -- a
    // single build/wheel gets the fast path on capable hardware without
    // risking an illegal-instruction crash on older CPUs the way the
    // opt-in SQT_NATIVE_ARCH compile flag would. NOT bit-identical between
    // the two paths (SIMD lane accumulation reorders the sum) -- verified
    // via a tolerance gate, not assumed; see tests/test_cpp_regression.py.
    const bool use_avx2 = detect_isa_features().avx2;

    auto recompute_window = [&](std::size_t start) {
        cx = x[start];
        cy = y[start];
        if (use_avx2) {
            rolling_beta_reduce_avx2(x, y, start, window, cx, cy, Sx, Sy, Sxy, Sxx);
        } else {
            Sx = Sy = Sxy = Sxx = 0.0;
            // Vectorization hint only, not a functional requirement -- a
            // 4-accumulator reduction the compiler may already auto-vectorize
            // at -O3/-march=native without it. MSVC's default /openmp only
            // implements OpenMP 2.0, which doesn't recognize `omp simd` (that's
            // 4.0+) -- confirmed this is a hard C7660 compile error there
            // (requires /openmp:experimental), not a silent no-op as initially
            // assumed, so this pragma is scoped to non-MSVC compilers only
            // rather than pulling in a project-wide experimental-flag change
            // for one vectorization hint.
#if defined(_OPENMP) && !defined(_MSC_VER)
            #pragma omp simd reduction(+:Sx,Sy,Sxy,Sxx)
#endif
            for (std::size_t j = start; j < start + static_cast<std::size_t>(window); ++j) {
                const double xd = x[j] - cx;
                const double yd = y[j] - cy;
                Sx  += xd;
                Sy  += yd;
                Sxy += xd * yd;
                Sxx += xd * xd;
            }
        }
        since_refresh = 0;
    };

    auto write_beta = [&](std::size_t i) {
        const double denom = W * Sxx - Sx * Sx;
        // Relative-epsilon threshold scaled to the denominator's own natural
        // magnitude (W*Sxx), not a fixed absolute 1e-14 -- the same
        // convention numerics.hpp exists to enforce. A fixed absolute cut
        // doesn't scale: it is simultaneously too lenient for large-magnitude
        // inputs and too aggressive for genuinely well-conditioned
        // small-magnitude ones. The helper's own max(scale, 1.0) floor
        // reintroduced exactly that second failure below unit scale, and is
        // gone -- rolling_beta on 1e-8-scale data used to return NaN for a
        // beta of exactly 2.
        if (!numerics::is_negligible_pivot(denom, W * Sxx))
            out[i] = (W * Sxy - Sx * Sy) / denom;
    };

    // Seed first window
    recompute_window(0);
    write_beta(static_cast<std::size_t>(window) - 1);

    // Slide. size_t throughout (not int): this loop is inherently serial
    // (Sx/Sy/Sxy/Sxx carry state across iterations, never OpenMP-
    // parallelized) and i/old can exceed INT_MAX for a large series.
    const std::size_t window_sz = static_cast<std::size_t>(window);
    for (std::size_t i = window_sz; i < n; ++i) {
        const std::size_t old = i - window_sz;
        const double xdi = x[i] - cx, ydi = y[i] - cy;
        const double xdo = x[old] - cx, ydo = y[old] - cy;
        Sx  += xdi - xdo;
        Sy  += ydi - ydo;
        Sxy += xdi * ydi - xdo * ydo;
        Sxx += xdi * xdi - xdo * xdo;
        ++since_refresh;

        if (since_refresh >= window_sz) {
            recompute_window(old + 1);
        }

        write_beta(i);
    }
}

std::vector<double> rolling_beta(
    const double* y,
    const double* x,
    std::size_t   n,
    int           window)
{
    std::vector<double> result(n);
    rolling_beta_into(y, x, n, window, result.data());
    return result;
}

}  // namespace sqt
