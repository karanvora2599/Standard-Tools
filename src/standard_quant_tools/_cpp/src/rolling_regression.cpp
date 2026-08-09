#include "sqt/rolling_regression.hpp"

#include "sqt/isa_dispatch.hpp"
#include "sqt/numerics.hpp"
#include "sqt/rolling_beta_avx2.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace sqt {

namespace {
    constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

    // ── Cholesky solve: A * beta = b ─────────────────────────────────────────
    // A is (p×p) symmetric positive definite (stored row-major).
    // Returns false if A is singular or near-singular (diagonal entry not
    // sufficiently positive relative to A's own magnitude -- see below).
    //
    // L_scratch/z_scratch are caller-owned buffers reused across every call
    // in a rolling-window loop (sized once, outside the loop) instead of
    // being freshly heap-allocated per call. No zero-fill of L_scratch is
    // needed even though it may hold stale values from a previous call:
    // every read of L[i*p+j] below (always j<=i) is to an entry this SAME
    // call already wrote earlier in its own iteration order (row i's entries
    // are written left-to-right before being read by a later row, and the
    // i==j diagonal entry is always written before any later row reads it) —
    // the old `std::vector<double> L(p*p, 0.0)` zero-fill was never actually
    // load-bearing for correctness, only for giving L a defined initial size.
    bool cholesky_solve(
        const std::vector<double>& A,
        const std::vector<double>& b,
        std::vector<double>&       beta,
        int p,
        std::vector<double>&       L_scratch,
        std::vector<double>&       z_scratch)
    {
        // Relative-epsilon singularity threshold, scaled to A's own
        // original diagonal magnitude -- replaces a fixed absolute
        // `s <= 1e-14` threshold that didn't scale with A's magnitude (a
        // matrix with e.g. ~1e9-scale factor values could have genuinely
        // significant diagonal entries the old absolute threshold would
        // never see as "small," while a well-conditioned but small-
        // magnitude matrix could be rejected too aggressively). Deliberately
        // NOT numerics::is_negligible_pivot() here: that helper tests
        // |value| against the threshold (correct for Gaussian-elimination
        // pivots, which can legitimately be negative), whereas a Cholesky
        // diagonal entry must be POSITIVE before its sqrt below -- a large-
        // magnitude negative `s` (definitely not positive-definite) must
        // still fail this check, exactly as the original `s <= 1e-14` did.
        double scale = 0.0;
        for (int i = 0; i < p; ++i) scale = std::max(scale, std::abs(A[i * p + i]));
        const double rel_eps = 1e-12;
        const double min_diag = rel_eps * std::max(scale, 1.0);

        std::vector<double>& L = L_scratch;
        L.resize(static_cast<std::size_t>(p) * static_cast<std::size_t>(p));
        for (int i = 0; i < p; ++i) {
            for (int j = 0; j <= i; ++j) {
                double s = A[i * p + j];
                for (int kk = 0; kk < j; ++kk)
                    s -= L[i * p + kk] * L[j * p + kk];
                if (i == j) {
                    if (s <= min_diag) return false;
                    L[i * p + i] = std::sqrt(s);
                } else {
                    L[i * p + j] = s / L[j * p + j];
                }
            }
        }
        // Forward: L z = b
        std::vector<double>& z = z_scratch;
        z.resize(static_cast<std::size_t>(p));
        for (int i = 0; i < p; ++i) {
            double s = b[i];
            for (int j = 0; j < i; ++j) s -= L[i * p + j] * z[j];
            z[i] = s / L[i * p + i];
        }
        // Back: L' beta = z
        beta.resize(p);
        for (int i = p - 1; i >= 0; --i) {
            double s = z[i];
            for (int j = i + 1; j < p; ++j) s -= L[j * p + i] * beta[j];
            beta[i] = s / L[i * p + i];
        }
        return true;
    }

    // ── Build XtX (p×p) and Xty (p) from a contiguous range of bars ──────────
    // Design matrix row for bar i: xi = [1, factors[i*k], ..., factors[i*k+k-1]].
    //
    // Only the lower triangle (c <= r) is computed. cholesky_solve()'s
    // decomposition loop (`for i: for j <= i: ... A[i*p+j] ...`) never reads
    // XtX[r*p+c] for c > r, so the upper triangle is provably dead work, not
    // merely redundant-but-mirrorable -- no mirror step is needed either,
    // since nothing downstream ever reads those entries.
    void build_normal_equations(
        const double* y,
        const double* factors,
        std::size_t start, std::size_t end,
        int k, int p,
        std::vector<double>& XtX,
        std::vector<double>& Xty)
    {
        std::fill(XtX.begin(), XtX.end(), 0.0);
        std::fill(Xty.begin(), Xty.end(), 0.0);
        // start/end are bar indices into the full n-length series -- size_t,
        // not int, since n (and so these indices) can exceed INT_MAX for a
        // large series.
        for (std::size_t i = start; i < end; ++i) {
            const double* fi = factors + i * static_cast<std::size_t>(k);
            for (int r = 0; r < p; ++r) {
                const double xr = (r == 0) ? 1.0 : fi[r - 1];
                for (int c = 0; c <= r; ++c) {
                    XtX[r * p + c] += xr * ((c == 0) ? 1.0 : fi[c - 1]);
                }
                Xty[r] += xr * y[i];
            }
        }
    }
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

    std::vector<double> XtX(p * p), Xty(p);
    std::vector<double> beta;
    std::vector<double> L_scratch, z_scratch;  // reused across every cholesky_solve call below

    // Recompute XtX from scratch every `window` steps to prevent drift.
    // size_t throughout the slide loop below (not int): this loop is
    // inherently serial (XtX/Xty carry state across iterations, so it's
    // never OpenMP-parallelized) and i/old can exceed INT_MAX for a large
    // series -- no signed-induction-variable constraint applies here.
    const std::size_t window_sz = static_cast<std::size_t>(window);
    const std::size_t refresh   = window_sz;

    // ── Seed: first full window ───────────────────────────────────────────────
    build_normal_equations(y, factors, 0, window_sz, static_cast<int>(k), p, XtX, Xty);
    if (cholesky_solve(XtX, Xty, beta, p, L_scratch, z_scratch)) {
        for (int j = 0; j < p; ++j)
            out[(window_sz - 1) * static_cast<std::size_t>(p) + static_cast<std::size_t>(j)] = beta[j];
    }

    // ── Slide window ─────────────────────────────────────────────────────────
    for (std::size_t i = window_sz; i < n; ++i) {
        const std::size_t old = i - window_sz;

        if ((i - window_sz + 1) % refresh == 0) {
            // Periodic full recompute to flush floating-point accumulation
            build_normal_equations(y, factors, old + 1, i + 1,
                                   static_cast<int>(k), p, XtX, Xty);
        } else {
            // Rank-1 update: add bar i, remove bar old.
            // Same lower-triangle-only reasoning as build_normal_equations()
            // above -- the upper triangle (c > r) is never read by
            // cholesky_solve(), so updating it is dead work.
            const double* fi_new = factors + i   * k;
            const double* fi_old = factors + old * k;
            for (int r = 0; r < p; ++r) {
                const double xr_n = (r == 0) ? 1.0 : fi_new[r - 1];
                const double xr_o = (r == 0) ? 1.0 : fi_old[r - 1];
                for (int c = 0; c <= r; ++c) {
                    const double xc_n = (c == 0) ? 1.0 : fi_new[c - 1];
                    const double xc_o = (c == 0) ? 1.0 : fi_old[c - 1];
                    XtX[r * p + c] += xr_n * xc_n - xr_o * xc_o;
                }
                Xty[r] += xr_n * y[i] - xr_o * y[old];
            }
        }

        if (cholesky_solve(XtX, Xty, beta, p, L_scratch, z_scratch)) {
            for (int j = 0; j < p; ++j)
                out[i * p + j] = beta[j];
        }
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
#if defined(SQT_HAS_OPENMP) && !defined(_MSC_VER)
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
        // convention numerics.hpp exists to enforce and that cholesky_solve
        // above already uses. A fixed absolute cut doesn't scale: it is
        // simultaneously too lenient for large-magnitude inputs and too
        // aggressive for genuinely well-conditioned small-magnitude ones.
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
