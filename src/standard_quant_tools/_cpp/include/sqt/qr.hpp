#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

// ── Shared rank-revealing least-squares backend ──────────────────────────────
//
// Householder QR with column pivoting, replacing the normal-equations solves
// (X'X b = X'y via Gaussian elimination or Cholesky) that the kernels used to
// carry independently.
//
// Why this exists, rather than each kernel keeping its own solve:
//
//   1. CONDITIONING. Forming X'X squares the condition number of X. A design
//      that a QR handles to full double precision can be numerically singular
//      by the time it reaches a Cholesky factorization -- which is not
//      hypothetical: rolling_factor_loadings returned all-NaN for factor
//      values around 1e-6 while the NumPy fallback returned correct
//      coefficients from the same inputs, purely because the Cholesky pivot
//      test compared every factor column against the ONE largest diagonal of
//      X'X (the intercept column's, equal to the window length) instead of
//      against each column's own magnitude. Column-pivoted QR ranks columns by
//      their own remaining norm, so a small-magnitude but perfectly
//      well-conditioned column is no longer rejected because a large-magnitude
//      column sits next to it.
//
//   2. ONE RANK POLICY. Every caller gets the same `rank`, computed the same
//      way, so "is this system rank-deficient?" has a single answer across
//      rolling factor loadings, cointegration OLS and the ADF regressions,
//      instead of three ad-hoc thresholds that disagreed with each other and
//      with the NumPy fallback.
//
//   3. RSS WITHOUT CANCELLATION. rss falls out of the transformed response as
//      a sum of squares of its own tail, so it is non-negative by
//      construction. The old RSS = y'y - beta'X'y form is a difference of two
//      large nearly-equal quantities and could land materially negative on an
//      ill-conditioned design -- which cointegration.cpp had to detect and
//      special-case precisely because it happens in practice.
//
// Sizes here are small (k is the regressor count: <= ~20 for an ADF
// regression, <= ~10 for factor loadings), so the O(T*k^2) factorization and
// the O(k^2) triangular solve are not worth specializing further.
namespace sqt::qr {

struct LstsqResult {
    std::vector<double> beta;  // length k, ORIGINAL column order; all-NaN when rank < k
    // Residual sum of squares, >= 0 by construction. NaN when rank < k: the
    // tail of Q'b past column k is the correct RSS only for a full-rank
    // solve, and for a rank-deficient one it silently OMITS the components
    // in [rank, k) that no column of R can reach -- an underestimate that
    // looks like an ordinary number. Nothing reads it in that state today
    // (every caller checks full_rank first), which is precisely why a wrong
    // value here would have gone unnoticed by the next caller that doesn't.
    double rss  = 0.0;
    int    rank = 0;           // numerical rank of X
    int    nobs = 0;           // rows T
    int    ncoef = 0;          // columns k
    bool   full_rank = false;  // rank == k

    // 2-norm each column was divided by before factorizing (see the
    // equilibration note in lstsq). Length k, ORIGINAL column order. Needed to
    // convert anything read off the factorization -- xtx_inv_diag does this --
    // back into the caller's units.
    std::vector<double> col_scale;

    // Residual degrees of freedom for an unbiased variance estimate.
    int df() const { return nobs - ncoef; }
};

// Least squares via Householder QR with column pivoting.
//
// @param A       T x k design matrix, ROW-MAJOR. Overwritten with the
//                factorization: its leading k x k upper triangle becomes R, in
//                PIVOTED column order.
// @param b       Length-T response. Overwritten with Q'b.
// @param T, k    Dimensions. Requires T >= k >= 1.
// @param perm    Scratch of length k; on return perm[j] is the ORIGINAL column
//                index that ended up in pivoted position j.
// @param rel_tol Rank tolerance RELATIVE to the largest |R_ii|. Deliberately no
//                absolute floor: a uniformly rescaled problem must produce a
//                uniformly rescaled answer, so the test is a pure ratio. A
//                max(scale, 1.0) style floor silently turns this into an
//                absolute test for data below unit magnitude and rejects
//                perfectly well-conditioned small-scale systems.
//
// Returns beta in ORIGINAL column order (un-pivoted). When the design is
// rank-deficient, beta is all-NaN and full_rank is false: callers decide what a
// rank-deficient window means for them rather than silently receiving one
// arbitrary member of an infinite solution set.
//
// The `scratch` overload lets a rolling loop hoist the length-T reflector
// buffer out of the per-window call, so sliding a window over a long series
// does not allocate once per bar.
inline LstsqResult lstsq(double* A, double* b, int T, int k, int* perm,
                         std::vector<double>& scratch, double rel_tol = 1e-12)
{
    LstsqResult res;
    res.nobs  = T;
    res.ncoef = k;
    res.beta.assign(static_cast<std::size_t>(k < 0 ? 0 : k),
                    std::numeric_limits<double>::quiet_NaN());
    if (k < 1 || T < k) return res;

    const std::size_t k_sz = static_cast<std::size_t>(k);
    auto at = [A, k_sz](int i, int j) -> double& {
        return A[static_cast<std::size_t>(i) * k_sz + static_cast<std::size_t>(j)];
    };

    for (int j = 0; j < k; ++j) perm[j] = j;
    scratch.resize(static_cast<std::size_t>(T));
    std::vector<double>& v = scratch;

    // ── Column equilibration ────────────────────────────────────────────
    // Each column is scaled to unit 2-norm before the factorization, and the
    // coefficients are scaled back at the end.
    //
    // Without this the rank test -- necessarily a ratio against the largest
    // |R_ii| -- cannot tell "these two columns are collinear" apart from
    // "these two columns are measured in units 1e13 apart". A design of
    // [intercept, factors*1e13] is perfectly well conditioned, but its
    // intercept column's R_ii is 1e13 times smaller than the factor columns',
    // so an un-equilibrated test declares it rank-deficient and returns NaN
    // where NumPy reports rank 4/4 and a correct answer. Equilibrating makes
    // the rank decision unit-free, which is the only thing it should ever have
    // depended on.
    //
    // Note this is the SAME class of bug, in the opposite direction, as the
    // Cholesky pivot test this file replaced: that one judged small columns
    // against a large one and rejected them; this one would have judged a
    // small column against large ones and rejected it.
    res.col_scale.assign(k_sz, 1.0);
    for (int j = 0; j < k; ++j) {
        double nrm = 0.0;
        for (int i = 0; i < T; ++i) { const double a = at(i, j); nrm += a * a; }
        nrm = std::sqrt(nrm);
        // An exactly-zero column has no scale to normalize by; leave it at 1.0
        // and let the rank test below reject it, which it will.
        if (nrm > 0.0 && std::isfinite(nrm)) {
            res.col_scale[static_cast<std::size_t>(j)] = nrm;
            const double inv = 1.0 / nrm;
            for (int i = 0; i < T; ++i) at(i, j) *= inv;
        }
    }

    for (int j = 0; j < k; ++j) {
        // ── Column pivoting ─────────────────────────────────────────────
        // Remaining column norms are recomputed rather than downdated.
        // Downdating is the usual optimization, but it loses accuracy exactly
        // on the ill-conditioned designs this factorization exists to handle,
        // and k is small enough that the extra work is irrelevant next to
        // getting the rank decision right.
        int    piv  = j;
        double best = -1.0;
        for (int c = j; c < k; ++c) {
            double s = 0.0;
            for (int i = j; i < T; ++i) { const double a = at(i, c); s += a * a; }
            if (s > best) { best = s; piv = c; }
        }
        if (piv != j) {
            for (int i = 0; i < T; ++i) std::swap(at(i, j), at(i, piv));
            std::swap(perm[j], perm[piv]);
        }

        // ── Householder reflector for the sub-column A[j.., j] ───────────
        double normx = 0.0;
        for (int i = j; i < T; ++i) { const double a = at(i, j); normx += a * a; }
        normx = std::sqrt(normx);
        if (!(normx > 0.0)) continue;  // exactly-zero column: R_jj stays 0

        const double ajj = at(j, j);
        // Sign chosen away from ajj so v[j] is formed by addition, not by
        // subtracting two nearly-equal numbers.
        const double alpha = (ajj > 0.0) ? -normx : normx;

        v[static_cast<std::size_t>(j)] = ajj - alpha;
        for (int i = j + 1; i < T; ++i) v[static_cast<std::size_t>(i)] = at(i, j);

        double vtv = 0.0;
        for (int i = j; i < T; ++i) {
            const double vi = v[static_cast<std::size_t>(i)];
            vtv += vi * vi;
        }
        if (!(vtv > 0.0)) continue;

        for (int c = j; c < k; ++c) {
            double s = 0.0;
            for (int i = j; i < T; ++i) s += v[static_cast<std::size_t>(i)] * at(i, c);
            s = 2.0 * s / vtv;
            for (int i = j; i < T; ++i) at(i, c) -= s * v[static_cast<std::size_t>(i)];
        }
        double sb = 0.0;
        for (int i = j; i < T; ++i)
            sb += v[static_cast<std::size_t>(i)] * b[static_cast<std::size_t>(i)];
        sb = 2.0 * sb / vtv;
        for (int i = j; i < T; ++i)
            b[static_cast<std::size_t>(i)] -= sb * v[static_cast<std::size_t>(i)];
    }

    // ── Rank from the R diagonal ────────────────────────────────────────
    // Column pivoting orders |R_ii| non-increasingly, so R[0][0] is the
    // largest and the test is a pure ratio against it.
    double rmax = 0.0;
    for (int j = 0; j < k; ++j) rmax = std::max(rmax, std::abs(at(j, j)));
    const double rtol = rel_tol * rmax;
    int rank = 0;
    for (int j = 0; j < k; ++j) {
        if (std::abs(at(j, j)) > rtol) ++rank;
        else break;  // pivoting guarantees the tail is the small end
    }
    res.rank      = rank;
    res.full_rank = (rank == k);

    // ── RSS from the transformed response's tail ────────────────────────
    // The components of Q'b that no column of R can reach. Non-negative by
    // construction: no cancellation, no clamping, no "went unexpectedly
    // negative" branch needed.
    //
    // Valid ONLY at full rank -- see LstsqResult::rss. Below full rank the
    // sum from k omits the [rank, k) components, so it understates the true
    // residual; report NaN rather than a plausible smaller number.
    if (!res.full_rank) {
        res.rss = std::numeric_limits<double>::quiet_NaN();
        return res;  // beta stays all-NaN
    }
    double rss = 0.0;
    for (int i = k; i < T; ++i) {
        const double bi = b[static_cast<std::size_t>(i)];
        rss += bi * bi;
    }
    res.rss = rss;

    // ── Back-substitution, then un-pivot ────────────────────────────────
    std::vector<double> x(k_sz);
    for (int i = k - 1; i >= 0; --i) {
        double s = b[static_cast<std::size_t>(i)];
        for (int j = i + 1; j < k; ++j) s -= at(i, j) * x[static_cast<std::size_t>(j)];
        x[static_cast<std::size_t>(i)] = s / at(i, i);
    }
    // Un-pivot AND un-equilibrate: solving with column j scaled by 1/s_j
    // returns a coefficient scaled by s_j, so divide it back out.
    for (int j = 0; j < k; ++j) {
        const std::size_t orig = static_cast<std::size_t>(perm[j]);
        res.beta[orig] = x[static_cast<std::size_t>(j)] / res.col_scale[orig];
    }

    return res;
}

// Allocating convenience form, for one-shot solves where the extra length-T
// buffer is not worth threading through the call site.
inline LstsqResult lstsq(double* A, double* b, int T, int k, int* perm,
                         double rel_tol = 1e-12)
{
    std::vector<double> scratch;
    return lstsq(A, b, T, k, perm, scratch, rel_tol);
}

// ── Nested-model RSS from ONE factorization ─────────────────────────────────
//
// When model j is exactly the first j columns of model k on the SAME rows --
// which is what a lag-selection sweep produces -- every model's residual sum
// of squares can be read off one factorization of the largest design:
//
//     RSS(j) = sum_{i >= j} (Q'b)_i^2
//
// The identity is exact, not an approximation. Householder reflection number
// m acts only on rows m..T-1, so applying reflections j+1..k leaves the tail
// sum from index j untouched: the value that sum has after the FULL
// factorization is the value it had after j reflections, which is by
// definition the residual of the first-j-columns model.
//
// This replaces k separate factorizations costing O(T*k^3) in total with one
// costing O(T*k^2). For adf_test's lag sweep, where k grows as n^(1/4) via
// Schwert's rule, that is the difference between O(n^1.75) and O(n^1.25) --
// measured before the change, `engle_granger` was O(n^1.99) and 1953x of its
// cost at n=8000 was this sweep.
//
// PIVOTING IS DELIBERATELY ABSENT. Column pivoting reorders columns, which
// destroys the nesting the identity depends on; a pivoted factorization of
// the full design says nothing about any prefix of it. The consequence is
// that the R diagonal is no longer ordered, so the rank test cannot stop at
// the first small entry the way lstsq's does -- see below.
//
// @param A     T x k design, ROW-MAJOR. Overwritten with the factorization.
// @param b     Length-T response. Overwritten with Q'b.
// @param out_rss        Length k+1. out_rss[j] = RSS of the first-j-columns
//                       model; NaN where j exceeds min(k, T).
// @param out_full_rank  Length k+1. Whether that prefix is full rank.
// @param rel_tol        Rank tolerance relative to the largest |R_ii|, same
//                       pure-ratio convention as lstsq.
inline void lstsq_nested_rss(double* A, double* b, int T, int k,
                             double* out_rss, unsigned char* out_full_rank,
                             double rel_tol = 1e-12)
{
    constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
    for (int j = 0; j <= (k < 0 ? 0 : k); ++j) { out_rss[j] = kNaN; out_full_rank[j] = 0; }
    if (k < 1 || T < 1) return;

    const std::size_t k_sz = static_cast<std::size_t>(k);
    auto at = [A, k_sz](int i, int j) -> double& {
        return A[static_cast<std::size_t>(i) * k_sz + static_cast<std::size_t>(j)];
    };

    // Column equilibration, for the same reason lstsq does it: the rank test
    // is a ratio against the largest |R_ii|, and without equilibration it
    // cannot tell "these columns are collinear" from "these columns are
    // measured in units 1e13 apart".
    //
    // RSS is invariant under it. Scaling a column by 1/s scales that
    // coefficient by s and leaves the fitted values -- and therefore the
    // residual -- untouched, so the equilibration affects the rank decision
    // only, which is exactly what it is here for.
    for (int j = 0; j < k; ++j) {
        double nrm = 0.0;
        for (int i = 0; i < T; ++i) { const double a = at(i, j); nrm += a * a; }
        nrm = std::sqrt(nrm);
        if (nrm > 0.0 && std::isfinite(nrm)) {
            const double inv = 1.0 / nrm;
            for (int i = 0; i < T; ++i) at(i, j) *= inv;
        }
    }

    // Only min(k, T) reflections exist. A caller sweeping nested models can
    // legitimately ask about prefixes shorter than T while the full design is
    // over-parameterized, so this is a supported case, not an error.
    const int kk = (k < T) ? k : T;

    std::vector<double> v(static_cast<std::size_t>(T));
    for (int j = 0; j < kk; ++j) {
        double normx = 0.0;
        for (int i = j; i < T; ++i) { const double a = at(i, j); normx += a * a; }
        normx = std::sqrt(normx);
        if (!(normx > 0.0)) continue;  // exactly-zero column: R_jj stays 0

        const double ajj = at(j, j);
        // Sign away from ajj, so v[j] is formed by addition rather than by
        // subtracting two nearly-equal numbers.
        const double alpha = (ajj > 0.0) ? -normx : normx;

        v[static_cast<std::size_t>(j)] = ajj - alpha;
        for (int i = j + 1; i < T; ++i) v[static_cast<std::size_t>(i)] = at(i, j);

        double vtv = 0.0;
        for (int i = j; i < T; ++i) {
            const double vi = v[static_cast<std::size_t>(i)];
            vtv += vi * vi;
        }
        if (!(vtv > 0.0)) continue;

        for (int c = j; c < k; ++c) {
            double s = 0.0;
            for (int i = j; i < T; ++i) s += v[static_cast<std::size_t>(i)] * at(i, c);
            s = 2.0 * s / vtv;
            for (int i = j; i < T; ++i) at(i, c) -= s * v[static_cast<std::size_t>(i)];
        }
        double sb = 0.0;
        for (int i = j; i < T; ++i)
            sb += v[static_cast<std::size_t>(i)] * b[static_cast<std::size_t>(i)];
        sb = 2.0 * sb / vtv;
        for (int i = j; i < T; ++i)
            b[static_cast<std::size_t>(i)] -= sb * v[static_cast<std::size_t>(i)];
    }

    // ── Rank, per prefix ────────────────────────────────────────────────
    // Unpivoted, so |R_ii| is NOT ordered and lstsq's "break at the first
    // small entry" shortcut does not apply. Prefix j is full rank iff EVERY
    // |R_ii| with i < j clears the tolerance, which is a running conjunction.
    double rmax = 0.0;
    for (int j = 0; j < kk; ++j) rmax = std::max(rmax, std::abs(at(j, j)));
    const double rtol = rel_tol * rmax;
    bool prefix_ok = true;
    out_full_rank[0] = 1;  // the empty model is trivially full rank
    for (int j = 0; j < kk; ++j) {
        if (!(std::abs(at(j, j)) > rtol)) prefix_ok = false;
        out_full_rank[j + 1] = prefix_ok ? 1u : 0u;
    }

    // ── RSS, per prefix ─────────────────────────────────────────────────
    // Accumulated from the tail forward so every value is a sum of squares
    // of a suffix: non-negative by construction, no cancellation.
    double acc = 0.0;
    for (int i = T - 1; i >= kk; --i) {
        const double bi = b[static_cast<std::size_t>(i)];
        acc += bi * bi;
    }
    out_rss[kk] = acc;
    for (int j = kk - 1; j >= 0; --j) {
        const double bj = b[static_cast<std::size_t>(j)];
        acc += bj * bj;
        out_rss[j] = acc;
    }
}

// Diagonal entry `idx` (in ORIGINAL column order) of (X'X)^{-1}, read off the
// same factorization lstsq just produced -- so a t-statistic and its
// coefficient always come from ONE decomposition rather than from a second,
// independently-conditioned solve.
//
// X P = Q R  =>  X'X = P R'R P'  =>  (X'X)^{-1} = P R^{-1} R^{-T} P', whose
// entry at pivoted position j is the squared norm of row j of R^{-1}.
//
// The factorization is of the EQUILIBRATED design, so the result is scaled
// back by the column norms lstsq recorded -- (X'X)^{-1} = D (Xs'Xs)^{-1} D with
// D = diag(1/s_j), so entry j picks up a 1/s_j^2.
//
// @param A     The factorized matrix lstsq wrote (leading k x k triangle = R).
// @param res   That same call's result, for its rank and column scales.
inline double xtx_inv_diag(const double* A, const LstsqResult& res,
                           const int* perm, int idx)
{
    constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
    const int k = res.ncoef;
    if (k < 1 || res.rank != k || idx < 0 || idx >= k) return kNaN;
    const std::size_t k_sz = static_cast<std::size_t>(k);
    auto at = [A, k_sz](int i, int j) -> double {
        return A[static_cast<std::size_t>(i) * k_sz + static_cast<std::size_t>(j)];
    };

    // Where did original column `idx` end up after pivoting?
    int pos = -1;
    for (int j = 0; j < k; ++j) if (perm[j] == idx) { pos = j; break; }
    if (pos < 0) return kNaN;

    // Row `pos` of R^{-1}. R^{-1} is upper triangular, so only columns >= pos
    // are nonzero.
    std::vector<double> row(k_sz, 0.0);
    row[static_cast<std::size_t>(pos)] = 1.0 / at(pos, pos);
    for (int j = pos + 1; j < k; ++j) {
        double s = 0.0;
        for (int l = pos; l < j; ++l) s += row[static_cast<std::size_t>(l)] * at(l, j);
        row[static_cast<std::size_t>(j)] = -s / at(j, j);
    }

    double s = 0.0;
    for (int j = pos; j < k; ++j) {
        const double rj = row[static_cast<std::size_t>(j)];
        s += rj * rj;
    }
    const double sc = res.col_scale[static_cast<std::size_t>(idx)];
    return s / (sc * sc);
}

}  // namespace sqt::qr
