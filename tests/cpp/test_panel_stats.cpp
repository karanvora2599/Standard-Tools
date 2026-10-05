/**
 * C++ unit tests for sqt::fit_preprocess_stats and
 * sqt::apply_preprocess_stats.
 *
 * These kernels replace a pandas expression, so the properties worth
 * asserting here are the ones a Python-side comparison would find hardest
 * to localize: the quantile interpolation rule, the ddof=1 divisor, NaN
 * being skipped by the moments but preserved by the transform, and the
 * degenerate columns where the Python path substitutes a value rather than
 * dividing by zero. The bit-level agreement with pandas itself is asserted
 * from Python, where pandas is available to compare against.
 *
 * Build:
 *   cmake -B build -DSQT_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
 *   cmake --build build --config Release
 *
 * Run directly:
 *   Windows : build\tests\cpp\Release\test_panel_stats.exe
 *   Linux   : ./build/tests/cpp/test_panel_stats
 *
 * Run via CTest:
 *   ctest --test-dir build --config Release -V
 */

#include "sqt/panel_stats.hpp"

#include "sqt/numerics.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

// ── Tiny assertion helpers ────────────────────────────────────────────────────

static int g_tests_run = 0;
static int g_tests_failed = 0;

static void expect(bool condition, const char* what) {
    ++g_tests_run;
    if (!condition) {
        ++g_tests_failed;
        std::printf("  FAIL: %s\n", what);
    }
}

static void expect_near(double got, double want, double tol, const char* what) {
    ++g_tests_run;
    const bool ok = (std::isnan(got) && std::isnan(want)) ||
                    std::fabs(got - want) <= tol;
    if (!ok) {
        ++g_tests_failed;
        std::printf("  FAIL: %s (got %.17g, want %.17g)\n", what, got, want);
    }
}

namespace {

struct Fitted {
    std::vector<double> lo, hi, mean, stdev;

    explicit Fitted(std::size_t n_cols)
        : lo(n_cols), hi(n_cols), mean(n_cols), stdev(n_cols) {}

    sqt::PreprocessStats view() {
        return sqt::PreprocessStats{lo.data(), hi.data(), mean.data(),
                                    stdev.data()};
    }
};

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();

}  // namespace

// ── Quantile interpolation ────────────────────────────────────────────────────

static void test_quantile_is_linearly_interpolated() {
    std::printf("test_quantile_is_linearly_interpolated\n");
    // 0..10 in one column. For q=0.25, h = (11-1)*0.25 = 2.5, so pandas
    // returns x[2] + 0.5*(x[3]-x[2]) = 2.5 -- NOT x[2]=2, which is what a
    // bare nth_element would give. This single case is the whole reason
    // interpolated_quantile exists.
    std::vector<double> values(11);
    for (std::size_t i = 0; i < 11; ++i) values[i] = static_cast<double>(i);

    Fitted fitted(1);
    const bool ok =
        sqt::fit_preprocess_stats(values.data(), 11, 1, 0.25, 0.75, fitted.view());
    expect(ok, "fit succeeds");
    expect_near(fitted.lo[0], 2.5, 1e-15, "q=0.25 interpolates to 2.5");
    expect_near(fitted.hi[0], 7.5, 1e-15, "q=0.75 interpolates to 7.5");
}

static void test_quantile_endpoints() {
    std::printf("test_quantile_endpoints\n");
    std::vector<double> values{5.0, 1.0, 4.0, 2.0, 3.0};
    Fitted fitted(1);
    sqt::fit_preprocess_stats(values.data(), 5, 1, 0.0, 1.0, fitted.view());
    expect_near(fitted.lo[0], 1.0, 1e-15, "q=0 is the minimum");
    expect_near(fitted.hi[0], 5.0, 1e-15, "q=1 is the maximum");
}

static void test_quantile_endpoints_on_a_long_column() {
    std::printf("test_quantile_endpoints_on_a_long_column\n");
    // Five values cannot catch a q=1 answer read from an unordered slot:
    // below 33 values nth_element insertion-sorts, which orders the buffer
    // as a side effect. 100 shuffled values can. Each (q_low, q_high) pair
    // runs the q_low partition first, which is what leaves slot n-1
    // holding something other than the maximum.
    const std::size_t n = 100;
    std::vector<double> values(n);
    // A fixed permutation of 0..99 (37 is coprime with 100), then scaled so
    // the extremes are not the obvious 0 and 99.
    for (std::size_t i = 0; i < n; ++i)
        values[i] = 0.25 + 1.5 * static_cast<double>((i * 37 + 11) % n);
    const double lo_true = 0.25;
    const double hi_true = 0.25 + 1.5 * 99.0;
    const double pairs[][2] = {{0.0, 1.0}, {0.01, 1.0}, {0.0, 0.99}};
    for (const auto& q : pairs) {
        Fitted fitted(1);
        sqt::fit_preprocess_stats(values.data(), n, 1, q[0], q[1], fitted.view());
        if (q[1] == 1.0)
            expect_near(fitted.hi[0], hi_true, 0.0, "q_high=1 is the maximum");
        if (q[0] == 0.0)
            expect_near(fitted.lo[0], lo_true, 0.0, "q_low=0 is the minimum");
    }
}

// ── Moments ───────────────────────────────────────────────────────────────────

static void test_std_uses_ddof_one() {
    std::printf("test_std_uses_ddof_one\n");
    // 1,2,3,4: mean 2.5. ddof=1 variance = 5/3, std = 1.29099...
    // ddof=0 would give sqrt(1.25) = 1.118..., so the two are easy to tell
    // apart. pandas' Series.std() is ddof=1 and fit_preprocessing uses it.
    std::vector<double> values{1.0, 2.0, 3.0, 4.0};
    Fitted fitted(1);
    sqt::fit_preprocess_stats(values.data(), 4, 1, 0.0, 1.0, fitted.view());
    expect_near(fitted.mean[0], 2.5, 1e-15, "mean of 1..4");
    expect_near(fitted.stdev[0], std::sqrt(5.0 / 3.0), 1e-15, "ddof=1 std");
}

static void test_moments_are_of_the_clipped_column() {
    std::printf("test_moments_are_of_the_clipped_column\n");
    // An extreme value pulled inside the winsorize bounds must not drag the
    // mean: the Python path clips first, then takes the moments.
    std::vector<double> values{0.0, 1.0, 2.0, 3.0, 1000.0};
    Fitted fitted(1);
    sqt::fit_preprocess_stats(values.data(), 5, 1, 0.0, 0.75, fitted.view());
    // q=0.75 over 5 points: h = 4*0.75 = 3 exactly, so hi = 3.0.
    expect_near(fitted.hi[0], 3.0, 1e-15, "upper bound is 3.0");
    // Clipped column is 0,1,2,3,3 -> mean 1.8.
    expect_near(fitted.mean[0], 1.8, 1e-15, "mean is of the CLIPPED column");
}

// ── Degenerate columns ────────────────────────────────────────────────────────

static void test_constant_column_gets_unit_std() {
    std::printf("test_constant_column_gets_unit_std\n");
    std::vector<double> values{7.0, 7.0, 7.0, 7.0};
    Fitted fitted(1);
    sqt::fit_preprocess_stats(values.data(), 4, 1, 0.01, 0.99, fitted.view());
    expect_near(fitted.mean[0], 7.0, 1e-15, "mean of a constant column");
    // Zero dispersion: 1.0 keeps the caller's division defined and leaves
    // the standardized value at 0, which is what the column deserves.
    expect_near(fitted.stdev[0], 1.0, 0.0, "constant column std is 1.0");

    std::vector<double> out(4);
    sqt::apply_preprocess_stats(values.data(), 4, 1, fitted.view(), out.data());
    for (std::size_t i = 0; i < 4; ++i)
        expect_near(out[i], 0.0, 1e-15, "constant column standardizes to 0");
}

static void test_single_row_column() {
    std::printf("test_single_row_column\n");
    std::vector<double> values{42.0};
    Fitted fitted(1);
    sqt::fit_preprocess_stats(values.data(), 1, 1, 0.01, 0.99, fitted.view());
    expect_near(fitted.lo[0], 42.0, 0.0, "single value is both bounds");
    expect_near(fitted.hi[0], 42.0, 0.0, "single value is both bounds");
    // One observation has no ddof=1 dispersion at all.
    expect_near(fitted.stdev[0], 1.0, 0.0, "one row -> std 1.0");
}

static void test_all_nan_column() {
    std::printf("test_all_nan_column\n");
    std::vector<double> values{kNaN, kNaN, kNaN};
    Fitted fitted(1);
    sqt::fit_preprocess_stats(values.data(), 3, 1, 0.01, 0.99, fitted.view());
    expect(std::isnan(fitted.lo[0]), "all-NaN column has NaN bounds");
    expect(std::isnan(fitted.mean[0]), "all-NaN column has NaN mean");
    expect_near(fitted.stdev[0], 1.0, 0.0, "all-NaN column still gets std 1.0");
}

// ── NaN and infinity handling ─────────────────────────────────────────────────

static void test_nan_is_skipped_by_the_moments() {
    std::printf("test_nan_is_skipped_by_the_moments\n");
    // Series.quantile and Series.std ignore missing values, so the moments
    // here must match those of {1,2,3,4} exactly.
    std::vector<double> with_gaps{1.0, kNaN, 2.0, 3.0, kNaN, 4.0};
    std::vector<double> without{1.0, 2.0, 3.0, 4.0};
    Fitted a(1), b(1);
    sqt::fit_preprocess_stats(with_gaps.data(), 6, 1, 0.0, 1.0, a.view());
    sqt::fit_preprocess_stats(without.data(), 4, 1, 0.0, 1.0, b.view());
    expect_near(a.mean[0], b.mean[0], 0.0, "NaN does not change the mean");
    expect_near(a.stdev[0], b.stdev[0], 0.0, "NaN does not change the std");
    expect_near(a.lo[0], b.lo[0], 0.0, "NaN does not change the bounds");
}

static void test_nan_survives_the_transform() {
    std::printf("test_nan_survives_the_transform\n");
    // Series.clip leaves a missing value missing rather than pinning it to
    // a bound -- so a gap must come out the far side still a gap, not a
    // fabricated observation sitting exactly on the winsorize boundary.
    std::vector<double> values{1.0, kNaN, 3.0};
    Fitted fitted(1);
    sqt::fit_preprocess_stats(values.data(), 3, 1, 0.0, 1.0, fitted.view());
    std::vector<double> out(3);
    sqt::apply_preprocess_stats(values.data(), 3, 1, fitted.view(), out.data());
    expect(std::isnan(out[1]), "NaN passes through apply untouched");
    expect(!std::isnan(out[0]) && !std::isnan(out[2]), "other rows transform");
}

static double from_bits(std::uint64_t bits) {
    double x;
    std::memcpy(&x, &bits, sizeof x);
    return x;
}

static std::uint64_t to_bits(double x) {
    std::uint64_t bits;
    std::memcpy(&bits, &x, sizeof bits);
    return bits;
}

static void test_is_nan_is_std_isnan() {
    std::printf("test_is_nan_is_std_isnan\n");
    // numerics::is_nan replaced std::isnan in apply's loop (CHANGELOG,
    // 2026-10-02). They must agree on every double; the edges of the NaN
    // range and a spread of random bit patterns stand in for all 2^64.
    const std::uint64_t edges[] = {
        0x0000000000000000ULL, 0x8000000000000000ULL,  // +-0
        0x0000000000000001ULL, 0x000FFFFFFFFFFFFFULL,  // subnormals
        0x7FEFFFFFFFFFFFFFULL, 0xFFEFFFFFFFFFFFFFULL,  // +-DBL_MAX
        0x7FF0000000000000ULL, 0xFFF0000000000000ULL,  // +-inf
        0x7FF0000000000001ULL, 0x7FF7FFFFFFFFFFFFULL,  // signalling NaN
        0x7FF8000000000000ULL, 0xFFF8000000000000ULL,  // quiet NaN, both signs
        0x7FFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,  // all-ones payloads
        0xFFF0000000000001ULL, 0x3FF0000000000000ULL,  // -sNaN, 1.0
    };
    bool all_agree = true;
    for (const std::uint64_t b : edges) {
        const double x = from_bits(b);
        all_agree = all_agree && (sqt::numerics::is_nan(x) == std::isnan(x));
    }
    expect(all_agree, "is_nan agrees with std::isnan at every edge pattern");
    std::uint64_t state = 0x9E3779B97F4A7C15ULL;
    bool random_agree = true;
    for (int i = 0; i < 1'000'000; ++i) {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        // Half the draws forced into the all-ones exponent, where NaN lives.
        const std::uint64_t b = (i % 2) ? (state | 0x7FF0000000000000ULL) : state;
        const double x = from_bits(b);
        random_agree = random_agree && (sqt::numerics::is_nan(x) == std::isnan(x));
    }
    expect(random_agree, "is_nan agrees with std::isnan on a million bit patterns");
}

static void test_every_nan_passes_through_apply_bit_for_bit() {
    std::printf("test_every_nan_passes_through_apply_bit_for_bit\n");
    // Whatever NaN arrives -- either sign, any payload, signalling -- is the
    // value that leaves, bit for bit; the infinities beside them are clipped
    // to the bounds like any other value.
    const std::uint64_t nan_bits[] = {
        0x7FF8000000000000ULL, 0xFFF8000000000000ULL, 0x7FF8000000000123ULL,
        0xFFFFFFFFFFFFFFFFULL, 0x7FF0000000000001ULL,
    };
    std::vector<double> values{1.0, 2.0, 3.0, 4.0, 5.0};
    Fitted fitted(1);
    sqt::fit_preprocess_stats(values.data(), 5, 1, 0.0, 1.0, fitted.view());
    std::vector<double> panel;
    for (const std::uint64_t b : nan_bits) panel.push_back(from_bits(b));
    panel.push_back(kInf);
    panel.push_back(-kInf);
    panel.push_back(-0.0);
    std::vector<double> out(panel.size());
    sqt::apply_preprocess_stats(panel.data(), panel.size(), 1, fitted.view(), out.data());
    bool kept = true;
    for (std::size_t i = 0; i < 5; ++i) kept = kept && to_bits(out[i]) == nan_bits[i];
    expect(kept, "every NaN leaves with the bits it arrived with");
    const double sd = fitted.stdev[0];
    expect(out[5] == (5.0 - 3.0) / sd, "+inf is clipped to the upper bound");
    expect(out[6] == (1.0 - 3.0) / sd, "-inf is clipped to the lower bound");
    expect(out[7] == (1.0 - 3.0) / sd, "-0.0 is a value, clipped like one");
}

static void test_infinity_is_not_treated_as_missing() {
    std::printf("test_infinity_is_not_treated_as_missing\n");
    // pandas treats only NaN as missing. An infinity is a real, if
    // pathological, order statistic and must participate in the quantile.
    std::vector<double> values{1.0, 2.0, 3.0, kInf};
    Fitted fitted(1);
    sqt::fit_preprocess_stats(values.data(), 4, 1, 0.0, 1.0, fitted.view());
    expect(std::isinf(fitted.hi[0]), "inf is the maximum, not skipped");
}

// ── Multi-column layout ───────────────────────────────────────────────────────

static void test_columns_are_independent() {
    std::printf("test_columns_are_independent\n");
    // Row-major (4, 3): each column has a different scale, and reading the
    // wrong stride would mix them. Column 1 is deliberately the constant
    // one so a stride bug shows up as a wrong std rather than a near miss.
    std::vector<double> values{
        1.0, 5.0, 100.0,
        2.0, 5.0, 200.0,
        3.0, 5.0, 300.0,
        4.0, 5.0, 400.0,
    };
    Fitted fitted(3);
    sqt::fit_preprocess_stats(values.data(), 4, 3, 0.0, 1.0, fitted.view());
    expect_near(fitted.mean[0], 2.5, 1e-15, "column 0 mean");
    expect_near(fitted.mean[1], 5.0, 1e-15, "column 1 mean");
    expect_near(fitted.mean[2], 250.0, 1e-13, "column 2 mean");
    expect_near(fitted.stdev[1], 1.0, 0.0, "column 1 is constant -> std 1.0");
    expect_near(fitted.stdev[0], std::sqrt(5.0 / 3.0), 1e-15, "column 0 std");

    std::vector<double> out(12);
    sqt::apply_preprocess_stats(values.data(), 4, 3, fitted.view(), out.data());
    // Column 0 standardized: (1-2.5)/1.290994 = -1.161895
    expect_near(out[0], (1.0 - 2.5) / std::sqrt(5.0 / 3.0), 1e-14,
                "row 0, column 0 standardized");
    expect_near(out[1], 0.0, 1e-15, "row 0, column 1 (constant) -> 0");
    expect_near(out[2], (100.0 - 250.0) / fitted.stdev[2], 1e-13,
                "row 0, column 2 standardized");
}

static void test_apply_can_write_into_its_own_input() {
    std::printf("test_apply_can_write_into_its_own_input\n");
    // The header documents that values and out may alias; the engine does
    // not rely on it today, but a caller reading that promise should find
    // it true.
    std::vector<double> values{1.0, 2.0, 3.0, 4.0};
    std::vector<double> expected(4);
    Fitted fitted(1);
    sqt::fit_preprocess_stats(values.data(), 4, 1, 0.0, 1.0, fitted.view());
    sqt::apply_preprocess_stats(values.data(), 4, 1, fitted.view(),
                                expected.data());
    sqt::apply_preprocess_stats(values.data(), 4, 1, fitted.view(),
                                values.data());
    for (std::size_t i = 0; i < 4; ++i)
        expect_near(values[i], expected[i], 0.0, "in-place matches out-of-place");
}

static void test_empty_panel_is_a_no_op() {
    std::printf("test_empty_panel_is_a_no_op\n");
    Fitted fitted(1);
    fitted.lo[0] = fitted.hi[0] = fitted.mean[0] = fitted.stdev[0] = -1.0;
    const bool ok = sqt::fit_preprocess_stats(nullptr, 0, 0, 0.01, 0.99,
                                              fitted.view());
    expect(ok, "null/empty input reports success rather than failing");
    expect_near(fitted.lo[0], -1.0, 0.0, "untouched on an empty panel");
}

static void test_zero_rows_with_columns() {
    std::printf("test_zero_rows_with_columns\n");
    // No rows but real columns: every column is "all missing", so the
    // all-NaN rule applies rather than a division by zero.
    std::vector<double> values;  // empty, but n_cols = 2
    Fitted fitted(2);
    sqt::fit_preprocess_stats(values.data(), 0, 2, 0.01, 0.99, fitted.view());
    expect(std::isnan(fitted.mean[0]) && std::isnan(fitted.mean[1]),
           "zero rows -> NaN means");
    expect_near(fitted.stdev[0], 1.0, 0.0, "zero rows -> std 1.0");
}


// -- rank_by_date -------------------------------------------------------------

static void test_rank_is_one_based_and_averages_ties() {
    std::printf("test_rank_is_one_based_and_averages_ties\n");
    // One date, four rows, one column: 5,5,5,9.
    const double values[] = {5.0, 5.0, 5.0, 9.0};
    const long long codes[] = {0, 0, 0, 0};
    double out[4] = {0, 0, 0, 0};
    const bool ok = sqt::rank_by_date(values, 4, 1, codes, 1, out);
    expect(ok, "rank_by_date reports success");
    // Ordinals 1,2,3 average to 2.0; the odd one out takes 4.
    expect_near(out[0], 2.0, 0.0, "tied rows share the mean ordinal");
    expect_near(out[1], 2.0, 0.0, "tied rows share the mean ordinal");
    expect_near(out[2], 2.0, 0.0, "tied rows share the mean ordinal");
    expect_near(out[3], 4.0, 0.0, "the untied row keeps its ordinal");
}

static void test_rank_skips_nan_and_preserves_it() {
    std::printf("test_rank_skips_nan_and_preserves_it\n");
    const double values[] = {10.0, kNaN, 30.0, 20.0};
    const long long codes[] = {0, 0, 0, 0};
    double out[4] = {0, 0, 0, 0};
    expect(sqt::rank_by_date(values, 4, 1, codes, 1, out), "success");
    expect(std::isnan(out[1]), "a missing value stays missing");
    // The three present values rank 1..3 -- the absent one does not shift them.
    expect_near(out[0], 1.0, 0.0, "present values rank among themselves");
    expect_near(out[3], 2.0, 0.0, "present values rank among themselves");
    expect_near(out[2], 3.0, 0.0, "present values rank among themselves");
}

static void test_rank_is_per_date_and_per_column() {
    std::printf("test_rank_is_per_date_and_per_column\n");
    // Two dates x two rows, two columns; row-major (n_rows, n_cols).
    // date 0: col0 = 1,2   col1 = 9,8
    // date 1: col0 = 5,4   col1 = 0,7
    const double values[] = {1.0, 9.0, 2.0, 8.0, 5.0, 0.0, 4.0, 7.0};
    const long long codes[] = {0, 0, 1, 1};
    double out[8] = {0};
    expect(sqt::rank_by_date(values, 4, 2, codes, 2, out), "success");
    expect_near(out[0], 1.0, 0.0, "date 0 col 0");
    expect_near(out[2], 2.0, 0.0, "date 0 col 0");
    expect_near(out[1], 2.0, 0.0, "date 0 col 1 ranks independently");
    expect_near(out[3], 1.0, 0.0, "date 0 col 1 ranks independently");
    expect_near(out[4], 2.0, 0.0, "date 1 col 0");
    expect_near(out[6], 1.0, 0.0, "date 1 col 0");
    expect_near(out[5], 1.0, 0.0, "date 1 col 1");
    expect_near(out[7], 2.0, 0.0, "date 1 col 1");
}

static void test_rank_does_not_need_sorted_dates() {
    std::printf("test_rank_does_not_need_sorted_dates\n");
    const double values[] = {1.0, 9.0, 2.0, 8.0};
    const long long codes[] = {1, 0, 1, 0};
    double out[4] = {0, 0, 0, 0};
    expect(sqt::rank_by_date(values, 4, 1, codes, 2, out), "success");
    expect_near(out[0], 1.0, 0.0, "interleaved dates rank within themselves");
    expect_near(out[2], 2.0, 0.0, "interleaved dates rank within themselves");
    expect_near(out[1], 2.0, 0.0, "interleaved dates rank within themselves");
    expect_near(out[3], 1.0, 0.0, "interleaved dates rank within themselves");
}

static void test_rank_all_nan_date_is_all_nan() {
    std::printf("test_rank_all_nan_date_is_all_nan\n");
    const double values[] = {kNaN, kNaN};
    const long long codes[] = {0, 0};
    double out[2] = {0, 0};
    expect(sqt::rank_by_date(values, 2, 1, codes, 1, out), "success");
    expect(std::isnan(out[0]) && std::isnan(out[1]),
           "a date with nothing present ranks nothing");
}

static void test_rank_empty_panel_is_a_no_op() {
    std::printf("test_rank_empty_panel_is_a_no_op\n");
    double out[1] = {7.0};
    expect(sqt::rank_by_date(nullptr, 0, 1, nullptr, 0, out), "empty is success");
    expect_near(out[0], 7.0, 0.0, "nothing was written");
}

// -- rows whose date code is outside [0, n_dates) ------------------------------

static void test_out_of_range_codes_come_back_nan() {
    std::printf("test_out_of_range_codes_come_back_nan\n");
    // The output is pre-filled with a sentinel standing in for whatever the
    // allocator left behind. A row the kernel cannot place used to keep it.
    constexpr double kSentinel = 12345.0;
    const double values[] = {1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0};
    const long long below[] = {0, 0, -1, 0};   // pd.factorize's code for NaT
    const long long above[] = {0, 0, 1, 0};    // == n_dates
    for (const long long* codes : {below, above}) {
        double ranked[8], standardized[8];
        for (double& v : ranked) v = kSentinel;
        for (double& v : standardized) v = kSentinel;
        expect(sqt::rank_by_date(values, 4, 2, codes, 1, ranked), "rank success");
        expect(sqt::standardize_by_date(values, 4, 2, codes, 1, 0.0, standardized),
               "standardize success");
        expect(std::isnan(ranked[4]) && std::isnan(ranked[5]),
               "the unplaceable row is NaN in every rank column");
        expect(std::isnan(standardized[4]) && std::isnan(standardized[5]),
               "the unplaceable row is NaN in every standardized column");
        bool sentinel_left = false;
        for (int i = 0; i < 8; ++i)
            sentinel_left = sentinel_left || ranked[i] == kSentinel ||
                            standardized[i] == kSentinel;
        expect(!sentinel_left, "every output element was written");
        // The placeable rows rank among themselves: 1, 2, 4 -> 1, 2, 3.
        expect_near(ranked[0], 1.0, 0.0, "placeable rows still rank");
        expect_near(ranked[6], 3.0, 0.0, "placeable rows still rank");
    }
}

static void test_in_range_codes_leave_no_nan() {
    std::printf("test_in_range_codes_leave_no_nan\n");
    // The null case: every code valid, nothing extra is written as NaN.
    const double values[] = {1.0, 2.0, 3.0, 4.0};
    const long long codes[] = {0, 1, 0, 1};
    double ranked[4] = {0, 0, 0, 0}, standardized[4] = {0, 0, 0, 0};
    expect(sqt::rank_by_date(values, 4, 1, codes, 2, ranked), "rank success");
    expect(sqt::standardize_by_date(values, 4, 1, codes, 2, 0.0, standardized),
           "standardize success");
    for (int i = 0; i < 4; ++i) {
        expect(!std::isnan(ranked[i]), "valid rows rank");
        expect(!std::isnan(standardized[i]), "valid rows standardize");
    }
}


// -- permutation_null_ic ------------------------------------------------------

static void test_permutation_is_reproducible_from_its_seed() {
    std::printf("test_permutation_is_reproducible_from_its_seed\n");
    const double target[] = {1.0, 2.0, 3.0, 4.0, 1.0, 4.0, 2.0, 3.0};
    const double values[] = {4.0, 3.0, 2.0, 1.0, 2.0, 1.0, 4.0, 3.0};
    const long long codes[] = {0, 0, 0, 0, 1, 1, 1, 1};
    double a[16] = {0}, b[16] = {0};
    expect(sqt::permutation_null_ic(target, values, codes, 8, 2, 16, 99ULL, true, a),
           "first run succeeds");
    expect(sqt::permutation_null_ic(target, values, codes, 8, 2, 16, 99ULL, true, b),
           "second run succeeds");
    bool identical = true;
    for (int i = 0; i < 16; ++i) if (a[i] != b[i]) identical = false;
    expect(identical, "the same seed reproduces the same draws");
}

static void test_permutation_seeds_differ() {
    std::printf("test_permutation_seeds_differ\n");
    const double target[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    const double values[] = {6.0, 5.0, 4.0, 3.0, 2.0, 1.0};
    const long long codes[] = {0, 0, 0, 0, 0, 0};
    double a[32] = {0}, b[32] = {0};
    sqt::permutation_null_ic(target, values, codes, 6, 1, 32, 1ULL, true, a);
    sqt::permutation_null_ic(target, values, codes, 6, 1, 32, 2ULL, true, b);
    bool any_difference = false;
    for (int i = 0; i < 32; ++i) if (a[i] != b[i]) any_difference = true;
    expect(any_difference, "a different seed draws differently");
}

static void test_permutation_stays_in_range() {
    std::printf("test_permutation_stays_in_range\n");
    const double target[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    const double values[] = {2.0, 8.0, 1.0, 5.0, 3.0, 7.0, 4.0, 6.0};
    const long long codes[] = {0, 0, 0, 0, 1, 1, 1, 1};
    double out[64] = {0};
    expect(sqt::permutation_null_ic(target, values, codes, 8, 2, 64, 7ULL, true, out),
           "success");
    bool in_range = true;
    for (int i = 0; i < 64; ++i) {
        if (!(out[i] >= -1.0000001 && out[i] <= 1.0000001)) in_range = false;
    }
    expect(in_range, "a mean of correlations stays inside [-1, 1]");
}

static void test_permutation_skips_single_row_dates() {
    std::printf("test_permutation_skips_single_row_dates\n");
    // Date 1 has one row and contributes nothing; date 0 has three.
    const double target[] = {1.0, 2.0, 3.0, 9.0};
    const double values[] = {3.0, 1.0, 2.0, 9.0};
    const long long codes[] = {0, 0, 0, 1};
    double out[8] = {0};
    expect(sqt::permutation_null_ic(target, values, codes, 4, 2, 8, 3ULL, true, out),
           "success");
    bool finite = true;
    for (int i = 0; i < 8; ++i) if (!std::isfinite(out[i])) finite = false;
    expect(finite, "a one-row date does not poison the mean");
}

static void test_permutation_zero_draws_is_a_no_op() {
    std::printf("test_permutation_zero_draws_is_a_no_op\n");
    double out[1] = {5.0};
    expect(sqt::permutation_null_ic(nullptr, nullptr, nullptr, 0, 0, 0, 1ULL,
                                    true, out),
           "zero draws is success");
    expect_near(out[0], 5.0, 0.0, "nothing was written");
}

// -- label_uniqueness ---------------------------------------------------------

// Rows of one entity on the same date take their places on its date axis in
// row order, the order of the Python fallback's stable argsort. A tied row's
// place decides which bars its label spans, so the weights follow it. The
// reference below is the fallback's arithmetic on a stable sort; 120 rows per
// entity are past the length at which a library sort stops insertion-sorting,
// where an unstable sort first reorders ties.
static void test_label_uniqueness_ties_keep_row_order() {
    std::printf("test_label_uniqueness_ties_keep_row_order\n");
    const std::size_t per_entity = 120, n_entities = 2;
    const std::size_t n = per_entity * n_entities;
    const long long nat = std::numeric_limits<long long>::min();
    std::vector<long long> dates(n), ends(n), entity(n);
    for (std::size_t k = 0; k < n; ++k) {
        // A scrambled row order (37 and 240 are coprime), entities
        // interleaved, three rows on each date, label ends 1-5 dates ahead.
        const std::size_t i = (k * 37) % n;
        const std::size_t local = i / n_entities;
        const long long day = static_cast<long long>(local / 3);
        dates[k] = day * 1000;
        ends[k] = (day + 1 + static_cast<long long>((local * 7) % 5)) * 1000;
        if (day >= 38) ends[k] = nat;
        entity[k] = static_cast<long long>(i % n_entities);
    }
    std::vector<double> got(n);
    expect(sqt::label_uniqueness(dates.data(), ends.data(), entity.data(), n,
                                 n_entities, got.data()),
           "label_uniqueness success");

    std::vector<double> want(n, 1.0);
    for (std::size_t e = 0; e < n_entities; ++e) {
        std::vector<std::size_t> rows;
        for (std::size_t k = 0; k < n; ++k)
            if (entity[k] == static_cast<long long>(e)) rows.push_back(k);
        std::stable_sort(rows.begin(), rows.end(),
                         [&](std::size_t a, std::size_t b) { return dates[a] < dates[b]; });
        const std::size_t m = rows.size();
        std::vector<long long> axis(m);
        for (std::size_t i = 0; i < m; ++i) axis[i] = dates[rows[i]];
        std::vector<std::size_t> end_pos(m);
        std::vector<double> delta(m + 1, 0.0);
        for (std::size_t i = 0; i < m; ++i) {
            std::size_t p = i;
            if (ends[rows[i]] != nat) {
                const auto up = std::upper_bound(axis.begin(), axis.end(), ends[rows[i]]);
                const auto dist = up - axis.begin();
                if (dist > 0 && static_cast<std::size_t>(dist - 1) > p)
                    p = static_cast<std::size_t>(dist - 1);
            }
            end_pos[i] = p;
            delta[i] += 1.0;
            delta[p + 1] -= 1.0;
        }
        std::vector<double> cumulative(m + 1, 0.0);
        double running = 0.0;
        for (std::size_t i = 0; i < m; ++i) {
            running += delta[i];
            cumulative[i + 1] = cumulative[i] + 1.0 / std::max(running, 1.0);
        }
        for (std::size_t i = 0; i < m; ++i)
            want[rows[i]] = (cumulative[end_pos[i] + 1] - cumulative[i]) /
                            static_cast<double>(end_pos[i] - i + 1);
    }
    double total = 0.0;
    for (double w : want) total += w;
    const double mean = total / static_cast<double>(n);
    bool tied_rows_differ = false;
    for (std::size_t k = 0; k < n; ++k) {
        want[k] /= mean;
        expect_near(got[k], want[k], 1e-12, "a tied row weighs as in row order");
        for (std::size_t j = k + 1; j < n; ++j)
            if (entity[j] == entity[k] && dates[j] == dates[k] && want[j] != want[k])
                tied_rows_differ = true;
    }
    expect(tied_rows_differ, "tied rows weigh differently, so their order shows");
}

int main() {
    std::printf("=== sqt panel_stats tests ===\n");
    test_quantile_is_linearly_interpolated();
    test_quantile_endpoints();
    test_quantile_endpoints_on_a_long_column();
    test_std_uses_ddof_one();
    test_moments_are_of_the_clipped_column();
    test_constant_column_gets_unit_std();
    test_single_row_column();
    test_all_nan_column();
    test_nan_is_skipped_by_the_moments();
    test_nan_survives_the_transform();
    test_is_nan_is_std_isnan();
    test_every_nan_passes_through_apply_bit_for_bit();
    test_infinity_is_not_treated_as_missing();
    test_columns_are_independent();
    test_apply_can_write_into_its_own_input();
    test_empty_panel_is_a_no_op();
    test_zero_rows_with_columns();
    test_rank_is_one_based_and_averages_ties();
    test_rank_skips_nan_and_preserves_it();
    test_rank_is_per_date_and_per_column();
    test_rank_does_not_need_sorted_dates();
    test_rank_all_nan_date_is_all_nan();
    test_rank_empty_panel_is_a_no_op();
    test_out_of_range_codes_come_back_nan();
    test_in_range_codes_leave_no_nan();
    test_permutation_is_reproducible_from_its_seed();
    test_permutation_seeds_differ();
    test_permutation_stays_in_range();
    test_permutation_skips_single_row_dates();
    test_permutation_zero_draws_is_a_no_op();
    test_label_uniqueness_ties_keep_row_order();

    std::printf("\n%d assertion(s), %d failed\n", g_tests_run, g_tests_failed);
    return g_tests_failed == 0 ? 0 : 1;
}
