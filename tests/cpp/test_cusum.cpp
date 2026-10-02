/**
 * C++ unit tests for sqt::cusum_peaks_into, the two-sided CUSUM scan over a
 * panel of paths.
 *
 * The Python suite (tests/cpp_bindings/test_cpp_cusum.py and
 * tests/analysis/test_liquidity_events_native.py) holds the kernel to the
 * numpy loop in analysis/liquidity_events.py bit for bit. What is checked
 * here is what the kernel owes on its own: hand-computed peaks, the
 * reference window, slack at zero, NaN and infinity, and that a panel --
 * however its rows fall into blocks -- answers exactly what each row
 * answers alone.
 *
 * Run via CTest:
 *   ctest --test-dir build --config Release -V -R cpp_cusum
 */

#include "sqt/cusum.hpp"

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

static int g_tests_run = 0;
static int g_tests_failed = 0;

#define CHECK(cond)                                                                   \
    do {                                                                              \
        ++g_tests_run;                                                                \
        if (!(cond)) {                                                                \
            ++g_tests_failed;                                                         \
            std::fprintf(stderr, "FAIL  %s  line %d: %s\n", __func__, __LINE__, #cond); \
        }                                                                             \
    } while (false)

namespace {

const double kNaN = std::numeric_limits<double>::quiet_NaN();
const double kInf = std::numeric_limits<double>::infinity();

// Bit-for-bit equality.
bool same_bits(double a, double b) { return std::memcmp(&a, &b, sizeof(double)) == 0; }

// The numpy loop, written out one row at a time. numpy's maximum: a NaN in
// either argument propagates, the first's if it is NaN; otherwise the
// larger. (No -0.0 reaches a maximum in this recursion, so the tie rule
// cannot show; see cusum.cpp.)
double np_max(double a, double b) { return (a >= b || std::isnan(a)) ? a : b; }

double reference_peak(const double* row, std::size_t n_cols, std::size_t n_reference,
                      double slack) {
    double up = 0.0, down = 0.0, peak = 0.0;
    for (std::size_t t = 1; t < n_cols; ++t) {
        up = np_max(0.0, (up + row[t]) - slack);
        down = np_max(0.0, (down - row[t]) - slack);
        if (t >= n_reference) peak = np_max(peak, np_max(up, down));
    }
    return peak;
}

std::vector<double> peaks_of(const std::vector<double>& z, std::size_t n_rows,
                             std::size_t n_cols, std::size_t n_reference, double slack) {
    std::vector<double> out(n_rows, -1.0);
    sqt::cusum_peaks_into(z.data(), n_rows, n_cols, n_reference, slack, out.data());
    return out;
}

double one_row(const std::vector<double>& row, std::size_t n_reference, double slack) {
    return peaks_of(row, 1, row.size(), n_reference, slack)[0];
}

}  // namespace

static void test_an_empty_panel_is_a_no_op() {
    sqt::cusum_peaks_into(nullptr, 0, 0, 0, 0.5, nullptr);
    sqt::cusum_peaks_into(nullptr, 0, 100, 5, 0.5, nullptr);
    CHECK(true);
}

static void test_a_known_crossing_by_hand() {
    // up: 1 - 0.5 = 0.5, then 0.5 + 2 - 0.5 = 2.0, then 2 + 3 - 0.5 = 4.5.
    // down never leaves zero. Column 0 is never read.
    const std::vector<double> row = {99.0, 1.0, 2.0, 3.0};
    CHECK(one_row(row, 0, 0.5) == 4.5);
    CHECK(one_row(row, 1, 0.5) == 4.5);
}

static void test_a_null_path_never_accumulates() {
    // Every step inside the slack: both sides stay at exactly zero.
    const std::vector<double> row = {0.0, 0.4, -0.4, 0.2, -0.5, 0.5, 0.0};
    CHECK(same_bits(one_row(row, 0, 0.5), 0.0));
}

static void test_slack_zero_is_the_plain_reflected_walk() {
    // up: 1, 0, 1, 0; down: 0, 1, 0, 1. Peak 1.
    const std::vector<double> row = {0.0, 1.0, -1.0, 1.0, -1.0};
    CHECK(one_row(row, 0, 0.0) == 1.0);
}

static void test_all_negative_steps_drive_only_the_down_side() {
    // down gains 1 - 0.5 = 0.5 per step from t = 1: 0.5 * (n - 1).
    const std::size_t n = 11;
    const std::vector<double> row(n, -1.0);
    CHECK(one_row(row, 0, 0.5) == 0.5 * static_cast<double>(n - 1));
}

static void test_the_reference_window_advances_but_is_not_scanned() {
    // A spike at t = 1 inside the window, then quiet: up decays by the slack.
    // Scanned from t = 3, the peak is what is left of the spike there.
    const std::vector<double> row = {0.0, 10.0, 0.0, 0.0, 0.0};
    // up: 9.5, 9.0, 8.5, 8.0
    CHECK(one_row(row, 0, 0.5) == 9.5);
    CHECK(one_row(row, 3, 0.5) == 8.5);
    CHECK(one_row(row, 4, 0.5) == 8.0);
    // A window as long as the row, or longer, scans nothing.
    CHECK(same_bits(one_row(row, 5, 0.5), 0.0));
    CHECK(same_bits(one_row(row, 500, 0.5), 0.0));
}

static void test_one_column_scans_nothing() {
    const std::vector<double> row = {1e9};
    CHECK(same_bits(one_row(row, 0, 0.5), 0.0));
}

static void test_negative_zero_never_reaches_the_peak() {
    // The kernel's maximum may break a tie either way (see cusum.cpp); that
    // is only safe because -0.0 cannot get into the recursion. Planted
    // everywhere it could come from, the peak is still +0.0 to the bit.
    const std::vector<double> row = {-0.0, -0.0, 0.0, -0.0, -0.0};
    CHECK(same_bits(one_row(row, 0, 0.0), 0.0));
    CHECK(same_bits(one_row(row, 0, -0.0), 0.0));
    CHECK(same_bits(one_row(row, 2, -0.0), 0.0));
}

static void test_nan_propagates_into_the_peak() {
    // A NaN anywhere from t = 1 makes up and down NaN for good, so the peak
    // is NaN -- also when the NaN sat inside the unscanned window.
    std::vector<double> row = {0.0, 1.0, kNaN, 1.0, 1.0};
    CHECK(std::isnan(one_row(row, 0, 0.5)));
    CHECK(std::isnan(one_row(row, 4, 0.5)));
    // In column 0 it is never read.
    row = {kNaN, 1.0, 2.0, 3.0};
    CHECK(one_row(row, 0, 0.5) == 4.5);
}

static void test_infinity_follows_ieee() {
    // +inf: up is inf for good, down is max(0, -inf) = 0, so the peak is inf.
    std::vector<double> row = {0.0, 1.0, kInf, -5.0};
    CHECK(one_row(row, 0, 0.5) == kInf);
    // An infinite slack: inf - inf would be NaN only for an infinite step;
    // a finite one keeps both sides at zero.
    row = {0.0, 1.0, 2.0, 3.0};
    CHECK(same_bits(one_row(row, 0, kInf), 0.0));
    // A NaN slack poisons everything.
    CHECK(std::isnan(one_row(row, 0, kNaN)));
}

static void test_matches_the_reference_loop_bit_for_bit() {
    std::mt19937_64 gen(20261001);
    std::normal_distribution<double> normal(0.0, 1.0);
    for (std::size_t n_rows : {1u, 3u, 4u, 5u, 17u, 200u}) {
        for (std::size_t n_cols : {1u, 2u, 10u, 333u, 2105u}) {
            std::vector<double> z(n_rows * n_cols);
            for (double& v : z) v = normal(gen) * 1.7 + 0.1;
            for (std::size_t n_reference : {0u, 1u, 5u, 631u}) {
                for (double slack : {0.0, 0.5, 1.25, -0.3}) {
                    const auto got = peaks_of(z, n_rows, n_cols, n_reference, slack);
                    bool all_same = true;
                    for (std::size_t r = 0; r < n_rows; ++r) {
                        const double want =
                            reference_peak(z.data() + r * n_cols, n_cols, n_reference, slack);
                        all_same = all_same && same_bits(got[r], want);
                    }
                    CHECK(all_same);
                }
            }
        }
    }
}

static void test_a_panel_answers_what_each_row_answers_alone() {
    // Rows are advanced in blocks, which may not change a bit. 203 rows
    // leave a remainder after whole blocks.
    const std::size_t n_rows = 203, n_cols = 2105;
    std::mt19937_64 gen(7);
    std::normal_distribution<double> normal(0.0, 1.0);
    std::vector<double> z(n_rows * n_cols);
    for (double& v : z) v = normal(gen);
    // Planted rows: a NaN, an infinity, a level shift.
    z[3 * n_cols + 700] = kNaN;
    z[8 * n_cols + 1500] = kInf;
    for (std::size_t t = 1200; t < n_cols; ++t) z[202 * n_cols + t] += 2.0;
    const auto panel = peaks_of(z, n_rows, n_cols, 631, 0.5);
    bool all_same = true;
    for (std::size_t r = 0; r < n_rows; ++r) {
        const std::vector<double> row(z.begin() + static_cast<std::ptrdiff_t>(r * n_cols),
                                      z.begin() + static_cast<std::ptrdiff_t>((r + 1) * n_cols));
        all_same = all_same && same_bits(panel[r], one_row(row, 631, 0.5));
    }
    CHECK(all_same);
    CHECK(std::isnan(panel[3]));
    CHECK(panel[8] == kInf);
    CHECK(panel[202] > 100.0);  // the shifted row crosses any sane threshold
}

int main() {
    test_an_empty_panel_is_a_no_op();
    test_a_known_crossing_by_hand();
    test_a_null_path_never_accumulates();
    test_slack_zero_is_the_plain_reflected_walk();
    test_all_negative_steps_drive_only_the_down_side();
    test_the_reference_window_advances_but_is_not_scanned();
    test_one_column_scans_nothing();
    test_negative_zero_never_reaches_the_peak();
    test_nan_propagates_into_the_peak();
    test_infinity_follows_ieee();
    test_matches_the_reference_loop_bit_for_bit();
    test_a_panel_answers_what_each_row_answers_alone();

    std::printf("%d/%d checks passed\n", g_tests_run - g_tests_failed, g_tests_run);
    return g_tests_failed == 0 ? 0 : 1;
}
