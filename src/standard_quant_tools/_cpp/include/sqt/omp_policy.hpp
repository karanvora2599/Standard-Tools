#pragma once

#include <algorithm>
#include <atomic>
#include <climits>
#include <cstddef>
#include <cstdlib>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace sqt {
namespace omp_policy {

/**
 * Whether a parallel region is worth entering, and how many threads it may
 * use.
 *
 * The kernels here previously parallelized whenever there was more than one
 * independent task (`if(num_tests > 1)`), which is the wrong question twice
 * over:
 *
 *   - TOO EAGER. Two tiny backtests cost more in thread startup and
 *     scheduling than they save. The decision depends on TOTAL WORK, not on
 *     the task count alone.
 *
 *   - TOO GREEDY. Standard Tools frequently runs inside something that is
 *     already parallel: a ProcessPoolExecutor screener, several agents, a
 *     web worker pool, replicated containers, the library's own search and
 *     fold thread pools. Each call grabbing every core massively
 *     oversubscribes the machine.
 *
 * WORK IS TIME, NOT A COUNT. A region costs the same to start whatever the
 * kernel, so the question is how long the work would take serially. Each
 * call site states its work as units (tasks x units per task, the same units
 * it always counted) AND what one unit costs serially, in nanoseconds,
 * measured for that kernel (`cost` below). A region goes parallel when
 * tasks x (ns_per_task + units_per_task x ns_per_unit) reaches
 * kMinSerialNs, 150 us.
 *
 * A single unit threshold could not do this: one unit is 1 ns of work in the
 * implied-volatility solve and 140 ns in an Engle-Granger test, so the old
 * 50,000-unit default was right for the option kernels and held back the
 * others until they had milliseconds of serial work. From 150 us the
 * parallel path measured 1.4-7x faster than the serial one on back-to-back
 * calls, and about even (0.75-1.2x) on a call made after the runtime's
 * workers had gone idle, whether they spin or sleep between regions
 * (OMP_WAIT_POLICY); at 250 us the option and preprocessing kernels gave up
 * 1.3-2.8x on calls just under it (CHANGELOG, 2026-10-02).
 *
 * SHARING THE CORES -- for the pooled sort only. Every call that goes
 * parallel is counted while it runs (`parallel_call`, an RAII guard declared
 * just before the region). The parallel sort behind a pooled rank
 * correlation (panel_stats.cpp's average_ranks) asks `shared_threads()`,
 * an equal share of the threads among the calls running, so several Python
 * threads ranking at once -- every binding releases the GIL -- split the
 * cores instead of each splitting its sort across all of them; that sort's
 * run count follows its thread count, and the share measured 1.2-1.6x
 * faster from 8 threads. Every other region takes `max_threads()`, the
 * configured count, as before: shared the same way, the per-date and
 * per-pair loops measured up to 17% slower, because a share fixed when a
 * region starts leaves cores idle as the calls around it finish. How many
 * threads ran a region is never observable in its output (see
 * tests/cpp_bindings/test_thread_count_determinism.py), so this changes
 * speed, never a result.
 *
 * Two environment variables, read once and cached:
 *
 *   SQT_NUM_THREADS   maximum threads any kernel may use (0/unset = OpenMP's
 *                     own default), and the total that concurrent pooled
 *                     sorts share. Set this to 1 inside a process pool.
 *   SQT_OMP_MIN_WORK  when set, the old rule exactly: a region goes
 *                     parallel at tasks x units_per_task >= this many units,
 *                     and the measured costs are ignored. Unset (or
 *                     unparsable, or negative): the time rule above.
 *
 * SCHEDULING. Every parallel loop in this codebase uses schedule(guided), not
 * schedule(static), unless its own comment earns the exception. static
 * assigns each thread an equal COUNT of iterations up front and never
 * rebalances, which is optimal only when the work per iteration and the
 * speed of each thread are both uniform. Neither is something a library can
 * assume:
 *
 *   - Work per iteration varies for real. rolling_hurst's per-window cost
 *     depends on how many boxes log_sizes yields for that window.
 *   - Threads are not equal. SMT siblings share a physical core; hybrid
 *     P/E designs (Intel 12th gen+, Apple silicon, ARM big.LITTLE) differ by
 *     ~2x; a cgroup CPU quota, a co-tenant process, or thermal throttling
 *     all skew throughput mid-run.
 *
 * Measured on one such machine before the change, scaling was NON-MONOTONIC:
 * batch_run_strategy took 44.5 ms on 6 threads and 60.4 ms on 8. That is the
 * signature of an equal split landing on unequal cores.
 *
 * The fix is deliberately NOT a tuned thread count. "Cap threads at this
 * box's fast-core count" is a statement about one machine and wrong to ship;
 * guided adapts to whatever the machine turns out to be. The verification bar
 * is likewise machine-independent: adding a thread must never make a kernel
 * slower.
 */
inline long long env_ll(const char* name, long long fallback) {
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)  // getenv is fine here: read-only, cached once
#endif
    const char* raw = std::getenv(name);
#ifdef _MSC_VER
#pragma warning(pop)
#endif
    if (raw == nullptr || *raw == '\0') return fallback;
    try {
        const long long v = std::stoll(raw);
        return (v < 0) ? fallback : v;
    } catch (...) {
        return fallback;  // unparsable: behave as though it were unset
    }
}

/** Estimated serial time, in nanoseconds, from which a region goes parallel. */
constexpr double kMinSerialNs = 150'000.0;

/** The unit threshold the policy used before costs: 2-argument callers only. */
constexpr long long kLegacyMinWork = 50'000;

/**
 * What one call's work costs serially: `ns_per_unit` for each unit the call
 * site counts, plus `ns_per_task` for each task whatever its size (a fixed
 * per-task set-up, such as reseeding a generator).
 */
struct serial_cost {
    double ns_per_unit;
    double ns_per_task;
};

/**
 * The measured serial cost of each parallel call site's unit.
 *
 * Measured on the raw bindings with SQT_NUM_THREADS=1, at sizes around the
 * decision (30 us to 3 ms of serial work), as the slope of time against
 * units -- so a call's fixed, unsplittable overhead (validation, output
 * allocation, bucketing by date) is not counted as work a region would
 * split. Intel i7-13620H, MSVC /arch:AVX2, 2026-10-02. Where the cost per
 * unit grows with the shape (longer series, bigger dates), the figure is the
 * low end, so a mis-estimate delays the parallel path rather than taking it
 * too early. These are ratios between kernels more than absolute times: a
 * faster machine runs every kernel faster, and the threshold's margin
 * absorbs the rest.
 */
namespace cost {
// backtest.cpp. Unit: one bar of one test / one bar of one indicator pair.
constexpr serial_cost batch_run_strategy{3.8, 0.0};
constexpr serial_cost batch_backtest_crossover{5.0, 0.0};
// cointegration.cpp. Unit: one bar of one pair. 140 ns at 252 bars and
// 250 ns at 1,260: the automatic lag search grows with the series.
constexpr serial_cost engle_granger{140.0, 0.0};
// hurst.cpp. Unit: one bar of one window. 25-34 ns (DFA) and 34-48 ns
// (R/S) for windows of 252 and 100 bars.
constexpr serial_cost hurst_dfa{25.0, 0.0};
constexpr serial_cost hurst_rs{34.0, 0.0};
// indicators.cpp. Unit: one bar of one ticker for one indicator; the panel
// averages the costs of the indicators a call asks for.
constexpr double rsi_bar = 6.5;
constexpr double adx_bar = 10.0;
constexpr double atr_bar = 6.5;
constexpr double bollinger_bar = 20.0;
constexpr double stochastic_bar = 45.0;
// monte_carlo.cpp. Unit: one day of one path, plus reseeding a 64-bit
// Mersenne Twister for every path.
constexpr serial_cost bootstrap_paths{1.9, 730.0};
constexpr serial_cost bootstrap_terminal{1.0, 740.0};
// options.cpp. Units: kImpliedVolWork per contract, kGreeksWork per
// valuation (about 475 ns per solve and 60 ns per full set of greeks).
// A selection of one or two greeks costs 13-30 ns a valuation, but the
// region still paid at the full set's threshold: a gamma-only grid of 4,000
// valuations took 32 us across the threads against 53 us on one, so a
// selection keeps the full set's cost rather than one of its own.
constexpr serial_cost implied_volatility{0.95, 0.0};
constexpr serial_cost greeks{1.2, 0.0};
// panel_stats.cpp. Units: one value of the panel (rows x columns), one row
// of a date's cross-section, one row of one permutation, one row of one
// entity.
constexpr serial_cost fit_preprocess{13.0, 0.0};
constexpr serial_cost apply_preprocess{2.2, 0.0};
// Pearson's per-date loop barely scales: measured 8 ns a row, but from the
// 18,750 rows that would put at the threshold, the parallel path was
// 0.74-0.81x on a call made after the workers went idle, and still only
// 1.07-1.19x at 37,500 rows. The cost is set so it switches where the old
// unit rule did, at 50,000 rows.
constexpr serial_cost cross_section_pearson{3.0, 0.0};
// correlation.cpp. Units: one value of a complete column's recursion; one
// row of one complete pair, summed in a vectorized row (0.16-0.23 ns); one
// row of one pair with a gap, pandas' loop as written (3.0-3.4 ns). The
// recursion measured 4.3-4.6 ns a value with the serial scan for gaps
// counted in.
constexpr serial_cost correlation_columns{3.0, 0.0};
constexpr serial_cost correlation_pairs{0.2, 0.0};
constexpr serial_cost correlation_gapped_pairs{3.0, 0.0};
constexpr serial_cost cross_section_spearman{40.0, 0.0};
constexpr serial_cost standardize_by_date{4.5, 0.0};
constexpr serial_cost rank_by_date{15.0, 0.0};
constexpr serial_cost permutation_null{6.0, 0.0};
constexpr serial_cost label_uniqueness{28.0, 0.0};
}  // namespace cost

/** SQT_OMP_MIN_WORK when it is set to a count of units; -1 otherwise. */
inline long long min_work_override() {
    static const long long value = env_ll("SQT_OMP_MIN_WORK", -1);
    return value;
}

/** SQT_NUM_THREADS; 0 when unset (OpenMP's own default). */
inline int thread_cap() {
    static const int value = static_cast<int>(
        std::min<long long>(env_ll("SQT_NUM_THREADS", 0), INT_MAX));
    return value;
}

namespace detail {

/**
 * The decision itself, with the environment passed in. `min_work_units` is
 * SQT_OMP_MIN_WORK, or -1 for the time rule.
 */
inline bool decide(std::size_t tasks, std::size_t units_per_task,
                   serial_cost c, int cap, long long min_work_units) noexcept {
    if (tasks <= 1) return false;
    if (cap == 1) return false;
    if (min_work_units >= 0) {
        const long long total = static_cast<long long>(tasks) *
                                static_cast<long long>(units_per_task);
        return total >= min_work_units;
    }
    const double serial_ns =
        static_cast<double>(tasks) *
        (c.ns_per_task + static_cast<double>(units_per_task) * c.ns_per_unit);
    return serial_ns >= kMinSerialNs;
}

/**
 * Threads for one region. Alone (`active` <= 1) it is the configured cap,
 * 0 meaning OpenMP's own default, exactly as before calls were counted.
 * Shared, it is an equal part of the total: the cap, or `openmp_default`
 * when there is none.
 */
inline int share(int cap, int openmp_default, int active) noexcept {
    if (active <= 1) return cap;
    const int total = cap > 0 ? cap : openmp_default;
    return std::max(1, total / active);
}

}  // namespace detail

/**
 * True when the work justifies entering a parallel region: `tasks`
 * independent tasks of `units_per_task` units each, at cost `c`.
 */
inline bool worth_parallel(std::size_t tasks, std::size_t units_per_task,
                           serial_cost c) {
    return detail::decide(tasks, units_per_task, c, thread_cap(),
                          min_work_override());
}

/**
 * The rule before costs were stated, for a call site that has not stated
 * one: tasks x units_per_task against SQT_OMP_MIN_WORK, default 50,000.
 * Every call site in this library states its cost.
 */
inline bool worth_parallel(std::size_t tasks, std::size_t units_per_task) {
    const long long override_units = min_work_override();
    return detail::decide(tasks, units_per_task, serial_cost{0.0, 0.0},
                          thread_cap(),
                          override_units >= 0 ? override_units : kLegacyMinWork);
}

/** How many calls are inside a parallel region right now, process-wide. */
inline std::atomic<int>& active_calls() noexcept {
    static std::atomic<int> count{0};
    return count;
}

/**
 * Threads a region may use: SQT_NUM_THREADS, or 0 for OpenMP's own default.
 * Not divided among concurrent calls; see the header comment.
 */
inline int max_threads() { return thread_cap(); }

/**
 * Threads for the pooled sort: an equal share of the total among the calls
 * now in a parallel region, or exactly max_threads() when this call is
 * alone. Read after the sort's own `parallel_call` is counted, so the share
 * includes it.
 */
inline int shared_threads() {
#ifdef _OPENMP
    const int openmp_default = omp_get_max_threads();
#else
    const int openmp_default = 1;
#endif
    return detail::share(thread_cap(), openmp_default,
                         active_calls().load(std::memory_order_relaxed));
}

/**
 * One kernel call's parallel decision, counted in `active_calls` for as
 * long as it is in scope when the call goes parallel -- the count the
 * pooled sort divides by. Declare it just before the region and use
 * `parallel()` as the region's `if` clause; the count drops when the call
 * returns, by exception too.
 */
class parallel_call {
public:
    explicit parallel_call(bool parallel) noexcept : parallel_(parallel) {
        if (parallel_) active_calls().fetch_add(1, std::memory_order_relaxed);
    }
    parallel_call(std::size_t tasks, std::size_t units_per_task, serial_cost c)
        : parallel_call(worth_parallel(tasks, units_per_task, c)) {}
    ~parallel_call() {
        if (parallel_) active_calls().fetch_sub(1, std::memory_order_relaxed);
    }
    parallel_call(const parallel_call&) = delete;
    parallel_call& operator=(const parallel_call&) = delete;

    bool parallel() const noexcept { return parallel_; }

private:
    const bool parallel_;
};

}  // namespace omp_policy
}  // namespace sqt
