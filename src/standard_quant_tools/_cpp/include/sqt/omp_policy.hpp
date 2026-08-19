#pragma once

#include <cstddef>
#include <cstdlib>
#include <string>

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
 *     scheduling than they save. The decision depends on TOTAL WORK
 *     (tasks x bars), not on the task count alone.
 *
 *   - TOO GREEDY. Standard Tools frequently runs inside something that is
 *     already parallel: a ProcessPoolExecutor screener, several agents, a
 *     web worker pool, replicated containers. Each call grabbing every core
 *     massively oversubscribes the machine, and nothing in the library said
 *     so or offered a way to stop it.
 *
 * Two environment variables, read once and cached:
 *
 *   SQT_NUM_THREADS   maximum threads any kernel may use (0/unset = OpenMP's
 *                     own default). Set this to 1 inside a process pool.
 *   SQT_OMP_MIN_WORK  minimum tasks x elements before a region goes parallel
 *                     at all (default 50,000).
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

inline long long min_work() {
    static const long long value = env_ll("SQT_OMP_MIN_WORK", 50'000LL);
    return value;
}

inline int max_threads() {
    static const int value = static_cast<int>(env_ll("SQT_NUM_THREADS", 0));
    return value;  // 0 => leave OpenMP's own choice alone
}

/** True when `tasks * work_per_task` justifies entering a parallel region. */
inline bool worth_parallel(std::size_t tasks, std::size_t work_per_task) {
    if (tasks <= 1) return false;
    if (max_threads() == 1) return false;
    const long long total =
        static_cast<long long>(tasks) * static_cast<long long>(work_per_task);
    return total >= min_work();
}

}  // namespace omp_policy
}  // namespace sqt
