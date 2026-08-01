#include "sqt/signal_state_machines.hpp"

#include <cmath>

namespace sqt {

std::vector<double> donchian_state_machine(
    const double* close,
    const double* entry_max,
    const double* exit_min,
    std::size_t   n)
{
    std::vector<double> values(n, 0.0);
    bool in_pos = false;
    for (std::size_t i = 0; i < n; ++i) {
        if (std::isnan(entry_max[i]) || std::isnan(exit_min[i])) continue;
        if (!in_pos && close[i] >= entry_max[i]) {
            in_pos = true;
        } else if (in_pos && close[i] <= exit_min[i]) {
            in_pos = false;
        }
        values[i] = in_pos ? 1.0 : 0.0;
    }
    return values;
}

std::vector<double> vwap_reversion_state_machine(
    const double* close,
    const double* vwap,
    double        entry_threshold,
    std::size_t   n)
{
    std::vector<double> values(n, 0.0);
    bool in_pos = false;
    for (std::size_t i = 0; i < n; ++i) {
        if (std::isnan(vwap[i])) continue;
        if (!in_pos && close[i] <= vwap[i] * (1.0 - entry_threshold)) {
            in_pos = true;
        } else if (in_pos && close[i] >= vwap[i]) {
            in_pos = false;
        }
        values[i] = in_pos ? 1.0 : 0.0;
    }
    return values;
}

}  // namespace sqt
