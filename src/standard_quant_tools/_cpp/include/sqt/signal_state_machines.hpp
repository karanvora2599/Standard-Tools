#pragma once

#include <cstddef>
#include <vector>

namespace sqt {

/**
 * Donchian breakout entry/exit hysteresis.
 *
 * Enter long when close[i] >= entry_max[i]; exit to flat when
 * close[i] <= exit_min[i]. A NaN in entry_max[i]/exit_min[i] (rolling
 * warmup) does NOT update the position state for that bar, and the output
 * for that bar carries the position already held (1.0 if in_pos, else
 * 0.0) rather than hardcoding 0.0 -- a caller reading this as a real
 * position series sees the position actually held through the gap, not a
 * phantom close/reopen blip. Matches _donchian_state_machine in
 * backtest/strategies.py exactly.
 *
 * @param close      Close prices, length n.
 * @param entry_max  Rolling entry-channel high (already shifted/lagged by
 *                    the Python caller), length n.
 * @param exit_min   Rolling exit-channel low (already shifted/lagged),
 *                    length n.
 * @param n          Number of bars.
 * @returns  Vector of length n: 1.0 = long, 0.0 = flat. Empty if n==0.
 */
std::vector<double> donchian_state_machine(
    const double* close,
    const double* entry_max,
    const double* exit_min,
    std::size_t   n);

/**
 * VWAP mean-reversion entry/exit hysteresis.
 *
 * Enter long when close[i] <= vwap[i] * (1 - entry_threshold); exit to
 * flat once close[i] >= vwap[i]. A NaN in vwap[i] (rolling warmup) does
 * NOT update the position state for that bar, and the output for that bar
 * carries the position already held (1.0 if in_pos, else 0.0) rather than
 * hardcoding 0.0 -- a caller reading this as a real position series sees
 * the position actually held through the gap, not a phantom close/reopen
 * blip. Matches _vwap_reversion_state_machine in backtest/strategies.py
 * exactly.
 *
 * @param close            Close prices, length n.
 * @param vwap             Rolling VWAP, length n.
 * @param entry_threshold  Fractional drop below VWAP that triggers entry.
 * @param n                Number of bars.
 * @returns  Vector of length n: 1.0 = long, 0.0 = flat. Empty if n==0.
 */
std::vector<double> vwap_reversion_state_machine(
    const double* close,
    const double* vwap,
    double        entry_threshold,
    std::size_t   n);

}  // namespace sqt
