import logging
from typing import Any, Dict, Tuple, Union

import numpy as np
import pandas as pd

from standard_quant_tools.error import ValidationError
from standard_quant_tools.metrics.risk_metrics import max_drawdown
from standard_quant_tools.portfolio.portfolio import build_portfolio

logger = logging.getLogger(__name__)

# Well-known approximate peak/trough windows for major historical drawdowns.
# These are informal, widely-cited market-history dates, not academically
# precise event-study boundaries -- good enough for "how would my current
# allocation have fared during X," not a research-grade event study.
_SCENARIOS: Dict[str, Tuple[str, str]] = {
    "black_monday_1987": ("1987-10-14", "1987-10-19"),
    "dotcom_2000": ("2000-03-24", "2002-10-09"),
    "gfc_2008": ("2008-09-01", "2009-03-09"),
    "volmageddon_2018": ("2018-01-26", "2018-02-09"),
    "covid_2020": ("2020-02-19", "2020-03-23"),
    "rate_shock_2022": ("2022-01-03", "2022-10-12"),
}


def list_stress_scenarios() -> Dict[str, Dict[str, str]]:
    """Named built-in scenarios and their (start, end) date windows."""
    return {name: {"start": s, "end": e} for name, (s, e) in _SCENARIOS.items()}


def scenario_dates(scenario: str) -> Tuple[str, str]:
    """Look up a named scenario's (start_date, end_date). Raises
    ValidationError for an unknown scenario name."""
    if scenario not in _SCENARIOS:
        raise ValidationError(
            f"Unknown scenario {scenario!r}. Available: {sorted(_SCENARIOS)}"
        )
    return _SCENARIOS[scenario]


def replay_stress_scenario(
    returns_df: pd.DataFrame,
    weights: Union[list, np.ndarray],
) -> Dict[str, Any]:
    """
    Replay a portfolio's weights against an already-sliced historical
    returns window (the caller slices returns_df to the scenario's date
    range before calling this — see get_stress_test_result's per-ticker
    fetch, which needs to tolerate individual tickers having no data for
    an old window before this function ever runs).

    Returns:
        UNITS: every `_pct` field here is a FRACTION, not a percentage --
        -0.20 means -20%. The suffix is a misnomer and it is kept because
        callers read these keys, but it is worth being explicit:
        `backtest/futures_engine.py` returns a field of the same name
        already multiplied by 100, so the two differ by exactly 100x and
        neither said so.

        Dict with portfolio_return_pct, max_drawdown_pct,
        worst_day_return_pct, worst_day_date, best_day_return_pct,
        best_day_date, n_trading_days.

    Raises:
        ValidationError: returns_df has no rows (empty scenario window).
    """
    if returns_df.empty:
        raise ValidationError(
            "returns_df is empty — no overlapping trading days for this scenario window"
        )

    portfolio_returns = build_portfolio(returns_df, weights)
    equity = (1.0 + portfolio_returns).cumprod()

    total_return = float(equity.iloc[-1] - 1.0)

    # THE PEAK HAS TO INCLUDE THE STARTING VALUE. `cumprod` never contains
    # 1.0, so a first-day crash sits AT the running maximum and shows no
    # drawdown at all: a -20% opening day reported max_drawdown_pct 0.0
    # next to portfolio_return_pct -18.39% and worst_day_return_pct -20%,
    # in the same result. A stress scenario whose whole point is the first
    # day is exactly where this bites.
    seeded = pd.concat([pd.Series([1.0]), equity], ignore_index=True)
    mdd = float(max_drawdown(seeded))

    worst_idx = portfolio_returns.idxmin()
    best_idx = portfolio_returns.idxmax()

    logger.debug(
        "[stress_test] n_days=%d  total_return=%.4f  max_drawdown=%.4f",
        len(portfolio_returns),
        total_return,
        mdd,
    )

    return {
        "portfolio_return_pct": total_return,
        "max_drawdown_pct": mdd,
        "worst_day_return_pct": float(portfolio_returns.loc[worst_idx]),
        "worst_day_date": (
            str(worst_idx.date()) if hasattr(worst_idx, "date") else str(worst_idx)
        ),
        "best_day_return_pct": float(portfolio_returns.loc[best_idx]),
        "best_day_date": (
            str(best_idx.date()) if hasattr(best_idx, "date") else str(best_idx)
        ),
        "n_trading_days": len(portfolio_returns),
    }
