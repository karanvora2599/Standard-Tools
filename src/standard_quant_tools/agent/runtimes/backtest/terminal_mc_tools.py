"""
Monte Carlo that keeps only where the paths ENDED.

WHY A SECOND SIMULATION TOOL. `run_monte_carlo_simulation` allocates
`n_simulations x horizon` and returns the distribution of paths, which is
the right answer when the question is about the journey -- worst drawdown
along the way, time underwater, path dependence.

It is the wrong shape when the question is only about the destination.
A million simulations over 252 days is roughly 2 GB of path matrix for a
handful of terminal quantiles, and the memory is what caps the simulation
count rather than the statistics. `simulate_forward_paths_terminal` keeps
only terminal outcomes, so the count can go where the precision actually
needs it.

The finance is identical -- same block bootstrap, same resampling. Nothing
here is new maths; it is an existing kernel that no tool could reach.
"""

from __future__ import annotations

import logging
import math
from typing import Annotated, Any, List, Optional

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

from standard_quant_tools.agent.runtimes.data.models import DataSource, resolve_source
from standard_quant_tools.backtest.monte_carlo import simulate_forward_paths_terminal

logger = logging.getLogger(__name__)


def _finite_or_none(value: Any) -> Any:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value if math.isfinite(float(value)) else None
    return value


Stat = Annotated[Optional[float], BeforeValidator(_finite_or_none)]


class TerminalMonteCarloInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    returns: DataSource = Field(
        ..., description="The return series: a symbol, a reference, or values."
    )
    horizon_days: int = Field(..., gt=0, description="How far forward to simulate.")
    n_simulations: int = Field(
        10_000,
        gt=0,
        le=5_000_000,
        description=(
            "Terminal-only storage means this can be large. The ceiling is "
            "wall-clock, not memory."
        ),
    )
    block_size: int = Field(
        20,
        gt=0,
        description=(
            "Bootstrap block length. Resampling single days destroys the "
            "serial correlation that drives drawdowns, so blocks are the "
            "default rather than an option."
        ),
    )
    initial_capital: float = Field(10_000.0, gt=0)
    seed: Optional[int] = Field(None, description="Set it for a reproducible answer.")


class TerminalMonteCarloResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n_simulations: int = 0
    horizon_days: int = 0
    initial_capital: Stat = None
    terminal_median: Stat = None
    terminal_p5: Stat = None
    terminal_p95: Stat = None
    prob_loss: Stat = Field(
        None, description="Fraction of paths ending below initial capital."
    )
    terminal_var_95: Stat = None
    terminal_cvar_95: Stat = None
    warnings: List[str] = Field(default_factory=list)


def run_terminal_monte_carlo(
    input_data: TerminalMonteCarloInput,
) -> TerminalMonteCarloResult:
    """Where the paths END, without ever holding the paths."""
    returns = resolve_source(input_data.returns, what="run_terminal_monte_carlo")

    result = simulate_forward_paths_terminal(
        returns,
        horizon_days=input_data.horizon_days,
        n_simulations=input_data.n_simulations,
        block_size=input_data.block_size,
        initial_capital=input_data.initial_capital,
        seed=input_data.seed,
    )

    warnings: List[str] = list(result.get("warnings", []))
    warnings.append(
        "TERMINAL outcomes only. This says nothing about the path -- the "
        "worst drawdown along the way and time underwater are not in here, "
        "and a distribution of endpoints can look benign while every path "
        "reaching it was unholdable. run_monte_carlo_simulation is the tool "
        "for the journey."
    )
    if input_data.seed is None:
        warnings.append(
            "No seed, so this answer is not reproducible. Set one before "
            "quoting a number anybody will act on."
        )
    if len(returns) < input_data.block_size * 5:
        warnings.append(
            f"only {len(returns)} observations against a block size of "
            f"{input_data.block_size}: the bootstrap is resampling a handful "
            "of distinct blocks, so the spread understates real uncertainty."
        )

    return TerminalMonteCarloResult(
        n_simulations=input_data.n_simulations,
        horizon_days=input_data.horizon_days,
        initial_capital=input_data.initial_capital,
        terminal_median=result.get("terminal_median"),
        terminal_p5=result.get("terminal_p5"),
        terminal_p95=result.get("terminal_p95"),
        prob_loss=result.get("prob_loss"),
        terminal_var_95=result.get("terminal_var_95"),
        terminal_cvar_95=result.get("terminal_cvar_95"),
        warnings=warnings,
    )


__all__ = [
    "TerminalMonteCarloInput",
    "TerminalMonteCarloResult",
    "run_terminal_monte_carlo",
]
