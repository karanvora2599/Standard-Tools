"""
The financing terms an agent tool takes, and what it reports back.

`FinancingSpecInput` is the input model of `backtest.financing.FinancingSpec`
-- a benchmark rate (scalar or dated), the credit and debit spreads around
it, a borrow rate per asset with an explicit default, and where short
proceeds sit. The model itself is defined beside the dataclass, in
`backtest/financing.py`, because modeling's `PortfolioSimSpec` takes it too
and modeling cannot import this package while it loads; this module is where
the tools read it.

The shared-cash tools -- the portfolio simulation, the pair backtest and a
model's portfolio evaluation -- take `financing` beside the scalar rates
they already took. The scalars are a shorthand for the same terms (a
constant benchmark with no spreads), so a call that sets one of them away
from its default AND passes `financing` is refused by name: picking one
would silently discard the other.

The run_strategy-based tools take the three scalar rates only (`cash_rate`,
`financing_rate`, `borrow_fee_bps`). A dated benchmark and a borrow rate per
asset are terms of a book, and the nested model costs every tool that
carries it its whole schema; the library's `run_strategy` and
`backtest_grid` take the full terms for a caller who needs them.

Nothing here imports the tool layer at load: `agent/models.py` imports this
module, and the date parser lives in a module that imports `agent/models.py`.
"""

from __future__ import annotations

from typing import Annotated, Any, Dict, Mapping, Optional

from pydantic import BeforeValidator, ConfigDict, Field

from standard_quant_tools.agent.runtimes._json_safe import (
    ExplainsNulls,
    Reason,
    finite_or_none,
)
from standard_quant_tools.backtest.financing import (
    _THIRTY_360,
    FinancingSpec,
    FinancingSpecInput,
    refuse_scalars_with_spec,
    require_financing_day_count,
    resolve_debit_rate,
)
from standard_quant_tools.error import ValidationError

_Stat = Annotated[Optional[float], BeforeValidator(finite_or_none)]

__all__ = [
    "FinancingSpecInput",
    "FinancingBenchmarkReport",
    "FINANCING_NULL_REASONS",
    "day_count_reason",
    "financing_spec",
    "strategy_financing_kwargs",
    "account_financing_kwargs",
    "cash_rate_field",
    "financing_rate_field",
    "borrow_fee_field",
]


# ── The input fields every run_strategy-based tool takes ────────────────
#
# No description of their own: fourteen tools carry each byte, and under the default tool
# detail the budget those bytes come out of is what decides how many tools
# are listed in full. The names say what they are, and risk_free_rate's
# description says what the two rates default to. Factories rather than
# shared FieldInfo objects: one Field instance reused across models is
# mutated by each model it is attached to.


def cash_rate_field() -> Any:
    """Annual rate idle cash earns, as a decimal; None is risk_free_rate."""
    return Field(None, ge=-1.0, le=1.0)


def financing_rate_field() -> Any:
    """Annual rate paid on exposure beyond 1x; None is risk_free_rate."""
    return Field(None, ge=-1.0, le=1.0)


def borrow_fee_field() -> Any:
    """Annual basis points a short pays on its size; 0.0 borrows free."""
    return Field(0.0, ge=0.0, le=100_000.0)


def day_count_reason(value: Any, name: str = "day_count") -> Any:
    """A before-validator body for a financing day count: 30/360 is refused
    with the reason (see backtest.financing.require_financing_day_count)
    rather than by the bare literal error; anything else passes through to
    the field's own check."""
    if isinstance(value, str) and _THIRTY_360.match(value):
        try:
            require_financing_day_count(value, name)
        except ValidationError as exc:
            raise ValueError(str(exc)) from None
    return value


# ── Reading the input ────────────────────────────────────────────────────


def financing_spec(
    model: Optional[FinancingSpecInput], tool: str
) -> Optional[FinancingSpec]:
    """
    The library spec from a tool's `financing`, with a dated benchmark's keys
    parsed as ISO dates by the same parser every dated input on this surface
    uses (so a bad key is named the same way everywhere).
    """
    if model is None:
        return None
    from standard_quant_tools.agent.runtimes._shared import parse_date_keys

    dated: Optional[Mapping[Any, float]] = None
    if isinstance(model.benchmark_rate, dict):
        dated = parse_date_keys(
            model.benchmark_rate, "financing.benchmark_rate", tool, finite=True
        )
    return model.to_spec(benchmark=dated)


def strategy_financing_kwargs(input_data: Any) -> Dict[str, Any]:
    """
    The financing keyword arguments a run_strategy-based tool passes to
    `run_strategy` / `backtest_grid`: its three scalar rates.

    These tools take the scalars only. A dated benchmark and a borrow rate
    per asset are terms of a book, which the shared-cash tools take as
    `financing`; the library's `run_strategy` takes them too. Every one of
    these tools ran with the rates at their defaults before it took them,
    which is exactly what the defaults reproduce.
    """
    # A tool that runs only registry strategies takes cash_rate alone: those
    # are long or flat, so financing and borrow keep their defaults there.
    return {
        "cash_rate": input_data.cash_rate,
        "financing_rate": getattr(input_data, "financing_rate", None),
        "borrow_fee_bps": float(getattr(input_data, "borrow_fee_bps", 0.0)),
    }


#: The shared-cash tools' scalar financing inputs at their defaults.
_ACCOUNT_DEFAULTS: Dict[str, Any] = {
    "risk_free_rate": 0.0,
    "cash_rate": None,
    "margin_interest_rate": None,
    "margin_spread_bps": 0.0,
    "borrow_fee_bps": 0.0,
    "short_proceeds": "net_cash",
    "short_rebate_rate": None,
    "day_count": "ACT/365F",
}


def account_financing_kwargs(input_data: Any, tool: str) -> Dict[str, Any]:
    """
    The financing keyword arguments a shared-cash tool (the portfolio
    simulation, the pair backtest, a model's portfolio evaluation) passes to
    `run_portfolio_simulation`: the scalar terms resolved -- a rate left as
    None takes the risk-free rate, and borrowing costs it plus
    `margin_spread_bps` -- or the spec.
    """
    model = getattr(input_data, "financing", None)
    if model is not None:
        refuse_scalars_with_spec(
            {name: getattr(input_data, name, None) for name in _ACCOUNT_DEFAULTS},
            _ACCOUNT_DEFAULTS,
            tool,
        )
        return {"financing": financing_spec(model, tool)}
    rf = float(input_data.risk_free_rate)
    rebate = input_data.short_rebate_rate
    return {
        "cash_rate": (
            rf if input_data.cash_rate is None else float(input_data.cash_rate)
        ),
        "margin_interest_rate": resolve_debit_rate(
            input_data.margin_interest_rate, rf, input_data.margin_spread_bps
        ),
        "borrow_fee_bps": float(input_data.borrow_fee_bps),
        "short_proceeds": input_data.short_proceeds,
        "short_rebate_rate": 0.0 if rebate is None else float(rebate),
        "day_count": input_data.day_count,
    }


# ── What the result says ─────────────────────────────────────────────────


_NO_RATE_USED = (
    "the run had a single bar, so no balance was carried and no rate was used"
)

#: Why a field of the financing block is null under a spec.
FINANCING_NULL_REASONS: Dict[str, Reason] = {
    "cash_rate": "the credit rate moved with the dated benchmark bar by bar; "
    "benchmark gives its first, last and mean, credit_spread_bps the spread "
    "under it",
    "margin_interest_rate": "the debit rate moved with the dated benchmark bar "
    "by bar; benchmark gives its first, last and mean, debit_spread_bps the "
    "spread over it",
    "short_rebate_rate": "the rebate moved with the dated benchmark bar by bar; "
    "benchmark gives its first, last and mean, rebate_spread_bps the spread "
    "under it",
    "borrow_fee_bps": "the assets held short paid different borrow rates "
    "(borrow_bps_by_asset), or nothing was held short and the terms set no "
    "default_borrow_bps",
}


class FinancingBenchmarkReport(ExplainsNulls):
    """The benchmark a run was financed over, as it was used."""

    model_config = ConfigDict(extra="forbid")
    null_reasons = {
        "rate": "the benchmark was a dated series; first, last and mean " "describe it",
        "first": _NO_RATE_USED,
        "last": _NO_RATE_USED,
        "mean": _NO_RATE_USED,
    }

    kind: str = Field(..., description="'scalar' or 'dated'.")
    rate: _Stat = Field(None, description="The scalar benchmark; null when dated.")
    first: _Stat = Field(None, description="The first rate a balance accrued at.")
    last: _Stat = Field(None, description="The last rate used.")
    mean: _Stat = Field(
        None,
        description="The mean rate, each weighted by the calendar days it was "
        "in force.",
    )
    first_date: Optional[str] = Field(
        None, description="Bar date of the first rate used (dated only)."
    )
    last_date: Optional[str] = Field(
        None, description="Bar date of the last rate used (dated only)."
    )
    fill: Optional[str] = Field(
        None, description="benchmark_fill, for a dated benchmark."
    )
    n_bars_filled: Optional[int] = Field(
        None,
        description="Bars that took an earlier date's rate under "
        "benchmark_fill='previous'.",
    )
