"""
A cost of capital from the caller's inputs: CAPM cost of equity, after-tax
cost of debt, WACC on market values, Hamada's unlevered and relevered beta,
and how much of the answer is the uncertainty in beta.

EVERY INPUT IS THE CALLER'S. The risk-free rate, the beta, the equity risk
premium, the cost of debt, the tax rate and the market values are
arguments; nothing is fetched and nothing is defaulted except a debt value
of zero. A cost of equity is the risk-free rate plus beta times the
premium, and the premium is the input that decides it -- 4% against 6% on
a beta of 1.2 is 240 bp of cost of equity -- so it is never supplied here.

    cost of equity        k_e  = r_f + beta x ERP
    after-tax debt        k_d (1 - t)
    WACC                  E/(D+E) k_e + D/(D+E) k_d (1 - t)
    unlevered beta        beta_U = beta_L / (1 + (1 - t) D/E)      (Hamada)
    relevered beta        beta_L* = beta_U (1 + (1 - t) D*/E*)

With no debt the equity weight is exactly 1 and the debt weight exactly 0,
so WACC is the cost of equity to the bit.

HAMADA'S ASSUMPTIONS ARE STATED, NOT HIDDEN: the debt is riskless (a debt
beta of zero), its amount is fixed, and its tax shield is worth t x D. At a
target leverage the relevered WACC holds the cost of debt where the caller
put it, though more debt usually costs more; the result says so.

THE SENSITIVITY IS TO BETA. A beta estimated by regression carries a
standard error, typically 0.1 to 0.3 on five years of monthly returns, and
k_e moves by ERP x SE for each standard error. Given one, the cost of
equity and the WACC are reported at beta - SE and beta + SE.

Refused, naming the argument and the value: an equity value that is not
positive, a negative debt value, a tax rate outside [0, 1), debt without a
cost of debt, a negative standard error or target leverage, and any
non-finite input.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from standard_quant_tools.error import ValidationError

__all__ = [
    "estimate_cost_of_capital",
    "relever_beta",
    "unlever_beta",
]


def _finite(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValidationError(f"{name}={value!r} is not a finite number.")
    return number


def _decimal_rate(value: Any, name: str) -> float:
    number = _finite(value, name)
    if abs(number) > 1.0:
        raise ValidationError(
            f"{name}={value!r} is not a decimal rate: rates here are within "
            "+/-1 (0.045 is 4.5%). Pass a percentage divided by 100."
        )
    return number


def _tax_rate(value: Any) -> float:
    number = _finite(value, "tax_rate")
    if not 0.0 <= number < 1.0:
        raise ValidationError(
            f"tax_rate={value!r} is outside [0, 1): it is a decimal share of "
            "income, so 21% is 0.21. A rate of 1 or more leaves no income "
            "after tax and no levered beta."
        )
    return number


def _leverage(value: Any, name: str) -> float:
    number = _finite(value, name)
    if number < 0.0:
        raise ValidationError(
            f"{name}={value!r} is negative; a debt-to-equity ratio is 0 or "
            "more. Net cash is 0 debt here, not negative debt."
        )
    return number


def unlever_beta(levered_beta: float, debt_to_equity: float, tax_rate: float) -> float:
    """Hamada: beta_U = beta_L / (1 + (1 - t) D/E)."""
    return float(levered_beta) / (
        1.0 + (1.0 - _tax_rate(tax_rate)) * _leverage(debt_to_equity, "debt_to_equity")
    )


def relever_beta(
    unlevered_beta: float, debt_to_equity: float, tax_rate: float
) -> float:
    """Hamada: beta_L = beta_U (1 + (1 - t) D/E)."""
    return float(unlevered_beta) * (
        1.0 + (1.0 - _tax_rate(tax_rate)) * _leverage(debt_to_equity, "debt_to_equity")
    )


def estimate_cost_of_capital(
    *,
    risk_free_rate: float,
    beta: float,
    equity_risk_premium: float,
    equity_value: float,
    tax_rate: float,
    debt_value: float = 0.0,
    cost_of_debt: Optional[float] = None,
    beta_standard_error: Optional[float] = None,
    target_debt_to_equity: Optional[float] = None,
) -> Dict[str, Any]:
    """
    CAPM cost of equity, after-tax cost of debt, WACC on market values,
    Hamada's unlevered beta and, at `target_debt_to_equity`, the relevered
    beta, cost of equity and WACC; at beta -/+ `beta_standard_error`, the
    cost of equity and WACC. `at_target` is None without a target and
    `beta_sensitivity` None without a standard error, and the after-tax
    cost of debt None without a cost of debt. The WACC at a target that
    adds debt nobody priced is NaN.
    """
    rf = _decimal_rate(risk_free_rate, "risk_free_rate")
    erp = _decimal_rate(equity_risk_premium, "equity_risk_premium")
    levered = _finite(beta, "beta")
    tax = _tax_rate(tax_rate)
    equity = _finite(equity_value, "equity_value")
    if equity <= 0.0:
        raise ValidationError(
            f"equity_value={equity_value!r} is not positive. WACC weights are "
            "market values, and equity with no positive market value has no "
            "cost of equity and no D/E; pass the market capitalization."
        )
    debt = _finite(debt_value, "debt_value")
    if debt < 0.0:
        raise ValidationError(
            f"debt_value={debt_value!r} is negative. Pass the market value of "
            "the debt, 0 with none; net cash is not negative debt here."
        )
    kd = None if cost_of_debt is None else _decimal_rate(cost_of_debt, "cost_of_debt")
    if debt > 0.0 and kd is None:
        raise ValidationError(
            f"debt_value={debt_value!r} needs cost_of_debt, the pre-tax yield "
            "the firm pays on new debt (e.g. 0.06); WACC cannot weight a cost "
            "it was not given."
        )
    se = None
    if beta_standard_error is not None:
        se = _finite(beta_standard_error, "beta_standard_error")
        if se < 0.0:
            raise ValidationError(
                f"beta_standard_error={beta_standard_error!r} is negative; a "
                "standard error is 0 or more."
            )
    target = (
        None
        if target_debt_to_equity is None
        else _leverage(target_debt_to_equity, "target_debt_to_equity")
    )

    notes: List[str] = []
    nan = float("nan")
    total = equity + debt
    w_equity = equity / total
    w_debt = debt / total
    # None without a cost of debt: with no debt WACC needs none.
    after_tax_debt = None if kd is None else kd * (1.0 - tax)
    debt_cost_term = 0.0 if after_tax_debt is None else w_debt * after_tax_debt

    def k_equity(b: float) -> float:
        return rf + b * erp

    def wacc(k_e: float) -> float:
        return w_equity * k_e + debt_cost_term

    d_to_e = debt / equity
    unlevered = unlever_beta(levered, d_to_e, tax)
    k_e = k_equity(levered)
    out: Dict[str, Any] = {
        "cost_of_equity": k_e,
        "after_tax_cost_of_debt": after_tax_debt,
        "equity_weight": w_equity,
        "debt_weight": w_debt,
        "debt_to_equity": d_to_e,
        "wacc": wacc(k_e),
        "unlevered_beta": unlevered,
        "unlevered_cost_of_equity": k_equity(unlevered),
        "at_target": None,
        "beta_sensitivity": None,
        "warnings": notes,
    }

    if target is not None:
        relevered = relever_beta(unlevered, target, tax)
        k_target = k_equity(relevered)
        w_debt_target = target / (1.0 + target)
        if target > 0.0 and kd is None:
            wacc_target = nan  # the target adds debt nobody priced
        else:
            debt_term = 0.0 if target == 0.0 else w_debt_target * kd * (1.0 - tax)
            wacc_target = (1.0 - w_debt_target) * k_target + debt_term
        out["at_target"] = {
            "debt_to_equity": target,
            "debt_weight": w_debt_target,
            "beta": relevered,
            "cost_of_equity": k_target,
            "wacc": wacc_target,
        }
        if target > d_to_e and kd is not None:
            notes.append(
                "at_target.wacc holds cost_of_debt where it was given; at "
                "higher leverage debt usually costs more, which would raise it."
            )

    if se is not None:
        low, high = levered - se, levered + se
        out["beta_sensitivity"] = {
            "beta_standard_error": se,
            "beta_low": low,
            "beta_high": high,
            "cost_of_equity_low": k_equity(low),
            "cost_of_equity_high": k_equity(high),
            "wacc_low": wacc(k_equity(low)),
            "wacc_high": wacc(k_equity(high)),
        }
    notes.append(
        "Hamada's betas take the debt as riskless (debt beta 0), its amount "
        "as fixed and its tax shield as t x D."
    )
    return out
