"""
Whether a series of VaR and ES forecasts was right: the exception series and
the tests read off it.

A VaR forecast at level c says the return will fall below minus the VaR
with probability p = 1 - c. Lined up against the returns that followed,
each date is an EXCEPTION or it is not, and a correct forecast makes the
exceptions independent Bernoulli(p) draws. Each test asks one part of that:

- Kupiec (1995), proportion of failures: is the exception RATE p?
  LR_pof = -2 ln[(1-p)^(T-x) p^x] + 2 ln[(1-x/T)^(T-x) (x/T)^x], chi^2(1).
- Christoffersen (1998), independence: does an exception make the next one
  more likely? A first-order Markov chain against independence, from the
  transition counts n_ij (state i on one date, j on the next), chi^2(1); and
  conditional coverage, LR_cc = LR_pof + LR_ind, chi^2(2).
- The Basel traffic light (Basel Committee 1996): on the last 250
  forecasts at 99%, 0-4 exceptions green, 5-9 yellow with a plus factor
  0.40, 0.50, 0.65, 0.75, 0.85, and 10 or more red with 1.00, added to the
  multiplier of 3.
- Acerbi and Szekely (2014), test 2 for expected shortfall:
  Z2 = sum_t X_t I_t / (T p ES_t) + 1, where X_t is the return and I_t the
  exception indicator. Its expected value is 0 when the forecasts are
  right and it is negative when the tail was understated; the p-value is
  the share of Z2 simulated under the forecasts below the one observed.

A forecast series gives only VaR and ES, not the whole distribution, so the
simulation needs one: each date's return is drawn from the NORMAL whose VaR
and ES at level c are that date's forecasts. Acerbi and Szekely found Z2's
null distribution close to the same across the distributions they tried,
which is what makes one family a fair stand-in; the result says which.

VaR and ES are losses, positive; an exception is a return below -VaR.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from standard_quant_tools.error import ValidationError

__all__ = [
    "BASEL_PLUS_FACTORS",
    "Z2_NULL_DISTRIBUTION",
    "acerbi_szekely_z2",
    "backtest_var_series",
    "basel_traffic_light",
    "christoffersen_independence",
    "kupiec_pof",
]

#: Plus factor by exception count in the yellow zone; green is 0, red 1.
BASEL_PLUS_FACTORS = {5: 0.40, 6: 0.50, 7: 0.65, 8: 0.75, 9: 0.85}
BASEL_OBSERVATIONS = 250
BASEL_CONFIDENCE = 0.99
BASEL_MULTIPLIER = 3.0

#: What Z2's p-value is simulated under, as the result reports it.
Z2_NULL_DISTRIBUTION = (
    "normal, one per date, with that date's VaR and ES at the stated level"
)

#: Draws for Z2's p-value.
Z2_SIMULATIONS = 10_000

#: Normal draws held at once (8 MB): the simulated series are drawn a block
#: of rows at a time, and since the generator fills rows in order, the block
#: size changes no draw.
_Z2_DRAWS_PER_BLOCK = 1_000_000


def _xlogy(x: float, y: float) -> float:
    """x ln y with 0 ln 0 = 0, the convention every likelihood here needs."""
    return 0.0 if x == 0 else x * math.log(y)


def _chi2_sf(statistic: float, df: int) -> float:
    from scipy import stats

    return float(stats.chi2.sf(statistic, df))


def kupiec_pof(n_exceptions: int, n_obs: int, tail_probability: float) -> Dict:
    """Kupiec's proportion-of-failures LR and its chi^2(1) p-value. Defined
    for every count, 0 and T included, by 0 ln 0 = 0."""
    x, t, p = int(n_exceptions), int(n_obs), float(tail_probability)
    rate = x / t
    lr = -2.0 * (_xlogy(t - x, 1.0 - p) + _xlogy(x, p)) + 2.0 * (
        _xlogy(t - x, 1.0 - rate) + _xlogy(x, rate)
    )
    lr = max(lr, 0.0)
    return {"lr": lr, "p_value": _chi2_sf(lr, 1)}


def christoffersen_independence(exceptions: np.ndarray) -> Dict[str, Any]:
    """
    Christoffersen's independence LR from the exception indicators, with the
    four transition counts. `lr` and `p_value` are None, with `null_reason`,
    where the test is undefined: no exception or nothing but exceptions
    (there is one state, so no transition to compare), and no transition out
    of one of the two states (its probability is 0/0).
    """
    hits = np.asarray(exceptions, dtype=bool)
    prev, nxt = hits[:-1], hits[1:]
    n00 = int(np.sum(~prev & ~nxt))
    n01 = int(np.sum(~prev & nxt))
    n10 = int(np.sum(prev & ~nxt))
    n11 = int(np.sum(prev & nxt))
    out: Dict[str, Any] = {
        "n00": n00,
        "n01": n01,
        "n10": n10,
        "n11": n11,
        "lr": None,
        "p_value": None,
        "null_reason": None,
    }
    if not hits.any():
        out["null_reason"] = (
            "there is no exception, so there is no clustering to test: every "
            "transition is from a quiet day to a quiet day"
        )
        return out
    if hits.all():
        out["null_reason"] = (
            "every date is an exception, so there is no quiet day to compare "
            "the day after an exception with"
        )
        return out
    if n10 + n11 == 0:
        out["null_reason"] = (
            "no exception is followed by another date (the only one is the "
            "last), so the chance of an exception after one is 0/0"
        )
        return out
    if n00 + n01 == 0:
        out["null_reason"] = (
            "no quiet day is followed by another date, so the chance of an "
            "exception after a quiet day is 0/0"
        )
        return out
    pi01 = n01 / (n00 + n01)
    pi11 = n11 / (n10 + n11)
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    restricted = _xlogy(n00 + n10, 1.0 - pi) + _xlogy(n01 + n11, pi)
    unrestricted = (
        _xlogy(n00, 1.0 - pi01)
        + _xlogy(n01, pi01)
        + _xlogy(n10, 1.0 - pi11)
        + _xlogy(n11, pi11)
    )
    lr = max(-2.0 * (restricted - unrestricted), 0.0)
    out.update(lr=lr, p_value=_chi2_sf(lr, 1))
    return out


def basel_traffic_light(n_exceptions: int) -> Dict[str, Any]:
    """The zone, plus factor and multiplier for an exception count on 250
    forecasts at 99%."""
    x = int(n_exceptions)
    if x <= 4:
        zone, plus = "green", 0.0
    elif x <= 9:
        zone, plus = "yellow", BASEL_PLUS_FACTORS[x]
    else:
        zone, plus = "red", 1.0
    return {
        "zone": zone,
        "plus_factor": plus,
        "multiplier": BASEL_MULTIPLIER + plus,
    }


def _z2(realized: np.ndarray, var: np.ndarray, es: np.ndarray, p: float) -> np.ndarray:
    """Z2 along the last axis: realized may be (T,) or (M, T)."""
    hits = realized < -var
    terms = np.where(hits, realized, 0.0) / (p * es)
    return terms.sum(axis=-1) / realized.shape[-1] + 1.0


def acerbi_szekely_z2(
    realized: np.ndarray,
    var: np.ndarray,
    es: np.ndarray,
    tail_probability: float,
    *,
    random_seed: int = 0,
    n_simulations: int = Z2_SIMULATIONS,
) -> Dict[str, Any]:
    """
    Z2 and its p-value simulated under the forecasts: each date's return
    drawn from N(mu_t, s_t^2) with s_t = (ES_t - VaR_t) / (phi(z)/p - z) and
    mu_t = s_t z - VaR_t, z = Phi^-1(1 - p), whose VaR and ES at level
    1 - p are exactly that date's. `None` values with `null_reason` when an
    ES is missing, not above its VaR, or not positive.
    """
    from scipy import stats

    if isinstance(n_simulations, bool) or int(n_simulations) < 1:
        raise ValidationError(
            f"n_simulations={n_simulations!r}: Z2's p-value is the share of "
            "simulated values below the observed one, so it needs at least one."
        )
    realized = np.asarray(realized, dtype=np.float64)
    var = np.asarray(var, dtype=np.float64)
    es = np.asarray(es, dtype=np.float64)
    p = float(tail_probability)
    out: Dict[str, Any] = {
        "z2": None,
        "p_value": None,
        "n_simulations": None,
        "null_distribution": None,
        "null_reason": None,
    }
    if not np.all(np.isfinite(es)):
        out["null_reason"] = (
            f"{int(np.sum(~np.isfinite(es)))} date(s) have no ES forecast, and Z2 "
            "averages the return over the ES of every date"
        )
        return out
    if np.any(es <= 0.0):
        out["null_reason"] = (
            f"ES is zero or negative on {int(np.sum(es <= 0.0))} date(s); Z2 "
            "divides by it as a loss"
        )
        return out
    if np.any(es <= var):
        out["null_reason"] = (
            f"ES is not above VaR on {int(np.sum(es <= var))} date(s). The mean "
            "of a tail beyond the VaR lies beyond it, and no distribution -- the "
            "normal the p-value is simulated under included -- has ES equal to VaR"
        )
        return out
    z = float(stats.norm.ppf(1.0 - p))
    scale = (es - var) / (float(stats.norm.pdf(z)) / p - z)
    location = scale * z - var
    observed = float(_z2(realized, var, es, p))
    rng = np.random.default_rng(int(random_seed))
    below = 0
    done = 0
    rows = max(1, _Z2_DRAWS_PER_BLOCK // len(realized))
    while done < n_simulations:
        size = min(rows, n_simulations - done)
        draws = location + scale * rng.standard_normal((size, len(realized)))
        below += int(np.sum(_z2(draws, var, es, p) < observed))
        done += size
    out.update(
        z2=observed,
        p_value=below / n_simulations,
        n_simulations=int(n_simulations),
        null_distribution=Z2_NULL_DISTRIBUTION,
    )
    return out


def backtest_var_series(
    frame: pd.DataFrame,
    confidence: float,
    *,
    random_seed: int = 0,
    n_simulations: int = Z2_SIMULATIONS,
    who: str = "backtest_var_forecasts",
) -> Dict[str, Any]:
    """
    Every test above on one forecast series.

    `frame` is date-indexed with columns `realized` and `var`, and `es`
    when expected shortfall is to be tested; at least two rows, every
    `realized` and `var` finite.
    """
    if not (0.0 < confidence < 1.0):
        raise ValidationError(
            f"{who}: confidence={confidence!r} must be strictly between 0 and 1."
        )
    for column in ("realized", "var"):
        if column not in frame.columns:
            raise ValidationError(
                f"{who}: the forecasts have no {column!r} column; they need "
                "'realized' and 'var', and 'es' to test expected shortfall."
            )
    if len(frame) < 2:
        raise ValidationError(
            f"{who}: {len(frame)} forecast(s); a backtest needs at least two, and "
            "the tests below have power only with many more."
        )
    realized = frame["realized"].to_numpy(dtype=np.float64)
    var = frame["var"].to_numpy(dtype=np.float64)
    for name, column in (("realized", realized), ("var", var)):
        if not np.all(np.isfinite(column)):
            raise ValidationError(
                f"{who}: {name} is missing or not finite on "
                f"{int(np.sum(~np.isfinite(column)))} date(s); every date tested "
                "needs a return and its VaR."
            )
    p = 1.0 - float(confidence)
    hits = realized < -var
    n_obs = int(len(hits))
    x = int(hits.sum())
    dates = frame.index
    warnings: List[str] = []
    kupiec = kupiec_pof(x, n_obs, p)
    christoffersen = christoffersen_independence(hits)
    cc_lr: Optional[float] = None
    cc_p: Optional[float] = None
    if christoffersen["lr"] is not None:
        cc_lr = kupiec["lr"] + christoffersen["lr"]
        cc_p = _chi2_sf(cc_lr, 2)
    if n_obs * p < 5:
        warnings.append(
            f"{n_obs} forecasts at {confidence:.1%} expect {n_obs * p:.2f} "
            "exceptions -- fewer than 5, so the chi-squared p-values are rough "
            "and the tests can only reject a large error."
        )

    basel: Dict[str, Any] = {
        "zone": None,
        "exceptions": None,
        "plus_factor": None,
        "multiplier": None,
        "first_date": None,
        "null_reason": None,
    }
    if abs(confidence - BASEL_CONFIDENCE) > 1e-12:
        basel["null_reason"] = (
            f"the traffic light is defined on 99% VaR and these are at "
            f"{confidence:.4g}"
        )
    elif n_obs < BASEL_OBSERVATIONS:
        basel["null_reason"] = (
            f"the traffic light counts the last {BASEL_OBSERVATIONS} forecasts "
            f"and there are {n_obs}"
        )
    else:
        recent = int(hits[-BASEL_OBSERVATIONS:].sum())
        basel.update(
            exceptions=recent,
            first_date=dates[-BASEL_OBSERVATIONS],
            **basel_traffic_light(recent),
        )

    if "es" in frame.columns:
        z2 = acerbi_szekely_z2(
            realized,
            var,
            frame["es"].to_numpy(dtype=np.float64),
            p,
            random_seed=random_seed,
            n_simulations=n_simulations,
        )
    else:
        z2 = {
            "z2": None,
            "p_value": None,
            "n_simulations": None,
            "null_distribution": None,
            "null_reason": "no ES forecasts were given, so there is no "
            "expected shortfall to test",
        }

    for label, value in (
        ("Kupiec's test of the exception rate", kupiec["p_value"]),
        ("Christoffersen's test of independence", christoffersen["p_value"]),
        ("the conditional-coverage test", cc_p),
        ("Acerbi and Szekely's Z2", z2["p_value"]),
    ):
        if value is not None and value < 0.05:
            warnings.append(f"{label} rejects at 5% (p = {value:.4f}).")

    return {
        "confidence": float(confidence),
        "n_observations": n_obs,
        "n_exceptions": x,
        "expected_exceptions": n_obs * p,
        "exception_rate": x / n_obs,
        "first_date": dates[0],
        "last_date": dates[-1],
        "exception_dates": list(dates[hits]),
        "kupiec": kupiec,
        "christoffersen": christoffersen,
        "conditional_coverage": {"lr": cc_lr, "p_value": cc_p},
        "basel": basel,
        "z2": z2,
        "warnings": warnings,
    }
