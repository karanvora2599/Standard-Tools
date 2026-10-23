"""
A portfolio's risk under a factor model, split into the factors and the
names' own risk -- for the portfolio, and against a benchmark.

THE MODEL. Asset returns are r = B f + e: B the loadings (asset x factor),
f the factor returns with covariance F, and e the specific returns, with
variances D, uncorrelated with the factors and with each other. Then the
covariance of the assets is

    Sigma = B F B' + D,

and a portfolio w has exposures x = B'w, factor variance x'Fx, specific
variance sum(w_i^2 d_i), and nothing else: the covariance between its
factor part and its specific part is zero BY CONSTRUCTION -- that is the
model's assumption, not a measurement, and the result states it rather than
leaving a reader to wonder where the cross term went. A book whose residuals
do correlate (two share classes of one company, a merger pair) breaks the
assumption, and nothing computed from B, F and D can see that.

SPECIFIC VARIANCE IS REQUIRED. Leaving D out treats every name as riskless
beyond its factors, which overstates the factor share of the risk without
limit: two names with a beta of 1 to a market at 1% volatility and 40%
specific volatility each are 100% market without D and 0.12% market with
it. `get_factor_exposure_budget` reported the first number as the second
for exactly that reason.

THE DECOMPOSITION IS EULER'S. Volatility is homogeneous of degree one in
the weights, so it is the sum of each weight times its marginal:

    sigma = sum_i w_i (Sigma w)_i / sigma
          = sum_k x_k (F x)_k / sigma + sum_i w_i^2 d_i / sigma,

by asset, and by factor plus the specific total. Each factor's contribution
is also exposure x factor volatility x the factor's correlation with the
portfolio, x_k sigma_k rho_k, the form of Menchero and Davis (2011) --
which separates "how much of the factor do I hold" from "how much does the
factor move" from "how much does it move WITH me". The same algebra on the
active weights w - b gives the tracking error and its contributions
(Grinold and Kahn, Active Portfolio Management, ch. 3).

UNITS ARE THE INPUTS'. F and D in annual variance give annual volatility
and tracking error; daily in, daily out. Nothing here annualises.

Every name is checked against every other: the covariance's factors are
the loadings' factors, exactly; every asset with loadings has a specific
variance and every specific variance an asset; every asset in the weights
or the benchmark has loadings. A disagreement is refused by name, because
the alternatives -- a missing loading read as zero, a held name left out --
each understate the risk without saying so.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from standard_quant_tools._blas import single_threaded_blas
from standard_quant_tools.error import ValidationError
from standard_quant_tools.numeric_contract import require_finite_covariance
from standard_quant_tools.portfolio.construction import (
    _finite_weights,
    _named_once,
    _repair_psd,
    _zero_variance_assets,
)

__all__ = ["decompose_factor_risk"]

#: How many offending names a refusal lists before counting the rest.
_SHOWN = 5

#: The statement carried by `factor_specific_covariance`.
_CROSS_TERM_IS_ZERO = 0.0


def _listed(items: Sequence[Any]) -> str:
    items = list(items)
    shown = ", ".join(repr(i) for i in items[:_SHOWN])
    more = len(items) - _SHOWN
    return f"[{shown}{f' and {more} more' if more > 0 else ''}]"


def _factor_covariance(
    factor_covariance: Any, who: str
) -> "tuple[pd.DataFrame, List[str]]":
    """The factor covariance as a named, checked frame, and any repair note.

    A mapping is read by name: every row must have a value for every
    factor, the diagonal included, and nothing else. A frame's row and
    column names must be the same factors (its columns are put in its rows'
    order)."""
    if isinstance(factor_covariance, pd.DataFrame):
        frame = factor_covariance.copy()
        frame.index = [str(r) for r in frame.index]
        frame.columns = [str(c) for c in frame.columns]
        _named_once(frame.index, "factor_covariance's rows", who)
        _named_once(frame.columns, "factor_covariance's columns", who)
        rows, columns = list(frame.index), list(frame.columns)
        if set(rows) != set(columns):
            raise ValidationError(
                f"{who}: factor_covariance's rows {_listed(rows)} and columns "
                f"{_listed(columns)} name different factors. A covariance has "
                "one row and one column per factor, under the same names."
            )
        frame = frame.loc[rows, rows]
    elif isinstance(factor_covariance, Mapping):
        named: Dict[str, Dict[str, Any]] = {}
        for key, row in factor_covariance.items():
            if not isinstance(row, Mapping):
                raise ValidationError(
                    f"{who}: factor_covariance[{str(key)!r}] is a "
                    f"{type(row).__name__}; each row is a map of factor to "
                    "covariance, read by name."
                )
            named[str(key)] = {str(c): v for c, v in row.items()}
        rows = list(named)
        _named_once(
            pd.Index([str(r) for r in factor_covariance]),
            "factor_covariance's rows",
            who,
        )
        expected = set(rows)
        ragged = []
        for key, row in named.items():
            lacks = [f for f in rows if f not in row]
            extra = [c for c in row if c not in expected]
            if lacks:
                ragged.append(f"factor_covariance[{key!r}] lacks {_listed(lacks)}")
            if extra:
                ragged.append(
                    f"factor_covariance[{key!r}] has {_listed(extra)}, which have "
                    "no row"
                )
        if ragged:
            raise ValidationError(
                f"{who}: factor_covariance is not square under its factor "
                f"names: {'; '.join(ragged[:_SHOWN])}. Every row needs a value "
                "for every factor, the diagonal included -- a missing "
                "covariance is not zero."
            )
        try:
            values = [[float(named[r][c]) for c in rows] for r in rows]
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                f"{who}: factor_covariance holds a value that is not a number "
                f"({exc})."
            ) from exc
        frame = pd.DataFrame(values, index=rows, columns=rows)
    else:
        raise ValidationError(
            f"{who}: factor_covariance is a {type(factor_covariance).__name__}; "
            "give it as a map of factor -> {factor: covariance} (or a frame "
            "with the factors as its rows and columns)."
        )
    if frame.shape[0] == 0:
        raise ValidationError(f"{who}: factor_covariance names no factor.")
    array = require_finite_covariance(
        frame.to_numpy(dtype=float), "factor_covariance", who
    )
    flat = _zero_variance_assets(frame.columns, np.diag(array))
    if flat:
        raise ValidationError(
            f"{who}: factor_covariance gives {_listed(flat)} a variance of zero "
            "(or a negative one, or float residue next to the largest). A "
            "factor that does not move carries no risk and has no correlation "
            "with anything; drop it from the loadings and the covariance."
        )
    frame = pd.DataFrame(array, index=frame.index, columns=frame.columns)
    return _repair_psd(frame, who)


def _loadings(factor_loadings: Any, factors: List[str], who: str) -> pd.DataFrame:
    """Asset x factor loadings, columns in the covariance's factor order.

    Every asset needs a loading on every factor of the covariance and on no
    other: absent is not zero, so a gap is refused by asset and factor, and
    a factor the covariance does not have cannot be given a risk."""
    if isinstance(factor_loadings, pd.DataFrame):
        frame = factor_loadings.copy()
        frame.index = [str(a) for a in frame.index]
        frame.columns = [str(c) for c in frame.columns]
        _named_once(frame.index, "factor_loadings' assets", who)
        _named_once(frame.columns, "factor_loadings' factors", who)
        columns = list(frame.columns)
        rows = {
            asset: dict(zip(columns, values))
            for asset, values in zip(frame.index, frame.to_numpy(dtype=object))
        }
    elif isinstance(factor_loadings, Mapping):
        rows = {}
        for asset, row in factor_loadings.items():
            if not isinstance(row, Mapping):
                raise ValidationError(
                    f"{who}: factor_loadings[{str(asset)!r}] is a "
                    f"{type(row).__name__}; each asset's loadings are a map of "
                    "factor to loading."
                )
            rows[str(asset)] = {str(f): v for f, v in row.items()}
        _named_once(pd.Index(list(rows)), "factor_loadings' assets", who)
    else:
        raise ValidationError(
            f"{who}: factor_loadings is a {type(factor_loadings).__name__}; "
            "give it as a map of asset -> {factor: loading}."
        )
    if not rows:
        raise ValidationError(f"{who}: factor_loadings names no asset.")
    known = set(factors)
    missing: List[str] = []
    unknown: Dict[str, None] = {}
    bad: List[str] = []
    for asset, row in rows.items():
        for factor in factors:
            if factor not in row or row[factor] is None or _is_nan(row[factor]):
                missing.append(f"{asset}/{factor}")
        for factor, value in row.items():
            if factor not in known:
                unknown[factor] = None
            elif value is not None and not _is_nan(value) and not _finite(value):
                bad.append(f"{asset}/{factor}")
    if unknown:
        raise ValidationError(
            f"{who}: factor_loadings name the factor(s) {_listed(list(unknown))}, "
            f"which factor_covariance does not have (it has {_listed(factors)}). "
            "A loading on a factor with no variance or covariances cannot be "
            "turned into a risk; add the factor to factor_covariance or drop it "
            "from the loadings."
        )
    if missing:
        raise ValidationError(
            f"{who}: {len(missing)} asset/factor loading(s) are missing: "
            f"{_listed(missing)}. Give every asset a loading on every factor of "
            "factor_covariance (0.0 where it genuinely has none); a missing "
            "loading is not a zero one."
        )
    if bad:
        raise ValidationError(
            f"{who}: factor_loadings are not finite at {_listed(bad)}; every "
            "loading must be a finite number."
        )
    return pd.DataFrame(
        [[float(rows[a][f]) for f in factors] for a in rows],
        index=list(rows),
        columns=factors,
    )


def _is_nan(value: Any) -> bool:
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _specific(specific_variance: Any, assets: List[str], who: str) -> np.ndarray:
    """Each asset's specific variance, in `assets` order, or a refusal naming
    the assets the two inputs disagree on or a bad value."""
    if specific_variance is None:
        raise ValidationError(
            f"{who}: specific_variance is required. Without it every name is "
            "riskless beyond its factors, and the factor share of the risk is "
            "overstated without limit; give each asset's idiosyncratic return "
            "variance (0.0 only if it genuinely has none)."
        )
    values = pd.Series(specific_variance, dtype=float)
    values.index = [str(a) for a in values.index]
    _named_once(values.index, "specific_variance", who)
    known = set(assets)
    absent = [a for a in assets if a not in values.index]
    extra = [a for a in values.index if a not in known]
    if absent or extra:
        parts = []
        if absent:
            parts.append(f"no entry for {_listed(absent)}")
        if extra:
            parts.append(f"entries for {_listed(extra)}, which have no loadings")
        raise ValidationError(
            f"{who}: specific_variance and factor_loadings must name the same "
            f"assets; specific_variance has {' and '.join(parts)}. They are two "
            "halves of one model, and an asset with only one of them is a "
            "model assembled from two different universes."
        )
    bad = {
        a: float(values[a])
        for a in assets
        if not math.isfinite(float(values[a])) or float(values[a]) < 0
    }
    if bad:
        raise ValidationError(
            f"{who}: specific_variance must be a finite, non-negative variance "
            f"for every asset; got {dict(list(bad.items())[:_SHOWN])}. It is a "
            "variance (volatility squared), in the units of factor_covariance."
        )
    return values.loc[assets].to_numpy(dtype=float)


def _weight_vector(
    weights: Any, field: str, assets: List[str], who: str
) -> "tuple[np.ndarray, List[str]]":
    """A weight map as a vector over the model's assets (zero where it names
    none), and the names it gave, in its order."""
    series = _finite_weights(weights, field, who)
    series.index = [str(a) for a in series.index]
    known = set(assets)
    absent = [a for a in series.index if a not in known]
    if absent:
        raise ValidationError(
            f"{who}: {field} name(s) {_listed(absent)}, which have no factor "
            "loadings. A held asset with no loadings would be left out of the "
            "risk -- understating it by exactly that position -- so it is "
            "refused rather than excluded. Give its loadings and its specific "
            "variance."
        )
    vector = np.zeros(len(assets), dtype=float)
    position = {a: i for i, a in enumerate(assets)}
    for name, value in series.items():
        vector[position[name]] = float(value)
    return vector, list(series.index)


def _ratio(numerator: float, denominator: float) -> float:
    """numerator / denominator, NaN (reported as null with its reason) when
    the denominator is zero."""
    return float(numerator / denominator) if denominator > 0 else float("nan")


def _side(
    v: np.ndarray,
    B: np.ndarray,
    F: np.ndarray,
    d: np.ndarray,
    factors: List[str],
    assets: List[str],
    rows: List[str],
    who: str,
    what: str,
) -> Dict[str, Any]:
    """The decomposition of one weight vector: the portfolio's, or the
    active weights against the benchmark."""
    with single_threaded_blas():
        with np.errstate(over="ignore", invalid="ignore"):
            x = B.T @ v
            Fx = F @ x
            factor_variance = float(x @ Fx)
            specific_by_asset = v * v * d
            specific_variance = float(np.sum(specific_by_asset))
            BFx = B @ Fx
    if not (
        math.isfinite(factor_variance)
        and math.isfinite(specific_variance)
        and np.all(np.isfinite(BFx))
    ):
        raise ValidationError(
            f"{who}: the {what} variance of these weights is beyond the float "
            "range. Give weights as fractions of the portfolio, loadings on "
            "their usual scale, and variances in return units."
        )
    variance = factor_variance + specific_variance
    volatility = math.sqrt(variance) if variance > 0 else 0.0
    factor_vols = np.sqrt(np.diag(F))
    by_factor = [
        {
            "factor": factor,
            "exposure": float(x[k]),
            "factor_volatility": float(factor_vols[k]),
            "correlation": _ratio(float(Fx[k]), float(factor_vols[k]) * volatility),
            "contribution": _ratio(float(x[k] * Fx[k]), volatility),
            "share": _ratio(float(x[k] * Fx[k]), variance),
        }
        for k, factor in enumerate(factors)
    ]
    position = {a: i for i, a in enumerate(assets)}
    by_asset = []
    for name in rows:
        i = position[name]
        total_i = float(BFx[i] + d[i] * v[i])
        by_asset.append(
            {
                "asset": name,
                "weight": float(v[i]),
                "marginal": _ratio(total_i, volatility),
                "contribution": _ratio(float(v[i] * total_i), volatility),
                "share": _ratio(float(v[i] * total_i), variance),
                "factor_contribution": _ratio(float(v[i] * BFx[i]), volatility),
                "specific_contribution": _ratio(
                    float(specific_by_asset[i]), volatility
                ),
            }
        )
    if volatility > 0:
        by_asset.sort(key=lambda r: r["contribution"], reverse=True)
    return {
        "volatility": volatility,
        "variance": variance,
        "factor_variance": factor_variance,
        "specific_variance": specific_variance,
        "factor_volatility": math.sqrt(max(factor_variance, 0.0)),
        "specific_volatility": math.sqrt(specific_variance),
        "factor_share": _ratio(factor_variance, variance),
        "specific_share": _ratio(specific_variance, variance),
        "factor_specific_covariance": _CROSS_TERM_IS_ZERO,
        "exposures": {f: float(x[k]) for k, f in enumerate(factors)},
        "by_factor": by_factor,
        "specific_contribution": _ratio(specific_variance, volatility),
        "by_asset": by_asset,
        "sum_of_contributions": (
            math.fsum(r["contribution"] for r in by_asset)
            if volatility > 0
            else float("nan")
        ),
    }


def decompose_factor_risk(
    weights: Mapping[str, float],
    factor_loadings: Any,
    factor_covariance: Any,
    specific_variance: Any,
    *,
    benchmark_weights: Optional[Mapping[str, float]] = None,
) -> Dict[str, Any]:
    """
    Total, factor and specific risk of a portfolio under Sigma = B F B' + D,
    and its tracking error against a benchmark, each split by factor and by
    asset with Euler contributions that sum to the volatility.

    Args:
        weights: asset -> weight. Every asset named needs loadings.
        factor_loadings: asset -> {factor: loading} (or an asset x factor
            frame). Every asset needs every factor of the covariance.
        factor_covariance: factor -> {factor: covariance} (or a square
            frame named by factor). Its units are the answer's.
        specific_variance: asset -> idiosyncratic return variance, one for
            every asset of the loadings and no other. Required.
        benchmark_weights: asset -> benchmark weight, for the active
            decomposition and the tracking error.

    Returns a dict: `volatility`, `tracking_error` (NaN without a
    benchmark), `benchmark_volatility`, `portfolio` and `active` (each:
    volatility, variance, factor and specific variance and volatility, their
    shares, the zero factor-specific covariance, exposures, `by_factor` with
    exposure x factor volatility x correlation, the specific contribution,
    `by_asset` with marginals and factor/specific contributions, and the sum
    of the contributions), `benchmark_exposures`, counts and `warnings`.
    A zero variance leaves its shares and contributions NaN (undefined).
    """
    who = "decompose_factor_risk"
    covariance, notes = _factor_covariance(factor_covariance, who)
    factors = [str(f) for f in covariance.columns]
    loadings = _loadings(factor_loadings, factors, who)
    assets = [str(a) for a in loadings.index]
    d = _specific(specific_variance, assets, who)
    w, named = _weight_vector(weights, "weights", assets, who)
    b = None
    rows = list(named)
    if benchmark_weights is not None:
        b, benchmark_named = _weight_vector(
            benchmark_weights, "benchmark_weights", assets, who
        )
        held = set(named)
        rows += [a for a in benchmark_named if a not in held]

    B = loadings.to_numpy(dtype=float)
    F = covariance.to_numpy(dtype=float)
    portfolio = _side(w, B, F, d, factors, assets, rows, who, "portfolio")
    active = None
    benchmark_volatility = float("nan")
    benchmark_exposures = None
    if b is not None:
        with np.errstate(over="ignore", invalid="ignore"):
            a = w - b
        if not np.all(np.isfinite(a)):
            raise ValidationError(
                f"{who}: weights minus benchmark_weights is beyond the float "
                "range. Give both as fractions of the portfolio."
            )
        active = _side(a, B, F, d, factors, assets, rows, who, "active")
        benchmark = _side(b, B, F, d, factors, assets, rows, who, "benchmark")
        benchmark_volatility = benchmark["volatility"]
        benchmark_exposures = benchmark["exposures"]

    warnings: List[str] = list(notes)
    if not portfolio["volatility"] > 0:
        warnings.append(
            "The portfolio has zero variance under this model -- no factor "
            "exposure and no specific variance on what it holds -- so there is "
            "no risk to apportion and its shares and contributions are null."
        )
    if active is not None and not active["volatility"] > 0:
        warnings.append(
            "The tracking error is zero: the portfolio has the benchmark's "
            "factor exposures and no active specific risk (the same weights, "
            "or active weights only on names with no specific variance and no "
            "net factor exposure), so there is no active risk to apportion."
        )
    hedges = [
        r["asset"]
        for r in portfolio["by_asset"]
        if r["weight"] != 0.0 and r["marginal"] < 0
    ]
    if hedges:
        warnings.append(
            f"{len(hedges)} position(s) have NEGATIVE marginal risk "
            f"({hedges[:_SHOWN]}): adding to them reduces the portfolio's "
            "volatility under this model, so they are hedges whether or not "
            "they were meant as one."
        )
    return {
        "n_assets": len(rows),
        "n_model_assets": len(assets),
        "n_factors": len(factors),
        "volatility": portfolio["volatility"],
        "tracking_error": (
            active["volatility"] if active is not None else float("nan")
        ),
        "benchmark_volatility": benchmark_volatility,
        "portfolio": portfolio,
        "active": active,
        "benchmark_exposures": benchmark_exposures,
        "warnings": warnings,
    }
