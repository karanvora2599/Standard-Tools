"""
Which candidate factors deserve a place in a risk model, judged out of
sample.

`estimate_factor_model` fits B, F and D for whatever factors it is handed;
nothing says which of them earn their place, and every number it returns is
in-sample, where adding a factor never makes the fit worse. The screen asks
four questions of each candidate, each the way it can fail in practice:

1. DOES IT EXPLAIN RETURNS IT WAS NOT FITTED TO? Walked forward: at each
   step the loadings of every asset on base + candidate (and on base alone)
   are estimated by least squares on the trailing `estimation_window` rows
   and applied to the next `step` rows' factor returns, contemporaneously --
   the explanatory model a risk model is, not a forecast. The out-of-sample
   incremental R-squared is

       (SSE_base - SSE_base+candidate) / SST,

   pooled over every asset and test row, SST the squared deviations of the
   returns from their estimation-window means. Pooled, so an asset counts
   by its variance, which is what a risk model is for. It is tested against
   CIRCULAR SHIFTS of the candidate -- the null `select_features` and
   `permutation_test_ic` draw from, `np.roll` by a uniform offset in
   [1, T-1] -- which keep the candidate's own autocorrelation and
   distribution and break only its link to the assets. A noise regressor
   makes the out-of-sample fit WORSE (it costs estimation error), so the
   null sits below zero and the candidate is compared with it rather than
   with zero: the p-value is (1 + #{shifted >= observed}) / (n_shifts + 1),
   never 0. The candidates' p-values are Benjamini-Hochberg adjusted
   together (`modeling.validation.comparison.bh_adjust`), so `fdr` bounds
   the expected share of false factors among the ones passed.
2. ARE ITS LOADINGS STABLE? A loading whose sign changes from window to
   window is not one the model can use. `sign_stability` is the share of
   windows in which each asset's loading keeps the sign of its full-sample
   loading, averaged over the assets weighted by |full-sample loading|, so
   an asset the factor does not move (whose loading is noise around zero
   and flips half the time) counts for little. `half_sample_correlation`
   is the uncentred correlation (the cosine) of the loadings fitted on the
   first and second halves of the sample: +1 when every loading keeps its
   size and sign, -1 when they all flip. Uncentred, because a factor every
   asset loads on alike has no cross-sectional spread for a centred
   correlation to read, and it would come out as noise.
3. IS IT SOMETHING ALREADY IN THE MODEL? The VIF of each candidate against
   the base factors and the other candidates, from the feature lab's
   `redundancy_report` (the inverse correlation matrix's diagonal, an exact
   dependency infinite and reported as None), at 10 or more redundant
   (`VIF_ACTION`), with the factor it correlates with most.
4. DOES IT MAKE THE RISK MODEL BETTER? The test a risk model exists to
   pass. At each step the global minimum-variance portfolio,
   w = S^-1 1 / 1'S^-1 1, is built from the factor model's covariance
   S = B F B' + D on the estimation window -- with base + candidate, with
   base alone, and from the Ledoit-Wolf shrunk sample covariance -- and held
   over the next step. Its realized volatility over all the steps is the
   score. The minimum-variance book depends on the covariance and nothing
   else, which makes it the cleanest out-of-sample test of one (Ledoit and
   Wolf 2003): a factor that "explains" returns but does not make that book
   less volatile has not earned a place in a risk model. Unconstrained, so
   the test reaches every direction the covariance gets wrong. One thing
   it cannot see: a factor every asset loads on ALIKE adds a multiple of
   11' to S, which leaves S^-1 1 pointing where it did (Sherman-Morrison),
   so the book -- fully invested, its exposure to that factor fixed at the
   common loading -- does not move. Such a factor fails this test however
   real it is; it is a risk the book cannot diversify, not one the model
   prices wrongly.

`selected` holds the candidates that pass all four: BH-adjusted p at most
`fdr`, stable loadings, VIF under 10 and a lower minimum-variance volatility
than the base model's. Every other candidate carries the reasons it failed.

WHAT IS IN-SAMPLE. The walk-forward fits, their errors and the
minimum-variance returns use only rows before each step (a row after t
changes nothing computed for a window ending at or before t). The
full-sample loadings stability compares against, the half-sample fits and
the VIFs describe the whole sample, and the p-values use the whole sample's
shifts. Screening k candidates on one sample is itself a selection: the
count is reported, and `selected_min_variance_volatility` -- the base model
with the selected factors, scored on the same steps that selected them -- is
labelled hindsight.

`T` rows, `N` assets: every least-squares fit is
`multi_factor.fit_least_squares` (rank-deficient designs refused by name),
the covariance is `estimate_factor_model`'s arithmetic on the window, and
all of it runs on one BLAS thread.
"""

from __future__ import annotations

import math
import zlib
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from standard_quant_tools._blas import single_threaded_blas
from standard_quant_tools.analysis.factor_model import _frame, _listed
from standard_quant_tools.analysis.multi_factor import fit_least_squares
from standard_quant_tools.constants import TRADING_DAYS_PER_YEAR
from standard_quant_tools.error import ValidationError

__all__ = [
    "MAX_NULL_CELLS",
    "STABLE_HALF_CORRELATION",
    "STABLE_SIGN_SHARE",
    "screen_factor_candidates",
]

_WHO = "screen_factor_candidates"

#: The null's size one call may reach: candidates x steps x (shifts + 1) x
#: (window + step) x assets, the cells its Frisch-Waugh-Lovell pass touches.
#: Measured at about 0.35 ns a cell (4 candidates, 48 steps, 201 offsets,
#: 273 rows, 50 assets: 0.18 s), so the ceiling is a null of under ten
#: seconds; past it the refusal names the knobs.
MAX_NULL_CELLS = 20_000_000_000

#: A candidate's loadings are stable when, weighted by |full-sample
#: loading|, they keep its sign in at least this share of windows...
STABLE_SIGN_SHARE = 0.75

#: ...and the two halves' loadings have at least this uncentred
#: correlation.
STABLE_HALF_CORRELATION = 0.5

#: Fewer walk-forward steps than this and the result says the scores rest
#: on few windows.
_FEW_STEPS = 10

#: Fewer estimation rows than this per coefficient and the result says the
#: window is short for the loadings it fits.
_ROWS_PER_COEFFICIENT = 10


def _whole(value: Any, name: str, low: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValidationError(f"{_WHO}: {name} must be a whole number, got {value!r}.")
    if int(value) < low:
        raise ValidationError(f"{_WHO}: {name}={value} is below {low}.")
    return int(value)


def _design(columns: Sequence[np.ndarray], n: int) -> np.ndarray:
    """The intercept and the given factor columns, side by side."""
    return np.column_stack([np.ones(n), *columns]) if columns else np.ones((n, 1))


def _window_fit(X: np.ndarray, Y: np.ndarray, names: List[str]):
    """Coefficients (p x N) of every asset on one window's design, and the
    per-asset residual variance SSR / (n - p)."""
    beta, residuals, _ssr = fit_least_squares(X, Y, names, _WHO)
    dof = X.shape[0] - X.shape[1]
    return beta, np.sum(residuals * residuals, axis=0) / dof


def factor_covariance(X: np.ndarray, Y: np.ndarray, names: List[str]) -> np.ndarray:
    """
    B F B' + D per period from one window: B the OLS loadings of each asset
    on the factor columns of `X` (its first column the intercept), F their
    sample covariance (n - 1), D the residual variances SSR / (n - p) --
    `estimate_factor_model`'s covariance on the same rows, before it is
    annualized. Call it under `single_threaded_blas()`.
    """
    beta, d = _window_fit(X, Y, names)
    k = X.shape[1] - 1
    if k == 0:
        return np.diag(d)
    B = beta[1:].T
    F = np.atleast_2d(np.cov(X[:, 1:], rowvar=False, ddof=1))
    S = (B @ F) @ B.T
    sigma = (S + S.T) / 2.0
    sigma[np.diag_indices(len(d))] += d
    return sigma


def _min_variance_weights(sigma: np.ndarray) -> Optional[np.ndarray]:
    """w = S^-1 1 / 1'S^-1 1, or None when S cannot be solved."""
    ones = np.ones(sigma.shape[0])
    try:
        x = np.linalg.solve(sigma, ones)
    except np.linalg.LinAlgError:
        return None
    total = float(x.sum())
    if not math.isfinite(total) or total == 0.0:
        return None
    return x / total


def _ledoit_wolf(window: np.ndarray, names: List[str]) -> np.ndarray:
    """The Ledoit-Wolf covariance `estimate_covariance` gives, per period."""
    from standard_quant_tools.portfolio.covariance import estimate_covariance

    out = estimate_covariance(
        pd.DataFrame(window, columns=names),
        method="ledoit_wolf",
        periods_per_year=1,
    )
    return np.array([[out["matrix"][r][c] for c in names] for r in names])


def _annualized_volatility(returns: np.ndarray, periods_per_year: int) -> float:
    if returns.size < 2 or not np.all(np.isfinite(returns)):
        return float("nan")
    return float(np.std(returns, ddof=1) * math.sqrt(periods_per_year))


def _shift_seed(random_seed: int, name: str) -> np.random.Generator:
    """One generator per candidate, from the seed and its name: which other
    candidates are screened beside it changes none of its draws."""
    return np.random.default_rng([int(random_seed), zlib.crc32(name.encode())])


def _constant_stretch(values: np.ndarray, window: int) -> bool:
    """Whether some `window` consecutive values, read circularly, are all
    equal: a stretch no regression on it can fit, which a shift can move
    into any estimation window."""
    wrapped = np.concatenate([values, values[: window - 1]])
    rolling = pd.Series(wrapped).rolling(window)
    spread = (rolling.max() - rolling.min()).to_numpy()[window - 1 :]
    return bool(np.any(spread == 0.0))


class _Walk:
    """The walk-forward steps over one panel: where each estimation window
    and each test block lie, and the per-step pieces every model shares."""

    def __init__(self, Y: np.ndarray, window: int, step: int) -> None:
        self.Y = Y
        self.window = window
        self.T = Y.shape[0]
        self.starts = list(range(window, self.T, step))
        self.blocks = [(s - window, s, min(s + step, self.T)) for s in self.starts]
        self.sst = np.array(
            [
                float(np.sum((Y[t0:t1] - Y[e0:t0].mean(axis=0)) ** 2))
                for e0, t0, t1 in self.blocks
            ]
        )

    def sse(self, X: np.ndarray, names: List[str]) -> np.ndarray:
        """Each step's out-of-sample squared error over its test block, the
        loadings fitted on its window."""
        out = np.empty(len(self.blocks))
        for i, (e0, t0, t1) in enumerate(self.blocks):
            beta, _resid, _ssr = fit_least_squares(X[e0:t0], self.Y[e0:t0], names, _WHO)
            error = self.Y[t0:t1] - X[t0:t1] @ beta
            out[i] = float(np.sum(error * error))
        return out

    def base_fits(self, X: np.ndarray, names: List[str]) -> List[tuple]:
        """Per window, the base model's residuals on the window and its
        errors on the test block: what every candidate's fit is built on."""
        fits = []
        for e0, t0, t1 in self.blocks:
            beta, resid, _ssr = fit_least_squares(X[e0:t0], self.Y[e0:t0], names, _WHO)
            fits.append((resid, self.Y[t0:t1] - X[t0:t1] @ beta))
        return fits

    def shifted_sse(
        self,
        X: np.ndarray,
        names: List[str],
        base: List[tuple],
        candidate: np.ndarray,
        offsets: np.ndarray,
    ) -> np.ndarray:
        """
        Each step's out-of-sample squared error of base + the candidate rolled
        by each of `offsets` (0 is the candidate itself): steps x offsets.

        By Frisch-Waugh-Lovell, not one regression per offset. On a window,
        the candidate's loadings in the joint fit are those of the base
        residuals Y~ on the candidate's own base residual c~,
        b = c~'Y~ / c~'c~, and the joint fit's error on the test block is
        the base model's error less u b', u the candidate's test values less
        their base fit -- the same least squares, solved for every offset
        at once. The observed candidate goes through the same arithmetic as
        its shifts, so the comparison that gives the p-value is like for
        like; it equals a direct fit of base + candidate to rounding.
        """
        T = self.T
        out = np.empty((len(self.blocks), len(offsets)))
        for i, (e0, t0, t1) in enumerate(self.blocks):
            resid, error = base[i]
            window = candidate[(np.arange(e0, t0)[:, None] - offsets[None, :]) % T]
            test = candidate[(np.arange(t0, t1)[:, None] - offsets[None, :]) % T]
            gamma, own, _ssr = fit_least_squares(X[e0:t0], window, names, _WHO)
            loadings = (own.T @ resid) / np.sum(own * own, axis=0)[:, None]
            u = test - X[t0:t1] @ gamma
            cross = np.sum(u * (error @ loadings.T), axis=0)
            spread = np.sum(u * u, axis=0) * np.sum(loadings * loadings, axis=1)
            out[i] = float(np.sum(error * error)) - 2.0 * cross + spread
        return out

    def loadings(self, X: np.ndarray, names: List[str], column: int) -> np.ndarray:
        """Each window's loading of every asset on X's `column` (steps x N)."""
        out = np.empty((len(self.blocks), self.Y.shape[1]))
        for i, (e0, t0, _t1) in enumerate(self.blocks):
            beta, _resid, _ssr = fit_least_squares(X[e0:t0], self.Y[e0:t0], names, _WHO)
            out[i] = beta[column]
        return out

    def min_variance_returns(self, covariance) -> np.ndarray:
        """The minimum-variance portfolio from `covariance(e0, t0)` on each
        window, held over its test block: the realized returns, NaN over a
        block whose covariance could not be solved."""
        pieces = []
        for e0, t0, t1 in self.blocks:
            w = _min_variance_weights(covariance(e0, t0))
            block = self.Y[t0:t1]
            pieces.append(block @ w if w is not None else np.full(len(block), np.nan))
        return np.concatenate(pieces)


def screen_factor_candidates(
    asset_returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
    *,
    base_factors: Sequence[str] = (),
    estimation_window: int = TRADING_DAYS_PER_YEAR,
    step: int = 21,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    n_shifts: int = 500,
    random_seed: int = 0,
    fdr: float = 0.10,
) -> Dict[str, Any]:
    """
    Screen every factor of `factor_returns` not in `base_factors` as a
    candidate for a risk model of `asset_returns`: out-of-sample incremental
    R-squared against circular shifts with Benjamini-Hochberg, loading
    stability, VIF, and the out-of-sample minimum-variance volatility of the
    risk model with and without it (see the module docstring).

    Args:
        asset_returns: one column per asset (at least two), one row per date.
        factor_returns: one column per factor on exactly the same dates; the
            base factors and the candidates.
        base_factors: the factors always in the model (a market factor);
            none leaves the intercept alone as the base.
        estimation_window: rows each window's loadings are fitted on.
        step: rows between refits, and each test block's length.
        periods_per_year: annualizes the volatilities (252 daily).
        n_shifts: circular shifts of each candidate for its p-value; the
            smallest p is 1 / (n_shifts + 1), and 500 lets a lone true
            factor among 50 candidates clear fdr 0.10 after BH.
        random_seed: seeds the shifts, per candidate by name.
        fdr: the Benjamini-Hochberg level a candidate's p must reach.

    Returns a dict: `assets`, `base_factors`, `candidates`, the walk's
    `n_observations`, `n_steps`, `n_test_rows`, `first_test_date`,
    `last_test_date`; `base_oos_r_squared`; per candidate under
    `candidates_screened` its `incremental_r_squared`, `p_value`,
    `p_value_bh`, `sign_stability`, `half_sample_correlation`, `vif`,
    `most_correlated_factor` and `most_correlated`,
    `min_variance_volatility`, `passed` and `failed` (the reasons), and
    `window_loadings` (steps x assets); `min_variance_volatility` for
    `base`, `ledoit_wolf`, `all_candidates` and `selected` (hindsight);
    `selected`; `steps` (per step: its first test date, SST and each
    model's SSE) and `portfolio_returns` (per test date, each model's
    minimum-variance return); and `warnings`.
    """
    from standard_quant_tools.modeling.analysis.feature_report import (
        VIF_ACTION,
        redundancy_report,
    )
    from standard_quant_tools.modeling.validation.comparison import bh_adjust

    window = _whole(estimation_window, "estimation_window", 2)
    stride = _whole(step, "step", 1)
    ppy = _whole(periods_per_year, "periods_per_year", 1)
    shifts = _whole(n_shifts, "n_shifts", 1)
    seed = _whole(random_seed, "random_seed", 0)
    if isinstance(fdr, bool) or not isinstance(fdr, (int, float)):
        raise ValidationError(f"{_WHO}: fdr must be a number, got {fdr!r}.")
    if not 0.0 < float(fdr) < 1.0:
        raise ValidationError(
            f"{_WHO}: fdr={fdr} is not a false discovery rate; it is strictly "
            "between 0 and 1 (0.10 is the default)."
        )
    assets_frame = _frame(asset_returns, "asset_returns", _WHO)
    factors_frame = _frame(factor_returns, "factor_returns", _WHO)
    assets = list(assets_frame.columns)
    factors = list(factors_frame.columns)
    both = [a for a in assets if a in set(factors)]
    if both:
        raise ValidationError(
            f"{_WHO}: {_listed(both)} named both as an asset and as a factor; "
            "an asset regressed on itself is an identity. Rename one side."
        )
    if not assets_frame.index.equals(factors_frame.index):
        raise ValidationError(
            f"{_WHO}: asset_returns ({len(assets_frame)} rows) and "
            f"factor_returns ({len(factors_frame)} rows) are not on the same "
            "dates. Align them on the dates they share first."
        )
    if len(assets) < 2:
        raise ValidationError(
            f"{_WHO}: {len(assets)} asset; a minimum-variance portfolio, the "
            "test a risk model is scored by, needs at least two."
        )
    base = [str(b) for b in base_factors]
    repeated = sorted({b for b in base if base.count(b) > 1})
    if repeated:
        raise ValidationError(f"{_WHO}: base_factors names {repeated} twice.")
    unknown = [b for b in base if b not in factors]
    if unknown:
        raise ValidationError(
            f"{_WHO}: base_factors {_listed(unknown)} are not among the "
            f"factors, which are {_listed(factors)}."
        )
    candidates = [f for f in factors if f not in set(base)]
    if not candidates:
        raise ValidationError(
            f"{_WHO}: every factor is a base factor, so there is no candidate "
            "to screen. Pass the candidates beside the base factors."
        )

    T = len(assets_frame)
    k_base = len(base)
    p_one = k_base + 2
    if window < p_one + 1:
        raise ValidationError(
            f"{_WHO}: estimation_window={window} rows for {p_one} coefficients "
            "(the intercept, the base factors and a candidate) leaves no "
            "residual variance; it needs at least "
            f"{p_one + 1}, and ten rows a coefficient to fit them well."
        )
    if T < window + 2 * stride:
        raise ValidationError(
            f"{_WHO}: {T} rows hold fewer than two steps of {stride} after the "
            f"first window of {window}. Give at least {window + 2 * stride} "
            "rows, or a shorter window or step."
        )
    if T // 2 < p_one + 1:
        raise ValidationError(
            f"{_WHO}: {T} rows leave halves of {T // 2}, too few to fit "
            f"{p_one} coefficients on each."
        )

    Y = assets_frame.to_numpy(dtype=float)
    Fb = [factors_frame[b].to_numpy(dtype=float) for b in base]
    C = {c: factors_frame[c].to_numpy(dtype=float) for c in candidates}
    flat = [c for c in candidates if _constant_stretch(C[c], window)]
    if flat:
        raise ValidationError(
            f"{_WHO}: {_listed(flat)} hold the same value for {window} rows or "
            "more (read circularly, as the shifts read them), so some window "
            "has nothing to fit a loading to. Leave such a factor out, or start "
            "it where it starts to vary."
        )
    walk = _Walk(Y, window, stride)
    n_steps = len(walk.blocks)
    cells = len(candidates) * n_steps * (shifts + 1) * (window + stride) * len(assets)
    if cells > MAX_NULL_CELLS:
        raise ValidationError(
            f"{_WHO}: {len(candidates)} candidates x {n_steps} steps x "
            f"{shifts + 1} offsets (the observed and {shifts} shifts) x "
            f"{window + stride} rows x {len(assets)} assets is {cells:,} cells, "
            f"more than the {MAX_NULL_CELLS:,} one call's null runs over. Lower "
            "n_shifts, raise step, or screen fewer candidates or assets at once."
        )

    base_names = ["intercept"] + base
    X_base = _design(Fb, T)
    test_index = assets_frame.index[walk.starts[0] :]
    warnings: List[str] = []
    screened: Dict[str, Dict[str, Any]] = {}

    with single_threaded_blas():
        sse_base = walk.sse(X_base, base_names)
        base_fits = walk.base_fits(X_base, base_names)
        returns = {
            "base": walk.min_variance_returns(
                lambda e0, t0: factor_covariance(X_base[e0:t0], Y[e0:t0], base_names)
            ),
            "ledoit_wolf": walk.min_variance_returns(
                lambda e0, t0: _ledoit_wolf(Y[e0:t0], assets)
            ),
        }
        sse: Dict[str, np.ndarray] = {}
        sst = float(walk.sst.sum())
        for c in candidates:
            names = base_names + [c]
            X = _design(Fb + [C[c]], T)
            # The full-sample fit first: a candidate the base factors span is
            # refused here, by name, before the null divides by its residual.
            full, _d = _window_fit(X, Y, names)
            # The observed candidate is offset 0; each shift an offset drawn
            # uniformly from [1, T-1], as np.roll would move it.
            offsets = np.concatenate(
                [[0], _shift_seed(seed, c).integers(1, T, size=shifts)]
            )
            per_step = walk.shifted_sse(X_base, base_names, base_fits, C[c], offsets)
            totals = per_step.sum(axis=0)
            sse[c] = per_step[:, 0]
            observed = float((sse_base.sum() - totals[0]) / sst)
            null = (sse_base.sum() - totals[1:]) / sst
            half = T // 2
            first, _d1 = _window_fit(X[:half], Y[:half], names)
            second, _d2 = _window_fit(X[half:], Y[half:], names)
            window_loadings = walk.loadings(X, names, -1)
            returns[c] = walk.min_variance_returns(
                lambda e0, t0, X=X, names=names: factor_covariance(
                    X[e0:t0], Y[e0:t0], names
                )
            )
            screened[c] = {
                "incremental_r_squared": observed,
                "p_value": float((1 + np.sum(null >= observed)) / (shifts + 1)),
                "null_mean": float(np.mean(null)),
                "sign_stability": _sign_stability(window_loadings, full[-1]),
                "half_sample_correlation": _cosine(first[-1], second[-1]),
                "window_loadings": pd.DataFrame(
                    window_loadings,
                    index=assets_frame.index[walk.starts],
                    columns=assets,
                ),
            }

        all_names = base_names + candidates
        p_all = len(all_names)
        if window > p_all:
            X_all = _design(Fb + [C[c] for c in candidates], T)
            try:
                returns["all_candidates"] = walk.min_variance_returns(
                    lambda e0, t0: factor_covariance(X_all[e0:t0], Y[e0:t0], all_names)
                )
            except ValidationError as exc:
                warnings.append(
                    "min_variance_volatility.all_candidates is null: the base "
                    "and every candidate together are linearly dependent in "
                    f"some window ({exc})."
                )
        else:
            warnings.append(
                f"min_variance_volatility.all_candidates is null: the base and "
                f"all {len(candidates)} candidates are {p_all} coefficients, "
                f"and an estimation window of {window} rows leaves them no "
                "residual variance."
            )

    redundancy = redundancy_report(factors_frame, factors)
    for c, value in zip(
        candidates, bh_adjust([screened[c]["p_value"] for c in candidates])
    ):
        screened[c]["p_value_bh"] = value
        screened[c].update(_most_correlated(c, factors, redundancy))
        screened[c]["min_variance_volatility"] = _annualized_volatility(returns[c], ppy)

    base_vol = _annualized_volatility(returns["base"], ppy)
    selected: List[str] = []
    for c in candidates:
        failed = _failures(screened[c], float(fdr), base_vol, VIF_ACTION)
        screened[c]["failed"] = failed
        screened[c]["passed"] = not failed
        if not failed:
            selected.append(c)

    with single_threaded_blas():
        if selected:
            chosen = base_names + selected
            X_chosen = _design(Fb + [C[c] for c in selected], T)
            returns["selected"] = walk.min_variance_returns(
                lambda e0, t0: factor_covariance(X_chosen[e0:t0], Y[e0:t0], chosen)
            )
        else:
            returns["selected"] = returns["base"]

    volatility = {
        name: _annualized_volatility(returns.get(name, np.array([])), ppy)
        for name in ("base", "ledoit_wolf", "all_candidates", "selected")
    }
    base_r2 = float(1.0 - sse_base.sum() / walk.sst.sum())
    warnings.extend(
        _notes(
            candidates,
            selected,
            n_steps,
            window,
            p_one,
            shifts,
            float(fdr),
        )
    )
    steps = pd.DataFrame(
        {"sst": walk.sst, "base": sse_base, **{c: sse[c] for c in candidates}},
        index=assets_frame.index[walk.starts],
    )
    portfolio = pd.DataFrame(returns, index=test_index)
    return {
        "assets": assets,
        "base_factors": base,
        "candidates": candidates,
        "n_observations": T,
        "n_steps": n_steps,
        "n_test_rows": int(len(test_index)),
        "first_test_date": test_index[0],
        "last_test_date": test_index[-1],
        "estimation_window": window,
        "step": stride,
        "base_oos_r_squared": base_r2,
        "candidates_screened": screened,
        "min_variance_volatility": volatility,
        "selected": selected,
        "steps": steps,
        "portfolio_returns": portfolio,
        "warnings": warnings,
    }


def _sign_stability(window_loadings: np.ndarray, full: np.ndarray) -> float:
    """The share of windows each asset's loading keeps its full-sample sign,
    averaged over the assets with weights |full-sample loading|."""
    weights = np.abs(full)
    total = float(weights.sum())
    if total == 0.0 or not math.isfinite(total):
        return float("nan")
    kept = (np.sign(window_loadings) == np.sign(full)[None, :]).mean(axis=0)
    return float(np.sum(weights * kept) / total)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """The uncentred correlation of two loading vectors."""
    scale = math.sqrt(float(np.sum(a * a)) * float(np.sum(b * b)))
    return float(np.sum(a * b)) / scale if scale > 0 else float("nan")


def _most_correlated(
    candidate: str, factors: List[str], redundancy: Dict[str, Any]
) -> Dict[str, Any]:
    """The candidate's VIF and the other factor it correlates with most."""
    vif = redundancy["vif"].get(candidate, 1.0) if len(factors) > 1 else 1.0
    row = redundancy["correlation"].get(candidate, {})
    others = [(f, float(row[f])) for f in factors if f != candidate and f in row]
    if not others:
        return {"vif": vif, "most_correlated_factor": None, "most_correlated": None}
    name, value = max(others, key=lambda item: (abs(item[1]), item[0]))
    return {"vif": vif, "most_correlated_factor": name, "most_correlated": value}


def _failures(
    row: Dict[str, Any], fdr: float, base_vol: float, vif_line: float
) -> List[str]:
    """Why a candidate is not selected, one reason per test it fails."""
    failed: List[str] = []
    if not row["p_value_bh"] <= fdr:
        failed.append(
            f"not significant: its BH-adjusted p-value is {row['p_value_bh']:.3g}, "
            f"above fdr {fdr:g} (out-of-sample incremental R-squared "
            f"{row['incremental_r_squared']:.4g}, its shifts' mean "
            f"{row['null_mean']:.4g})"
        )
    sign, half = row["sign_stability"], row["half_sample_correlation"]
    if not (sign >= STABLE_SIGN_SHARE and half >= STABLE_HALF_CORRELATION):
        failed.append(
            f"unstable loadings: they keep their full-sample sign in "
            f"{sign:.0%} of windows (stable is {STABLE_SIGN_SHARE:.0%}) and the "
            f"two halves' loadings correlate at {half:.2f} (stable is "
            f"{STABLE_HALF_CORRELATION:g})"
        )
    vif = row["vif"]
    if vif is None or not vif < vif_line:
        size = (
            "infinite (an exact combination of the others)"
            if vif is None
            else (f"{vif:.1f}")
        )
        failed.append(
            f"redundant: VIF {size}, at or above {vif_line:g}; it correlates "
            f"{row['most_correlated']:.2f} with {row['most_correlated_factor']!r}"
        )
    own = row["min_variance_volatility"]
    if not (math.isfinite(own) and math.isfinite(base_vol) and own < base_vol):
        failed.append(
            "does not lower the risk model's out-of-sample minimum-variance "
            f"volatility: {own:.4%} with it, {base_vol:.4%} without"
        )
    return failed


def _notes(
    candidates: List[str],
    selected: List[str],
    n_steps: int,
    window: int,
    p_one: int,
    shifts: int,
    fdr: float,
) -> List[str]:
    notes = [
        f"{len(candidates)} candidate(s) were screened on this sample, and "
        "choosing among them is itself a selection: the selected factors' "
        "scores are the best of that many tries. "
        "min_variance_volatility.selected scores them on the same steps that "
        "selected them, so it is hindsight; judge them on later data.",
        "In-sample: sign_stability compares each window with the full-sample "
        "loading, half_sample_correlation and the VIFs describe the whole "
        "sample. The incremental R-squared and the minimum-variance "
        "volatilities are out of sample.",
    ]
    floor = 1.0 / (shifts + 1)
    if len(candidates) * floor > fdr:
        notes.append(
            f"With n_shifts={shifts} the smallest p-value is {floor:.4f}, so a "
            f"lone true factor among {len(candidates)} candidates cannot reach "
            f"fdr {fdr:g} after Benjamini-Hochberg. Raise n_shifts to at least "
            f"{math.ceil(len(candidates) / fdr)}."
        )
    if n_steps < _FEW_STEPS:
        notes.append(
            f"{n_steps} walk-forward steps: the stability and the "
            "minimum-variance volatilities rest on few windows."
        )
    if window < _ROWS_PER_COEFFICIENT * p_one:
        notes.append(
            f"An estimation window of {window} rows is fewer than "
            f"{_ROWS_PER_COEFFICIENT} per coefficient ({p_one}): each "
            "window's loadings are loose, which costs every candidate "
            "out-of-sample fit."
        )
    if not selected:
        notes.append(
            "No candidate passed all four tests; the reasons are per candidate."
        )
    return notes
