"""
Liquidity measured from BARS, when there is no tick feed.

The existing microstructure tools measure spreads from trades and quotes and
refuse to run without them, which is the right refusal: a quoted spread is a
quoted spread and nothing computed from daily bars is one. But "no tick data"
is the normal case, and it does not mean the questions go away. Every
estimator here recovers a liquidity measure from OHLCV, and every one of them
names what it is a proxy FOR and how it fails.

THE ESTIMATORS AND WHAT THEY ASSUME:

- **Roll (1984)** infers the effective spread from the negative serial
  covariance of price changes -- bid-ask bounce makes consecutive returns
  mean-revert, and the size of that reversal is the spread. It assumes the
  efficient price is a random walk with no autocorrelation of its own, so
  any genuine momentum or reversal in the underlying contaminates it. On a
  trending stock the covariance turns positive and the estimator is
  undefined; that is reported rather than papered over with a zero.

  IT RETURNS A SPREAD WHEN THERE IS NONE, and this is the failure worth
  knowing about because nothing in the formula reveals it. Measured on a
  simulated random walk with a spread of EXACTLY ZERO and a 1% daily
  volatility, Roll's estimator returned 0.098 on a $100 stock -- a
  confident-looking 10 bps conjured entirely from sampling noise. Two
  things produce it: the lag-1 autocovariance has a standard error of
  var(dp)/sqrt(n), which swamps -(s/2)^2 whenever the spread is small
  relative to volatility, and the estimator only takes a square root when
  the covariance lands NEGATIVE, so the positive half of the noise is
  silently discarded and what survives is biased upward. `significant` and
  `smallest_detectable_spread` exist for exactly this: on the same
  simulation the floor came back at 0.289, correctly declaring the 0.098
  unmeasurable. Planted spreads of 0.50 and above cleared it and were
  recovered to within 8%.
- **Corwin-Schultz (2012)** uses the high-low range over one and two days.
  The insight is that the range contains both volatility and the spread, but
  volatility scales with time while the spread does not -- so two horizons
  identify them separately. It needs true intraday highs and lows, and it
  degrades on a stock that gaps overnight.
- **Amihud (2002)** is |return| per dollar traded: how far the price moves
  to absorb a dollar. It is not a spread and is not in any unit anyone can
  interpret directly -- it is only meaningful RELATIVE to the same stock's
  history or a peer's, which is why the result reports a percentile rather
  than encouraging the raw number to be read.
- **Kyle's lambda** is the regression of price change on signed order flow.
  The honest version needs signed trades; from bars, the sign comes from the
  tick rule, which is right about 85% of the time on liquid names and worse
  on illiquid ones -- exactly where the measure matters most.

WHAT NONE OF THEM DO: measure the spread you will actually pay. They measure
a historical average round-trip cost under a model. The spread at the moment
you send an order depends on the book at that moment, and no daily bar
contains it.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from standard_quant_tools.error import ValidationError

logger = logging.getLogger(__name__)

#: Below this many observations none of these estimators mean anything. Roll
#: needs a covariance, Corwin-Schultz needs overlapping two-day windows, and
#: Amihud needs enough days for a percentile to exist.
MIN_OBSERVATIONS = 30

#: Trading days per year, for annualizing anything that needs it.


def _require_columns(
    frame: pd.DataFrame, needed: Sequence[str], who: str
) -> pd.DataFrame:
    frame = pd.DataFrame(frame)
    lower = {str(c).lower(): c for c in frame.columns}
    missing = [c for c in needed if c not in lower]
    if missing:
        raise ValidationError(
            f"{who}: missing required column(s) {missing}. Got "
            f"{sorted(str(c) for c in frame.columns)}."
        )
    out = frame[[lower[c] for c in needed]].copy()
    out.columns = list(needed)
    return out.astype(float)


def _enough(n: int, who: str, minimum: int = MIN_OBSERVATIONS) -> None:
    if n < minimum:
        raise ValidationError(
            f"{who}: {n} usable observations, and this estimator needs at "
            f"least {minimum}. Below that the estimate's standard error is "
            "larger than the quantity being estimated."
        )


# ── spread from the covariance of price changes ─────────────────────────


def roll_spread(
    prices: pd.Series,
    *,
    window: Optional[int] = None,
) -> Dict[str, Any]:
    """
    The effective spread implied by bid-ask bounce, after Roll (1984).

    THE IDEA: if trades arrive randomly at the bid and the ask, consecutive
    price changes are negatively correlated purely from bouncing between the
    two. Under Roll's assumptions that covariance is -(s/2)^2, so the spread
    is 2*sqrt(-cov). No quotes required -- the spread is recovered from the
    footprint it leaves in the trade prices.

    WHEN IT BREAKS, and it breaks often. The derivation assumes the
    efficient price is a random walk with no autocorrelation of its own. Real
    return series have plenty: momentum makes the covariance less negative
    (understating the spread), and a strong trend pushes it POSITIVE, at
    which point the square root is undefined.

    That last case is the important one. The literature's usual fix is to
    set the spread to zero when the covariance is positive, which produces a
    tidy series with a systematic downward bias -- and the zeros cluster in
    exactly the trending periods where liquidity is most interesting. This
    returns `None` and counts the undefined windows instead, because "we
    could not measure it" and "it was zero" are different facts and only one
    of them is true.
    """
    values = pd.Series(prices).astype(float).dropna()
    _enough(len(values), "roll_spread")

    changes = values.diff().dropna()
    # The standard error of a lag-1 autocovariance, under the null that the
    # series is white noise, is var(dp)/sqrt(n). This is the number that
    # decides whether the estimate means anything -- see `significant` below.
    change_variance = float(changes.var(ddof=1))
    covariance_se = change_variance / math.sqrt(max(len(changes), 1))

    if window is None:
        covariance = float(np.cov(changes.iloc[1:], changes.iloc[:-1])[0, 1])
        defined = covariance < 0
        spread = 2.0 * math.sqrt(-covariance) if defined else None
        mean_price = float(values.mean())
        rolling_summary = None
        undefined_fraction = 0.0 if defined else 1.0
    else:
        window = int(window)
        if window < 10:
            raise ValidationError(
                f"roll_spread: window={window} is too short for a covariance."
            )
        estimates: List[Optional[float]] = []
        array = changes.to_numpy()
        for end in range(window, len(array) + 1):
            chunk = array[end - window : end]
            cov = float(np.cov(chunk[1:], chunk[:-1])[0, 1])
            estimates.append(2.0 * math.sqrt(-cov) if cov < 0 else None)
        usable = [e for e in estimates if e is not None]
        undefined_fraction = 1.0 - len(usable) / len(estimates) if estimates else 1.0
        spread = float(np.median(usable)) if usable else None
        mean_price = float(values.mean())
        rolling_summary = {
            "n_windows": len(estimates),
            "n_undefined": len(estimates) - len(usable),
            "median_spread": spread,
            "p25_spread": float(np.percentile(usable, 25)) if usable else None,
            "p75_spread": float(np.percentile(usable, 75)) if usable else None,
        }
        covariance = float("nan")
        covariance_se = covariance_se * math.sqrt(len(changes) / window)

    # The smallest spread this sample could distinguish from zero. Below
    # this, ANY estimate the formula returns is sampling noise that happened
    # to land on the negative side.
    detectable = 2.0 * math.sqrt(2.0 * covariance_se)
    significant = (
        bool(covariance < -2.0 * covariance_se) if math.isfinite(covariance) else None
    )

    warnings: List[str] = []
    if spread is not None and significant is False:
        warnings.append(
            f"THIS ESTIMATE IS NOT DISTINGUISHABLE FROM ZERO. The serial "
            f"covariance is {covariance:.3g} against a standard error of "
            f"{covariance_se:.3g}, so the {spread:.4f} reported is what "
            "sampling noise produces on a series with no spread at all. "
            "Roll's estimator is also biased UPWARD here, because a "
            "spread is only computed when the covariance lands negative -- "
            "the positive half of the noise is discarded. Measured on a "
            "pure random walk with a zero spread, it returns roughly "
            f"{detectable:.3f} on a sample this size."
        )
    if spread is not None and significant:
        warnings.append(
            f"The smallest spread this sample could distinguish from zero "
            f"is about {detectable:.4f}. Estimates near that floor are "
            "noise; this one is above it."
        )
    if spread is None:
        warnings.append(
            "The serial covariance of price changes is POSITIVE, so Roll's "
            "estimator is undefined here -- there is momentum in this series "
            "rather than bid-ask bounce. Returning None rather than zero: "
            "'we could not measure it' and 'the spread was zero' are "
            "different facts, and the usual convention of substituting zero "
            "biases every downstream average downward."
        )
    if undefined_fraction > 0.25:
        warnings.append(
            f"{undefined_fraction:.0%} of windows were undefined (positive "
            "covariance). The median over the rest is conditioned on the "
            "periods where the estimator happened to work, which are the "
            "less trending ones -- so it understates the average spread."
        )
    warnings.append(
        "Roll's spread is an EFFECTIVE spread under a model, not a quoted "
        "one. It assumes trades arrive randomly at bid and ask and that the "
        "efficient price has no autocorrelation of its own; real momentum "
        "biases it downward."
    )

    return {
        "n_observations": int(len(values)),
        "serial_covariance": covariance if math.isfinite(covariance) else None,
        "covariance_standard_error": float(covariance_se),
        "significant": significant,
        "smallest_detectable_spread": float(detectable),
        "spread_estimate": spread,
        "spread_bps": (
            float(spread / mean_price * 1e4)
            if spread is not None and mean_price > 0
            else None
        ),
        "half_spread_bps": (
            float(spread / mean_price * 5e3)
            if spread is not None and mean_price > 0
            else None
        ),
        "mean_price": mean_price,
        "undefined_fraction": float(undefined_fraction),
        "rolling": rolling_summary,
        "warnings": warnings,
    }


def corwin_schultz_spread(ohlc: pd.DataFrame) -> Dict[str, Any]:
    """
    The spread implied by the high-low range, after Corwin and Schultz (2012).

    THE IDEA is genuinely clever: a day's high-low range contains both the
    stock's volatility and its spread. Volatility scales with the square root
    of time, and the spread does not -- it is paid once whatever the horizon.
    So comparing a one-day range against a two-day range identifies them
    separately, with no quote data at all.

    IT PRODUCES NEGATIVE ESTIMATES, routinely, and this is the thing to know
    before using it. On roughly 10-30% of days the algebra yields a negative
    spread, which is a sampling artefact rather than a measurement. Corwin
    and Schultz's own recommendation is to set those to zero before
    averaging. That is done here AND reported: `negative_fraction` is
    returned, and above about a third the whole estimate should be treated as
    noise rather than as a small spread.

    OVERNIGHT GAPS BREAK IT. The derivation assumes the price is continuous
    between the two days. A stock that gaps on news has a two-day range
    inflated by the gap rather than by the spread, which biases the estimate
    UP. The adjustment for that is applied here, but a name that gaps
    routinely is outside what the estimator was built for.
    """
    frame = _require_columns(ohlc, ["high", "low"], "corwin_schultz_spread")
    frame = frame.dropna()
    _enough(len(frame), "corwin_schultz_spread")
    if (frame["low"] <= 0).any() or (frame["high"] < frame["low"]).any():
        raise ValidationError(
            "corwin_schultz_spread: found a non-positive low or a high below "
            "its low. The estimator takes logs of the ratio, so this is not "
            "recoverable data noise."
        )

    high = frame["high"].to_numpy()
    low = frame["low"].to_numpy()

    # Two-day high and low, adjusted for overnight gaps: if the whole of day
    # two sits above day one, the gap is not spread and is removed.
    high2 = np.maximum(high[1:], high[:-1])
    low2 = np.minimum(low[1:], low[:-1])

    beta = np.log(high[1:] / low[1:]) ** 2 + np.log(high[:-1] / low[:-1]) ** 2
    gamma = np.log(high2 / low2) ** 2

    root2 = math.sqrt(2.0)
    denominator = 3.0 - 2.0 * root2
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / denominator - np.sqrt(
        gamma / denominator
    )
    spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))

    negative = spread < 0
    negative_fraction = float(negative.mean())
    # Corwin-Schultz's own recommendation. Applied, and reported.
    floored = np.where(negative, 0.0, spread)

    warnings: List[str] = []
    if negative_fraction > 0.33:
        warnings.append(
            f"{negative_fraction:.0%} of daily estimates came out NEGATIVE "
            "and were floored at zero, as Corwin-Schultz recommend. Above "
            "about a third, treat the average as noise rather than as a "
            "small spread -- the flooring turns a symmetric error into a "
            "one-sided bias, and at this rate the bias dominates."
        )
    elif negative_fraction > 0:
        warnings.append(
            f"{negative_fraction:.0%} of daily estimates were negative and "
            "floored at zero. That is normal for this estimator (10-30% is "
            "typical) and it biases the mean upward slightly."
        )
    warnings.append(
        "Assumes the price is continuous between the two days. A name that "
        "gaps on news has its two-day range inflated by the gap rather than "
        "by the spread, biasing the estimate up. The standard gap "
        "adjustment is applied, but a routinely-gapping name is outside "
        "what this was built for."
    )

    return {
        "n_observations": int(len(frame)),
        "n_estimates": int(spread.size),
        "spread_estimate": float(floored.mean()),
        "spread_bps": float(floored.mean() * 1e4),
        "median_spread_bps": float(np.median(floored) * 1e4),
        "negative_fraction": negative_fraction,
        "raw_mean_bps": float(spread.mean() * 1e4),
        "warnings": warnings,
    }


# ── how far the price moves per dollar ──────────────────────────────────


def amihud_illiquidity(
    ohlcv: pd.DataFrame,
    *,
    window: int = 21,
) -> Dict[str, Any]:
    """
    Amihud's illiquidity ratio: how far the price moves per dollar traded.

    ILLIQ = mean(|return| / dollar volume). A stock whose price moves 2% on
    $1m of volume is less liquid than one that moves 0.5% on the same, and
    this ratio says so. It is the most widely used liquidity proxy in the
    academic literature, largely because it needs nothing but daily bars.

    THE RAW NUMBER IS UNINTERPRETABLE and that is the main trap. Its units
    are return per dollar, so it scales inversely with the stock's dollar
    volume -- a large-cap's ILLIQ is a thousand times smaller than a
    microcap's, and neither number means anything on its own. It is only
    usable as a RELATIVE measure: this stock against its own history, or
    against a peer group measured the same way over the same window. The
    result therefore leads with the percentile of the current reading within
    its own history, and the raw value is returned second.

    IT IS NOT A SPREAD. It conflates the spread, the depth of the book, and
    the information content of trades, and it cannot separate them. A stock
    that moves a lot per dollar because it is genuinely volatile scores as
    illiquid here even if its book is deep.
    """
    frame = _require_columns(ohlcv, ["close", "volume"], "amihud_illiquidity")
    frame = frame.dropna()
    frame = frame[(frame["close"] > 0) & (frame["volume"] > 0)]
    _enough(len(frame), "amihud_illiquidity")

    returns = frame["close"].pct_change(fill_method=None).abs()
    dollar_volume = frame["close"] * frame["volume"]
    ratio = (returns / dollar_volume).dropna()
    # Scaled by 1e6 so the numbers are readable; the scaling is stated rather
    # than silently applied, because published values use several conventions.
    scaled = ratio * 1e6

    window = max(2, int(window))
    rolling = scaled.rolling(window).mean().dropna()
    current = float(rolling.iloc[-1]) if len(rolling) else float(scaled.mean())
    percentile = float((rolling < current).mean() * 100.0) if len(rolling) > 1 else None

    trend = None
    if len(rolling) >= 2 * window:
        first = float(rolling.iloc[: len(rolling) // 2].mean())
        second = float(rolling.iloc[len(rolling) // 2 :].mean())
        if first > 0:
            trend = float((second / first - 1.0) * 100.0)

    warnings = [
        "The RAW value is not interpretable. Its units are return per "
        "dollar, so it scales inversely with dollar volume -- a large cap's "
        "reading is orders of magnitude below a microcap's and neither "
        "number means anything alone. Read the percentile against this "
        "name's own history, or compare peers measured identically over the "
        "same window.",
        "Amihud is NOT a spread. It conflates spread, book depth and the "
        "information content of trades and cannot separate them: a "
        "genuinely volatile stock scores as illiquid even with a deep book.",
        "Values are scaled by 1e6 for readability. Published figures use "
        "several scaling conventions, so do not compare these against a "
        "paper's numbers without checking its scaling.",
    ]
    if percentile is not None and percentile > 90:
        warnings.append(
            f"The current reading sits at the {percentile:.0f}th percentile "
            "of this name's own history -- it is currently far less liquid "
            "than usual. Sizing off a full-sample average would understate "
            "today's cost."
        )

    return {
        "n_observations": int(len(ratio)),
        "window": window,
        "current_illiquidity": current,
        "current_percentile": percentile,
        "mean_illiquidity": float(scaled.mean()),
        "median_illiquidity": float(scaled.median()),
        "trend_pct": trend,
        "scaling": "1e6",
        "mean_dollar_volume": float(dollar_volume.mean()),
        "warnings": warnings,
    }


def kyle_lambda(
    ohlcv: pd.DataFrame,
    *,
    window: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Kyle's lambda: the price impact of a unit of signed order flow.

    THE REGRESSION is price change on signed volume. Lambda is the slope --
    the depth of the market, in price per share of imbalance. It is the one
    number here with a direct trading interpretation: multiply it by the size
    you intend to trade and you have an estimate of the impact you will
    cause.

    THE SIGN COMES FROM THE TICK RULE and that is the weak link. Kyle's model
    is about SIGNED order flow, meaning buyer- versus seller-initiated
    volume, which requires trades matched against quotes. From bars, the sign
    of the day's return stands in for it. On a liquid name the tick rule
    agrees with the true classification about 85% of the time; on an illiquid
    one it is materially worse -- and illiquid names are where lambda
    matters. The misclassification attenuates the slope toward zero, so this
    UNDERSTATES impact, and it understates it most where impact is largest.

    R-SQUARED IS THE NUMBER TO CHECK. A lambda from a regression that
    explains 2% of the variance is a number with a standard error larger
    than itself. It is returned next to the estimate rather than buried.
    """
    frame = _require_columns(ohlcv, ["close", "volume"], "kyle_lambda")
    frame = frame.dropna()
    frame = frame[(frame["close"] > 0) & (frame["volume"] > 0)]
    _enough(len(frame), "kyle_lambda")

    price_change = frame["close"].diff()
    # The tick rule: the day's direction signs the day's volume.
    signed_volume = np.sign(price_change) * frame["volume"]
    data = pd.DataFrame({"dp": price_change, "signed": signed_volume}).dropna()
    data = data[data["signed"] != 0]
    _enough(len(data), "kyle_lambda", minimum=20)

    def _fit(chunk: pd.DataFrame):
        x = chunk["signed"].to_numpy()
        y = chunk["dp"].to_numpy()
        design = np.column_stack([np.ones_like(x), x])
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
        fitted = design @ coefficients
        total = float(((y - y.mean()) ** 2).sum())
        r2 = float(1.0 - ((y - fitted) ** 2).sum() / total) if total > 0 else 0.0
        return float(coefficients[1]), r2

    lam, r_squared = _fit(data)

    rolling = None
    if window:
        window = int(window)
        values = []
        for end in range(window, len(data) + 1):
            values.append(_fit(data.iloc[end - window : end])[0])
        if values:
            rolling = {
                "n_windows": len(values),
                "median_lambda": float(np.median(values)),
                "p25": float(np.percentile(values, 25)),
                "p75": float(np.percentile(values, 75)),
                "latest": float(values[-1]),
            }

    mean_price = float(frame["close"].mean())
    mean_volume = float(frame["volume"].mean())
    # What lambda says a 1%-of-ADV order costs, which is the interpretable form.
    impact_1pct = lam * 0.01 * mean_volume
    warnings: List[str] = []
    if r_squared < 0.05:
        warnings.append(
            f"R-squared is {r_squared:.3f}: signed volume explains almost "
            "none of the price variation, so this lambda has a standard "
            "error larger than itself. Do not size an order off it."
        )
    if lam <= 0:
        warnings.append(
            "Lambda came out non-positive, which is not economically "
            "meaningful -- buying should push the price up. It usually "
            "means the tick-rule signing is failing on this name, which "
            "happens on illiquid or heavily-crossed names."
        )
    warnings.append(
        "Order flow is signed by the TICK RULE, not by matching trades "
        "against quotes. That is right about 85% of the time on liquid "
        "names and worse on illiquid ones; misclassification attenuates the "
        "slope toward zero, so this understates impact -- and understates "
        "it most exactly where impact is largest."
    )

    return {
        "n_observations": int(len(data)),
        "kyle_lambda": lam,
        "r_squared": r_squared,
        "impact_of_1pct_adv": float(impact_1pct),
        "impact_of_1pct_adv_bps": (
            float(impact_1pct / mean_price * 1e4) if mean_price > 0 else None
        ),
        "mean_price": mean_price,
        "mean_volume": mean_volume,
        "rolling": rolling,
        "warnings": warnings,
    }


# ── flow ────────────────────────────────────────────────────────────────


def order_flow_imbalance(
    ohlcv: pd.DataFrame,
    *,
    window: int = 5,
) -> Dict[str, Any]:
    """
    Signed volume imbalance from bars, and whether it predicts the next
    return.

    The imbalance is (buy volume - sell volume) / total volume, with the
    split taken from the tick rule. On tick data this is one of the more
    reliable short-horizon predictors; from daily bars it is a much weaker
    thing, and the honest way to present it is with its own predictive test
    attached rather than as a signal to be trusted.

    THE AUTOCORRELATION TEST IS THE POINT. A genuine imbalance series is
    persistent -- informed flow arrives in pieces over days. A tick-rule
    imbalance built from daily returns inherits whatever autocorrelation the
    RETURNS have, which is roughly none, so a persistence near zero is the
    expected result and means the measure carries no information at this
    frequency. It is reported so the caller can see that rather than
    assuming otherwise.
    """
    frame = _require_columns(ohlcv, ["close", "volume"], "order_flow_imbalance")
    frame = frame.dropna()
    frame = frame[frame["volume"] > 0]
    _enough(len(frame), "order_flow_imbalance")

    returns = frame["close"].pct_change(fill_method=None)
    direction = np.sign(returns)
    signed = direction * frame["volume"]
    window = max(2, int(window))

    rolling_signed = signed.rolling(window).sum()
    rolling_total = frame["volume"].rolling(window).sum()
    imbalance = (rolling_signed / rolling_total).dropna()
    if imbalance.empty:
        raise ValidationError(
            f"order_flow_imbalance: window={window} left no complete windows."
        )

    # Does today's imbalance say anything about tomorrow's return?
    forward = returns.shift(-1).reindex(imbalance.index)
    pair = pd.DataFrame({"imb": imbalance, "fwd": forward}).dropna()
    predictive = float(pair["imb"].corr(pair["fwd"])) if len(pair) > 10 else None

    # PERSISTENCE IS MEASURED ON NON-OVERLAPPING WINDOWS. The rolling series
    # at window=5 shares four of its five observations with the previous
    # point, so its lag-1 autocorrelation is about 0.8 whatever the data
    # does -- measured at +0.76 on pure noise. That number describes the
    # window, not the flow. Stepping by `window` removes the overlap and
    # leaves an autocorrelation that means something.
    non_overlapping = imbalance.iloc[::window]
    persistence = (
        float(non_overlapping.autocorr(lag=1)) if len(non_overlapping) > 10 else None
    )
    overlapping_persistence = (
        float(imbalance.autocorr(lag=1)) if len(imbalance) > 10 else None
    )

    warnings = [
        "Buy/sell volume is split by the TICK RULE on daily bars, which is "
        "a much weaker signing than trade-versus-quote matching. Treat this "
        "as a coarse directional summary, not as order flow.",
    ]
    warnings.append(
        "`persistence` is measured on NON-OVERLAPPING windows. The rolling "
        "series shares window-1 of its observations with the previous "
        "point, so its raw autocorrelation is roughly 1 - 1/window "
        "whatever the data does -- +0.76 on pure noise at window=5. "
        "`overlapping_persistence` is that artefact, returned only so the "
        "difference is visible."
    )
    if persistence is not None and abs(persistence) < 0.1:
        warnings.append(
            f"Imbalance persistence is {persistence:.3f} -- essentially "
            "none. Real informed flow arrives in pieces and persists across "
            "days; an imbalance built from daily returns inherits the "
            "returns' own autocorrelation, which is near zero. That is the "
            "expected result here and it means this carries no information "
            "at this frequency."
        )
    if predictive is not None and abs(predictive) > 0.15:
        warnings.append(
            f"Imbalance correlates {predictive:+.3f} with the NEXT day's "
            "return. Before trading it, check it is not the same "
            "short-horizon reversal that a signed-volume measure picks up "
            "mechanically from the bid-ask bounce."
        )

    return {
        "n_observations": int(len(imbalance)),
        "n_non_overlapping": int(len(non_overlapping)),
        "window": window,
        "current_imbalance": float(imbalance.iloc[-1]),
        "mean_imbalance": float(imbalance.mean()),
        "std_imbalance": float(imbalance.std(ddof=1)) if len(imbalance) > 1 else 0.0,
        "persistence": persistence,
        "overlapping_persistence": overlapping_persistence,
        "next_day_correlation": predictive,
        # VOLUME, not bars. This was `(direction > 0).mean()` -- the
        # fraction of up DAYS, which is a different quantity under the same
        # field name that `analysis/microstructure.py` computes correctly
        # size-weighted. Measured on the same synthetic tape: 0.495050 here
        # against 0.000979 there, a factor of 506, same package.
        "buy_volume_fraction": (
            float(frame["volume"][direction > 0].sum() / frame["volume"].sum())
            if float(frame["volume"].sum()) > 0
            else float("nan")
        ),
        "warnings": warnings,
    }


def estimate_vpin(
    ohlcv: pd.DataFrame,
    *,
    n_buckets: int = 50,
    window: int = 50,
) -> Dict[str, Any]:
    """
    VPIN -- volume-synchronized probability of informed trading, after
    Easley, Lopez de Prado and O'Hara (2012).

    THE IDEA is to stop measuring in clock time. Information arrives with
    VOLUME, not with the clock, so the series is cut into equal-volume
    buckets rather than equal-time bars. Within each bucket the order flow
    is split into buy and sell, and VPIN is the average absolute imbalance
    across a rolling window of buckets. A high reading means flow is
    persistently one-sided, which is what informed trading looks like from
    outside.

    IT IS CONTESTED, and a tool that presents it as settled is misleading.
    The original paper's claim that VPIN spiked before the 2010 flash crash
    was challenged (Andersen and Bondarenko, 2014) on the grounds that the
    metric is largely a transformation of volatility, and that the flash
    crash result depends on the sample construction. What is not in dispute
    is that it measures one-sidedness of flow; whether that is "informed
    trading" is a model assumption.

    FROM BARS IT IS A SHADOW OF ITSELF. VPIN was designed for trade-level
    data, where volume buckets contain hundreds of trades and the bulk
    classification has something to work with. Built from daily bars each
    bucket is a handful of days and the classification is the tick rule.
    What is returned is a defensible time series of flow one-sidedness; it
    is not the VPIN of the paper, and it is labelled accordingly.
    """
    frame = _require_columns(ohlcv, ["close", "volume"], "estimate_vpin")
    frame = frame.dropna()
    frame = frame[frame["volume"] > 0]
    _enough(len(frame), "estimate_vpin")

    n_buckets = max(5, int(n_buckets))
    returns = frame["close"].pct_change(fill_method=None).fillna(0.0)
    volume = frame["volume"].to_numpy()
    total_volume = float(volume.sum())
    bucket_size = total_volume / n_buckets

    # Walk the bars, filling equal-volume buckets and splitting each bar's
    # volume by the sign of its return.
    buys: List[float] = []
    sells: List[float] = []
    current_buy = current_sell = filled = 0.0
    signs = np.sign(returns.to_numpy())
    for v, s in zip(volume, signs):
        remaining = float(v)
        while remaining > 0:
            room = bucket_size - filled
            take = min(remaining, room)
            if s >= 0:
                current_buy += take
            else:
                current_sell += take
            filled += take
            remaining -= take
            if filled >= bucket_size - 1e-9:
                buys.append(current_buy)
                sells.append(current_sell)
                current_buy = current_sell = filled = 0.0
    if filled > 0:
        buys.append(current_buy)
        sells.append(current_sell)

    buys_a = np.asarray(buys)
    sells_a = np.asarray(sells)
    totals = buys_a + sells_a
    valid = totals > 0
    imbalance = np.zeros_like(totals)
    imbalance[valid] = np.abs(buys_a[valid] - sells_a[valid]) / totals[valid]

    window = max(2, min(int(window), imbalance.size))
    series = pd.Series(imbalance).rolling(window).mean().dropna()
    if series.empty:
        raise ValidationError(
            f"estimate_vpin: window={window} over {imbalance.size} buckets "
            "left nothing to average."
        )
    current = float(series.iloc[-1])
    percentile = float((series < current).mean() * 100.0) if len(series) > 1 else None

    warnings = [
        "This is VPIN computed from DAILY BARS with tick-rule signing. The "
        "original is a trade-level measure where each volume bucket holds "
        "hundreds of trades and bulk classification has something to work "
        "with. What this returns is a defensible series of flow "
        "one-sidedness -- it is not the VPIN of the paper.",
        "VPIN is CONTESTED. Andersen and Bondarenko (2014) argue it is "
        "largely a transformation of volatility and that the flash-crash "
        "result depends on sample construction. It measures one-sidedness "
        "of flow; calling that 'informed trading' is a model assumption, "
        "not a measurement.",
    ]
    if percentile is not None and percentile > 90:
        warnings.append(
            f"The current reading is at the {percentile:.0f}th percentile of "
            "its own history. Given the caveats above, read that as 'flow "
            "has been unusually one-sided', which is also what a trending "
            "market looks like."
        )

    return {
        "n_buckets": int(imbalance.size),
        "bucket_volume": float(bucket_size),
        "window": window,
        "current_vpin": current,
        "current_percentile": percentile,
        "mean_vpin": float(series.mean()),
        "max_vpin": float(series.max()),
        "warnings": warnings,
    }


# ── the shape of the trading day ────────────────────────────────────────


def intraday_volume_profile(
    bars: pd.DataFrame,
    *,
    n_buckets: int = 13,
) -> Dict[str, Any]:
    """
    How volume distributes across the trading day, and what that implies for
    a participation schedule.

    THE U-SHAPE IS THE FACT EVERY EXECUTION SCHEDULE IS BUILT ON. Volume
    concentrates at the open and the close, with a midday trough that is
    routinely a third of the opening bucket. A VWAP algorithm that spread an
    order evenly across the clock would over-participate at lunch -- paying
    impact into a thin book -- and under-participate at the close, missing
    the cheapest liquidity of the day.

    THE CLOSE HAS BEEN GROWING and a profile fitted to old data will be
    wrong about it. Closing auctions have taken a rising share of US equity
    volume for a decade, driven by index fund flows. A profile estimated
    over several years averages across that trend; the result reports the
    close's share separately so it can be sanity-checked against the recent
    subsample.

    Needs INTRADAY bars with a datetime index. Daily bars have no intraday
    shape to describe and are refused rather than aggregated into a
    meaningless single bucket.
    """
    frame = pd.DataFrame(bars)
    lower = {str(c).lower(): c for c in frame.columns}
    if "volume" not in lower:
        raise ValidationError(
            f"intraday_volume_profile: no 'volume' column. Got "
            f"{sorted(str(c) for c in frame.columns)}."
        )
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValidationError(
            "intraday_volume_profile: needs a DatetimeIndex to know when in "
            "the day each bar sits. A frame indexed by position has no "
            "intraday shape to describe."
        )
    volume = pd.Series(frame[lower["volume"]].astype(float).values, index=frame.index)
    volume = volume.dropna()
    _enough(len(volume), "intraday_volume_profile")

    minutes = volume.index.hour * 60 + volume.index.minute
    if minutes.nunique() < 3:
        raise ValidationError(
            f"intraday_volume_profile: the index has only "
            f"{minutes.nunique()} distinct time(s) of day, so these are "
            "daily bars rather than intraday ones. There is no intraday "
            "profile in daily data."
        )

    n_buckets = max(3, int(n_buckets))
    lo, hi = int(minutes.min()), int(minutes.max())
    span = max(hi - lo, 1)
    bucket = np.minimum(((minutes - lo) / span * n_buckets).astype(int), n_buckets - 1)

    grouped = volume.groupby(bucket).agg(["sum", "mean", "count"])
    total = float(grouped["sum"].sum())
    profile: List[Dict[str, Any]] = []
    for index, row in grouped.iterrows():
        start_minute = lo + span * int(index) / n_buckets
        profile.append(
            {
                "bucket": int(index),
                "start_time": f"{int(start_minute) // 60:02d}:{int(start_minute) % 60:02d}",
                "share_of_volume": float(row["sum"] / total) if total > 0 else 0.0,
                "mean_volume": float(row["mean"]),
                "n_bars": int(row["count"]),
            }
        )

    shares = np.array([p["share_of_volume"] for p in profile])
    even = 1.0 / len(shares)
    first, last = shares[0], shares[-1]
    trough = float(shares.min())
    trough_bucket = int(np.argmin(shares))
    u_shaped = bool(
        first > even and last > even and trough_bucket not in (0, len(shares) - 1)
    )

    warnings = [
        "A schedule that spreads an order evenly across the CLOCK "
        "over-participates in the midday trough -- paying impact into a "
        "thin book -- and under-participates at the close, which is usually "
        "the cheapest liquidity of the day. Schedule against this profile, "
        "not against time.",
    ]
    if not u_shaped:
        warnings.append(
            "This profile is NOT U-shaped, which is unusual for a listed "
            "equity. Check the sample: a partial day, a single session, or "
            "a name whose volume is dominated by one scheduled auction all "
            "produce this."
        )
    if last > 0.20:
        warnings.append(
            f"The final bucket carries {last:.0%} of the day's volume. "
            "Closing auction share has risen for a decade on index flows, "
            "so a profile fitted over several years understates today's "
            "close -- check this against the recent subsample before using "
            "it to schedule."
        )

    return {
        "n_bars": int(len(volume)),
        "n_buckets": len(profile),
        "profile": profile,
        "u_shaped": u_shaped,
        "open_share": float(first),
        "close_share": float(last),
        "trough_share": trough,
        "trough_bucket": trough_bucket,
        "open_to_trough_ratio": float(first / trough) if trough > 0 else None,
        "warnings": warnings,
    }


def implementation_shortfall(
    *,
    decision_price: float,
    arrival_price: float,
    fills: Sequence[Dict[str, float]],
    target_quantity: float,
    final_price: float,
    side: str = "buy",
) -> Dict[str, Any]:
    """
    What an execution actually cost, decomposed after Perold (1988).

    EVERY OTHER COST TOOL HERE IS A MODEL. `estimate_trade_cost` predicts,
    `get_capacity_report` bounds, `plan_rebalance` schedules -- all before
    the fact, all under assumptions. This is the measurement, and it is the
    number those models should be checked against.

    THE FOUR COMPONENTS and why they are separated. The total gap between
    the decision price and what was actually achieved splits into:

    - **DELAY COST** -- the price moved between the decision and the first
      order reaching the market. This is a workflow problem, not a trading
      one, and it is frequently the largest term. No amount of clever
      execution recovers it.
    - **IMPACT COST** -- the price moved while the order was working,
      relative to arrival. This is the part an execution algorithm controls.
    - **OPPORTUNITY COST** -- the shares never filled, priced at the closing
      level. An algorithm that beats VWAP by never completing has simply
      moved its cost into this term, which is why a shortfall report without
      it is not a report.
    - **FEES**, which are known and are separated so they do not flatter or
      contaminate the parts that are measured.

    THE SIGN CONVENTION: a POSITIVE shortfall is a cost. It is stated
    because both conventions exist in the wild and the sign is the first
    thing misread.

    THE DECISION PRICE IS THE HARD PART, and it is an input here rather than
    inferred because only the caller knows it. Using the arrival price
    instead -- which is common, because it is easy -- sets the delay cost to
    zero by construction, and delay is often where the money went.
    """
    decision_price = float(decision_price)
    arrival_price = float(arrival_price)
    final_price = float(final_price)
    target_quantity = float(target_quantity)
    side = str(side).lower()
    if side not in ("buy", "sell"):
        raise ValidationError(
            f"implementation_shortfall: side must be 'buy' or 'sell', got {side!r}"
        )
    for name, value in (
        ("decision_price", decision_price),
        ("arrival_price", arrival_price),
        ("final_price", final_price),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValidationError(f"{name} must be positive and finite, got {value!r}")
    if target_quantity <= 0:
        raise ValidationError(
            "implementation_shortfall: target_quantity must be positive. Use "
            "`side` to express direction -- a negative quantity and a sell "
            "side together would double-negate."
        )
    if not fills:
        raise ValidationError(
            "implementation_shortfall: no fills. An order that never traded "
            "is entirely opportunity cost, which this can report -- but it "
            "needs an empty list to be passed deliberately rather than by "
            "omission, so supply [] explicitly if that is the case."
        )

    filled = 0.0
    notional = 0.0
    fees = 0.0
    for i, fill in enumerate(fills):
        quantity = float(fill.get("quantity", 0.0))
        price = float(fill.get("price", 0.0))
        if quantity < 0:
            raise ValidationError(
                f"implementation_shortfall: fill {i} has a negative quantity. "
                "Direction is carried by `side`, not by the fill signs."
            )
        if price <= 0 or not math.isfinite(price):
            raise ValidationError(
                f"implementation_shortfall: fill {i} has a non-positive price."
            )
        filled += quantity
        notional += quantity * price
        fees += float(fill.get("fee", 0.0))

    if filled <= 0:
        raise ValidationError(
            "implementation_shortfall: the fills sum to zero quantity."
        )
    if filled > target_quantity * 1.0001:
        raise ValidationError(
            f"implementation_shortfall: filled {filled} against a target of "
            f"{target_quantity}. An overfill is a reconciliation problem "
            "rather than an execution one, and the decomposition would be "
            "meaningless."
        )

    average_fill = notional / filled
    unfilled = max(target_quantity - filled, 0.0)
    # A buy pays MORE than the reference; a sell receives LESS. One sign.
    direction = 1.0 if side == "buy" else -1.0

    delay = direction * (arrival_price - decision_price) * filled
    impact = direction * (average_fill - arrival_price) * filled
    opportunity = direction * (final_price - decision_price) * unfilled
    total = delay + impact + opportunity + fees

    denominator = decision_price * target_quantity

    def _bps(value: float) -> float:
        return float(value / denominator * 1e4) if denominator > 0 else 0.0

    fill_rate = float(filled / target_quantity)
    components = [
        {"component": "delay", "dollars": float(delay), "bps": _bps(delay)},
        {"component": "impact", "dollars": float(impact), "bps": _bps(impact)},
        {
            "component": "opportunity",
            "dollars": float(opportunity),
            "bps": _bps(opportunity),
        },
        {"component": "fees", "dollars": float(fees), "bps": _bps(fees)},
    ]
    largest = max(components, key=lambda c: abs(c["dollars"]))

    warnings: List[str] = [
        "POSITIVE is a COST. Both sign conventions exist in the wild and "
        "this is the first thing misread.",
    ]
    if largest["component"] == "delay" and abs(delay) > abs(total) * 0.4:
        warnings.append(
            f"DELAY is the largest component at {_bps(delay):.1f} bps -- the "
            "price moved between the decision and the order reaching the "
            "market. That is a workflow problem rather than an execution "
            "one, and no algorithm recovers it."
        )
    if fill_rate < 0.95:
        warnings.append(
            f"Only {fill_rate:.0%} of the order filled, and the remainder is "
            f"priced as {_bps(opportunity):.1f} bps of opportunity cost. An "
            "algorithm that beats its benchmark by not completing has moved "
            "its cost here rather than saved it."
        )
    if abs(arrival_price - decision_price) < 1e-12:
        warnings.append(
            "The arrival price equals the decision price, so the delay cost "
            "is zero BY CONSTRUCTION. That usually means the arrival price "
            "was passed for both because the true decision price was not "
            "recorded -- in which case delay is not measured here, it is "
            "assumed away."
        )
    return {
        "side": side,
        "target_quantity": target_quantity,
        "filled_quantity": float(filled),
        "unfilled_quantity": float(unfilled),
        "fill_rate": fill_rate,
        "decision_price": decision_price,
        "arrival_price": arrival_price,
        "average_fill_price": float(average_fill),
        "final_price": final_price,
        "total_shortfall_dollars": float(total),
        "total_shortfall_bps": _bps(total),
        "components": components,
        "largest_component": largest["component"],
        "n_fills": len(fills),
        "warnings": warnings,
    }


__all__ = [
    "implementation_shortfall",
    "MIN_OBSERVATIONS",
    "amihud_illiquidity",
    "corwin_schultz_spread",
    "estimate_vpin",
    "intraday_volume_profile",
    "kyle_lambda",
    "order_flow_imbalance",
    "roll_spread",
]
