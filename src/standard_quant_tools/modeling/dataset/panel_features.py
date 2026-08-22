"""
Batch computation of the technical features, for the whole universe at once.

build_dataset's natural shape is a loop: for each entity, for each feature,
call the per-ticker wrapper. Measured, that loop spends most of its time not
in the indicator arithmetic but in the pandas round trip around it — Series
to NumPy, validation, logging, Series reconstruction, once per entity per
feature. indicators/panel.py already exists to remove exactly that overhead
(one native call for the whole universe, tickers computed in parallel), and
was measured at 11.9x over the per-ticker loop. This module is the adapter
that lets build_dataset reach it.

WHEN THE FAST PATH IS USED, AND WHY THE GUARD IS STRICT

`technical_indicators_panel` stacks the universe onto ONE index, and the
index it uses is the intersection of every ticker's bars. That is the only
shape a dense matrix can have, and it is not equivalent to computing each
entity over its own full history:

  * a ticker with a shorter history truncates the panel for everyone, so
    entities would lose rows they are entitled to, and
  * every indicator here is path-dependent (Wilder smoothing, EMAs), so
    starting a series later changes its warm-up and therefore its VALUES,
    not merely its coverage

Both of those would move numbers, quietly, on a change whose entire purpose
is speed. So the fast path is taken only when every entity's index is
IDENTICAL, which makes the intersection a no-op and the two paths exactly
equivalent. A universe with mid-sample IPOs, delistings, or entities on
different holiday calendars simply falls back to the per-entity loop, which
is correct there and always was.

The transforms for the derived features (atr_pct, bollinger_pct_b) are
imported from features/risk.py rather than reimplemented here, so there is
one definition of each and no way for the two paths to drift apart.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from standard_quant_tools.indicators.panel import HAS_CPP, technical_indicators_panel

from ..features.risk import atr_pct_from_atr, pct_b_from_bands

logger = logging.getLogger(__name__)

# feature id -> (panel indicator, {feature param: panel kwarg}, field)
#
# `field` names the column to take out of a multi-column indicator, or None
# when the indicator returns one value per bar. Features needing a transform
# on top (atr_pct, bollinger_pct_b) are handled explicitly in _extract.
_PANEL_FEATURES: Dict[str, Tuple[str, Dict[str, str], Optional[str]]] = {
    "technical.rsi": ("rsi", {"period": "rsi_period"}, None),
    "technical.adx": ("adx", {"period": "adx_period"}, "ADX"),
    "technical.stochastic_k": (
        "stochastic_oscillator",
        {"k_period": "stoch_k_period", "d_period": "stoch_d_period"},
        "Stoch_K",
    ),
    "risk.atr_pct": ("atr", {"period": "atr_period"}, None),
    "risk.bollinger_pct_b": (
        "bollinger_bands",
        {"period": "bollinger_period", "num_std": "bollinger_num_std"},
        None,
    ),
}

_REQUIRED_COLUMNS = ("High", "Low", "Close")


def _indices_identical(ohlcv_by_entity: Mapping[str, pd.DataFrame]) -> bool:
    """True when every entity carries exactly the same bar index."""
    reference: Optional[pd.Index] = None
    for frame in ohlcv_by_entity.values():
        if reference is None:
            reference = frame.index
        elif not reference.equals(frame.index):
            return False
    return reference is not None


def _extract(
    feature_id: str,
    field: Optional[str],
    indicator_frame: pd.DataFrame,
    symbol: str,
    close: pd.Series,
) -> pd.Series:
    """One entity's column out of a panel result, plus any transform."""
    if feature_id == "risk.atr_pct":
        return atr_pct_from_atr(indicator_frame[symbol], close)
    if feature_id == "risk.bollinger_pct_b":
        return pct_b_from_bands(
            close,
            indicator_frame[(symbol, "BB_Upper")],
            indicator_frame[(symbol, "BB_Lower")],
        )
    if field is None:
        return indicator_frame[symbol]
    return indicator_frame[(symbol, field)]


def _batch(requests: List[Tuple[str, str, Dict[str, Any], Optional[str]]]):
    """
    Pack requests into panel calls.

    A single call parameterizes each indicator once (rsi_period, adx_period,
    ...), so two aliases of the same feature at different periods cannot
    share one call. Requests are packed greedily into as few calls as
    possible, which is one call for the overwhelmingly common case where no
    indicator is requested twice.
    """
    batches: List[List[Tuple[str, str, Dict[str, Any], Optional[str]]]] = []
    for request in requests:
        indicator = request[1]
        for batch in batches:
            if all(existing[1] != indicator for existing in batch):
                batch.append(request)
                break
        else:
            batches.append([request])
    return batches


def compute_panel_features(
    feature_specs: Sequence[Any],
    feature_defs: Sequence[Any],
    resolved_params: Sequence[Dict[str, Any]],
    ohlcv_by_entity: Mapping[str, pd.DataFrame],
) -> Dict[str, Dict[str, pd.Series]]:
    """
    Compute every panel-eligible feature for the whole universe at once.

    Returns {output_name: {symbol: Series}} covering only the features that
    were eligible; the caller computes the rest per entity as before. An
    empty dict means the fast path did not apply, which is a normal outcome
    and not an error.
    """
    if not HAS_CPP:
        # The pure-Python panel fallback loops per ticker anyway, so there
        # is nothing to win and a stacking cost to pay.
        return {}
    if len(ohlcv_by_entity) < 2:
        return {}

    requests: List[Tuple[str, str, Dict[str, Any], Optional[str]]] = []
    for fs, definition, params in zip(feature_specs, feature_defs, resolved_params):
        mapping = _PANEL_FEATURES.get(definition.id)
        if mapping is None:
            continue
        indicator, param_map, field = mapping
        # An unrecognized parameter would silently be dropped from the
        # panel call and computed at the default instead, so anything not
        # in the map disqualifies the feature rather than being ignored.
        if set(params) - set(param_map):
            continue
        kwargs = {param_map[key]: value for key, value in params.items()}
        requests.append((fs.output_name, indicator, kwargs, field))

    if not requests:
        return {}
    if not _indices_identical(ohlcv_by_entity):
        logger.debug(
            "[modeling] panel feature path skipped: entity indices differ "
            "(ragged history), falling back to the per-entity loop"
        )
        return {}
    if any(
        column not in frame.columns
        for frame in ohlcv_by_entity.values()
        for column in _REQUIRED_COLUMNS
    ):
        # The panel stacker needs High/Low/Close for every ticker even when
        # the requested indicator only reads Close.
        return {}

    feature_ids = {
        fs.output_name: definition.id
        for fs, definition in zip(feature_specs, feature_defs)
    }
    symbols = list(ohlcv_by_entity)
    out: Dict[str, Dict[str, pd.Series]] = {}
    for batch in _batch(requests):
        kwargs: Dict[str, Any] = {}
        for _, _, batch_kwargs, _ in batch:
            kwargs.update(batch_kwargs)
        indicators = [indicator for _, indicator, _, _ in batch]
        panel = technical_indicators_panel(
            ohlcv_by_entity, indicators=indicators, **kwargs
        )
        for output_name, indicator, _, field in batch:
            frame = panel[indicator]
            out[output_name] = {
                symbol: _extract(
                    feature_ids[output_name],
                    field,
                    frame,
                    symbol,
                    ohlcv_by_entity[symbol]["Close"],
                )
                for symbol in symbols
            }
    logger.debug(
        "[modeling] panel feature path: %d feature(s) over %d entities in "
        "%d native call(s)",
        len(out),
        len(symbols),
        len(_batch(requests)),
    )
    return out
