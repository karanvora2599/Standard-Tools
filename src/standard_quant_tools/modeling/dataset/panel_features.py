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
is speed. So a panel call only ever stacks entities whose indices are
IDENTICAL, which makes the intersection a no-op and the two paths exactly
equivalent.

The universe is therefore split into groups of entities that share one
index, and each group of two or more is served by its own panel call. An
entity whose history matches no other -- a mid-sample IPO, a delisting, a
different holiday calendar -- is left to the per-entity loop, which is
correct for it and always was. One such entity used to send the WHOLE
universe to that loop (see the CHANGELOG entry of 2026-10-01); now it costs
only its own share. Padding a different history with NaN to join a larger
panel is not done: the kernels treat a NaN inside a window as a gap, so an
entity on another holiday calendar would come out different from its own
loop, and grouping is exact by construction without having to prove the
leading- and trailing-pad cases indicator by indicator.

The transforms for the derived features (atr_pct, bollinger_pct_b) are
imported from features/risk.py rather than reimplemented here, so there is
one definition of each and no way for the two paths to drift apart.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from standard_quant_tools.error import ValidationError
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


def _aligned_groups(ohlcv_by_entity: Mapping[str, pd.DataFrame]) -> List[List[str]]:
    """
    The entities, grouped by identical bar index.

    Groups come in order of first appearance and members in universe order,
    so the result is a function of the mapping alone. Candidates are
    bucketed on (length, first bar, last bar) before the full `equals`
    comparison, which keeps a universe of all-different histories linear
    rather than comparing every pair.
    """
    buckets: Dict[Tuple[Any, ...], List[Tuple[pd.Index, List[str]]]] = {}
    groups: List[List[str]] = []
    for symbol, frame in ohlcv_by_entity.items():
        index = frame.index
        key = (len(index), index[0], index[-1]) if len(index) else ()
        candidates = buckets.setdefault(key, [])
        for reference, members in candidates:
            if reference.equals(index):
                members.append(symbol)
                break
        else:
            members = [symbol]
            candidates.append((index, members))
            groups.append(members)
    return groups


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


def _serve_group(
    group: Mapping[str, pd.DataFrame],
    batches: List[List[Tuple[str, str, Dict[str, Any], Optional[str]]]],
    feature_ids: Mapping[str, str],
) -> Dict[str, Dict[str, pd.Series]]:
    """Every batch of requests for one group of identically indexed
    entities: {output_name: {symbol: Series}}."""
    out: Dict[str, Dict[str, pd.Series]] = {}
    for batch in batches:
        kwargs: Dict[str, Any] = {}
        for _, _, batch_kwargs, _ in batch:
            kwargs.update(batch_kwargs)
        indicators = [indicator for _, indicator, _, _ in batch]
        panel = technical_indicators_panel(group, indicators=indicators, **kwargs)
        for output_name, indicator, _, field in batch:
            frame = panel[indicator]
            out[output_name] = {
                symbol: _extract(
                    feature_ids[output_name],
                    field,
                    frame,
                    symbol,
                    entity["Close"],
                )
                for symbol, entity in group.items()
            }
    return out


def compute_panel_features(
    feature_specs: Sequence[Any],
    feature_defs: Sequence[Any],
    resolved_params: Sequence[Dict[str, Any]],
    ohlcv_by_entity: Mapping[str, pd.DataFrame],
) -> Dict[str, Dict[str, pd.Series]]:
    """
    Compute every panel-eligible feature for the whole universe at once.

    Returns {output_name: {symbol: Series}} covering only the features that
    were eligible, and within each only the entities a panel call served;
    the caller computes everything else per entity as before. An empty dict
    means the fast path did not apply, which is a normal outcome and not an
    error.

    Entities are served in groups that share an identical bar index (see
    the module docstring for why nothing looser is exact). An entity in no
    group of two or more, or one missing a column the stacker needs, is
    absent from the result and falls to the per-entity loop.
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
    # The panel stacker needs High/Low/Close for every ticker it is handed,
    # even when the requested indicator only reads Close, so an entity
    # without them is not stacked at all; nor is one with no bars, which
    # the stacker refuses (build_dataset never gets that far with one).
    stackable = {
        symbol: frame
        for symbol, frame in ohlcv_by_entity.items()
        if len(frame.index)
        and all(column in frame.columns for column in _REQUIRED_COLUMNS)
    }
    groups = [members for members in _aligned_groups(stackable) if len(members) >= 2]
    if not groups:
        logger.debug(
            "[modeling] panel feature path skipped: no two entities share a "
            "bar index, falling back to the per-entity loop"
        )
        return {}

    feature_ids = {
        fs.output_name: definition.id
        for fs, definition in zip(feature_specs, feature_defs)
    }
    batches = _batch(requests)
    out: Dict[str, Dict[str, pd.Series]] = {}
    n_served = 0
    for members in groups:
        group = {symbol: stackable[symbol] for symbol in members}
        try:
            served = _serve_group(group, batches, feature_ids)
        except ValidationError:
            # A refusal from the panel -- an infinity in a stacked column,
            # say -- is what a universe stacked WHOLE has always met, so it
            # stands. A group that is only part of the universe used to be
            # computed by the per-entity loop, so it is handed back to it,
            # and the loop answers or refuses exactly as it did before.
            if len(members) == len(ohlcv_by_entity):
                raise
            logger.debug(
                "[modeling] panel feature path declined a group of %d "
                "entities; computing them per entity",
                len(members),
            )
            continue
        for output_name, by_symbol in served.items():
            out.setdefault(output_name, {}).update(by_symbol)
        n_served += len(members)
    logger.debug(
        "[modeling] panel feature path: %d feature(s) over %d of %d entities "
        "in %d group(s), %d native call(s); the other %d computed per entity",
        len(out),
        n_served,
        len(ohlcv_by_entity),
        len(groups),
        len(groups) * len(batches),
        len(ohlcv_by_entity) - n_served,
    )
    return out
