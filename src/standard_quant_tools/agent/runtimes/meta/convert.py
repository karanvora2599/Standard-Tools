"""
Kind-to-kind conversions for the handoff interconnect.

Kept out of the tool module because this is the table that decides which
producers can reach which consumers, and it should be readable as a table.
Every entry is a conversion that is genuinely well defined -- there is
deliberately no "best effort" path, because a handoff that quietly
guesses is worse than one that refuses.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd

from standard_quant_tools.error import ValidationError

#: (source kind, destination kind) pairs this module can perform.
CONVERSIONS: Tuple[Tuple[str, str], ...] = (
    ("predictions", "signal_panel"),
    ("predictions", "score_panel"),
    ("score_panel", "weight_panel"),
    ("signal_panel", "score_panel"),
    ("equity_curve", "returns_panel"),
)


def convert(input_data: Any, source: Any) -> Tuple[Any, List[str]]:
    """Perform one conversion, returning the value and any caveats."""
    pair = (source.kind, input_data.to_kind)
    if pair not in CONVERSIONS:
        raise ValidationError(
            f"no conversion from {source.kind!r} to {input_data.to_kind!r}. "
            f"Available: {sorted(CONVERSIONS)}. There is deliberately no "
            "best-effort path — a handoff that guesses is worse than one "
            "that refuses."
        )
    return _HANDLERS[pair](input_data, source)


def _predictions_to_signal_panel(input_data, source):
    from standard_quant_tools.agent.runtimes import handoff
    from standard_quant_tools.modeling.bridge import oos_predictions_to_signal_panel

    frame = handoff.resolve(input_data.ref, expect="predictions")
    if input_data.task is None:
        raise ValidationError(
            "converting predictions to a signal panel needs `task`: a "
            "regression prediction thresholded as a probability produces a "
            "nonsensical but valid-looking panel, which is a wrong answer "
            "rather than an error."
        )

    # Reuses the modeling bridge rather than reimplementing the rule, so a
    # converted panel is identical to what that function produces.
    from standard_quant_tools.backtest.artifacts import save_artifact

    # The bridge requires a datetime64 date column. Coerced HERE rather
    # than demanded of every publisher: a `predictions` reference can be
    # produced by anything, and rejecting a frame whose dates are strings
    # would push a trivially fixable representation detail onto every
    # producer in the fleet. A date that cannot be parsed still fails.
    if "date" in frame.columns and not pd.api.types.is_datetime64_any_dtype(
        frame["date"]
    ):
        frame = frame.copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="raise")

    uri = save_artifact(
        frame, input_data.run_id, f"{input_data.name}_src", overwrite=True
    )
    panel = oos_predictions_to_signal_panel(
        oos_predictions_uri=uri,
        task=input_data.task,
        deadband=input_data.deadband,
        proba_threshold=input_data.proba_threshold,
        long_only=input_data.long_only,
    )
    notes = [
        "Magnitude is discarded on purpose: a signal panel's value is read "
        "as a leverage multiplier, so passing a 0.02 forward-return "
        "prediction through unchanged would size a 2%-leveraged position.",
    ]
    if input_data.task == "classification" and input_data.long_only:
        notes.append(
            "long_only=True: the negative class is flat rather than short. "
            "'Not predicted up' is not the same claim as 'predicted down'."
        )
    return panel, notes


def _predictions_to_score_panel(input_data, source):
    from standard_quant_tools.agent.runtimes import handoff

    frame = handoff.resolve(input_data.ref, expect="predictions")
    required = {"date", "entity", "prediction"}
    missing = required - set(frame.columns)
    if missing:
        raise ValidationError(
            f"a predictions frame needs {sorted(required)}; missing {sorted(missing)}"
        )
    # `task` was accepted and ignored here, while the SIGNAL panel
    # conversion refuses without it -- and for the reason that applies just
    # as well to this one: "a regression prediction thresholded as a
    # probability produces a nonsensical but valid-looking panel, which is
    # a wrong answer rather than an error." A classifier's predictions are
    # probabilities, so every score is positive and the sign carries no
    # direction. Two of the three sizers downstream recentre and hide it;
    # `vol_scaled` does not, and neither does any consumer that reads the
    # sign directly.
    offset = 0.0
    notes = [
        "Scores are the raw predictions, unscaled. They are only meaningful "
        "cross-sectionally unless the model was trained to a calibrated "
        "target."
    ]
    if input_data.task == "classification":
        offset = float(input_data.proba_threshold)
        notes.append(
            f"Classification: probabilities recentred by subtracting "
            f"proba_threshold ({offset:g}), so a score's SIGN is the "
            "predicted direction. Raw probabilities are all positive, which "
            "reads as an all-long book to any consumer that does not "
            "recentre cross-sectionally."
        )
    elif input_data.task is None:
        notes.append(
            "`task` was not given, so the predictions are passed through "
            "unchanged. If these came from a CLASSIFIER they are "
            "probabilities in [0, 1] and every score is positive -- pass "
            "task='classification' to have them recentred."
        )

    panel: Dict[str, Dict[str, float]] = {}
    for row in frame.itertuples(index=False):
        panel.setdefault(str(row.entity), {})[str(row.date)] = (
            float(row.prediction) - offset
        )
    return panel, notes


def _score_panel_to_weight_panel(input_data, source):
    import pandas as pd

    from standard_quant_tools.agent.runtimes import handoff
    from standard_quant_tools.backtest.sizing import (
        rank_weighted,
        vol_scaled,
        zscore_normalized,
    )

    sizers = {
        "rank_weighted": rank_weighted,
        "zscore_normalized": zscore_normalized,
        "vol_scaled": vol_scaled,
    }
    if input_data.construction_method not in sizers:
        raise ValidationError(
            f"construction_method must be one of {sorted(sizers)}, got "
            f"{input_data.construction_method!r}"
        )
    # `vol_scaled` divides each score by that name's TRAILING REALIZED
    # VOLATILITY, which a score panel does not carry -- it needs a returns
    # frame aligned to the same columns. It was listed as available and
    # then raised a bare `TypeError: vol_scaled() missing 1 required
    # positional argument: 'returns_df'`, so the tool advertised an option
    # that could never work and failed in a way that read as a bug in the
    # library rather than a wrong argument.
    if input_data.construction_method == "vol_scaled":
        raise ValidationError(
            "construction_method='vol_scaled' cannot be built from a score "
            "panel alone: it divides each name's score by that name's "
            "trailing realized volatility, and a score panel carries no "
            "returns to measure that from. Use 'rank_weighted' or "
            "'zscore_normalized' here, or size with vol scaling inside "
            "run_portfolio_simulation, which has the price history."
        )
    scores = handoff.resolve(input_data.ref, expect="score_panel")
    frame = pd.DataFrame(scores)
    weights = sizers[input_data.construction_method](
        frame, gross_leverage=input_data.gross_leverage
    )
    # `publish` takes the frame directly now; this hand-rolled copy of
    # `handoff._frame_to_mapping` was the second of three.
    return weights, [
        f"Weights built by backtest.sizing.{input_data.construction_method}, "
        "the same constructor run_portfolio_simulation would have used — "
        "not a reimplementation of it."
    ]


def _signal_panel_to_score_panel(input_data, source):
    from standard_quant_tools.agent.runtimes import handoff

    panel = handoff.resolve(input_data.ref, expect="signal_panel")
    return panel, [
        "A signal panel is already -1/0/+1, so this only relabels the kind. "
        "The information magnitude would have carried was discarded when "
        "the signal panel was made, and this does not recover it."
    ]


def _equity_curve_to_returns(input_data, source):
    """An equity curve into the returns every metric tool actually wants.

    THE REFUSAL THAT NEEDED A REMEDY. An `equity_curve` is levels -- the
    kind says so, "account value per bar" -- and `calculate_series_metrics`
    documents its input as a RETURN series. Handing it the curve used to
    produce a Sharpe of 302 against a true 0.30, because mean/std of a level
    series is scale-invariant and looks like an ordinary ratio.

    `resolve_source` refuses that now, and told the caller to "convert first
    -- .pct_change().dropna()", which is not something an agent holding a
    REFERENCE can do. That made the refusal a dead end: right answer,
    nowhere to go. This is the way out.

    `returns_panel` rather than a new series kind. It is documented as a
    wide frame of per-asset returns and a single asset is the degenerate
    case of that; `resolve_source` already unwraps a one-column frame. A
    `return_series` kind would double the taxonomy for one shape, which is
    the trade `handoff.py` declines elsewhere for the same reason.
    """
    from standard_quant_tools.agent.runtimes import handoff

    curve = handoff.resolve(input_data.ref, expect="equity_curve")
    if isinstance(curve, pd.DataFrame):
        if curve.shape[1] != 1:
            raise ValidationError(
                f"{input_data.ref!r} resolves to {curve.shape[1]} columns; an "
                "equity curve is one series."
            )
        curve = curve.iloc[:, 0]
    curve = pd.Series(curve).astype("float64")
    if len(curve) < 2:
        raise ValidationError(
            f"{input_data.ref!r} has {len(curve)} point(s); a return needs a "
            "prior value to be measured against."
        )

    non_positive = int((curve <= 0).sum())
    returns = curve.pct_change(fill_method=None).dropna()
    if returns.empty:
        raise ValidationError(
            f"{input_data.ref!r} produced no finite returns -- every bar was "
            "null or unchanged from a null."
        )

    notes = [
        f"{len(curve)} equity points -> {len(returns)} returns. The FIRST BAR "
        "IS DROPPED: it has no prior value, and carrying it as 0.0 would add "
        "a fabricated flat period that pulls the mean toward zero and the "
        "volatility down with it.",
        "Simple returns, not log. The metrics that consume this compound "
        "them back with (1 + r).cumprod(), so a log return would restate the "
        "curve it came from.",
    ]
    if non_positive:
        notes.append(
            f"WARNING: {non_positive} equity value(s) were <= 0, so the "
            "returns around them are not meaningful -- an account that "
            "reached zero has no percentage change to report."
        )
    return returns.to_frame(name="return"), notes


_HANDLERS = {
    ("predictions", "signal_panel"): _predictions_to_signal_panel,
    ("predictions", "score_panel"): _predictions_to_score_panel,
    ("score_panel", "weight_panel"): _score_panel_to_weight_panel,
    ("signal_panel", "score_panel"): _signal_panel_to_score_panel,
    ("equity_curve", "returns_panel"): _equity_curve_to_returns,
}
