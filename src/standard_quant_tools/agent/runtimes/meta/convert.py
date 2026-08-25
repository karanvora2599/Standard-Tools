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
    panel: Dict[str, Dict[str, float]] = {}
    for row in frame.itertuples(index=False):
        panel.setdefault(str(row.entity), {})[str(row.date)] = float(row.prediction)
    return panel, [
        "Scores are the raw predictions, unscaled. They are only meaningful "
        "cross-sectionally unless the model was trained to a calibrated "
        "target."
    ]


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
    scores = handoff.resolve(input_data.ref, expect="score_panel")
    frame = pd.DataFrame(scores)
    weights = sizers[input_data.construction_method](
        frame, gross_leverage=input_data.gross_leverage
    )
    panel = {
        str(column): {
            str(index): float(value)
            for index, value in weights[column].dropna().items()
        }
        for column in weights.columns
    }
    return panel, [
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


_HANDLERS = {
    ("predictions", "signal_panel"): _predictions_to_signal_panel,
    ("predictions", "score_panel"): _predictions_to_score_panel,
    ("score_panel", "weight_panel"): _score_panel_to_weight_panel,
    ("signal_panel", "score_panel"): _signal_panel_to_score_panel,
}
