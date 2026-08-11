"""
modeling.bridge: the model -> backtest bridge. Deliberately a plain
Python function, not a 6th agent tool — the 5-tool modeling surface
stays exactly 5. This is the "artifacts, not tool calls" boundary
between the modeling registry and the existing 46-tool agent.tools
registry:

    run_model_experiment(...)                # modeling, 1 of 5 tools
        -> RunModelExperimentResult.oos_predictions_uri
    oos_predictions_to_signal_panel(...)      # this module, plain Python
        -> {ticker: {date: value}}
    run_signal_panel_backtest(...)            # agent.tools, the OTHER registry

Uses `run_model_experiment`'s walk-forward out-of-sample predictions,
never `score_model`'s single as-of snapshot: `score_model`'s model is the
FINAL full-panel refit, and using it to "predict" historical dates it was
trained on would be leakage, producing a falsely optimistic backtest. Each
fold's predictions come from a model that never saw that fold's dates
during training, and together they already span the whole dataset's date
range.

Two distinct leakage channels matter here, and only the first is closed by
fold construction alone:

  1. FEATURE overlap — a test-fold row's features coming from the training
     window. Prevented by the split itself plus `embargo`.
  2. LABEL overlap — a TRAINING row whose forward-return target is only
     resolved by prices inside the test window. Prevented by the
     target-overlap purge in engine.py, which drops any training row whose
     recorded `label_end_date` reaches the first test date.

Before that purge existed, a horizon larger than the embargo (e.g.
horizon=20, embargo=0) left training labels built from test-period prices,
so "leakage-safe by construction" was not accurate. Both channels are now
handled, but the guarantee rests on that purge running — it is not a
property of walk-forward splitting on its own.

Converts predictions to DIRECTION-valid signal values (sign of the
regression prediction, or a thresholded classifier probability), never
raw SignalType.SCORE values. Investigated `run_signal_panel_backtest`
directly: it never normalizes a SCORE value — the value is multiplied
straight into `strategy_return = lagged_signal * market_return` as a raw
leverage multiplier. Passing a raw forward-return prediction like `0.02`
through as SCORE would produce an economically meaningless ~2%-leveraged
position. DIRECTION sidesteps this entirely — it's units-invariant
regardless of the model's prediction scale, the same reasoning
`CustomSignalBacktestInput` already uses to default to DIRECTION over
SCORE.
"""

from typing import Dict, Literal

import numpy as np

from standard_quant_tools.error import ValidationError

from . import artifacts as _artifacts


def oos_predictions_to_signal_panel(
    oos_predictions_uri: str,
    task: Literal["regression", "classification"],
    proba_threshold: float = 0.5,
    deadband: float = 0.0,
    long_only: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    Load a walk-forward out-of-sample predictions artifact (columns
    date, entity, prediction — as persisted by run_model_experiment) and
    reshape it into the {ticker: {date: value}} shape
    SignalPanelBacktestInput.signal_panel needs, with values valid for
    SignalType.DIRECTION (each exactly -1.0, 0.0, or 1.0).

    Args:
        task: must match the ModelSpec.task the model was trained with.
            regression -> sign(prediction), 0.0 inside +/-deadband.
            classification -> prediction is already a positive-class
            probability (see engine.py::_predict_fold): 1.0 if
            proba > proba_threshold else 0.0 when long_only (default —
            a classifier's "not predicted up" class isn't necessarily
            short-worthy); with long_only=False, symmetric: 1.0 above
            proba_threshold, -1.0 below (1 - proba_threshold), else 0.0.
        deadband: regression only. |prediction| <= deadband -> flat (0.0),
            avoiding a leveraged position on a near-zero, likely-noise
            prediction.
        proba_threshold: classification only, must be in (0, 1); when
            long_only=False, must additionally be >= 0.5 (a symmetric
            decision boundary below the midpoint is ambiguous — the long
            and short conditions would overlap).

    Returns:
        {ticker: {date: value}} — pass directly as
        SignalPanelBacktestInput.signal_panel with signal_type=SignalType.DIRECTION.

    Raises:
        ValidationError: deadband < 0, proba_threshold outside (0, 1), or
        proba_threshold < 0.5 with long_only=False.
    """
    if deadband < 0:
        raise ValidationError(f"deadband must be >= 0, got {deadband}")
    if not (0.0 < proba_threshold < 1.0):
        raise ValidationError(
            f"proba_threshold must be in (0, 1), got {proba_threshold}"
        )
    if not long_only and proba_threshold < 0.5:
        raise ValidationError(
            "proba_threshold must be >= 0.5 when long_only=False (a symmetric long/short "
            f"decision boundary below the midpoint is ambiguous), got {proba_threshold}"
        )

    predictions_df = _artifacts.load_artifact(oos_predictions_uri)
    dates = predictions_df["date"].dt.strftime("%Y-%m-%d")
    raw = predictions_df["prediction"].to_numpy(dtype=float)

    if task == "regression":
        direction = np.sign(raw)
        direction[np.abs(raw) <= deadband] = 0.0
    else:
        if long_only:
            direction = np.where(raw > proba_threshold, 1.0, 0.0)
        else:
            direction = np.where(
                raw > proba_threshold,
                1.0,
                np.where(raw < (1.0 - proba_threshold), -1.0, 0.0),
            )

    signal_panel: Dict[str, Dict[str, float]] = {}
    for entity, date, value in zip(predictions_df["entity"], dates, direction):
        signal_panel.setdefault(entity, {})[date] = float(value)
    return signal_panel
