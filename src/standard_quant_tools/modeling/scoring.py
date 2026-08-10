"""
score_model: load a registered model + the exact DatasetSpec that trained
it, rebuild the same features as of a given date for a (possibly new)
universe, predict, and write predictions to a Parquet artifact via
modeling.artifacts.save_artifact.

Reuses dataset.builder.build_dataset(include_target=False) — the scoring
path deliberately skips target construction, since a forward-return
target needs `horizon` bars of future data that don't exist for "today".
Applies the SAME preprocessing stats the registered model's final refit
used (persisted by registry.model_registry.save_model), not freshly
fit stats on the scoring universe — otherwise the same input row could
score differently depending on which other tickers happened to be in the
scoring call.
"""

from typing import Any, Dict, List

import pandas as pd

from standard_quant_tools.error import ValidationError

from . import artifacts as _artifacts
from .dataset.builder import build_dataset
from .features.transforms import apply_preprocessing
from .registry.model_registry import (
    load_manifest,
    load_model,
    load_preprocessing_stats,
)
from .specs import DatasetSpec, _parse_date
from .validation.metrics import positive_class_proba


def score_model(
    model_id: str, as_of: str, universe: List[str], lookback_days: int = 400
) -> Dict[str, Any]:
    """
    Args:
        lookback_days: calendar days of history fetched before `as_of` so
            every requested feature's lookback window has enough data —
            widen this if the model's features use unusually large
            windows (e.g. a custom feature with lookback > 252 bars).

    Raises:
        ValidationError: no registered model with `model_id`, or no
        entity in `universe` has a scoreable row as of `as_of`.
    """
    try:
        as_of_ts = _parse_date(as_of, "as_of")
    except ValueError as exc:
        # _parse_date raises plain ValueError so it also works unmodified
        # inside a pydantic validator (ScoreModelInput._valid_date) --
        # here, called directly from a plain function, re-raise as this
        # module's own established error type instead of leaking a raw
        # ValueError inconsistent with every other score_model failure.
        raise ValidationError(str(exc)) from exc
    manifest = load_manifest(model_id)
    stats = load_preprocessing_stats(model_id)
    estimator = load_model(model_id)

    dataset_spec_path = _artifacts.run_dir(manifest.dataset_id) / "dataset_spec.json"
    original_spec_dict = _artifacts.load_json(str(dataset_spec_path))

    start = (as_of_ts - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    # Reconstruct through DatasetSpec(**...) rather than
    # original_spec.model_copy(update=...) -- model_copy does NOT re-run
    # validators in pydantic v2, so it would silently bypass the
    # duplicate-symbol/start-before-end checks DatasetSpec defines for
    # every other caller.
    scoring_spec = DatasetSpec(
        **{**original_spec_dict, "universe": universe, "start": start, "end": as_of}
    )

    built = build_dataset(scoring_spec, include_target=False)
    panel = built["panel"]
    latest = panel.sort_values("date").groupby("entity", as_index=False).tail(1)
    if latest.empty:
        raise ValidationError(
            f"score_model: no scoreable rows as of {as_of!r} for universe {universe} — "
            "try a larger lookback_days."
        )

    missing_entities = sorted(set(universe) - set(latest["entity"]))

    X = apply_preprocessing(latest[manifest.feature_ids], stats)
    if manifest.task == "regression":
        predictions = estimator.predict(X.to_numpy())
    else:
        predictions = positive_class_proba(estimator, X.to_numpy())

    predictions_df = pd.DataFrame(
        {
            "entity": latest["entity"].to_numpy(),
            "date": latest["date"].to_numpy(),
            "prediction": predictions,
        }
    )
    run_name = f"predictions_{as_of_ts.strftime('%Y%m%d')}"
    predictions_uri = _artifacts.save_artifact(
        predictions_df, run_id=model_id, name=run_name, overwrite=True
    )

    return {
        "model_id": model_id,
        "as_of": as_of,
        "predictions_uri": predictions_uri,
        "n_entities": int(len(predictions_df)),
        "missing_entities": missing_entities,
        "summary_stats": {
            "mean": float(predictions_df["prediction"].mean()),
            "std": float(predictions_df["prediction"].std()) if len(predictions_df) > 1 else 0.0,
            "min": float(predictions_df["prediction"].min()),
            "max": float(predictions_df["prediction"].max()),
        },
    }
