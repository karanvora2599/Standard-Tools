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

import hashlib
import json
from typing import Any, Dict, List

import pandas as pd

from standard_quant_tools.error import ValidationError

from . import artifacts as _artifacts
from .dataset.builder import build_dataset
from .features.transforms import apply_preprocessing
from .registry.model_registry import (
    load_dataset_spec,
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

    # ── Future-trained-model guard ────────────────────────────────────────
    # The registered estimator is refit on the ENTIRE training panel, so it
    # has already seen every date up to train_end_date. Scoring an as_of at
    # or before that date produces a prediction that LOOKS point-in-time but
    # was made by a model trained on the very future it is "predicting" --
    # the exact mistake the walk-forward OOS predictions exist to avoid.
    # Nothing previously compared as_of against the training window at all.
    #
    # For a genuine historical evaluation use the model's OOS predictions
    # artifact (each fold predicted only dates outside its own training
    # window); see modeling.bridge.oos_predictions_to_signal_panel.
    if manifest.train_end_date is not None:
        train_end_ts = _parse_date(manifest.train_end_date, "train_end_date")
        if as_of_ts <= train_end_ts:
            raise ValidationError(
                f"score_model: as_of {as_of!r} is not after this model's training data, "
                f"which ends {manifest.train_end_date}. The registered estimator is refit "
                "on the full training panel, so scoring a date it was trained on returns a "
                "future-trained prediction disguised as a historical one. For historical "
                "evaluation use the model's walk-forward OOS predictions "
                f"({manifest.oos_predictions_uri}) via "
                "modeling.bridge.oos_predictions_to_signal_panel, which are genuinely "
                "out-of-sample; use score_model only for dates after training."
            )

    stats = load_preprocessing_stats(model_id)
    estimator = load_model(model_id)

    # The model's OWN bundled, content-verified copy -- not
    # SQT_RUNS_DIR/<dataset_id>/dataset_spec.json. Reading the dataset
    # directory made scoring depend on that directory surviving, and let an
    # edit there (say RSI period 14 -> 100) silently redefine the features
    # fed to an already-registered estimator, with no integrity check and
    # no change in model_id.
    original_spec_dict = load_dataset_spec(model_id)

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
    # The artifact name includes a digest of the scored universe, not just
    # the date. With `predictions_YYYYMMDD` alone, scoring [AAPL, MSFT] and
    # then [AAPL, NVDA] for the same as_of overwrote the first file, so an
    # audit record written by the earlier call pointed at contents produced
    # by the later one -- a silently wrong provenance trail rather than a
    # missing one. overwrite=True is kept so re-scoring the SAME universe
    # on the same date is idempotent.
    universe_digest = hashlib.sha256(
        json.dumps(sorted(universe)).encode("utf-8")
    ).hexdigest()[:8]
    run_name = f"predictions_{as_of_ts.strftime('%Y%m%d')}_{universe_digest}"
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
            "std": (
                float(predictions_df["prediction"].std())
                if len(predictions_df) > 1
                else 0.0
            ),
            "min": float(predictions_df["prediction"].min()),
            "max": float(predictions_df["prediction"].max()),
        },
    }
