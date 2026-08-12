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
from typing import Any, Dict, List, Optional

import pandas as pd

from standard_quant_tools.error import ValidationError

from . import artifacts as _artifacts
from .dataset.builder import build_dataset
from .features.base import FeatureScope
from .features.registry import get_feature
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
    model_id: str,
    as_of: str,
    universe: List[str],
    lookback_days: int = 400,
    max_staleness_days: Optional[int] = None,
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
    # has already seen every date up to its information cutoff. Scoring an
    # as_of at or before that produces a prediction that LOOKS
    # point-in-time but was made by a model trained on the very future it is
    # "predicting" -- the exact mistake the walk-forward OOS predictions
    # exist to avoid.
    #
    # The cutoff is training_information_cutoff (max label_end_date), NOT
    # train_end_date (max feature date). A horizon-h forward-return target
    # reads Close[t+h] to build the label for a row dated t, so the training
    # data consumed prices h bars past its last feature date. Gating on the
    # feature date left exactly that horizon-wide window -- ~28 calendar
    # days at h=20 -- accepting an as_of whose future the model had already
    # been shown.
    #
    # Falls back to train_end_date only for manifests written before
    # training_information_cutoff existed: that is the older, weaker
    # guarantee, which is still better than no guard, and the message says
    # which one is in force so a stale manifest is not mistaken for a
    # verified one.
    cutoff_value = manifest.training_information_cutoff
    cutoff_field = "training_information_cutoff"
    if cutoff_value is None:
        cutoff_value = manifest.train_end_date
        cutoff_field = "train_end_date"
    if cutoff_value is not None:
        cutoff_ts = _parse_date(cutoff_value, cutoff_field)
        if as_of_ts <= cutoff_ts:
            weaker = (
                ""
                if cutoff_field == "training_information_cutoff"
                else (
                    " (This model predates the label-aware cutoff, so the check used its "
                    "last FEATURE date; its labels consumed prices beyond that, meaning "
                    "the true unsafe window extends further than this message states. "
                    "Retrain to get the exact cutoff.)"
                )
            )
            raise ValidationError(
                f"score_model: as_of {as_of!r} is not after this model's training "
                f"information cutoff, {cutoff_value}. The registered estimator is refit on "
                "the full training panel, and its forward-return labels consumed prices "
                "through that date, so scoring at or before it returns a future-trained "
                "prediction disguised as a historical one. For historical evaluation use "
                f"the model's walk-forward OOS predictions ({manifest.oos_predictions_uri}) "
                "via modeling.bridge.oos_predictions_to_signal_panel, which are genuinely "
                f"out-of-sample; use score_model only for dates after training.{weaker}"
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

    # ── Universe-scope features pin the universe ──────────────────────────
    # Scoring a DIFFERENT universe than the model trained on is fine for
    # entity-scope features -- AAPL's RSI doesn't change because MSFT was
    # added to the request. It is NOT fine for a UNIVERSE-scope feature:
    # factors.pca_loading / pca_factor_return are computed from the whole
    # universe's return matrix, so [AAPL, MSFT, NVDA] and [AAPL, XOM, JPM]
    # produce a completely different PCA basis. The estimator would receive
    # a different variable under the same feature column, with nothing in
    # the result indicating the input had been redefined.
    trained_universe = list(original_spec_dict.get("universe") or [])
    universe_scope_features = []
    for feature_entry in original_spec_dict.get("features") or []:
        feature_id = feature_entry.get("id")
        try:
            definition = get_feature(feature_id)
        except Exception:
            # An unresolvable feature is a separate failure that
            # build_dataset below reports properly; don't mask it here.
            continue
        if definition.scope is FeatureScope.UNIVERSE:
            universe_scope_features.append(feature_id)

    if universe_scope_features and sorted(universe) != sorted(trained_universe):
        raise ValidationError(
            f"score_model: this model uses universe-scope feature(s) "
            f"{sorted(set(universe_scope_features))}, which are computed from the "
            f"ENTIRE universe's return matrix, so the scoring universe must match "
            f"the training universe exactly. Trained on "
            f"{sorted(trained_universe)}, asked to score {sorted(universe)}. "
            "Scoring a different set would feed the estimator a different "
            "factor basis under the same feature name — a silently different "
            "variable, not a smaller sample. Score the training universe, or "
            "train a new model on the universe you want to score."
        )

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

    # ── One cross-section, one date ───────────────────────────────────────
    # `latest` is each entity's OWN most recent surviving row, which is not
    # necessarily the same date for every entity: a symbol that stopped
    # trading, halted, or simply has a shorter history contributes an older
    # bar. Returning those together silently mixed observation dates into
    # something reported as a single as_of cross-section -- and for a
    # cross-sectional model that is not a smaller cross-section, it is a
    # ranking that no longer compares contemporaneous information.
    #
    # missing_entities never caught this: it only saw entities with NO row
    # at all, so a stale one looked like a fully successful score.
    #
    # effective_score_date is the most recent date actually available, which
    # is also (deliberately) not assumed equal to as_of -- a holiday, or a
    # provider whose window ends earlier, legitimately moves it earlier.
    effective_ts = pd.Timestamp(latest["date"].max())
    stale_mask = pd.to_datetime(latest["date"]) < effective_ts
    stale_entities = {
        str(row_entity): pd.Timestamp(row_date).strftime("%Y-%m-%d")
        for row_entity, row_date in zip(
            latest.loc[stale_mask, "entity"], latest.loc[stale_mask, "date"]
        )
    }
    latest = latest.loc[~stale_mask]

    # ── How old is that shared date? ──────────────────────────────────────
    # Enforcing ONE cross-section date makes every returned prediction
    # internally consistent, but says nothing about how old that date is: a
    # universe whose data stopped six months ago still yields a perfectly
    # uniform -- and entirely stale -- cross-section, which previously came
    # back looking like a completely successful score.
    #
    # Always reported, so the gap is visible whether or not a limit was
    # requested; only enforced when the caller states one, because how much
    # staleness is still decision-useful is a property of the strategy, not
    # something this function can pick on their behalf.
    staleness_days = int((as_of_ts - effective_ts).days)
    if max_staleness_days is not None and staleness_days > max_staleness_days:
        raise ValidationError(
            f"score_model: the newest available observation is "
            f"{effective_ts.strftime('%Y-%m-%d')}, {staleness_days} calendar days "
            f"before as_of {as_of!r}, exceeding max_staleness_days="
            f"{max_staleness_days}. Every entity agrees on that date, so this is "
            "not a per-symbol gap — the whole universe's data ends there. Check "
            "the provider window and that these symbols still trade, or raise "
            "max_staleness_days if a prediction this old is still useful."
        )

    # Stale entities are reported separately rather than folded into
    # missing_entities: "no data at all" and "data, but from an older bar"
    # are different conditions with different fixes, and collapsing them
    # would hide which one actually happened.
    missing_entities = sorted(
        set(universe) - set(latest["entity"]) - set(stale_entities)
    )

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
        # The date the predictions were actually computed from, which is not
        # necessarily the date that was asked for.
        "effective_score_date": effective_ts.strftime("%Y-%m-%d"),
        "staleness_days": staleness_days,
        "predictions_uri": predictions_uri,
        "n_entities": int(len(predictions_df)),
        "missing_entities": missing_entities,
        "stale_entities": stale_entities,
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
