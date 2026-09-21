"""
score_model: load a registered model + the exact DatasetSpec that trained
it, rebuild the same features as of a given date for a (possibly new)
universe, predict, and write predictions to a Parquet artifact via
modeling.artifacts.save_artifact.

Reuses dataset.builder.build_dataset(include_target=False) — the scoring
path deliberately skips target construction, since a forward-return
target needs `horizon` bars of future data that don't exist for "today".
Applies the SAME fitted pipeline the registered model's final refit used
(persisted by registry.model_registry.save_model). For a POOLED model
that is a set of statistics, so the same input row scores the same
whichever other tickers are in the call. For a CROSS-SECTIONAL model it
is not: the transform fits nothing and standardizes within the rows the
call contains, so every row's score depends on which other entities
were scored with it -- narrowing a trained universe of eight names to
three inverted a forest's ranking (findings D15). Such a model is
therefore pinned to its training universe unless the caller passes
universe_policy='allow', and the result then says what was refit.
"""

import hashlib
import json
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from standard_quant_tools.audit.hashing import hash_dataframe
from standard_quant_tools.error import ValidationError

from . import artifacts as _artifacts
from .adapters import accepts_missing, get_adapter
from .assets import canonical_universe
from .dataset.builder import build_dataset
from .estimators.registry import get_estimator_class
from .features.base import FeatureScope
from .features.registry import get_feature
from .features.transforms import apply_preprocessing, standardize_cross_sectional
from .preprocessing import FoldContext, apply_pipeline
from .registry.feature_provenance import feature_provenance_from_spec
from .registry.model_registry import (
    load_dataset_spec,
    load_distribution,
    load_manifest,
    load_model,
    load_model_spec,
    load_preprocessing_state,
    load_preprocessing_stats,
)


def _deployed_preprocessing(manifest, model_id: str) -> Dict[str, Any]:
    """
    The transform the deployed estimator was fitted under.

    From the manifest when it records one. A manifest written before the
    field existed is resolved from the model's own bundled spec -- but only
    when that spec says `pooled`, because that is the one transform such a
    model was certainly refit under. For a legacy spec that says
    `cross_sectional` the refit of the day did NOT honour it: the estimator
    was fitted on the pooled statistics while the folds were validated
    cross-sectionally, and no transform applied here describes a pipeline
    that was validated. Scoring it would return a number with two
    contradictory provenances, so it is refused with the remedy.
    """
    if manifest.preprocessing:
        return dict(manifest.preprocessing)
    spec = load_model_spec(model_id)
    if spec.preprocessing.normalization != "pooled":
        raise ValidationError(
            f"score_model: model {model_id!r} was validated under "
            f"preprocessing.normalization={spec.preprocessing.normalization!r} "
            "but predates the refit that honours it: its deployed estimator "
            "was fitted on the pooled statistics, so no transform applied now "
            "reproduces the pipeline its OOS metrics describe. Retrain to "
            "register a model whose deployed transform is the validated one. "
            "For historical evaluation the model's walk-forward OOS "
            "predictions remain valid."
        )
    return spec.preprocessing.model_dump()


def _deployed_is_cross_sectional(
    manifest, state: Optional[Dict[str, Any]], preprocessing: Dict[str, Any]
) -> bool:
    """
    Whether the deployed pipeline standardizes within the scoring
    cross-section: from the fitted state's steps when there is one, else
    from the manifest's resolved steps, else from the scheme name.
    """
    steps: List[str] = []
    if state is not None:
        steps = [str(s.get("type")) for s in (state.get("steps") or [])]
    elif manifest.preprocessing.get("steps"):
        steps = [str(s.get("type")) for s in manifest.preprocessing["steps"]]
    if "cross_sectional_standardize" in steps:
        return True
    scheme = (manifest.preprocessing or preprocessing or {}).get("normalization")
    return scheme == "cross_sectional"


from .specs import DatasetSpec, _parse_date


def _interval_statistics(
    predictions_df: pd.DataFrame,
) -> tuple[Dict[str, float], List[str]]:
    """
    What the conformal band a scored model emits actually looks like.

    `summary_stats` describes the POINT prediction and says nothing about
    the interval beside it, so a band thirty times the width of the whole
    cross-section -- measured, on a ridge model whose predictions ranged
    over 0.0024 while the mean interval width was 0.169 -- came back from
    scoring looking exactly like a tight one. The width is the model's own
    statement of how much it does not know, and no view returned it.

    Empty for a point-only model: an absent interval is not a zero-width
    one, and a dict of zeros would read as perfect confidence.

    There is deliberately no coverage number here. Coverage is the fraction
    of realized outcomes that fell inside the band, and at `as_of` the
    outcomes do not exist yet -- computing one against anything available
    now would mean scoring the interval on the data it was calibrated on.
    """
    if not {"lower", "upper"} <= set(predictions_df.columns):
        return {}, []

    widths = predictions_df["upper"].to_numpy(dtype=float) - predictions_df[
        "lower"
    ].to_numpy(dtype=float)
    widths = widths[np.isfinite(widths)]
    if widths.size == 0:
        return {}, []

    stats: Dict[str, float] = {
        "interval_mean_width": float(np.mean(widths)),
        "interval_median_width": float(np.median(widths)),
        "interval_min_width": float(np.min(widths)),
        "interval_max_width": float(np.max(widths)),
        "n_intervals": int(widths.size),
    }

    # The band against what it is a band AROUND. An absolute width means
    # nothing without the scale of the predictions it brackets: 0.169 is
    # narrow for a price and absurd for a daily return forecast, and the
    # ratio is what says which case this is.
    predictions = predictions_df["prediction"].to_numpy(dtype=float)
    predictions = predictions[np.isfinite(predictions)]
    warnings: List[str] = []
    spread = (
        float(np.max(predictions) - np.min(predictions)) if predictions.size else 0.0
    )
    # A one-name cross-section has a spread of exactly zero, and so does a
    # model that predicted the same number for everything. Dividing by it
    # would emit an infinity; the key is absent instead, because "the ratio
    # is undefined here" is a different claim from "the ratio is enormous".
    if np.isfinite(spread) and spread > 0:
        ratio = stats["interval_mean_width"] / spread
        if np.isfinite(ratio):
            stats["interval_width_over_prediction_spread"] = float(ratio)
            if ratio > 1.0:
                warnings.append(
                    "the prediction interval is wider than the entire "
                    f"cross-section's spread: mean width {stats['interval_mean_width']:.6g} "
                    f"against a prediction range of {spread:.6g} ({ratio:.1f}x). "
                    "Every name's band contains every other name's point "
                    "prediction, so the interval says the model cannot "
                    "distinguish them at all -- the RANKING may still be "
                    "usable, the level is not, and any position size derived "
                    "from the width will be. Check the conformal alpha and "
                    "the calibration residuals the radius came from."
                )

    return {
        key: value
        for key, value in stats.items()
        if not isinstance(value, float) or np.isfinite(value)
    }, warnings


def score_model(
    model_id: str,
    as_of: str,
    universe: List[str],
    lookback_days: int = 400,
    max_staleness_days: Optional[int] = None,
    universe_policy: str = "strict",
) -> Dict[str, Any]:
    """
    Args:
        universe_policy: 'strict' (default) refuses to score a
            cross-sectional model on a universe that is not the one it
            was trained on; 'allow' scores it and returns a warning
            saying the transform was refit on the scoring cross-section
            and how its width compares with the training one. A model
            with universe-scope features is refused either way.
        lookback_days: calendar days of history fetched before `as_of` so
            every requested feature's lookback window has enough data —
            widen this if the model's features use unusually large
            windows (e.g. a custom feature with lookback > 252 bars).

    Raises:
        ValidationError: no registered model with `model_id`, or no
        entity in `universe` has a scoreable row as of `as_of`.
    """
    # Entities are asset keys in canonical form, whatever spelling arrived,
    # so `missing_entities` and the panel agree on names.
    universe = canonical_universe(universe)
    if universe_policy not in ("strict", "allow"):
        raise ValidationError(
            f"score_model: universe_policy={universe_policy!r}; expected "
            "'strict' or 'allow'."
        )
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

    # The fitted pipeline state is the record of the deployed transform. A
    # model registered before it existed falls back to the statistics file
    # and the manifest's scheme, which is the Phase 0 stop-gap, and is
    # refused by name when neither describes a validated pipeline.
    state = load_preprocessing_state(model_id)
    stats: Dict[str, Any] = {}
    preprocessing: Dict[str, Any] = {}
    if state is None:
        stats = load_preprocessing_stats(model_id)
        preprocessing = _deployed_preprocessing(manifest, model_id)
    estimator = load_model(model_id)

    # The model's OWN bundled, content-verified copy -- not
    # SQT_RUNS_DIR/<dataset_id>/dataset_spec.json. Reading the dataset
    # directory made scoring depend on that directory surviving, and let an
    # edit there (say RSI period 14 -> 100) silently redefine the features
    # fed to an already-registered estimator, with no integrity check and
    # no change in model_id.
    original_spec_dict = load_dataset_spec(model_id)

    # A panel registered from outside carries no recipe. Every column in it
    # was computed by something this library has never seen, so the feature
    # rebuild below has nothing to rebuild FROM -- it would look up ids like
    # "ofi_100ms" in FEATURE_REGISTRY, find nothing, and fail several frames
    # deeper with a message about an unknown feature rather than about the
    # actual situation. Refused here, by name, with the thing to do instead.
    if original_spec_dict.get("provider") == "external":
        raise ValidationError(
            f"model {model_id!r} was trained on an externally registered "
            "panel, so score_model cannot run: scoring rebuilds features "
            "from the model's bundled spec, and these features were "
            "computed outside this library. Register a panel covering the "
            "scoring window with register_external_panel, predict with the "
            "estimator directly, and use score_predictions -- which was "
            "built for predictions this library never produced."
        )

    # ── Feature implementations must still be the trained ones ────────────
    # The manifest recorded each column's implementation hash at
    # registration, but nothing compared it against today's code. So
    # editing a feature function and then scoring an existing model fed the
    # registered estimator a differently-defined input under the same
    # column name, with the provenance only recording -- after the fact,
    # for anyone who went looking -- that something had changed.
    #
    # Scoped honestly: this catches a change to the FEATURE FUNCTION
    # itself, which is the case the field exists for (especially a custom
    # feature registered at runtime from outside the repo). It does NOT
    # catch a rewrite of a shared primitive the wrapper calls -- editing
    # indicators.momentum.rsi leaves _technical_rsi's source, and therefore
    # this hash, identical. git_commit_sha/package_version in the manifest
    # are the coarser signal for that.
    recorded_provenance = manifest.feature_provenance or {}
    if recorded_provenance:
        current = feature_provenance_from_spec(original_spec_dict.get("features"))
        drifted = {
            column: (
                record.get("implementation_hash"),
                current[column].get("implementation_hash"),
            )
            for column, record in recorded_provenance.items()
            if column in current
            and record.get("implementation_hash") not in (None, "unavailable")
            and current[column].get("implementation_hash")
            != record.get("implementation_hash")
        }
        if drifted:
            detail = ", ".join(
                f"{column} (trained {was}, now {now})"
                for column, (was, now) in drifted.items()
            )
            raise ValidationError(
                f"score_model: the implementation of {len(drifted)} feature(s) has "
                f"changed since model {model_id!r} was trained: {detail}. The "
                "registered estimator learned coefficients against the OLD "
                "definition, so scoring now would feed it a differently-computed "
                "input under the same column name and return a prediction that "
                "looks valid but is not the model that was validated. Retrain "
                "against the current feature code, or score with the code the "
                f"model was trained on (git_commit_sha {manifest.git_commit_sha})."
            )

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

    # ── A cross-sectional model is pinned to its universe too ─────────
    # `cross_sectional_standardize` fits nothing: it standardizes within
    # the rows of the call. The module docstring promised that the same
    # input row scores the same whichever other tickers are in the
    # call, and for such a model that is false by construction --
    # narrowing eight trained names to three moved one row's score by
    # 544% and inverted the ranking (findings D15). The pin above fired
    # only for universe-scope features; this one fires for the
    # transform, in the same voice, and can be waived by name.
    cross_sectional = _deployed_is_cross_sectional(manifest, state, preprocessing)
    universe_differs = sorted(universe) != sorted(trained_universe)
    warnings: List[str] = []
    if cross_sectional and universe_differs and universe_policy == "strict":
        raise ValidationError(
            "score_model: this model standardizes each feature WITHIN the "
            "scoring cross-section (preprocessing step "
            "cross_sectional_standardize), so every row's score depends on "
            "which other entities are scored with it -- a subset is not a "
            "smaller sample but a different transform. Trained on "
            f"{sorted(trained_universe)}, asked to score {sorted(universe)}. "
            "Score the training universe, train a new model on the universe "
            "you want to score, or pass universe_policy='allow' to refit the "
            "transform on this cross-section and receive a warning saying so."
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
    if cross_sectional and universe_differs:
        width = manifest.training_cross_section or {}
        trained_width = (
            f"median {width['median']:g} (min {width['min']}, max {width['max']})"
            if width
            else "unrecorded for this model"
        )
        warnings.append(
            f"cross-sectional transform refit on the scoring cross-section of "
            f"{len(latest)} entities; the training cross-section was {trained_width} "
            "entities per date. Every returned score depends on which entities "
            "were in this call, and is not comparable with a score of the same "
            "entity from a call with a different universe."
        )

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

    # The transform the deployed estimator was validated and refit under.
    # A cross-sectional model standardizes within the scoring date's own
    # cross-section -- `latest` is exactly one date after the stale filter
    # above -- which is the contemporaneous information the folds used and
    # a live model also has. A pooled model applies the persisted
    # statistics. Applying the pooled statistics to both was the
    # deployment mismatch the engine's refit comment records.
    if state is not None:
        X = apply_pipeline(
            state, latest[manifest.feature_ids], FoldContext.from_frame(latest)
        )
    elif preprocessing.get("normalization") == "cross_sectional":
        X = standardize_cross_sectional(
            latest[manifest.feature_ids],
            latest["date"].to_numpy(),
            float(preprocessing.get("clip_sigma", 3.0)),
        )
    else:
        X = apply_preprocessing(latest[manifest.feature_ids], stats)
    # A hole the pipeline left, for an estimator that cannot take one, is
    # refused here with the step that would close it rather than several
    # frames down inside sklearn -- the same check the engine makes per fold.
    if np.isnan(X.to_numpy(dtype=np.float64)).any() and not accepts_missing(
        get_estimator_class(manifest.task, manifest.estimator_type)
    ):
        holed = latest.loc[X.isna().any(axis=1).to_numpy(), "entity"].tolist()
        raise ValidationError(
            f"score_model: {len(holed)} entity row(s) carry a missing feature "
            f"after the model's preprocessing pipeline ({holed[:5]}"
            f"{'...' if len(holed) > 5 else ''}), and estimator "
            f"{manifest.estimator_type!r} does not accept missing values. "
            "The model was registered without an `impute` step; retrain with "
            "one, or score a universe whose features are complete as of "
            f"{as_of!r}."
        )
    # Through the SAME adapter the folds and the deployed refit used. This
    # was a two-way branch -- regression got `predict`, everything else got
    # `positive_class_proba` -- written when those were the only two tasks.
    # A ranker is neither: LGBMRanker and XGBRanker have no predict_proba
    # at all, so a registered ranking model trained, validated, and then
    # failed here with an AttributeError from inside the library. The
    # adapter already answers "how do I get a score out of this task's
    # estimator" for the engine; scoring asking the question its own way
    # is how the two drift.
    predictions = get_adapter(manifest.task).score(estimator, X)

    # The distribution the model was registered with, emitted under the
    # same columns the validation reported on: one per quantile level, and
    # `lower`/`upper` from the deployed conformal radius. Empty for a
    # point-only model, so its frame is exactly what it was.
    distribution, quantile_models = load_distribution(model_id)
    extra_columns: Dict[str, Any] = {}
    for column, quantile_model in quantile_models.items():
        extra_columns[column] = np.asarray(quantile_model.predict(X.to_numpy()))
    conformal = distribution.get("conformal") or None
    if conformal:
        radius = float(conformal["radius"])
        extra_columns["lower"] = np.asarray(predictions, dtype=float) - radius
        extra_columns["upper"] = np.asarray(predictions, dtype=float) + radius

    predictions_df = pd.DataFrame(
        {
            "entity": latest["entity"].to_numpy(),
            "date": latest["date"].to_numpy(),
            "prediction": predictions,
            **extra_columns,
        }
    )
    # The artifact name includes a digest of the scored universe, not just
    # the date. With `predictions_YYYYMMDD` alone, scoring [AAPL, MSFT] and
    # then [AAPL, NVDA] for the same as_of overwrote the first file, so an
    # audit record written by the earlier call pointed at contents produced
    # by the later one -- a silently wrong provenance trail rather than a
    # missing one. overwrite=True is kept so re-scoring the SAME universe
    # on the same date is idempotent.
    # 16 hex chars, matching the digest length used everywhere else in this
    # package (audit/hashing.py, artifacts.hash_file). The previous 8 was
    # only 32 bits -- fine for making a filename unique-ish, too short to
    # lean on when it is part of an artifact's identity.
    universe_digest = hashlib.sha256(
        json.dumps(sorted(universe)).encode("utf-8")
    ).hexdigest()[:16]
    # Content-addressed, so a written artifact is IMMUTABLE.
    #
    # The name previously covered only (date, universe) and was written with
    # overwrite=True, so re-scoring the same model/date/universe -- after a
    # provider revised its data, say -- replaced the file in place. An audit
    # record written by the earlier call still pointed at that URI, which
    # now returned different bytes: a silently wrong provenance trail rather
    # than a missing one, and the harder kind to notice because the link
    # still resolves.
    #
    # Including the content digest means identical re-scores resolve to the
    # same path (idempotent, no file proliferation) while any change in the
    # predictions produces a new path, leaving the old one intact for
    # whoever recorded it.
    content_digest = hash_dataframe(predictions_df)
    run_name = (
        f"predictions_{as_of_ts.strftime('%Y%m%d')}_{universe_digest}_{content_digest}"
    )
    predictions_uri = _artifacts.save_artifact(
        predictions_df, run_id=model_id, name=run_name, overwrite=True
    )
    # The RAW feature rows these predictions were made from, beside them
    # under the same suffix, so `monitor_model` can ask whether the inputs
    # the model was handed today look like the inputs it was trained on.
    features_df = latest[["entity", "date", *manifest.feature_ids]].reset_index(
        drop=True
    )
    features_uri = _artifacts.save_artifact(
        features_df,
        run_id=model_id,
        name=run_name.replace("predictions_", "features_", 1),
        overwrite=True,
    )

    # Appended to `warnings` before it is returned, so an implausible band
    # is read beside the number it makes implausible.
    interval_stats, interval_warnings = _interval_statistics(predictions_df)
    warnings.extend(interval_warnings)

    return {
        "model_id": model_id,
        "as_of": as_of,
        "features_uri": features_uri,
        # The date the predictions were actually computed from, which is not
        # necessarily the date that was asked for.
        "effective_score_date": effective_ts.strftime("%Y-%m-%d"),
        "staleness_days": staleness_days,
        "predictions_uri": predictions_uri,
        # Returned so a caller (or an audit record) can assert later that
        # the file at predictions_uri is still the one this call produced,
        # without re-deriving it from the frame.
        "predictions_hash": content_digest,
        "n_entities": int(len(predictions_df)),
        "missing_entities": missing_entities,
        "stale_entities": stale_entities,
        "warnings": warnings,
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
        # The band BESIDE the point prediction, which summary_stats above
        # describes and which said nothing about the interval. Empty for a
        # point-only model -- see _interval_statistics.
        "interval_stats": interval_stats,
    }
