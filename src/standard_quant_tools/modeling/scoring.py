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

survival_curves: the same load, the same gates, the same feature matrix,
and then the question a risk score cannot answer. A survival estimator's
`predict` orders the entities -- who reaches the event first -- while the
curve S(t | x) says how likely one particular entity is to still be
waiting at t. Both come off the same fitted model; only the second is a
probability, and only the second can be read against a deadline. The two
entry points share `_scoring_context` so that a curve and a score for the
same (model, date, universe) rest on one set of gates rather than two.
"""

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

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


def _deployed_preprocessing(
    manifest, model_id: str, caller: str = "score_model"
) -> Dict[str, Any]:
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
            f"{caller}: model {model_id!r} was validated under "
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


@dataclass
class _ScoringContext:
    """
    Everything a prediction needs, and nothing about what KIND of
    prediction it is: the registered model, the aligned feature matrix,
    and the record of which entities did not make it into that matrix.

    Whatever an estimator is asked for at `as_of` -- a point score, a
    band, a survival curve -- rests on exactly the same gates (the
    training-information cutoff, the feature-implementation provenance,
    the universe pins, the one-date cross-section, the staleness limit)
    and exactly the same fitted transform. Two entry points each running
    those gates in their own words is how one of them ends up enforcing
    a weaker set; they run once here instead, and each entry point does
    only the arithmetic that makes it different.
    """

    manifest: Any
    estimator: Any
    universe: List[str]
    as_of_ts: pd.Timestamp
    latest: pd.DataFrame
    X: pd.DataFrame
    effective_ts: pd.Timestamp
    staleness_days: int
    stale_entities: Dict[str, str]
    missing_entities: List[str]
    warnings: List[str]


def _scoring_context(
    model_id: str,
    as_of: str,
    universe: List[str],
    lookback_days: int = 400,
    max_staleness_days: Optional[int] = None,
    universe_policy: str = "strict",
    caller: str = "score_model",
) -> _ScoringContext:
    """
    Run every gate and build the feature matrix the registered estimator
    was fitted to consume.

    `caller` is only the name the refusals use for themselves, so a
    message written for one entry point does not tell the reader to fix
    a call they did not make. The gates themselves are identical by
    construction -- that is the whole reason this is one function.
    """
    # Entities are asset keys in canonical form, whatever spelling arrived,
    # so `missing_entities` and the panel agree on names.
    universe = canonical_universe(universe)
    if universe_policy not in ("strict", "allow"):
        raise ValidationError(
            f"{caller}: universe_policy={universe_policy!r}; expected "
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
                f"{caller}: as_of {as_of!r} is not after this model's training "
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
        preprocessing = _deployed_preprocessing(manifest, model_id, caller)
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
            f"panel, so {caller} cannot run: scoring rebuilds features "
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
                f"{caller}: the implementation of {len(drifted)} feature(s) has "
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
            f"{caller}: this model uses universe-scope feature(s) "
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
            f"{caller}: this model standardizes each feature WITHIN the "
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
            f"{caller}: no scoreable rows as of {as_of!r} for universe {universe} — "
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
            f"{caller}: the newest available observation is "
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
            f"{caller}: {len(holed)} entity row(s) carry a missing feature "
            f"after the model's preprocessing pipeline ({holed[:5]}"
            f"{'...' if len(holed) > 5 else ''}), and estimator "
            f"{manifest.estimator_type!r} does not accept missing values. "
            "The model was registered without an `impute` step; retrain with "
            "one, or score a universe whose features are complete as of "
            f"{as_of!r}."
        )

    return _ScoringContext(
        manifest=manifest,
        estimator=estimator,
        universe=universe,
        as_of_ts=as_of_ts,
        latest=latest,
        X=X,
        effective_ts=effective_ts,
        staleness_days=staleness_days,
        stale_entities=stale_entities,
        missing_entities=missing_entities,
        warnings=warnings,
    )


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
    context = _scoring_context(
        model_id,
        as_of,
        universe,
        lookback_days=lookback_days,
        max_staleness_days=max_staleness_days,
        universe_policy=universe_policy,
    )
    manifest = context.manifest
    estimator = context.estimator
    universe = context.universe
    as_of_ts = context.as_of_ts
    latest = context.latest
    X = context.X
    effective_ts = context.effective_ts
    staleness_days = context.staleness_days
    stale_entities = context.stale_entities
    missing_entities = context.missing_entities
    warnings = context.warnings

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


#: The survival estimators that can say how likely a row is to still be
#: waiting at t, not only which row goes first. Named in the refusal,
#: because "this one cannot" is half an answer.
_CURVE_ESTIMATORS = ("cox_ph", "xgboost_cox", "xgboost_aft")


def _survival_time_grid(
    estimator: Any,
    manifest: Any,
    model_id: str,
    times: Optional[Sequence[float]],
    n_times: int,
) -> "tuple[np.ndarray, Optional[np.ndarray]]":
    """
    Where to read the curve, and the baseline knots it is read against.

    An explicit `times` is taken as given -- a deadline is the caller's,
    not the model's. Otherwise the grid is `n_times` quantiles of the
    model's own baseline event times, so the points land where the
    training data actually observed events rather than spread evenly
    across a range that may be mostly empty. A Cox-family estimator
    carries those knots; an accelerated-failure-time booster carries a
    fitted distribution instead, and its grid falls back to the span of
    training durations its validation recorded (the Brier horizons on the
    manifest). With neither, there is nothing to guess from and the
    caller is asked for `times`.
    """
    knots = getattr(estimator, "baseline_times_", None)
    if knots is not None:
        knots = np.asarray(knots, dtype=float)
        if knots.size == 0:
            knots = None

    if times is not None:
        grid = np.asarray(list(times), dtype=float)
        if grid.size < 2:
            raise ValidationError(
                f"survival_curves: `times` has {grid.size} point(s); a curve "
                "needs at least two to be a curve. Pass a longer grid, or "
                "leave `times` unset to read the model's own baseline knots."
            )
        if not np.isfinite(grid).all() or (grid < 0).any():
            raise ValidationError(
                "survival_curves: every entry of `times` must be a finite, "
                "non-negative duration in the units the model's target was "
                "measured in (seconds, bars, days -- whatever the training "
                "label counted). A negative time has no survival probability."
            )
        if not (np.diff(grid) > 0).all():
            raise ValidationError(
                "survival_curves: `times` must be strictly increasing. S(t) "
                "is a non-increasing step function of t, so an out-of-order "
                "grid would return a curve that appears to rise; the order of "
                "the columns is part of the claim."
            )
        return grid, knots

    n_times = int(n_times)
    if n_times < 2:
        raise ValidationError(
            f"survival_curves: n_times={n_times}; a curve needs at least two "
            "points. Raise n_times, or pass `times` explicitly."
        )

    if knots is not None:
        # Quantiles, not a linear span: the knots ARE the observed event
        # times, so their quantiles put the grid where the durations are
        # and not in whatever empty stretch a heavy right tail leaves.
        # Duplicates collapse, because two identical columns would be two
        # identical answers charged as two.
        return np.unique(np.quantile(knots, np.linspace(0.0, 1.0, n_times))), knots

    metrics = manifest.oos_metrics or {}
    low = metrics.get("brier_horizon_min")
    high = metrics.get("brier_horizon_max")
    if low is None or high is None or not np.isfinite([low, high]).all() or high <= low:
        raise ValidationError(
            f"survival_curves: the estimator registered for model {model_id!r} "
            f"({type(estimator).__name__}) carries no baseline event times, and "
            "this model's validation recorded no horizon span to fall back on, "
            "so there is nothing from which to pick a default grid. Pass "
            "`times` explicitly -- the durations this model was trained on are "
            "in the units of its target, and only you know which deadline the "
            "decision turns on."
        )
    return np.linspace(float(low), float(high), n_times), knots


def survival_curves(
    model_id: str,
    as_of: str,
    universe: List[str],
    *,
    lookback_days: int = 400,
    max_staleness_days: Optional[int] = None,
    universe_policy: str = "strict",
    times: Optional[Sequence[float]] = None,
    n_times: int = 32,
) -> Dict[str, Any]:
    """
    S(t | x) for each entity as of a date: how likely each one is to still
    be waiting at t, beside the risk score that only ranks them.

    Everything up to the feature matrix is `score_model`'s, gate for gate
    (`_scoring_context`). What is different is the question asked of the
    fitted estimator afterwards: `predict` gives a hazard, which orders
    the cross-section and has no units, while `predict_survival_function`
    gives a probability at each of `times`, which can be read against a
    deadline. `median_survival` is the first grid point where that
    probability falls to 0.5 or below, and is None when the curve never
    crosses inside the grid -- deliberately not the grid's last point,
    which would report "we stopped looking" as "it happened here".

    Raises:
        ValidationError: the model is not a survival model; its estimator
        produces a risk and no curve; `times` is not a strictly increasing
        non-negative grid; or any gate `score_model` enforces.
    """
    manifest = load_manifest(model_id)
    if manifest.task != "survival":
        raise ValidationError(
            f"survival_curves: model {model_id!r} was trained for the "
            f"{manifest.task!r} task, which has no survival curve. S(t | x) is "
            "read off a baseline hazard, and only a survival label -- a "
            "duration together with whether its event was observed -- can "
            f"estimate one; a {manifest.task!r} model predicts a level, and "
            "there is no time axis to put it on. Use score_model for this "
            "model, or train one on a survival target (task='survival', e.g. "
            "a time_to_fill panel registered with its event_column) and ask "
            "again."
        )

    context = _scoring_context(
        model_id,
        as_of,
        universe,
        lookback_days=lookback_days,
        max_staleness_days=max_staleness_days,
        universe_policy=universe_policy,
        caller="survival_curves",
    )
    estimator = context.estimator
    if not hasattr(estimator, "predict_survival_function"):
        raise ValidationError(
            f"survival_curves: model {model_id!r} is fitted with "
            f"{type(estimator).__name__} (estimator_type "
            f"{manifest.estimator_type!r}), which produces a risk score and no "
            "survival function. The risk is NOT returned in its place: it "
            "orders the entities and is not a probability, so reading it "
            "against a deadline would put a number on a scale it does not "
            "have. The survival estimators that carry a curve are "
            f"{', '.join(_CURVE_ESTIMATORS)} -- retrain with one of those, or "
            "call score_model, which is where the ordering lives."
        )

    grid, knots = _survival_time_grid(estimator, manifest, model_id, times, n_times)
    matrix = np.asarray(
        estimator.predict_survival_function(context.X.to_numpy(), grid), dtype=float
    )
    entities = [str(entity) for entity in context.latest["entity"].to_numpy()]
    if matrix.shape != (len(entities), grid.size):
        raise ValidationError(
            f"survival_curves: {type(estimator).__name__}."
            "predict_survival_function returned a "
            f"{matrix.shape} matrix for {len(entities)} entities and "
            f"{grid.size} times. A curve tool cannot align rows to entities "
            "it cannot count, and silently reshaping would attach one name's "
            "probabilities to another."
        )
    # The same adapter score_model uses, so a curve and a score for the
    # same call carry the SAME risk numbers rather than two paths' worth.
    risk = np.asarray(
        get_adapter(manifest.task).score(estimator, context.X), dtype=float
    )

    # The first grid point at or below 0.5. `np.argmax` on the boolean
    # returns 0 for a row that never crosses, which is why the crossing
    # flag is carried separately -- a median of grid[0] and "never reached
    # 0.5" are opposite claims.
    crossed = matrix <= 0.5
    any_crossing = crossed.any(axis=1)
    first_crossing = np.argmax(crossed, axis=1)

    per_entity: List[Dict[str, Any]] = []
    for row, entity in enumerate(entities):
        per_entity.append(
            {
                "entity": entity,
                "risk": float(risk[row]),
                "survival_at_times": [float(value) for value in matrix[row]],
                "median_survival": (
                    float(grid[first_crossing[row]])
                    if bool(any_crossing[row])
                    else None
                ),
            }
        )

    warnings = list(context.warnings)
    if knots is not None:
        if float(grid[-1]) < float(knots[-1]):
            warnings.append(
                f"this grid ends at t={float(grid[-1]):.6g}, before the model's "
                f"last baseline event time ({float(knots[-1]):.6g}): every curve "
                "is TRUNCATED there. A median_survival of None therefore means "
                "the curve had not fallen to 0.5 by the end of this grid, NOT "
                "that the event never comes -- widen `times`, or leave it unset "
                "to span the baseline."
            )
        if float(grid[-1]) > float(knots[-1]):
            warnings.append(
                f"this grid extends to t={float(grid[-1]):.6g}, past the model's "
                f"last baseline event time ({float(knots[-1]):.6g}). The baseline "
                "cumulative hazard is a step function with no step out there, so "
                "every probability beyond that point is the last observed value "
                "held flat -- an extrapolation, not an estimate, and it will "
                "understate the hazard for as far as it runs."
            )
        warnings.append(
            "proportional hazards: these curves share one baseline, scaled by "
            "each entity's risk. The ORDERING between them is what the model "
            f"learned from {int(knots.size)} baseline event time(s); the LEVEL "
            "of any one of them is the baseline's, estimated on the training "
            "durations. If the base rate has moved since training -- a slower "
            "book, a wider spread regime -- the ranking can still be right "
            "while every probability is off, and nothing in this result can "
            "tell the two apart."
        )
    else:
        warnings.append(
            "this estimator carries no baseline event times: the level of every "
            "curve comes from the fitted accelerated-failure-time distribution "
            "and the scale its spec fixed, not from an empirical hazard. The "
            "shape is a modelling assumption, so read the ordering with more "
            "confidence than the probabilities."
        )
    n_never = int((~any_crossing).sum())
    if n_never:
        warnings.append(
            f"{n_never} of {len(entities)} entity curve(s) never reach 0.5 on "
            "this grid, so their median_survival is None. That is the honest "
            "answer for a horizon this grid does not cover; it is not a claim "
            "that the event does not happen."
        )

    return {
        "model_id": model_id,
        "as_of": as_of,
        # The date the curves were actually computed from, which is not
        # necessarily the date that was asked for -- see score_model.
        "effective_score_date": context.effective_ts.strftime("%Y-%m-%d"),
        "times": [float(value) for value in grid],
        "per_entity": per_entity,
        # The cross-section's average curve, which is a description of
        # this call's universe and not of the model's baseline: it moves
        # when the universe does.
        "survival_mean_curve": [float(value) for value in matrix.mean(axis=0)],
        "n_baseline_knots": int(knots.size) if knots is not None else 0,
        "n_entities": len(entities),
        "missing_entities": context.missing_entities,
        "stale_entities": context.stale_entities,
        "warnings": warnings,
    }
