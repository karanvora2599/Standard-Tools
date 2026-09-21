"""
The portfolio simulator, on a REFERENCE rather than a model id.

WHY THE SAME SIMULATION NEEDED A SECOND DOOR. `evaluate_model_portfolio`
takes a `model_id` and nothing else, so an ensemble -- the whole point of
`build_model_ensemble` -- could be correlated, described and published and
then never traded: passing its reference fails at identifier validation
before anything reads a frame. The same is true of an externally computed
alpha and of a converted panel. Everything below the manifest lookup in
`portfolio_eval.py` was already frame-only, which is why this is a door
and not an implementation: `_simulate_predictions_portfolio` runs for both
entry points, so a reference and a registered model reach byte-identical
weights.

What the manifest supplied, and what stands in for it here:

    task                 -> a required input, because guessing it is how a
                            classifier's probabilities become an
                            all-long book that looks like a portfolio
    validation_method    -> the cpcv refusal, by the `path` column that
                            shape carries
    predictions + digest -> the reference, and the producer recorded on
                            its sidecar, in `provenance`. There is no
                            registered hash to check against here; the
                            handoff store is the root of trust and the
                            result says so rather than implying a
                            verification that did not happen
    dataset spec         -> `dataset_id` to inherit the five price fields,
                            or all four required ones explicitly

`manifest.distribution` was never read by this path: `scale_by_uncertainty`
reads the frame's own `lower`/`upper` columns, so a reference carrying them
sizes by interval width exactly as a registered model's artifact does.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from standard_quant_tools.error import ValidationError

from .. import artifacts as _artifacts
from ..dataset.builder import dataset_spec_hash
from ..portfolio_eval import (
    evaluate_predictions_portfolio as _evaluate_predictions_portfolio,
)
from ..specs import DatasetSpec
from .portfolio_models import (
    EvaluatePredictionsPortfolioInput,
    EvaluatePredictionsPortfolioResult,
)

logger = logging.getLogger(__name__)

#: The tool description, here so the module that registers the tool does
#: not have to restate what this door is for.
EVALUATE_PREDICTIONS_PORTFOLIO_DESCRIPTION = (
    "Evaluate ANY published predictions reference as a shared-cash "
    "portfolio: transform them into target weights and simulate them with "
    "costs, returning Sharpe, drawdown, turnover and exposure. The same "
    "simulator evaluate_model_portfolio runs, reached by reference instead "
    "of by model_id -- which is what makes an ensemble from "
    "build_model_ensemble, an externally computed alpha, or a converted "
    "panel tradeable rather than only describable. `task` is REQUIRED and "
    "is the one thing a reference cannot tell you: a classifier's "
    "probabilities are all positive, so read as regression scores they "
    "produce a long-everything book that simulates cleanly and means "
    "nothing. Prices come from `dataset_id` (inheriting its interval, "
    "provider, calendar, start and end) or from those fields given "
    "explicitly. Unlike evaluate_model_portfolio there is no registered "
    "content hash behind the predictions -- the reference and its producer "
    "are recorded in provenance instead, and a tampered copy is not "
    "detectable here. For a registered model, prefer evaluate_model_"
    "portfolio."
)

#: The fields that have no default and no manifest to come from. `calendar`
#: is deliberately not among them: absent, the annualization falls back to
#: 252 with a warning, which is a stated approximation rather than a
#: guess about which venue these entities trade on.
_REQUIRED_EXPLICIT = ("interval", "provider", "start_date", "end_date")


def _dataset_spec_fields(dataset_id: str) -> Dict[str, Optional[str]]:
    """
    The five price fields off a built dataset's own spec.

    Verified against the hash recorded when the dataset was built, for the
    reason run_model_experiment verifies it: an edited `start`, `interval`
    or `provider` changes which bars this book is priced against, and a
    silently different price series produces a clean equity curve for a
    portfolio nobody could have held.
    """
    directory = _artifacts.run_dir(dataset_id)
    meta_path = directory / "dataset_meta.json"
    spec_path = directory / "dataset_spec.json"
    if not spec_path.exists():
        raise ValidationError(
            f"no dataset_spec.json for dataset_id={dataset_id!r}"
            + (
                " -- dataset_meta.json is written last, so its absence also "
                "means a previous build_model_dataset call did not complete."
                if not meta_path.exists()
                else "."
            )
            + " Pass a dataset built by build_model_dataset, or give "
            "interval, provider, start_date and end_date explicitly instead."
        )

    spec_dict = _artifacts.load_json(str(spec_path))
    if meta_path.exists():
        meta = _artifacts.load_json(str(meta_path))
        stored = meta.get("spec_hash")
        if stored is not None:
            version = int(meta.get("spec_hash_version", 1))
            actual = dataset_spec_hash(DatasetSpec(**spec_dict), version=version)
            if actual != stored:
                raise ValidationError(
                    f"dataset {dataset_id!r}: dataset_spec.json no longer "
                    f"matches the hash recorded when it was built (expected "
                    f"{stored}, found {actual}, hash version {version}). The "
                    "price window, provider and interval this simulation "
                    "would fetch are read from that file, so the equity "
                    "curve would be priced against bars the dataset never "
                    "used -- rebuild the dataset, or give the five fields "
                    "explicitly if that is what you mean to do."
                )

    missing = [key for key in ("start", "end") if not spec_dict.get(key)]
    if missing:
        raise ValidationError(
            f"dataset {dataset_id!r} records no {missing} in its spec, and "
            "the simulation needs a price window. Give start_date and "
            "end_date explicitly."
        )
    return {
        "interval": str(spec_dict.get("interval", "1d")),
        "provider": str(spec_dict.get("provider", "yfinance")),
        "calendar": (str(spec_dict["calendar"]) if spec_dict.get("calendar") else None),
        "start": str(spec_dict["start"]),
        "end": str(spec_dict["end"]),
    }


def _price_fields(
    input_data: EvaluatePredictionsPortfolioInput,
) -> Dict[str, Optional[str]]:
    """Which bars to price the book against: inherited, overridden, or
    refused by name when neither source exists."""
    if input_data.dataset_id is None:
        absent = [
            field for field in _REQUIRED_EXPLICIT if getattr(input_data, field) is None
        ]
        if absent:
            raise ValidationError(
                "evaluate_predictions_portfolio does not know which bars to "
                f"price this book against: {absent} were not given and there "
                "is no dataset_id to inherit them from. A predictions "
                "reference carries no dataset, so neither is a default this "
                "tool can pick -- the wrong provider or window would simulate "
                "cleanly against the wrong prices. Pass dataset_id=<the "
                "dataset these predictions were made on> to inherit interval, "
                "provider, calendar, start and end from it, or pass all of "
                "interval, provider, start_date and end_date. `calendar` "
                "stays optional: without it an intraday interval is "
                "annualized with 252 bars and the result says so."
            )
        fields: Dict[str, Optional[str]] = {
            "interval": input_data.interval,
            "provider": input_data.provider,
            "calendar": input_data.calendar,
            "start": input_data.start_date,
            "end": input_data.end_date,
        }
        return fields

    fields = _dataset_spec_fields(input_data.dataset_id)
    # An explicit field OVERRIDES the inherited one rather than conflicting
    # with it: pricing a model's predictions on a longer window, or through
    # a different provider, is a legitimate thing to ask for, and refusing
    # the combination would mean restating all five to change one.
    for field, key in (
        ("interval", "interval"),
        ("provider", "provider"),
        ("calendar", "calendar"),
        ("start_date", "start"),
        ("end_date", "end"),
    ):
        value = getattr(input_data, field)
        if value is not None:
            fields[key] = value
    return fields


def evaluate_predictions_portfolio(
    input_data: EvaluatePredictionsPortfolioInput,
) -> EvaluatePredictionsPortfolioResult:
    """Simulate a published predictions reference as a shared-cash portfolio."""
    from standard_quant_tools.agent.runtimes import handoff

    frame = handoff.resolve(input_data.predictions_ref, expect="predictions")
    # Described BEFORE the simulation, not after: the producer is part of
    # what this result claims, and discovering the sidecar is unreadable
    # after a minute of price fetching and simulating would throw the
    # answer away for a field nobody can add later.
    described: Dict[str, Any] = handoff.describe(input_data.predictions_ref)
    fields = _price_fields(input_data)

    result = _evaluate_predictions_portfolio(
        frame,
        input_data.task,
        interval=str(fields["interval"]),
        provider_name=str(fields["provider"]),
        calendar=fields["calendar"],
        start=str(fields["start"]),
        end=str(fields["end"]),
        transform=input_data.transform,
        portfolio=input_data.portfolio,
        run_id=input_data.run_id,
        source=input_data.predictions_ref,
    )

    producer = described.get("producer")
    result["provenance"] = {
        "source_ref": input_data.predictions_ref,
        "producer": producer,
        # Named `source_content_hash`, not `oos_predictions_hash`: it is
        # the digest of what was read just now, not a digest recorded when
        # something was registered and checked against afterwards. The two
        # are different claims and the key says which one this is.
        "source_content_hash": described.get("content_hash")
        or described.get("fingerprint"),
        "dataset_id": input_data.dataset_id,
        **result["provenance"],
    }
    if not producer:
        result["warnings"].append(
            f"{input_data.predictions_ref} records no producer on its "
            "sidecar, so what computed these predictions is not written "
            "down anywhere in this result. Publish with producer=<tool or "
            "model that made them> if this track record is going to be read "
            "later by someone who was not there."
        )
    logger.debug(
        "[evaluate_predictions_portfolio] ref=%s  producer=%s  dataset=%s",
        input_data.predictions_ref,
        producer,
        input_data.dataset_id,
    )
    return EvaluatePredictionsPortfolioResult(**result)


__all__ = [
    "EVALUATE_PREDICTIONS_PORTFOLIO_DESCRIPTION",
    "EvaluatePredictionsPortfolioInput",
    "EvaluatePredictionsPortfolioResult",
    "evaluate_predictions_portfolio",
]
