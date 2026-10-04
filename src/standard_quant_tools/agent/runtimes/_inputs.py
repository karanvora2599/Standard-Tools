"""
Building a tool's input from the arguments a caller sent, with a refusal that
says where a misplaced argument belongs.

WHY THE REFUSAL IS REWRITTEN. Every input model forbids an argument it does
not take, which is right: a hallucinated name must not run on defaults. But
pydantic's text for that is "Extra inputs are not permitted", and for an
absent one "Field required". Neither says what the tool does take, or which
tool takes the argument that was sent. In one live modeling session seven
first calls were refused that way, and five of them were made to a tool whose
sibling takes exactly what was sent -- `get_dataset_metadata(dataset_id=...)`
for a dataset the modeling runtime built, `get_feature_drift` without a
feature where `screen_feature_stability` takes the dataset alone -- with
nothing in the refusal pointing there.

WHAT IS KEPT. The exception is still a `pydantic.ValidationError` with the
same title and the same errors in the same order, each with its `type` and
`loc` unchanged, so anything that branches on them -- `validate_tool_call`'s
missing/unknown/invalid classification, an MCP client, Carbon -- sees the
same structure. Only two messages change: an `extra_forbidden` error says
what the tool takes, the nearest argument name, and which tools take the
argument instead; a `missing` error that has a curated sibling names it. An
accepted call is constructed exactly as before, inside one extra function
call.

The hints are derived where they can be -- which tools take `dataset_id` is
read off the dispatch tables, so a tool added tomorrow is named tomorrow --
and curated where the right answer is a judgement (`score_model` without
`as_of` most likely wanted `inspect_model`). A test checks that every name a
curated hint uses is a real tool or argument. See the CHANGELOG entry of
2026-10-04.
"""

from __future__ import annotations

import difflib
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Type

import pydantic
from pydantic_core import PydanticCustomError

__all__ = [
    "EXTRA_HINTS",
    "ID_ARGUMENTS",
    "MISSING_HINTS",
    "build_input",
    "explain_refusal",
]

#: Arguments that name material another tool reads: an id, a reference, a
#: ticker. Sent to a tool that does not take one, the useful answer is which
#: tools do, because the caller is holding the value and looking for its
#: door.
ID_ARGUMENTS = frozenset(
    {"dataset_id", "model_id", "model_ids", "predictions_ref", "symbol", "ref"}
)

#: How many tools a "taken by" sentence names before counting the rest.
_TAKERS_SHOWN = 5

#: How many optional arguments "It takes ..." names before counting the
#: rest. The largest input model has 31 fields, and a refusal that lists
#: all of them buries the one sentence the caller needed.
_OPTIONAL_SHOWN = 12

_INSPECT_DATASET = (
    "inspect_dataset(dataset_id) in the modeling runtime describes a dataset "
    "built by build_model_dataset or registered by register_external_panel."
)
_PBO_IS_FOR_CONFIGURATIONS = (
    "PBO compares two or more configurations, so trial_returns takes one "
    "return series per configuration, and one model is one configuration. "
    "compare_models(model_ids=[...]) in the modeling runtime compares "
    "registered models, and method='paired' tests whether the gap exceeds "
    "the noise."
)

#: (tool, argument it does not take) -> what the caller most likely wanted.
EXTRA_HINTS: Dict[Tuple[str, str], str] = {
    ("get_dataset_metadata", "dataset_id"): (
        "get_dataset_metadata reports what a data provider guarantees for one "
        "symbol; it does not read a dataset. " + _INSPECT_DATASET
    ),
    ("estimate_backtest_overfitting", "model_id"): _PBO_IS_FOR_CONFIGURATIONS,
    ("estimate_backtest_overfitting", "model_ids"): _PBO_IS_FOR_CONFIGURATIONS,
    ("inspect_model", "dataset_id"): (
        "inspect_model describes a registered model; " + _INSPECT_DATASET
    ),
    ("list_datasets", "dataset_id"): (
        "list_datasets lists every dataset; " + _INSPECT_DATASET
    ),
    ("score_model", "dataset_id"): (
        "score_model fetches new bars for universe through as_of and scores "
        "them with a registered model. The model's out-of-sample metrics on "
        "the dataset it was trained on need no fetch: they are in "
        "inspect_model(model_id)."
    ),
}

_SCREEN_STABILITY = (
    "screen_feature_stability takes dataset_id alone and returns these "
    "numbers for every feature of the dataset."
)

#: (tool, required argument left out) -> the sibling that does not need it.
#: Each single-feature feature_lab tool has an all-feature counterpart, and
#: a call without `feature` is usually a request for that counterpart.
MISSING_HINTS: Dict[Tuple[str, str], str] = {
    ("get_feature_drift", "feature"): _SCREEN_STABILITY,
    ("get_feature_regime_stability", "feature"): _SCREEN_STABILITY,
    ("run_feature_permutation_test", "feature"): (
        "screen_feature_significance takes dataset_id alone and tests every "
        "feature of the dataset against the same kind of null."
    ),
    ("get_feature_ic_decay", "feature"): (
        "For every feature at once, analyze_features in the modeling runtime "
        "returns this curve per feature under report.leakage."
    ),
    ("profile_feature", "feature"): (
        "analyze_features in the modeling runtime scores every feature of a "
        "dataset at once."
    ),
    ("score_model", "as_of"): (
        "score_model fetches bars for every name in universe through as_of "
        "and predicts that date. A registered model's out-of-sample metrics "
        "need no fetch: they are in inspect_model(model_id)."
    ),
}
MISSING_HINTS[("score_model", "universe")] = MISSING_HINTS[("score_model", "as_of")]


def build_input(
    tool_name: str, model_cls: Type[pydantic.BaseModel], arguments: Mapping[str, Any]
) -> pydantic.BaseModel:
    """`model_cls(**arguments)`, with a refusal that says where to go.

    Raises the same `pydantic.ValidationError` the constructor raises --
    same title, same error types and locations -- with the message of an
    unknown or a missing argument rewritten (see the module docstring).
    """
    try:
        return model_cls(**arguments)
    except pydantic.ValidationError as exc:
        explained = explain_refusal(tool_name, model_cls, exc)
        if explained is exc:
            raise
        raise explained from None


def explain_refusal(
    tool_name: str,
    model_cls: Type[pydantic.BaseModel],
    exc: pydantic.ValidationError,
) -> pydantic.ValidationError:
    """`exc` with its unknown- and missing-argument messages rewritten, or
    `exc` itself when there is nothing to add.

    Never raises: a fault in composing a hint must not turn a refusal into a
    crash, so any failure here returns the original exception untouched.
    """
    try:
        errors = exc.errors()
        lines, changed = _rewritten(tool_name, model_cls, errors)
        if not changed:
            return exc
        return pydantic.ValidationError.from_exception_data(
            exc.title,
            lines,
            hide_input=bool(model_cls.model_config.get("hide_input_in_errors")),
        )
    except Exception:  # noqa: BLE001 - the original refusal is still correct
        return exc


def _rewritten(
    tool_name: str, model_cls: Type[pydantic.BaseModel], errors: Sequence[Dict]
) -> Tuple[List[Any], bool]:
    fields = list(model_cls.model_fields)
    lines: List[Any] = []
    changed = False
    described = False
    hinted = set()
    for error in errors:
        loc = error["loc"]
        key = loc[0] if len(loc) == 1 and isinstance(loc[0], str) else None
        if error["type"] == "extra_forbidden" and key is not None:
            parts = [f"{tool_name} takes no argument {key!r}."]
            # What the tool takes is said once, on the first unknown
            # argument, rather than repeated under each.
            if not described:
                parts.append(_what_it_takes(model_cls))
                described = True
            near = _nearest(key, fields)
            if near:
                parts.append(f"Did you mean {' or '.join(near)}?")
            hint = EXTRA_HINTS.get((tool_name, key))
            if hint is None and key in ID_ARGUMENTS and not near:
                hint = _takers(tool_name, key)
            if hint and hint not in hinted:
                parts.append(hint)
                hinted.add(hint)
            lines.append(_custom("extra_forbidden", " ".join(parts), error))
            changed = True
        elif (
            error["type"] == "missing"
            and (tool_name, key) in MISSING_HINTS
            and MISSING_HINTS[(tool_name, key)] not in hinted
        ):
            hint = MISSING_HINTS[(tool_name, key)]
            hinted.add(hint)
            # Pydantic's own words first, so text that matched on "Field
            # required" still does.
            lines.append(_custom("missing", f"Field required. {hint}", error))
            changed = True
        else:
            lines.append(error)
    return lines, changed


def _custom(kind: str, message: str, error: Dict) -> Dict[str, Any]:
    """One error line of the same type and location with a new message.

    No context is passed, so the message is not treated as a template and a
    brace in an argument name is printed as written.
    """
    return {
        "type": PydanticCustomError(kind, message),
        "loc": error["loc"],
        "input": error.get("input"),
    }


def _what_it_takes(model_cls: Type[pydantic.BaseModel]) -> str:
    required = [n for n, f in model_cls.model_fields.items() if f.is_required()]
    optional = [n for n, f in model_cls.model_fields.items() if not f.is_required()]
    sentence = "It takes " + (", ".join(required) or "no required argument")
    if optional:
        shown = ", ".join(optional[:_OPTIONAL_SHOWN])
        rest = len(optional) - _OPTIONAL_SHOWN
        if rest > 0:
            shown += f" and {rest} more, which describe_tool lists"
        sentence += f" (optional: {shown})"
    return sentence + "."


def _nearest(key: str, fields: Sequence[str]) -> List[str]:
    """Argument names the caller probably meant: close spellings, then a
    name that is the start of the one sent (`as_of_date` for `as_of`)."""
    near = difflib.get_close_matches(key, fields, n=2, cutoff=0.75)
    for name in fields:
        if name in near or len(name) < 4 or len(key) < 4:
            continue
        if key.startswith(name) or name.startswith(key):
            near.append(name)
    return near[:2]


def _takers(tool_name: str, key: str) -> Optional[str]:
    """Which tools take `key`: in this tool's own runtime when any do, else
    which runtimes have one. Read off the dispatch tables, so it cannot name
    a tool that does not exist."""
    from standard_quant_tools.agent.runtimes import all_runtimes, owner_of

    runtimes = all_runtimes()
    home = owner_of(tool_name)
    if home is not None:
        same = sorted(
            name
            for name, (_fn, model) in runtimes[home].dispatch_table.items()
            if name != tool_name and key in model.model_fields
        )
        if same:
            shown = ", ".join(same[:_TAKERS_SHOWN])
            if len(same) > _TAKERS_SHOWN:
                shown += f" and {len(same) - _TAKERS_SHOWN} more"
            return f"In the {home} runtime, {key} is taken by {shown}."
    elsewhere = sorted(
        name
        for name, runtime in runtimes.items()
        if name != home
        and any(
            key in model.model_fields for _fn, model in runtime.dispatch_table.values()
        )
    )
    if not elsewhere:
        return None
    where = " and ".join(
        [", ".join(elsewhere[:-1]), elsewhere[-1]] if len(elsewhere) > 1 else elsewhere
    )
    plural = "runtimes have tools" if len(elsewhere) > 1 else "runtime has tools"
    scope = f"No {home} tool" if home is not None else "No tool here"
    return f"{scope} takes {key}; the {where} {plural} that do."
