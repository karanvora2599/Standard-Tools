"""
modeling_dispatch: routes an LLM tool call to the correct modeling tool
function, mirroring agent.tools.dispatch() exactly — but over
MODELING_TOOL_DISPATCH, never the 46-entry _TOOL_DISPATCH. Reuses
audit._run_and_record as-is, so every modeling tool call is still
audit-logged (ModelSpec/DatasetSpec hashes ride in the existing
DecisionRecord.input payload) without a parallel audit implementation.
"""

from typing import Any, Dict

from standard_quant_tools._jsonsafe import sanitize_for_json
from standard_quant_tools.audit.dispatch import _run_and_record

from .tools import MODELING_TOOL_DISPATCH


def modeling_dispatch(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Args:
        tool_name: one of list_features / build_model_dataset /
            run_model_experiment / score_model / inspect_model.
        arguments: parsed tool arguments dict from the LLM tool call.

    Returns:
        result.model_dump() — a plain dict, JSON-serializable.

    Raises:
        ValueError: unknown tool name.
        pydantic.ValidationError: arguments don't match the tool's input schema.
    """
    if tool_name not in MODELING_TOOL_DISPATCH:
        raise ValueError(
            f"Unknown modeling tool: {tool_name!r}. Available: "
            f"{sorted(MODELING_TOOL_DISPATCH.keys())}"
        )
    fn, input_model = MODELING_TOOL_DISPATCH[tool_name]
    model_instance = input_model(**arguments)
    # Same JSON-safety boundary agent.tools.dispatch applies. Modeling
    # results genuinely produce non-finite values -- a classification AUC on
    # a single-class fold, HistGradientBoosting's absent feature importance,
    # an ICIR with zero IC dispersion -- and json.dumps would emit the
    # non-standard NaN token for each, which strict parsers reject.
    return sanitize_for_json(_run_and_record(tool_name, fn, model_instance))
