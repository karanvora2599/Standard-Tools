"""The `meta` runtime's registry: what it advertises and what it
can execute. The two are built from one list, so a tool cannot be
advertised without being dispatchable or the reverse."""

from standard_quant_tools.agent.models import (
    ArgumentProblem,
    CompareDataSourcesInput,
    CompareDecisionsInput,
    ConvertReferenceInput,
    ConvertReferenceResult,
    DataCapabilitiesInput,
    DescribeArtifactInput,
    DescribeReferenceInput,
    DescribeReferenceResult,
    DescribeToolInput,
    DescribeToolResult,
    ExplainDecisionInput,
    ExportAuditBundleInput,
    ListReferenceKindsInput,
    ListReferenceKindsResult,
    ListStrategiesInput,
    ListStressScenariosInput,
    ReferenceKind,
    ReplayDecisionInput,
    TemporalContractInput,
    ValidateToolCallInput,
    ValidateToolCallResult,
    VerifyAuditIntegrityInput,
)

from .scope_tools import (  # noqa: F401
    SCOPE_TOOL_DEFS,
    SCOPE_TOOL_DISPATCH,
    compare_artifacts,
    describe_runtime,
    estimate_tool_cost,
)
from .tools import (
    compare_data_sources,
    compare_decisions,
    convert_reference,
    describe_artifact,
    describe_data_capabilities,
    describe_reference,
    describe_temporal_contract,
    describe_tool,
    explain_decision,
    export_audit_bundle,
    list_reference_kinds,
    list_strategies,
    list_stress_scenarios,
    replay_decision,
    validate_tool_call,
    verify_audit_integrity,
)

#: (name, description, input model) — the single source for both
#: the advertised schema and the dispatch table below.
TOOL_DEFS = [
    (
        "compare_data_sources",
        "Fetch the same fundamentals from two providers and report where they disagree, separating a SCALE difference (a constant ratio -- a missed unit conversion, fixable by arithmetic) from a DEFINITION difference (systematic with no constant ratio -- the two are computing different quantities and no conversion exists) from noise. FinancialRatios already documents that Polygon derives debt_to_equity from total liabilities and yfinance reports it as a percentage; this checks it rather than leaving it in a docstring. Fetches from both providers.",
        CompareDataSourcesInput,
    ),
    (
        "describe_temporal_contract",
        "What a data source can say about WHEN its facts became knowable, asked BEFORE fetching anything. A quarterly filing describes 30 September and is published on 25 October, so a model that joins it on the quarter end carries three weeks of hindsight per row. Read pit_safe first — False means do not build this dataset from this source — then reproduces_history, which is stricter: a snapshot source joins without leaking the future and still shows a backtest restated numbers nobody had. Fetches nothing.",
        TemporalContractInput,
    ),
    (
        "describe_tool",
        "One tool's full contract — arguments, result fields, owning runtime, and whether calling it fetches data or writes an artifact. Works for tools this caller is not scoped to; describing a tool is not calling it.",
        DescribeToolInput,
    ),
    (
        "validate_tool_call",
        "Check arguments against a tool's schema WITHOUT calling it, including the strategy parameter contract that the JSON schema cannot express. Catches a hallucinated or out-of-range argument before it costs a fetch and a run.",
        ValidateToolCallInput,
    ),
    (
        "describe_reference",
        "What a handoff reference points at — its content kind, shape, date span and which runtime published it. References are how bulk values cross runtimes without passing through the conversation.",
        DescribeReferenceInput,
    ),
    (
        "list_reference_kinds",
        "Every content kind a handoff reference can carry and what converts to what — the map of which producer outputs can reach which consumer inputs. Offline.",
        ListReferenceKindsInput,
    ),
    (
        "convert_reference",
        "Turn one kind of published value into another and publish the result: raw model predictions into a signal panel, scores into portfolio weights. This is what lets a producer and a consumer that were never written for each other compose.",
        ConvertReferenceInput,
    ),
    (
        "explain_decision",
        "What one recorded tool call did: inputs, the market data it read with the content hashes those inputs had at the time, which execution path ran (C++/Numba/Python), timing, and the git commit and package version it ran under.",
        ExplainDecisionInput,
    ),
    (
        "replay_decision",
        "Re-run a recorded call and classify the result: reproduced, data_changed (the inputs were revised, so a different answer is expected), code_changed (inputs identical, output differs — the only case implicating the library), or not_comparable.",
        ReplayDecisionInput,
    ),
    (
        "compare_decisions",
        "Diff two recorded calls — tool, inputs, output hash, git commit — and say which of the candidate causes the evidence supports.",
        CompareDecisionsInput,
    ),
    (
        "verify_audit_integrity",
        "Check the audit log's tamper-evident hash chain, for one day or the whole trail, optionally including that day's Ed25519 checkpoint signature. Read-only.",
        VerifyAuditIntegrityInput,
    ),
    (
        "export_audit_bundle",
        "Package a date range of the audit log plus its chain index and manifest into one zip. Writes a new file; modifies no existing record.",
        ExportAuditBundleInput,
    ),
    (
        "describe_artifact",
        "Shape, date span, per-column statistics and both ends of a persisted Parquet artifact, by URI. Read what a run produced instead of re-running it.",
        DescribeArtifactInput,
    ),
    (
        "list_strategies",
        "Every built-in strategy's parameter contract: names, kinds, defaults, bounds and cross-parameter relations. Offline. Call this before guessing a strategy's parameters.",
        ListStrategiesInput,
    ),
    (
        "list_stress_scenarios",
        "The named historical crash windows run_stress_test accepts, with each window's dates. Offline.",
        ListStressScenariosInput,
    ),
    (
        "describe_data_capabilities",
        "What a data provider can serve — tick trades, top-of-book quotes, async OHLCV, supported intervals, and its adjusted/survivorship/point-in-time guarantees. Fetches no market data. Call this before a tool that needs a capability the active provider may not have.",
        DataCapabilitiesInput,
    ),
]

# The discovery tools declared in scope_tools.py,
# concatenated rather than pasted so the group stays readable as a
# unit and cannot half-register.
TOOL_DEFS = TOOL_DEFS + SCOPE_TOOL_DEFS

TOOL_DISPATCH = {name: (globals()[name], model) for name, _d, model in TOOL_DEFS}

#: This runtime's slice of the library-wide routing taxonomy.
TOOL_CATEGORY = {
    "describe_tool": "discovery",
    "validate_tool_call": "discovery",
    "describe_reference": "discovery",
    "list_reference_kinds": "discovery",
    "convert_reference": "discovery",
    "explain_decision": "provenance",
    "replay_decision": "provenance",
    "compare_decisions": "provenance",
    "verify_audit_integrity": "provenance",
    "export_audit_bundle": "provenance",
    "describe_artifact": "provenance",
    "list_strategies": "discovery",
    "list_stress_scenarios": "discovery",
    "describe_data_capabilities": "discovery",
    "describe_temporal_contract": "discovery",
    "compare_data_sources": "discovery",
}

TOOL_DISPATCH.update(SCOPE_TOOL_DISPATCH)
TOOL_CATEGORY.update({name: "discovery" for name in SCOPE_TOOL_DISPATCH})

__all__ = [
    "estimate_tool_cost",
    "describe_runtime",
    "compare_artifacts",
    "describe_tool",
    "validate_tool_call",
    "describe_reference",
    "list_reference_kinds",
    "convert_reference",
    "TOOL_CATEGORY",
    "TOOL_DEFS",
    "TOOL_DISPATCH",
    "compare_decisions",
    "describe_artifact",
    "describe_data_capabilities",
    "explain_decision",
    "export_audit_bundle",
    "list_strategies",
    "list_stress_scenarios",
    "replay_decision",
    "verify_audit_integrity",
]
