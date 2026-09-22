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
    ReadReferenceInput,
    ReferenceKind,
    ReplayDecisionInput,
    TemporalContractInput,
    ValidateToolCallInput,
    ValidateToolCallResult,
    VerifyAuditIntegrityInput,
)

from .artifact_tools import ListArtifactsInput, list_artifacts
from .audit_tools import (
    AuditLogInput,
    FindDecisionsInput,
    describe_audit_log,
    find_decisions,
)
from .config_tools import EffectiveConfigInput, describe_effective_config
from .contract_tools import NumericContractInput, describe_numeric_contract
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
    read_reference,
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
        "read_reference",
        "The actual values at chosen rows of a handoff reference — by date, or the first/last N. `describe_reference` says what a reference holds; this says what is IN it. Bounded on purpose: ask for the rows you intend to cite, because references exist to keep bulk values out of the conversation.",
        ReadReferenceInput,
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
        "Check the audit log's tamper-evident hash chain, for one day or the whole trail, optionally including that day's Ed25519 checkpoint signature. The verdict separates intact, tampered, no_trail and recording_disabled, because an empty directory is not an intact one, and signature_state names which of six things a failed checkpoint check means. Read-only.",
        VerifyAuditIntegrityInput,
    ),
    (
        "export_audit_bundle",
        "Package a date range of the audit log plus its chain index, any checkpoint sidecars and a manifest into one zip; a range covering no day file is refused rather than exported as a bundle of nothing. Writes a new file; modifies no existing record.",
        ExportAuditBundleInput,
    ),
    (
        "describe_artifact",
        "Shape, date span, per-column statistics and both ends of a persisted Parquet artifact, by URI or by the store key list_artifacts reports. Read what a run produced instead of re-running it.",
        DescribeArtifactInput,
    ),
    (
        "list_artifacts",
        "Every artifact this library has persisted, or one run's: key, absolute URI, size, last-modified time and -- on request -- the content hash. Tools hand back a URI once, in one response, and after that the file existed with no way to find it. The hash is opt-in because it is the only part that opens the files rather than their directory entries; it is the same digest describe_artifact reports, so the two compare directly. Read-only.",
        ListArtifactsInput,
    ),
    (
        "describe_audit_log",
        "What the decision log holds and what it is configured to do: which dates, how many records, how large, and the recording, redaction, retention and signing settings that decide what a count of zero means. Per day, on request, whether it is held, sealed or carries a signed checkpoint. The retention window is reported as a PREVIEW of what a policy would make eligible -- nothing here deletes, seals, holds or releases anything, because the chain cannot tell a policy-driven deletion from the tampering it exists to detect. The redaction salt is reported as set or unset, never as a value.",
        AuditLogInput,
    ),
    (
        "find_decisions",
        "Search the decision log by tool, status and date, and get back the request ids explain_decision, replay_decision and compare_decisions take. dispatch() returns the payload alone, so an in-process caller otherwise has no way to obtain one and those three tools are unreachable. It is also the only way to read a FAILED call: an error record is written like any other and nothing else surfaces one. Reads only; nothing is re-run.",
        FindDecisionsInput,
    ),
    (
        "describe_numeric_contract",
        "The numerical rules every public boundary in this library enforces -- an infinity refused, an all-NaN series refused, prices strictly positive, an equity curve's START positive, an annualization ceiling, a covariance symmetric to 1e-9, a bool refused as a count -- with the threshold each bites at, an excerpt of the message it raises, and why the line is where it is. These ran on every call and were reported nowhere, so the only way to learn one was to trigger it after paying for the fetch. Offline and static.",
        NumericContractInput,
    ),
    (
        "describe_effective_config",
        "Every SQT_* setting this process reads, resolved through the functions that read it rather than echoed from the environment -- so an unset variable still reports the value in force. Covers recording, redaction, retention and signing of the decision log, the artifact and cache roots, the native-extension switch, provider credentials and the model registry. A secret reports only whether it is set: disclosing the redaction salt would undo the redaction it configures. Reads configuration and cannot change it.",
        EffectiveConfigInput,
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
        "What a data provider can serve — tick trades, top-of-book quotes, L2 depth, order events, point-in-time records, its own temporal contract, async OHLCV, supported intervals, and its adjusted/survivorship/point-in-time guarantees. Also reports the persistent cache: how many files, how large, and how many were written under a format version nothing reads any more (counted, never deleted). Fetches no market data. Call this before a tool that needs a capability the active provider may not have — depth and order events are served by one provider only, and point-in-time records by a different one.",
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
    "read_reference": "discovery",
    "list_reference_kinds": "discovery",
    "convert_reference": "discovery",
    "explain_decision": "provenance",
    "replay_decision": "provenance",
    "compare_decisions": "provenance",
    "verify_audit_integrity": "provenance",
    "export_audit_bundle": "provenance",
    "describe_audit_log": "provenance",
    "find_decisions": "provenance",
    "describe_artifact": "provenance",
    "list_artifacts": "discovery",
    "describe_numeric_contract": "discovery",
    "describe_effective_config": "discovery",
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
    "describe_audit_log",
    "describe_data_capabilities",
    "describe_effective_config",
    "describe_numeric_contract",
    "explain_decision",
    "export_audit_bundle",
    "find_decisions",
    "list_artifacts",
    "list_strategies",
    "list_stress_scenarios",
    "replay_decision",
    "verify_audit_integrity",
]
