"""The `meta` runtime's registry: what it advertises and what it
can execute. The two are built from one list, so a tool cannot be
advertised without being dispatchable or the reverse."""

from standard_quant_tools.agent.models import (
    CompareDecisionsInput,
    DataCapabilitiesInput,
    DescribeArtifactInput,
    ExplainDecisionInput,
    ExportAuditBundleInput,
    ListStrategiesInput,
    ListStressScenariosInput,
    ReplayDecisionInput,
    VerifyAuditIntegrityInput,
)

from .tools import (
    compare_decisions,
    describe_artifact,
    describe_data_capabilities,
    explain_decision,
    export_audit_bundle,
    list_strategies,
    list_stress_scenarios,
    replay_decision,
    verify_audit_integrity,
)

#: (name, description, input model) — the single source for both
#: the advertised schema and the dispatch table below.
TOOL_DEFS = [
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

TOOL_DISPATCH = {name: (globals()[name], model) for name, _d, model in TOOL_DEFS}

#: This runtime's slice of the library-wide routing taxonomy.
TOOL_CATEGORY = {
    "explain_decision": "provenance",
    "replay_decision": "provenance",
    "compare_decisions": "provenance",
    "verify_audit_integrity": "provenance",
    "export_audit_bundle": "provenance",
    "describe_artifact": "provenance",
    "list_strategies": "discovery",
    "list_stress_scenarios": "discovery",
    "describe_data_capabilities": "discovery",
}

__all__ = [
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
