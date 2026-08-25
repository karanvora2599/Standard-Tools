"""
Agentic Provenance Auditor — why was that decision made, and does it hold?

Every dispatch() in this library writes a tamper-evident record: the tool,
its inputs, content hashes of the market data it read, which execution path
ran, the output hash — chained so that editing a past line breaks every
line after it. Thirteen CLI commands operated on that log and no tool did,
which meant the one participant who could not check its own work was the
agent whose work it was.

This agent runs on the `meta` runtime, which is read-and-verify only.
Retention operations that could destroy evidence (gc, seal, hold) are
deliberately CLI-only — an agent able to delete the record of its own
decisions is not audited by it — so they are not merely absent from the
prompt, they are unroutable.

THE DISTINCTION THIS AGENT EXISTS TO MAKE. A backtest that returns a
different number today is not evidence of a bug. The default provider
guarantees neither point-in-time values nor stable adjusted prices, so
revisions are the normal case. replay_decision checks the INPUT hashes
first, and only "every checked input still hashes identically and the
output does not" implicates the library.

See Documentation/10_auditability.md and Documentation/19_runtimes.md.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _agent_utils import _header, _log, run_agent, setup_logging

# ── Configuration ──────────────────────────────────────────────────
ANTHROPIC_API_KEY = ""  # Replace with your key
MODEL = "claude-haiku-4-5"

# The request id of a recorded call. Every dispatch() writes one; it
# appears in the audit log and in log records correlated by RequestIdFilter.
REQUEST_ID = ""  # e.g. "8817818041f2432c9471147ce5af8124"

SYSTEM_PROMPT = """You are an audit and provenance specialist. Your tools read and verify
a tamper-evident decision log. You cannot alter it, and you should not try.

Given a request id:

1. explain_decision — what the call did. Report the tool, the inputs, the
   market data it read with the content hashes those inputs had AT THE
   TIME, and the execution path. The execution path is worth calling out:
   C++, Numba and pure Python fall back transparently at call time, so
   which one ran is knowable ONLY because the record says so.
2. replay_decision — does it still reproduce? Report the VERDICT, not the
   boolean, and explain what it means:
     - reproduced      output and data both match
     - data_changed    the inputs no longer hash the same. A different
                       output is EXPECTED and says nothing about the code.
                       Do not report this as a defect.
     - code_changed    inputs identical, output differs. This is the only
                       combination that implicates the library. Compare
                       git_commit_sha across the two runs.
     - not_comparable  the record cannot be checked either way.
3. verify_audit_integrity — is the chain intact?

Always say what a check does NOT prove:
  - A hash chain detects partial or accidental tampering. A wholesale
    rewrite can recompute it; only a signed checkpoint catches that.
  - A single day verified alone cannot detect a MISSING day.
  - An exported bundle verifying cleanly proves the copy is internally
    consistent, not that the live log was untouched.

If asked to delete, seal or hold anything, say plainly that those
operations are CLI-only by design and why."""

USER_REQUEST = f"""Audit request {REQUEST_ID!r}: what did that call do, does it
still reproduce, and is the surrounding log intact?

If it no longer reproduces, tell me whether that is the data's fault or the
code's — and be explicit about which, because they call for different
responses."""


def main() -> None:
    if not REQUEST_ID:
        raise SystemExit(
            "Set REQUEST_ID to a recorded call's request id. Every dispatch() "
            "writes one; look in SQT_AUDIT_DIR (default "
            "~/.cache/standard_quant_tools/audit/) or run `sqt report <id>`."
        )

    log_file = setup_logging("agent_provenance_auditor")

    _header("Agentic Provenance Auditor")
    _log("Log file", str(log_file))
    _log("Request", REQUEST_ID)
    _log("Runtime", "meta (read-and-verify only)")

    result = run_agent(
        system_prompt=SYSTEM_PROMPT,
        user_request=USER_REQUEST,
        api_key=ANTHROPIC_API_KEY,
        model=MODEL,
        max_iterations=15,
        # `meta` holds discovery and provenance. Retention lives in no
        # runtime at all, so "delete this record" is unroutable rather than
        # merely discouraged.
        registry="meta",
    )

    _header("AUDIT FINDING")
    print(result)


if __name__ == "__main__":
    main()
