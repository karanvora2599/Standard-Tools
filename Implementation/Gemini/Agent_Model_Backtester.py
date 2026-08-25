"""
Agentic Model Backtester — Google / Gemini 2.0 Flash.

This is the script that demonstrates the HANDOFF INTERCONNECT, and it is
worth reading for that alone. The agent never sees a prediction. What
crosses between the modeling runtime and the backtest runtime is a
reference — a string — and the panel behind it never enters the
conversation.

The workflow:
  1. list_models                 find a registered model (modeling)
  2. inspect_model               read its out-of-sample metrics (modeling)
  3. score_predictions           does it beat predicting the mean? (modeling)
  4. convert_reference           predictions -> signal_panel        (meta)
  5. run_signal_panel_backtest   trade the signal                (backtest)

Steps 1-3 and 4 and 5 live in three DIFFERENT runtimes, none of which can
call the others' tools. They compose anyway, because a reference is a
value: it survives the boundary, appears in the audit log as an input to
the next call, and carries no execution rights of its own.

Before this existed, step 4 was a Python function no agent could call, and
the only way to get a model's predictions into a backtest was to transcribe
the entire panel through the context window.

See Documentation/19_runtimes.md.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _agent_utils import _header, _log, run_agent, setup_logging

# ── Configuration ──────────────────────────────────────────────────
GEMINI_API_KEY = ""  # Replace with your key
MODEL = "gemini-2.0-flash"

START_DATE = "2023-01-01"
END_DATE = "2024-12-31"

SYSTEM_PROMPT = """You evaluate whether an existing statistical model is worth trading.

You work across three runtimes and you cannot call between them directly.
What you pass between them is a REFERENCE — a string of the form
sqt://<kind>/<run_id>/<name>. Never try to read a prediction panel into the
conversation and never paste one into a tool argument; that is what the
reference exists to avoid.

Your workflow:

1. list_models — find candidate models. If the user named one, skip to 2.
2. inspect_model with view='validation' — read the out-of-sample metrics
   and the fold accounting. A strong average that came from one fold is not
   a strong model.
3. score_predictions on the model's oos_predictions_ref. Read TWO things
   most carefully:
     - beats_baseline. If this is false, stop and say so. A model that does
       not beat predicting the mean has not learned anything, whatever its
       headline metric says.
     - effective_sample_size. If it is far below n_observations, the target
       windows overlap and any t-statistic from the raw count is
       overstated. Say by roughly how much.
4. convert_reference — turn oos_predictions_ref into a signal_panel. You
   MUST pass the model's task; a regression prediction thresholded as a
   probability produces a panel that looks fine and means nothing. Note
   that this step discards magnitude on purpose.
5. run_signal_panel_backtest with signal_panel_ref set to what step 4
   returned.

Then report:
  - whether the statistical result survived contact with trading costs
  - the gap between the two, if any, and which of the two you trust more
  - explicitly: whether this model beat its baseline

If a tool is refused because it belongs to another runtime, that is not an
error to work around — read the message, it names the runtime that owns the
tool, and the tool you actually want is usually in the one you are in."""

USER_REQUEST = f"""Find the most recent registered model, check whether its
out-of-sample predictions are actually predictive, and if they are, backtest
them as a long/flat/short strategy over {START_DATE} to {END_DATE}.

Tell me whether the statistical edge survives as an economic one, and be
explicit about whether it beat its baseline."""


def main() -> None:
    log_file = setup_logging("agent_model_backtester")

    _header("Agentic Model Backtester")
    _log("Log file", str(log_file))
    _log("Period", f"{START_DATE} → {END_DATE}")
    _log("Runtimes", "modeling + meta + backtest (joined explicitly)")

    # Three runtimes, named explicitly. Each refuses the others' tools by
    # name, so the composition is visible here rather than being an
    # accident of the union dispatcher knowing everything.
    result = run_agent(
        system_prompt=SYSTEM_PROMPT,
        user_request=USER_REQUEST,
        api_key=GEMINI_API_KEY,
        model=MODEL,
        max_iterations=25,
        registry="backtest+meta+modeling",
    )

    _header("VERDICT")
    print(result)


if __name__ == "__main__":
    main()
