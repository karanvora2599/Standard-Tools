"""
Agentic Model Builder — Google Gemini 2.0 Flash.

Gemini drives the MODELING runtime end to end: catalog the features, build a
panel, judge those features before fitting anything, fit and walk-forward
validate a model, then evaluate its out-of-sample predictions as a portfolio.

This is the one example script that does not use the 46-tool analysis
surface. `standard_quant_tools.modeling.agent` is a separate 8-tool registry
that the library never merges into the other one (Documentation/15_modeling.md
explains why), so this script passes registry="modeling" to run_agent() and
gets those eight tools and their own dispatch function together.

There is no route_request() call here either, and that is not an oversight:
the router exists to narrow 46 similarly-shaped tools down to the relevant
category. Eight tools in one ordered pipeline have nothing to narrow -- every
one of them is used, in sequence -- so routing would be overhead with no
selection ambiguity to remove.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _agent_utils import _header, _log, run_agent, setup_logging

# ── Configuration ──────────────────────────────────────────────────
GEMINI_API_KEY = ""  # Replace with your key
MODEL = "gemini-2.0-flash"

UNIVERSE = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMD",
    "INTC",
    "QCOM",
    "AVGO",
    "TXN",
    "AMAT",
    "LRCX",
    "KLAC",
    "ADI",
    "MU",
    "NXPI",
    "MRVL",
    "ON",
]
START_DATE = "2019-01-01"
END_DATE = "2024-12-31"
HORIZON = 5  # forward-return horizon, in bars

SYSTEM_PROMPT = """You are a quantitative researcher building a cross-sectional return model.

You have eight tools, and they form ONE ORDERED PIPELINE. Later steps depend
on ids returned by earlier ones, so do not reorder them or skip ahead.

Step 1 — Establish what this install can do
  list_modeling_capabilities. Read which estimators, targets, tasks and
  validation schemes are actually available HERE. lightgbm and xgboost are
  optional and may be absent — check before naming one, rather than
  discovering it at fit time.

Step 2 — Choose features deliberately
  list_features. Pick features that measure DIFFERENT things. Two momentum
  features at neighbouring lookbacks are one feature counted twice, and the
  redundancy report in step 4 will say so.

Step 3 — Build the dataset
  build_model_dataset with a spec carrying universe, start, end, features and
  target. It returns a dataset_id. Every later step needs that exact id.

Step 4 — Judge the features BEFORE fitting anything
  analyze_features on the dataset_id. This is the step most likely to change
  your mind, so read it properly:
  - coverage: a feature missing on most rows costs rows for every other
    feature too, because alignment drops the whole row.
  - IC / rank IC / ICIR: rank IC is the one that matches how the model is
    scored cross-sectionally.
  - the redundancy matrix: near-duplicate features split importance between
    themselves and make both look weak.
  - the leakage screen: treat a flag as a CLAIM TO CHECK, not a verdict.
    Look at the lead-lag IC curve. A feature that peaks at shift 0 and decays
    smoothly either side is usually a slow-moving state variable (ADX,
    realized volatility), not a leak. A leak looks like a sharp isolated
    spike at 0 against a flat curve elsewhere.
  If something is redundant or genuinely leaky, rebuild the dataset without
  it rather than fitting anyway and explaining the result away afterwards.

Step 5 — Fit and validate
  run_model_experiment with the dataset_id and a model spec. Validation is
  walk-forward with an embargo and a target-overlap purge; use it. It returns
  a model_id and out-of-sample metrics.

Step 6 — Read the validation honestly
  inspect_model with view="validation", then view="feature_importance".
  A mean IC earned steadily across ten folds and the same mean carried by two
  folds are different claims. Say which one you have.

Step 7 — Ask whether it would have made money
  evaluate_model_portfolio on the model_id. Out-of-sample IC is NOT an answer
  to that question — a model can rank names well and still lose after costs.
  This is the tool that answers it, so call it before recommending anything.

Step 8 — Write the research note
  ## UNIVERSE AND FEATURES
  ## FEATURE DIAGNOSTICS      (coverage, IC, redundancy, leakage verdict)
  ## MODEL AND VALIDATION     (fold-by-fold, not just the mean)
  ## PORTFOLIO EVALUATION     (after costs)
  ## WHAT I WOULD NOT CLAIM   (what this test cannot establish)

Cite exact numbers from the tool results. Never describe an in-sample number
as out-of-sample, and never present IC as a return."""

USER_REQUEST = f"""
Build and validate a cross-sectional {HORIZON}-day forward-return model on the
semiconductor universe, then tell me whether it would actually have been
tradeable.

Universe : {', '.join(UNIVERSE)}
Period   : {START_DATE} to {END_DATE}
Target   : forward return over {HORIZON} bars

Work through the full pipeline:
1. Check what this install supports before choosing an estimator
2. Pick a set of features that measure genuinely different things
3. Build the dataset
4. Analyze the features before fitting — report coverage, rank IC, redundancy
   and the leakage screen, and say whether any flag is real
5. Fit with walk-forward validation
6. Report performance fold by fold, not just the average
7. Evaluate the out-of-sample predictions as a portfolio with costs charged
8. Write the research note, including what this test does NOT establish
""".strip()


if __name__ == "__main__":
    log_file = setup_logging("agent_model_builder_gemini")

    _header("Agentic Model Builder — Gemini 2.0 Flash")
    _log("Log file", str(log_file))
    _log("Registry", "modeling (8 tools)")
    _log("Universe", f"{len(UNIVERSE)} tickers")
    _log("Period", f"{START_DATE} → {END_DATE}")
    _log("Horizon", f"{HORIZON} bars")

    result = run_agent(
        system_prompt=SYSTEM_PROMPT,
        user_request=USER_REQUEST,
        api_key=GEMINI_API_KEY,
        model=MODEL,
        max_iterations=30,
        registry="modeling",
    )

    _header("MODEL RESEARCH NOTE")
    print(result)
