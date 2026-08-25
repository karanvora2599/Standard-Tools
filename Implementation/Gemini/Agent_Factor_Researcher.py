"""
Agentic Factor Researcher — Google Gemini 2.0 Flash.
Gemini runs a multi-factor study and synthesises a factor research note.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _agent_utils import _header, _log, route_request, run_agent, setup_logging

# ── Configuration ──────────────────────────────────────────────────
GEMINI_API_KEY = ""  # Replace with your key
MODEL = "gemini-2.0-flash"

ASSETS = ["NVDA", "AMD", "INTC", "QCOM", "AMAT", "LRCX", "KLAC"]
START_DATE = "2020-01-01"
END_DATE = "2024-12-31"

FACTORS = {
    "market": "SPY",
    "size": "IWM",
    "value": "IWD",
    "momentum": "MTUM",
    "quality": "QUAL",
}

SYSTEM_PROMPT = """You are a quantitative factor researcher studying return attribution across a sector.

Your workflow:

Step 1 — Per-asset factor regression
  For each asset, call run_factor_regression with all 5 factors and rolling_window=60.
  Note significant factors (p < 0.05) and alpha direction.

Step 2 — Rolling stability check
  Inspect rolling_alpha_tail and rolling_loadings_tail for each asset.
  Flag if any loading changed sign or doubled/halved in the last 20 windows.

Step 3 — PCA on the sector
  Call run_pca_analysis with all assets (3 components).
  How many PCs explain 80% of variance? What does PC1 represent economically?

Step 4 — Hurst analysis on each asset
  Call run_hurst_analysis (DFA, rolling_window=63).
  Trending vs mean-reverting regime comparison across the sector.

Step 5 — Write the factor research note
  ## SECTOR FACTOR PROFILE
  ## PER-ASSET FACTOR LOADINGS TABLE
  ## FACTOR STABILITY ANALYSIS
  ## LATENT RISK FACTORS (PCA)
  ## RETURN PROCESS REGIMES (HURST)
  ## INVESTMENT IMPLICATIONS

Cite exact p-values, loadings, R², and Hurst values throughout."""

factor_names_str = ", ".join(FACTORS.keys())

USER_REQUEST = f"""
Conduct a multi-factor return attribution study on the semiconductor sector.

Assets  : {', '.join(ASSETS)}
Period  : {START_DATE} to {END_DATE}

Factors (name → proxy ticker):
{chr(10).join(f'  {k:<12} → {v}' for k, v in FACTORS.items())}

For each asset: factor regression (60-bar rolling) + Hurst analysis (63-bar rolling).
Then: PCA on the whole sector (3 components).
Write a complete factor research note with investment implications.
Use exact numbers from all tool results.
""".strip()


if __name__ == "__main__":
    log_file = setup_logging("agent_factor_researcher_gemini")

    _header("Agentic Factor Researcher — Gemini 2.0 Flash")
    _log("Log file", str(log_file))
    _log("Assets", ", ".join(ASSETS))
    _log("Factors", factor_names_str)
    _log("Period", f"{START_DATE} → {END_DATE}")

    routed_categories = route_request(USER_REQUEST, api_key=GEMINI_API_KEY, model=MODEL)

    result = run_agent(
        system_prompt=SYSTEM_PROMPT,
        user_request=USER_REQUEST,
        api_key=GEMINI_API_KEY,
        model=MODEL,
        max_iterations=30,
        categories=routed_categories,
        # factor, cointegration and PCA structure are `research`.
        # The router still narrows WITHIN the runtime; the runtime
        # is what makes the narrowing enforceable rather than
        # advisory. See Documentation/19_runtimes.md.
        registry="research",
    )

    _header("FACTOR RESEARCH NOTE")
    print(result)
