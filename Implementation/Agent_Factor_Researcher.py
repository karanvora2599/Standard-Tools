"""
Agentic Factor Researcher — Claude autonomously runs a multi-factor research
study: it identifies which factors drive returns for a given stock, checks
whether the loadings are stable over time (rolling OLS), and compares the
factor profile across multiple assets in the same sector.

The agent:
  1. Runs multi-factor regression (market / size / value / momentum) on each asset
  2. Identifies statistically significant factors (p < 0.05) per asset
  3. Checks rolling factor loadings to detect regime shifts in factor exposure
  4. Runs PCA on the sector to find latent risk factors beyond the named ones
  5. Runs Hurst analysis to characterise each asset's return process
  6. Synthesises findings into a factor research note with investment implications
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from _agent_utils import setup_logging, run_agent, _header, _log

# ── Configuration ──────────────────────────────────────────────────
ANTHROPIC_API_KEY = ""   # Replace with your key
MODEL             = "claude-haiku-4-5"

# Semiconductor sector
ASSETS     = ["NVDA", "AMD", "INTC", "QCOM", "AMAT", "LRCX", "KLAC"]
START_DATE = "2020-01-01"
END_DATE   = "2024-12-31"

# Factor proxies
FACTORS = {
    "market":   "SPY",   # broad market
    "size":     "IWM",   # small-cap premium (size factor)
    "value":    "IWD",   # value premium
    "momentum": "MTUM",  # momentum factor ETF
    "quality":  "QUAL",  # quality factor ETF
}

SYSTEM_PROMPT = """You are a quantitative factor researcher studying return attribution across a sector.

Your workflow:

Step 1 — Per-asset factor regression
  For each asset in the universe, call run_factor_regression with:
    factor_tickers: [SPY, IWM, IWD, MTUM, QUAL]
    factor_names:   [market, size, value, momentum, quality]
    rolling_window: 60   (60-day rolling loadings — last 20 bars returned)
  Note which factors are statistically significant (p < 0.05) and the alpha direction.

Step 2 — Rolling stability check
  For each asset, look at rolling_alpha_tail and rolling_loadings_tail.
  Flag if any factor loading has changed sign or doubled/halved in the last 20 windows.
  This reveals time-varying factor exposure — important for risk management.

Step 3 — PCA on the sector
  Call run_pca_analysis with all assets. Analyse:
  - How many PCs explain 80% of variance?
  - Which assets load heavily on PC1 (systematic risk) vs idiosyncratic PCs?
  - Is PC1 simply the market factor or does it represent a sector-specific risk?

Step 4 — Hurst analysis on each asset
  Call run_hurst_analysis for each asset with rolling_window=63.
  Compare regimes — trending vs mean-reverting — across the sector.
  Trending assets are better candidates for momentum strategies;
  mean-reverting assets suit pairs / stat-arb approaches.

Step 5 — Write the factor research note
  Structure:
  ## SECTOR FACTOR PROFILE
  Summary of dominant factors across the sector.

  ## PER-ASSET FACTOR LOADINGS TABLE
  Table of significant factors, alpha, R² for each asset.

  ## FACTOR STABILITY ANALYSIS
  Which assets have stable vs time-varying exposures.

  ## LATENT RISK FACTORS (PCA)
  What the principal components represent economically.

  ## RETURN PROCESS REGIMES (HURST)
  Which assets trend vs mean-revert and strategy implications.

  ## INVESTMENT IMPLICATIONS
  Specific factor-based trade ideas derived from the analysis.

Be rigorous. Cite exact p-values, factor loadings, R², and Hurst values."""

factor_names_str   = ", ".join(FACTORS.keys())
factor_tickers_str = ", ".join(FACTORS.values())

USER_REQUEST = f"""
Conduct a multi-factor return attribution study on the semiconductor sector.

Assets  : {', '.join(ASSETS)}
Period  : {START_DATE} to {END_DATE}

Factors (name → proxy ticker):
{chr(10).join(f'  {k:<12} → {v}' for k, v in FACTORS.items())}

For each asset:
1. Run full factor regression (all 5 factors) with a 60-bar rolling window
2. Identify significant factors and note any rolling instability
3. Run Hurst analysis (DFA method, 63-bar rolling window)

Then:
4. Run PCA on the whole sector (3 components)
5. Write a complete factor research note with investment implications

Use exact numbers from all tool results in your analysis.
""".strip()


if __name__ == "__main__":
    log_file = setup_logging("agent_factor_researcher")

    _header("Agentic Factor Researcher — Claude Haiku")
    _log("Log file", str(log_file))
    _log("Assets",   ", ".join(ASSETS))
    _log("Factors",  factor_names_str)
    _log("Period",   f"{START_DATE} → {END_DATE}")

    result = run_agent(
        system_prompt=SYSTEM_PROMPT,
        user_request=USER_REQUEST,
        api_key=ANTHROPIC_API_KEY,
        model=MODEL,
        max_iterations=30,
    )

    _header("FACTOR RESEARCH NOTE")
    print(result)
