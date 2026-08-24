"""
Workflow prompts: the reusable studies this library is for.

A CORRECTION TO THE PLAN. `Development/mcp_plan.md` §7 proposed reusing the
nine worker system prompts from `Multi_Agent_Implementation/worker_agents.py`
verbatim, on the grounds that a sixth copy of the model-builder prompt is
duplication. That does not survive contact with the protocol: an MCP prompt
is a USER-invoked template that takes arguments and produces a user message,
while a worker system prompt SCOPES an agent that already has a fixed tool
list. They are different artifacts with different readers.

So these are written for this surface. What they borrow from the worker
prompts is the judgement, not the text -- the insistence on reading
validation fold by fold, on treating a leakage flag as a claim rather than a
verdict, and on not confusing out-of-sample IC with money.

WHY SO FEW. Five, not fifty-four. A prompt per tool would be a menu nobody
wrote; these are the multi-step studies where the ORDER and the caveats are
the value, which is exactly what a tool description cannot carry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Sequence, Tuple


@dataclass(frozen=True)
class PromptArg:
    name: str
    description: str
    required: bool = True


@dataclass(frozen=True)
class WorkflowPrompt:
    name: str
    title: str
    description: str
    arguments: Tuple[PromptArg, ...]
    render: Callable[[Mapping[str, str]], str]

    def build(self, values: Mapping[str, str]) -> str:
        missing = [
            a.name for a in self.arguments if a.required and not values.get(a.name)
        ]
        if missing:
            raise ValueError(f"prompt {self.name!r} needs argument(s) {missing}")
        return self.render(values)


def _get(values: Mapping[str, str], key: str, default: str = "") -> str:
    value = values.get(key)
    return value if value not in (None, "") else default


def _screen_and_backtest(v: Mapping[str, str]) -> str:
    return f"""Screen this universe, then backtest what survives.

Universe : {_get(v, 'universe')}
Criteria : {_get(v, 'criteria')}
Period   : {_get(v, 'period', 'the last 3 years')}

1. Screen the universe against the criteria and report which tickers pass,
   with the exact values that got them through.
2. For each survivor, run the backtest and report Sharpe, max drawdown,
   total return and trade count.
3. Rank them, and say plainly which results are too few trades to mean
   anything -- a Sharpe from 4 trades is not a Sharpe.

Use the numbers the tools return. Do not estimate anything you could
measure, and do not carry a ticker forward that the screen rejected."""


def _factor_research_note(v: Mapping[str, str]) -> str:
    return f"""Write a factor research note on this sector.

Assets  : {_get(v, 'assets')}
Factors : {_get(v, 'factors', 'market (SPY), size (IWM), value (IWD)')}
Period  : {_get(v, 'period', 'the last 5 years')}

1. Run the factor regression per asset with a rolling window. Report the
   loadings, their p-values, and R².
2. Check rolling stability: flag any loading that changed sign or halved
   or doubled over the window tail. A stable average hiding an unstable
   path is the finding, not a footnote.
3. Run PCA across the sector. Say how many components reach 80% of
   variance, and whether PC1 is just the market.
4. Run the Hurst analysis per asset and group them into trending versus
   mean-reverting.
5. Write the note: sector profile, per-asset loadings table, stability,
   latent factors, regimes, and what you would NOT conclude from this."""


def _pair_trade_study(v: Mapping[str, str]) -> str:
    return f"""Assess this pair for a mean-reversion trade.

Pair   : {_get(v, 'symbol_a')} / {_get(v, 'symbol_b')}
Period : {_get(v, 'period', 'the last 5 years')}

1. Test for cointegration. Report the p-value, the hedge ratio, and the
   half-life. If it does not cointegrate, say so and stop -- do not proceed
   to backtest a spread that has no reason to revert.
2. If it does, estimate the hedge ratio both statically and with the Kalman
   filter, and say whether they disagree.
3. Backtest the pair with realistic costs.
4. Report the result with the half-life next to the holding period. A
   half-life longer than the average hold is the trade fighting itself."""


def _build_and_validate_model(v: Mapping[str, str]) -> str:
    return f"""Build a cross-sectional forward-return model and tell me whether it
would actually have been tradeable.

Universe : {_get(v, 'universe')}
Horizon  : {_get(v, 'horizon', '5')} bars
Period   : {_get(v, 'period', 'the last 5 years')}

Work the pipeline in order:
1. Check what this install supports before naming an estimator -- lightgbm
   and xgboost are optional and may be absent here.
2. Choose features that measure DIFFERENT things. Two momentum features at
   neighbouring lookbacks are one feature counted twice.
3. Build the dataset. Report the dataset_id.
4. Analyze the features BEFORE fitting: coverage, rank IC, redundancy, and
   the leakage screen. Treat a leakage flag as a claim to check against the
   lead-lag curve, not a verdict -- a slow-moving state feature such as ADX
   or realized volatility peaks at shift 0 honestly. A real leak is a sharp
   isolated spike against an otherwise flat curve.
5. Fit with walk-forward validation.
6. Report performance FOLD BY FOLD. A mean IC earned steadily across ten
   folds and the same mean carried by two are different claims.
7. Evaluate the out-of-sample predictions as a portfolio with costs
   charged. Out-of-sample IC does not answer "would this have made money";
   this step does.
8. Close with what this test does NOT establish."""


def _risk_review(v: Mapping[str, str]) -> str:
    return f"""Review the risk in this portfolio.

Holdings : {_get(v, 'holdings')}
Period   : {_get(v, 'period', 'the last 3 years')}

1. Decompose the risk: marginal contribution per position and the PCA
   variance breakdown. Name the positions carrying more risk than weight.
2. Report the correlation structure, and say whether the diversification is
   real or whether several names are one bet.
3. Stress the allocation against the relevant historical windows. Report
   any ticker without history that far back rather than quietly dropping it.
4. Report liquidity: which positions would be slow or expensive to exit.
5. Summarize the three risks you would actually raise, and say which of
   them this data cannot measure."""


PROMPTS: Tuple[WorkflowPrompt, ...] = (
    WorkflowPrompt(
        name="screen_and_backtest",
        title="Screen a universe, then backtest the survivors",
        description=(
            "Filter a ticker universe on fundamental/technical criteria, "
            "backtest each survivor, and rank them -- with an explicit check "
            "on which results have too few trades to interpret."
        ),
        arguments=(
            PromptArg("universe", "Tickers to screen, comma-separated."),
            PromptArg("criteria", "Screening criteria in plain language."),
            PromptArg("period", "Date range or description.", required=False),
        ),
        render=_screen_and_backtest,
    ),
    WorkflowPrompt(
        name="factor_research_note",
        title="Multi-factor attribution study",
        description=(
            "Per-asset factor regressions, rolling-stability checks, PCA "
            "across the sector, and Hurst regimes, written up as a note."
        ),
        arguments=(
            PromptArg("assets", "Assets to study, comma-separated."),
            PromptArg("factors", "Factor name -> proxy ticker.", required=False),
            PromptArg("period", "Date range or description.", required=False),
        ),
        render=_factor_research_note,
    ),
    WorkflowPrompt(
        name="pair_trade_study",
        title="Cointegration and pair-trade assessment",
        description=(
            "Test a pair for cointegration, estimate the hedge ratio two "
            "ways, and backtest the spread -- stopping early if the pair "
            "does not cointegrate."
        ),
        arguments=(
            PromptArg("symbol_a", "First ticker."),
            PromptArg("symbol_b", "Second ticker."),
            PromptArg("period", "Date range or description.", required=False),
        ),
        render=_pair_trade_study,
    ),
    WorkflowPrompt(
        name="build_and_validate_model",
        title="Build, validate and evaluate a cross-sectional model",
        description=(
            "The full modeling pipeline: capabilities, features, dataset, "
            "feature report, walk-forward fit, fold-by-fold validation, and "
            "a portfolio evaluation with costs charged."
        ),
        arguments=(
            PromptArg("universe", "Tickers for the model universe."),
            PromptArg("horizon", "Forward-return horizon in bars.", required=False),
            PromptArg("period", "Date range or description.", required=False),
        ),
        render=_build_and_validate_model,
    ),
    WorkflowPrompt(
        name="risk_review",
        title="Portfolio risk decomposition and stress review",
        description=(
            "Risk attribution, correlation structure, historical stress "
            "replay and liquidity, summarized as the risks worth raising."
        ),
        arguments=(
            PromptArg("holdings", "Positions, e.g. 'AAPL 30%, MSFT 40%, NVDA 30%'."),
            PromptArg("period", "Date range or description.", required=False),
        ),
        render=_risk_review,
    ),
)

BY_NAME: Dict[str, WorkflowPrompt] = {p.name: p for p in PROMPTS}


def names() -> List[str]:
    return [p.name for p in PROMPTS]


def required_categories(prompt: WorkflowPrompt) -> Sequence[str]:
    """
    Which tool categories a prompt actually needs.

    Used to warn at `prompts/get` when a prompt is invoked against a server
    that was not started with the categories it depends on -- otherwise the
    model receives a workflow it has no tools to execute and improvises,
    which is the worst of both.
    """
    return _PROMPT_CATEGORIES[prompt.name]


_PROMPT_CATEGORIES: Dict[str, Tuple[str, ...]] = {
    "screen_and_backtest": ("screener", "backtest_execution"),
    "factor_research_note": ("quant_research",),
    "pair_trade_study": ("quant_research", "backtest_execution"),
    "build_and_validate_model": ("modeling",),
    "risk_review": ("portfolio_risk", "analysis"),
}
