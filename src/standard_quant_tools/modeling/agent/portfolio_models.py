"""
The input and result for evaluating PUBLISHED predictions as a portfolio.

Why these live beside `portfolio_tools.py` rather than in `models.py`:
the module pair is how a tool that needs none of `tools.py`'s private
helpers stays out of that file's seam (`dataset_tools.py` is the
precedent). Everything here is a reference-shaped restatement of
`EvaluateModelPortfolioInput`/`Result` -- the field definitions are
COPIED rather than imported, because the two inputs are deliberately
diverging: one identifies a registered model and reads five dataset
fields off its manifest, and this one is handed a reference that has no
manifest to read.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from ..specs import PortfolioSimSpec, PredictionTransformSpec, Task


class EvaluatePredictionsPortfolioInput(BaseModel):
    # An argument this tool does not take is REJECTED, not ignored.
    # Pydantic's default would drop it silently, so a typo or a
    # hallucinated name ran on defaults while the caller believed it
    # had configured something -- the same failure strategy_params.py
    # exists to stop one layer down, at the boundary where a model is
    # the one choosing the names.
    model_config = ConfigDict(extra="forbid")

    predictions_ref: str = Field(
        ...,
        description="A published `predictions` reference: an ensemble from "
        "build_model_ensemble, a converted panel, or any (date, entity, "
        "prediction) frame published with kind='predictions'. A raw artifact "
        "path is refused -- it carries no kind, so nothing can check that it "
        "is predictions at all.",
    )
    task: Task = Field(
        ...,
        description="How to read the prediction column. Required, and "
        "deliberately not inferred: 'classification' probabilities are "
        "recentred on 0.5 so a score's sign is a direction, while "
        "'regression', 'ranking' and 'survival' scores pass through signed "
        "as they are. Getting it wrong on a classifier produces a "
        "long-everything book that looks like a portfolio. A registered "
        "model's task is on its manifest; a reference has none, which is why "
        "this is an argument.",
    )
    dataset_id: Optional[str] = Field(
        None,
        description="A dataset built by build_model_dataset, used ONLY for "
        "the five fields that say which bars to price this book against: "
        "interval, provider, calendar, start and end. Give this OR all four "
        "of interval/provider/start_date/end_date. Nothing is read from the "
        "dataset's panel and the predictions are not checked against it.",
    )
    interval: Optional[str] = Field(
        None,
        description="Bar interval to fetch prices at ('1d', '1h', ...). "
        "Inherited from dataset_id when that is given; an explicit value "
        "overrides it.",
    )
    provider: Optional[str] = Field(
        None,
        description="Data provider to price the simulation with. Inherited "
        "from dataset_id when that is given; an explicit value overrides it.",
    )
    calendar: Optional[str] = Field(
        None,
        description="Exchange calendar the interval's bars-per-year is "
        "derived from, which is what annualizes Sharpe, CAGR, volatility and "
        "turnover. Optional even without a dataset_id: absent, an intraday "
        "interval falls back to 252 and says so in `warnings` rather than "
        "refusing.",
    )
    start_date: Optional[str] = Field(
        None,
        description="First date to fetch prices from (YYYY-MM-DD). Earlier "
        "than the first prediction on purpose -- volatility_scale needs "
        "trailing history before the first rebalance. Inherited from "
        "dataset_id when that is given.",
    )
    end_date: Optional[str] = Field(
        None,
        description="Last date to fetch prices to (YYYY-MM-DD). Later than "
        "the last prediction on purpose -- a next_open fill needs a bar "
        "after the final rebalance. Inherited from dataset_id when that is "
        "given.",
    )
    transform: PredictionTransformSpec = Field(
        default_factory=PredictionTransformSpec,
        description="How the predictions become target weights. Defaults to "
        "a dollar-neutral, weekly-rebalanced, rank-weighted portfolio capped "
        "at 5% per name.",
    )
    portfolio: PortfolioSimSpec = Field(
        default_factory=PortfolioSimSpec,
        description="Simulation parameters (capital, costs, fill convention, "
        "leverage limits). Defaults to next-open fills, 10bps commission and "
        "5bps slippage, unlevered.",
    )
    run_id: str = Field(
        ...,
        description="Run id the target-weight and equity-curve artifacts are "
        "written under. A reference has no model directory to write beside, "
        "so the caller names the run.",
    )


class EvaluatePredictionsPortfolioResult(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    source_ref: str = Field(
        ...,
        description="The reference these metrics were computed from. Stands "
        "where evaluate_model_portfolio returns model_id: there is no "
        "registered model here, and the reference is the whole of the "
        "identity.",
    )
    metrics: Dict[str, float] = Field(
        ...,
        description=(
            "Economic performance of the simulated account: cumulative "
            "return, CAGR, annualized volatility, Sharpe, Sortino, max "
            "drawdown, Calmar, turnover, mean gross/net exposure, position "
            "count, and estimated_cost_drag_pct. These are what the "
            "predictions are worth AFTER costs and position sizing — a "
            "different question from an IC or an R2, which measure "
            "predictive accuracy and can be strong while these are negative."
        ),
    )
    transform_diagnostics: Dict[str, Any] = Field(
        ...,
        description="What the prediction -> weight step actually produced: "
        "names per date, book sizes, realized gross/net exposure, dates that "
        "could not reach the target gross under the position cap, and dates "
        "with no position at all.",
    )
    coverage: Dict[str, Any] = Field(
        ...,
        description="Entities, prediction dates, rebalance dates actually "
        "traded, and simulated bars — how much of a track record this number "
        "rests on.",
    )
    target_weights_uri: str = Field(
        ...,
        description="Persisted (date x entity) target-weight panel that drove "
        "the simulation. Content-addressed, so re-running with different "
        "transform settings writes a new artifact rather than replacing one an "
        "audit record still points at.",
    )
    equity_curve_uri: str
    provenance: Dict[str, Any] = Field(
        ...,
        description="The source reference and the PRODUCER recorded on its "
        "sidecar, plus the weight and equity-curve hashes, the five dataset "
        "fields and both specs. There is no registered content hash to check "
        "the predictions against, unlike evaluate_model_portfolio: the "
        "handoff store is the root of trust here, and the provenance says so "
        "by naming the reference and who published it instead.",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Conditions that change how these metrics should be read: "
        "a look-ahead fill convention, rebalance dates dropped, books that "
        "could not reach target gross, an ambiguous annualization factor, and "
        "anything the simulator itself raised (insolvency, negative cash). "
        "A reference carries no dataset coverage warnings — those live on a "
        "model's manifest, and nothing here can vouch for how the underlying "
        "universe was assembled.",
    )


__all__ = [
    "EvaluatePredictionsPortfolioInput",
    "EvaluatePredictionsPortfolioResult",
]
