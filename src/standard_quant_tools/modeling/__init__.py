"""
standard_quant_tools.modeling — a second, independent runtime alongside
the analysis/backtest surface in `standard_quant_tools.agent`.

Reuses this codebase's existing indicator/analysis math, the Parquet
artifact store (`backtest.artifacts`), and the audit pipeline
(`audit.dispatch`), but exposes its own agent surface via
`standard_quant_tools.modeling.agent` — never merged into
`standard_quant_tools.agent.get_agent_tools()`/`TOOL_CATEGORY`. See
Documentation/15_modeling.md for the full architecture rationale.
"""

from .bridge import oos_predictions_to_signal_panel
from .dataset.builder import build_dataset
from .engine import run_experiment
from .features.registry import list_features, register_feature
from .model_sleeves import build_model_sleeves
from .portfolio_eval import (
    evaluate_model_portfolio,
    mean_variance_weights,
    transform_predictions_to_weights,
)
from .scoring import score_model
from .specs import (
    DatasetSpec,
    EstimatorSpec,
    FeatureSpec,
    MeanVarianceSpec,
    ModelSpec,
    PortfolioSimSpec,
    PredictionTransformSpec,
    TargetSpec,
    ValidationSpec,
)

__all__ = [
    "DatasetSpec",
    "EstimatorSpec",
    "FeatureSpec",
    "MeanVarianceSpec",
    "ModelSpec",
    "PortfolioSimSpec",
    "PredictionTransformSpec",
    "TargetSpec",
    "ValidationSpec",
    "build_dataset",
    "build_model_sleeves",
    "evaluate_model_portfolio",
    "list_features",
    "mean_variance_weights",
    "oos_predictions_to_signal_panel",
    "register_feature",
    "run_experiment",
    "score_model",
    "transform_predictions_to_weights",
]
