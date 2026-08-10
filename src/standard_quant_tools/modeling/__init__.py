"""
standard_quant_tools.modeling — a second, independent runtime alongside
the 46-tool analysis/backtest surface in `standard_quant_tools.agent`.

Reuses this codebase's existing indicator/analysis math, the Parquet
artifact store (`backtest.artifacts`), and the audit pipeline
(`audit.dispatch`), but exposes its own 5-tool agent surface via
`standard_quant_tools.modeling.agent` — never merged into
`standard_quant_tools.agent.get_agent_tools()`/`TOOL_CATEGORY`. See
Documentation/15_modeling.md for the full architecture rationale.
"""

from .bridge import oos_predictions_to_signal_panel
from .dataset.builder import build_dataset
from .engine import run_experiment
from .features.registry import list_features, register_feature
from .scoring import score_model
from .specs import (
    DatasetSpec,
    EstimatorSpec,
    FeatureSpec,
    ModelSpec,
    TargetSpec,
    ValidationSpec,
)

__all__ = [
    "DatasetSpec",
    "EstimatorSpec",
    "FeatureSpec",
    "ModelSpec",
    "TargetSpec",
    "ValidationSpec",
    "build_dataset",
    "list_features",
    "oos_predictions_to_signal_panel",
    "register_feature",
    "run_experiment",
    "score_model",
]
