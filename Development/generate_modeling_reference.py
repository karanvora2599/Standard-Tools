"""
Generate `Documentation/29_modeling_reference.md` from the modeling registries.

WHY GENERATED RATHER THAN WRITTEN. The modeling guide carried a
hand-written feature table that said 21 entries when the registry held 23,
and seven docstrings quoted a 6-tool, 46-tool or 16-tool surface long
after the counts had moved. `modeling_capabilities()` already treats the
live registries as the truth for an agent; this makes the documentation
follow the same principle, the way `generate_tool_index.py` already does
for the tool surface. `tests/docs/test_documentation.py` regenerates this
file and fails if what is on disk differs, so a feature, estimator or
target added without regenerating breaks the suite in the same commit.

MACHINE-INDEPENDENT BY CONSTRUCTION. `ESTIMATOR_REGISTRY` only holds what
this machine could import, so the optional lightgbm and xgboost entries are
taken from `boosting.OPTIONAL_ESTIMATORS`, a static declaration that exists
whether or not the library does. Nothing here reads a version, a path or an
install state; the output is the same on every machine at the same commit.

RUN IT WITH:  python Development/generate_modeling_reference.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

OUTPUT = ROOT / "Documentation" / "29_modeling_reference.md"

HEADER = """# Modeling reference

Every feature, estimator, target and spec option the modeling runtime
knows, read from its registries. **Generated** by
`Development/generate_modeling_reference.py` -- a test regenerates it and
fails if this file has drifted, so an entry added without regenerating
breaks the suite in the commit that added it. The prose that explains
these lives in [15_modeling.md](15_modeling.md); this is the catalog.

The estimators marked *optional* register only when their library is
installed. They are listed from a static declaration so this document is
the same on every machine; `list_modeling_capabilities` reports which of
them the running install actually has.
"""


def _cell(value: Any) -> str:
    text = str(value) if value is not None else ""
    return text.replace("|", "\\|").replace("\n", " ")


def _table(columns: List[str], rows: List[List[Any]]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        lines.append("| " + " | ".join(_cell(v) for v in row) + " |")
    return "\n".join(lines)


def _params(mapping: Dict[str, Any]) -> str:
    if not mapping:
        return ""
    return ", ".join(f"`{k}={v!r}`" for k, v in sorted(mapping.items()))


def _features() -> str:
    from standard_quant_tools.modeling.features.registry import list_features

    rows = []
    for definition in list_features():
        rows.append(
            [
                f"`{definition.id}`",
                definition.scope.value,
                definition.temporal_support.value,
                definition.lookback,
                ", ".join(definition.requires) or "Close",
                _params(definition.default_params),
                definition.description,
            ]
        )
    return (
        f"## Features ({len(rows)})\n\n"
        + _table(
            ["id", "scope", "temporal", "lookback", "requires", "default params", "description"],
            rows,
        )
    )


def _estimators() -> str:
    from standard_quant_tools.modeling.adapters import get_adapter
    from standard_quant_tools.modeling.estimators import boosting
    from standard_quant_tools.modeling.estimators.registry import (
        ESTIMATOR_REGISTRY,
        allowed_params,
    )

    optional = boosting.OPTIONAL_ESTIMATORS
    rows = []
    for (task, name), cls in sorted(ESTIMATOR_REGISTRY.items()):
        if (task, name) in optional:
            continue
        capabilities = get_adapter(task).capabilities(cls)
        flags = [
            label
            for key, label in (
                ("supports_sample_weight", "sample weights"),
                ("supports_probability", "probabilities"),
                ("needs_groups", "query groups"),
                ("exposes_coefficients", "coefficients"),
                ("exposes_feature_importance", "importances"),
            )
            if capabilities.get(key)
        ]
        rows.append(
            [
                task,
                f"`{name}`",
                f"`{cls.__module__}.{cls.__qualname__}`",
                ", ".join(f"`{p}`" for p in allowed_params(task, name)),
                ", ".join(flags),
            ]
        )
    optional_rows = []
    for (task, name), (library, schema) in sorted(optional.items()):
        optional_rows.append(
            [
                task,
                f"`{name}`",
                f"*optional: `{library}`*",
                ", ".join(f"`{p}`" for p in schema.allowed_names),
            ]
        )
    return (
        f"## Estimators ({len(rows)} always available, {len(optional_rows)} optional)\n\n"
        "Parameter values are bounded as well as named; see "
        "[15_modeling.md](15_modeling.md#parameter-values-are-bounded-not-just-named).\n\n"
        + _table(["task", "name", "class", "allowed params", "capabilities"], rows)
        + "\n\n### Optional\n\n"
        + _table(["task", "name", "requires", "allowed params"], optional_rows)
    )


def _preprocessing() -> str:
    from standard_quant_tools.modeling.preprocessing import list_preprocessors

    rows = []
    for definition in list_preprocessors():
        rows.append(
            [
                f"`{definition.id}`",
                ", ".join(f"`{p}`" for p in definition.schema_.allowed_names) or "",
                _params(definition.default_params),
                "stateless" if definition.stateless else "fitted on train",
                "yes" if definition.column_wise else "no",
                definition.description,
            ]
        )
    return (
        f"## Preprocessing steps ({len(rows)})\n\n"
        "Composed in order by `PreprocessingSpec.steps`; each is fitted on the "
        "fold's training rows and its state applied to the test rows, then "
        "persisted with the model as `preprocessing_state.json`. "
        "`normalization='pooled'` resolves to `winsorize` then `zscore`; "
        "`'cross_sectional'` to `cross_sectional_standardize`.\n\n"
        + _table(
            ["id", "params", "defaults", "state", "column-wise", "description"], rows
        )
    )


def _targets() -> str:
    from standard_quant_tools.modeling.specs import TARGET_KINDS

    rows = [
        [
            f"`{name}`",
            ", ".join(kind.tasks),
            "yes" if kind.buildable else "external only",
            "continuous" if kind.continuous else "discrete",
            kind.description,
        ]
        for name, kind in sorted(TARGET_KINDS.items())
    ]
    buildable = sum(1 for kind in TARGET_KINDS.values() if kind.buildable)
    return (
        f"## Targets ({buildable} buildable from prices, "
        f"{len(rows) - buildable} external only)\n\n"
        + _table(["id", "tasks", "buildable", "kind", "description"], rows)
    )


def _spec_options() -> str:
    from standard_quant_tools.modeling.capabilities import _literal_options
    from standard_quant_tools.modeling.specs import (
        EstimatorSpec,
        PredictionTransformSpec,
        PreprocessingSpec,
        SearchSpec,
        ValidationSpec,
        WeightingSpec,
    )

    rows = [
        ["`ValidationSpec.method`", _literal_options(ValidationSpec, "method")],
        ["`ValidationSpec.scheme`", _literal_options(ValidationSpec, "scheme")],
        ["`PreprocessingSpec.normalization`", _literal_options(PreprocessingSpec, "normalization")],
        ["`WeightingSpec.method`", _literal_options(WeightingSpec, "method")],
        ["`SearchSpec.method`", _literal_options(SearchSpec, "method")],
        ["`SearchSpec.scoring`", _literal_options(SearchSpec, "scoring")],
        ["`EstimatorSpec.calibration`", _literal_options(EstimatorSpec, "calibration")],
        ["`PredictionTransformSpec.method`", _literal_options(PredictionTransformSpec, "method")],
        [
            "`PredictionTransformSpec.rebalance_frequency`",
            _literal_options(PredictionTransformSpec, "rebalance_frequency"),
        ],
    ]
    return "## Spec options\n\n" + _table(
        ["field", "choices"],
        [[field, ", ".join(f"`{c}`" for c in choices)] for field, choices in rows],
    )


def _limits() -> str:
    from standard_quant_tools.modeling import limits

    rows = [
        ["`MAX_LAG`", limits.MAX_LAG, "deepest single lag, in bars"],
        ["`MAX_LAGS_PER_FEATURE`", limits.MAX_LAGS_PER_FEATURE, "lags one feature may request"],
        ["`MAX_EXPANDED_COLUMNS`", limits.MAX_EXPANDED_COLUMNS, "ceiling on the expanded panel"],
    ]
    return "## Limits\n\n" + _table(["name", "value", "meaning"], rows)


def render() -> str:
    sections = [
        HEADER,
        _features(),
        _estimators(),
        _preprocessing(),
        _targets(),
        _spec_options(),
        _limits(),
    ]
    return "\n\n".join(section.rstrip() for section in sections) + "\n"


def main() -> None:
    OUTPUT.write_text(render(), encoding="utf-8", newline="\n")
    print(f"wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
