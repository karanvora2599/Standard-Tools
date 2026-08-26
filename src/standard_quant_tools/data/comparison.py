"""
Two providers, the same question, different answers.

`FinancialRatios` already documents that `debt_to_equity` means different
things depending on where it came from: Polygon derives it from total
LIABILITIES, which include payables and deferred revenue, so it is
systematically higher than a debt-based ratio for reasons that have nothing
to do with leverage. yfinance reports it as a percentage where the canonical
form is a plain ratio.

That is written down, in a docstring, which somebody has to read. Nothing
checks it. A screen that ranks on `debt_to_equity` across a universe fetched
from two providers is comparing two different quantities and will produce a
confident ordering with no error anywhere.

WHAT THIS MODULE DOES. Fetch the same field from two sources and report
where they disagree, separating three cases that look identical in a diff
and need different responses:

  - **scale** -- a constant ratio between them, which is a unit conversion
    somebody missed. Mechanical, and the fix is arithmetic.
  - **definition** -- a systematic offset with no constant ratio, which
    means the two are measuring different things. No conversion exists; one
    of them has to be chosen deliberately.
  - **noise** -- small, unsystematic differences. Vendors disagree at the
    margin about everything and this is not a finding.

Getting those three confused is what makes a data-quality report either
ignored or acted on wrongly.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

#: Relative difference below which two vendors are simply rounding
#: differently. Not a tolerance to tune -- it is the level at which a
#: disagreement stops being evidence of anything.
NOISE_THRESHOLD = 0.01

#: How close the pairwise ratios must be for a divergence to be called a
#: SCALE difference rather than a definition one. A true unit error gives an
#: essentially exact constant (100, or 1/100); real data gives it with a
#: little slop. Above this the ratio is not constant and the two fields are
#: measuring different things.
SCALE_CONSISTENCY = 0.05


def _finite_pairs(left: Dict[str, Any], right: Dict[str, Any], field: str):
    """(entity, a, b) for entities where BOTH sides reported a number."""
    pairs = []
    for entity in sorted(set(left) & set(right)):
        a = getattr(left[entity], field, None)
        b = getattr(right[entity], field, None)
        if a is None or b is None:
            continue
        try:
            a, b = float(a), float(b)
        except (TypeError, ValueError):
            continue
        if math.isfinite(a) and math.isfinite(b):
            pairs.append((entity, a, b))
    return pairs


def classify_divergence(
    pairs: Sequence, *, noise: float = NOISE_THRESHOLD
) -> Dict[str, Any]:
    """
    Decide whether a field's disagreement is scale, definition or noise.

    The scale test is the interesting one: take the ratio b/a for every
    entity and ask whether it is CONSTANT. A unit error gives the same ratio
    everywhere -- that is what a unit is. A definition difference gives a
    ratio that wanders, because the two quantities respond differently to
    different balance sheets.

    Entities where either side is zero are excluded from the ratio, not from
    the comparison: 0/0 and x/0 say nothing about scale, and letting them in
    would make the ratio wander for arithmetic reasons and mislabel a clean
    unit error as a definition difference.
    """
    if not pairs:
        return {
            "verdict": "no_overlap",
            "n_compared": 0,
            "detail": "no entity had a value from both sources",
        }

    a = np.array([p[1] for p in pairs], dtype=float)
    b = np.array([p[2] for p in pairs], dtype=float)

    scale = np.maximum(np.abs(a), np.abs(b))
    relative = np.where(scale > 0, np.abs(a - b) / np.where(scale > 0, scale, 1), 0.0)
    max_relative = float(np.max(relative))
    median_relative = float(np.median(relative))

    if max_relative <= noise:
        return {
            "verdict": "agree",
            "n_compared": len(pairs),
            "max_relative_difference": max_relative,
            "median_relative_difference": median_relative,
            "detail": "the two sources agree to within rounding",
        }

    usable = (np.abs(a) > 0) & (np.abs(b) > 0)
    ratios = b[usable] / a[usable]
    ratio_spread = (
        float(np.std(ratios) / abs(np.mean(ratios)))
        if ratios.size and abs(np.mean(ratios)) > 0
        else float("inf")
    )
    mean_ratio = float(np.mean(ratios)) if ratios.size else float("nan")

    if ratios.size >= 2 and ratio_spread <= SCALE_CONSISTENCY:
        return {
            "verdict": "scale",
            "n_compared": len(pairs),
            "max_relative_difference": max_relative,
            "median_relative_difference": median_relative,
            "ratio": mean_ratio,
            "ratio_spread": ratio_spread,
            "detail": (
                f"the second source is consistently {mean_ratio:.4g}x the "
                "first, which is a unit conversion rather than a "
                "disagreement about the quantity"
            ),
        }

    return {
        "verdict": "definition",
        "n_compared": len(pairs),
        "max_relative_difference": max_relative,
        "median_relative_difference": median_relative,
        "ratio_spread": ratio_spread,
        "detail": (
            "the difference is systematic but the ratio is not constant, so "
            "the two sources are computing different quantities. No "
            "conversion exists -- pick one deliberately and record which"
        ),
    }


def compare_ratio_sources(
    left: Dict[str, Any],
    right: Dict[str, Any],
    *,
    left_name: str = "left",
    right_name: str = "right",
    fields: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Compare two providers' `FinancialRatios`, field by field.

    Takes already-fetched results rather than fetching, so the comparison is
    testable without a network and so a caller who already has both is not
    made to fetch twice.
    """
    from standard_quant_tools.data.base import FinancialRatios

    if fields is None:
        fields = [
            name
            for name, field in FinancialRatios.model_fields.items()
            if name != "definition_notes"
        ]

    per_field: List[Dict[str, Any]] = []
    for field in fields:
        pairs = _finite_pairs(left, right, field)
        result = classify_divergence(pairs)
        result["field"] = field
        # The worst offenders by name, so a caller can go and look.
        if result["verdict"] in ("scale", "definition") and pairs:
            worst = sorted(
                pairs,
                key=lambda p: -abs(p[1] - p[2]) / max(abs(p[1]), abs(p[2]), 1e-12),
            )[:3]
            result["examples"] = [
                {"entity": e, left_name: a, right_name: b} for e, a, b in worst
            ]
        per_field.append(result)

    declared = _declared_notes(left, right, left_name, right_name)
    warnings = _warnings(per_field, left_name, right_name, declared)

    return {
        "left": left_name,
        "right": right_name,
        "n_entities_compared": len(set(left) & set(right)),
        "fields": per_field,
        "declared_definition_notes": declared,
        "warnings": warnings,
    }


def _declared_notes(left, right, left_name, right_name) -> List[Dict[str, str]]:
    """
    Definition differences the providers already declared themselves.

    Surfaced next to the measured ones because they answer different halves
    of the same question: a declared note says "we know this is different",
    and a measured divergence says "and here is how much".
    """
    out = []
    for name, side in ((left_name, left), (right_name, right)):
        for entity, ratios in sorted(side.items()):
            notes = getattr(ratios, "definition_notes", None) or {}
            for field, note in sorted(notes.items()):
                out.append(
                    {
                        "source": name,
                        "entity": entity,
                        "field": field,
                        "note": str(note),
                    }
                )
    return out


def _warnings(per_field, left_name, right_name, declared) -> List[str]:
    out: List[str] = []
    for result in per_field:
        if result["verdict"] == "scale":
            out.append(
                f"{result['field']}: {right_name} is {result['ratio']:.4g}x "
                f"{left_name} across all {result['n_compared']} entities. That "
                "is a unit conversion, not a disagreement -- but ranking a "
                "universe on a mix of the two would order it by which "
                "provider answered."
            )
        elif result["verdict"] == "definition":
            out.append(
                f"{result['field']}: {left_name} and {right_name} differ "
                f"systematically with no constant ratio across "
                f"{result['n_compared']} entities, so they are computing "
                "different quantities. No conversion exists; choose one and "
                "record which."
            )
    if declared:
        fields = sorted({d["field"] for d in declared})
        out.append(
            f"the provider(s) themselves declare a definition difference for "
            f"{fields}. Read declared_definition_notes -- a declared "
            "difference is not a bug and is not convertible either."
        )
    return out


__all__ = [
    "NOISE_THRESHOLD",
    "SCALE_CONSISTENCY",
    "classify_divergence",
    "compare_ratio_sources",
]
