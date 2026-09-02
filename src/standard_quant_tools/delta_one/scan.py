"""
Many pairs at once, ranked by how unusual their basis is right now.

WHY A SCAN IS A DIFFERENT TOOL. `basis_history` answers "where is this
basis against its own history" and `detect_basis_dislocation` answers "did
this basis structurally shift". Both take ONE pair. A desk does not look at
one pair -- it looks at every future it trades against every underlying and
asks which one is worth attention this morning.

That is not a loop a model should write. Running twenty pairs through two
functions and sorting the output is deterministic work, and asking an LLM
to do it costs twenty round trips and gets the ranking wrong when two
z-scores are close.

NOTHING NEW IS COMPUTED HERE. Every number comes from `basis_history` or
`detect_basis_dislocation`. This module aligns the inputs, calls them per
pair, and orders the result -- and reports the pairs it could NOT evaluate
rather than dropping them, because a pair that failed to align is a data
problem the caller needs to see, not an absence of signal.

RANKED BY |z|, NOT BY BASIS. A 40 bps basis is not interesting if that
name always trades at 40 bps. The whole point of scanning is to find the
one sitting away from its own history, which is what the z-score measures
and what the raw level does not.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from standard_quant_tools.delta_one.basis import basis_history, detect_basis_dislocation
from standard_quant_tools.error import ValidationError
from standard_quant_tools.numeric_contract import require_positive_int

logger = logging.getLogger(__name__)

__all__ = ["basis_scan"]


def basis_scan(
    pairs: Sequence[Mapping[str, Any]],
    *,
    window: Optional[int] = None,
    detect_shifts: bool = True,
    min_observations: int = 30,
    top_n: int = 10,
) -> Dict[str, Any]:
    """
    Rank a set of spot/futures pairs by how far each basis sits from its own
    history.

    Each entry of `pairs` needs `label`, `spot` and `futures`; optionally
    `time_to_expiry` (a per-observation sequence, which is what turns a
    basis in points into an annualized rate) and `multiplier`.

    `detect_shifts` additionally runs the CUSUM change detector per pair.
    It is on by default because the two answer different questions -- a
    basis can sit at a 2-sigma z-score having been there for months, which
    is a level, or it can have moved there last week, which is an event --
    and a scan that reported only the level would rank those the same.

    Returns every pair it could evaluate in `ranked`, ordered by absolute
    z-score, plus `skipped` for the ones it could not, each with the reason.
    A pair is never silently dropped: a spot and futures series that do not
    align is a data problem, and a scan that quietly returned fewer rows
    than it was given would hide it.

    Raises:
        ValidationError: `pairs` is empty, or an entry is missing `label`,
        `spot` or `futures`.
    """
    if not pairs:
        raise ValidationError(
            "basis_scan: no pairs were given. This ranks a set against each "
            "other; a set of nothing has no ranking."
        )
    min_observations = require_positive_int(
        min_observations, "min_observations", "basis_scan"
    )
    top_n = require_positive_int(top_n, "top_n", "basis_scan")

    ranked: List[Dict[str, Any]] = []
    skipped: List[Dict[str, str]] = []

    for i, entry in enumerate(pairs):
        label = str(entry.get("label", f"pair_{i}"))
        spot = entry.get("spot")
        futures = entry.get("futures")
        if spot is None or futures is None:
            skipped.append({"label": label, "reason": "missing 'spot' or 'futures'"})
            continue
        if len(spot) != len(futures):
            skipped.append(
                {
                    "label": label,
                    "reason": (
                        f"spot has {len(spot)} observations and futures has "
                        f"{len(futures)}; they describe the same instrument "
                        "at the same times or they describe nothing"
                    ),
                }
            )
            continue
        if len(spot) < min_observations:
            skipped.append(
                {
                    "label": label,
                    "reason": (
                        f"{len(spot)} observations, below the "
                        f"{min_observations} this scan requires. A z-score "
                        "against a handful of points is noise with a number."
                    ),
                }
            )
            continue

        expiry = entry.get("time_to_expiry")
        try:
            history = basis_history(
                spot=list(spot),
                futures=list(futures),
                window=window,
                time_to_expiry=list(expiry) if expiry is not None else None,
            )
        except ValidationError as exc:
            skipped.append({"label": label, "reason": str(exc)})
            continue

        row: Dict[str, Any] = {
            "label": label,
            "n_observations": history["n_observations"],
            "current_basis_bps": history["current_basis_bps"],
            "mean_bps": history["mean_bps"],
            "std_bps": history["std_bps"],
            "zscore": history["zscore"],
            "percentile": history["percentile"],
            "half_life_observations": history["half_life_observations"],
            "annualized": history.get("annualized"),
            "warnings": list(history.get("warnings", [])),
        }

        if detect_shifts:
            try:
                shift = detect_basis_dislocation(
                    spot=list(spot),
                    futures=list(futures),
                    time_to_expiry=list(expiry) if expiry is not None else None,
                )
                row["shift_detected"] = bool(shift.get("triggered"))
                row["shift_severity"] = shift.get("severity")
                row["shift_at"] = shift.get("first_crossing")
                row["shift_in_reference_sd"] = shift.get("shift_in_reference_sd")
                row["warnings"].extend(shift.get("warnings", []))
            except ValidationError as exc:
                # The level still stands even when the detector refuses --
                # reporting the level and NOTING the refusal beats dropping
                # a pair whose z-score was computed fine.
                row["shift_detected"] = None
                row["shift_severity"] = None
                row["warnings"].append(f"Change detection could not run: {exc}")

        ranked.append(row)

    # ORDERED BY |z|. A basis that is always wide is not news; one sitting
    # away from its own history is. A None z-score (a degenerate basis with
    # no dispersion) sorts last rather than first, which is what an
    # unguarded sort on None would otherwise do.
    ranked.sort(
        key=lambda r: (
            abs(r["zscore"])
            if r["zscore"] is not None and np.isfinite(r["zscore"])
            else -np.inf
        ),
        reverse=True,
    )

    warnings: List[str] = []
    if skipped:
        warnings.append(
            f"{len(skipped)} of {len(pairs)} pairs could not be evaluated and "
            "are listed in `skipped` with a reason each. They are not "
            "absences of signal."
        )
    flat = [r for r in ranked if r["zscore"] is None]
    if flat:
        warnings.append(
            f"{len(flat)} pairs have no basis dispersion at all, so their "
            "z-score is undefined and they sort last. A perfectly constant "
            "basis usually means a stale or synthetic series rather than a "
            "perfectly tracked one."
        )
    if detect_shifts:
        fired = [r for r in ranked if r.get("shift_detected")]
        if fired:
            warnings.append(
                f"{len(fired)} pairs show a structural shift, which is a "
                "different finding from a wide level: the basis MOVED "
                "rather than merely being far from its mean. Those are the "
                "ones with a date attached in `shift_at`."
            )

    logger.debug(
        "[basis_scan] evaluated=%d  skipped=%d  window=%s",
        len(ranked),
        len(skipped),
        window,
    )

    return {
        "n_pairs": int(len(pairs)),
        "n_evaluated": int(len(ranked)),
        "n_skipped": int(len(skipped)),
        "window": window,
        "ranked": ranked[:top_n],
        "skipped": skipped,
        "warnings": warnings,
    }
