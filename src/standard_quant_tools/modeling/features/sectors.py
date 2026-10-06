"""
The entity -> sector map `group_demean` neutralises within, built once.

WHY THIS RETURNS A WARNING AND NOT ONLY A MAP. A provider's sector is
TODAY'S classification. Applied to a panel that starts ten years ago it
says Meta was in Communication Services in 2015, which it was not — it was
Information Technology until the 2018 GICS restructure — and it says
nothing at all about the names that have since been acquired or delisted.
That is the same survivorship shape this repo already documents for
universe membership, and a map that arrived without the sentence would be
used as though it were point-in-time.

So the warning travels WITH the map. A caller that wants it in the record
puts it there; a caller that ignores it had to ignore something.

THE MAP IS BUILT ONCE AND PASSED IN, which is the point of this module
existing at all rather than `group_demean` calling a provider itself. A
step that looked up a sector at fit time would neutralise differently next
month against the same panel, so a model registered today would not
reproduce: its manifest would describe a pipeline whose behaviour lives
outside it. Passed in, the map is hashed into the dataset and the model.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

#: What a provider reports for a name it cannot classify. Mapped to no
#: group rather than to a group called "Unknown": names that share only
#: "nobody knows" are not a sector, and demeaning them against each other
#: would invent a factor out of ignorance.
UNKNOWN = "Unknown"


def sector_groups(
    symbols: Sequence[str], provider: Any = None
) -> Tuple[Dict[str, str], List[str]]:
    """
    `({symbol: sector}, warnings)` for the names a provider can classify.

    A symbol the provider does not know, or reports as "Unknown", is LEFT
    OUT of the map rather than given a shared label. `group_demean` makes
    an unmapped row NaN, which is the honest answer — a name with no sector
    has no sector-relative position — whereas a shared "Unknown" bucket
    would demean unrelated companies against each other and call the result
    a neutralisation.

    `provider` defaults to the configured one. Failures per symbol are
    collected rather than raised: one unclassifiable name should not stop
    a universe being mapped, and the count is in the warnings.
    """
    if provider is None:
        from standard_quant_tools.data import get_provider

        provider = get_provider()

    mapped: Dict[str, str] = {}
    unknown: List[str] = []
    failed: List[str] = []
    for symbol in symbols:
        try:
            info = provider.get_ticker_info(str(symbol))
        except Exception:
            failed.append(str(symbol))
            continue
        sector = str(getattr(info, "sector", "") or "").strip()
        if not sector or sector == UNKNOWN:
            unknown.append(str(symbol))
            continue
        mapped[str(symbol)] = sector

    warnings = [
        "THIS IS TODAY'S CLASSIFICATION, APPLIED TO EVERY DATE IN THE "
        "PANEL. A sector is not point-in-time here: a name that moved "
        "between sectors is recorded only where it sits now, and a name "
        "that has since been acquired or delisted is not recorded at all. "
        "That is the same survivorship shape this library documents for "
        "universe membership, and a neutralisation built on it removes "
        "today's sector from yesterday's returns."
    ]
    if unknown:
        warnings.append(
            f"{len(unknown)} of {len(symbols)} symbol(s) have no sector and "
            f"are NOT in the map ({unknown[:5]}{'...' if len(unknown) > 5 else ''}). "
            "group_demean makes their rows NaN, which is the honest answer: "
            "a name with no sector has no sector-relative position. They are "
            "deliberately not pooled into one 'Unknown' group — names that "
            "share only 'nobody knows' are not a sector, and demeaning them "
            "against each other would invent a factor out of ignorance."
        )
    if failed:
        warnings.append(
            f"{len(failed)} symbol(s) could not be looked up at all "
            f"({failed[:5]}{'...' if len(failed) > 5 else ''}) and are not in "
            "the map. One unclassifiable name does not stop a universe being "
            "mapped, but these rows will be NaN after the demean."
        )
    return mapped, warnings


def group_sizes(groups: Dict[str, str]) -> Dict[str, int]:
    """How many names each group holds, for reading before fitting.

    A group of one demeans to NaN, so a map whose groups are mostly
    singletons neutralises almost the whole panel away — worth seeing
    before a run rather than as an all-NaN feature afterwards.
    """
    sizes: Dict[str, int] = {}
    for label in groups.values():
        sizes[label] = sizes.get(label, 0) + 1
    return dict(sorted(sizes.items(), key=lambda kv: (-kv[1], kv[0])))


def singleton_warning(groups: Dict[str, str]) -> Optional[str]:
    """One sentence when most of the map cannot be neutralised, else None."""
    sizes = group_sizes(groups)
    singletons = [label for label, size in sizes.items() if size < 2]
    if not singletons:
        return None
    covered = sum(size for size in sizes.values() if size > 1)
    return (
        f"{len(singletons)} of {len(sizes)} group(s) hold one name "
        f"({singletons[:5]}{'...' if len(singletons) > 5 else ''}), so "
        f"{len(groups) - covered} of {len(groups)} mapped names have no peer "
        "to be demeaned against and become NaN. A group of one has no "
        "within-group position to report."
    )


__all__ = [
    "UNKNOWN",
    "group_sizes",
    "sector_groups",
    "singleton_warning",
]
