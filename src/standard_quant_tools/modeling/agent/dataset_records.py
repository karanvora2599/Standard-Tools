"""
A dataset's recorded definition, read without its panel.

A built dataset is two JSON files beside its panel: `dataset_meta.json`,
what the build observed (rows, entities, row loss, warnings, hashes), and
`dataset_spec.json`, what was asked for (each column's FeatureSpec, the
target, the interval and calendar). Several tools need the second without
the panel: `estimate_feature_warmup` prices a dataset's own features,
`check_leakage` maps a column name back to its catalog id, and
`inspect_dataset` reports both files. Reading the panel to answer those
would cost a Parquet read and a hash over every row for a question about a
few hundred bytes of JSON.

The spec is VERIFIED against the hash the build recorded before anything
uses it, by the same check `run_model_experiment` applies before it copies
the spec into a model: an edited `dataset_spec.json` (RSI period 14 changed
to 100, say) describes features the panel was not built from, and answering
from it would describe a dataset that does not exist.

Kept apart from `tools.py` so the discovery tools, which import nothing
from that module, can read a dataset too.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from standard_quant_tools.error import ValidationError

from .. import artifacts as _artifacts
from ..dataset.builder import dataset_spec_hash
from ..specs import DatasetSpec, FeatureSpec

__all__ = [
    "is_external",
    "read_dataset_meta",
    "recorded_features",
    "verified_dataset_spec",
]


def read_dataset_meta(dataset_id: str) -> Tuple[Dict[str, Any], Path]:
    """A dataset's `dataset_meta.json` and its directory. The panel is not
    read."""
    directory = _artifacts.run_dir(dataset_id)
    meta_path = directory / "dataset_meta.json"
    if not meta_path.exists():
        raise ValidationError(
            f"no dataset with dataset_id={dataset_id!r} — "
            "dataset_meta.json is written last, so its absence also means a "
            "previous build_model_dataset call did not complete."
        )
    return _artifacts.load_json(str(meta_path)), directory


def verified_dataset_spec(
    dataset_id: str, meta: Dict[str, Any], directory: Path
) -> Dict[str, Any]:
    """`dataset_spec.json`, refused if it no longer matches the hash the
    build recorded.

    The version the stored hash was written under is honoured: a dataset
    persisted before versions existed is version 1, which hashed every
    field, and is recomputed that way rather than refused for having been
    built under an older form of the check.
    """
    stored_spec_hash = meta.get("spec_hash")
    spec_hash_version = int(meta.get("spec_hash_version", 1))
    spec_dict = _artifacts.load_json(str(directory / "dataset_spec.json"))
    if stored_spec_hash is not None:
        actual_spec_hash = dataset_spec_hash(
            DatasetSpec(**spec_dict), version=spec_hash_version
        )
        if actual_spec_hash != stored_spec_hash:
            raise ValidationError(
                f"dataset {dataset_id!r}: dataset_spec.json no longer matches "
                f"the hash recorded when it was built (expected {stored_spec_hash}, found "
                f"{actual_spec_hash}, hash version {spec_hash_version}). The panel was "
                "built from the original spec, so training would register a model whose "
                "bundled feature definitions differ from the data it learned on — rebuild "
                "the dataset instead. "
                + (
                    "An UPGRADE can also cause this for a version-1 hash without "
                    "anything being edited: that form covered every field of the "
                    "spec, so a release that added one changed it for every dataset "
                    "persisted earlier. Rebuilding records a version-2 hash, which "
                    "excludes fields nobody set and survives the next such release."
                    if spec_hash_version == 1
                    else ""
                )
            )
    return spec_dict


def is_external(meta: Dict[str, Any]) -> bool:
    """Registered by `register_external_panel` rather than built here: its
    columns were computed elsewhere and have no catalog definition."""
    return meta.get("storage") == "external" or meta.get("provider") == "external"


def recorded_features(spec_dict: Dict[str, Any]) -> List[FeatureSpec]:
    """The FeatureSpecs a recorded DatasetSpec carries, in column order."""
    return [FeatureSpec(**feature) for feature in spec_dict.get("features") or []]
