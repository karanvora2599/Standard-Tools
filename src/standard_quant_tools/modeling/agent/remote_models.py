"""
The inputs and results for reaching a model store that is not this one.

Why these live beside `remote_tools.py` rather than in `models.py`: the
module pair is how a tool that needs none of `tools.py`'s private helpers
stays out of that file's seam (`dataset_tools.py` is the precedent).

Both inputs default `store_url` to the environment variable that
configures the mirror, so an operator who set it once before the process
started does not have to repeat it on every call -- and an agent that has
no store configured is told which variable would supply one rather than
being handed a default it cannot see.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

#: Said the same way in both inputs, because the answer to "where is the
#: store" is the same question twice and a reader comparing the two
#: schemas should not have to work out whether the wording hides a
#: difference.
_STORE_URL_DESCRIPTION = (
    "The artifact store holding the packages: a directory path, a "
    "`file://` URL, or an object store `fsspec` can open (`s3://bucket/"
    "prefix`, `gs://...`, `memory://...`). Defaults to the store "
    "SQT_MODEL_MIRROR_URL names -- the one registrations in this process "
    "are pushed to -- and is refused by name when neither is given."
)


class ListRemoteModelsInput(BaseModel):
    # An argument this tool does not take is REJECTED, not ignored.
    # Pydantic's default would drop it silently, so a typo or a
    # hallucinated name ran on defaults while the caller believed it had
    # configured something.
    model_config = ConfigDict(extra="forbid")

    store_url: Optional[str] = Field(None, description=_STORE_URL_DESCRIPTION)
    limit: int = Field(
        200,
        gt=0,
        le=1000,
        description="How many model ids to return. `n_total` reports how "
        "many the store holds, so a truncated listing says so rather than "
        "looking like the whole store.",
    )


class ListRemoteModelsResult(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    store_url: str = Field(
        ...,
        description="The store that was listed, as given or as inherited "
        "from the environment -- so a result read later says which store "
        "these ids are from.",
    )
    models: List[str] = Field(
        default_factory=list,
        description="Model ids with a manifest in the store, sorted, "
        "truncated to `limit`. A package is found by its model id, which "
        "is the only key prefix one is written under; nothing else in the "
        "store is listed.",
    )
    n_total: int = Field(
        ...,
        description="How many model ids the store holds, before `limit` "
        "was applied.",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Conditions that change how this listing should be "
        "read: a truncated result, or an empty store.",
    )


class PullModelPackageInput(BaseModel):
    # `model_id` starts with `model_`, which pydantic protects by default.
    model_config = ConfigDict(protected_namespaces=(), extra="forbid")

    model_id: str = Field(
        ...,
        description="The model id to register locally, as list_remote_"
        "models reports it. A manifest in the store that names a different "
        "id is refused rather than registered under the id it was asked "
        "for.",
    )
    store_url: Optional[str] = Field(None, description=_STORE_URL_DESCRIPTION)
    require_signature: bool = Field(
        False,
        description="Refuse a package that carries no signature. Default "
        "False because a shop with no signing key would otherwise be "
        "unable to pull anything; True is what you want when the store is "
        "not the one you wrote to. A signature that IS present is checked "
        "either way, before anything is written locally.",
    )
    public_key_path: Optional[str] = Field(
        None,
        description="Path to the Ed25519 public key the signature must be "
        "under. Without it a signature is checked against the key it "
        "carries, which proves the manifest and the signature were written "
        "together and not that anyone you trust wrote them -- `key_pinned` "
        "on the result says which of the two this pull established.",
    )
    overwrite: bool = Field(
        False,
        description="Replace a model of this id already registered here. "
        "Default False: a registered model's directory is the evidence "
        "behind every score and promotion recorded against it, and "
        "replacing it silently would leave those claims pointing at "
        "different bytes.",
    )


class PullModelPackageResult(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    store_url: str = Field(..., description="The store the package was read from.")
    ok: bool = Field(
        ...,
        description="Every file the manifest hashes is present and hashes "
        "to what it recorded, and the signature (when there is one) "
        "verified.",
    )
    verified: List[str] = Field(
        default_factory=list,
        description="Files whose bytes hash to what the manifest recorded, "
        "re-checked on local disk after the copy.",
    )
    mismatched: List[str] = Field(
        default_factory=list,
        description="Files present whose hash no longer matches. Empty "
        "after a successful pull -- a file that changed in flight is "
        "refused by name before anything is registered.",
    )
    missing: List[str] = Field(
        default_factory=list,
        description="Files the manifest hashes that are not there.",
    )
    unhashed: List[str] = Field(
        default_factory=list,
        description="Files that arrived which the manifest does NOT cover: "
        "the signature, the promotion log, scoring outputs. Named so a "
        "reader knows what the hashes do not vouch for.",
    )
    signature: Optional[Dict[str, Any]] = Field(
        None,
        description="The verified signature record -- algorithm, public "
        "key, when it was signed, the manifest digest it covers -- or None "
        "when the package is unsigned and no signature was required.",
    )
    key_pinned: bool = Field(
        False,
        description="Whether the signature was checked against a key YOU "
        "supplied. False means it verified under the key it carried, which "
        "establishes that the manifest and the signature travelled "
        "together, not that they came from a signer you trust.",
    )
    stage: str = Field(
        ...,
        description="The lifecycle stage the pulled package's promotion "
        "log carries: the log travels with the model, so a model promoted "
        "to `validated` elsewhere arrives validated rather than as a fresh "
        "candidate.",
    )
    registry_dir: str = Field(
        ...,
        description="The local directory the package was registered into, "
        "under this process's runs root.",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="What this pull did NOT establish: an unsigned package "
        "accepted because no signature was required, a signature under an "
        "unpinned key, and files that arrived outside the manifest's "
        "hashes.",
    )


__all__ = [
    "ListRemoteModelsInput",
    "ListRemoteModelsResult",
    "PullModelPackageInput",
    "PullModelPackageResult",
]
