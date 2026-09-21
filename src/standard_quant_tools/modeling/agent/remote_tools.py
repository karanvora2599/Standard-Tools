"""
The model store, from the agent's side: list what is in it, pull one in.

WHY THESE TWO DOORS. The registry could already push a verified package
to a directory or a bucket -- `SQT_MODEL_MIRROR_URL` sends every
registration and every promotion there as it happens -- and could read
one back, verifying each file against the manifest it travels with
before any of it is written locally. None of that was reachable from a
tool: no tool took a store at all, so a package could be pushed to a
mirror and never listed, pulled or verified by the agent that would use
it. The mirror was write-only through the tool surface.

WHAT A PULL ESTABLISHES, AND WHAT IT DOES NOT. Every hashed file is
checked against the manifest as it arrives; the signature, when there is
one or when one is required, is verified over the store's bytes BEFORE
anything is written; the manifest is written last, so a pull that fails
at any point leaves a directory that is not a registered model rather
than a registered model that is wrong. What it does not establish is who
signed: an unpinned key verifies that the manifest and the signature
were written together and nothing more, and an unsigned package that was
accepted because no signature was required says so in `warnings` rather
than returning a quiet `ok=True`.

The promotion log travels with the package, so `stage` is what somebody
decided about this model wherever it was validated -- which is the whole
point of moving a package rather than retraining one.
"""

from __future__ import annotations

import logging
import os
from typing import List

from standard_quant_tools.artifact_store import ArtifactStore, store_from_url
from standard_quant_tools.error import ValidationError

from .. import artifacts as _artifacts
from ..registry.lifecycle import current_stage
from ..registry.mirror import MIRROR_URL_ENV
from ..registry.package import list_remote_models as _list_remote_models
from ..registry.package import pull_model_package as _pull_model_package
from .remote_models import (
    ListRemoteModelsInput,
    ListRemoteModelsResult,
    PullModelPackageInput,
    PullModelPackageResult,
)

logger = logging.getLogger(__name__)

#: The tool descriptions, here so the module that registers the tools
#: does not have to restate what these doors are for.
LIST_REMOTE_MODELS_DESCRIPTION = (
    "List the model ids in an artifact store -- a directory, or a bucket "
    "reached through fsspec -- so a package registered on another machine "
    "can be found before it is pulled. The store defaults to the one "
    "SQT_MODEL_MIRROR_URL names, which is where this process pushes every "
    "registration and promotion, so with a mirror configured this is the "
    "inventory of what has been published. Only packages are listed: a "
    "model id with a manifest under it, sorted. Nothing is downloaded and "
    "nothing is verified here -- pull_model_package does both."
)

PULL_MODEL_PACKAGE_DESCRIPTION = (
    "Register a model package from an artifact store into THIS runs "
    "directory, verified on the way in, so a model trained and validated "
    "elsewhere can be inspected, scored, monitored and backtested here. "
    "Every file the manifest hashes is checked against it as it arrives "
    "and the manifest is written last, so a pull that fails registers "
    "nothing. Pass require_signature=True (and public_key_path=<the key "
    "you trust>) when the store is not one you wrote to: without a pinned "
    "key a signature proves only that the manifest and the signature "
    "travelled together, and the result's key_pinned says which of the "
    "two it established. The promotion log travels with the package, so "
    "`stage` is the stage somebody promoted it to wherever it came from. "
    "Refuses rather than overwrites when the id is already registered "
    "here."
)

#: What `store_from_url` can actually open, for the refusal below. A bare
#: path and `file://` are the local filesystem; everything else is an
#: fsspec protocol, and the two named here are the ones this package is
#: tested against beside the in-memory store.
_WORKING_SCHEMES = "a directory path, file://, memory://, s3:// or gs://"


def _resolved_store_url(store_url: "str | None", tool: str) -> str:
    """The store to address, or a refusal naming both ways to give one."""
    url = store_url or os.environ.get(MIRROR_URL_ENV)
    if not url:
        raise ValidationError(
            f"{tool}: no artifact store to address. Pass store_url=<the "
            f"store holding the packages> or set {MIRROR_URL_ENV} in the "
            "environment before the process starts, which is also what "
            "makes this process mirror its own registrations there. A "
            f"store_url is {_WORKING_SCHEMES}."
        )
    return str(url)


def _store_for(url: str, tool: str) -> ArtifactStore:
    """
    The store an URL names, with an unusable scheme refused by name.

    fsspec raises a bare `ValueError("Protocol not known: ...")` for a
    scheme it has no implementation for, and a missing optional
    dependency surfaces the same way. Either is a fact about the URL the
    caller passed, so it is answered here as a refusal that names what
    would work instead of propagating as an unhandled error from three
    layers down.
    """
    try:
        return store_from_url(url)
    except ValidationError:
        raise
    except Exception as exc:  # noqa: BLE001 - whatever fsspec raised, it is the URL
        raise ValidationError(
            f"{tool}: {url!r} is not a store this runtime can open ({exc}). "
            f"A store_url is {_WORKING_SCHEMES}; an object store also needs "
            "the fsspec filesystem for its protocol installed (s3fs, gcsfs)."
        ) from exc


def list_remote_models(input_data: ListRemoteModelsInput) -> ListRemoteModelsResult:
    """Every model id with a manifest in an artifact store."""
    url = _resolved_store_url(input_data.store_url, "list_remote_models")
    store = _store_for(url, "list_remote_models")
    found = _list_remote_models(store)
    warnings: List[str] = []
    if not found:
        warnings.append(
            f"{url} holds no model packages. A package lands there when a "
            f"registration happens in a process with {MIRROR_URL_ENV} set "
            "to this store, or when mirror_model_package is called on one."
        )
    elif len(found) > input_data.limit:
        warnings.append(
            f"{len(found)} models in the store, {input_data.limit} returned. "
            "Raise `limit` to see the rest; the listing is sorted by model "
            "id, which is not a time order."
        )
    logger.debug(
        "[list_remote_models] store=%s  found=%d  limit=%d",
        url,
        len(found),
        input_data.limit,
    )
    return ListRemoteModelsResult(
        store_url=url,
        models=found[: input_data.limit],
        n_total=len(found),
        warnings=warnings,
    )


def pull_model_package(input_data: PullModelPackageInput) -> PullModelPackageResult:
    """Register one package from an artifact store, verified on the way in."""
    url = _resolved_store_url(input_data.store_url, "pull_model_package")
    store = _store_for(url, "pull_model_package")
    report = _pull_model_package(
        input_data.model_id,
        store,
        require_signature=input_data.require_signature,
        public_key=input_data.public_key_path,
        overwrite=input_data.overwrite,
    )
    signature = report.signature
    warnings: List[str] = []
    if signature is None:
        # Said on every unsigned pull, not only when it looks suspicious:
        # the package was accepted because nothing required a signature,
        # and a result that reported ok=True without saying so would let
        # an agent record "verified" for an integrity check that has no
        # author behind it.
        warnings.append(
            f"model {input_data.model_id!r} arrived UNSIGNED and was "
            "accepted because require_signature was False. Its content "
            "hashes agree with its own manifest, which says the package is "
            "internally consistent and says nothing about who assembled "
            "it. Re-pull with require_signature=True and "
            "public_key_path=<the key you trust> if this model is going to "
            "be promoted."
        )
    elif not signature.get("key_pinned"):
        warnings.append(
            f"model {input_data.model_id!r} is signed by key "
            f"{str(signature.get('public_key'))[:16]}..., which was not "
            "checked against a key you supplied. That the manifest and the "
            "signature were written together is all this establishes; pass "
            "public_key_path to establish who wrote them."
        )
    if report.unhashed:
        warnings.append(
            f"{report.unhashed} came with the package and are not covered "
            "by its content hashes -- the signature, the promotion log and "
            "any scoring output are recorded beside the manifest rather "
            "than inside it."
        )
    logger.debug(
        "[pull_model_package] model=%s  store=%s  ok=%s  verified=%d",
        input_data.model_id,
        url,
        report.ok,
        len(report.verified),
    )
    return PullModelPackageResult(
        model_id=report.model_id,
        store_url=url,
        ok=report.ok,
        verified=report.verified,
        mismatched=report.mismatched,
        missing=report.missing,
        unhashed=report.unhashed,
        signature=signature,
        # There is no top-level field for this: the pinning is a property
        # of the check that was run, and it is recorded on the record that
        # check produced.
        key_pinned=bool((signature or {}).get("key_pinned", False)),
        stage=current_stage(input_data.model_id),
        registry_dir=str(_artifacts.run_dir(input_data.model_id)),
        warnings=warnings,
    )


__all__ = [
    "LIST_REMOTE_MODELS_DESCRIPTION",
    "PULL_MODEL_PACKAGE_DESCRIPTION",
    "ListRemoteModelsInput",
    "ListRemoteModelsResult",
    "PullModelPackageInput",
    "PullModelPackageResult",
    "list_remote_models",
    "pull_model_package",
]
