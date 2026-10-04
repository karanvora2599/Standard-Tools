"""
A skops bundle beside the joblib: the estimator, loadable without pickle.

WHY A SECOND FORMAT. `model.joblib` is verified against the manifest
before it is deserialized, and the manifest can be signed, so a swapped
binary is refused before it can run. That is a guard around pickle, not
a replacement for it: joblib IS pickle, and pickle executes code from
the file by design. `skops` serializes an sklearn estimator as its
declared state and, at load, constructs only the types the loader was
told to trust -- an unknown type is refused before anything is built.
A model that carries both formats can be loaded either way; the choice
is `load_model(format=...)` or `SQT_MODEL_FORMAT`.

WHAT IS TRUSTED. skops trusts sklearn's estimators by default, but not
every object a fitted one holds: a forest's or a gradient-boosting
model's trees (`sklearn.tree._tree.Tree`), a histogram-boosting model's
(`TreePredictor`), a calibrated classifier's per-fold calibrators and an
MLP's optimizer are outside its defaults, so every such bundle was
refused at load. Those five types, which `skops.io.get_untrusted_types`
reports for the estimators this library registers, are trusted by name
(`TRUSTED_TYPES`). This package's estimators -- the numpy Cox model, the
quantile wrappers -- are trusted by their module prefix, because they
are this code. Anything else in a bundle is named and refused: a
LightGBM or XGBoost booster, which skops writes but whose loading would
mean trusting that library's own state, or a type from somewhere this
registry did not write.

WHEN THERE IS NO BUNDLE. skops cannot serialize everything (a booster
holding a native handle, say). Registration then keeps joblib alone and
the manifest's `formats` says so; asking for skops on such a model is
refused by name rather than answered with joblib.

THE SAME MODEL, THE SAME BYTES. A skops archive as `skops.io.dump` writes
it differs between two runs that fit the same model: each zip member
carries the wall-clock time it was written, each array is stored under
the memory address of the object it came from (`2269133876912.npy`) and
named by that address in `schema.json`'s `__id__` and `file` keys, a
bytes value is stored under a random UUID, and a tree's node records
carry seven padding bytes each, whatever the allocator left there. The
manifest hashes the file, so `model.skops`'s content hash never
reproduced. `dump_estimator` rewrites the archive before it is saved: the
ids numbered 1, 2, 3 in the order `schema.json` first names them, each
member file renamed to match, padding bytes zero, and every member dated
1980-01-01 with fixed attributes. The loader reads ids only to tell one
object from another, members only by the names `schema.json` gives, and
an array's fields but never its padding, so what loads is unchanged; an
archive written before loads as it always did.

AND THE SAME JOBLIB. `model.joblib` of a fresh fit reproduced, but two
things in it did not follow the model. A tree's node records are pickled
with their padding bytes, and a forest loaded from a file carries
whatever its loader left there, so two loads of one file re-dumped to two
hashes. And a histogram-boosting model pickles its bin mapper's
`n_threads`, the OpenMP thread count it was fitted under (the physical
core count when nothing limits it), so the same model fitted under
limits of 1 and 4 threads, or on a 10-core and a 2-core machine, gave
different bytes. `save_joblib` writes the joblib, and `dump_estimator`
the bundle, under `reproducible_state`: node padding zeroed in place, and
`n_threads` recorded as None for the length of the dump, then restored.
"""

from __future__ import annotations

import io
import json
import logging
import zipfile
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from standard_quant_tools._env import env_str
from standard_quant_tools.error import ValidationError

from .. import artifacts as _artifacts

logger = logging.getLogger(__name__)

TRUSTED_PREFIX = "standard_quant_tools."
FORMAT_ENV = "SQT_MODEL_FORMAT"
FORMATS = ("joblib", "skops")

#: The scikit-learn types outside skops' defaults that the estimators this
#: library registers hold once fitted, as `skops.io.get_untrusted_types`
#: names them (skops 0.15, scikit-learn 1.9): trees of a random forest,
#: gradient boosting and the quantile gradient boosting; trees of a
#: histogram-boosting model; the per-fold calibrators of a calibrated
#: classifier; the optimizer of an MLP. A fixed list, so a bundle naming
#: any other type is still refused by name.
TRUSTED_TYPES = (
    "sklearn.calibration._CalibratedClassifier",
    "sklearn.calibration._SigmoidCalibration",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
    "sklearn.neural_network._stochastic_optimizers.AdamOptimizer",
    "sklearn.tree._tree.Tree",
)

#: Where `reproducible_state` looks for trees and bin mappers: objects
#: from these packages are walked through their attributes; any other
#: object (a LightGBM booster, say) is not entered.
_WALKED_PACKAGES = ("sklearn", "standard_quant_tools")


def skops_available() -> bool:
    try:
        import skops.io  # noqa: F401
    except ImportError:
        return False
    return True


def _require() -> None:
    if not skops_available():
        raise ValidationError(
            "a skops bundle needs the `skops` package "
            "(`pip install standard_quant_tools[skops]`)."
        )


def default_format() -> str:
    """The format `load_model` uses when not told: `SQT_MODEL_FORMAT`, else
    joblib. A blank value is unset, not a format to refuse."""
    value = (env_str(FORMAT_ENV) or "joblib").lower()
    if value not in FORMATS:
        raise ValidationError(
            f"{FORMAT_ENV}={value!r} is not a model format; the formats are {list(FORMATS)}."
        )
    return value


def dump_estimator(directory: Path, name: str, estimator: Any) -> Optional[str]:
    """
    Write `<name>.skops` under `directory`, atomically; None when skops is
    not installed or cannot serialize this estimator, which is logged and
    recorded on the manifest rather than raised -- the joblib is the
    format every registration has, the bundle is the one it has when it
    can.
    """
    if not skops_available():
        return None
    import skops.io as sio

    buffer = io.BytesIO()
    try:
        with reproducible_state(estimator):
            sio.dump(estimator, buffer)
    except Exception as exc:  # noqa: BLE001 - any failure means "no bundle"
        logger.warning(
            "[modeling] skops could not serialize %s (%s); the model is "
            "registered with joblib only.",
            type(estimator).__name__,
            exc,
        )
        return None
    path = Path(directory) / f"{name}.skops"
    _artifacts._atomic_write_bytes(path, deterministic_archive(buffer.getvalue()))
    return str(path)


def save_joblib(directory: Path, name: str, obj: Any) -> str:
    """
    `artifacts.save_joblib` under `reproducible_state`: the same model gives
    the same `<name>.joblib`, whether it was fitted here, loaded from a file,
    or fitted under another OpenMP thread count. `obj` may be an estimator
    or a container of them (the quantile models' dict).
    """
    with reproducible_state(obj):
        return _artifacts.save_joblib(directory, name, obj)


def _padding_mask(dtype: Any) -> Any:
    """
    For a structured dtype whose records hold bytes no field covers, a
    boolean mask over one record's bytes that is True on those; None for a
    packed or unstructured dtype, which has nothing to clear.

    A fitted tree's node record is such a dtype: seven fields of eight
    bytes and one of one, in a 64-byte record, so seven bytes per node are
    whatever the allocator left there.
    """
    import numpy as np

    if dtype.names is None:
        return None
    covered = np.zeros(dtype.itemsize, dtype=bool)
    for name in dtype.names:
        field, offset = dtype.fields[name][:2]
        covered[offset : offset + field.itemsize] = True
    if covered.all():
        return None
    return ~covered


def _zero_node_padding(tree: Any) -> None:
    """
    Set the padding bytes of a fitted `sklearn.tree._tree.Tree`'s node
    records to zero, in the tree's own memory.

    `Tree.__getstate__()["nodes"]` is a view onto the records the tree
    predicts from, and it is what pickle writes, padding included. No field
    is touched, so the tree predicts, reports and pickles every field as it
    did. A view that is not writeable, or not a plain run of records, is
    left alone.
    """
    nodes = tree.__getstate__()["nodes"]
    runs = _padding_runs(nodes.dtype)
    if (
        not runs
        or nodes.size == 0
        or not nodes.flags.writeable
        or not nodes.flags.c_contiguous
    ):
        return
    records = nodes.view("u1").reshape(nodes.size, nodes.dtype.itemsize)
    for start, stop in runs:
        records[:, start:stop] = 0


@lru_cache(maxsize=16)
def _padding_runs(dtype: Any) -> Tuple[Tuple[int, int], ...]:
    """`_padding_mask` as (start, stop) byte ranges -- (57, 64) for a node
    record -- so a record array is cleared by a strided write per range
    rather than through a boolean index."""
    mask = _padding_mask(dtype)
    if mask is None:
        return ()
    runs: List[Tuple[int, int]] = []
    start = None
    for offset, padding in enumerate([bool(b) for b in mask] + [False]):
        if padding and start is None:
            start = offset
        elif not padding and start is not None:
            runs.append((start, offset))
            start = None
    return tuple(runs)


def _trees_and_bin_mappers(obj: Any) -> Tuple[List[Any], List[Any]]:
    """
    Every fitted tree (`sklearn.tree._tree.Tree`) and every
    histogram-boosting bin mapper reachable from `obj`: through dicts,
    lists, tuples, object arrays (a gradient-boosting model's
    `estimators_`) and the attributes of scikit-learn and package objects
    (a forest's trees, a calibrated classifier's per-fold estimators, the
    quantile models' dict). Each is listed once.
    """
    import numpy as np

    try:
        from sklearn.tree._tree import Tree
    except ImportError:  # pragma: no cover - a scikit-learn without it
        Tree = None
    try:
        from sklearn.ensemble._hist_gradient_boosting.binning import _BinMapper
    except ImportError:  # pragma: no cover - a scikit-learn without it
        _BinMapper = None

    trees: List[Any] = []
    mappers: List[Any] = []
    seen = set()
    stack = [obj]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if Tree is not None and isinstance(node, Tree):
            trees.append(node)
        elif _BinMapper is not None and isinstance(node, _BinMapper):
            mappers.append(node)
        elif isinstance(node, dict):
            stack.extend(node.values())
        elif isinstance(node, (list, tuple)):
            stack.extend(node)
        elif isinstance(node, np.ndarray):
            if node.dtype == object:
                stack.extend(node.ravel().tolist())
        elif type(node).__module__.split(".", 1)[0] in _WALKED_PACKAGES:
            stack.extend(getattr(node, "__dict__", {}).values())
    return trees, mappers


@contextmanager
def reproducible_state(obj: Any) -> Iterator[None]:
    """
    Hold `obj` in the state its pickle should record while it is dumped.

    Every fitted tree's node padding is zeroed, in place and for good: the
    bytes belong to no field and are never read. Every histogram-boosting
    bin mapper's `n_threads` -- the OpenMP thread count its model was
    fitted under -- reads None, scikit-learn's own default for a bin
    mapper, and is restored on exit, so the model in memory is left as it
    was. scikit-learn reads that count only when the bin mapper bins data,
    which happens inside `fit` (a refit builds a new one); `predict` picks
    its thread count when it is called. A model loaded from the dump
    therefore predicts as the dumped one does, on the thread count the
    loading process gives it.
    """
    trees, mappers = _trees_and_bin_mappers(obj)
    for tree in trees:
        _zero_node_padding(tree)
    recorded = [(mapper, mapper.n_threads) for mapper in mappers]
    try:
        for mapper, _threads in recorded:
            mapper.n_threads = None
        yield
    finally:
        for mapper, threads in recorded:
            mapper.n_threads = threads


#: The date every member of a rewritten archive carries: the earliest a zip
#: entry can hold, and a constant.
_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)
#: Unix, whatever the machine: `ZipInfo` records 0 on Windows and 3
#: elsewhere, which would make the bytes depend on the operating system.
_CREATE_SYSTEM = 3
#: rw-------, what skops' own `writestr` records for every member.
_EXTERNAL_ATTR = 0o600 << 16
_SCHEMA = "schema.json"


def _is_id(key: str, value: Any) -> bool:
    return key == "__id__" and isinstance(value, int) and not isinstance(value, bool)


def _first_seen(node: Any, ids: Dict[int, int], files: List[str]) -> None:
    """Every `__id__` (an int) and every `file` (a member name) under
    `node`, numbered or listed in the order `schema.json` gives them."""
    if isinstance(node, dict):
        for key, value in node.items():
            if _is_id(key, value):
                ids.setdefault(value, len(ids) + 1)
            elif key == "file" and isinstance(value, str):
                if value not in files:
                    files.append(value)
            else:
                _first_seen(value, ids, files)
    elif isinstance(node, list):
        for value in node:
            _first_seen(value, ids, files)


def _renamed(node: Any, ids: Dict[int, int], names: Dict[str, str]) -> Any:
    if isinstance(node, dict):
        out = {}
        for key, value in node.items():
            if _is_id(key, value):
                out[key] = ids[value]
            elif key == "file" and isinstance(value, str):
                out[key] = names.get(value, value)
            else:
                out[key] = _renamed(value, ids, names)
        return out
    if isinstance(node, list):
        return [_renamed(value, ids, names) for value in node]
    return node


def _member_names(files: List[str], ids: Dict[int, int]) -> Dict[str, str]:
    """Each member `schema.json` names, renamed: `<id>.npy` after its
    object's new id, anything else (a bytes value's `<uuid>.bin`)
    `member-<n>` with its suffix, numbered in the same order."""
    names: Dict[str, str] = {}
    others = 0
    for name in files:
        stem, dot, suffix = name.rpartition(".")
        base, ext = (stem, f".{suffix}") if dot else (name, "")
        if base.isdigit() and int(base) in ids:
            names[name] = f"{ids[int(base)]}{ext}"
        else:
            others += 1
            names[name] = f"member-{others}{ext}"
    return names


def _zeroed_padding(payload: bytes) -> bytes:
    """
    An `.npy` member whose structured dtype leaves bytes that belong to no
    field, with those bytes set to zero; any other member as it was.

    A fitted tree's node array is such a dtype: seven fields of eight bytes
    and one of one, in a 64-byte record, so seven bytes per node are
    whatever the allocator left there. They are never read, and they made
    two dumps of one forest differ. The `.npy` header and every byte a
    field covers are kept as skops wrote them.
    """
    import numpy as np

    try:
        array = np.load(io.BytesIO(payload), allow_pickle=False)
    except Exception:  # noqa: BLE001 - not an array this can read: keep it
        return payload
    dtype = array.dtype
    padding = _padding_mask(dtype)
    if padding is None or array.size == 0:
        return payload  # unstructured or packed: nothing to zero
    # The records are the payload's last `nbytes`, each `itemsize` long in
    # either memory order; only the bytes no field covers are cleared, in
    # place, so the header and every field's bytes are the ones skops wrote.
    header = len(payload) - array.nbytes
    data = np.frombuffer(bytearray(payload[header:]), dtype=np.uint8).reshape(
        array.size, dtype.itemsize
    )
    data[:, padding] = 0
    return payload[:header] + data.tobytes()


def deterministic_archive(data: bytes) -> bytes:
    """
    A skops archive rewritten so the same model gives the same bytes: each
    `__id__` numbered by its first appearance in `schema.json`, each member
    renamed to match (see `_member_names`), members in skops' own order,
    each dated 1980-01-01 with fixed attributes and its own compression,
    and an array's padding bytes zero (see `_zeroed_padding`).
    `schema.json` is written back as skops writes it: two-space indent,
    keys in their order.

    An archive this does not recognise -- no `schema.json`, a `file` key
    naming a member the archive does not hold, a member no `file` key
    names -- is returned as it was: rewriting what is not understood could
    change what loads.
    """
    with zipfile.ZipFile(io.BytesIO(data)) as source:
        infos = source.infolist()
        members = {info.filename for info in infos}
        if _SCHEMA not in members:
            logger.debug("a skops archive without %s is kept as written", _SCHEMA)
            return data
        schema = json.loads(source.read(_SCHEMA))
        ids: Dict[int, int] = {}
        files: List[str] = []
        _first_seen(schema, ids, files)
        if set(files) | {_SCHEMA} != members:
            logger.debug(
                "a skops archive whose members are not the files its schema "
                "names is kept as written"
            )
            return data
        names = _member_names(files, ids)
        out = io.BytesIO()
        with zipfile.ZipFile(out, "w") as target:
            for info in infos:
                if info.filename == _SCHEMA:
                    renamed = _renamed(schema, ids, names)
                    payload = json.dumps(renamed, indent=2).encode("utf-8")
                elif info.filename.endswith(".npy"):
                    payload = _zeroed_padding(source.read(info.filename))
                else:
                    payload = source.read(info.filename)
                fixed = zipfile.ZipInfo(
                    names.get(info.filename, info.filename), date_time=_ZIP_EPOCH
                )
                fixed.compress_type = info.compress_type
                fixed.create_system = _CREATE_SYSTEM
                fixed.external_attr = _EXTERNAL_ATTR
                target.writestr(fixed, payload)
    return out.getvalue()


def untrusted_types(path: str) -> List[str]:
    """The types in a bundle beyond skops' defaults, as skops names them."""
    _require()
    import skops.io as sio

    return [str(t) for t in sio.get_untrusted_types(file=str(path))]


def load_estimator(path: str) -> Any:
    """
    Load a bundle, trusting skops' defaults, the scikit-learn types in
    `TRUSTED_TYPES` and this package's own types.

    Any other type is refused BY NAME before anything is constructed. That
    is the property the format is for: a bundle that names a type this
    registry never writes did not come from this registry.
    """
    _require()
    import skops.io as sio

    resolved = Path(path)
    if not resolved.exists():
        raise ValidationError(f"artifact not found: {path}")
    beyond_default = untrusted_types(str(resolved))
    foreign = [
        t
        for t in beyond_default
        if not (t.startswith(TRUSTED_PREFIX) or t in TRUSTED_TYPES)
    ]
    if foreign:
        raise ValidationError(
            f"{resolved.name} names type(s) outside skops' defaults, the "
            "scikit-learn types this library's estimators hold and this "
            f"package: {foreign[:5]}. A skops bundle is loaded without "
            "executing pickle precisely so an unknown type cannot run code; "
            "this bundle is refused, not loaded."
        )
    return sio.load(str(resolved), trusted=beyond_default)


__all__ = [
    "FORMATS",
    "FORMAT_ENV",
    "TRUSTED_PREFIX",
    "TRUSTED_TYPES",
    "default_format",
    "deterministic_archive",
    "dump_estimator",
    "load_estimator",
    "reproducible_state",
    "save_joblib",
    "skops_available",
    "untrusted_types",
]
