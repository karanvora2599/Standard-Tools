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
LightGBM or XGBoost booster, whose loading would mean trusting that
library's own state, or a type from somewhere this registry did not
write.

WHEN THERE IS NO BUNDLE. skops cannot serialize everything (a booster
holding a native handle, say), and what it can serialize the loader may
refuse: a LightGBM or XGBoost model's booster. Registration then keeps
joblib alone, logs why -- the exception, or the types the loader would
refuse -- and the manifest's `formats` says `["joblib"]`; asking for
skops on such a model is refused by name rather than answered with
joblib. A bundle the loader refuses is not written, because a format
listed on the manifest that can never load is not a format the model
has.

THE SAME MODEL, THE SAME BYTES. A skops archive as `skops.io.dump` writes
it differs between two runs that fit the same model: each zip member
carries the wall-clock time it was written, each array is stored under
the memory address of the object it came from (`2269133876912.npy`) and
named by that address in `schema.json`'s `__id__` and `file` keys, a
bytes value is stored under a random UUID, and a tree's node records
carry seven padding bytes each, whatever the allocator left there. The
manifest hashes the file, so `model.skops`'s content hash never
reproduced. `dump_estimator` rewrites the archive before it is saved:
ids numbered 1, 2, 3 in the order `schema.json` first names them, each
member file renamed to match, padding bytes zero, and every member dated
1980-01-01 with fixed attributes. The loader reads ids only to tell one
object from another, members only by the names `schema.json` gives, and
an array's fields but never its padding; an archive written before loads
as it always did.

AND THE SAME BYTES WHEREVER THE MODEL CAME FROM. skops gives one `__id__`
to every reference to one object, so the ids also recorded which objects
the model in memory happened to share, and that is not the model's
state: a fit shares a float or a string between attributes that a load
holds as separate copies, joblib loads each reference to an array as its
own array, and a model loaded from the bundle shares every `inf`. The
same forest therefore wrote one bundle fitted and another loaded. The
rewrite numbers ids by what the nodes hold instead: a node that loads
as an immutable value (a number, string, None, type, function, bytes, a
numpy scalar, or a tuple of these) gets one id per distinct value, and
every other node -- an array, a list, a dict, an estimator, a random
state -- one id per place it appears, its arrays written once for each.
A fitted model, the same model loaded from its joblib, and the same
model loaded from its bundle write one bundle. What loads is the same
estimator, the same values and predictions; what changes is which
objects are one object: a mutable object two attributes shared loads as
two equal copies, as it does from the joblib, and equal immutable values
load as one.

AND THE SAME JOBLIB. `model.joblib` of a fresh fit reproduced, but some
things in it did not follow the model. A tree's node records are pickled
with their padding bytes, and a forest loaded from a file carries
whatever its loader left there, so two loads of one file re-dumped to two
hashes. A histogram-boosting model pickles its bin mapper's `n_threads`,
the OpenMP thread count it was fitted under (the physical core count
when nothing limits it), so the same model fitted under limits of 1 and
4 threads, or on a 10-core and a 2-core machine, gave different bytes.
A LightGBM model records the `n_jobs` the engine hands it -- its share
of the budget -- three times: as its own parameter, in its booster's
parameters, and in the booster's model text (`[num_threads: 4]`).
`save_joblib` writes the joblib, and `dump_estimator` the bundle, under
`reproducible_state`: node padding zeroed in place, `n_threads` recorded
as None, a LightGBM model's thread count recorded as 0 -- LightGBM's
documented "the OpenMP runtime's count" -- for the length of the dump,
then restored.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import re
import sys
import warnings
import zipfile
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from standard_quant_tools._env import env_str
from standard_quant_tools.artifact_store import hash_bytes
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
#: The skops release `TRUSTED_TYPES` was read off, and the oldest the
#: `skops` extra allows. Another release's defaults can differ, so its
#: `get_untrusted_types` can name types this list lacks, which the loader
#: refuses by name.
TRUSTED_TYPES_SKOPS = "0.15"

#: Where `reproducible_state` looks for trees, bin mappers and LightGBM
#: models: objects from these packages are walked through their
#: attributes; any other object (an XGBoost booster, say) is not entered.
_WALKED_PACKAGES = ("sklearn", "standard_quant_tools")
#: What a LightGBM model's thread count reads while it is dumped: 0 is
#: LightGBM's documented "the OpenMP runtime's count", read when predict
#: is called. (None would be the physical core count, set by LightGBM
#: whatever OpenMP limit the caller holds.)
_LIGHTGBM_DUMPED_THREADS = 0
#: `num_threads` and the aliases LightGBM documents for it.
_LIGHTGBM_THREAD_PARAMS = frozenset(
    {"num_threads", "num_thread", "nthread", "nthreads", "n_jobs"}
)
#: The line of a LightGBM model text that records its thread count.
_LIGHTGBM_THREADS_LINE = re.compile(r"^\[num_threads: [^\]\n]*\]$", re.MULTILINE)


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


def _refused(types: List[str]) -> List[str]:
    """The types among `types` (what `get_untrusted_types` names) that
    `load_estimator` refuses: neither this package's nor in
    `TRUSTED_TYPES`."""
    return [
        t for t in types if not (t.startswith(TRUSTED_PREFIX) or t in TRUSTED_TYPES)
    ]


def _release(version: str) -> Tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", version)[:2])


def _older_skops_note() -> str:
    """A sentence naming the installed skops when it is older than the one
    `TRUSTED_TYPES` was read off, else empty."""
    try:
        import skops
    except ImportError:  # pragma: no cover - callers have checked
        return ""
    installed = str(getattr(skops, "__version__", ""))
    if not installed or _release(installed) >= _release(TRUSTED_TYPES_SKOPS):
        return ""
    return (
        f" TRUSTED_TYPES was derived on skops {TRUSTED_TYPES_SKOPS}; this "
        f"process has skops {installed}, whose defaults can name types "
        f"skops {TRUSTED_TYPES_SKOPS} does not. The `skops` extra requires "
        f"{TRUSTED_TYPES_SKOPS} or later."
    )


def _bundle(estimator: Any, level: int = logging.WARNING) -> Optional[bytes]:
    """
    The archive `dump_estimator` writes for `estimator`: dumped under
    `reproducible_state` and rewritten by `deterministic_archive`. None
    -- logged at `level` -- when skops is not installed, cannot serialize
    the estimator or read back what it wrote, or writes a type
    `load_estimator` refuses.
    """
    if not skops_available():
        return None
    import skops.io as sio

    buffer = io.BytesIO()
    try:
        with reproducible_state(estimator):
            sio.dump(estimator, buffer)
    except Exception as exc:  # noqa: BLE001 - any failure means "no bundle"
        logger.log(
            level,
            "[modeling] skops could not serialize %s (%s); the model is "
            "registered with joblib only.",
            type(estimator).__name__,
            exc,
        )
        return None
    data = buffer.getvalue()
    try:
        refused = _refused([str(t) for t in sio.get_untrusted_types(data=data)])
    except Exception as exc:  # noqa: BLE001 - a bundle skops cannot audit
        logger.log(
            level,
            "[modeling] skops could not read back the bundle of %s (%s); the "
            "model is registered with joblib only.",
            type(estimator).__name__,
            exc,
        )
        return None
    if refused:
        logger.log(
            level,
            "[modeling] the skops bundle of %s would hold type(s) the loader "
            "refuses: %s; the model is registered with joblib only.%s",
            type(estimator).__name__,
            refused[:5],
            _older_skops_note(),
        )
        return None
    return deterministic_archive(data)


def dump_estimator(directory: Path, name: str, estimator: Any) -> Optional[str]:
    """
    Write `<name>.skops` under `directory`, atomically; None when skops is
    not installed, cannot serialize this estimator, or would write a type
    `load_estimator` refuses (a LightGBM or XGBoost booster), which is
    logged and recorded on the manifest rather than raised -- the joblib
    is the format every registration has, the bundle is the one it has
    when it can be loaded.
    """
    data = _bundle(estimator)
    if data is None:
        return None
    path = Path(directory) / f"{name}.skops"
    _artifacts._atomic_write_bytes(path, data)
    return str(path)


def state_hash(estimator: Any) -> Optional[str]:
    """
    The content hash registration records for `estimator`'s `model.skops`,
    computed in memory: the same for a fitted model, for that model loaded
    from its joblib and for that model loaded from its bundle. None when
    registration would write no bundle for it (see `dump_estimator`).
    """
    data = _bundle(estimator, level=logging.DEBUG)
    return None if data is None else hash_bytes(data)


def save_joblib(directory: Path, name: str, obj: Any) -> str:
    """
    `artifacts.save_joblib` under `reproducible_state`: the same model gives
    the same `<name>.joblib`, whether it was fitted here, loaded from a file,
    or fitted under another thread count. `obj` may be an estimator or a
    container of them (the quantile models' dict).
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


def _fitted_parts(obj: Any) -> Tuple[List[Any], List[Any], List[Any]]:
    """
    Every fitted tree (`sklearn.tree._tree.Tree`), every histogram-boosting
    bin mapper and every LightGBM scikit-learn model reachable from `obj`:
    through dicts, lists, tuples, object arrays (a gradient-boosting
    model's `estimators_`) and the attributes of scikit-learn and package
    objects (a forest's trees, a calibrated classifier's per-fold
    estimators, the quantile models' dict). Each is listed once. LightGBM
    is looked for only once something has imported it: no model of it can
    exist before.
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
    LGBMModel = getattr(sys.modules.get("lightgbm.sklearn"), "LGBMModel", None)

    trees: List[Any] = []
    mappers: List[Any] = []
    lightgbm: List[Any] = []
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
        elif LGBMModel is not None and isinstance(node, LGBMModel):
            lightgbm.append(node)
        elif isinstance(node, dict):
            stack.extend(node.values())
        elif isinstance(node, (list, tuple)):
            stack.extend(node)
        elif isinstance(node, np.ndarray):
            if node.dtype == object:
                stack.extend(node.ravel().tolist())
        elif type(node).__module__.split(".", 1)[0] in _WALKED_PACKAGES:
            stack.extend(getattr(node, "__dict__", {}).values())
    return trees, mappers, lightgbm


def _trees_and_bin_mappers(obj: Any) -> Tuple[List[Any], List[Any]]:
    """The trees and histogram-boosting bin mappers `_fitted_parts` finds."""
    trees, mappers, _lightgbm = _fitted_parts(obj)
    return trees, mappers


def _thread_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """`params` (a LightGBM booster's) with every name of `num_threads` it
    holds reading `_LIGHTGBM_DUMPED_THREADS`, keys in their order."""
    return {
        key: (_LIGHTGBM_DUMPED_THREADS if key in _LIGHTGBM_THREAD_PARAMS else value)
        for key, value in params.items()
    }


def _booster_without_threads(booster: Any) -> Any:
    """
    A stand-in for a LightGBM booster that pickles as it does except for
    its thread count: built from the booster's model text with the
    `[num_threads: N]` line reading `_LIGHTGBM_DUMPED_THREADS`, carrying
    the booster's own attributes, its parameters' thread count likewise.
    The booster itself is not touched.
    """
    import lightgbm

    text = booster.model_to_string(num_iteration=-1)
    stand_in = lightgbm.Booster(
        model_str=_LIGHTGBM_THREADS_LINE.sub(
            f"[num_threads: {_LIGHTGBM_DUMPED_THREADS}]", text
        )
    )
    handle = stand_in._handle
    state = dict(booster.__dict__)
    state["_handle"] = handle
    if isinstance(state.get("params"), dict):
        state["params"] = _thread_params(state["params"])
    stand_in.__dict__.clear()
    stand_in.__dict__.update(state)
    return stand_in


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

    A LightGBM model's `n_jobs` reads 0 and its booster is swapped for one
    whose parameters and model text record 0 threads, and both are put
    back on exit. LightGBM's scikit-learn `predict` reads `n_jobs` when it
    is called and hands 0 to its C++ library as "the OpenMP runtime's
    count", so a loaded model predicts on whatever count the loading
    process's OpenMP limit gives it -- the count `score_model` sets --
    with the same predictions: LightGBM predicts each row on its own.
    The booster's recorded counts are not read by `predict`.
    """
    trees, mappers, lightgbm = _fitted_parts(obj)
    for tree in trees:
        _zero_node_padding(tree)
    recorded = [(mapper, mapper.n_threads) for mapper in mappers]
    boosted = [
        (model, model.n_jobs, getattr(model, "_Booster", None)) for model in lightgbm
    ]
    try:
        for mapper, _threads in recorded:
            mapper.n_threads = None
        for model, _jobs, booster in boosted:
            # Set as an attribute: `set_params` would also copy the value
            # into the model's `_other_params`, which is pickled too.
            model.n_jobs = _LIGHTGBM_DUMPED_THREADS
            if booster is not None:
                model._Booster = _booster_without_threads(booster)
        yield
    finally:
        for mapper, threads in recorded:
            mapper.n_threads = threads
        for model, jobs, booster in boosted:
            model.n_jobs = jobs
            if booster is not None:
                model._Booster = booster


#: The date every member of a rewritten archive carries: the earliest a zip
#: entry can hold, and a constant.
_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)
#: Unix, whatever the machine: `ZipInfo` records 0 on Windows and 3
#: elsewhere, which would make the bytes depend on the operating system.
_CREATE_SYSTEM = 3
#: rw-------, what skops' own `writestr` records for every member.
_EXTERNAL_ATTR = 0o600 << 16
_SCHEMA = "schema.json"
#: A node that names another node's `__id__` rather than carrying its own
#: state. skops' loader reads it; skops 0.15's writer does not emit it.
_REFERENCE_LOADER = "CachedNode"
#: Node loaders whose object is an immutable value whatever object it was
#: written from: JSON numbers, strings, booleans and None; types; module
#: functions; bytes; slices. A tuple of values is one too, and so is a
#: numpy scalar (see `_loads_as_value`).
_VALUE_LOADERS = frozenset(
    {"JsonNode", "TypeNode", "FunctionNode", "BytesNode", "SliceNode"}
)


def _is_id(key: str, value: Any) -> bool:
    return key == "__id__" and isinstance(value, int) and not isinstance(value, bool)


def _is_node(value: Any) -> bool:
    return isinstance(value, dict) and _is_id("__id__", value.get("__id__"))


@lru_cache(maxsize=64)
def _is_numpy_scalar(name: str) -> bool:
    """Whether numpy's attribute `name` is an immutable scalar type."""
    import numpy as np

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kind = getattr(np, name, None)
    return (
        isinstance(kind, type)
        and issubclass(kind, np.generic)
        and not issubclass(kind, np.void)
    )


def _loads_as_value(node: Dict[str, Any]) -> bool:
    """Whether `node` loads as an immutable value, given that every node it
    holds does: see `_VALUE_LOADERS`, a tuple, or a numpy scalar."""
    loader = node.get("__loader__")
    if loader in _VALUE_LOADERS or loader == "TupleNode":
        return True
    return (
        loader == "NdArrayNode"
        and node.get("type") == "numpy"
        and node.get("__module__") == "numpy"
        and isinstance(node.get("__class__"), str)
        and _is_numpy_scalar(node["__class__"])
    )


class _Survey:
    """
    What one pass over `schema.json` finds: every member a `file` key
    names, whether a reference node is present, whether a `file` key sits
    outside a node, and a digest of what each value node holds -- every key
    but `__id__`, a member by its bytes, a node inside by its digest --
    keyed by the `id()` of the node's dict. A value node is one that loads
    as an immutable value and holds nothing that does not.
    """

    def __init__(self, schema: Any, payloads: Dict[str, bytes]) -> None:
        self.files: set = set()
        self.referenced = False
        self.loose = False
        self.digests: Dict[int, str] = {}
        self._payloads = payloads
        self._visit(schema)

    def _visit(self, value: Any) -> Tuple[Any, bool]:
        # (what `value` holds, as a digest can be taken of it; whether
        # every node inside it is a value node)
        if isinstance(value, dict):
            node = _is_node(value)
            original = value.get("file")
            if isinstance(original, str):
                self.files.add(original)
                self.loose = self.loose or not node
            if value.get("__loader__") == _REFERENCE_LOADER:
                self.referenced = True
            if node and not _loads_as_value(value):
                # Not a value whatever it holds: only the nodes inside it
                # can be.
                for key, item in value.items():
                    if key != "file":
                        self._visit(item)
                return None, False
            form: Dict[str, Any] = {}
            pure = True
            for key, item in value.items():
                if node and key == "__id__":
                    continue
                if key == "file" and isinstance(item, str):
                    payload = self._payloads.get(item, b"")
                    form[key] = hashlib.sha256(payload).hexdigest()
                    continue
                form[key], held = self._visit(item)
                pure = pure and held
            if not node or not pure:
                return form, pure
            digest = hashlib.sha256(
                json.dumps(form, sort_keys=True).encode("utf-8")
            ).hexdigest()
            self.digests[id(value)] = digest
            return digest, True
        if isinstance(value, list):
            forms = [self._visit(item) for item in value]
            return [form for form, _ in forms], all(held for _, held in forms)
        return value, True


class _Conflict(Exception):
    """Two places that would share a member name hold different bytes."""


class _Rewrite:
    """
    `schema.json` with every `__id__` and `file` renamed (`schema`), and
    {new member name: the member it is written from} in the order the
    schema names them (`sources`). Raises `_Conflict` when two places
    that would share a name hold different bytes.

    Ids are numbered in the order `schema.json` names them: a value node's
    by its digest (one id per distinct value), any other node's by where
    it appears (one id per place). `as_recorded` keeps the ids' own
    grouping instead -- one new id per id written -- for an archive holding
    a reference node, whose `__id__` names another node's.

    A member named after its node's `__id__` (`<id>.npy`) takes the node's
    new id; any other (a bytes value's `<uuid>.bin`) is `member-<n>` with
    its suffix, numbered in the order the schema names them, one per value
    for a value node.
    """

    def __init__(
        self,
        schema: Any,
        survey: _Survey,
        payloads: Dict[str, bytes],
        as_recorded: bool,
    ) -> None:
        self._digests = {} if as_recorded else survey.digests
        self._payloads = payloads
        self._as_recorded = as_recorded
        self._ids: Dict[Tuple[str, Any], int] = {}
        self._places = 0
        self._others: Dict[Tuple[str, str], str] = {}
        self.sources: Dict[str, str] = {}
        self.schema = self._render(schema)

    def _render(self, value: Any) -> Any:
        if isinstance(value, dict):
            out: Dict[str, Any] = {}
            new_id = None
            for key, item in value.items():
                if _is_id(key, item):
                    if self._as_recorded:
                        group: Tuple[str, Any] = ("id", item)
                    elif id(value) in self._digests:
                        group = ("value", self._digests[id(value)])
                    else:
                        group = ("place", self._places)
                    self._places += 1
                    new_id = self._ids.setdefault(group, len(self._ids) + 1)
                    out[key] = new_id
                elif key == "file" and isinstance(item, str):
                    out[key] = item  # renamed below, once the node's id is known
                else:
                    out[key] = self._render(item)
            original = value.get("file")
            if isinstance(original, str):
                out["file"] = self._member(value, original, new_id)
            return out
        if isinstance(value, list):
            return [self._render(item) for item in value]
        return value

    def _member(self, node: Dict[str, Any], original: str, new_id: Any) -> str:
        stem, dot, suffix = original.rpartition(".")
        base, ext = (stem, f".{suffix}") if dot else (original, "")
        if base == str(node["__id__"]):
            name = f"{new_id}{ext}"
        else:
            group = (
                ("value", self._digests[id(node)])
                if id(node) in self._digests
                else ("file", original)
            )
            if group not in self._others:
                self._others[group] = f"member-{len(self._others) + 1}{ext}"
            name = self._others[group]
        earlier = self.sources.setdefault(name, original)
        if earlier != original and self._payloads[earlier] != self._payloads[original]:
            raise _Conflict(name)
        return name


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
    A skops archive rewritten so the same model gives the same bytes,
    whether it was fitted or loaded: ids numbered in the order
    `schema.json` names them -- one per distinct value for a node that
    loads as an immutable value, one per place for any other node -- and
    each member renamed to match and written once per name (see
    `_Rewrite`), in the order the schema names them (the order skops writes
    them) with `schema.json` last, each dated 1980-01-01 with fixed
    attributes and its own compression, and an array's padding bytes zero
    (see `_zeroed_padding`). `schema.json` is written back as skops writes
    it: two-space indent, keys in their order.

    An archive holding a reference node keeps its ids' grouping, numbered
    the same way. An archive this does not recognise -- no `schema.json`,
    a `file` key naming a member the archive does not hold or sitting
    outside a node, a member no `file` key names, one name for two
    different members -- is returned as it was: rewriting what is not
    understood could change what loads.
    """
    with zipfile.ZipFile(io.BytesIO(data)) as source:
        infos = source.infolist()
        members = {info.filename for info in infos}
        if _SCHEMA not in members:
            logger.debug("a skops archive without %s is kept as written", _SCHEMA)
            return data
        schema = json.loads(source.read(_SCHEMA))
        payloads = {
            info.filename: (
                _zeroed_padding(source.read(info.filename))
                if info.filename.endswith(".npy")
                else source.read(info.filename)
            )
            for info in infos
            if info.filename != _SCHEMA
        }
    survey = _Survey(schema, payloads)
    if survey.files | {_SCHEMA} != members or survey.loose:
        logger.debug(
            "a skops archive whose members are not the files its schema "
            "names is kept as written"
        )
        return data
    try:
        rewrite = _Rewrite(schema, survey, payloads, as_recorded=survey.referenced)
    except _Conflict:
        logger.debug("a skops archive naming two members alike is kept as written")
        return data
    compression = {info.filename: info.compress_type for info in infos}
    renamed = json.dumps(rewrite.schema, indent=2).encode("utf-8")
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as target:
        for name, original in [*rewrite.sources.items(), (_SCHEMA, _SCHEMA)]:
            fixed = zipfile.ZipInfo(name, date_time=_ZIP_EPOCH)
            fixed.compress_type = compression[original]
            fixed.create_system = _CREATE_SYSTEM
            fixed.external_attr = _EXTERNAL_ATTR
            target.writestr(
                fixed, renamed if original == _SCHEMA else payloads[original]
            )
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
    registry never writes did not come from this registry. Under a skops
    older than `TRUSTED_TYPES_SKOPS` the refusal says so.
    """
    _require()
    import skops.io as sio

    resolved = Path(path)
    if not resolved.exists():
        raise ValidationError(f"artifact not found: {path}")
    beyond_default = untrusted_types(str(resolved))
    foreign = _refused(beyond_default)
    if foreign:
        raise ValidationError(
            f"{resolved.name} names type(s) outside skops' defaults, the "
            "scikit-learn types this library's estimators hold and this "
            f"package: {foreign[:5]}. A skops bundle is loaded without "
            "executing pickle precisely so an unknown type cannot run code; "
            f"this bundle is refused, not loaded.{_older_skops_note()}"
        )
    return sio.load(str(resolved), trusted=beyond_default)


__all__ = [
    "FORMATS",
    "FORMAT_ENV",
    "TRUSTED_PREFIX",
    "TRUSTED_TYPES",
    "TRUSTED_TYPES_SKOPS",
    "default_format",
    "deterministic_archive",
    "dump_estimator",
    "load_estimator",
    "reproducible_state",
    "save_joblib",
    "skops_available",
    "state_hash",
    "untrusted_types",
]
