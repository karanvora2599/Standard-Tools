"""
A dataset's content hash does not depend on the pandas version.

`audit.hash_dataframe` covers each column's `str(dtype)`. pandas 3 prints a
text column as `str` where pandas 2 prints `object`, and parses a date
string to `datetime64[s]` where pandas 2 gives `datetime64[ns]`, so a
dataset built under one pandas was refused under the other as "no longer
matches the hash recorded when it was built" by all 21 tools that load a
dataset, though not a byte of it had changed (the CHANGELOG entry of
2026-10-04).

New datasets record `audit.canonical_frame_hash` as version 2, which covers
the same names, types and values through a representation that is the same
under every pandas. A dataset recorded before (version 1, no version key)
keeps its hash -- manifests and fold node hashes carry it -- and verifies
as read or under the other pandas's dtype names and datetime resolutions.

The literals below were computed under pandas 2.3.3 and pandas 3.0.5 and
are asserted under whichever one runs the suite: CI runs Python 3.10, which
resolves pandas 2, and Python 3.11 and 3.12, which resolve pandas 3.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.audit.hashing import canonical_frame_hash, hash_dataframe
from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling import artifacts as _artifacts
from standard_quant_tools.modeling.agent.models import (
    BuildModelDatasetInput,
    InspectModelInput,
    RegisterExternalPanelInput,
    RunModelExperimentInput,
)
from standard_quant_tools.modeling.agent.preview_models import PlanModelExperimentInput
from standard_quant_tools.modeling.agent.preview_tools import plan_model_experiment
from standard_quant_tools.modeling.agent.tools import (
    _load_dataset_panel,
    build_model_dataset,
    inspect_model,
    register_external_panel,
    run_model_experiment,
)
from standard_quant_tools.modeling.dataset.integrity import (
    DATA_HASH_VERSION,
    panel_file_stats,
)
from standard_quant_tools.modeling.registry.model_registry import load_manifest
from standard_quant_tools.modeling.specs import (
    DatasetSpec,
    EstimatorSpec,
    FeatureSpec,
    ModelSpec,
    TargetSpec,
    ValidationSpec,
)

PANDAS_3 = int(pd.__version__.split(".")[0]) >= 3

#: `hash_dataframe` of `_tiny()` as each pandas computes it: the values are
#: hashed alike, and the schema says `object` under pandas 2, `str` under 3.
TINY_LEGACY_PANDAS_2 = "2c7c3d523759a27c"
TINY_LEGACY_PANDAS_3 = "dd82d212c60d10d0"
#: `canonical_frame_hash` of `_tiny()`, the same under both.
TINY_CANONICAL = "f446f07073bcbb43"
#: `canonical_frame_hash` of `_kinds()`, the same under both.
KINDS_CANONICAL = "a89da2bf0512cf3d"

#: An external CSV and the `hash_dataframe` of the panel each pandas loads
#: from it: `datetime64[ns]` and `object` under pandas 2, `datetime64[s]`
#: and `str` under pandas 3 (with NaT label ends on the last date).
CSV_TEXT = (
    "date,entity,alpha,target\n"
    "2024-01-02,AAA,0.1,0.5\n"
    "2024-01-02,BBB,-0.2,-0.5\n"
    "2024-01-03,AAA,0.3,0.25\n"
    "2024-01-03,BBB,-0.4,-0.25\n"
    "2024-01-04,AAA,0.5,0.125\n"
    "2024-01-04,BBB,-0.6,-0.125\n"
)
CSV_LEGACY_PANDAS_2 = "6728fe0236f3dbac"
CSV_LEGACY_PANDAS_3 = "8ab830a79f9382e4"
CSV_CANONICAL = "50cdae16ced3062c"

INTEGRITY_LOGGER = "standard_quant_tools.modeling.dataset.integrity"


def _tiny() -> pd.DataFrame:
    dates = pd.DatetimeIndex(
        ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"], tz="UTC"
    ).as_unit("ns")
    panel = pd.DataFrame(
        {
            "date": dates,
            "entity": ["AAA", "BBB", "AAA", "BBB"],
            "f1": [0.1, -0.2, 0.3, -0.4],
            "target": [0.5, -0.5, 0.25, -0.25],
        }
    )
    panel["label_end_date"] = (panel["date"] + pd.Timedelta(days=1)).dt.as_unit("ns")
    return panel


def _kinds() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "i64": np.array([1, -2, 3], dtype="int64"),
            "i_na": pd.array([1, None, 3], dtype="Int64"),
            "u8": np.array([0, 7, 255], dtype="uint8"),
            "flag": [True, False, True],
            "flag_na": pd.array([True, None, False], dtype="boolean"),
            "f32": np.array([0.5, np.nan, -1.25], dtype="float32"),
            "cat": pd.Categorical(["x", "y", "x"]),
            "span": pd.to_timedelta(["1D", "2h", None]),
            "ny": pd.DatetimeIndex(
                ["2024-03-10 01:00", "2024-03-10 03:00", None], tz="America/New_York"
            ),
            "mixed": pd.Series([1, "a", None], dtype=object),
            "text_na": pd.Series(["a", None, np.nan], dtype=object),
        }
    )
    frame.index = pd.Index(["r1", "r2", "r3"], name="row")
    return frame


def _legacy_respelled(panel: pd.DataFrame, text_dtype: str) -> str:
    """`hash_dataframe` as a pandas that spells every text column
    `text_dtype` would compute it: the values digest is the one this pandas
    computes, only the schema differs."""
    rows = pd.util.hash_pandas_object(panel, index=True).to_numpy()
    digest = hashlib.sha256(np.asarray(rows).tobytes()).hexdigest()
    schema = json.dumps(
        [
            [str(name), text_dtype if name == "entity" else str(dtype)]
            for name, dtype in zip(panel.columns, panel.dtypes)
        ],
        sort_keys=False,
    )
    return hashlib.sha256(f"{schema}|{digest}".encode("utf-8")).hexdigest()[:16]


def _legacy_dataset(panel: pd.DataFrame, data_hash: str, **extra) -> str:
    """A dataset directory as a build before version 2 left it: the panel
    and a `data_hash` with no version key."""
    dataset_id = f"ds_{uuid.uuid4().hex[:12]}"
    _artifacts.save_artifact(panel, run_id=dataset_id, name="panel")
    _artifacts.save_json(
        _artifacts.run_dir(dataset_id),
        "dataset_meta",
        {
            "feature_ids": ["f1"],
            "target_id": "forward_return:1",
            "data_hash": data_hash,
            **extra,
        },
    )
    return dataset_id


#: 2026-10-01 12:00:00 UTC in nanoseconds: a modification time set
#: explicitly, so a test does not depend on the file system's clock
#: granularity (two writes within one tick share a time on some).
REGISTERED_NS = 1_790_856_000_000_000_000
REGISTERED_UTC = "2026-10-01 12:00:00.000000000 UTC"


def _write_csv(path, text: str = CSV_TEXT, mtime_ns=None) -> str:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(text)
    if mtime_ns is not None:
        os.utime(path, ns=(mtime_ns, mtime_ns))
    return str(path)


def _refusal(dataset_id: str) -> str:
    with pytest.raises(ValidationError) as caught:
        _load_dataset_panel(dataset_id)
    return str(caught.value)


def _register_csv(path):
    return register_external_panel(
        RegisterExternalPanelInput(
            path=str(path),
            targets=[
                {
                    "name": "primary",
                    "column": "target",
                    "horizon": 1,
                    "target_type": "forward_return",
                }
            ],
        )
    )


def _meta_path(dataset_id: str):
    return _artifacts.run_dir(dataset_id) / "dataset_meta.json"


def _rewrite_meta(dataset_id: str, **changes) -> dict:
    """Edit dataset_meta.json in place; a value of None deletes the key."""
    path = _meta_path(dataset_id)
    meta = json.loads(path.read_text(encoding="utf-8"))
    for key, value in changes.items():
        if value is None:
            meta.pop(key, None)
        else:
            meta[key] = value
    path.write_text(json.dumps(meta), encoding="utf-8")
    return meta


class TestTheCanonicalHash:
    def test_the_pinned_values_hold_under_this_pandas(self):
        """The same literal under pandas 2 and pandas 3: the point of the
        hash. `hash_dataframe` of the same frame is pinned beside it to show
        the difference being removed is real under this pandas."""
        assert canonical_frame_hash(_tiny()) == TINY_CANONICAL
        assert canonical_frame_hash(_kinds()) == KINDS_CANONICAL
        expected_legacy = TINY_LEGACY_PANDAS_3 if PANDAS_3 else TINY_LEGACY_PANDAS_2
        assert hash_dataframe(_tiny()) == expected_legacy

    @pytest.mark.parametrize("dtype", [object, "string"])
    def test_text_storage_does_not_change_it(self, dtype):
        panel = _tiny()
        panel["entity"] = panel["entity"].astype(dtype)
        assert canonical_frame_hash(panel) == TINY_CANONICAL

    def test_none_and_nan_in_a_text_column_hash_alike(self):
        with_none = pd.DataFrame({"t": pd.Series(["a", None, "b"], dtype=object)})
        with_nan = pd.DataFrame({"t": pd.Series(["a", np.nan, "b"], dtype=object)})
        as_string = with_none.astype("string")
        assert (
            canonical_frame_hash(with_none)
            == canonical_frame_hash(with_nan)
            == canonical_frame_hash(as_string)
        )

    @pytest.mark.parametrize("unit", ["us", "ms", "s"])
    def test_datetime_resolution_does_not_change_it(self, unit):
        panel = _tiny()
        for column in ("date", "label_end_date"):
            panel[column] = panel[column].dt.as_unit(unit)
        assert canonical_frame_hash(panel) == TINY_CANONICAL

    def test_a_materialized_default_index_is_the_range_index(self):
        panel = _tiny()
        panel.index = pd.Index(np.arange(len(panel), dtype="int64"))
        assert canonical_frame_hash(panel) == TINY_CANONICAL

    def test_nullable_and_numpy_storage_hash_alike(self):
        numpy_backed = pd.DataFrame(
            {"i": np.array([1, 2], dtype="int64"), "b": [True, False]}
        )
        nullable = numpy_backed.astype({"i": "Int64", "b": "boolean"})
        assert canonical_frame_hash(numpy_backed) == canonical_frame_hash(nullable)

    @pytest.mark.parametrize(
        "edit",
        [
            "one_ulp",
            "drop_last_row",
            "relabel_entity",
            "date_plus_one_day",
            "reorder_columns",
            "rename_column",
            "float32",
            "nan",
            "drop_time_zone",
            "swap_rows",
        ],
    )
    def test_every_edit_changes_it(self, edit):
        panel = _tiny()
        if edit == "one_ulp":
            panel.loc[0, "f1"] = np.nextafter(panel.loc[0, "f1"], np.inf)
        elif edit == "drop_last_row":
            panel = panel.iloc[:-1]
        elif edit == "relabel_entity":
            panel.loc[0, "entity"] = "AAB"
        elif edit == "date_plus_one_day":
            panel.loc[0, "date"] = panel.loc[0, "date"] + pd.Timedelta(days=1)
        elif edit == "reorder_columns":
            panel = panel[list(reversed(panel.columns))]
        elif edit == "rename_column":
            panel = panel.rename(columns={"f1": "f2"})
        elif edit == "float32":
            panel["f1"] = panel["f1"].astype("float32")
        elif edit == "nan":
            panel.loc[0, "f1"] = np.nan
        elif edit == "drop_time_zone":
            panel["date"] = panel["date"].dt.tz_localize(None)
        elif edit == "swap_rows":
            panel = panel.iloc[[1, 0, 2, 3]].reset_index(drop=True)
        assert canonical_frame_hash(panel) != TINY_CANONICAL

    def test_frames_without_rows_or_columns_hash(self):
        assert canonical_frame_hash(pd.DataFrame()) != canonical_frame_hash(
            pd.DataFrame({"a": pd.Series([], dtype=float)})
        )


class TestAVersion1DatasetLoadsUnderEitherPandas:
    @pytest.mark.parametrize("recorded", [TINY_LEGACY_PANDAS_2, TINY_LEGACY_PANDAS_3])
    def test_a_hash_recorded_under_either_pandas_verifies(self, recorded):
        """The hash a pandas-2 build recorded and the one a pandas-3 build
        recorded both verify here: one as read, the other with the text
        column's dtype spelled the way the other pandas spells it."""
        dataset_id = _legacy_dataset(_tiny(), recorded)
        before = _meta_path(dataset_id).read_bytes()
        panel, meta, _ = _load_dataset_panel(dataset_id)
        assert meta["data_hash"] == recorded
        assert canonical_frame_hash(panel) == TINY_CANONICAL
        # Verified, never rewritten: manifests already carry this value.
        assert _meta_path(dataset_id).read_bytes() == before

    def test_a_variant_match_logs_one_debug_line_and_a_direct_match_none(self, caplog):
        own = TINY_LEGACY_PANDAS_3 if PANDAS_3 else TINY_LEGACY_PANDAS_2
        other = TINY_LEGACY_PANDAS_2 if PANDAS_3 else TINY_LEGACY_PANDAS_3
        caplog.set_level(logging.DEBUG, logger=INTEGRITY_LOGGER)
        _load_dataset_panel(_legacy_dataset(_tiny(), own))
        assert not [r for r in caplog.records if r.name == INTEGRITY_LOGGER]
        caplog.clear()
        _load_dataset_panel(_legacy_dataset(_tiny(), other))
        lines = [r.getMessage() for r in caplog.records if r.name == INTEGRITY_LOGGER]
        assert len(lines) == 1
        assert other in lines[0] and "text spelled" in lines[0]

    def test_an_edited_panel_is_refused_naming_this_pandas(self):
        """The footer says this pandas wrote the panel, so a miss under the
        same pandas is an edit, and the refusal says so."""
        recorded = TINY_LEGACY_PANDAS_3 if PANDAS_3 else TINY_LEGACY_PANDAS_2
        edited = _tiny()
        edited.loc[0, "f1"] = 0.11
        dataset_id = _legacy_dataset(edited, recorded)
        with pytest.raises(ValidationError) as caught:
            _load_dataset_panel(dataset_id)
        message = str(caught.value)
        assert "no longer matches" in message
        assert (
            f"written by pandas {pd.__version__} and this process runs pandas "
            f"{pd.__version__}, so the data has changed" in message
        )
        assert str(_artifacts.run_dir(dataset_id) / "panel.parquet") in message

    def test_a_miss_across_pandas_versions_says_it_cannot_tell(self):
        edited = _tiny()
        edited.loc[0, "f1"] = 0.11
        dataset_id = _legacy_dataset(
            edited, TINY_LEGACY_PANDAS_2, built_with={"pandas": "1.5.3"}
        )
        with pytest.raises(ValidationError) as caught:
            _load_dataset_panel(dataset_id)
        message = str(caught.value)
        assert "could not be verified" in message
        assert (
            f"written by pandas 1.5.3 and this process runs pandas {pd.__version__}"
            in message
        )
        assert "cannot tell which" in message
        assert "Rebuilding the dataset records a hash that does not depend" in message

    def test_a_version_2_hash_without_its_key_verifies(self):
        """Metadata written by hand from a build's `data_hash` that left out
        `data_hash_version`: still a full content hash, so it verifies."""
        panel, _meta, _ = _load_dataset_panel(_legacy_dataset(_tiny(), TINY_CANONICAL))
        assert len(panel) == 4

    def test_a_dataset_without_a_hash_is_not_checked(self):
        dataset_id = _legacy_dataset(_tiny(), "0" * 16)
        _rewrite_meta(dataset_id, data_hash=None)
        panel, _meta, _ = _load_dataset_panel(dataset_id)
        assert len(panel) == 4

    def test_an_unknown_version_is_refused(self):
        dataset_id = _legacy_dataset(_tiny(), TINY_CANONICAL, data_hash_version=3)
        with pytest.raises(ValidationError, match="recorded as version 3"):
            _load_dataset_panel(dataset_id)


class TestExternalPanels:
    def test_registration_records_version_2_and_loads(self, tmp_path):
        result = _register_csv(_write_csv(tmp_path / "panel.csv"))
        meta = json.loads(_meta_path(result.dataset_id).read_text(encoding="utf-8"))
        assert meta["data_hash"] == CSV_CANONICAL
        assert meta["data_hash_version"] == DATA_HASH_VERSION == 2
        assert meta["built_with"]["pandas"] == pd.__version__
        panel, _meta, _ = _load_dataset_panel(result.dataset_id)
        assert canonical_frame_hash(panel) == CSV_CANONICAL

    @pytest.mark.parametrize("recorded", [CSV_LEGACY_PANDAS_2, CSV_LEGACY_PANDAS_3])
    def test_a_version_1_registration_under_either_pandas_loads(
        self, tmp_path, recorded
    ):
        """A CSV registered before version 2 recorded `hash_dataframe` of the
        panel as that pandas parsed it: `[ns]` dates and `object` text under
        pandas 2, `[s]` and `str` under pandas 3. Both verify here."""
        result = _register_csv(_write_csv(tmp_path / "panel.csv"))
        _rewrite_meta(
            result.dataset_id,
            data_hash=recorded,
            data_hash_version=None,
            built_with=None,
        )
        panel, meta, _ = _load_dataset_panel(result.dataset_id)
        assert meta["data_hash"] == recorded
        assert len(panel) == 6

    def test_an_edited_file_is_refused_by_its_own_path(self, tmp_path):
        """The refusal named `panel.parquet` for a panel that has none."""
        path = _write_csv(tmp_path / "panel.csv")
        result = _register_csv(path)
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write(CSV_TEXT.replace("0.1,0.5", "0.15,0.5"))
        with pytest.raises(ValidationError) as caught:
            _load_dataset_panel(result.dataset_id)
        message = str(caught.value)
        assert f"the panel at {path} no longer matches the content hash" in message
        assert "does not depend on the pandas or pyarrow version" in message
        assert "Register the panel again." in message
        assert "panel.parquet" not in message

    def test_an_edited_version_1_file_says_which_pandas_is_unknown(self, tmp_path):
        path = _write_csv(tmp_path / "panel.csv")
        result = _register_csv(path)
        own = CSV_LEGACY_PANDAS_3 if PANDAS_3 else CSV_LEGACY_PANDAS_2
        _rewrite_meta(
            result.dataset_id, data_hash=own, data_hash_version=None, built_with=None
        )
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write(CSV_TEXT.replace("0.1,0.5", "0.15,0.5"))
        with pytest.raises(ValidationError) as caught:
            _load_dataset_panel(result.dataset_id)
        message = str(caught.value)
        assert f"the panel at {path} could not be verified" in message
        assert (
            "The dataset does not record which pandas built it, and this process "
            f"runs pandas {pd.__version__}." in message
        )


class TestAnExternalRefusalSaysWhetherTheFileMoved:
    """`register_external_panel` recorded the file's name, size and
    modification time as `panel_fingerprint` and nothing read it, so a
    refused external panel said nothing about whether its file had been
    touched -- the fact that tells a pandas difference from an edit (the
    CHANGELOG entry of 2026-10-04). A registration now also records the
    three apart as `panel_file_stats`, and a refusal says which moved."""

    def test_registration_records_the_three_apart(self, tmp_path):
        path = _write_csv(tmp_path / "panel.csv", mtime_ns=REGISTERED_NS)
        result = _register_csv(path)
        meta = json.loads(_meta_path(result.dataset_id).read_text(encoding="utf-8"))
        stats = meta["panel_file_stats"]
        assert stats == panel_file_stats(path)
        assert stats["files"] == 1
        assert stats["bytes"] == len(CSV_TEXT.encode("utf-8"))
        assert stats["modified_ns"] == REGISTERED_NS
        assert meta["panel_fingerprint"] == result.fingerprint

    def test_an_unchanged_file_points_an_undecided_refusal_at_pandas(self, tmp_path):
        """A version-1 hash nothing reproduces, from a file nobody touched:
        the likelier cause is the pandas, and the refusal says so."""
        path = _write_csv(tmp_path / "panel.csv", mtime_ns=REGISTERED_NS)
        result = _register_csv(path)
        _rewrite_meta(
            result.dataset_id,
            data_hash="0" * 16,
            data_hash_version=None,
            built_with=None,
        )
        message = _refusal(result.dataset_id)
        assert f"the panel at {path} could not be verified" in message
        assert (
            "The file's name, size and modification time are unchanged since "
            "registration. Writing to a file moves its modification time unless "
            "something sets it back, so a pandas difference is the likelier "
            "cause." in message
        )
        assert message.endswith(
            "Registering the panel again records a hash that does not depend on "
            "the pandas version."
        )

    def test_an_edited_file_says_what_moved(self, tmp_path):
        path = _write_csv(tmp_path / "panel.csv", mtime_ns=REGISTERED_NS)
        result = _register_csv(path)
        own = CSV_LEGACY_PANDAS_3 if PANDAS_3 else CSV_LEGACY_PANDAS_2
        _rewrite_meta(
            result.dataset_id, data_hash=own, data_hash_version=None, built_with=None
        )
        size = len(CSV_TEXT.encode("utf-8"))
        _write_csv(
            path,
            CSV_TEXT.replace("0.1,0.5", "0.15,0.5"),
            mtime_ns=REGISTERED_NS + 1_500_000_000,
        )
        message = _refusal(result.dataset_id)
        assert (
            f"Since registration, the file's size changed from {size} to "
            f"{size + 1} bytes and its modification time changed from "
            f"{REGISTERED_UTC} to 2026-10-01 12:00:01.500000000 UTC. That points "
            "to an edit rather than a pandas difference." in message
        )

    def test_a_touched_file_names_the_time_alone(self, tmp_path):
        """The same bytes, a later modification time: only the time moved."""
        path = _write_csv(tmp_path / "panel.csv", mtime_ns=REGISTERED_NS)
        result = _register_csv(path)
        _rewrite_meta(result.dataset_id, data_hash="0" * 16)
        os.utime(path, ns=(REGISTERED_NS, REGISTERED_NS + 1_000))
        message = _refusal(result.dataset_id)
        assert (
            "Since registration, the file's modification time changed from "
            f"{REGISTERED_UTC} to 2026-10-01 12:00:00.000001000 UTC. Register "
            "the panel again." in message
        )

    def test_an_edit_that_keeps_size_and_time_is_still_refused(self, tmp_path):
        """Null case for the hint: an edit of the same length with the time
        set back leaves all three as registered. The content hash refuses
        it, and the sentence names the edit as one explanation."""
        path = _write_csv(tmp_path / "panel.csv", mtime_ns=REGISTERED_NS)
        result = _register_csv(path)
        _write_csv(path, CSV_TEXT.replace("0.1,0.5", "0.2,0.5"), mtime_ns=REGISTERED_NS)
        message = _refusal(result.dataset_id)
        assert f"the panel at {path} no longer matches the content hash" in message
        assert (
            "The file's name, size and modification time are unchanged since "
            "registration. Writing to a file moves its modification time unless "
            "something sets it back, so either an edit set it back or the file "
            "now parses to different values" in message
        )
        assert message.endswith("Register the panel again.")

    def test_a_registration_with_the_digest_alone_says_whether_it_moved(self, tmp_path):
        """A panel registered before `panel_file_stats` existed has only
        the digest: the refusal says whether the file moved, not which of
        the three did."""
        path = _write_csv(tmp_path / "panel.csv", mtime_ns=REGISTERED_NS)
        result = _register_csv(path)
        _rewrite_meta(
            result.dataset_id,
            panel_file_stats=None,
            data_hash="0" * 16,
            data_hash_version=None,
            built_with=None,
        )
        assert "unchanged since registration" in _refusal(result.dataset_id)

        _write_csv(path, mtime_ns=REGISTERED_NS + 1_000)
        message = _refusal(result.dataset_id)
        assert (
            "The file's name, size or modification time changed since "
            "registration; the registration recorded their digest alone, so "
            "which of them is not known. That points to an edit rather than a "
            "pandas difference." in message
        )

    def test_a_directory_names_the_files_added(self, tmp_path):
        """A partitioned panel with a partition added since registration."""
        directory = tmp_path / "panel"
        directory.mkdir()
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"]
                ),
                "entity": ["AAA", "BBB", "AAA", "BBB"],
                "alpha": [0.1, -0.2, 0.3, -0.4],
                "target": [0.5, -0.5, 0.25, -0.25],
            }
        )
        frame.iloc[:2].to_parquet(directory / "part-0.parquet", index=False)
        frame.iloc[2:].to_parquet(directory / "part-1.parquet", index=False)
        result = _register_csv(directory)

        later = frame.iloc[2:].assign(date=pd.to_datetime(["2024-01-04"] * 2))
        later.to_parquet(directory / "part-2.parquet", index=False)
        message = _refusal(result.dataset_id)
        assert (
            "Since registration, files were added to, removed from or renamed "
            f"in {directory} (2 files at registration, 3 now)." in message
        )


def _spec() -> DatasetSpec:
    return DatasetSpec(
        universe=["AAA", "BBB", "CCC"],
        start="2022-01-01",
        end="2023-12-31",
        features=[FeatureSpec(id="technical.rsi"), FeatureSpec(id="market.momentum")],
        target=TargetSpec(horizon=5),
        benchmark="SPY",
    )


def _ridge() -> ModelSpec:
    return ModelSpec(
        task="regression",
        estimator=EstimatorSpec(type="ridge", params={"alpha": 1.0}),
        validation=ValidationSpec(train_window=150, test_window=30, embargo=5),
        random_seed=1,
    )


class TestBuiltDatasets:
    @pytest.fixture
    def dataset_id(self, patched_multi_factory):
        return build_model_dataset(BuildModelDatasetInput(spec=_spec())).dataset_id

    def test_a_build_records_version_2_with_its_libraries(self, dataset_id):
        """The recorded hash is the one the reloaded panel reproduces, not
        only the one the in-memory frame had."""
        meta = json.loads(_meta_path(dataset_id).read_text(encoding="utf-8"))
        assert meta["data_hash_version"] == 2
        assert set(meta["built_with"]) == {"python", "pandas", "numpy", "pyarrow"}
        assert meta["built_with"]["pandas"] == pd.__version__
        reloaded = _artifacts.load_artifact(
            str(_artifacts.run_dir(dataset_id) / "panel.parquet")
        )
        assert meta["data_hash"] == canonical_frame_hash(reloaded)

    def test_an_edited_panel_is_refused_in_the_version_2_wording(self, dataset_id):
        path = _artifacts.run_dir(dataset_id) / "panel.parquet"
        panel = _artifacts.load_artifact(str(path))
        panel.loc[0, "target"] = panel.loc[0, "target"] + 1e-9
        panel.to_parquet(path)
        with pytest.raises(ValidationError) as caught:
            _load_dataset_panel(dataset_id)
        message = str(caught.value)
        assert f"the panel at {path} no longer matches the content hash" in message
        assert "covers the column names, their types and every value" in message
        assert "does not depend on the pandas or pyarrow version" in message
        assert message.endswith("Rebuild the dataset.")

    def test_a_run_records_the_version_and_inspect_model_shows_it(self, dataset_id):
        result = run_model_experiment(
            RunModelExperimentInput(dataset_id=dataset_id, spec=_ridge())
        )
        manifest = load_manifest(result.model_id)
        meta = json.loads(_meta_path(dataset_id).read_text(encoding="utf-8"))
        assert manifest.dataset_hash == meta["data_hash"]
        assert manifest.dataset_hash_version == 2
        lineage = inspect_model(
            InspectModelInput(model_id=result.model_id, view="lineage")
        )
        assert lineage.data["dataset_hash_version"] == 2

    @pytest.mark.parametrize("spelling", ["as read", "the other pandas"])
    def test_a_version_1_dataset_keeps_its_identity(self, dataset_id, spelling):
        """A run on a dataset recorded before version 2 records the stored
        hash -- not the canonical one, not a recomputed one -- and its fold
        node hashes are the ones that hash produces, so runs on it before
        and after this release share their identity."""
        reloaded = _artifacts.load_artifact(
            str(_artifacts.run_dir(dataset_id) / "panel.parquet")
        )
        assert _legacy_respelled(reloaded, str(reloaded["entity"].dtype)) == (
            hash_dataframe(reloaded)
        )
        if spelling == "as read":
            stored = hash_dataframe(reloaded)
        else:
            stored = _legacy_respelled(reloaded, "object" if PANDAS_3 else "str")
        assert stored != canonical_frame_hash(reloaded)
        _rewrite_meta(
            dataset_id, data_hash=stored, data_hash_version=None, built_with=None
        )
        before = _meta_path(dataset_id).read_bytes()

        planned = plan_model_experiment(
            PlanModelExperimentInput(dataset_id=dataset_id, spec=_ridge())
        )
        result = run_model_experiment(
            RunModelExperimentInput(dataset_id=dataset_id, spec=_ridge())
        )
        manifest = load_manifest(result.model_id)
        assert manifest.dataset_hash == stored
        assert manifest.dataset_hash_version == 1
        folds = result.validation_report["folds"]
        assert folds and [f["node_hash"] for f in folds] == [
            f.node_hash for f in planned.folds
        ]
        assert _meta_path(dataset_id).read_bytes() == before

        # The node hashes do carry the dataset hash: the same panel under
        # its version-2 record gives different ones.
        _rewrite_meta(
            dataset_id,
            data_hash=canonical_frame_hash(reloaded),
            data_hash_version=2,
        )
        rerun = run_model_experiment(
            RunModelExperimentInput(dataset_id=dataset_id, spec=_ridge())
        )
        rerun_hashes = [f["node_hash"] for f in rerun.validation_report["folds"]]
        assert rerun_hashes != [f["node_hash"] for f in folds]
        assert json.dumps(rerun.oos_metrics, sort_keys=True) == json.dumps(
            result.oos_metrics, sort_keys=True
        )
