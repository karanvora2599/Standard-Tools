"""
Redaction reaches the keys of a mapping.

`SQT_AUDIT_REDACT_FIELDS` could hide values, inside lists too, but never a
mapping's keys -- so an input keyed by the sensitive value itself
(positions keyed by account, weights keyed by symbol) wrote it to the log
whatever the policy said. A segment ending in `{}` now names a mapping whose
keys are redacted: each key becomes the placeholder that text gets as a
value, salted the same way, and the values and the mapping's size are kept.
"""

import json

import pytest
from pydantic import BaseModel

from standard_quant_tools import audit
from standard_quant_tools.audit.dispatch import _run_and_record

INPUT = {
    "positions": {"ACC-1": {"ssn": "123-45-6789", "qty": 5}, "ACC-2": {"qty": 7}},
    "weights": {"AAPL": 0.6, "MSFT": 0.4},
    "book": {"accounts": {"ACC-9": 1.0}},
    "legs": [{"AAPL": 1}, {"MSFT": 2}],
}


def _is_placeholder(value) -> bool:
    return isinstance(value, str) and value.startswith("<redacted:")


def _value_placeholder(text):
    """What `text` becomes when it is redacted as a VALUE."""
    return audit._redact({"v": text}, ["v"])["v"]


@pytest.fixture(autouse=True)
def _salted(monkeypatch):
    monkeypatch.setenv("SQT_AUDIT_REDACT_SALT", "a-stable-secret")


class TestAKeysSegmentRedactsTheKeys:
    def test_every_key_is_a_placeholder_and_every_value_is_kept(self):
        out = audit._redact(INPUT, ["weights{}"])
        assert all(_is_placeholder(k) for k in out["weights"])
        assert sorted(out["weights"].values()) == [0.4, 0.6]
        written = json.dumps(out["weights"])
        assert "AAPL" not in written and "MSFT" not in written

    def test_the_order_and_size_of_the_mapping_are_kept(self):
        out = audit._redact(INPUT, ["positions{}"])
        assert list(out["positions"]) == [
            _value_placeholder("ACC-1"),
            _value_placeholder("ACC-2"),
        ]
        assert list(out["positions"].values()) == list(INPUT["positions"].values())

    def test_a_key_is_salted_exactly_like_the_same_text_as_a_value(self, monkeypatch):
        """Two records that hid one account compare equal on it, whether it
        was a key in one and a value in the other."""
        key = next(iter(audit._redact(INPUT, ["weights{}"])["weights"]))
        assert key == _value_placeholder("AAPL")

        monkeypatch.setenv("SQT_AUDIT_REDACT_SALT", "another-secret")
        rekeyed = next(iter(audit._redact(INPUT, ["weights{}"])["weights"]))
        assert rekeyed != key and rekeyed == _value_placeholder("AAPL")

    def test_a_nested_mapping(self):
        out = audit._redact(INPUT, ["book.accounts{}"])
        assert list(out["book"]["accounts"]) == [_value_placeholder("ACC-9")]
        assert list(out["book"]["accounts"].values()) == [1.0]

    def test_a_list_of_mappings_has_each_mappings_keys_redacted(self):
        out = audit._redact(INPUT, ["legs{}"])
        assert [list(leg) for leg in out["legs"]] == [
            [_value_placeholder("AAPL")],
            [_value_placeholder("MSFT")],
        ]
        assert [list(leg.values()) for leg in out["legs"]] == [[1], [2]]

    def test_a_keys_segment_mid_path_goes_on_into_every_value(self):
        out = audit._redact(INPUT, ["positions{}.ssn"])
        first, second = out["positions"].values()
        assert all(_is_placeholder(k) for k in out["positions"])
        assert _is_placeholder(first["ssn"]) and first["qty"] == 5
        assert second == {"qty": 7}

    def test_the_order_paths_are_listed_in_does_not_matter(self):
        """Keys are renamed after every path has been walked, so a value
        path through a raw key still finds it."""
        forward = audit._redact(INPUT, ["positions{}", "positions.ACC-1.ssn"])
        backward = audit._redact(INPUT, ["positions.ACC-1.ssn", "positions{}"])
        assert forward == backward
        assert _is_placeholder(next(iter(forward["positions"].values()))["ssn"])

    def test_a_mapping_named_twice_is_hashed_once(self):
        twice = audit._redact(INPUT, ["weights{}", "weights{}"])
        assert twice == audit._redact(INPUT, ["weights{}"])

    def test_colliding_placeholders_keep_both_entries(self, monkeypatch):
        from standard_quant_tools.audit import redaction

        monkeypatch.setattr(redaction, "_placeholder_for", lambda v: "<redacted:0>")
        out = audit._redact(INPUT, ["weights{}"])
        assert out["weights"] == {"<redacted:0>": 0.6, "<redacted:0>~2": 0.4}

    def test_the_original_is_not_mutated(self):
        before = repr(INPUT)
        audit._redact(INPUT, ["positions{}", "weights{}", "legs{}"])
        assert repr(INPUT) == before


class TestOnlyANamedMappingIsTouched:
    def test_an_unlisted_mapping_keeps_its_keys(self):
        """Null case."""
        out = audit._redact(INPUT, ["positions{}"])
        assert out["weights"] == INPUT["weights"]
        assert out["book"] == INPUT["book"]

    def test_a_mapping_named_without_braces_is_still_one_value(self):
        """Null case: the old meaning of naming a mapping is unchanged."""
        out = audit._redact(INPUT, ["weights"])
        assert _is_placeholder(out["weights"])

    def test_braces_on_something_that_is_not_a_mapping_match_nothing(self):
        assert audit._redact({"account": "ACC-1"}, ["account{}"]) == {
            "account": "ACC-1"
        }


class TestTheErrorMessageLosesTheKeysToo:
    def test_a_redacted_key_is_scrubbed_with_its_placeholder(self):
        text = "no price for AAPL in weights; MSFT ok"
        out = audit.redact_text(text, INPUT, ["weights{}"])
        assert "AAPL" not in out and "MSFT" not in out
        keys = list(audit._redact(INPUT, ["weights{}"])["weights"])
        assert out == f"no price for {keys[0]} in weights; {keys[1]} ok"

    def test_an_unlisted_mappings_keys_stay_in_the_message(self):
        """Null case."""
        text = "no price for AAPL"
        assert audit.redact_text(text, INPUT, ["positions{}"]) == text


class _KeyedInput(BaseModel):
    positions: dict
    note: str = "x"


class TestTheWrittenRecordHidesTheKeys:
    def test_through_the_audited_path(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SQT_AUDIT_DIR", str(tmp_path))
        monkeypatch.setenv("SQT_AUDIT_REDACT_FIELDS", "positions{}")

        def _tool(model):
            raise KeyError(f"unknown account ACC-1 (of {len(model.positions)})")

        with pytest.raises(KeyError):
            _run_and_record(
                "keyed_tool",
                _tool,
                _KeyedInput(positions={"ACC-1": 100.0, "ACC-2": 50.0}),
            )
        day = audit._iter_day_files(tmp_path)[0]
        text = day.read_text(encoding="utf-8")
        assert "ACC-1" not in text and "ACC-2" not in text
        record = json.loads(text.splitlines()[0])
        assert sorted(record["input"]["positions"].values()) == [50.0, 100.0]
        assert _value_placeholder("ACC-1") in record["input"]["positions"]
        assert _value_placeholder("ACC-1") in record["error_message"]
        assert audit.verify_audit_trail_integrity(tmp_path) == []

    def test_without_the_policy_the_keys_are_written(self, tmp_path, monkeypatch):
        """Null case."""
        monkeypatch.setenv("SQT_AUDIT_DIR", str(tmp_path))
        monkeypatch.delenv("SQT_AUDIT_REDACT_FIELDS", raising=False)
        _run_and_record(
            "keyed_tool",
            lambda model: {"ok": True},
            _KeyedInput(positions={"ACC-1": 100.0}),
        )
        day = audit._iter_day_files(tmp_path)[0]
        record = json.loads(day.read_text(encoding="utf-8").splitlines()[0])
        assert record["input"]["positions"] == {"ACC-1": 100.0}
