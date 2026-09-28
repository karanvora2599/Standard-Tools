"""
Redaction reaches fields inside lists, scrubs whole tokens from an error
message, and says so when it is unsalted.

A configured path stopped at the first list it met, so `positions[].symbol`
and `nested.deep[].ssn` redacted nothing -- and a policy that covers nothing
looked exactly like one that matched nothing. The error-message scrub
replaced every SUBSTRING, so redacting a quantity of 5 rewrote
`123-45-6789` as `123-4<redacted:...>-6789`. The unsalted-placeholder notice
went to a logger nobody attaches. See the CHANGELOG entry of 2026-09-28.
"""

import warnings

import pytest

from standard_quant_tools import audit
from standard_quant_tools.audit import redaction
from standard_quant_tools.audit.redaction import UnsaltedRedactionWarning

INPUT = {
    "account_id": "ACC-1",
    "positions": [{"symbol": "AAPL", "qty": 5}, {"symbol": "MSFT", "qty": 7}],
    "nested": {"deep": [{"ssn": "123-45-6789"}, {"ssn": "987-65-4321"}]},
    "tags": ["alpha", "beta"],
}


def _is_placeholder(value) -> bool:
    return isinstance(value, str) and value.startswith("<redacted:")


class TestAPathReachesIntoLists:
    def test_an_explicit_list_segment_redacts_every_element(self):
        out = audit._redact(INPUT, ["positions[].symbol"])
        assert all(_is_placeholder(p["symbol"]) for p in out["positions"])
        assert [p["qty"] for p in out["positions"]] == [5, 7]

    def test_a_list_met_mid_path_is_walked_too(self):
        out = audit._redact(INPUT, ["positions.symbol"])
        assert all(_is_placeholder(p["symbol"]) for p in out["positions"])

    def test_a_nested_list_segment(self):
        out = audit._redact(INPUT, ["nested.deep[].ssn"])
        assert all(_is_placeholder(d["ssn"]) for d in out["nested"]["deep"])

    def test_a_path_ending_in_a_list_segment_redacts_each_element(self):
        out = audit._redact(INPUT, ["tags[]"])
        assert len(out["tags"]) == 2 and all(_is_placeholder(t) for t in out["tags"])
        assert out["tags"][0] != out["tags"][1]

    def test_a_whole_list_named_without_a_segment_is_one_placeholder(self):
        """Null case: naming a list itself redacts it as one value, as it
        always did."""
        out = audit._redact(INPUT, ["positions"])
        assert _is_placeholder(out["positions"])

    def test_a_path_matching_nothing_leaves_the_input_identical(self):
        assert audit._redact(INPUT, ["positions[].isin", "absent.field"]) == INPUT

    def test_the_original_is_not_mutated(self):
        before = repr(INPUT)
        audit._redact(INPUT, ["positions[].symbol", "nested.deep[].ssn"])
        assert repr(INPUT) == before


class TestTheErrorMessageIsScrubbedByToken:
    def test_a_redacted_number_leaves_other_numbers_alone(self):
        raw = {"quantity": 5, "ssn": "123-45-6789"}
        text = "quantity 5 rejected for 123-45-6789 at step 5."
        out = audit.redact_text(text, raw, ["quantity"])
        assert "123-45-6789" in out
        assert out.count("<redacted:") == 2
        assert out.endswith(".")

    def test_a_value_inside_a_decimal_is_not_a_token(self):
        out = audit.redact_text("price 1.5 and 5.0 and 5", {"q": 5}, ["q"])
        assert "1.5" in out and "5.0" in out and out.endswith(">")

    def test_values_reached_through_lists_are_scrubbed(self):
        text = "no data for AAPL or MSFT"
        out = audit.redact_text(text, INPUT, ["positions[].symbol"])
        assert "AAPL" not in out and "MSFT" not in out
        placeholders = audit._redact(INPUT, ["positions[].symbol"])["positions"]
        assert placeholders[0]["symbol"] in out and placeholders[1]["symbol"] in out

    def test_each_value_is_replaced_by_its_own_placeholder(self):
        """ "ACC" used to be replaced inside "ACC-1" first, leaving
        `<redacted:...>-1` where the longer value's placeholder belonged."""
        raw = {"a": "ACC", "b": "ACC-1"}
        out = audit.redact_text("account ACC-1 and ACC", raw, ["a", "b"])
        longer = audit._redact({"b": "ACC-1"}, ["b"])["b"]
        shorter = audit._redact({"a": "ACC"}, ["a"])["a"]
        assert out == f"account {longer} and {shorter}"

    def test_a_message_without_the_value_is_unchanged(self):
        """Null case."""
        assert audit.redact_text("unrelated error", INPUT, ["account_id"]) == (
            "unrelated error"
        )


class TestTheUnsaltedNoticeIsSeen:
    @pytest.fixture(autouse=True)
    def _fresh_process(self, monkeypatch):
        monkeypatch.setattr(redaction, "_warned_no_salt", False)

    def test_it_is_a_warning_once_per_process(self, monkeypatch):
        monkeypatch.delenv("SQT_AUDIT_REDACT_SALT", raising=False)
        with pytest.warns(UnsaltedRedactionWarning, match="SQT_AUDIT_REDACT_SALT"):
            audit._redact({"account_id": "1"}, ["account_id"])
        with warnings.catch_warnings():
            warnings.simplefilter("error", UnsaltedRedactionWarning)
            audit._redact({"account_id": "2"}, ["account_id"])

    def test_a_salted_redaction_says_nothing(self, monkeypatch):
        """Null case."""
        monkeypatch.setenv("SQT_AUDIT_REDACT_SALT", "a-stable-secret")
        with warnings.catch_warnings():
            warnings.simplefilter("error", UnsaltedRedactionWarning)
            audit._redact({"account_id": "1"}, ["account_id"])
