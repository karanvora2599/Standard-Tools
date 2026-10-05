"""
The promotion log says an edit cannot hide, and now something would show it.

`lifecycle.py` claimed it outright: the history is "append-only ... and
nothing about its promotion history can be edited without the edit
showing". Nothing would have. There was no digest and no chain in the
module; the machinery that exists -- `torn_fragments`, `_TORN_PREFIX` --
detects a TRUNCATED WRITE, which is an interrupted append, not an edit.
Append-only was a convention of the writer, not a property of the file.

The line someone would want gone is the one `promote_model` writes on
purpose: `package_check_waived`, recorded so a promotion made over a
failing package stays legible. That evidence and the file protecting it
were the same unprotected file, which is what the waiver test plants.

WHAT IS AND IS NOT CLAIMED. Each record carries the digest of the previous
line's bytes, rooted in the manifest digest. An edited, removed or
reordered line is detected. A log rewritten end to end, every link
recomputed, is NOT -- the manifest digest is readable, so the root can be
reproduced -- and the last test pins that limit so no later reader mistakes
this for more than it is.
"""

import hashlib
import json

import pytest

from standard_quant_tools.modeling import artifacts as _artifacts
from standard_quant_tools.modeling.agent.models import PromoteModelInput
from standard_quant_tools.modeling.agent.tools import promote_model
from standard_quant_tools.modeling.registry.lifecycle import (
    PROMOTIONS_FILE,
    current_stage,
    promotions,
    verify_promotion_chain,
)

from .test_scoring import _dataset_spec, _train_a_model_with_spec

NO_PREV = "carries no `prev`"
TAMPERED = "edited, removed or reordered"


def _log(model_id: str):
    return _artifacts.run_dir(model_id) / PROMOTIONS_FILE


def _lines(model_id: str):
    return [line for line in _log(model_id).read_bytes().split(b"\n") if line.strip()]


def _rewrite(model_id: str, lines) -> None:
    _log(model_id).write_bytes(b"".join(line + b"\n" for line in lines))


def _promote(model_id: str, to_stage: str, reason: str = "the folds agreed"):
    return promote_model(
        PromoteModelInput(model_id=model_id, to_stage=to_stage, reason=reason)
    )


def _manifest_root(model_id: str) -> str:
    manifest = _artifacts.run_dir(model_id) / "manifest.json"
    return hashlib.sha256(manifest.read_bytes()).hexdigest()


@pytest.fixture
def promoted(patched_multi_factory, request):
    """A model with a three-record history, which is the shortest log in
    which a line can be edited in the MIDDLE."""
    model_id = _train_a_model_with_spec(
        _dataset_spec(), dataset_id=f"ds_chain_{request.node.name[:24]}"
    )
    _promote(model_id, "validated")
    _promote(model_id, "staging")
    _promote(model_id, "production")
    return model_id


class TestACleanLogVerifies:
    def test_three_promotions_chain(self, promoted):
        assert verify_promotion_chain(promoted) == []
        assert current_stage(promoted) == "production"

    def test_the_first_record_is_rooted_in_the_manifest(self, promoted):
        """Not in a constant: the manifest is the package's commit point
        and the one file a signature covers."""
        first = json.loads(_lines(promoted)[0])
        assert first["prev"] == _manifest_root(promoted)

    def test_an_empty_log_has_no_findings(self, patched_multi_factory):
        model_id = _train_a_model_with_spec(
            _dataset_spec(), dataset_id="ds_chain_never_promoted"
        )
        assert verify_promotion_chain(model_id) == []
        assert promotions(model_id) == []


class TestAnEditIsDetected:
    def test_a_changed_reason_breaks_the_link(self, promoted):
        lines = _lines(promoted)
        record = json.loads(lines[1])
        record["reason"] = "something nobody said"
        lines[1] = json.dumps(record, sort_keys=True).encode("utf-8")
        _rewrite(promoted, lines)
        findings = verify_promotion_chain(promoted)
        assert findings, "an edited record left the chain intact"
        # The edit is at record 2, so record 3's link is the one that fails.
        assert "record 3 of 3" in findings[0], findings

    def test_a_removed_record_breaks_the_link(self, promoted):
        lines = _lines(promoted)
        del lines[1]
        _rewrite(promoted, lines)
        findings = verify_promotion_chain(promoted)
        assert findings
        assert TAMPERED in findings[0], findings

    def test_a_reordered_pair_breaks_the_link(self, promoted):
        lines = _lines(promoted)
        lines[0], lines[1] = lines[1], lines[0]
        _rewrite(promoted, lines)
        assert verify_promotion_chain(promoted)

    def test_a_legacy_record_is_unchained_not_broken(self, promoted):
        """A log from before the chain is not a damaged log. Reported so
        nobody reads silence as a guarantee, but distinguished from an
        edit -- the same distinction the content hashes draw between a
        pre-hashing package and a removed key."""
        stripped = []
        for line in _lines(promoted):
            record = json.loads(line)
            record.pop("prev", None)
            stripped.append(json.dumps(record, sort_keys=True).encode("utf-8"))
        _rewrite(promoted, stripped)
        findings = verify_promotion_chain(promoted)
        assert len(findings) == 3
        assert all(NO_PREV in f for f in findings)
        assert not any(TAMPERED in f for f in findings)


class TestTheDecisionRecordsTheDamage:
    def test_a_promotion_onto_a_broken_history_says_so_in_its_evidence(
        self, patched_multi_factory
    ):
        """The reader months later is reading the LOG, not the return value
        of the call that wrote it."""
        model_id = _train_a_model_with_spec(
            _dataset_spec(), dataset_id="ds_chain_evidence"
        )
        _promote(model_id, "validated")
        _promote(model_id, "staging")
        # Record 0, not the tip: nothing in the file commits to the last
        # record's bytes, which is a documented limit and is pinned below.
        lines = _lines(model_id)
        record = json.loads(lines[0])
        record["reason"] = "tampered"
        lines[0] = json.dumps(record, sort_keys=True).encode("utf-8")
        _rewrite(model_id, lines)

        result = _promote(model_id, "production")
        assert result.promotion_chain_findings
        recorded = [
            e
            for e in promotions(model_id)[-1].evidence
            if e.startswith("promotion_log_chain_broken")
        ]
        assert recorded, promotions(model_id)[-1].evidence

    def test_a_clean_promotion_records_no_damage(self, promoted):
        result = _promote(promoted, "archived")
        assert result.promotion_chain_findings == []
        assert not any(
            e.startswith("promotion_log_chain_broken")
            for e in promotions(promoted)[-1].evidence
        )

    def test_the_waiver_line_can_no_longer_be_deleted_quietly(
        self, patched_multi_factory
    ):
        """The line this protects. `promote_model` writes
        `package_check_waived` so a promotion made over a failing package
        stays legible; removing that record is the edit the log existed to
        make visible and could not."""
        model_id = _train_a_model_with_spec(
            _dataset_spec(), dataset_id="ds_chain_waiver"
        )
        with open(_artifacts.run_dir(model_id) / "model.joblib", "ab") as handle:
            handle.write(b"\x00")
        promote_model(
            PromoteModelInput(
                model_id=model_id,
                to_stage="validated",
                reason="shipping anyway",
                require_verified_package=False,
            )
        )
        promote_model(
            PromoteModelInput(
                model_id=model_id,
                to_stage="staging",
                reason="and onwards",
                require_verified_package=False,
            )
        )
        assert any(
            e.startswith("package_check_waived")
            for e in promotions(model_id)[0].evidence
        )
        # Delete the waived decision and keep the stage that followed it.
        _rewrite(model_id, _lines(model_id)[1:])
        assert verify_promotion_chain(model_id)


class TestTheLimitIsStated:
    def test_a_log_rewritten_end_to_end_is_not_detected(self, promoted):
        """Pinned deliberately. The root is the manifest digest, which is
        readable, so every link can be recomputed. Detecting this needs a
        witness kept outside the package, which this module does not have
        and its docstring does not claim.

        If this test ever fails, the module gained a property worth
        documenting rather than a bug worth fixing.
        """
        prev = _manifest_root(promoted)
        rebuilt = []
        for line in _lines(promoted):
            record = json.loads(line)
            record["reason"] = "rewritten wholesale"
            record["prev"] = prev
            raw = json.dumps(record, sort_keys=True).encode("utf-8") + b"\n"
            rebuilt.append(raw[:-1])
            prev = hashlib.sha256(raw).hexdigest()
        _rewrite(promoted, rebuilt)
        assert verify_promotion_chain(promoted) == []
        assert promotions(promoted)[0].reason == "rewritten wholesale"

    def test_an_edit_to_the_LAST_record_is_not_detected(self, promoted):
        """The other stated limit. Nothing in the file hashes the tip, so
        the newest decision is unprotected until another lands on top of
        it -- the same shape as the audit chain's own unchecked tail.

        If this test ever fails, the module gained a property worth
        documenting rather than a bug worth fixing.
        """
        lines = _lines(promoted)
        record = json.loads(lines[-1])
        record["reason"] = "the tip is not covered"
        lines[-1] = json.dumps(record, sort_keys=True).encode("utf-8")
        _rewrite(promoted, lines)
        assert verify_promotion_chain(promoted) == []
        # And appending one more decision closes it retroactively.
        _promote(promoted, "archived")
        assert verify_promotion_chain(promoted) == []
        lines = _lines(promoted)
        record = json.loads(lines[-2])
        record["reason"] = "now it has a successor"
        lines[-2] = json.dumps(record, sort_keys=True).encode("utf-8")
        _rewrite(promoted, lines)
        assert verify_promotion_chain(promoted)
