"""
A stale extension is indistinguishable from an absent one, and costs more.

Every kernel is guarded by `hasattr(_cpp_core, "<name>")` at import. That is
the right design -- a missing symbol falls back to Python silently rather
than raising, so an old build still runs correctly -- but it means an
extension compiled before the newest kernels were added loads cleanly,
reports itself available, and quietly runs the slow path for whatever it
does not carry.

Observed while surveying this repo: a `.pyd` built for 3.11 exported 40 of
the 42 symbols, was missing `rank_by_date` and `permutation_null_ic`, and
cost about 18x on the operations that use them. Nothing said so. The only
symptom was that a run was slower than the numbers in the docs.

So `list_modeling_capabilities` reports the count alongside the boolean, and
these tests keep the expected number honest -- a kernel added without
updating it fails here rather than making every future report say "stale".
"""

from __future__ import annotations

import pytest

from standard_quant_tools.modeling.capabilities import (
    _EXPECTED_NATIVE_EXPORTS,
    modeling_capabilities,
)

_cpp_core = pytest.importorskip(
    "standard_quant_tools._sqt_core", reason="needs the compiled extension"
)


def _exports():
    return [name for name in dir(_cpp_core) if not name.startswith("_")]


class TestTheExpectedCountIsTheRealOne:
    def test_it_matches_what_the_extension_exports(self):
        """The drift guard. Adding a kernel without updating the constant
        would make every capability report claim a stale build."""
        assert len(_exports()) == _EXPECTED_NATIVE_EXPORTS, (
            f"the extension exports {len(_exports())} symbols and "
            f"_EXPECTED_NATIVE_EXPORTS says {_EXPECTED_NATIVE_EXPORTS}. If a "
            "kernel was added, update the constant."
        )

    def test_a_current_build_is_not_reported_stale(self):
        detail = modeling_capabilities()["native_extension_detail"]
        assert detail["available"] is True
        assert detail["stale"] is False
        assert detail["exports"] == _EXPECTED_NATIVE_EXPORTS

    def test_the_two_kernels_a_stale_build_was_missing_are_present(self):
        """Named rather than counted: these are the two whose absence was
        measured at ~18x, and a count alone would not say which."""
        assert hasattr(_cpp_core, "rank_by_date")
        assert hasattr(_cpp_core, "permutation_null_ic")

    def test_the_report_names_the_file_it_loaded(self):
        """Two interpreters on one machine can load different builds. The
        path is how a reader tells which one answered."""
        detail = modeling_capabilities()["native_extension_detail"]
        assert detail["path"] and "_sqt_core" in detail["path"]


class TestAStaleBuildWouldBeCaught:
    def test_a_short_export_list_reads_as_stale(self, monkeypatch):
        """Simulated by raising the expectation, which is the same
        comparison from the other side."""
        import standard_quant_tools.modeling.capabilities as caps

        monkeypatch.setattr(
            caps, "_EXPECTED_NATIVE_EXPORTS", _EXPECTED_NATIVE_EXPORTS + 5
        )
        detail = caps.modeling_capabilities()["native_extension_detail"]
        assert detail["stale"] is True
        assert "Rebuild it" in detail["note"]

    def test_the_note_says_what_the_symptom_would_be(self, monkeypatch):
        import standard_quant_tools.modeling.capabilities as caps

        monkeypatch.setattr(
            caps, "_EXPECTED_NATIVE_EXPORTS", _EXPECTED_NATIVE_EXPORTS + 1
        )
        detail = caps.modeling_capabilities()["native_extension_detail"]
        assert "falling back to Python silently" in detail["note"]
