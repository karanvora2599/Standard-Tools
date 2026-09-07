"""
The helper that was written fourteen times, and the one that only looks
like a fifteenth.

`_finite_or_none` existed in fifteen modules. Every copy was live and used
exactly once, so nothing here is about dead code -- it is about what
fourteen copies of a correct three-line function cost. They had drifted into
four spellings, and the docstring explaining why the function exists at all
survived in two of them.

The test that matters is the last one. `modeling.agent.feature_models` keeps
its own version on purpose, because it is a DIFFERENT function wearing the
same name: it coerces with `float()` and swallows the conversion error, so a
string goes in and `None` comes out. Anyone finishing the consolidation by
deleting that copy too would change what the feature tools return, silently,
for every non-numeric value they touch.
"""

from __future__ import annotations

import pytest

from standard_quant_tools.agent.runtimes._json_safe import finite_or_none
from standard_quant_tools.modeling.agent.feature_models import (
    _finite_or_none as coercing_variant,
)


class TestNonFiniteBecomesNull:
    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_a_non_finite_float_becomes_none(self, value):
        """`json.dumps(float("nan"))` emits a bare NaN token, which is not
        valid JSON. Several MCP clients reject that at the transport layer,
        failing the whole response rather than the one field that could not
        be computed."""
        assert finite_or_none(value) is None

    @pytest.mark.parametrize("value", [0.0, -1.5, 1e300, 3, -7, True, False])
    def test_anything_finite_passes_through_unchanged(self, value):
        result = finite_or_none(value)
        assert result == value
        assert type(result) is type(value)

    @pytest.mark.parametrize("value", ["nan", "abc", None, [float("nan")], {}])
    def test_a_non_number_is_not_this_function_s_to_reinterpret(self, value):
        assert finite_or_none(value) == value


class TestTheFifteenthCopyIsNotACopy:
    """
    The reason `modeling.agent.feature_models` keeps its own.

    Its contract is "a statistic, or None when it could not be computed",
    which is not the same as "a non-finite float becomes null". Merging the
    two would be a silent behaviour change in whichever direction it went.
    """

    def test_the_coercing_variant_nulls_a_string_where_the_shared_one_passes_it(
        self,
    ):
        assert coercing_variant("abc") is None
        assert finite_or_none("abc") == "abc"

    def test_the_coercing_variant_returns_a_float_where_the_shared_one_keeps_the_type(
        self,
    ):
        assert coercing_variant(3) == 3.0
        assert isinstance(coercing_variant(3), float)
        assert type(finite_or_none(3)) is int

    def test_they_agree_on_the_case_they_both_exist_for(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            assert coercing_variant(value) is None
            assert finite_or_none(value) is None
        assert coercing_variant(2.5) == finite_or_none(2.5) == 2.5


class TestEveryRuntimeUsesTheSharedOne:
    def test_no_module_redefines_it_except_the_documented_exception(self):
        """
        Fourteen private definitions became one import. This fails if a
        fifteenth is added rather than imported -- which is how there came
        to be four spellings of it in the first place.
        """
        import ast
        import pathlib

        src = pathlib.Path(__file__).resolve().parents[2] / "src"
        allowed = {"standard_quant_tools/modeling/agent/feature_models.py"}

        redefiners = []
        for path in (src / "standard_quant_tools").rglob("*.py"):
            rel = path.relative_to(src).as_posix()
            if rel in allowed:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name in (
                    "_finite_or_none",
                    "finite_or_none",
                ):
                    if not rel.endswith("_json_safe.py"):
                        redefiners.append(rel)

        assert redefiners == [], (
            "these modules define their own instead of importing the shared "
            f"helper: {redefiners}"
        )
