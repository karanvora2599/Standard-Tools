"""
The native module's docstrings read as text in `help()`.

Three of them were written with a doubled backslash, so `help()` printed a
literal backslash-n where each line break should have been (43 of them), and
stochastic_oscillator wrote %K as `%%K`, the printf escape, in a string no
printf ever sees.
"""

import pytest

_cpp = pytest.importorskip(
    "standard_quant_tools._sqt_core",
    reason="native extension not built",
)


def _docstrings():
    for name in sorted(dir(_cpp)):
        doc = getattr(getattr(_cpp, name), "__doc__", None)
        if name.startswith("_") or not doc:
            continue
        yield name, doc


@pytest.mark.parametrize("name,doc", list(_docstrings()))
def test_no_escape_sequence_survives_into_the_text(name, doc):
    assert "\\n" not in doc, name
    assert "%%" not in doc, name


def test_the_three_that_had_them_now_break_lines():
    for name in (
        "run_portfolio_simulation",
        "technical_indicators_panel",
        "batch_engle_granger",
    ):
        assert getattr(_cpp, name).__doc__.count("\n") > 5, name
    assert "%K" in _cpp.stochastic_oscillator.__doc__
