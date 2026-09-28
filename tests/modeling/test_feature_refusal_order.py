"""
A refusal that names missing features says the same thing in every process.

`compare_feature_sets` checked `set(left) | set(right)` one feature at a
time and stopped at the first missing one, so WHICH feature the refusal
named depended on string-hash order: the same call, run twice, could blame
a different feature. Every missing feature is now named, once, in the order
the caller gave them. One missing feature keeps the wording it always had.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pandas as pd
import pytest

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.agent import tools as modeling_tools
from standard_quant_tools.modeling.agent.feature_models import CompareFeatureSetsInput
from standard_quant_tools.modeling.agent.feature_tools import compare_feature_sets

_PROBE = textwrap.dedent("""
    import pandas as pd

    from standard_quant_tools.modeling.agent import tools as modeling_tools
    from standard_quant_tools.modeling.agent.feature_models import (
        CompareFeatureSetsInput,
    )
    from standard_quant_tools.modeling.agent.feature_tools import (
        compare_feature_sets,
    )

    panel = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=4),
            "entity": ["AAA"] * 4,
            "target": [0.1, -0.2, 0.3, 0.0],
            "ret_1": [0.01, 0.02, -0.01, 0.0],
        }
    )
    modeling_tools._load_dataset_panel = lambda dataset_id: (panel, {}, None)
    try:
        compare_feature_sets(
            CompareFeatureSetsInput(
                dataset_id="ds_probe",
                left=["missing_alpha", "ret_1"],
                right=["missing_beta", "missing_gamma"],
            )
        )
    except Exception as exc:
        print(str(exc))
    """)


def _panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=4),
            "entity": ["AAA"] * 4,
            "target": [0.1, -0.2, 0.3, 0.0],
            "ret_1": [0.01, 0.02, -0.01, 0.0],
            "ret_5": [0.03, -0.02, 0.01, 0.0],
        }
    )


def test_the_refusal_is_the_same_under_every_hash_seed():
    """Planted: under different PYTHONHASHSEED values the refusal named
    different features."""
    texts = set()
    for seed in ("0", "1", "2", "3", "4", "5"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        completed = subprocess.run(
            [sys.executable, "-c", _PROBE],
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
            check=True,
        )
        texts.add(completed.stdout.strip())
    assert len(texts) == 1, texts
    (text,) = texts
    assert "['missing_alpha', 'missing_beta', 'missing_gamma']" in text


def test_one_missing_feature_keeps_its_wording(monkeypatch):
    """Null: the single-feature message other tests and callers match on."""
    panel = _panel()
    monkeypatch.setattr(
        modeling_tools, "_load_dataset_panel", lambda dataset_id: (panel, {}, None)
    )
    with pytest.raises(ValidationError, match="has no feature 'ret_2'"):
        compare_feature_sets(
            CompareFeatureSetsInput(
                dataset_id="ds_probe", left=["ret_1", "ret_2"], right=["ret_5"]
            )
        )


def test_a_feature_named_on_both_sides_is_named_once(monkeypatch):
    panel = _panel()
    monkeypatch.setattr(
        modeling_tools, "_load_dataset_panel", lambda dataset_id: (panel, {}, None)
    )
    with pytest.raises(ValidationError) as excinfo:
        compare_feature_sets(
            CompareFeatureSetsInput(
                dataset_id="ds_probe",
                left=["zeta", "ret_1"],
                right=["zeta", "alpha"],
            )
        )
    assert "has no features ['zeta', 'alpha']" in str(excinfo.value)
