"""
A portfolio evaluation's weights and equity-curve hashes do not depend on
the pandas version.

`evaluate_model_portfolio` and `evaluate_predictions_portfolio` named the
target-weights and equity-curve artifacts after `audit.hash_dataframe` of
each frame and returned the same values as `target_weights_hash` and
`equity_curve_hash`. That hash covers the resolution a frame's dates are
stored at, and both frames are indexed by dates read from the predictions
and the prices, which pandas 3 stores at `[us]` where pandas 2 stores
`[ns]`: identical weights were hashed and named differently under the two
(the CHANGELOG entry of 2026-10-04).

They are now `audit.canonical_frame_hash`, recorded with
`frame_hash_version` 2 in the result's provenance. The literals below were
computed under pandas 2.3.3 and pandas 3.0.5 and are asserted under
whichever one runs the suite: CI runs Python 3.10, which resolves pandas 2,
and Python 3.11 and 3.12, which resolve pandas 3.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.audit.hashing import canonical_frame_hash, hash_dataframe
from standard_quant_tools.data.factory import DataFactory
from standard_quant_tools.modeling import artifacts as _artifacts
from standard_quant_tools.modeling.portfolio_eval import (
    FRAME_HASH_VERSION,
    evaluate_predictions_portfolio,
)
from standard_quant_tools.modeling.specs import PredictionTransformSpec

from .conftest import make_ohlcv

ENTITIES = ["AAA", "BBB", "CCC", "DDD", "EEE"]

#: `canonical_frame_hash` of `_weights(unit)` at either resolution.
WEIGHTS_CANONICAL = "caf7ea90fe8178f7"
#: `hash_dataframe` of `_weights(unit)`: the index's stored integers differ.
WEIGHTS_LEGACY_NS = "7bf320a2ac765d17"
WEIGHTS_LEGACY_US = "05ed19d82b9c8b40"

#: `target_weights_hash` of `_evaluate(unit)`, whichever unit the dates
#: arrive at and whichever pandas runs it.
EVALUATED_WEIGHTS = "660227a02d7b3ab8"


def _weights(unit: str) -> pd.DataFrame:
    """A target-weights frame as each pandas builds it: the same dates,
    stored at `[ns]` under pandas 2 and `[us]` under pandas 3."""
    dates = pd.DatetimeIndex(
        ["2024-01-05", "2024-01-12", "2024-01-19"], name="date"
    ).as_unit(unit)
    return pd.DataFrame(
        {
            "AAA": [0.25, -0.125, 0.0],
            "BBB": [-0.25, 0.125, 0.5],
            "CCC": [0.0, 0.0, -0.5],
        },
        index=dates,
    )


def _predictions(unit: str) -> pd.DataFrame:
    """Five names a day over 120 business days, ranked by arithmetic alone
    so each date's ordering is exact on every platform."""
    dates = make_ohlcv("AAA").index[300:420].as_unit(unit)
    rows = [
        (date, entity, float((7 * day + 3 * position) % 11) + 0.01 * position)
        for day, date in enumerate(dates)
        for position, entity in enumerate(ENTITIES)
    ]
    frame = pd.DataFrame(rows, columns=["date", "entity", "prediction"])
    frame["date"] = frame["date"].astype(f"datetime64[{unit}]")
    return frame


@pytest.fixture
def prices_at(monkeypatch) -> Dict[str, str]:
    """Serves `conftest.make_ohlcv` bars with the index at the unit the
    returned dict names, so one process can stand in for either pandas."""
    state = {"unit": "ns"}

    class _Provider:
        def get_ohlcv(self, symbol, start, end, interval="1d"):
            frame = make_ohlcv(symbol)
            frame.index = frame.index.as_unit(state["unit"])
            return frame

        async def get_ohlcv_async(self, symbol, start, end, interval="1d"):
            return self.get_ohlcv(symbol, start, end, interval)

    monkeypatch.setattr(DataFactory, "get_provider", lambda *a, **k: _Provider())
    return state


def _evaluate(unit: str, prices_at: Dict[str, str]) -> dict:
    prices_at["unit"] = unit
    return evaluate_predictions_portfolio(
        _predictions(unit),
        "regression",
        interval="1d",
        provider_name="mock",
        calendar=None,
        start="2022-01-01",
        end="2023-12-31",
        transform=PredictionTransformSpec(
            method="cross_sectional_rank", rebalance_frequency="weekly"
        ),
        run_id="portfolio_hash",
    )


class TestTheHashDoesNotDependOnPandas:
    def test_the_pinned_values_hold_under_this_pandas(self):
        """The earlier hash of the same weights differs with the index's
        resolution; the one recorded now does not."""
        assert canonical_frame_hash(_weights("ns")) == WEIGHTS_CANONICAL
        assert canonical_frame_hash(_weights("us")) == WEIGHTS_CANONICAL
        assert hash_dataframe(_weights("ns")) == WEIGHTS_LEGACY_NS
        assert hash_dataframe(_weights("us")) == WEIGHTS_LEGACY_US

    def test_a_changed_weight_changes_it(self):
        """Null case: the hash still identifies the weights."""
        moved = _weights("ns")
        moved.iloc[0, 0] = np.nextafter(0.25, 1.0)
        assert canonical_frame_hash(moved) != WEIGHTS_CANONICAL


class TestAnEvaluationRecordsTheCurrentForm:
    def test_dates_at_either_resolution_give_one_hash_and_one_name(self, prices_at):
        """The predictions' and the prices' dates at `[ns]` (pandas 2) and
        at `[us]` (pandas 3): the same weights and the same equity curve,
        so the same hashes and the same artifact names."""
        at_ns = _evaluate("ns", prices_at)
        at_us = _evaluate("us", prices_at)
        for key in ("target_weights_hash", "equity_curve_hash"):
            assert at_ns["provenance"][key] == at_us["provenance"][key]
        assert at_ns["target_weights_uri"] == at_us["target_weights_uri"]
        assert at_ns["equity_curve_uri"] == at_us["equity_curve_uri"]
        assert at_ns["provenance"]["target_weights_hash"] == EVALUATED_WEIGHTS

    def test_the_hashes_are_of_the_frames_written(self, prices_at):
        result = _evaluate("us", prices_at)
        provenance = result["provenance"]
        assert provenance["frame_hash_version"] == FRAME_HASH_VERSION == 2

        weights = _artifacts.load_artifact(result["target_weights_uri"])
        equity = _artifacts.load_artifact(result["equity_curve_uri"])
        if isinstance(equity, pd.DataFrame):
            equity = equity.iloc[:, 0]
        assert canonical_frame_hash(weights) == provenance["target_weights_hash"]
        assert (
            canonical_frame_hash(equity.rename("equity").to_frame())
            == provenance["equity_curve_hash"]
        )
        assert Path(result["target_weights_uri"]).stem == (
            f"target_weights_{provenance['target_weights_hash']}"
        )
        assert Path(result["equity_curve_uri"]).stem == (
            f"portfolio_equity_{provenance['equity_curve_hash']}"
        )
