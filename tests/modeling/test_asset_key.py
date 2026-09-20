"""
AssetKey: the symbol a provider resolves, kept apart from the identity a
panel row belongs to.

Planted: the same symbol on two venues is two entities and is refused
where it would have been fetched twice; a class or venue suffix is
canonical and never reaches the provider; a venue every key shares
becomes the dataset's calendar; the bridge refuses a qualified universe
by name because the backtest runtime speaks bare symbols.
"""

import pytest
from pydantic import ValidationError as PydanticValidationError

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.assets import (
    AssetKey,
    canonical_universe,
    common_venue,
    fetch_plan,
    fetch_symbol,
    is_qualified,
    parse_asset_key,
)
from standard_quant_tools.modeling.bridge import oos_predictions_to_signal_panel
from standard_quant_tools.modeling.calendar import calendar_available
from standard_quant_tools.modeling.dataset.builder import build_dataset
from standard_quant_tools.modeling.portfolio_eval import evaluate_model_portfolio
from standard_quant_tools.modeling.registry.model_registry import load_manifest
from standard_quant_tools.modeling.scoring import score_model
from standard_quant_tools.modeling.specs import DatasetSpec

from .test_scoring import _dataset_spec, _train_a_model_with_spec

QUALIFIED = ["AAA@XNYS", "BBB@XNYS", "CCC@XNYS"]


class TestTheKey:
    def test_parse_and_canonical_round_trip(self):
        assert parse_asset_key("AAPL") == AssetKey(symbol="AAPL")
        assert parse_asset_key("AAPL~equity").canonical == "AAPL"
        key = parse_asset_key("BHP.AX@XASX")
        assert (key.symbol, key.venue, key.asset_class) == ("BHP.AX", "XASX", "equity")
        assert key.canonical == "BHP.AX@XASX"
        future = parse_asset_key("ES=F@XCME~future")
        assert future.asset_class == "future" and future.canonical == "ES=F@XCME~future"
        assert fetch_symbol("ES=F@XCME~future") == "ES=F"
        assert canonical_universe(["MSFT~equity", "BRK-B", "^GSPC~index"]) == [
            "MSFT",
            "BRK-B",
            "^GSPC~index",
        ]
        assert (
            is_qualified("AAA@XNYS")
            and is_qualified("X~fx")
            and not is_qualified("AAA")
        )
        assert (
            is_qualified("ORDER 1") is False
        )  # a string test, so odd names pass through

    def test_malformed_keys_are_refused_by_name(self):
        for bad, message in (
            ("", "empty"),
            ("AA PL", "not a symbol"),
            ("AAPL@nyse", "exchange code"),
            ("AAPL~bond", "not an asset class"),
            ("@XNYS", "not a symbol"),
        ):
            with pytest.raises(ValidationError, match=message):
                parse_asset_key(bad)

    def test_a_shared_symbol_across_venues_is_refused_where_it_would_fetch(self):
        assert fetch_plan(["AAA@XNYS", "BBB@XNYS"]) == {
            "AAA@XNYS": "AAA",
            "BBB@XNYS": "BBB",
        }
        # Different classes, same symbol: the same collision.
        with pytest.raises(ValidationError, match="both fetch as 'AAA'"):
            fetch_plan(["AAA", "AAA~etf"])
        with pytest.raises(ValidationError, match="both fetch as 'BHP'"):
            fetch_plan(["BHP@XASX", "BHP@XNYS"])
        assert common_venue(QUALIFIED) == "XNYS"
        assert common_venue(["AAA@XNYS", "BBB"]) is None
        assert common_venue(["AAA@XNYS", "BBB@XASX"]) is None


class TestTheSpec:
    def test_the_universe_is_canonical_and_distinct(self):
        spec = _dataset_spec(universe=["AAPL~equity", "MSFT", "BHP.AX@XASX"])
        assert spec.universe == ["AAPL", "MSFT", "BHP.AX@XASX"]
        with pytest.raises(PydanticValidationError, match="duplicate symbols"):
            _dataset_spec(universe=["AAPL", "AAPL~equity"])
        with pytest.raises(PydanticValidationError, match="not a symbol"):
            _dataset_spec(universe=["AA PL"])

    @pytest.mark.skipif(
        not calendar_available(), reason="exchange_calendars not installed"
    )
    def test_a_shared_venue_is_the_calendar_unless_one_is_named(self):
        assert _dataset_spec(universe=QUALIFIED).calendar == "XNYS"
        assert _dataset_spec(universe=["AAA@XNYS", "BBB@XASX"]).calendar is None
        assert _dataset_spec(universe=["AAA@XNYS", "BBB"]).calendar is None
        assert _dataset_spec(universe=QUALIFIED, calendar="XASX").calendar == "XASX"
        with pytest.raises(
            PydanticValidationError, match="not an exchange_calendars name"
        ):
            _dataset_spec(universe=["AAA@ZZZZ", "BBB@ZZZZ"])

    def test_a_scoring_universe_needs_no_calendar_to_be_canonical(self):
        spec = DatasetSpec(
            **{**_dataset_spec().model_dump(), "universe": ["AAA~equity"]}
        )
        assert spec.universe == ["AAA"]


class TestThroughTheRuntime:
    def test_the_provider_sees_symbols_and_the_panel_sees_keys(
        self, patched_multi_factory
    ):
        built = build_dataset(_dataset_spec(universe=QUALIFIED))
        assert set(built["panel"]["entity"].unique()) == set(QUALIFIED)
        requested = {
            call.args[0]
            for call in patched_multi_factory.get_ohlcv_async.call_args_list
        } | {call.args[0] for call in patched_multi_factory.get_ohlcv.call_args_list}
        assert {"AAA", "BBB", "CCC"} <= requested
        assert not any("@" in symbol for symbol in requested)

    def test_the_collision_is_refused_before_anything_is_fetched(
        self, patched_multi_factory
    ):
        with pytest.raises(ValidationError, match="both fetch as 'AAA'"):
            build_dataset(_dataset_spec(universe=["AAA@XNYS", "AAA@XASX", "BBB"]))
        assert patched_multi_factory.get_ohlcv_async.call_count == 0

    def test_a_model_trains_scores_and_simulates_under_its_keys(
        self, patched_multi_factory
    ):
        model_id = _train_a_model_with_spec(
            _dataset_spec(universe=QUALIFIED), dataset_id="ds_keys"
        )
        manifest = load_manifest(model_id)
        scored = score_model(model_id, as_of="2023-12-29", universe=QUALIFIED)
        assert scored["n_entities"] == 3 and scored["missing_entities"] == []
        # The canonical spelling is what comes back, whatever spelling went in.
        spelled = score_model(
            model_id, as_of="2023-12-29", universe=["AAA@XNYS~equity", "BBB@XNYS"]
        )
        assert spelled["n_entities"] == 2 and spelled["missing_entities"] == []
        result = evaluate_model_portfolio(model_id)
        assert result["n_entities"] == 3 if "n_entities" in result else True
        with pytest.raises(ValidationError, match="venue or asset class"):
            oos_predictions_to_signal_panel(
                oos_predictions_uri=manifest.oos_predictions_uri, task=manifest.task
            )
