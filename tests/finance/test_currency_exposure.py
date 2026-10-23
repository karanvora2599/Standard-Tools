"""
The currency exposure of a set of positions after its FX hedges, and the
local / FX / cross split of each position's return, against answers worked
by hand and identities that hold for any input.

THE PLANTED BOOK, base USD, at rates exact in binary:

    SAP   +1,000 x 120     EUR x 1.125      = +135,000.00
    7203  -2,000 x 2,900   JPY x 0.0078125  =  -45,312.50
    VOD   +4,000 x 0.75    GBP x 1.25       =   +3,750.00
    AAPL  +100   x 190     USD              =  +19,000.00

hedged with {'EURUSD': -60,000} (sell 60,000 EUR, 67,500 USD) and
{'EURGBP': -20,000} (sell 20,000 EUR, buy 18,000 GBP = 22,500 USD). EUR's
net is 135,000 - 67,500 - 22,500 = 45,000 and its hedge ratio 90,000 /
135,000 = 2/3; GBP's net is 3,750 + 22,500 = 26,250 and its ratio -6, a
hedge that adds to the exposure; JPY is unhedged; USD gains the 67,500.

The identities: each currency's exposure is the sum of its lines' local
notionals times the rate; the three return terms add up to the base return
exactly and to the return worked from prices to 1e-14; relabelling the
currencies -- the same permutation of codes and rates everywhere --
changes no number; a fully hedged currency has a ratio of 1 and no net
exposure; a position in the base currency has no FX term.
"""

from __future__ import annotations

import copy
import itertools
import math
import random
from fractions import Fraction
from typing import Any, Dict, List

import pydantic
import pytest

from standard_quant_tools.agent.runtimes import resolve
from standard_quant_tools.agent.runtimes.finance.currency_models import (
    CurrencyExposureInput,
)
from standard_quant_tools.error import ValidationError
from standard_quant_tools.finance import currency_exposure, exposure_summary
from standard_quant_tools.mcp.catalog import _reads_market_data
from standard_quant_tools.mcp.schemas import dereference

from ..surface import synth


def _line(instrument: str, quantity: float, price: float, currency: str, **more):
    return {
        "instrument": instrument,
        "quantity": quantity,
        "price": price,
        "currency": currency,
        **more,
    }


PLANTED: List[Dict[str, Any]] = [
    _line("SAP", 1000, 120.0, "EUR"),
    _line("7203", -2000, 2900.0, "JPY"),
    _line("VOD", 4000, 0.75, "GBP"),
    _line("AAPL", 100, 190.0, "USD"),
]
RATES = {"EUR": 1.125, "JPY": 0.0078125, "GBP": 1.25}
HEDGES = {"EURUSD": -60_000.0, "EURGBP": -20_000.0}
FX_RETURNS = {"EUR": 0.0125, "JPY": -0.03, "GBP": 0.005}
LOCAL = {"SAP": 0.04, "7203": -0.015, "VOD": 0.02, "AAPL": 0.01}


def _with_returns(lines: List[Dict[str, Any]], local=LOCAL) -> List[Dict[str, Any]]:
    return [{**line, "local_return": local[line["instrument"]]} for line in lines]


def _rows(result: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {row["currency"]: row for row in result["currencies"]}


class TestThePlantedBook:
    def test_exposure_per_currency_is_the_worked_one(self):
        result = currency_exposure(PLANTED, "USD", RATES, hedges=HEDGES)
        rows = _rows(result)
        assert rows["EUR"]["position_net"] == 135_000.0
        assert rows["EUR"]["hedge_net"] == -90_000.0
        assert rows["EUR"]["net"] == 45_000.0
        assert rows["EUR"]["local_net"] == 40_000.0
        assert rows["EUR"]["hedge_ratio"] == pytest.approx(2 / 3, rel=1e-15)
        assert rows["GBP"]["hedge_net"] == 22_500.0
        assert rows["GBP"]["net"] == 26_250.0
        assert rows["GBP"]["local_net"] == 3_000.0 + 18_000.0
        assert rows["GBP"]["hedge_ratio"] == -6.0
        assert rows["JPY"]["net"] == -45_312.5
        assert rows["JPY"]["hedge_ratio"] == 0.0
        assert rows["USD"]["hedge_net"] == 67_500.0
        assert math.isnan(rows["USD"]["hedge_ratio"])
        assert result["hedges"] == 2
        assert result["foreign_net"] == 45_000.0 - 45_312.5 + 26_250.0
        assert result["foreign_gross"] == 45_000.0 + 45_312.5 + 26_250.0
        assert result["unhedged_foreign_gross"] == 135_000.0 + 45_312.5 + 3_750.0

    def test_the_positions_totals_are_exposure_summarys(self):
        """The hedges move exposure between currencies; the positions' own
        gross, net, long and short are the shared valuation's, unchanged."""
        result = currency_exposure(PLANTED, "USD", RATES, hedges=HEDGES)
        summary = exposure_summary(PLANTED, "USD", RATES)
        for name in ("gross", "net", "long", "short"):
            assert result[name] == getattr(summary, name)

    def test_a_hedge_has_no_value_of_its_own_in_the_base_currency(self):
        """Each hedge's two legs are equal and opposite in the base currency,
        bit for bit, so the hedges sum to nothing across the currencies."""
        result = currency_exposure(PLANTED, "USD", RATES, hedges=HEDGES)
        assert math.fsum(row["hedge_net"] for row in result["currencies"]) == 0.0


class TestNotionalIdentities:
    @pytest.mark.parametrize("seed", range(5))
    def test_each_currencys_exposure_is_its_local_notionals_times_its_rate(self, seed):
        """position_net is the fsum of each line's quantity x price x
        multiplier x rate, each correctly rounded; it agrees with local_net
        x rate to rounding. Random lines and rates that are not exact."""
        rng = random.Random(seed)
        codes = ["EUR", "JPY", "GBP", "CHF", "USD"]
        rates = {c: rng.uniform(0.005, 2.0) for c in codes if c != "USD"}
        lines = [
            _line(
                f"X{i}",
                rng.choice([-1, 1]) * rng.uniform(1, 1e6),
                rng.uniform(0.01, 5000),
                rng.choice(codes),
                multiplier=rng.choice([1.0, 10.0, 50.0]),
            )
            for i in range(40)
        ]
        result = currency_exposure(lines, "USD", rates)
        for row in result["currencies"]:
            mine = [line for line in lines if line["currency"] == row["currency"]]
            rate = 1.0 if row["currency"] == "USD" else rates[row["currency"]]
            exact = sum(
                Fraction(line["quantity"])
                * Fraction(line["price"])
                * Fraction(line["multiplier"])
                * Fraction(rate)
                for line in mine
            )
            assert row["position_net"] == pytest.approx(float(exact), rel=1e-15)
            assert row["position_net"] == pytest.approx(
                row["local_net"] * rate, rel=1e-12
            )


class TestTheReturnSplit:
    def test_the_three_terms_add_up_to_the_base_return_exactly(self):
        result = currency_exposure(
            _with_returns(PLANTED), "USD", RATES, hedges=HEDGES, fx_returns=FX_RETURNS
        )
        for line in result["returns"]["lines"]:
            parts = [line["local_return"], line["fx_return"], line["cross_return"]]
            assert math.fsum(parts) == line["base_return"]
            assert abs(sum(parts) - line["base_return"]) <= 1e-16
            pnl = [line["local_pnl"], line["fx_pnl"], line["cross_pnl"]]
            assert math.fsum(pnl) == line["pnl"]

    @pytest.mark.parametrize("seed", range(20))
    def test_the_base_return_is_the_one_worked_from_prices(self, seed):
        """An independent oracle: from a starting and an ending price and
        rate, the base-currency return is (P1 X1) / (P0 X0) - 1 in exact
        arithmetic. The split's base_return matches it to 1e-14."""
        rng = random.Random(seed)
        p0, p1 = rng.uniform(1, 500), rng.uniform(1, 500)
        x0, x1 = rng.uniform(0.005, 2), rng.uniform(0.005, 2)
        local, fx = p1 / p0 - 1, x1 / x0 - 1
        result = currency_exposure(
            [_line("A", 100, p0, "EUR", local_return=local)],
            "USD",
            {"EUR": x0},
            fx_returns={"EUR": fx},
        )
        line = result["returns"]["lines"][0]
        oracle = (Fraction(p1) * Fraction(x1)) / (Fraction(p0) * Fraction(x0)) - 1
        assert abs(line["base_return"] - float(oracle)) <= 1e-14
        assert line["cross_return"] == local * fx

    def test_a_position_in_the_base_currency_has_no_fx_term(self):
        result = currency_exposure(
            _with_returns(PLANTED), "USD", RATES, fx_returns=FX_RETURNS
        )
        aapl = next(l for l in result["returns"]["lines"] if l["instrument"] == "AAPL")
        assert aapl["fx_return"] == 0.0 and math.copysign(1, aapl["fx_return"]) == 1
        assert aapl["cross_return"] == 0.0 and aapl["fx_pnl"] == 0.0
        assert aapl["cross_pnl"] == 0.0
        assert aapl["base_return"] == 0.01
        assert aapl["pnl"] == 19_000.0 * 0.01

    def test_a_short_in_the_base_currency_has_no_negative_zero(self):
        """A short base line falling 5%: no FX term is +0.0, not -0.0."""
        result = currency_exposure(
            [_line("S", -10, 50.0, "USD", local_return=-0.05)], "USD"
        )
        line = result["returns"]["lines"][0]
        assert math.copysign(1, line["fx_pnl"]) == 1
        assert math.copysign(1, line["cross_pnl"]) == 1

    def test_the_book_totals_are_the_lines_and_the_hedges(self):
        result = currency_exposure(
            _with_returns(PLANTED), "USD", RATES, hedges=HEDGES, fx_returns=FX_RETURNS
        )
        split = result["returns"]
        lines = split["lines"]
        assert split["local_pnl"] == math.fsum(l["local_pnl"] for l in lines)
        assert split["fx_pnl"] == math.fsum(l["fx_pnl"] for l in lines)
        # The hedge legs: -67,500 USD of EUR (two legs) and +22,500 of GBP.
        assert split["hedge_pnl"] == pytest.approx(
            -90_000.0 * 0.0125 + 22_500.0 * 0.005, rel=1e-15
        )
        assert split["pnl"] == pytest.approx(
            split["local_pnl"]
            + split["fx_pnl"]
            + split["cross_pnl"]
            + split["hedge_pnl"],
            rel=1e-15,
        )
        assert split["fx_pnl_by_currency"]["EUR"] == pytest.approx(
            45_000.0 * 0.0125, rel=1e-15
        )

    def test_a_full_hedge_cancels_the_fx_pnl_and_leaves_the_cross(self):
        """Sell exactly the EUR held: the hedge's FX P&L is minus the
        position's, bit for bit, and the cross term -- the gain on the gain --
        is what a static hedge leaves."""
        lines = [_line("SAP", 1000, 120.0, "EUR", local_return=0.04)]
        result = currency_exposure(
            lines,
            "USD",
            {"EUR": 1.1},
            hedges={"EURUSD": -120_000.0},
            fx_returns={"EUR": 0.02},
        )
        row = _rows(result)["EUR"]
        assert row["net"] == 0.0 and row["hedge_ratio"] == 1.0
        split = result["returns"]
        assert split["fx_pnl"] + split["hedge_pnl"] == 0.0
        assert split["cross_pnl"] == pytest.approx(132_000.0 * 0.04 * 0.02, rel=1e-14)


class TestRelabelling:
    def _relabelled(self, mapping: Dict[str, str]):
        lines = [
            {**line, "currency": mapping[line["currency"]]}
            for line in _with_returns(PLANTED)
        ]
        rates = {mapping[c]: r for c, r in RATES.items()}
        hedges = {mapping[k[:3]] + mapping[k[3:]]: v for k, v in HEDGES.items()}
        fx = {mapping[c]: r for c, r in FX_RETURNS.items()}
        return currency_exposure(
            lines, mapping["USD"], rates, hedges=hedges, fx_returns=fx
        )

    @pytest.mark.parametrize(
        "codes",
        [("CHF", "SEK", "AUD", "NZD"), ("JPY", "USD", "EUR", "GBP")],
        ids=["new codes", "a permutation moving the base"],
    )
    def test_a_consistent_permutation_of_codes_and_rates_changes_nothing(self, codes):
        """Codes are labels: renaming USD, EUR, JPY and GBP consistently --
        in the lines, the rates, the hedges, the FX returns and the base --
        changes every number by not one bit, though the currencies sort
        into another order."""
        original = ("USD", "EUR", "JPY", "GBP")
        mapping = dict(zip(original, codes))
        before = self._relabelled(dict(zip(original, original)))
        after = self._relabelled(mapping)
        for name in ("gross", "net", "foreign_net", "foreign_gross"):
            assert after[name] == before[name]
        old_rows, new_rows = _rows(before), _rows(after)
        for code, row in old_rows.items():
            new = new_rows[mapping[code]]
            for name in ("position_net", "hedge_net", "net", "gross", "local_net"):
                assert new[name] == row[name]
            assert (
                math.isnan(new["hedge_ratio"]) and math.isnan(row["hedge_ratio"])
            ) or (new["hedge_ratio"] == row["hedge_ratio"])
        for old, new in zip(before["returns"]["lines"], after["returns"]["lines"]):
            for name in ("base_return", "pnl", "fx_pnl", "cross_pnl"):
                assert new[name] == old[name]
        for name in ("pnl", "hedge_pnl", "fx_pnl", "cross_pnl"):
            assert after["returns"][name] == before["returns"][name]

    def test_the_order_of_lines_and_hedges_changes_no_total(self):
        rng = random.Random(3)
        base = currency_exposure(
            _with_returns(PLANTED), "USD", RATES, hedges=HEDGES, fx_returns=FX_RETURNS
        )
        for _ in range(10):
            lines = _with_returns(PLANTED)
            rng.shuffle(lines)
            hedges = dict(reversed(list(HEDGES.items())))
            shuffled = currency_exposure(
                lines, "USD", RATES, hedges=hedges, fx_returns=FX_RETURNS
            )
            assert shuffled["currencies"] == base["currencies"]
            for name in ("pnl", "hedge_pnl", "local_pnl", "fx_pnl", "cross_pnl"):
                assert shuffled["returns"][name] == base["returns"][name]


class TestRefusals:
    def test_a_hedge_in_a_currency_with_no_rate_is_refused_naming_it(self):
        with pytest.raises(
            ValidationError, match=r"no rate for CHF, in hedges\['EURCHF'\]"
        ):
            currency_exposure(PLANTED, "USD", RATES, hedges={"EURCHF": -1e4})

    def test_one_pair_written_twice_is_refused(self):
        with pytest.raises(ValidationError, match="names EURUSD twice"):
            currency_exposure(
                PLANTED, "USD", RATES, hedges={"EURUSD": -1.0, "EUR/USD": -2.0}
            )

    @pytest.mark.parametrize(
        "hedges,match",
        [
            ({"EURUSD": 0.0}, "zero hedge"),
            ({"EURUSD": math.nan}, "not finite"),
            ({"EUR": -1.0}, "not a currency pair"),
        ],
    )
    def test_a_hedge_that_is_not_one_is_refused(self, hedges, match):
        with pytest.raises(ValidationError, match=match):
            currency_exposure(PLANTED, "USD", RATES, hedges=hedges)

    def test_a_missing_fx_return_is_refused_naming_the_currency(self):
        with pytest.raises(ValidationError, match="no return for GBP"):
            currency_exposure(
                _with_returns(PLANTED),
                "USD",
                RATES,
                fx_returns={"EUR": 0.01, "JPY": 0.0},
            )

    def test_the_base_currencys_own_fx_return_is_zero(self):
        with pytest.raises(ValidationError, match="exactly 0"):
            currency_exposure(
                _with_returns(PLANTED),
                "USD",
                RATES,
                fx_returns={**FX_RETURNS, "USD": 0.01},
            )

    def test_an_fx_return_at_or_below_minus_one_is_refused(self):
        with pytest.raises(ValidationError, match="cannot fall to zero"):
            currency_exposure(
                _with_returns(PLANTED),
                "USD",
                RATES,
                fx_returns={**FX_RETURNS, "EUR": -1.0},
            )

    def test_the_split_is_of_every_line_or_none(self):
        lines = _with_returns(PLANTED)
        del lines[2]["local_return"]
        with pytest.raises(ValidationError, match=r"missing on positions\[2\] 'VOD'"):
            currency_exposure(lines, "USD", RATES, fx_returns=FX_RETURNS)

    def test_fx_returns_without_local_returns_is_refused(self):
        with pytest.raises(ValidationError, match="no position has a local_return"):
            currency_exposure(PLANTED, "USD", RATES, fx_returns=FX_RETURNS)

    def test_no_returns_means_no_split(self):
        assert currency_exposure(PLANTED, "USD", RATES)["returns"] is None


class TestTheTool:
    def _arguments(self) -> Dict[str, Any]:
        return {
            "positions": _with_returns(PLANTED),
            "base_currency": "USD",
            "fx_rates": RATES,
            "hedges": HEDGES,
            "fx_returns": FX_RETURNS,
        }

    def test_it_dispatches_and_explains_its_nulls(self):
        result = resolve("finance").dispatch("get_currency_exposure", self._arguments())
        rows = _rows(result)
        assert rows["EUR"]["hedge_ratio"] == pytest.approx(2 / 3)
        assert rows["USD"]["hedge_ratio"] is None
        assert any("hedge_ratio is null" in w for w in result["warnings"])
        assert len(result["returns"]["lines"]) == 4

    def test_the_input_refuses_a_hedge_without_a_rate_at_hedges(self):
        arguments = {**self._arguments(), "hedges": {"EURCHF": -1.0}}
        with pytest.raises(pydantic.ValidationError) as caught:
            CurrencyExposureInput(**arguments)
        assert caught.value.errors()[0]["loc"] == ("hedges",)

    def test_a_nan_local_return_is_refused_at_its_line(self):
        arguments = copy.deepcopy(self._arguments())
        arguments["positions"][1]["local_return"] = math.nan
        with pytest.raises(pydantic.ValidationError) as caught:
            CurrencyExposureInput(**arguments)
        assert caught.value.errors()[0]["loc"][:3] == ("positions", 1, "local_return")

    def test_it_takes_no_ticker_and_is_synthesizable(self):
        schema = dereference(CurrencyExposureInput.model_json_schema())
        assert not _reads_market_data(schema)
        arguments, reason = synth.synthesize(CurrencyExposureInput)
        assert arguments is not None, reason


def test_every_pair_of_the_planted_hedges_nets_to_zero_in_base():
    """Over every subset of a larger hedge set -- spot and cross pairs, both
    signs -- the hedge legs sum to nothing in the base currency."""
    hedges = {"EURUSD": -1e5, "USDJPY": 3e4, "EURGBP": -2.5e4, "GBPJPY": 1e3}
    for k in range(1, len(hedges) + 1):
        for keys in itertools.combinations(hedges, k):
            result = currency_exposure(
                PLANTED, "USD", RATES, hedges={key: hedges[key] for key in keys}
            )
            assert math.fsum(row["hedge_net"] for row in result["currencies"]) == 0.0
