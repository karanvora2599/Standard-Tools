"""
Sector-neutralisation, on a blocker that had gone stale.

It was deferred for want of "per-ticker beta/sector metadata this repo does
not carry", and `TickerInfo.sector` exists with providers implementing
`get_ticker_info`. The metadata was there.

THE GROUPS ARE A PARAMETER AND NOT A LOOKUP, which is the design and not a
convenience. A step that asked a provider for a sector at fit time would
neutralise differently next month against the same panel, so a model
registered today would not reproduce — its manifest would describe a
pipeline whose behaviour lives outside it. It is also the survivorship
shape this repo documents for universe membership: today's classification
applied to history. `sector_groups` builds the map once and returns that
warning WITH it, so a caller who ignores it had to ignore something.
"""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.features.sectors import (
    group_sizes,
    sector_groups,
    singleton_warning,
)
from standard_quant_tools.modeling.preprocessing.base import FoldContext
from standard_quant_tools.modeling.preprocessing.registry import (
    PREPROCESSOR_REGISTRY,
)
from standard_quant_tools.modeling.preprocessing.steps import GroupDemean

BANKS = ("BANK_A", "BANK_B")
MINERS = ("MINE_A", "MINE_B")
GROUPS = {
    **{name: "Financials" for name in BANKS},
    **{name: "Materials" for name in MINERS},
}


def _ctx(entities, n_dates=2):
    dates = np.repeat(
        np.array(["2024-01-01", "2024-01-02"][:n_dates], dtype="datetime64[ns]"),
        len(entities),
    )
    return FoldContext(dates=dates, entities=np.array(list(entities) * n_dates))


def _demean(values, entities, groups=None, n_dates=2):
    return GroupDemean(groups=groups if groups is not None else GROUPS).transform(
        pd.DataFrame({"mom": values}), {}, _ctx(entities, n_dates)
    )["mom"]


class TestTheSectorComesOut:
    def test_the_level_goes_and_the_spread_stays(self):
        """A momentum feature on banks and miners carries the industry's
        move as well as the name's; a model fitted on it learns the
        rotation and reports it as stock selection."""
        out = _demean(
            [10.0, 12.0, 50.0, 54.0, 11.0, 13.0, 51.0, 55.0],
            BANKS + MINERS,
        )
        assert list(out) == pytest.approx([-1.0, 1.0, -2.0, 2.0, -1.0, 1.0, -2.0, 2.0])

    def test_each_date_is_its_own_cross_section(self):
        """Stateless for the reason cross_sectional_standardize is: the
        groups are formed from that date's own rows."""
        out = _demean([1.0, 3.0, 100.0, 300.0, 5.0, 7.0, 10.0, 30.0], BANKS + MINERS)
        assert list(out[:2]) == pytest.approx([-1.0, 1.0])
        assert list(out[4:6]) == pytest.approx([-1.0, 1.0])

    def test_a_feature_that_is_entirely_sector_becomes_zero(self):
        """Which is the point: nothing of it was about the name."""
        out = _demean([5.0, 5.0, 9.0, 9.0, 5.0, 5.0, 9.0, 9.0], BANKS + MINERS)
        assert list(out) == pytest.approx([0.0] * 8)


class TestTheDegenerateCases:
    def test_a_group_of_one_is_nan_not_zero(self):
        """A singleton's mean is the name itself, so demeaning gives a
        fabricated 'average for its sector' that destroys the feature while
        looking like a measurement. The same call `_stage_rank` makes."""
        groups = {"BANK_A": "Financials", "MINE_A": "Materials"}
        out = _demean([1.0, 2.0, 3.0, 4.0], ("BANK_A", "MINE_A"), groups, n_dates=2)
        assert out.isna().all()

    def test_an_unmapped_entity_is_nan(self):
        """A name with no sector has no sector-relative position."""
        groups = {name: "Financials" for name in BANKS}
        out = _demean(
            [10.0, 12.0, 50.0, 54.0, 11.0, 13.0, 51.0, 55.0],
            BANKS + MINERS,
            groups,
        )
        assert not out.iloc[:2].isna().any()
        assert out.iloc[2:4].isna().all()

    def test_a_missing_value_does_not_poison_its_group(self):
        out = _demean(
            [10.0, np.nan, 50.0, 54.0, 11.0, 13.0, 51.0, 55.0],
            BANKS + MINERS,
        )
        # One bank has no value; the other has no peer left, so NaN.
        assert out.iloc[:2].isna().all()
        assert not out.iloc[2:4].isna().any()


class TestItRefusesRatherThanGuess:
    def test_without_entities_there_is_no_cross_section(self):
        with pytest.raises(ValidationError, match="entity of each row"):
            GroupDemean(groups=GROUPS).transform(
                pd.DataFrame({"mom": [1.0, 2.0]}),
                {},
                FoldContext(
                    dates=np.array(["2024-01-01"] * 2, dtype="datetime64[ns]")
                ),
            )

    def test_without_a_map_it_will_not_invent_one(self):
        """The refusal names why the groups are a parameter."""
        with pytest.raises(ValidationError, match="would neutralise differently"):
            _demean([1.0, 2.0, 3.0, 4.0], BANKS, groups={}, n_dates=2)

    def test_mismatched_dates(self):
        with pytest.raises(ValidationError, match="one date per row"):
            GroupDemean(groups=GROUPS).transform(
                pd.DataFrame({"mom": [1.0, 2.0]}),
                {},
                FoldContext(
                    dates=np.array(["2024-01-01"], dtype="datetime64[ns]"),
                    entities=np.array(["BANK_A", "BANK_B"]),
                ),
            )


class TestTheMapCarriesItsWarning:
    class _Provider:
        def __init__(self, sectors):
            self._sectors = sectors

        def get_ticker_info(self, symbol):
            if symbol not in self._sectors:
                raise KeyError(symbol)

            class _Info:
                sector = self._sectors[symbol]

            return _Info()

    def test_the_survivorship_sentence_is_always_there(self):
        """A map that arrived without it would be used as though it were
        point-in-time."""
        _map, warnings = sector_groups(
            ["A", "B"], self._Provider({"A": "Financials", "B": "Materials"})
        )
        assert any("TODAY'S CLASSIFICATION" in w for w in warnings)
        assert any("survivorship" in w for w in warnings)

    def test_an_unknown_sector_is_left_out_rather_than_pooled(self):
        """Names that share only 'nobody knows' are not a sector, and
        demeaning them against each other would invent a factor out of
        ignorance."""
        mapped, warnings = sector_groups(
            ["A", "B"], self._Provider({"A": "Financials", "B": "Unknown"})
        )
        assert mapped == {"A": "Financials"}
        assert any("not pooled into one 'Unknown' group" in w for w in warnings)

    def test_a_lookup_failure_does_not_stop_the_universe(self):
        mapped, warnings = sector_groups(
            ["A", "B"], self._Provider({"A": "Financials"})
        )
        assert mapped == {"A": "Financials"}
        assert any("could not be looked up" in w for w in warnings)

    def test_the_group_sizes_can_be_read_before_fitting(self):
        assert group_sizes(GROUPS) == {"Financials": 2, "Materials": 2}

    def test_a_map_of_singletons_is_warned_about(self):
        """Worth seeing before a run rather than as an all-NaN feature
        afterwards."""
        assert singleton_warning(GROUPS) is None
        line = singleton_warning({"A": "Financials", "B": "Materials"})
        assert line and "no peer to be demeaned against" in line


class TestItIsRegistered:
    def test_the_catalog_carries_it(self):
        definition = PREPROCESSOR_REGISTRY["group_demean"]
        assert definition.cls is GroupDemean
        assert definition.default_params == {"groups": {}}
        assert "PARAMETER, not a lookup" in definition.description

    def test_it_is_stateless(self):
        assert GroupDemean.stateless is True
        assert GroupDemean(groups=GROUPS).fit(
            pd.DataFrame({"mom": [1.0]}),
            FoldContext(
                dates=np.array(["2024-01-01"], dtype="datetime64[ns]"),
                entities=np.array(["BANK_A"]),
            ),
        ) == {}
