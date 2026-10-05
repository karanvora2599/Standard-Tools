"""
`source` names the vendor and `tier` names the layer that answered.

They were one field, and three of four providers put the layer in it:

    yfinance   source="session_cache" / "disk_cache" / "live_fetch"
    polygon    the same three for bars, and "polygon" for everything else
    bloomberg  source="live_fetch"
    databento  source="databento:EQUS.SUMMARY:session_cache"

Only the last named a vendor at all, and even that one glued the layer on the
end. The cost is not cosmetic, because the revision detector keys a window's
observations BY THIS FIELD, so a naming collision inverts its verdict in both
directions at once:

  - one vendor read from two layers looks like two sources disagreeing;
  - two vendors that both said "live_fetch" look like one source restating.

The second is the worse of the two. Two vendors holding different hashes for
the same window is the one thing `sources_disagree` exists to catch, and it
was reported as a revision instead.

These tests are about the field, not about any provider: they build the
records by hand and run the detector over them, because the question is what
the detector concludes and a provider fake would only restate the fixture.
"""

from __future__ import annotations

import pytest

from standard_quant_tools.audit import context


@pytest.fixture()
def recording():
    """An open decision record, so the recorders write somewhere."""
    token = context._data_sources_var.set([])
    try:
        yield lambda: list(context._data_sources_var.get() or [])
    finally:
        context._data_sources_var.reset(token)


def test_a_frame_access_records_the_vendor_and_the_layer(recording):
    import pandas as pd

    frame = pd.DataFrame({"Close": [1.0, 2.0]}, index=pd.date_range("2024-01-02", periods=2))
    context.record_frame_access(
        "AAPL", "2024-01-02", "2024-01-03", "1d", "yfinance", frame, tier="disk_cache"
    )
    (entry,) = recording()
    assert entry["source"] == "yfinance"
    assert entry["tier"] == "disk_cache"


def test_a_data_access_records_them_too(recording):
    context.record_data_access(
        "AAPL", "2024-01-02", "2024-01-03", "1d", "polygon", "abc123", tier="live_fetch"
    )
    (entry,) = recording()
    assert entry["source"] == "polygon" and entry["tier"] == "live_fetch"


def test_a_recorder_given_no_tier_writes_no_tier_key(recording):
    """Absent, not null.

    A record from before this split carries no `tier`, and a reader must not
    have to tell "the layer was not recorded" from "the layer was None".
    """
    context.record_data_access("AAPL", "2024-01-02", "2024-01-03", "1d", "polygon", "x")
    (entry,) = recording()
    assert "tier" not in entry


class TestTheRevisionDetectorReadsItCorrectly:
    """What the field is FOR.

    `gates.data_revisions` in the engine groups by `source`; the shape of the
    bug is the grouping, so these build the records the detector reads.
    """

    @staticmethod
    def _grouped(entries):
        """The detector's grouping step, which is where the field is read."""
        windows: dict = {}
        for entry in entries:
            key = (entry["symbol"], entry["start"], entry["end"], entry["interval"])
            windows.setdefault(key, {}).setdefault(entry["source"], []).append(
                entry["content_hash"]
            )
        return windows

    def test_one_vendor_read_from_two_layers_is_one_source(self, recording):
        for tier, digest in (("live_fetch", "h1"), ("disk_cache", "h2")):
            context.record_data_access(
                "AAPL", "2024-01-02", "2024-01-31", "1d", "yfinance", digest, tier=tier
            )
        (sources,) = self._grouped(recording()).values()
        assert list(sources) == ["yfinance"], "one vendor, however many layers"
        assert sources["yfinance"] == ["h1", "h2"], (
            "two observations of one source, which is a REVISION to look at "
            "-- not two sources disagreeing"
        )

    def test_two_vendors_are_two_sources_though_both_fetched_live(self, recording):
        for vendor, digest in (("yfinance", "h1"), ("bloomberg", "h2")):
            context.record_data_access(
                "AAPL", "2024-01-02", "2024-01-31", "1d", vendor, digest,
                tier="live_fetch",
            )
        (sources,) = self._grouped(recording()).values()
        assert sorted(sources) == ["bloomberg", "yfinance"], (
            "both said live_fetch, and under the old field they collapsed into "
            "one source -- so a real cross-vendor disagreement, the thing the "
            "detector exists to find, was reported as a revision"
        )

    def test_two_databento_feeds_stay_two_sources(self, recording):
        """The dataset belongs IN the source, and only the layer moved out.

        EQUS.SUMMARY and XNAS.ITCH really are different feeds, and a hash
        difference between them is a disagreement worth reporting.
        """
        for dataset, digest in (("EQUS.SUMMARY", "h1"), ("XNAS.ITCH", "h2")):
            context.record_data_access(
                "AAPL", "2024-01-02", "2024-01-31", "1d",
                f"databento:{dataset}", digest, tier="live_fetch",
            )
        (sources,) = self._grouped(recording()).values()
        assert sorted(sources) == ["databento:EQUS.SUMMARY", "databento:XNAS.ITCH"]

    def test_one_databento_feed_read_twice_is_one_source(self, recording):
        for tier, digest in (("live_fetch", "h1"), ("session_cache", "h2")):
            context.record_data_access(
                "AAPL", "2024-01-02", "2024-01-31", "1d",
                "databento:EQUS.SUMMARY", digest, tier=tier,
            )
        (sources,) = self._grouped(recording()).values()
        assert list(sources) == ["databento:EQUS.SUMMARY"]


class TestEveryProviderNamesItself:
    """The positive control on the whole change.

    Four providers, one convention. A provider that goes back to writing a
    layer name into `source` fails here rather than quietly inverting a
    verdict in a view nobody re-reads.
    """

    @pytest.mark.parametrize(
        ("module", "vendor"),
        [
            ("yfinance_provider", "yfinance"),
            ("polygon_provider", "polygon"),
            ("bloomberg_provider", "bloomberg"),
        ],
    )
    def test_no_provider_passes_a_layer_name_as_its_source(self, module, vendor):
        import inspect

        import standard_quant_tools.data as data_package

        source = inspect.getsource(getattr(data_package, module))
        for layer in ("live_fetch", "session_cache", "disk_cache"):
            assert f'source="{layer}"' not in source, (
                f"{module} passes {layer!r} as its source; it is a tier"
            )
        assert f'source="{vendor}"' in source

    def test_databento_does_not_glue_a_layer_onto_its_dataset(self):
        import inspect

        from standard_quant_tools.data import databento_provider

        source = inspect.getsource(databento_provider)
        for layer in ("session_cache", "disk_cache"):
            assert f'{{dataset}}:{layer}' not in source
            assert f'{{first}}:{layer}' not in source
