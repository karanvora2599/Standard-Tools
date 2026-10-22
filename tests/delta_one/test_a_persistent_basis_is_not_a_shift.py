"""
A persistent basis is not a shifted one.

THE BUG. The three CUSUM detectors on the delta one surface --
`detect_basis_dislocation`, `basis_scan` over it, and the streaming
`update_spread_monitor` -- standardized every observation by the baseline's
own standard deviation and compared the running sum with a threshold
calibrated on INDEPENDENT noise. A basis is not independent: it wanders from
its baseline for weeks with nothing having shifted, and a CUSUM sums the
wander. On spreads with no shift in them at all:

    lag-1 autocorrelation          0        0.5      0.9
    detect_basis_dislocation      2/40     26/40    40/40   (250 obs, 9.0)
    basis_scan, ten pairs                             10/10
    the monitor, 1,500 ticks      1/20     18/20    20/20   (warm-up 60)

and the monitor's 20 of 20 at 0.9 held at warm-ups of 150 and 300. Fixed,
on the same seeds: 1, 2 and 3 of 40; none of ten; and 1, 1 and 6 of 20,
falling to 2 and 0 of 20 at warm-ups of 150 and 300.

`cusum` had computed the lag-1 autocorrelation and the false-alarm rate it
implied -- 0.64 and 0.99 at 0.5 and 0.9 -- and the basis detector dropped
both. See the CHANGELOG entry for the fix.

Every spread below is an AR(1) basis around 20 bps with a 2 bps marginal sd,
on top of a spot random walk, so the persistence is known by construction.
Each null case has its detection case beside it: a fix that silenced the
detectors would pass every null here and fail every shift.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

from standard_quant_tools.agent.runtimes.delta_one import tools as delta_one_tools
from standard_quant_tools.agent.runtimes.delta_one.models import (
    BasisDislocationInput,
    SpreadMonitorInput,
)
from standard_quant_tools.delta_one.basis import detect_basis_dislocation
from standard_quant_tools.delta_one.scan import basis_scan
from standard_quant_tools.delta_one.streaming import (
    new_spread_monitor,
    update_spread_monitor,
)


def _basis(n, phi, seed, shift_sd=0.0, at=None, sd=2.0, level=20.0):
    """Spot and futures whose basis in bps is an AR(1) with lag-1
    autocorrelation `phi` and marginal sd `sd`, stepping by `shift_sd`
    marginal sds at `at`."""
    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal(n)
    shocks[0] /= np.sqrt(1.0 - phi * phi)  # the stationary start
    wander = lfilter([1.0], [1.0, -phi], shocks) * np.sqrt(1.0 - phi * phi) * sd
    if shift_sd:
        wander[at:] += shift_sd * sd
    spot = 5000.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))
    return spot, spot * (1.0 + (level + wander) / 10_000.0)


def _detect(spot, futures, **kwargs):
    return detect_basis_dislocation(
        spot=spot.tolist(), futures=futures.tolist(), **kwargs
    )


class TestTheBatchDetector:
    def test_a_persistent_basis_with_no_shift_rarely_fires(self):
        """
        Forty no-shift bases of 250 observations at each persistence. The
        threshold is now the 95th percentile of an AR(1) null with the
        basis's own lag-1 autocorrelation, measured on the reference window:
        1, 2 and 3 of 40 fire, where the fixed 9.0 fired on 2, 26 and 40.
        """
        for phi in (0.0, 0.5, 0.9):
            fired = sum(
                _detect(*_basis(250, phi, seed))["triggered"] for seed in range(40)
            )
            assert fired <= 5, f"{fired}/40 no-shift bases fired at lag-1 {phi}"

    def test_the_fixed_threshold_still_fires_on_that_noise_and_says_so(self):
        """The bug, kept reachable on purpose: a caller's 9.0 is used as
        given, and the result now reads it against this basis's memory."""
        spot, futures = _basis(250, 0.9, 0)
        result = _detect(spot, futures, threshold=9.0)
        assert result["triggered"]
        assert result["threshold_source"] == "caller"
        assert result["false_alarm_rate_at_threshold"] > 0.5
        assert any(
            "crosses" in w and "leave `threshold` unset" in w
            for w in result["warnings"]
        )

    def test_a_real_shift_is_still_found(self):
        """Twenty bases each, shifted at observation 150 of 250: 2 sd on an
        independent or AR(0.5) basis is found every time, and 3 sd on an
        AR(0.9) basis 17 times in 20 -- persistence costs power, because a
        2-sd excursion is what such a basis does by itself."""
        for phi, shift, floor in ((0.0, 2.0, 20), (0.5, 2.0, 20), (0.9, 3.0, 14)):
            found = sum(
                _detect(*_basis(250, phi, 1000 + seed, shift, 150))["triggered"]
                for seed in range(20)
            )
            assert found >= floor, f"{found}/20 shifts of {shift} sd at lag-1 {phi}"

    def test_the_diagnostics_are_returned_not_dropped(self):
        result = _detect(*_basis(250, 0.9, 3))
        assert result["threshold_source"] == "calibrated"
        assert abs(result["lag1_autocorrelation"] - 0.9) < 0.1
        assert abs(result["false_alarm_rate_at_threshold"] - 0.05) < 0.02
        assert result["threshold"] > 9.0
        assert any("lag-1 autocorrelation" in w for w in result["warnings"])

    def test_the_segmenter_s_breaks_in_persistent_noise_are_qualified(self):
        """Binary segmentation assumes independent observations and found
        breaks in 92 of 100 no-shift AR(0.9) channels. With the CUSUM quiet,
        its breaks are reported with that caveat beside them."""
        result = _detect(*_basis(250, 0.9, 5))
        assert not result["triggered"]
        assert result["n_breaks"] > 0
        assert any(
            "CUSUM did not flag" in w and "do not read these as shifts" in w
            for w in result["warnings"]
        )

    def test_the_tool_carries_the_diagnostics_on_the_wire(self):
        spot, futures = _basis(250, 0.9, 3)
        result = delta_one_tools.detect_basis_dislocation(
            BasisDislocationInput(
                spot_prices=spot.tolist(), futures_prices=futures.tolist()
            )
        )
        assert result.threshold_source == "calibrated"
        assert abs(result.lag1_autocorrelation - 0.9) < 0.1
        assert result.false_alarm_rate_at_threshold is not None


class TestTheScan:
    def test_ten_persistent_bases_and_one_that_moved(self):
        """Ten AR(0.9) bases with no shift -- all ten were flagged at the
        fixed 9.0 -- and one that stepped by 4 sd. Only the one that moved
        is flagged, and each row says what it was judged at."""
        pairs = []
        for k in range(10):
            spot, futures = _basis(250, 0.9, 500 + k)
            pairs.append(dict(label=f"P{k}", spot=spot, futures=futures))
        spot, futures = _basis(250, 0.9, 777, 4.0, 150)
        pairs.append(dict(label="moved", spot=spot, futures=futures))

        result = basis_scan(pairs, top_n=20)
        flagged = {row["label"] for row in result["ranked"] if row["shift_detected"]}
        assert flagged == {"moved"}
        assert result["threshold"] is None
        for row in result["ranked"]:
            assert row["shift_threshold"] > 9.0
            assert row["shift_lag1_autocorrelation"] > 0.6


def _stream(phi, seed, warmup, shift_sd=0.0, at=750, n=1500):
    spot, futures = _basis(n, phi, seed, shift_sd, at)
    reference = np.full(n, 5000.0)
    return update_spread_monitor(
        new_spread_monitor(warmup=warmup),
        primary=(reference * futures / spot).tolist(),
        reference=reference.tolist(),
    )


class TestTheMonitor:
    def test_a_persistent_stream_with_no_shift_rarely_alerts(self):
        """
        Twenty 1,500-tick streams each. Standardized by the long-run sd,
        the AR(0.5) streams alert once (18 of 20 before) and the AR(0.9)
        streams twice with a 150-tick warm-up and never with 300 (20 of 20
        before, at both).
        """
        for phi, warmup, ceiling in ((0.0, 60, 2), (0.5, 60, 3), (0.9, 150, 4)):
            alerted = sum(
                _stream(phi, 2000 + seed, warmup)["triggered"] for seed in range(20)
            )
            assert alerted <= ceiling, (phi, warmup, alerted)
        assert not any(
            _stream(0.9, 2000 + seed, 300)["triggered"] for seed in range(20)
        )

    def test_a_short_warm_up_on_a_persistent_stream_is_flagged(self):
        """At the default 60 ticks an AR(0.9) warm-up holds about five
        effective observations, and 6 of these 20 streams alert with no
        shift -- far below 20 of 20, and not 5%, which is why the result
        says so rather than leaving it to be discovered."""
        outs = [_stream(0.9, 2000 + seed, 60) for seed in range(20)]
        assert sum(out["triggered"] for out in outs) <= 9
        for out in outs:
            assert out["effective_warmup"] < 20
            assert any("longer warm-up" in w for w in out["warnings"])
        quiet = _stream(0.0, 2000, 60)
        assert quiet["effective_warmup"] > 20
        assert not any("longer warm-up" in w for w in quiet["warnings"])

    def test_a_real_shift_still_alerts(self):
        """2 sd on an independent stream alerts every time; 3 sd on an
        AR(0.9) stream 17 of 20 times with a 150-tick warm-up."""
        assert all(
            _stream(0.0, 3000 + seed, 60, 2.0)["triggered"] for seed in range(20)
        )
        found = sum(
            _stream(0.9, 3000 + seed, 150, 3.0)["triggered"] for seed in range(20)
        )
        assert found >= 14, found

    def test_the_tool_carries_the_diagnostics_on_the_wire(self):
        spot, futures = _basis(400, 0.9, 11)
        reference = np.full(400, 5000.0)
        result = delta_one_tools.monitor_spread_stream(
            SpreadMonitorInput(
                primary_prices=(reference * futures / spot).tolist(),
                reference_prices=reference.tolist(),
                warmup=200,
            )
        )
        assert result.lag1_autocorrelation > 0.6
        assert result.baseline_long_run_std > 2.0 * result.baseline_std
        assert result.effective_warmup < 20
        assert result.state["version"] == 4
