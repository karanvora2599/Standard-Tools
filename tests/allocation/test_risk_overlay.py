"""
Risk overlays held to their closed forms, and to the rule that an exposure
is decided from the past.

Volatility targeting on i.i.d. returns realizes within 5% of its target
over eighty years of days, and its exposure at t is target / sigma_t with
sigma_t the EWMA through t - 1, recomputed here. CPPI's cushion is C_t =
C_{t-1} (1 + m r_t) (Black & Perold 1992), so a one-period gap worse than
-1/m breaches the floor exactly once, by exactly C (1 + m r) -- and the
overlay is then cash-locked; a gap a hair smaller does not breach. With no
floor at all CPPI is the constant exposure m. TIPP's floor is the floor
fraction of the peak. Changing any return after t changes no exposure at
or before t, under every method.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError as PydanticValidationError

from standard_quant_tools.agent.runtimes import handoff, resolve
from standard_quant_tools.agent.runtimes.streams import returns_ref_streams
from standard_quant_tools.allocation.overlays import apply_overlay
from standard_quant_tools.error import ValidationError

P = 252


def _series(values, start="2001-01-02", name="s"):
    return pd.Series(
        np.asarray(values, dtype=float),
        index=pd.bdate_range(start, periods=len(values)),
        name=name,
    )


def _dated(series):
    return {d.strftime("%Y-%m-%d"): float(v) for d, v in series.items()}


def _call(arguments):
    return resolve("allocation").dispatch("apply_risk_overlay", arguments)


class TestTheVolatilityTarget:
    def test_iid_returns_realize_within_five_percent_of_the_target(self):
        """20,000 i.i.d. normal days at 16% a year, a 10% target and no
        binding cap: realized / target in [0.95, 1.05]. (It sits about 3%
        above one: the EWMA's noise raises E[1 / sigma_t^2].)"""
        r = _series(np.random.default_rng(1).normal(0.0003, 0.01, 20_000))
        out = apply_overlay(
            r, method="vol_target", periods_per_year=P, max_leverage=10.0
        )
        assert 0.95 <= out["realized_over_target"] <= 1.05
        assert out["share_at_max_leverage"] == 0.0

    def test_the_exposure_is_target_over_the_lagged_ewma(self):
        """sigma_t^2 seeded with the mean square of the first `warmup`
        returns, then lambda sigma^2 + (1 - lambda) r_{t-1}^2: the exposure
        at t uses nothing from t on."""
        r = _series(np.random.default_rng(2).normal(0.0, 0.012, 400))
        out = apply_overlay(
            r,
            method="vol_target",
            periods_per_year=P,
            target_volatility=0.08,
            ewma_lambda=0.9,
            warmup=30,
            max_leverage=5.0,
        )
        values = r.to_numpy()
        variance = float(np.mean(values[:30] ** 2))
        expected = []
        for t in range(30, 400):
            expected.append(min(5.0, 0.08 / math.sqrt(P) / math.sqrt(variance)))
            variance = 0.9 * variance + 0.1 * values[t] ** 2
        np.testing.assert_allclose(out["exposure"].to_numpy(), expected, rtol=1e-14)
        assert out["exposure"].index[0] == r.index[30]
        assert out["n_periods"] == 370 and out["warmup_periods"] == 30

    def test_the_cap_binds_and_is_counted(self):
        r = _series(np.random.default_rng(3).normal(0.0, 0.002, 300))
        out = apply_overlay(r, method="vol_target", periods_per_year=P)
        assert out["exposure_max"] == 1.0
        assert out["share_at_max_leverage"] == 1.0


class TestCppiAndTheGap:
    @staticmethod
    def _path(m, f, before, gap, after):
        return _series(list(before) + [gap] + list(after))

    def test_a_gap_worse_than_one_over_m_breaches_once_by_the_closed_form(self):
        """Small random returns, then a -30% day at m = 4 (-1/m = -25%),
        then more days. The cushion before the gap is C0 prod(1 + m r)
        with C0 = 0.2; the gap leaves wealth C (1 + m r) below the floor;
        one breach, and the exposure is zero from the next day on."""
        rng = np.random.default_rng(4)
        before = rng.normal(0.0005, 0.004, 40)
        after = rng.normal(0.001, 0.01, 30)
        m, f = 4.0, 0.8
        r = self._path(m, f, before, -0.30, after)
        out = apply_overlay(
            r, method="cppi", periods_per_year=P, multiplier=m, floor=f, max_leverage=50
        )
        cushion = (1 - f) * np.prod(1 + m * before)
        assert out["floor_breaches"] == 1
        breach = cushion * (1 + m * -0.30)
        assert breach < 0
        assert out["worst_breach"] == pytest.approx(breach / f, rel=1e-10)
        assert out["cash_locked"] is True
        assert out["cash_locked_from"] == r.index[41]
        assert np.all(out["exposure"].to_numpy()[41:] == 0.0)
        # The exposure before the gap is m x cushion / wealth, cushion by
        # the product: the whole path is the closed form.
        c = (1 - f) * np.concatenate([[1.0], np.cumprod(1 + m * before)])
        np.testing.assert_allclose(
            out["exposure"].to_numpy()[:41], m * c / (f + c), rtol=1e-12
        )

    def test_a_gap_just_inside_one_over_m_does_not_breach(self):
        r = self._path(4.0, 0.8, [0.0] * 10, -0.2499, [0.0] * 10)
        out = apply_overlay(
            r, method="cppi", periods_per_year=P, multiplier=4.0, floor=0.8
        )
        assert out["floor_breaches"] == 0
        assert out["cash_locked"] is False
        assert math.isnan(out["worst_breach"])

    def test_no_floor_is_the_constant_exposure_m(self):
        """floor = 0 puts the floor infinitely far away: the cushion is all of
        wealth, so the exposure is m every period and, at zero cost, the
        account earns m r + (1 - m) c exactly."""
        r = _series(np.random.default_rng(5).normal(0.0002, 0.01, 250))
        out = apply_overlay(
            r,
            method="cppi",
            periods_per_year=P,
            multiplier=2.5,
            floor=0.0,
            max_leverage=3.0,
            cash_rate=0.02,
        )
        assert np.all(out["exposure"].to_numpy() == 2.5)
        np.testing.assert_allclose(
            out["overlaid"].to_numpy(),
            2.5 * r.to_numpy() + (1 - 2.5) * 0.02 / P,
            rtol=0,
            atol=1e-17,
        )
        assert out["floor_breaches"] == 0

    def test_tipp_ratchets_its_floor_with_the_peak(self):
        """At a zero cash rate TIPP's floor is the floor fraction of the
        running peak of wealth, never lower."""
        r = _series(np.random.default_rng(6).normal(0.0006, 0.01, 300))
        out = apply_overlay(
            r, method="tipp", periods_per_year=P, multiplier=3.0, floor=0.85
        )
        wealth = np.cumprod(1 + out["overlaid"].to_numpy())
        peak = max(1.0, float(wealth.max()))
        assert out["final_floor"] == pytest.approx(0.85 * peak, rel=1e-12)
        cppi = apply_overlay(
            r, method="cppi", periods_per_year=P, multiplier=3.0, floor=0.85
        )
        assert cppi["final_floor"] == pytest.approx(0.85, rel=1e-15)


class TestNothingIsDecidedFromTheFuture:
    @pytest.mark.parametrize("method", ["vol_target", "cppi", "tipp"])
    def test_changing_returns_after_t_changes_no_exposure_up_to_t(self, method):
        """Every return after t is replaced; every exposure at or before t is
        the same bits."""
        rng = np.random.default_rng(7)
        base = rng.normal(0.0004, 0.012, 500)
        changed = base.copy()
        t = 300
        changed[t + 1 :] = rng.normal(-0.01, 0.05, 500 - t - 1)
        one = apply_overlay(_series(base), method=method, periods_per_year=P)
        two = apply_overlay(_series(changed), method=method, periods_per_year=P)
        cut = one["exposure"].index.get_loc(_series(base).index[t])
        a = one["exposure"].to_numpy()[: cut + 2]
        b = two["exposure"].to_numpy()[: cut + 2]
        assert a.tobytes() == b.tobytes()
        assert one["exposure"].to_numpy()[cut + 2 :].tobytes() != (
            two["exposure"].to_numpy()[cut + 2 :].tobytes()
        )


class TestCosts:
    def test_cost_is_bps_times_turnover(self):
        """The first entry from cash counts as turnover."""
        r = _series(np.random.default_rng(8).normal(0.0004, 0.012, 300))
        free = apply_overlay(r, method="vol_target", periods_per_year=P)
        paid = apply_overlay(r, method="vol_target", periods_per_year=P, cost_bps=10)
        assert paid["turnover"] == free["turnover"]
        assert paid["cost_total"] == pytest.approx(free["turnover"] * 1e-3, rel=1e-12)
        diff = free["overlaid"].to_numpy() - paid["overlaid"].to_numpy()
        changes = np.abs(np.diff(free["exposure"].to_numpy(), prepend=0.0))
        np.testing.assert_allclose(diff, changes * 1e-3, rtol=0, atol=1e-16)


class TestTheTool:
    def test_a_run_id_publishes_the_stream_and_the_exposure(self):
        """The overlaid stream is a returns_panel any stream tool reads, the
        exposure an analytic_series; a repeat at the same run_id is
        refused, as every publisher's is."""
        r = _series(np.random.default_rng(9).normal(0.0004, 0.012, 200))
        out = _call({"returns": _dated(r), "run_id": "overlay_pub"})
        stream = returns_ref_streams(out["overlaid_returns_ref"], what="test")
        assert list(stream) == ["returns"]
        local = apply_overlay(
            r.rename("returns"), method="vol_target", periods_per_year=P
        )
        np.testing.assert_array_equal(
            stream["returns"].to_numpy(), local["overlaid"].to_numpy()
        )
        exposure = handoff.resolve(out["exposure_ref"], expect="analytic_series")
        np.testing.assert_array_equal(exposure.to_numpy(), local["exposure"].to_numpy())
        with pytest.raises(ValidationError, match="already published"):
            _call({"returns": _dated(r), "run_id": "overlay_pub"})

    def test_a_reference_reads_as_the_inline_stream(self):
        rng = np.random.default_rng(10)
        frame = pd.DataFrame(
            rng.normal(0.0003, 0.01, (260, 2)),
            index=pd.bdate_range("2019-01-02", periods=260),
            columns=["a", "b"],
        )
        ref = handoff.publish(
            frame, kind="returns_panel", run_id="overlay_ref", name="p"
        )
        by_ref = _call({"returns_ref": ref, "column": "b", "method": "tipp"})
        inline = _call({"returns": _dated(frame["b"]), "method": "tipp"})
        for key in ("realized_volatility", "max_drawdown_after", "turnover"):
            assert by_ref[key] == inline[key]
        assert by_ref["stream"] == "b"
        with pytest.raises(ValidationError, match="Name it in column"):
            _call({"returns_ref": ref})

    def test_the_figures_a_method_does_not_have_are_unset(self):
        r = _series(np.random.default_rng(11).normal(0.0003, 0.01, 200))
        vt = _call({"returns": _dated(r)})
        assert vt["floor_breaches"] is None and vt["final_floor"] is None
        assert not any("floor" in w for w in vt["warnings"])
        cp = _call({"returns": _dated(r), "method": "cppi"})
        assert cp["target_volatility"] is None and cp["realized_over_target"] is None
        assert cp["floor_breaches"] == 0
        assert any("worst_breach is null" in w for w in cp["warnings"])

    @pytest.mark.parametrize(
        "extra,needle",
        [
            ({"multiplier": 4.0}, "not read by method='vol_target'"),
            ({"method": "cppi", "target_volatility": 0.2}, "not read by method='cppi'"),
            ({"column": "a"}, "column"),
            ({"returns_ref": "sqt://returns_panel/x/y"}, "exactly one"),
            ({"cash_rate": 1e308}, "cash_rate"),
            ({"floor": 1.0}, "floor"),
        ],
    )
    def test_the_input_model_refuses_by_name(self, extra, needle):
        r = _series(np.random.default_rng(12).normal(0.0, 0.01, 100))
        with pytest.raises(PydanticValidationError, match=needle):
            _call({"returns": _dated(r), **extra})

    def test_a_stream_no_longer_than_the_warmup_is_refused(self):
        r = _series(np.random.default_rng(13).normal(0.0, 0.01, 21))
        with pytest.raises(ValidationError, match="warmup"):
            _call({"returns": _dated(r)})
