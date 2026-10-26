"""
`PredictionTransformSpec.method='mean_variance'`: a model's predictions
into the exact cost-aware optimizer.

Each piece is held to an answer computed without the code under test:

- the realized rank IC against Spearman's closed form for a bivariate
  normal, (6/pi) asin(rho/2), on predictions built to correlate with their
  own forward returns at a planted rho;
- the trailing IC's shrink and overlap count against the formula, and its
  point-in-time cut against dates whose windows had not closed;
- the weights against the mean-variance closed forms: proportional to z
  when the covariance is sigma^2 I and nothing binds, and the two-asset
  fractional Kelly book f * S^-1 alpha;
- the no-trade region, where the optimizer's own optimality condition,
  checked independently, says the previous book is optimal;
- a future perturbation: prices and predictions after t changed, nothing
  at or before t moves.
"""

import math

import numpy as np
import pandas as pd
import pytest
from scipy.special import ndtri
from scipy.stats import rankdata

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.portfolio_eval import (
    cross_sectional_z,
    holding_period_bars,
    mean_variance_weights,
    realized_rank_ic,
    trailing_ic,
    transform_predictions_to_weights,
)
from standard_quant_tools.modeling.specs import (
    MeanVarianceSpec,
    PredictionTransformSpec,
)

# ── Planted data ────────────────────────────────────────────────────────


def _random_walk_closes(n_bars: int, names, seed: int, vol: float = 0.01):
    """Closes whose log returns are iid normal, so a forward log return
    over any horizon is exactly normal across names."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_bars, freq="B")
    log_returns = rng.normal(0.0, vol, (n_bars, len(names)))
    log_returns[0] = 0.0
    return pd.DataFrame(
        100.0 * np.exp(np.cumsum(log_returns, axis=0)), index=dates, columns=names
    )


def _predictions_with_ic(closes: pd.DataFrame, horizon: int, rho: float, seed: int):
    """Predictions dated s equal to rho * (standardized forward log return
    from s) + sqrt(1 - rho^2) * noise: bivariate normal with Pearson
    correlation rho against the return they are scored on, on every date
    with a closed window."""
    rng = np.random.default_rng(seed)
    logs = np.log(closes.to_numpy())
    forward = logs[horizon:] - logs[:-horizon]
    standard = (forward - forward.mean(axis=1, keepdims=True)) / forward.std(
        axis=1, keepdims=True
    )
    noise = rng.normal(0.0, 1.0, standard.shape)
    values = rho * standard + math.sqrt(1.0 - rho * rho) * noise
    return pd.DataFrame(
        values, index=closes.index[: len(values)], columns=closes.columns
    )


# ── The realized IC ─────────────────────────────────────────────────────


class TestRealizedRankIC:
    @pytest.mark.parametrize("rho", [0.1, 0.3])
    def test_a_planted_ic_is_recovered(self, rho):
        """Spearman's rho of a bivariate normal with Pearson correlation
        rho is (6/pi) asin(rho/2). Rank correlation is invariant to the
        monotone map from log to simple return, so the closed form holds
        for the simple returns the function reads. 600 dates of 150 names:
        the per-date standard error is about 0.08, so the mean's is about
        0.003 and a tolerance of 0.012 is four of them."""
        names = [f"N{i:03d}" for i in range(150)]
        closes = _random_walk_closes(610, names, seed=11)
        predictions = _predictions_with_ic(closes, horizon=5, rho=rho, seed=12)
        frame = realized_rank_ic(predictions, closes, horizon=5)
        expected = 6.0 / math.pi * math.asin(rho / 2.0)
        assert len(frame) == len(predictions)
        assert abs(frame["ic"].mean() - expected) < 0.012

    def test_each_row_is_known_only_when_its_window_closes(self):
        """`known_at` is the bar the forward return was last priced on, and
        a prediction whose window runs past the calendar has no IC yet."""
        names = ["A", "B", "C", "D"]
        closes = _random_walk_closes(40, names, seed=1)
        predictions = pd.DataFrame(
            np.random.default_rng(2).normal(size=(40, 4)),
            index=closes.index,
            columns=names,
        )
        frame = realized_rank_ic(predictions, closes, horizon=3)
        assert (frame["known_at"] - frame["position"] == 3).all()
        assert frame["known_at"].max() == 39
        assert len(frame) == 37

    def test_one_date_against_a_direct_computation(self):
        """The ranks of the predictions against the ranks of the realized
        forward returns, correlated by hand."""
        names = ["A", "B", "C", "D", "E"]
        closes = _random_walk_closes(12, names, seed=3)
        predictions = pd.DataFrame(
            np.random.default_rng(4).normal(size=(12, 5)),
            index=closes.index,
            columns=names,
        )
        frame = realized_rank_ic(predictions, closes, horizon=2)
        row = closes.to_numpy()
        forward = row[6] / row[4] - 1.0
        expected = np.corrcoef(rankdata(predictions.iloc[4]), rankdata(forward))[0, 1]
        assert frame.set_index("position").loc[4, "ic"] == pytest.approx(expected)


class TestTrailingIC:
    def _frame(self, values, spacing=1, horizon=5):
        positions = np.arange(len(values)) * spacing
        return pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=len(values)),
                "position": positions,
                "known_at": positions + horizon,
                "ic": values,
                "n_names": 50,
            }
        )

    def test_the_shrink_is_one_minus_one_over_t_squared(self):
        """Weekly-spaced ICs at a five-bar horizon do not overlap, so the
        t-statistic counts every date."""
        values = np.array([0.08, 0.02, 0.06, -0.01, 0.05, 0.04, 0.03, 0.07])
        frame = self._frame(values, spacing=5)
        out = trailing_ic(
            frame, 10_000, lookback=100, min_dates=4, horizon=5, shrink=True
        )
        mean = values.mean()
        t = mean / (values.std(ddof=1) / math.sqrt(len(values)))
        assert out["n_ic_effective"] == len(values)
        assert out["ic_t_stat"] == pytest.approx(t)
        assert out["ic_used"] == pytest.approx(mean * (1.0 - 1.0 / t**2))

    def test_overlapping_windows_count_once_per_horizon(self):
        """Daily ICs over a five-bar horizon: 50 dates span 49 bars, so
        ten non-overlapping windows, and t is divided by sqrt(10), not
        sqrt(50)."""
        values = np.random.default_rng(5).normal(0.03, 0.1, 50)
        out = trailing_ic(
            self._frame(values),
            10_000,
            lookback=100,
            min_dates=4,
            horizon=5,
            shrink=False,
        )
        assert out["n_ic_effective"] == 10
        expected_t = values.mean() / (values.std(ddof=1) / math.sqrt(10))
        assert out["ic_t_stat"] == pytest.approx(expected_t)
        assert out["ic_used"] == pytest.approx(max(values.mean(), 0.0))

    def test_only_closed_windows_are_read(self):
        """At position p a row is usable once known_at <= p; the last
        `lookback` of those are averaged."""
        values = np.arange(1, 21, dtype=float) / 100.0
        frame = self._frame(values)
        out = trailing_ic(frame, 12, lookback=5, min_dates=2, horizon=5, shrink=False)
        # Rows at positions 0..7 have known_at 5..12; the last five are 3..7.
        assert out["n_ic_dates"] == 5
        assert out["ic_raw"] == pytest.approx(values[3:8].mean())

    def test_a_negative_ic_is_no_view(self):
        values = np.full(30, -0.05) + np.random.default_rng(6).normal(0, 0.01, 30)
        for shrink in (True, False):
            out = trailing_ic(
                self._frame(values),
                10_000,
                lookback=100,
                min_dates=4,
                horizon=1,
                shrink=shrink,
            )
            assert out["ic_raw"] < 0.0
            assert out["ic_used"] == 0.0

    def test_before_min_dates_there_is_no_ic(self):
        out = trailing_ic(
            self._frame(np.full(10, 0.05)),
            10_000,
            lookback=100,
            min_dates=11,
            horizon=1,
            shrink=False,
        )
        assert out["ic_warmup"] is True
        assert out["ic_raw"] is None and out["ic_used"] == 0.0


# ── The score ───────────────────────────────────────────────────────────


class TestCrossSectionalZ:
    def test_zscore_is_the_standardized_prediction(self):
        values = np.array([1.0, 2.0, 4.0, np.nan, 9.0])
        z = cross_sectional_z(values, "zscore")
        finite = values[np.isfinite(values)]
        expected = (finite - finite.mean()) / finite.std()
        assert np.isnan(z[3])
        np.testing.assert_allclose(z[np.isfinite(values)], expected, rtol=1e-12)

    def test_rank_normal_is_the_inverse_normal_of_the_ranks(self):
        values = np.array([0.3, -2.0, 50.0, 0.1])
        z = cross_sectional_z(values, "rank_normal")
        raw = ndtri((rankdata(values) - 0.5) / 4)
        np.testing.assert_allclose(z, (raw - raw.mean()) / raw.std(), rtol=1e-12)

    def test_an_outlier_is_one_rank_under_rank_normal(self):
        """The reason it exists: a prediction a hundred times the others
        scores as the top rank and no more."""
        ordinary = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        outlier = np.array([0.1, 0.2, 0.3, 0.4, 50.0])
        np.testing.assert_array_equal(
            cross_sectional_z(ordinary, "rank_normal"),
            cross_sectional_z(outlier, "rank_normal"),
        )
        assert cross_sectional_z(outlier, "zscore")[-1] > 1.9

    def test_no_spread_is_no_view(self):
        np.testing.assert_array_equal(
            cross_sectional_z(np.full(4, 0.7), "zscore"), np.zeros(4)
        )


class TestHoldingPeriod:
    @pytest.mark.parametrize(
        "frequency, periods, expected",
        [("daily", 252, 1.0), ("weekly", 252, 252 / 52), ("monthly", 252, 21.0)],
    )
    def test_bars_between_rebalances(self, frequency, periods, expected):
        assert holding_period_bars(frequency, periods) == pytest.approx(expected)


# ── Refusals ────────────────────────────────────────────────────────────


class TestTheSpecRefuses:
    @pytest.mark.parametrize(
        "field, value",
        [
            ("long_quantile", 0.3),
            ("short_quantile", 0.1),
            ("volatility_scale", True),
            ("volatility_lookback", 60),
        ],
    )
    def test_a_field_the_method_never_reads(self, field, value):
        with pytest.raises(ValueError, match=field):
            PredictionTransformSpec(method="mean_variance", **{field: value})

    def test_an_optimizer_beside_another_method(self):
        with pytest.raises(ValueError, match="optimizer is read only"):
            PredictionTransformSpec(method="cross_sectional_rank", optimizer={})

    @pytest.mark.parametrize(
        "kwargs, named",
        [
            ({"kelly_fraction": 0.5}, "kelly_fraction"),
            ({"sizing": "kelly", "target_volatility": 0.2}, "target_volatility"),
            ({"sizing": "kelly", "risk_aversion": 3.0}, "risk_aversion"),
            ({"ic": 0.05, "ic_shrink": True}, "ic_shrink"),
            ({"ic_min_dates": 300}, "ic_min_dates"),
            ({"cost_aware": False, "impact": 0.1}, "cost_aware"),
            ({"bogus": 1}, "bogus"),
        ],
    )
    def test_an_optimizer_field_that_means_nothing(self, kwargs, named):
        with pytest.raises(ValueError, match=named):
            MeanVarianceSpec(**kwargs)

    def test_the_defaults_and_a_round_trip_are_accepted(self):
        """A full dump of a spec validates again: the refusals compare to
        defaults, not to which fields were passed."""
        spec = PredictionTransformSpec(
            method="mean_variance",
            optimizer={"sizing": "kelly", "kelly_fraction": 0.5},
        )
        assert PredictionTransformSpec(**spec.model_dump()) == spec

    def test_the_per_date_transform_points_at_the_optimizer_path(self):
        panel = pd.DataFrame(
            np.ones((3, 3)),
            index=pd.date_range("2024-01-01", periods=3),
            columns=["A", "B", "C"],
        )
        with pytest.raises(ValidationError, match="mean_variance_weights"):
            transform_predictions_to_weights(
                panel, PredictionTransformSpec(method="mean_variance")
            )


# ── The weights against closed forms ────────────────────────────────────

#: The Hadamard return unit, a power of two so every return is exact.
_UNIT = 2.0**-7


def _hadamard_closes(names, n_bars: int, period: int = 16) -> pd.DataFrame:
    """Closes whose bar returns cycle through the rows of a Hadamard matrix
    (its non-constant columns, scaled by `_UNIT`). Any `period` consecutive
    returns are a full cycle, so every window of that length has a sample
    covariance of `_UNIT^2 * period / (period - 1) * I`: equal variances,
    no correlation, the same at every date."""
    from scipy.linalg import hadamard

    pattern = hadamard(period)[:, 1 : len(names) + 1] * _UNIT
    returns = np.zeros((n_bars, len(names)))
    for t in range(1, n_bars):
        returns[t] = pattern[t % period]
    dates = pd.date_range("2021-01-04", periods=n_bars, freq="B")
    return pd.DataFrame(
        np.cumprod(1.0 + returns, axis=0), index=dates, columns=list(names)
    )


def _spec(**optimizer):
    """Daily rebalancing, a fixed IC and no binding exposure limits unless
    a test asks for them."""
    return PredictionTransformSpec(
        method="mean_variance",
        rebalance_frequency="daily",
        gross_exposure=10.0,
        max_position_weight=1.0,
        optimizer={"score": "zscore", "covariance": "sample", **optimizer},
    )


def _standardized(row) -> np.ndarray:
    row = np.asarray(row, dtype=float)
    return (row - row.mean()) / row.std()


class TestClosedForms:
    def test_identity_risk_without_costs_is_proportional_to_z(self):
        """Maximizing alpha'w - (gamma/2) s^2 w'w with alpha = IC s z gives
        w = IC z / (gamma s), and z summing to zero satisfies the
        dollar-neutral budget with a zero multiplier. One bar holding and
        one bar of IC horizon, so alpha is IC * s * z exactly."""
        names = [f"N{j}" for j in range(8)]
        closes = _hadamard_closes(names, 40)
        rng = np.random.default_rng(21)
        predictions = pd.DataFrame(
            rng.normal(size=(40, 8)), index=closes.index, columns=names
        )
        date = closes.index[20]
        spec = _spec(
            ic=0.05,
            risk_lookback=16,
            sizing="risk_aversion",
            risk_aversion=100.0,
            cost_aware=False,
        )
        weights, _, per_date = mean_variance_weights(
            predictions,
            [date],
            closes,
            spec,
            linear_cost=0.0,
            periods_per_year=252,
            fill_price="close",
        )
        s = _UNIT * math.sqrt(16 / 15)
        expected = 0.05 * _standardized(predictions.loc[date]) / (100.0 * s)
        np.testing.assert_allclose(weights.iloc[0].to_numpy(), expected, rtol=1e-9)
        assert per_date.iloc[0]["status"] == "solved"

    def test_the_cost_band_shrinks_each_trade_by_kappa(self):
        """From cash with a linear cost kappa and identity risk: a name
        whose |alpha| is inside kappa stays at zero, and every other moves
        to (alpha - kappa * sign(alpha)) / (gamma s^2). The z-scores are
        symmetric, so the budget's multiplier is zero."""
        names = [f"N{j}" for j in range(6)]
        closes = _hadamard_closes(names, 30)
        row = [-1.5, -1.2, -0.05, 0.05, 1.2, 1.5]
        predictions = pd.DataFrame([row] * 30, index=closes.index, columns=names)
        date = closes.index[20]
        kappa, gamma, ic = 0.0004, 100.0, 0.1
        spec = _spec(
            ic=ic, risk_lookback=16, sizing="risk_aversion", risk_aversion=gamma
        )
        weights, _, _ = mean_variance_weights(
            predictions,
            [date],
            closes,
            spec,
            linear_cost=kappa,
            periods_per_year=252,
            fill_price="close",
        )
        s = _UNIT * math.sqrt(16 / 15)
        alpha = ic * s * _standardized(row)
        traded = np.abs(alpha) > kappa
        expected = np.where(
            traded, (alpha - kappa * np.sign(alpha)) / (gamma * s * s), 0.0
        )
        np.testing.assert_allclose(
            weights.iloc[0].to_numpy(), expected, rtol=1e-9, atol=1e-15
        )
        assert (weights.iloc[0].to_numpy()[~traded] == 0.0).all()

    def test_the_two_asset_kelly_book(self):
        """sizing='kelly' is risk aversion 1/f, so unconstrained the book
        is f * S^-1 alpha. The budget is set to that book's own net, where
        its multiplier is zero, and every other limit is slack, so the
        answer is the closed form. S and the volatilities are computed
        here from the closes with numpy."""
        rng = np.random.default_rng(31)
        chol = np.linalg.cholesky([[1.0, 0.6], [0.6, 1.0]])
        returns = rng.normal(size=(80, 2)) @ chol.T * np.array([0.012, 0.008])
        returns[0] = 0.0
        closes = pd.DataFrame(
            100.0 * np.cumprod(1.0 + returns, axis=0),
            index=pd.date_range("2021-01-04", periods=80, freq="B"),
            columns=["A", "B"],
        )
        predictions = pd.DataFrame(
            [[0.3, -0.2]] * 80, index=closes.index, columns=["A", "B"]
        )
        date = closes.index[70]
        window = closes.pct_change().iloc[11:71].to_numpy()
        covariance = np.cov(window, rowvar=False)
        alpha = 0.002 * np.sqrt(np.diag(covariance)) * np.array([1.0, -1.0])
        fraction = 0.5
        kelly = fraction * np.linalg.solve(covariance, alpha)
        spec = PredictionTransformSpec(
            method="mean_variance",
            rebalance_frequency="daily",
            gross_exposure=float(2.0 * np.abs(kelly).sum()),
            net_exposure=float(kelly.sum()),
            max_position_weight=1.0,
            optimizer={
                "ic": 0.002,
                "score": "zscore",
                "covariance": "sample",
                "risk_lookback": 60,
                "sizing": "kelly",
                "kelly_fraction": fraction,
                "cost_aware": False,
            },
        )
        weights, _, _ = mean_variance_weights(
            predictions,
            [date],
            closes,
            spec,
            linear_cost=0.0,
            periods_per_year=252,
            fill_price="close",
        )
        np.testing.assert_allclose(weights.iloc[0].to_numpy(), kelly, rtol=1e-8)


def _held(previous, filled_at, marks, cost):
    """The previous target as weights of the equity it has become, from
    the account's own arithmetic: positions grown by mark / fill, cash the
    remainder less the cost the fill paid."""
    growth = np.asarray(marks, dtype=float) / np.asarray(filled_at, dtype=float)
    equity = 1.0 - cost + float(np.sum(previous * (growth - 1.0)))
    return previous * growth / equity


def _ohlcv(closes: pd.DataFrame, name: str) -> pd.DataFrame:
    """Bars whose open is the previous close, so a next-open fill is at
    the price the decision before it was marked to."""
    close = closes[name]
    opening = close.shift(1).fillna(close.iloc[0])
    return pd.DataFrame(
        {
            "Open": opening,
            "High": np.maximum(opening, close) * 1.001,
            "Low": np.minimum(opening, close) * 0.999,
            "Close": close,
            "Volume": 1e6,
        }
    )


class TestTheNoTradeRegion:
    KAPPA, GAMMA, IC = 0.0004, 100.0, 0.1

    def _run(self, kappa):
        names = [f"N{j}" for j in range(6)]
        closes = _hadamard_closes(names, 30)
        first = [-1.5, -1.2, -0.05, 0.05, 1.2, 1.5]
        moved = [-1.4, -1.1, -0.3, 0.3, 1.1, 1.4]
        predictions = pd.DataFrame([first] * 30, index=closes.index, columns=names)
        predictions.iloc[21] = moved
        spec = _spec(
            ic=self.IC,
            risk_lookback=16,
            sizing="risk_aversion",
            risk_aversion=self.GAMMA,
            cost_aware=kappa > 0,
        )
        dates = closes.index[[20, 21]]
        weights, _, per_date = mean_variance_weights(
            predictions,
            dates,
            closes,
            spec,
            linear_cost=kappa,
            periods_per_year=252,
            fill_price="close",
        )
        return closes, predictions, weights, per_date

    def _held_on_the_second_date(self, closes, weights, per_date, kappa):
        first = weights.iloc[0].to_numpy()
        return _held(
            first,
            closes.iloc[20].to_numpy(),
            closes.iloc[21].to_numpy(),
            kappa * per_date.iloc[0]["turnover"],
        )

    def test_an_alpha_moved_inside_the_band_keeps_the_held_book(self):
        """The second date's alphas move toward zero on every traded name
        and stay inside kappa on the others. The book held then is the
        first date's target drifted a bar at close fills, less the cost of
        building it. Holding it is optimal when some multiplier puts every
        name's gradient alpha - gamma S w0 within kappa, so the gradient's
        spread is at most 2 kappa: checked here from numpy's own
        covariance, and the optimizer then returns the held book, with no
        turnover."""
        kappa = self.KAPPA
        closes, predictions, weights, per_date = self._run(kappa)
        held = self._held_on_the_second_date(closes, weights, per_date, kappa)
        window = closes.pct_change().iloc[6:22].to_numpy()
        covariance = np.cov(window, rowvar=False)
        alpha = (
            self.IC * np.sqrt(np.diag(covariance)) * _standardized(predictions.iloc[21])
        )
        gradient = alpha - self.GAMMA * covariance @ held
        assert gradient.max() - gradient.min() < 2 * kappa - 1e-6
        assert (weights.iloc[0].to_numpy() != 0.0).sum() == 4
        assert not np.allclose(held, weights.iloc[0].to_numpy(), rtol=1e-6)
        np.testing.assert_allclose(weights.iloc[1].to_numpy(), held, rtol=1e-14)
        assert per_date.iloc[1]["turnover"] == 0.0
        assert per_date.iloc[1]["share_held"] == 1.0

    def test_the_simulator_then_trades_nothing(self):
        """The same weights through run_portfolio_simulation at close
        fills and the same rate: the second rebalance trades no name. The
        engine skips a trade under 1e-9 shares, and the held book matches
        the account's to floating-point rounding, far inside that: no
        turnover on the date, and the equity curve is the one with no
        second rebalance at all. Null case: the previous target as the
        second row, the convention this replaced, trades the drift."""
        from standard_quant_tools.backtest.portfolio_engine import (
            run_portfolio_simulation,
        )

        kappa = self.KAPPA
        closes, _, weights, _ = self._run(kappa)
        prices = {name: _ohlcv(closes, name) for name in closes.columns}

        def simulate(frame):
            return run_portfolio_simulation(
                prices,
                frame,
                initial_capital=1_000_000.0,
                commission_pct=kappa,
                slippage_pct=0.0,
                fill_price="close",
            )

        both = simulate(weights)
        once = simulate(weights.iloc[:1])
        assert both["rebalance_log"]["turnover_pct"].tolist()[1] == 0.0
        np.testing.assert_allclose(
            both["equity_curve"].to_numpy(), once["equity_curve"].to_numpy(), rtol=1e-12
        )
        stale = weights.copy()
        stale.iloc[1] = weights.iloc[0]
        assert simulate(stale)["rebalance_log"]["turnover_pct"].tolist()[1] > 1e-4

    def test_without_costs_the_same_move_trades(self):
        """Null case: the band is what held the book."""
        _, _, _, per_date = self._run(0.0)
        assert per_date.iloc[1]["turnover"] > 1e-4


class TestTheSimulatorTradesWhatTheOptimizerChose:
    def test_each_rebalance_trades_the_optimizers_turnover(self):
        """A planted universe at next-open fills whose opens are the
        previous closes, so the held book marked at a decision's close is
        the book at its fill. On every rebalance the simulator's turnover
        (notional over post-cost equity, to its six decimals) is the
        optimizer's own, turnover / (1 - rate * turnover): it trades what
        was chosen and nothing on the names left alone. Measured against
        the target-to-target change instead, it does not match, so the
        drift is real here."""
        from standard_quant_tools.backtest.portfolio_engine import (
            run_portfolio_simulation,
        )

        closes, predictions = _planted_universe(91)
        opens = pd.DataFrame(
            {name: _ohlcv(closes, name)["Open"] for name in closes.columns}
        )
        rate = 0.0015
        weights, _, per_date = mean_variance_weights(
            predictions,
            predictions.index[::5],
            closes,
            _pit_spec(),
            linear_cost=rate,
            periods_per_year=252,
            fill_price="next_open",
            opens=opens,
            target_horizon=5,
        )
        simulated = run_portfolio_simulation(
            {name: _ohlcv(closes, name) for name in closes.columns},
            weights,
            initial_capital=1_000_000.0,
            commission_pct=0.001,
            slippage_pct=0.0005,
            fill_price="next_open",
        )["rebalance_log"]
        chosen = per_date["turnover"].to_numpy()
        expected = chosen / (1.0 - rate * chosen)
        traded = simulated["turnover_pct"].to_numpy()
        assert len(traded) == len(chosen)
        np.testing.assert_allclose(traded, expected, atol=2e-6)
        assert (per_date["share_held"].dropna() > 0).sum() > 10
        target_change = np.abs(weights.diff().fillna(weights)).sum(axis=1).to_numpy()
        assert np.abs(traded - target_change).max() > 1e-3


# ── Point in time ───────────────────────────────────────────────────────


def _planted_universe(seed: int, n_names: int = 12, n_bars: int = 360, ic=0.06):
    """A market factor, two sectors and an AR(1) signal that drives the
    next bar's idiosyncratic drift; the predictions are the signal plus
    equal-size noise, from bar 100 on."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-02", periods=n_bars)
    names = [f"P{j:02d}" for j in range(n_names)]
    beta = rng.normal(1.0, 0.25, n_names)
    sector = np.arange(n_names) % 2
    idio = np.exp(rng.normal(math.log(0.015), 0.3, n_names))
    signal = np.zeros((n_bars, n_names))
    for t in range(1, n_bars):
        signal[t] = 0.8 * signal[t - 1] + 0.6 * rng.normal(size=n_names)
    drift = np.zeros_like(signal)
    drift[1:] = 1.15 * ic * signal[:-1]
    returns = (
        rng.normal(0.0003, 0.01, (n_bars, 1)) * beta
        + rng.normal(0.0, 0.006, (n_bars, 2))[:, sector]
        + idio * (drift + rng.normal(size=(n_bars, n_names)))
    )
    returns[0] = 0.0
    closes = pd.DataFrame(
        100.0 * np.cumprod(1.0 + returns, axis=0), index=dates, columns=names
    )
    predictions = pd.DataFrame(
        signal + rng.normal(size=signal.shape), index=dates, columns=names
    ).iloc[100:-6]
    return closes, predictions


def _pit_spec(**optimizer):
    return PredictionTransformSpec(
        method="mean_variance",
        max_position_weight=0.2,
        max_turnover=0.5,
        optimizer={
            "ic_lookback": 60,
            "ic_min_dates": 10,
            "risk_lookback": 60,
            **optimizer,
        },
    )


class TestPointInTime:
    @pytest.mark.parametrize(
        "optimizer",
        [{}, {"sizing": "kelly", "kelly_fraction": 0.5, "ic_shrink": True}],
    )
    def test_the_future_changes_nothing_at_or_before_t(self, optimizer):
        """Closes and opens after t scaled by random factors and
        predictions after t redrawn, at next-open fills: every weight and
        every per-date diagnostic at or before t is the same to the bit,
        and the dates after t do change."""
        from standard_quant_tools.modeling.portfolio_eval import select_rebalance_dates

        closes, predictions = _planted_universe(41)
        opens = pd.DataFrame(
            {name: _ohlcv(closes, name)["Open"] for name in closes.columns}
        )
        schedule = select_rebalance_dates(predictions.index, "weekly")
        spec = _pit_spec(**optimizer)
        kwargs = dict(
            linear_cost=0.0015,
            periods_per_year=252,
            target_horizon=5,
            fill_price="next_open",
        )
        weights, _, per_date = mean_variance_weights(
            predictions, schedule, closes, spec, opens=opens, **kwargs
        )
        cut = schedule[len(schedule) // 2]
        rng = np.random.default_rng(42)
        later = closes.index > cut
        moved_closes = closes.copy()
        moved_closes.loc[later] *= np.exp(rng.normal(0, 0.05, (int(later.sum()), 12)))
        moved_opens = opens.copy()
        moved_opens.loc[later] *= np.exp(rng.normal(0, 0.05, (int(later.sum()), 12)))
        moved_predictions = predictions.copy()
        after = predictions.index > cut
        moved_predictions.loc[after] = rng.normal(size=(int(after.sum()), 12))
        moved, _, moved_per_date = mean_variance_weights(
            moved_predictions, schedule, moved_closes, spec, opens=moved_opens, **kwargs
        )
        pd.testing.assert_frame_equal(
            moved.loc[:cut], weights.loc[:cut], check_exact=True
        )
        pd.testing.assert_frame_equal(
            moved_per_date.loc[:cut], per_date.loc[:cut], check_exact=True
        )
        assert not moved.loc[moved.index > cut].equals(weights.loc[weights.index > cut])
        assert (per_date["status"] == "solved").sum() > 10

    def test_the_fill_after_t_moves_only_the_next_held_book(self):
        """The trade decided at t fills at the open after t, which is in
        t's future: changing that one open changes nothing at or before t,
        and changes the book held at the next rebalance, which is marked
        from that fill."""
        from standard_quant_tools.modeling.portfolio_eval import select_rebalance_dates

        closes, predictions = _planted_universe(47)
        opens = pd.DataFrame(
            {name: _ohlcv(closes, name)["Open"] for name in closes.columns}
        )
        schedule = select_rebalance_dates(predictions.index, "weekly")
        kwargs = dict(
            linear_cost=0.0015,
            periods_per_year=252,
            target_horizon=5,
            fill_price="next_open",
        )
        weights, _, per_date = mean_variance_weights(
            predictions, schedule, closes, _pit_spec(), opens=opens, **kwargs
        )
        cut = schedule[len(schedule) // 2]
        fill = closes.index[closes.index.get_loc(cut) + 1]
        moved_opens = opens.copy()
        moved_opens.loc[fill] *= 1.03
        moved, _, moved_per_date = mean_variance_weights(
            predictions, schedule, closes, _pit_spec(), opens=moved_opens, **kwargs
        )
        pd.testing.assert_frame_equal(
            moved.loc[:cut], weights.loc[:cut], check_exact=True
        )
        pd.testing.assert_frame_equal(
            moved_per_date.loc[:cut], per_date.loc[:cut], check_exact=True
        )
        following = schedule[schedule > cut][0]
        assert (
            moved_per_date.loc[following, "turnover"]
            != per_date.loc[following, "turnover"]
        )

    def test_next_open_without_opens_is_refused(self):
        closes, predictions = _planted_universe(48)
        with pytest.raises(ValidationError, match="opens"):
            mean_variance_weights(
                predictions,
                predictions.index[::5],
                closes,
                _pit_spec(),
                linear_cost=0.0015,
                periods_per_year=252,
                fill_price="next_open",
            )

    def test_each_solve_scales_by_the_trailing_realized_ic(self):
        """The IC a date's alphas carry is the mean of the realized ICs
        whose five-bar windows had closed by that date, recomputed here
        from `realized_rank_ic` by position; and on a universe built for
        an IC near 0.06 the realized one is near it."""
        closes, predictions = _planted_universe(43, n_names=40, n_bars=700)
        schedule = pd.DatetimeIndex(predictions.index[::5])
        spec = _pit_spec(ic_shrink=False, ic_lookback=200)
        _, diagnostics, per_date = mean_variance_weights(
            predictions,
            schedule,
            closes,
            spec,
            linear_cost=0.0015,
            periods_per_year=252,
            fill_price="close",
            target_horizon=5,
        )
        frame = realized_rank_ic(predictions, closes, 5)
        assert diagnostics["mean_variance"]["ic_horizon_source"] == "target_horizon"
        assert 0.03 < frame["ic"].mean() < 0.09
        positions = closes.index.get_indexer(schedule)
        checked = 0
        for date, position in zip(schedule, positions):
            closed = frame[frame["known_at"] <= position].tail(200)
            if len(closed) < 10:
                assert pd.isna(per_date.loc[date, "ic_raw"])
                continue
            assert per_date.loc[date, "ic_raw"] == pytest.approx(closed["ic"].mean())
            checked += 1
        assert checked > 50


class TestEndToEnd:
    def test_an_evaluation_runs_the_optimizer(self, monkeypatch):
        """evaluate_predictions_portfolio with the synthetic provider: the
        weights it simulated are dollar neutral, inside gross and the
        cap, the target horizon reached the IC, the warm-up is reported,
        and the same inputs give the same weights."""
        from standard_quant_tools.data.factory import DataFactory
        from standard_quant_tools.modeling import artifacts as _artifacts
        from standard_quant_tools.modeling.portfolio_eval import (
            evaluate_predictions_portfolio,
        )

        closes, predictions = _planted_universe(61, n_names=12, n_bars=400)

        class _Provider:
            def get_ohlcv(self, symbol, start, end, interval="1d"):
                close = closes[symbol]
                opening = close.shift(1).fillna(close.iloc[0])
                return pd.DataFrame(
                    {
                        "Open": opening,
                        "High": np.maximum(opening, close) * 1.001,
                        "Low": np.minimum(opening, close) * 0.999,
                        "Close": close,
                        "Volume": 1e6,
                    }
                )

        monkeypatch.setattr(DataFactory, "get_provider", lambda *a, **k: _Provider())
        frame = (
            predictions.stack()
            .rename("prediction")
            .rename_axis(index=["date", "entity"])
            .reset_index()
        )

        def evaluate():
            return evaluate_predictions_portfolio(
                frame,
                "regression",
                interval="1d",
                provider_name="mock",
                calendar=None,
                start=str(closes.index[0].date()),
                end=str(closes.index[-1].date()),
                transform=_pit_spec(),
                run_id="mean_variance_e2e",
                target_horizon=5,
            )

        result = evaluate()
        weights = _artifacts.load_artifact(result["target_weights_uri"])
        assert np.abs(weights.sum(axis=1)).max() < 1e-8
        assert weights.abs().sum(axis=1).max() <= 1.0 + 1e-8
        assert weights.abs().max().max() <= 0.2 + 1e-8
        summary = result["transform_diagnostics"]["mean_variance"]
        assert summary["ic_horizon"] == 5
        assert summary["n_dates_solved"] == len(weights)
        assert summary["n_dates_ic_warmup"] > 0
        assert any("had no alpha" in w for w in result["warnings"])
        assert (
            result["provenance"]["transform_spec"]["optimizer"]["risk_lookback"] == 60
        )
        assert np.isfinite(result["metrics"]["sharpe_ratio"])
        again = evaluate()
        assert (
            again["provenance"]["target_weights_hash"]
            == result["provenance"]["target_weights_hash"]
        )


# ── Refusals with data ──────────────────────────────────────────────────


class TestALongOnlyBook:
    @pytest.mark.parametrize(
        "optimizer",
        [{}, {"sizing": "kelly"}, {"sizing": "risk_aversion", "risk_aversion": 5.0}],
    )
    def test_fully_invested_and_long_only_on_every_date(self, optimizer):
        """net == gross: every weight at or above zero, the budget met on
        every date, every solve certified. The gross cap is the budget
        here, so it is not passed a second time."""
        closes, predictions = _planted_universe(71, n_names=15)
        spec = PredictionTransformSpec(
            method="mean_variance",
            gross_exposure=1.0,
            net_exposure=1.0,
            max_position_weight=0.2,
            optimizer={
                "risk_lookback": 60,
                "ic_lookback": 60,
                "ic_min_dates": 10,
                **({"target_volatility": 0.4} if not optimizer else optimizer),
            },
        )
        weights, diagnostics, per_date = mean_variance_weights(
            predictions,
            predictions.index[::5],
            closes,
            spec,
            linear_cost=0.0015,
            periods_per_year=252,
            fill_price="close",
            target_horizon=5,
        )
        solved = per_date["status"] == "solved"
        assert solved.sum() > 20
        assert (weights.to_numpy() >= 0.0).all()
        np.testing.assert_allclose(weights[solved].sum(axis=1), 1.0, atol=1e-10)
        assert diagnostics["mean_variance"]["n_uncertified"] == 0


class TestRefusalsWithData:
    def test_a_volatility_target_below_a_long_only_book(self):
        """Fully invested and long-only, the book carries the market's
        volatility; a cap far under it cannot be met, and the refusal says
        which field to move."""
        closes, predictions = _planted_universe(51)
        spec = PredictionTransformSpec(
            method="mean_variance",
            gross_exposure=1.0,
            net_exposure=1.0,
            max_position_weight=0.2,
            optimizer={
                "target_volatility": 0.005,
                "risk_lookback": 60,
                "ic_lookback": 60,
                "ic_min_dates": 10,
            },
        )
        with pytest.raises(ValidationError, match="target_volatility"):
            mean_variance_weights(
                predictions,
                predictions.index[::5],
                closes,
                spec,
                linear_cost=0.0015,
                periods_per_year=252,
                fill_price="close",
            )

    def test_a_rebalance_date_off_the_price_calendar(self):
        closes, predictions = _planted_universe(52)
        with pytest.raises(ValidationError, match="calendar"):
            mean_variance_weights(
                predictions,
                [pd.Timestamp("2030-01-01")],
                closes,
                _pit_spec(),
                linear_cost=0.0015,
                periods_per_year=252,
                fill_price="close",
            )

    def test_a_negative_cost(self):
        closes, predictions = _planted_universe(53)
        with pytest.raises(ValidationError, match="linear_cost"):
            mean_variance_weights(
                predictions,
                predictions.index[::5],
                closes,
                _pit_spec(),
                linear_cost=-0.001,
                periods_per_year=252,
                fill_price="close",
            )


class TestAnAnswerThatIsNotAnOptimum:
    """The optimizer reports whether its answer certified. One that stopped
    short (a singular system, a cycle) is the last feasible point it held,
    not an optimum, so the run is refused naming the date; one that missed
    the certificate's tolerance by rounding is traded and counted."""

    def _run(self, monkeypatch, residual):
        import standard_quant_tools.portfolio.rebalance_optimizer as module

        real = module.optimize_rebalance

        def stopped(*args, **kwargs):
            out = real(*args, **kwargs)
            certificate = dict(out["certificate"], dual_feasibility=residual)
            return {
                **out,
                "certified": False,
                "certificate": certificate,
                "warnings": ["the method stopped"],
            }

        monkeypatch.setattr(module, "optimize_rebalance", stopped)
        closes, predictions = _planted_universe(81)
        return mean_variance_weights(
            predictions,
            predictions.index[::5],
            closes,
            _pit_spec(),
            linear_cost=0.0015,
            periods_per_year=252,
            fill_price="close",
        )

    def test_a_point_the_method_stopped_at_is_refused_by_date(self, monkeypatch):
        with pytest.raises(ValidationError, match="returned no optimum on 20"):
            self._run(monkeypatch, 0.3)

    def test_a_rounding_miss_is_traded_and_counted(self, monkeypatch):
        _, diagnostics, per_date = self._run(monkeypatch, 1e-10)
        solved = int((per_date["status"] == "solved").sum())
        assert diagnostics["mean_variance"]["n_uncertified"] == solved > 0
        assert any("by rounding" in w for w in diagnostics["warnings"])
