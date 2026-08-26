"""
The `portfolio` runtime: turn a view into a position, and price it.

Optimal weights, risk attribution, position sizing, stress replay, capacity
limits, and what trading actually costs -- estimated from bars when that is
all there is, and measured from ticks when a provider serves them. The
OHLCV liquidity proxies sit beside the tick measurements deliberately: they
are the same question at two data fidelities, and `check_spread_proxy`
exists to compare them.
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd

from standard_quant_tools.agent.models import (
    CapacityReportInput,
    CapacityReportResult,
    ChannelResult,
    EstimateCovarianceInput,
    EstimateCovarianceResult,
    EstimateTradeCostInput,
    EstimateTradeCostResult,
    LiquidityAnalysisInput,
    LiquidityAnalysisResult,
    LiquidityEventsInput,
    LiquidityEventsResult,
    MicrostructureInput,
    MicrostructureResult,
    PlanRebalanceInput,
    PlanRebalanceResult,
    PortfolioOptimizationInput,
    PortfolioOptimizationResult,
    PositionSizerInput,
    PositionSizerResult,
    RebalanceStep,
    RiskAttributionInput,
    RiskAttributionResult,
    SizeBucket,
    SpreadProxyCheckInput,
    SpreadProxyCheckResult,
    StressTestInput,
    StressTestResult,
    TimeBucket,
    TradeCostLeg,
    TradeProfileInput,
    TradeProfileResult,
    UnavailableChannel,
    UnreachableName,
)
from standard_quant_tools.analysis.microstructure import (
    intraday_volume_profile as _intraday_volume_profile,
)
from standard_quant_tools.analysis.microstructure import (
    microstructure_summary as _microstructure_summary,
)
from standard_quant_tools.analysis.microstructure import (
    trade_size_profile as _trade_size_profile,
)
from standard_quant_tools.backtest.constraints import (
    capacity_report as _capacity_report,
)
from standard_quant_tools.backtest.constraints import (
    days_to_liquidate as _days_to_liquidate,
)
from standard_quant_tools.backtest.constraints import (
    sector_exposure as _sector_exposure,
)
from standard_quant_tools.backtest.costs import (
    directional_commission,
    fixed_bps_spread,
    impact_cost,
    maker_taker_cost,
    margin_interest,
    pct_of_range_spread,
    per_share_commission,
    percentage_commission,
    short_borrow_cost,
)
from standard_quant_tools.backtest.liquidity import (
    amihud_illiquidity,
    corwin_schultz_spread,
)
from standard_quant_tools.backtest.stress_test import (
    replay_stress_scenario,
    scenario_dates,
)
from standard_quant_tools.data.base import DataProvider
from standard_quant_tools.data.factory import DataFactory
from standard_quant_tools.error import ValidationError
from standard_quant_tools.indicators.volatility import atr
from standard_quant_tools.metrics.return_metrics import annualized_volatility, cagr
from standard_quant_tools.metrics.risk_metrics import (
    cvar,
    information_ratio,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    var_historical,
)
from standard_quant_tools.portfolio.optimize import (
    _check_covariance_estimable,
    _small_sample_warnings,
    black_litterman,
    build_bl_views,
    mean_variance_optimize,
    risk_parity_weights,
)
from standard_quant_tools.portfolio.portfolio import (
    fetch_returns_sync,
)
from standard_quant_tools.validation import last_finite


def _rounded(value: Any, digits: int = 4) -> Optional[float]:
    """Round an optional float, keeping None as None.

    Non-finite values become None rather than NaN: these summaries are
    JSON, and a NaN token is rejected by strict parsers. None reads as
    "not measurable here", which is what it means.
    """
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return round(number, digits)


def run_portfolio_optimization(
    input_data: PortfolioOptimizationInput,
) -> PortfolioOptimizationResult:
    """
    Produce portfolio weights via Markowitz mean-variance (max_sharpe,
    min_volatility, target_return, target_volatility), risk parity, or
    Black-Litterman — unlike get_portfolio_analysis, which only scores
    weights the caller already picked.
    """
    logger.debug(
        "[portfolio_optimization] tickers=%s  method=%s  %s → %s",
        input_data.tickers,
        input_data.method,
        input_data.start_date,
        input_data.end_date,
    )
    returns_df = fetch_returns_sync(
        input_data.tickers, input_data.start_date, input_data.end_date
    )
    warnings: List[str] = []

    # Label weights from the columns the covariance was actually built from,
    # not from the requested list. PortfolioOptimizationInput now rejects
    # duplicates, which was the way the two could differ, but the invariant
    # worth holding is that a weight is named by the series it was computed
    # from -- so these stay coupled by construction rather than by a
    # validator elsewhere continuing to hold.
    solved_tickers = list(returns_df.columns)
    warnings.extend(_small_sample_warnings(returns_df.shape[0], len(solved_tickers)))

    if input_data.method == "risk_parity":
        # risk_parity and black_litterman bypass mean_variance_optimize, so
        # they need the same gate it applies -- otherwise a covariance that
        # is singular by construction reaches them too.
        _check_covariance_estimable(
            returns_df.shape[0], len(solved_tickers), returns_df.cov().to_numpy()
        )
        mu, cov = _mean_cov_for_tools(returns_df, input_data.periods_per_year)
        budget = (
            np.array([input_data.risk_budget[t] for t in input_data.tickers])
            if input_data.risk_budget is not None
            else None
        )
        rp = risk_parity_weights(cov, risk_budget=budget)
        if not rp["converged"]:
            warnings.append(
                "risk parity iteration did not converge within max_iterations "
                "— weights are the best achieved, not exact equal risk contribution"
            )
        w = rp["weights"]
        exp_ret = float(w @ mu)
        exp_vol = float(np.sqrt(w @ cov @ w))
        sharpe = (
            (exp_ret - input_data.risk_free_rate) / exp_vol if exp_vol > 1e-12 else 0.0
        )
        return PortfolioOptimizationResult(
            tickers=input_data.tickers,
            method=input_data.method,
            weights={t: round(float(wi), 6) for t, wi in zip(solved_tickers, w)},
            expected_return=round(exp_ret, 6),
            expected_volatility=round(exp_vol, 6),
            sharpe_ratio=round(sharpe, 4),
            converged=rp["converged"],
            risk_contributions={
                t: round(float(c), 6)
                for t, c in zip(solved_tickers, rp["risk_contributions"])
            },
            warnings=warnings,
        )

    if input_data.method == "black_litterman":
        _check_covariance_estimable(
            returns_df.shape[0], len(solved_tickers), returns_df.cov().to_numpy()
        )
        mu, cov = _mean_cov_for_tools(returns_df, input_data.periods_per_year)
        n = len(input_data.tickers)
        mkt_w = (
            np.array([input_data.market_weights[t] for t in input_data.tickers])
            if input_data.market_weights is not None
            else np.full(n, 1.0 / n)
        )
        assert input_data.views  # validated non-empty by the Pydantic model
        P, Q, omega = build_bl_views(
            input_data.tickers,
            [v.model_dump() for v in input_data.views],
            cov,
            tau=input_data.tau,
        )
        bl = black_litterman(
            cov,
            mkt_w,
            P,
            Q,
            risk_aversion=input_data.risk_aversion,
            tau=input_data.tau,
            omega=omega,
        )
        w = bl["implied_weights"]
        exp_ret = float(w @ bl["posterior_returns"])
        exp_vol = float(np.sqrt(w @ bl["posterior_cov"] @ w))
        sharpe = (
            (exp_ret - input_data.risk_free_rate) / exp_vol if exp_vol > 1e-12 else 0.0
        )
        return PortfolioOptimizationResult(
            tickers=input_data.tickers,
            method=input_data.method,
            weights={t: round(float(wi), 6) for t, wi in zip(solved_tickers, w)},
            expected_return=round(exp_ret, 6),
            expected_volatility=round(exp_vol, 6),
            sharpe_ratio=round(sharpe, 4),
            converged=True,
            warnings=warnings,
        )

    result = mean_variance_optimize(
        returns_df,
        objective=input_data.method,
        risk_free_rate=input_data.risk_free_rate,
        target_return=input_data.target_return,
        target_volatility=input_data.target_volatility,
        allow_short=input_data.allow_short,
        max_weight=input_data.max_weight,
        periods_per_year=input_data.periods_per_year,
    )
    # The optimizer's own caveats (currently the small-sample covariance
    # warning) travel with the result rather than being dropped at this
    # boundary -- a 22x volatility understatement from too short a window is
    # exactly what an agent consuming this needs told.
    warnings.extend(result.get("warnings", []))
    if not result["converged"]:
        warnings.append(
            "optimizer did not converge — constraints may be infeasible; "
            "treat weights as approximate"
        )
    return PortfolioOptimizationResult(
        tickers=result["tickers"],
        method=input_data.method,
        weights={t: round(float(wi), 6) for t, wi in result["weights"].items()},
        expected_return=round(result["expected_return"], 6),
        expected_volatility=round(result["expected_volatility"], 6),
        sharpe_ratio=round(result["sharpe_ratio"], 4),
        converged=result["converged"],
        warnings=warnings,
    )


def _mean_cov_for_tools(
    returns_df: pd.DataFrame, periods_per_year: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Same annualized (mean, cov) computation portfolio.optimize's private
    _mean_cov does — reimplemented here rather than imported since it's a
    two-line pandas call and importing a leading-underscore helper across
    module boundaries would couple this file to that module's internals."""
    mu = returns_df.mean().to_numpy(dtype=float) * periods_per_year
    cov = returns_df.cov().to_numpy(dtype=float) * periods_per_year
    return mu, cov


def get_portfolio_risk_attribution(
    input_data: RiskAttributionInput,
) -> RiskAttributionResult:
    """
    Deep portfolio risk decomposition: portfolio-level metrics, per-asset
    marginal risk contributions (fractional, summing to 1), PCA decomposition
    of the asset universe, and an optional multi-factor regression on the
    aggregate portfolio returns.
    """
    from standard_quant_tools.analysis.multi_factor import (
        multi_factor_regression as _mfr,
    )
    from standard_quant_tools.analysis.pca import pca_returns as _pca

    logger.debug(
        "[risk_attribution] tickers=%s  weights=%s  vs %s  %s → %s",
        input_data.tickers,
        [round(w, 4) for w in input_data.weights],
        input_data.benchmark,
        input_data.start_date,
        input_data.end_date,
    )

    provider = DataFactory.get_provider()
    weights = np.array(input_data.weights, dtype=float)

    returns_df = pd.DataFrame(
        {
            t: provider.get_ohlcv(t, input_data.start_date, input_data.end_date)[
                "Close"
            ].pct_change()
            for t in input_data.tickers
        }
    ).dropna()

    bench_ret = (
        provider.get_ohlcv(
            input_data.benchmark, input_data.start_date, input_data.end_date
        )["Close"]
        .pct_change()
        .dropna()
    )

    port_arr: np.ndarray = returns_df.values @ weights
    port_ret = pd.Series(port_arr, index=returns_df.index)
    port_equity = (1 + port_ret).cumprod()

    common = port_ret.index.intersection(bench_ret.index)
    bench_aligned = bench_ret.loc[common]
    port_aligned = port_ret.loc[common]

    # ── Portfolio-level metrics ────────────────────────────────────
    ann_ret = float(cagr(port_equity))
    ann_vol = float(annualized_volatility(port_ret))
    sr = float(sharpe_ratio(port_ret, input_data.risk_free_rate))
    sort_r = float(sortino_ratio(port_ret, input_data.risk_free_rate))
    mdd = float(max_drawdown(port_equity))
    v95 = float(var_historical(port_ret, 0.95))
    cv95 = float(cvar(port_ret, 0.95))
    ir = float(information_ratio(port_aligned, bench_aligned))

    # ── Marginal Risk Contribution (fraction of portfolio variance) ─
    cov_ann = returns_df.cov().values * 252
    port_var = float(weights @ cov_ann @ weights)
    mcr = (
        (cov_ann @ weights) * weights / port_var
        if port_var > 0
        else np.zeros(len(weights))
    )
    asset_risk_contribs = {
        t: round(float(mcr[i]), 6) for i, t in enumerate(input_data.tickers)
    }

    # ── PCA decomposition ─────────────────────────────────────────
    n_comp = min(input_data.n_components, len(input_data.tickers))
    pca_res = _pca(returns_df, n_components=n_comp)
    evr = {
        k: round(float(v), 4) for k, v in pca_res["explained_variance_ratio"].items()
    }
    loadings_mat = pca_res["loadings"].values  # (n_assets, n_comp)
    port_exposures = weights @ loadings_mat  # (n_comp,)
    pc_names = list(pca_res["explained_variance_ratio"].index)
    port_pc_exposures = {
        pc_names[i]: round(float(port_exposures[i]), 4) for i in range(n_comp)
    }

    # ── Optional factor regression on portfolio returns ───────────
    factor_loadings: Optional[Dict[str, float]] = None
    factor_r2: Optional[float] = None
    factor_alpha: Optional[float] = None

    if input_data.factor_tickers:
        names = input_data.factor_names or input_data.factor_tickers
        factor_df = pd.DataFrame(
            {
                name: provider.get_ohlcv(
                    tick, input_data.start_date, input_data.end_date
                )["Close"].pct_change()
                for name, tick in zip(names, input_data.factor_tickers)
            }
        ).dropna()

        mfr = _mfr(port_ret, factor_df)
        factor_loadings = {k: round(float(v), 4) for k, v in mfr["loadings"].items()}
        factor_r2 = round(float(mfr["r_squared"]), 4)
        factor_alpha = round(float(mfr["alpha"]), 6)

    return RiskAttributionResult(
        tickers=input_data.tickers,
        weights=list(input_data.weights),
        annualized_return=round(ann_ret, 4),
        annualized_volatility=round(ann_vol, 4),
        sharpe_ratio=round(sr, 4),
        sortino_ratio=round(sort_r, 4),
        max_drawdown=round(mdd, 6),
        var_95=round(v95, 6),
        cvar_95=round(cv95, 6),
        information_ratio=round(ir, 4),
        asset_risk_contributions=asset_risk_contribs,
        pca_variance_explained=evr,
        portfolio_pc_exposures=port_pc_exposures,
        factor_loadings=factor_loadings,
        factor_r_squared=factor_r2,
        factor_alpha=factor_alpha,
    )


def run_stress_test(input_data: StressTestInput) -> StressTestResult:
    """
    Replay a portfolio's current weights against a named historical crash
    window (or a custom date range) using each ticker's own real historical
    returns for that window. A ticker with no data that far back (e.g. it
    hadn't listed yet) is isolated into tickers_missing_data rather than
    failing the whole call — the replay proceeds on whatever subset of
    tickers actually has data.
    """
    if input_data.scenario == "custom":
        # StressTestInput._check_custom_dates already guarantees both are
        # set when scenario == "custom" (raises otherwise) — asserted here
        # too so this stays a str, not Optional[str], for every caller below.
        assert input_data.custom_start_date is not None
        assert input_data.custom_end_date is not None
        scenario_start = input_data.custom_start_date
        scenario_end = input_data.custom_end_date
    else:
        scenario_start, scenario_end = scenario_dates(input_data.scenario)

    logger.debug(
        "[stress_test] tickers=%s  scenario=%s  %s → %s",
        input_data.tickers,
        input_data.scenario,
        scenario_start,
        scenario_end,
    )

    provider = DataFactory.get_provider()
    returns_by_ticker: Dict[str, pd.Series] = {}
    tickers_missing_data: List[str] = []
    for t in input_data.tickers:
        try:
            close = provider.get_ohlcv(t, scenario_start, scenario_end)["Close"]
            returns = close.pct_change().dropna()
            if returns.empty:
                raise ValueError("no trading days with data in this window")
            returns_by_ticker[t] = returns
        except Exception as exc:
            logger.debug(
                "[stress_test] %s has no data for %s → %s (%s)",
                t,
                scenario_start,
                scenario_end,
                exc,
            )
            tickers_missing_data.append(t)

    if not returns_by_ticker:
        raise ValidationError(
            f"None of {input_data.tickers} have any data for {scenario_start} → "
            f"{scenario_end} — cannot replay this scenario."
        )

    tickers_used = list(returns_by_ticker.keys())
    returns_df = pd.DataFrame(returns_by_ticker).dropna()

    if input_data.weights is None:
        weights = np.full(len(input_data.tickers), 1.0 / len(input_data.tickers))
    else:
        weights = np.array(input_data.weights, dtype=float)
    # Re-normalize weights over just the tickers that actually have data,
    # preserving their relative proportions, rather than silently treating
    # a missing ticker's weight as if it had earned a 0% return.
    weight_by_ticker = dict(zip(input_data.tickers, weights))
    used_weights = np.array([weight_by_ticker[t] for t in tickers_used])
    used_weights_sum = used_weights.sum()
    if used_weights_sum <= 0:
        raise ValidationError(
            "the tickers with available data for this scenario carry zero "
            "total weight — cannot replay."
        )
    used_weights = used_weights / used_weights_sum

    result = replay_stress_scenario(returns_df, used_weights)

    return StressTestResult(
        scenario=input_data.scenario,
        scenario_start_date=scenario_start,
        scenario_end_date=scenario_end,
        tickers_used=tickers_used,
        tickers_missing_data=tickers_missing_data,
        portfolio_return_pct=round(result["portfolio_return_pct"], 6),
        max_drawdown_pct=round(result["max_drawdown_pct"], 6),
        worst_day_return_pct=round(result["worst_day_return_pct"], 6),
        worst_day_date=result["worst_day_date"],
        best_day_return_pct=round(result["best_day_return_pct"], 6),
        best_day_date=result["best_day_date"],
        n_trading_days=result["n_trading_days"],
    )


def get_position_size(input_data: PositionSizerInput) -> PositionSizerResult:
    """
    Compute risk-adjusted position size using ATR-based stop-loss sizing
    and optionally Kelly criterion when strategy statistics are provided.

    Fixed-risk sizing: shares = (account × risk_pct) / (atr_multiplier × ATR)
    Kelly sizing:      f = (b×p − q) / b  where b = avg_win/avg_loss,
                       p = win_rate, q = 1−win_rate. Half-Kelly is recommended.
    """
    logger.debug(
        "[position_size] %s  equity=%.0f  risk_pct=%.2f%%  atr_period=%d  atr_mult=%.1f",
        input_data.symbol,
        input_data.account_equity,
        input_data.risk_per_trade_pct * 100,
        input_data.atr_period,
        input_data.atr_multiplier,
    )
    provider = DataFactory.get_provider()
    df = provider.get_ohlcv(
        input_data.symbol, input_data.start_date, input_data.end_date
    )

    last_close = float(df["Close"].iloc[-1])
    atr_series = atr(
        df["High"], df["Low"], df["Close"], period=input_data.atr_period
    ).dropna()
    last_atr = last_finite(atr_series, "atr_series")

    stop_distance = last_atr * input_data.atr_multiplier
    dollar_risk = input_data.account_equity * input_data.risk_per_trade_pct

    shares_fr = max(int(dollar_risk / stop_distance), 0) if stop_distance > 0 else 0
    pos_val_fr = shares_fr * last_close
    port_pct_fr = (
        pos_val_fr / input_data.account_equity if input_data.account_equity > 0 else 0.0
    )

    kelly_fraction: Optional[float] = None
    shares_hk: Optional[int] = None
    pos_val_hk: Optional[float] = None
    port_pct_hk: Optional[float] = None

    has_kelly_inputs = (
        input_data.win_rate is not None
        and input_data.avg_win_pct is not None
        and input_data.avg_loss_pct is not None
    )

    if has_kelly_inputs:
        assert input_data.win_rate is not None
        assert input_data.avg_win_pct is not None
        assert input_data.avg_loss_pct is not None
        wr, aw, al = (
            input_data.win_rate,
            input_data.avg_win_pct,
            input_data.avg_loss_pct,
        )
        b = aw / al if al > 0 else 0.0
        raw_kelly = (b * wr - (1.0 - wr)) / b if b > 0 else 0.0
        kelly_fraction = round(max(raw_kelly, 0.0), 4)

        half_kelly_equity = input_data.account_equity * kelly_fraction * 0.5
        shares_hk = max(int(half_kelly_equity / last_close), 0) if last_close > 0 else 0
        pos_val_hk = shares_hk * last_close
        port_pct_hk = (
            pos_val_hk / input_data.account_equity
            if input_data.account_equity > 0
            else 0.0
        )

    use_kelly = (
        has_kelly_inputs
        and kelly_fraction is not None
        and kelly_fraction > 0
        and (shares_hk or 0) > 0
    )
    recommended_sizing = "half_kelly" if use_kelly else "fixed_risk"
    _rec_shares: int = (shares_hk or 0) if use_kelly else shares_fr  # type: ignore[assignment]
    recommended_value = _rec_shares * last_close
    logger.debug(
        "[position_size] close=%.4f  ATR=%.4f  stop=%.4f  sizing=%s  shares=%d  value=%.2f  kelly=%s",
        last_close,
        last_atr,
        stop_distance,
        recommended_sizing,
        _rec_shares,
        recommended_value,
        str(kelly_fraction),
    )

    return PositionSizerResult(
        symbol=input_data.symbol,
        last_close=round(last_close, 4),
        atr=round(last_atr, 4),
        atr_pct=round(last_atr / last_close * 100, 4) if last_close > 0 else 0.0,
        stop_distance=round(stop_distance, 4),
        shares_fixed_risk=shares_fr,
        position_value_fixed_risk=round(pos_val_fr, 2),
        portfolio_pct_fixed_risk=round(port_pct_fr, 4),
        max_loss_fixed_risk=round(shares_fr * stop_distance, 2),
        kelly_fraction=kelly_fraction,
        shares_half_kelly=shares_hk,
        position_value_half_kelly=(
            round(pos_val_hk, 2) if pos_val_hk is not None else None
        ),
        portfolio_pct_half_kelly=(
            round(port_pct_hk, 4) if port_pct_hk is not None else None
        ),
        recommended_sizing=recommended_sizing,
        recommended_shares=_rec_shares,
        recommended_position_value=round(recommended_value, 2),
    )


def get_capacity_report(input_data: CapacityReportInput) -> CapacityReportResult:
    """
    How much account size a target-weight portfolio can support before
    positions become too large relative to each ticker's own trading
    volume. Reuses backtest/constraints.py's capacity_report/
    days_to_liquidate — no new math, just data fetching and formatting for
    JSON tool-calling.
    """
    logger.debug(
        "[capacity_report] tickers=%d  max_participation=%.3f",
        len(input_data.tickers),
        input_data.max_participation,
    )
    provider = DataFactory.get_provider()

    avg_dollar_volumes: Dict[str, float] = {}
    avg_share_volumes: Dict[str, float] = {}
    last_close: Dict[str, float] = {}
    for t in input_data.tickers:
        df = provider.get_ohlcv(t, input_data.start_date, input_data.end_date)
        if "Volume" not in df.columns:
            raise ValueError(f"OHLCV for {t!r} is missing a 'Volume' column")
        dollar_vol = (
            (df["Close"] * df["Volume"])
            .rolling(input_data.adv_lookback, min_periods=1)
            .mean()
        )
        share_vol = df["Volume"].rolling(input_data.adv_lookback, min_periods=1).mean()
        avg_dollar_volumes[t] = float(dollar_vol.iloc[-1])
        avg_share_volumes[t] = float(share_vol.iloc[-1])
        last_close[t] = float(df["Close"].iloc[-1])

    raw = _capacity_report(
        input_data.tickers,
        avg_dollar_volumes,
        input_data.target_weights,
        input_data.max_participation,
    )

    per_ticker_out: Dict[str, Optional[float]] = {
        t: (None if v == float("inf") else round(v, 2))
        for t, v in raw["per_ticker"].items()
    }
    max_account_size = raw["max_account_size"]
    max_account_size_out: Optional[float] = (
        None if max_account_size == float("inf") else round(max_account_size, 2)
    )

    days_to_liquidate_out: Dict[str, float] = {}
    for t in input_data.tickers:
        weight = abs(input_data.target_weights[t])
        if weight <= 0 or max_account_size_out is None or avg_share_volumes[t] <= 0:
            days_to_liquidate_out[t] = 0.0
            continue
        notional_at_capacity = max_account_size_out * weight
        shares_at_capacity = (
            notional_at_capacity / last_close[t] if last_close[t] > 0 else 0.0
        )
        days_to_liquidate_out[t] = round(
            _days_to_liquidate(
                shares_at_capacity, avg_share_volumes[t], input_data.max_participation
            ),
            2,
        )

    warnings: List[str] = []
    sector_exp: Optional[Dict[str, float]] = None
    if input_data.include_sector_exposure:
        sectors: Dict[str, str] = {}
        for t in input_data.tickers:
            try:
                info = provider.get_ticker_info(t)
                sectors[t] = info.sector
            except Exception as exc:
                sectors[t] = "Unknown"
                warnings.append(f"could not fetch sector for {t!r}: {exc}")
        sector_exp = _sector_exposure(input_data.target_weights, sectors)

    logger.debug(
        "[capacity_report] binding=%s  max_account_size=%s",
        raw["binding_ticker"],
        max_account_size_out,
    )

    return CapacityReportResult(
        tickers=input_data.tickers,
        per_ticker_max_account_size=per_ticker_out,
        binding_ticker=raw["binding_ticker"],
        max_account_size=max_account_size_out,
        days_to_liquidate_at_capacity=days_to_liquidate_out,
        sector_exposure=sector_exp,
        warnings=warnings,
    )


def get_liquidity_metrics(
    input_data: LiquidityAnalysisInput,
) -> LiquidityAnalysisResult:
    """
    Amihud illiquidity ratio and Corwin-Schultz spread estimator per
    ticker — academic proxies for market depth and bid/ask spread derived
    purely from OHLCV, since no real bid/ask data exists in this library.
    """
    logger.debug(
        "[liquidity_metrics] tickers=%s  window=%d",
        input_data.tickers,
        input_data.window,
    )
    provider = DataFactory.get_provider()
    window = input_data.window

    per_ticker: Dict[str, Dict[str, float]] = {}
    for t in input_data.tickers:
        df = provider.get_ohlcv(t, input_data.start_date, input_data.end_date)
        if "Volume" not in df.columns:
            raise ValueError(f"OHLCV for {t!r} is missing a 'Volume' column")

        returns = df["Close"].pct_change()
        dollar_volume = df["Close"] * df["Volume"]
        avg_dollar_volume = dollar_volume.rolling(window, min_periods=1).mean()

        amihud = amihud_illiquidity(returns, dollar_volume, window=window)
        cs_spread = corwin_schultz_spread(df["High"], df["Low"], window=window)

        amihud_valid = amihud.dropna()
        cs_valid = cs_spread.dropna()

        per_ticker[t] = {
            "avg_dollar_volume": round(float(avg_dollar_volume.iloc[-1]), 2),
            "amihud_illiquidity": (
                round(float(amihud_valid.iloc[-1]), 6)
                if not amihud_valid.empty
                else 0.0
            ),
            "corwin_schultz_spread_bps": (
                round(float(cs_valid.iloc[-1]) * 10_000.0, 4)
                if not cs_valid.empty
                else 0.0
            ),
        }

    least_liquid = max(per_ticker, key=lambda t: per_ticker[t]["amihud_illiquidity"])
    most_liquid = min(per_ticker, key=lambda t: per_ticker[t]["amihud_illiquidity"])

    return LiquidityAnalysisResult(
        tickers=input_data.tickers,
        per_ticker=per_ticker,
        least_liquid_ticker=least_liquid,
        most_liquid_ticker=most_liquid,
    )


def _tick_provider(source: str) -> Any:
    """A provider that actually serves ticks, or an error saying who does.

    DataProvider.get_trades raises NotImplementedError with a good message
    already; this fires FIRST so the failure names the tool's own
    precondition and points at describe_data_capabilities, rather than
    surfacing from three frames deep after a fetch has been attempted.
    """
    try:
        provider = DataFactory.get_provider(source)
    except (NotImplementedError, ValueError) as exc:
        raise ValidationError(str(exc)) from exc
    except Exception as exc:
        raise ValidationError(
            f"the {source!r} provider could not be constructed: {exc}. Call "
            "describe_data_capabilities to see what this environment can "
            "actually reach."
        ) from exc

    if type(provider).get_trades is DataProvider.get_trades:
        raise ValidationError(
            f"the {source!r} provider has no tick feed, so this tool cannot "
            "run on it. Bar data is not a substitute — spreads and signed "
            "order flow are not recoverable from an OHLCV row, and nothing "
            "here will invent them. Call describe_data_capabilities to see "
            "which provider serves trades, or use get_liquidity_metrics for "
            "the OHLCV-derived proxies."
        )
    return provider


def _fetch_ticks(provider: Any, symbol: str, start: str, end: str, limit: Any):
    """Trades, and quotes when the provider has them."""
    trades = provider.get_trades(symbol, start, end, limit=limit)
    quotes = None
    if type(provider).get_quotes is not DataProvider.get_quotes:
        try:
            quotes = provider.get_quotes(symbol, start, end, limit=limit)
        except NotImplementedError:
            quotes = None
    return trades, quotes


def get_microstructure_metrics(
    input_data: MicrostructureInput,
) -> MicrostructureResult:
    """
    Measured spreads and signed order flow from tick data.

    This is what `get_liquidity_metrics` estimates. That tool derives a
    spread from OHLCV bars via Corwin-Schultz and says plainly that the
    result is a proxy; this one reads the trades and quotes and measures it.

    Three numbers matter and they are not interchangeable. The QUOTED
    spread is what crossing the book costs at an instant. The EFFECTIVE
    spread is what trades actually paid relative to the prevailing
    midpoint, which differs whenever fills happen inside the quotes or
    sweep through them — that is, most of the time, which is why a backtest
    charging the quoted spread is not charging what trading costs. With a
    realized horizon, the effective spread splits into what the liquidity
    provider KEPT and what the trade MOVED, and those two halves imply
    opposite fixes: impact says trade smaller, realized says trade
    elsewhere.

    Averages are size-weighted as well as count-weighted. The count-weighted
    figure answers "what did a typical print cost" and is dominated by the
    odd-lot tail; the size-weighted one answers "what did a typical share
    cost", which is the question a strategy sizing a position is asking.
    """
    provider = _tick_provider(input_data.source)
    trades, quotes = _fetch_ticks(
        provider,
        input_data.symbol,
        input_data.start,
        input_data.end,
        input_data.limit,
    )
    if trades is None or trades.empty:
        raise ValidationError(
            f"no trades returned for {input_data.symbol} between "
            f"{input_data.start} and {input_data.end}. Check the window is "
            "inside market hours and that the plan tier includes trades."
        )

    horizon = (
        pd.Timedelta(seconds=input_data.realized_horizon_seconds)
        if input_data.realized_horizon_seconds is not None
        else None
    )
    summary = _microstructure_summary(trades, quotes, horizon)

    notes: List[str] = list(summary.get("notes", []))
    if input_data.limit is not None and len(trades) >= input_data.limit:
        notes.append(
            f"Exactly {input_data.limit} trades came back, which is the "
            "limit — this is one page, not the whole window, and the "
            "measures below describe the part that was fetched. Narrow the "
            "window rather than raising the limit."
        )
    dropped = int(len(trades)) - int(summary.get("n_signed", 0))
    if dropped > 0:
        notes.append(
            f"{dropped} trade(s) could not be classified as buyer- or "
            "seller-initiated and are excluded from the spread measures. "
            "They are dropped rather than defaulted: a coin-flip side would "
            "put noise into every average."
        )
    if (
        quotes is not None
        and summary.get("effective_spread_bps_size_weighted") is not None
    ):
        effective = summary["effective_spread_bps_size_weighted"]
        quoted = summary.get("quoted_spread_bps_mean")
        if quoted and effective > quoted * 1.2:
            notes.append(
                "The effective spread exceeds the quoted spread, so trades "
                "were sweeping through the top of book rather than filling "
                "inside it. Top-of-book depth is the binding constraint "
                "here, and no shipped provider exposes the rest of it."
            )

    return MicrostructureResult(
        symbol=input_data.symbol,
        start=input_data.start,
        end=input_data.end,
        n_trades=int(summary["n_trades"]),
        n_quotes=summary.get("n_quotes"),
        n_signed=int(summary.get("n_signed", 0)),
        total_volume=float(summary["total_volume"]),
        vwap=round(float(summary["vwap"]), 6),
        buy_volume_fraction=round(float(summary["buy_volume_fraction"]), 6),
        quoted_spread_bps_mean=_rounded(summary.get("quoted_spread_bps_mean")),
        quoted_spread_bps_median=_rounded(summary.get("quoted_spread_bps_median")),
        quote_imbalance_mean=_rounded(summary.get("quote_imbalance_mean")),
        effective_spread_bps_mean=_rounded(summary.get("effective_spread_bps_mean")),
        effective_spread_bps_size_weighted=_rounded(
            summary.get("effective_spread_bps_size_weighted")
        ),
        realized_spread_bps_size_weighted=_rounded(
            summary.get("realized_spread_bps_size_weighted")
        ),
        price_impact_bps_size_weighted=_rounded(
            summary.get("price_impact_bps_size_weighted")
        ),
        notes=notes,
    )


def get_trade_profile(input_data: TradeProfileInput) -> TradeProfileResult:
    """
    How a symbol's volume is distributed across trade sizes and times of day.

    Both distributions change what a given order actually is. A book where
    most volume arrives in a few large prints behaves nothing like one where
    the same daily total arrives in thousands of small ones, at an identical
    ADV — so an ADV-participation limit means different things in the two.
    And US equity volume is U-shaped, so a fixed-time order is a much larger
    share of the available liquidity at midday than at the close.

    Size buckets are quantiles rather than a fixed share grid, because a
    grid that suits one symbol misreads another by orders of magnitude.
    """
    provider = _tick_provider(input_data.source)
    trades = provider.get_trades(
        input_data.symbol, input_data.start, input_data.end, limit=input_data.limit
    )
    if trades is None or trades.empty:
        raise ValidationError(
            f"no trades returned for {input_data.symbol} between "
            f"{input_data.start} and {input_data.end}."
        )

    sizes = _trade_size_profile(trades, buckets=input_data.size_buckets)
    times = _intraday_volume_profile(trades, freq=input_data.intraday_freq)

    notes: List[str] = []
    if input_data.limit is not None and len(trades) >= input_data.limit:
        notes.append(
            f"Exactly {input_data.limit} trades came back, which is the "
            "limit — this profile describes one page, not the whole window."
        )
    if sizes["largest_bucket_volume_fraction"] > 0.5:
        notes.append(
            "Over half the volume is in the largest size bucket: this name "
            "trades in blocks. An ADV-based capacity limit assumes volume "
            "you can actually join, and most of this is not that."
        )

    return TradeProfileResult(
        symbol=input_data.symbol,
        n_trades=int(sizes["n_trades"]),
        total_volume=float(sizes["total_volume"]),
        median_size=float(sizes["median_size"]),
        size_buckets=[SizeBucket(**bucket) for bucket in sizes["buckets"]],
        largest_bucket_volume_fraction=float(sizes["largest_bucket_volume_fraction"]),
        intraday_buckets=[TimeBucket(**bucket) for bucket in times["buckets"]],
        peak_time=times["peak_time"],
        peak_volume_fraction=float(times["peak_volume_fraction"]),
        notes=notes,
    )


#: Within this ratio of the measured spread, the OHLCV proxy is close
#: enough that a backtest using it is not materially mispriced. Wide
#: because Corwin-Schultz is a bar-derived estimator and was never meant to
#: be exact -- the useful question is which SIDE it errs on.
_PROXY_CLOSE_BAND = 0.25


def check_spread_proxy(input_data: SpreadProxyCheckInput) -> SpreadProxyCheckResult:
    """
    Measure the spread from ticks, compute the OHLCV proxy for the same
    name, and report which way the proxy is wrong.

    `get_liquidity_metrics` exists because tick data is usually absent, and
    its docstring says its numbers are proxies. A proxy cannot check
    itself. This tool does, and the direction of the error is what matters:
    a proxy that OVERSTATES the spread makes a backtest pessimistic, which
    is safe. One that UNDERSTATES it means every backtest charging costs
    from it has been reporting returns that are too good, and by roughly
    the ratio reported here.

    The two windows are separate on purpose. Corwin-Schultz needs a rolling
    window of daily bars, so the bar window normally reaches much further
    back than the tick window it is being checked against.
    """
    provider = _tick_provider(input_data.source)
    trades, quotes = _fetch_ticks(
        provider,
        input_data.symbol,
        input_data.start,
        input_data.end,
        input_data.limit,
    )
    if trades is None or trades.empty:
        raise ValidationError(
            f"no trades returned for {input_data.symbol}; there is nothing "
            "to check the proxy against."
        )
    if quotes is None or quotes.empty:
        raise ValidationError(
            "no quotes returned, so the effective spread cannot be measured "
            "and there is no ground truth to compare the proxy with."
        )

    summary = _microstructure_summary(trades, quotes, None)
    measured = summary.get("effective_spread_bps_size_weighted")
    if measured is None or not math.isfinite(measured):
        raise ValidationError(
            "the measured effective spread is not finite — too few "
            "classifiable trades matched a prevailing quote to compare."
        )

    bars = provider.get_ohlcv(
        input_data.symbol, input_data.bar_start_date, input_data.bar_end_date
    )
    if bars.empty:
        raise ValidationError(
            f"no OHLCV bars for {input_data.symbol} between "
            f"{input_data.bar_start_date} and {input_data.bar_end_date}; the "
            "proxy needs bars to be computed from."
        )
    cs = corwin_schultz_spread(bars["High"], bars["Low"], window=input_data.window)
    cs_valid = cs.dropna()
    if cs_valid.empty:
        raise ValidationError(
            f"Corwin-Schultz produced no value over {len(bars)} bars with "
            f"window={input_data.window}; widen the bar window."
        )
    proxy_bps = float(cs_valid.mean()) * 10_000.0

    returns = bars["Close"].pct_change()
    dollar_volume = bars["Close"] * bars["Volume"]
    amihud = amihud_illiquidity(returns, dollar_volume, window=input_data.window)
    amihud_valid = amihud.dropna()

    ratio = proxy_bps / measured if measured else float("nan")
    if abs(ratio - 1.0) <= _PROXY_CLOSE_BAND:
        verdict = "proxy_close"
    elif ratio > 1.0:
        verdict = "proxy_overstates"
    else:
        verdict = "proxy_understates"

    notes = [
        "The proxy is computed from daily bars over "
        f"{input_data.bar_start_date}..{input_data.bar_end_date}; the "
        f"measurement is from ticks over {input_data.start}..{input_data.end}. "
        "They describe overlapping but not identical periods, which is "
        "inherent to checking a bar estimator against tick data."
    ]
    if verdict == "proxy_understates":
        notes.append(
            f"The proxy charges roughly {ratio:.2f}x the measured spread, so "
            "a backtest priced from it has been charging too little. Returns "
            "computed that way are optimistic, and by more the more it "
            "trades."
        )
    elif verdict == "proxy_overstates":
        notes.append(
            "The proxy charges more than the spread actually measured, so a "
            "backtest priced from it is pessimistic — safe, but it may be "
            "rejecting strategies that would have cleared their costs."
        )

    logger.debug(
        "[check_spread_proxy] %s measured=%.2fbps proxy=%.2fbps verdict=%s",
        input_data.symbol,
        measured,
        proxy_bps,
        verdict,
    )
    return SpreadProxyCheckResult(
        symbol=input_data.symbol,
        measured_effective_spread_bps=round(float(measured), 4),
        measured_quoted_spread_bps=round(
            float(summary.get("quoted_spread_bps_mean", float("nan"))), 4
        ),
        corwin_schultz_spread_bps=round(proxy_bps, 4),
        proxy_error_bps=round(proxy_bps - float(measured), 4),
        proxy_ratio=round(float(ratio), 4),
        amihud_illiquidity=(
            round(float(amihud_valid.iloc[-1]), 8) if not amihud_valid.empty else 0.0
        ),
        verdict=verdict,
        notes=notes,
    )


def estimate_trade_cost(
    input_data: EstimateTradeCostInput,
) -> EstimateTradeCostResult:
    """
    Price one hypothetical trade under a composed cost model, itemized.

    Every leg is one of backtest/costs.py's pure functions, called with the
    caller's own numbers — this is the same arithmetic a backtest applies
    per fill, made answerable without running one. Two of those functions
    (`maker_taker_cost`, `pct_of_range_spread`) had no other agent-callable
    path at all.

    The result reports `breakeven_move_bps` as TWICE the one-way total,
    because an entry has to earn its exit's cost as well as its own, and a
    one-way figure quietly understates what the trade must make.
    """
    notional = input_data.notional
    legs: List[TradeCostLeg] = []
    notes: List[str] = []

    def _add(component: str, model: str, cost: float) -> None:
        legs.append(
            TradeCostLeg(
                component=component,
                model=model,
                cost=round(float(cost), 6),
                bps_of_notional=round(float(cost) / notional * 10_000.0, 4),
            )
        )

    if input_data.commission_model == "pct":
        _add(
            "commission",
            "percentage_commission",
            percentage_commission(notional, input_data.commission_pct),
        )
    elif input_data.commission_model == "per_share":
        shares = input_data.shares
        assert shares is not None  # guaranteed by the input model's validator
        commission = per_share_commission(
            shares, input_data.per_share_rate, input_data.min_commission
        )
        _add("commission", "per_share_commission", commission)
        if commission <= input_data.min_commission + 1e-12:
            notes.append(
                f"Commission is at the {input_data.min_commission} minimum, "
                "not the per-share rate — the trade is small enough that the "
                "floor dominates, so cost does not scale with size here."
            )
    elif input_data.commission_model == "directional":
        _add(
            "commission",
            "directional_commission",
            directional_commission(
                notional,
                input_data.buy_rate,
                input_data.sell_rate,
                input_data.side == "buy",
            ),
        )
    elif input_data.commission_model == "maker_taker":
        cost = maker_taker_cost(
            notional,
            input_data.taker_rate,
            input_data.maker_rate,
            input_data.is_maker,
        )
        _add("commission", "maker_taker_cost", cost)
        if cost < 0:
            notes.append(
                "Commission is NEGATIVE: this is a maker rebate, so the "
                "exchange pays for the fill. It offsets the other legs "
                "rather than being free money — check the total, not this leg."
            )

    if input_data.spread_model == "fixed_bps":
        _add(
            "spread",
            "fixed_bps_spread",
            fixed_bps_spread(notional, input_data.spread_bps),
        )
    elif input_data.spread_model == "pct_of_range":
        _add(
            "spread",
            "pct_of_range_spread",
            pct_of_range_spread(
                notional,
                float(input_data.bar_high),  # type: ignore[arg-type]
                float(input_data.bar_low),  # type: ignore[arg-type]
                float(input_data.bar_close),  # type: ignore[arg-type]
                input_data.range_pct,
            ),
        )

    if input_data.avg_dollar_volume is not None and input_data.volatility is not None:
        participation = notional / input_data.avg_dollar_volume
        _add(
            "impact",
            "impact_cost",
            impact_cost(
                notional,
                input_data.avg_dollar_volume,
                input_data.volatility,
                input_data.impact_coefficient,
            ),
        )
        if participation > 0.1:
            notes.append(
                f"This trade is {participation:.1%} of average dollar volume. "
                "The square-root impact model is calibrated for small "
                "participation rates; at this size the estimate is an "
                "extrapolation, and the real constraint is probably capacity "
                "rather than cost — see get_capacity_report."
            )

    if input_data.short_borrow_bps > 0:
        _add(
            "borrow",
            "short_borrow_cost",
            short_borrow_cost(
                notional, input_data.short_borrow_bps, input_data.holding_days
            ),
        )
    if input_data.margin_cash < 0 and input_data.margin_annual_rate > 0:
        _add(
            "margin_interest",
            "margin_interest",
            margin_interest(
                input_data.margin_cash,
                input_data.margin_annual_rate,
                input_data.holding_days,
            ),
        )

    total = float(sum(leg.cost for leg in legs))
    total_bps = total / notional * 10_000.0
    logger.debug(
        "[estimate_trade_cost] notional=%.2f legs=%d total_bps=%.2f",
        notional,
        len(legs),
        total_bps,
    )
    return EstimateTradeCostResult(
        notional=notional,
        side=input_data.side,
        legs=legs,
        total_cost=round(total, 6),
        total_bps=round(total_bps, 4),
        breakeven_move_bps=round(total_bps * 2.0, 4),
        notes=notes,
    )


def detect_liquidity_events(
    input_data: LiquidityEventsInput,
) -> LiquidityEventsResult:
    """
    Which part of the market changed, not merely that it did.

    "NVDA moved 1.4 sigma" describes one channel — price — and it is the
    channel that changes LAST. A liquidity event usually shows up first in
    the spread, in one-sided flow, or in depth leaving one side of the book,
    and by the time the mid has moved the interesting part is over. This
    runs a CUSUM change detector across several channels and reports which
    of them broke:

        spread shock            very high
        effective_spread shock  very high
        signed_volume shock     high
        (mid_return did not trigger)

    A channel that cannot be computed is REPORTED with what it needs, never
    silently dropped — dropping it would let a caller ask for order-flow
    imbalance, get a clean report with no OFI row, and conclude the flow was
    balanced.

    Read `shift` rather than `peak_statistic` when judging size. The
    statistic is accumulated and unbounded; the shift is in the channel's
    own units. When `degenerate_baseline` is set, the reference window
    barely varied and the statistic is a ratio to a denominator near zero —
    the shift is the only number worth reading.
    """
    from standard_quant_tools.analysis.liquidity_events import (
        detect_liquidity_events as _detect,
    )
    from standard_quant_tools.data.factory import DataFactory

    logger.debug(
        "[detect_liquidity_events] %s channels=%s",
        input_data.symbol,
        input_data.channels,
    )
    provider = DataFactory.get_provider(input_data.source or "yfinance")

    trades = quotes = None
    fetch_notes: List[str] = []
    for name, method in (("trades", "get_trades"), ("quotes", "get_quotes")):
        try:
            frame = getattr(provider, method)(
                input_data.symbol, input_data.start_date, input_data.end_date
            )
            if name == "trades":
                trades = frame
            else:
                quotes = frame
        except NotImplementedError as exc:
            fetch_notes.append(f"{name}: {exc}")
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            fetch_notes.append(f"{name}: {exc}")

    report = _detect(
        channels=input_data.channels,
        trades=trades,
        quotes=quotes,
        freq=input_data.freq,
        threshold=input_data.threshold,
        reference_fraction=input_data.reference_fraction,
    )
    return LiquidityEventsResult(
        symbol=input_data.symbol,
        channels_run=report["channels_run"],
        n_triggered=report["n_triggered"],
        worst_channel=report["worst_channel"],
        summary=report["summary"],
        results=[ChannelResult(**r) for r in report["results"]],
        unavailable=[UnavailableChannel(**u) for u in report["unavailable"]],
        warnings=report["warnings"] + fetch_notes,
    )


def plan_rebalance(input_data: PlanRebalanceInput) -> PlanRebalanceResult:
    """
    A day-by-day path from the weights you hold to the weights you want.

    Every optimizer here returns a target vector and implicitly assumes you
    arrive instantly and for free. You do not, and the two costs pull in
    opposite directions: trade fast and pay market impact, trade slow and
    keep holding the portfolio you were trying to leave.

    So this returns the SCHEDULE and both costs rather than one number.
    `urgency` is the only judgement call and it is exposed rather than made
    for you.

    THE THING THIS SURFACES that nothing else does: a target weight the
    market cannot supply. An optimizer will happily put 5% in a name whose
    daily volume supports 0.2% — the weight vector is valid, the backtest
    fills at the close, and the position is simply never attainable in the
    size the model assumed. It appears here in `unreachable`, with the number
    of days it would really take.
    """
    from standard_quant_tools.portfolio.rebalance import plan_rebalance as _plan

    logger.debug(
        "[plan_rebalance] %d -> %d names, urgency=%.2f",
        len(input_data.current_weights),
        len(input_data.target_weights),
        input_data.urgency,
    )
    result = _plan(
        input_data.current_weights,
        input_data.target_weights,
        portfolio_value=input_data.portfolio_value,
        adv=input_data.adv,
        max_participation=input_data.max_participation,
        max_days=input_data.max_days,
        urgency=input_data.urgency,
        impact_coefficient=input_data.impact_coefficient,
    )
    return PlanRebalanceResult(
        n_days=result["n_days"],
        total_turnover=result["total_turnover"],
        total_cost_bps=result.get("total_cost_bps"),
        total_cost_dollars=result.get("total_cost_dollars"),
        converged=result["converged"],
        residual_distance=result.get("residual_distance"),
        schedule=[RebalanceStep(**s) for s in result["schedule"]],
        unreachable=[UnreachableName(**u) for u in result["unreachable"]],
        warnings=result["warnings"],
    )


def estimate_covariance(
    input_data: EstimateCovarianceInput,
) -> EstimateCovarianceResult:
    """
    A covariance matrix, plus the diagnostics that say whether to trust it.

    The optimizer already warns about conditioning. Shrinkage is the ANSWER
    to that warning rather than a caveat about it: a covariance over N assets
    has N(N+1)/2 parameters, so 40 assets on 120 days is about six numbers
    per parameter, and the smallest eigenvalues — the directions an optimizer
    levers into because they look like free risk reduction — are the ones
    estimated worst.

    Read `observations_per_parameter` and `condition_number` before the
    matrix. Measured on 40 assets and 120 days: sample gives a condition
    number of 205, Ledoit-Wolf 143, EWMA 228 (worse, because it lowers the
    effective sample size), and shrunk EWMA 35.

    The matrix comes back ANNUALIZED, matching every other risk number here.
    """
    from standard_quant_tools.portfolio.covariance import (
        estimate_covariance as _estimate,
    )

    logger.debug(
        "[estimate_covariance] %d tickers method=%s",
        len(input_data.tickers),
        input_data.method,
    )
    returns = fetch_returns_sync(
        input_data.tickers, input_data.start_date, input_data.end_date
    )
    result = _estimate(
        returns,
        method=input_data.method,
        halflife=input_data.halflife,
        periods_per_year=input_data.periods_per_year,
    )
    return EstimateCovarianceResult(**result)
