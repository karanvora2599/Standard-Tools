"""The `research` runtime's registry: what it advertises and what it
can execute. The two are built from one list, so a tool cannot be
advertised without being dispatchable or the reverse."""

from standard_quant_tools.agent.models import (
    AdvancedIndicatorsInput,
    AnalysisInput,
    CointegrationInput,
    CorrelationAnalysisInput,
    DataQualityReportInput,
    ExtendedRiskInput,
    FactorRegressionInput,
    FundamentalsInput,
    GarchVolatilityForecastInput,
    HurstInput,
    ImpliedVolatilityInput,
    KalmanHedgeRatioInput,
    OptionPricingInput,
    PairScannerInput,
    PCAInput,
    PortfolioInput,
    RallyDetectionInput,
    RollingBetaInput,
    ScreenerInput,
    TailRiskInput,
    TechnicalInput,
    TechnicalPanelInput,
    VolatilityEstimatorsInput,
)

from .tools import (
    analyze_stock_risk,
    get_advanced_indicators,
    get_correlation_analysis,
    get_data_quality_report,
    get_extended_risk_metrics,
    get_implied_volatility,
    get_option_pricing,
    get_portfolio_analysis,
    get_rally_signal,
    get_rolling_beta,
    get_stock_fundamentals,
    get_tail_risk_metrics,
    get_technical_analysis,
    get_technical_panel,
    get_volatility_estimators,
    run_cointegration_test,
    run_factor_regression,
    run_garch_volatility_forecast,
    run_hurst_analysis,
    run_kalman_hedge_ratio,
    run_pca_analysis,
    run_screener,
    scan_pairs,
)

#: (name, description, input model) — the single source for both
#: the advertised schema and the dispatch table below.
TOOL_DEFS = [
    (
        "analyze_stock_risk",
        "Full risk analysis: alpha, beta, Sharpe, VaR, CVaR.",
        AnalysisInput,
    ),
    (
        "get_technical_analysis",
        "Compute configurable technical indicators.",
        TechnicalInput,
    ),
    (
        "get_portfolio_analysis",
        "Multi-asset portfolio metrics.",
        PortfolioInput,
    ),
    (
        "run_screener",
        "Filter a stock universe by fundamental and technical criteria.",
        ScreenerInput,
    ),
    (
        "run_factor_regression",
        "Multi-factor OLS regression: alpha, loadings, t-stats, p-values, R².",
        FactorRegressionInput,
    ),
    (
        "run_cointegration_test",
        "Engle-Granger cointegration: hedge ratio, half-life, spread z-score signal.",
        CointegrationInput,
    ),
    (
        "run_kalman_hedge_ratio",
        "Time-varying hedge ratio via a Kalman filter — a staleness diagnostic companion to run_cointegration_test's static OLS hedge ratio.",
        KalmanHedgeRatioInput,
    ),
    (
        "run_pca_analysis",
        "PCA on multi-asset returns: explained variance, loadings, factor contributions.",
        PCAInput,
    ),
    (
        "get_correlation_analysis",
        "Correlation matrix, avg pairwise correlation, most/least correlated pair, and diversification ratio for a universe.",
        CorrelationAnalysisInput,
    ),
    (
        "run_hurst_analysis",
        "Hurst exponent (DFA/R-S): regime classification and optional rolling breakdown.",
        HurstInput,
    ),
    (
        "get_rally_signal",
        "Detect a rally via 5 confirming signals: return z-score, ADX trend strength, DI+/DI- direction, Hurst trending regime, and new-high breakout.",
        RallyDetectionInput,
    ),
    (
        "get_volatility_estimators",
        "Realized volatility via Parkinson, Garman-Klass, and Yang-Zhang estimators vs. plain close-to-close.",
        VolatilityEstimatorsInput,
    ),
    (
        "run_garch_volatility_forecast",
        "GARCH(1,1) conditional volatility: fits how variance evolves over time and forecasts it forward, unlike get_volatility_estimators' backward-looking realized estimates.",
        GarchVolatilityForecastInput,
    ),
    (
        "scan_pairs",
        "Scan a ticker universe for cointegrated pairs, ranked by half-life.",
        PairScannerInput,
    ),
    (
        "get_stock_fundamentals",
        "Fetch company metadata and key financial ratios (PE, P/B, debt/equity, ROE, market cap).",
        FundamentalsInput,
    ),
    (
        "get_advanced_indicators",
        "Compute Parabolic SAR (trend), Wilder ATR (volatility), and MFI (volume-flow oscillator).",
        AdvancedIndicatorsInput,
    ),
    (
        "get_rolling_beta",
        "Compute rolling OLS beta to detect beta drift over time vs a benchmark.",
        RollingBetaInput,
    ),
    (
        "get_extended_risk_metrics",
        "Extended risk: Calmar ratio, Treynor ratio, parametric VaR 95/99, historical VaR 99, CVaR 99.",
        ExtendedRiskInput,
    ),
    (
        "get_tail_risk_metrics",
        "Extreme Value Theory tail risk (Peaks-Over-Threshold GPD fit): VaR/CVaR extrapolated from the fitted tail, compared against the naive historical quantile.",
        TailRiskInput,
    ),
    (
        "get_data_quality_report",
        "Dataset provenance (adjusted/survivorship-free/point-in-time guarantees) plus missing-bar/stale-price/price-jump detection on a symbol's OHLCV.",
        DataQualityReportInput,
    ),
    (
        "get_option_pricing",
        "Black-Scholes-Merton price and Greeks (delta, gamma, vega, theta, rho) for a European option.",
        OptionPricingInput,
    ),
    (
        "get_implied_volatility",
        "Solve for Black-Scholes-Merton implied volatility from an observed European option price.",
        ImpliedVolatilityInput,
    ),
    (
        "get_technical_panel",
        "Indicators (RSI/ADX/ATR/Bollinger/Stochastic) for a whole ticker universe in one native call, reported at the latest bar. Use instead of one get_technical_analysis call per ticker when screening.",
        TechnicalPanelInput,
    ),
]

TOOL_DISPATCH = {name: (globals()[name], model) for name, _d, model in TOOL_DEFS}

#: This runtime's slice of the library-wide routing taxonomy.
TOOL_CATEGORY = {
    "analyze_stock_risk": "analysis",
    "get_technical_analysis": "analysis",
    "get_portfolio_analysis": "analysis",
    "run_screener": "screener",
    "run_factor_regression": "quant_research",
    "run_cointegration_test": "quant_research",
    "run_kalman_hedge_ratio": "quant_research",
    "run_pca_analysis": "quant_research",
    "get_correlation_analysis": "quant_research",
    "run_hurst_analysis": "quant_research",
    "get_rally_signal": "analysis",
    "get_volatility_estimators": "analysis",
    "run_garch_volatility_forecast": "analysis",
    "scan_pairs": "quant_research",
    "get_stock_fundamentals": "screener",
    "get_advanced_indicators": "analysis",
    "get_rolling_beta": "analysis",
    "get_extended_risk_metrics": "analysis",
    "get_tail_risk_metrics": "analysis",
    "get_data_quality_report": "analysis",
    "get_option_pricing": "analysis",
    "get_implied_volatility": "analysis",
    "get_technical_panel": "analysis",
}

__all__ = [
    "TOOL_CATEGORY",
    "TOOL_DEFS",
    "TOOL_DISPATCH",
    "analyze_stock_risk",
    "get_advanced_indicators",
    "get_correlation_analysis",
    "get_data_quality_report",
    "get_extended_risk_metrics",
    "get_implied_volatility",
    "get_option_pricing",
    "get_portfolio_analysis",
    "get_rally_signal",
    "get_rolling_beta",
    "get_stock_fundamentals",
    "get_tail_risk_metrics",
    "get_technical_analysis",
    "get_technical_panel",
    "get_volatility_estimators",
    "run_cointegration_test",
    "run_factor_regression",
    "run_garch_volatility_forecast",
    "run_hurst_analysis",
    "run_kalman_hedge_ratio",
    "run_pca_analysis",
    "run_screener",
    "scan_pairs",
]
