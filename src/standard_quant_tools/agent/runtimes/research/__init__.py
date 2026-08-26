"""The `research` runtime's registry: what it advertises and what it
can execute. The two are built from one list, so a tool cannot be
advertised without being dispatchable or the reverse."""

from standard_quant_tools.agent.models import (
    AdvancedIndicatorsInput,
    AnalysisInput,
    ChangePointInput,
    CointegrationInput,
    CorrelationAnalysisInput,
    DataQualityReportInput,
    ExtendedRiskInput,
    FactorRegressionInput,
    FundamentalsInput,
    GarchVolatilityForecastInput,
    GrangerInput,
    HurstInput,
    ImpliedVolatilityInput,
    KalmanHedgeRatioInput,
    OptionPricingInput,
    PairScannerInput,
    PartialCorrelationInput,
    PCAInput,
    PortfolioInput,
    RallyDetectionInput,
    RegimeDetectionInput,
    RollingBetaInput,
    ScreenerInput,
    StationarityInput,
    TailDependenceInput,
    TailRiskInput,
    TechnicalInput,
    TechnicalPanelInput,
    VolatilityEstimatorsInput,
)

from .diagnostic_tools import (  # noqa: F401
    DIAGNOSTIC_TOOL_DEFS,
    DIAGNOSTIC_TOOL_DISPATCH,
    get_drawdown_profile,
    get_entropy_measures,
    get_lead_lag_matrix,
    get_sharpe_stability,
    run_seasonality_analysis,
    test_autocorrelation,
    test_structural_break,
)
from .inference_tools import (  # noqa: F401
    INFERENCE_TOOL_DEFS,
    INFERENCE_TOOL_DISPATCH,
    compare_distributions,
    decompose_returns,
    estimate_tail_index,
    get_bootstrap_interval,
    get_correlation_stability,
    test_normality,
)
from .tools import (
    analyze_stock_risk,
    analyze_tail_dependence,
    detect_change_points,
    detect_regimes,
    get_advanced_indicators,
    get_correlation_analysis,
    get_data_quality_report,
    get_extended_risk_metrics,
    get_implied_volatility,
    get_option_pricing,
    get_partial_correlation,
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
    run_stationarity_tests,
    scan_pairs,
    test_granger_causality,
)

#: (name, description, input model) — the single source for both
#: the advertised schema and the dispatch table below.
TOOL_DEFS = [
    (
        "detect_change_points",
        "When the process generating a series CHANGED, by binary segmentation on the mean. run_hurst_analysis says what KIND of process a series is; this says when it stopped being that one, which the first cannot -- a single Hurst exponent over a sample containing a break describes neither regime. Read `gain` on each break: a marginal call then looks marginal instead of looking like a boundary.",
        ChangePointInput,
    ),
    (
        "get_partial_correlation",
        "The correlation between two assets once the common drivers are removed from BOTH. Two stocks in one sector correlate at 0.7 and it says almost nothing; take out the market and the sector and what is left is the part actually about those two companies. That residual is what a pair trade lives on, and the raw correlation systematically overstates it.",
        PartialCorrelationInput,
    ),
    (
        "test_granger_causality",
        "Whether one series helps predict another beyond that series' own past. NOT causality: a common driver produces it and so does a faster-updating proxy for the same information. It establishes temporal precedence in a linear model, which is necessary for a tradeable lead and nowhere near sufficient. Every lag is tested and the smallest p-value reported, so treat it as a screen rather than a test result.",
        GrangerInput,
    ),
    (
        "analyze_tail_dependence",
        "Whether two assets move together IN THE TAIL, which is the only regime a diversification claim has to survive. A full-sample correlation of 0.3 is compatible with two assets that are independent day to day and fall together every time it matters. Read n_tail_observations alongside the estimate: at a 1% quantile on a year of data that is two or three points.",
        TailDependenceInput,
    ),
    (
        "run_stationarity_tests",
        "ADF, KPSS and the variance ratio, with the four-way verdict spelled out. The two tests have OPPOSITE nulls, which is the whole reason to run both: failing to reject ADF is not evidence of a unit root, and the verdict separates 'the data says non-stationary' from 'the data says nothing'. 'contradictory' usually means a structural break rather than either answer.",
        StationarityInput,
    ),
    (
        "detect_regimes",
        "Label each observation with a volatility regime, by a Gaussian mixture. A MIXTURE rather than a hidden Markov model: it has no transition matrix, so it flips on single observations where an HMM would smooth, and `persistence` reports how often it does -- below about 0.8 the labels describe noise. Regimes come back sorted by volatility so regime 0 is always the calm one.",
        RegimeDetectionInput,
    ),
    (
        "analyze_stock_risk",
        "Full risk profile of one asset against a benchmark: alpha, beta, "
        "Sharpe, VaR and CVaR in a single call. The starting point for "
        "'what is this thing like', and the place most analyses begin. "
        "Every number here is a point estimate over the whole sample -- "
        "use get_bootstrap_interval for what the Sharpe's error bar "
        "actually is, and get_sharpe_stability to check the edge did not "
        "decay inside the window.",
        AnalysisInput,
    ),
    (
        "get_technical_analysis",
        "Technical indicators for ONE ticker, with the parameters you choose "
        "-- RSI, MACD, Bollinger, moving averages and the rest. Use "
        "get_technical_panel instead when the question spans a universe: "
        "it computes the same indicators for every ticker in one native "
        "call, and looping this tool per ticker is the slow way to the "
        "same answer.",
        TechnicalInput,
    ),
    (
        "get_portfolio_analysis",
        "Risk and return metrics for a basket held at fixed weights: "
        "portfolio volatility, correlation structure, and contribution by "
        "position. Describes a portfolio you specify rather than choosing "
        "one -- run_portfolio_optimization and optimize_risk_parity "
        "choose, and get_marginal_risk_contribution answers the follow-up "
        "question of where the risk in this basket actually comes from.",
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
        "get_technical_panel",
        "Indicators (RSI/ADX/ATR/Bollinger/Stochastic) for a whole ticker universe in one native call, reported at the latest bar. Use instead of one get_technical_analysis call per ticker when screening.",
        TechnicalPanelInput,
    ),
]

# The quant_research tools declared in diagnostic_tools.py,
# concatenated rather than pasted so the group stays readable as a
# unit and cannot half-register.
TOOL_DEFS = TOOL_DEFS + DIAGNOSTIC_TOOL_DEFS

# The quant_research tools declared in inference_tools.py,
# concatenated rather than pasted so the group stays readable as a
# unit and cannot half-register.
TOOL_DEFS = TOOL_DEFS + INFERENCE_TOOL_DEFS

TOOL_DISPATCH = {name: (globals()[name], model) for name, _d, model in TOOL_DEFS}

#: This runtime's slice of the library-wide routing taxonomy.
TOOL_CATEGORY = {
    "analyze_stock_risk": "analysis",
    "analyze_tail_dependence": "quant_research",
    "detect_change_points": "quant_research",
    "detect_regimes": "quant_research",
    "get_partial_correlation": "quant_research",
    "run_stationarity_tests": "quant_research",
    "test_granger_causality": "quant_research",
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
    "get_technical_panel": "analysis",
}

TOOL_DISPATCH.update(DIAGNOSTIC_TOOL_DISPATCH)
TOOL_CATEGORY.update({name: "quant_research" for name in DIAGNOSTIC_TOOL_DISPATCH})

TOOL_DISPATCH.update(INFERENCE_TOOL_DISPATCH)
TOOL_CATEGORY.update({name: "quant_research" for name in INFERENCE_TOOL_DISPATCH})

__all__ = [
    "test_normality",
    "estimate_tail_index",
    "get_bootstrap_interval",
    "compare_distributions",
    "get_correlation_stability",
    "decompose_returns",
    "test_autocorrelation",
    "run_seasonality_analysis",
    "get_entropy_measures",
    "get_sharpe_stability",
    "get_drawdown_profile",
    "get_lead_lag_matrix",
    "test_structural_break",
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
