# Modeling reference

Every feature, estimator, target and spec option the modeling runtime
knows, read from its registries. **Generated** by
`Development/generate_modeling_reference.py` -- a test regenerates it and
fails if this file has drifted, so an entry added without regenerating
breaks the suite in the commit that added it. The prose that explains
these lives in [15_modeling.md](15_modeling.md); this is the catalog.

The estimators marked *optional* register only when their library is
installed. They are listed from a static declaration so this document is
the same on every machine; `list_modeling_capabilities` reports which of
them the running install actually has.

## Features (23)

| id | scope | temporal | lookback | requires | default params | description |
|---|---|---|---|---|---|---|
| `factors.pca_factor_return` | universe | pit_safe | 252 | Close | `refit_every=21`, `window=252` | That date's realized universe return projected onto the currently-held PC1 loadings — a shared macro factor (same value for every entity that date). |
| `factors.pca_loading` | universe | pit_safe | 252 | Close | `refit_every=21`, `window=252` | Entity's loading on PC1 of the universe return panel, refit every `refit_every` bars and forward-filled between refits. |
| `market.momentum` | entity | pit_safe | 20 | Close | `lookback=20` | Trailing close-to-close return over `lookback` bars. |
| `market.new_high_breakout` | entity | pit_safe | 20 | High, Close | `period=20` | 1.0 if Close breaks above the prior `period`-bar High (today's own bar excluded), else 0.0. NaN until `period` bars of history exist — the warm-up is unknown, not a confirmed non-breakout. |
| `market.psar_trend` | entity | pit_safe | 1 | High, Low | `af_max=0.2`, `af_start=0.02`, `af_step=0.02` | Parabolic SAR trend direction: 1.0 (uptrend) or -1.0 (downtrend). |
| `network.avg_correlation` | universe | pit_safe | 126 | Close | `refit_every=21`, `window=126` | Entity's mean correlation to the rest of the universe over a trailing window, refit every `refit_every` bars. Scale-free, unlike a PC1 loading: it says how much company a name keeps rather than how much variance it contributes. |
| `network.mst_degree` | universe | pit_safe | 126 | Close | `refit_every=21`, `window=126` | Entity's degree in the minimum spanning tree of the universe's correlation-distance matrix (Mantegna). Local topology, not a global factor: a hub is a name others route through, which a PC1 loading cannot express. |
| `risk.atr_pct` | entity | pit_safe | 14 | High, Low, Close | `period=14` | Wilder's Average True Range as a fraction of Close (normalized, comparable across differently-priced stocks). |
| `risk.bollinger_pct_b` | entity | pit_safe | 20 | Close | `num_std=2.0`, `period=20` | Position of Close within its Bollinger Bands: 0=lower band, 1=upper band, 0.5 when a flat window collapses the bands onto the mean. |
| `risk.garman_klass_volatility` | entity | pit_safe | 20 | Open, High, Low, Close | `period=20` | Garman-Klass OHLC realized volatility (annualized). |
| `risk.parkinson_volatility` | entity | pit_safe | 20 | High, Low | `period=20` | Parkinson high-low range realized volatility (annualized). |
| `risk.realized_volatility` | entity | pit_safe | 20 | Open, High, Low, Close | `period=20` | Yang-Zhang realized volatility (annualized). |
| `risk.rolling_beta` | entity | pit_safe | 60 | Close | `window=60` | Rolling OLS beta of the entity's returns against DatasetSpec.benchmark. |
| `risk.rolling_drawdown` | entity | pit_safe | 252 | Close | `window=252` | Drawdown of Close from its trailing `window`-bar peak (0 at a new high, negative otherwise). |
| `statistical.hurst` | entity | pit_safe | 200 | Close | `method='dfa'`, `window=200` | Rolling Hurst exponent — >0.55 trending, <0.45 mean-reverting. |
| `technical.adx` | entity | pit_safe | 14 | High, Low, Close | `period=14` | Average Directional Index — trend strength, unsigned. |
| `technical.macd_histogram` | entity | pit_safe | 26 | Close | `fast=12`, `signal=9`, `slow=26` | MACD histogram (MACD line minus its signal line) — trend-momentum divergence. |
| `technical.rsi` | entity | pit_safe | 14 | Close | `period=14` | Relative Strength Index — momentum oscillator, 0-100. |
| `technical.stochastic_k` | entity | pit_safe | 14 | High, Low, Close | `d_period=3`, `k_period=14` | Stochastic oscillator %K — momentum vs. recent high-low range, 0-100. |
| `technical.williams_r` | entity | pit_safe | 14 | High, Low, Close | `period=14` | Williams %R momentum oscillator, -100 (oversold) to 0 (overbought). |
| `volume.mfi` | entity | pit_safe | 14 | High, Low, Close, Volume | `period=14` | Money Flow Index — volume-weighted RSI, 0-100. |
| `volume.obv_roc` | entity | pit_safe | 20 | Close, Volume | `lookback=20` | Rate of change of On-Balance Volume over `lookback` bars. |
| `volume.vwap_deviation` | entity | pit_safe | 20 | High, Low, Close, Volume | `period=20` | (Close - VWAP) / VWAP over a trailing `period`-bar window. |

## Estimators (18 always available, 6 optional)

Parameter values are bounded as well as named; see [15_modeling.md](15_modeling.md#parameter-values-are-bounded-not-just-named).

| task | name | class | allowed params | capabilities |
|---|---|---|---|---|
| classification | `gradient_boosting` | `sklearn.ensemble._gb.GradientBoostingClassifier` | `learning_rate`, `max_depth`, `n_estimators` | sample weights, probabilities, importances |
| classification | `hist_gradient_boosting` | `sklearn.ensemble._hist_gradient_boosting.gradient_boosting.HistGradientBoostingClassifier` | `learning_rate`, `max_depth`, `max_iter` | sample weights, probabilities |
| classification | `logistic` | `sklearn.linear_model._logistic.LogisticRegression` | `C`, `fit_intercept`, `l1_ratio`, `max_iter`, `penalty`, `solver` | sample weights, probabilities, coefficients |
| classification | `mlp` | `standard_quant_tools.modeling.estimators.neural.PanelMLPClassifier` | `alpha`, `early_stopping`, `learning_rate_init`, `max_iter`, `n_hidden_layers`, `n_hidden_units`, `random_state` | probabilities |
| classification | `random_forest` | `sklearn.ensemble._forest.RandomForestClassifier` | `max_depth`, `n_estimators` | sample weights, probabilities, importances |
| classification | `sgd` | `standard_quant_tools.modeling.estimators.online.ProbabilisticSGDClassifier` | `alpha`, `eta0`, `fit_intercept`, `l1_ratio`, `learning_rate`, `loss`, `max_iter`, `penalty`, `random_state`, `tol` | sample weights, probabilities, coefficients |
| regression | `elastic_net` | `sklearn.linear_model._coordinate_descent.ElasticNet` | `alpha`, `fit_intercept`, `l1_ratio`, `max_iter` | sample weights, coefficients |
| regression | `gradient_boosting` | `sklearn.ensemble._gb.GradientBoostingRegressor` | `learning_rate`, `max_depth`, `n_estimators` | sample weights, importances |
| regression | `hist_gradient_boosting` | `sklearn.ensemble._hist_gradient_boosting.gradient_boosting.HistGradientBoostingRegressor` | `learning_rate`, `max_depth`, `max_iter` | sample weights |
| regression | `huber` | `sklearn.linear_model._huber.HuberRegressor` | `alpha`, `epsilon`, `fit_intercept`, `max_iter` | sample weights, coefficients |
| regression | `lasso` | `sklearn.linear_model._coordinate_descent.Lasso` | `alpha`, `fit_intercept`, `max_iter` | sample weights, coefficients |
| regression | `linear` | `sklearn.linear_model._base.LinearRegression` | `fit_intercept` | sample weights, coefficients |
| regression | `mlp` | `standard_quant_tools.modeling.estimators.neural.PanelMLPRegressor` | `alpha`, `early_stopping`, `learning_rate_init`, `max_iter`, `n_hidden_layers`, `n_hidden_units`, `random_state` |  |
| regression | `quantile` | `sklearn.linear_model._quantile.QuantileRegressor` | `alpha`, `fit_intercept`, `quantile`, `solver` | sample weights, coefficients |
| regression | `quantile_gradient_boosting` | `standard_quant_tools.modeling.estimators.boosting.QuantileGradientBoostingRegressor` | `alpha`, `learning_rate`, `max_depth`, `n_estimators` | sample weights, importances |
| regression | `random_forest` | `sklearn.ensemble._forest.RandomForestRegressor` | `max_depth`, `n_estimators` | sample weights, importances |
| regression | `ridge` | `sklearn.linear_model._ridge.Ridge` | `alpha`, `fit_intercept`, `max_iter` | sample weights, coefficients |
| regression | `sgd` | `sklearn.linear_model._stochastic_gradient.SGDRegressor` | `alpha`, `eta0`, `fit_intercept`, `l1_ratio`, `learning_rate`, `loss`, `max_iter`, `penalty`, `random_state`, `tol` | sample weights |

### Optional

| task | name | requires | allowed params |
|---|---|---|---|
| classification | `lightgbm` | *optional: `lightgbm`* | `colsample_bytree`, `learning_rate`, `max_depth`, `min_child_samples`, `n_estimators`, `num_leaves`, `reg_alpha`, `reg_lambda`, `subsample` |
| classification | `xgboost` | *optional: `xgboost`* | `colsample_bytree`, `learning_rate`, `max_depth`, `min_child_weight`, `n_estimators`, `reg_alpha`, `reg_lambda`, `subsample` |
| ranking | `lightgbm_ranker` | *optional: `lightgbm`* | `colsample_bytree`, `learning_rate`, `max_depth`, `min_child_samples`, `n_estimators`, `num_leaves`, `reg_alpha`, `reg_lambda`, `subsample` |
| ranking | `xgboost_ranker` | *optional: `xgboost`* | `colsample_bytree`, `learning_rate`, `max_depth`, `min_child_weight`, `n_estimators`, `reg_alpha`, `reg_lambda`, `subsample` |
| regression | `lightgbm` | *optional: `lightgbm`* | `colsample_bytree`, `learning_rate`, `max_depth`, `min_child_samples`, `n_estimators`, `num_leaves`, `reg_alpha`, `reg_lambda`, `subsample` |
| regression | `xgboost` | *optional: `xgboost`* | `colsample_bytree`, `learning_rate`, `max_depth`, `min_child_weight`, `n_estimators`, `reg_alpha`, `reg_lambda`, `subsample` |

## Preprocessing steps (8)

Composed in order by `PreprocessingSpec.steps`; each is fitted on the fold's training rows and its state applied to the test rows, then persisted with the model as `preprocessing_state.json`. `normalization='pooled'` resolves to `winsorize` then `zscore`; `'cross_sectional'` to `cross_sectional_standardize`.

| id | params | defaults | state | column-wise | description |
|---|---|---|---|---|---|
| `cross_sectional_standardize` | `clip_sigma` | `clip_sigma=3.0` | stateless | yes | Standardize within each date's cross-section and clip at clip_sigma, so what reaches the model is each entity's position relative to its peers that day. Stateless: nothing crosses the fold boundary. |
| `impute` | `fill_value`, `strategy` | `fill_value=0.0`, `strategy='median'` | fitted on train | yes | Fill NaN with the training fold's median or mean, or a constant. A missing test value receives the TRAINING statistic, never the test fold's own. |
| `missing_indicator` |  |  | stateless | yes | Add a <column>__missing indicator (1.0 where NaN) for every input column, keeping the originals. Meaningful once the dataset's missing-data policy lets NaN reach the engine; pair with impute. |
| `pca_whiten` | `n_components`, `whiten` | `n_components=8`, `whiten=True` | fitted on train | no | Replace the columns with their leading n_components principal components, fitted on the training fold and scaled to unit variance when whiten is set. Not column-wise: every output depends on every input. Refuses NaN; put impute before it. |
| `quantile_transform` | `n_quantiles`, `output` | `n_quantiles=1000`, `output='normal'` | fitted on train | yes | Map each column through its training-fold empirical distribution: rank-gauss for output='normal', a uniform [0, 1] for 'uniform'. Removes the shape of the distribution entirely, tails included. |
| `robust_scale` | `scale_to_normal` | `scale_to_normal=True` | fitted on train | yes | Centre by the training median and scale by the median absolute deviation, which a single extreme print cannot move; scale_to_normal makes the MAD a consistent estimate of the standard deviation on Gaussian data. |
| `winsorize` | `lower`, `upper` | `lower=0.01`, `upper=0.99` | fitted on train | yes | Clip each column to its training-fold quantiles, so a single extreme print cannot set the scale for everything that follows. |
| `zscore` |  |  | fitted on train | yes | Centre and scale each column by its training-fold mean and standard deviation. Leaves the market factor inside every feature; pair with cross_sectional_standardize for a model judged on cross-sectional IC. |

## Targets (6 buildable from prices, 12 external only)

| id | tasks | buildable | kind | description |
|---|---|---|---|---|
| `adverse_selection` | regression, ranking | external only | continuous | How much the mid moves against a fill after it happens. The cost of being the one who was willing to trade. |
| `fill_probability` | classification | external only | discrete | Whether a passive order resting at a stated level fills within the horizon. Needs queue position and cancellations, so no bar-derived series can produce it. |
| `forward_direction` | classification | yes | discrete | That forward return binarized against `threshold`. |
| `forward_return` | regression, ranking | yes | continuous | The return from t to t+horizon. |
| `forward_return_market_neutral` | regression, ranking | yes | continuous | Forward return minus that date's equal-weighted mean. |
| `forward_return_rank` | regression, ranking | yes | continuous | Its rank within the date's cross-section, in [-0.5, 0.5]. |
| `forward_return_vol_scaled` | regression, ranking | yes | continuous | Forward return over the entity's own trailing volatility. |
| `future_depth` | regression, ranking | external only | continuous | Resting size at t+horizon. What will be THERE to trade against, which a spread forecast does not answer -- a tight quote for a hundred shares and a tight quote for fifty thousand cost the same to cross and are not the same liquidity. |
| `future_markout` | regression, ranking | external only | continuous | Mid move measured FROM a fill, signed by the side taken. The standard read on whether a trade was well-placed. |
| `future_microprice_return` | regression, ranking | external only | continuous | Return of the size-weighted touch price. Leads the mid when the book is lopsided, which is exactly when the mid is least informative. |
| `future_mid_return` | regression, ranking | external only | continuous | Return of the MIDPOINT over the horizon. Not the same as a trade-price return: the mid moves without a trade and is where a passive order is measured from. |
| `future_ofi` | regression, ranking | external only | continuous | Signed order-flow imbalance over the horizon, from book updates. Predicting FLOW rather than price: the quantity that moves the price, one step earlier. |
| `future_spread` | regression, ranking | external only | continuous | The quoted spread at t+horizon. A liquidity forecast rather than a price one -- what it will COST to cross, not where the price goes. |
| `future_trade_intensity` | regression, ranking | external only | continuous | Trades per unit time over the horizon. Distinct from volume: one block and two hundred odd lots are the same volume and completely different information. |
| `future_volume` | regression, ranking | external only | continuous | Traded volume over the horizon. Bar volume can approximate this at daily frequency, but not at the horizons this exists for, where the question is how much prints in the next thirty seconds. |
| `next_mid_direction` | classification | external only | discrete | Whether the midpoint's next move is up or down. |
| `time_to_fill` | regression | external only | continuous | How long that order waits before filling. CENSORED by construction -- an order that never fills has no time, and recording it as the horizon rather than as unfilled biases every estimate toward patience. |
| `triple_barrier` | classification | yes | discrete | Which barrier is touched first: up, down, or neither. |

## Spec options

| field | choices |
|---|---|
| `ValidationSpec.method` | `walk_forward`, `purged_kfold` |
| `ValidationSpec.scheme` | `rolling`, `expanding` |
| `PreprocessingSpec.normalization` | `pooled`, `cross_sectional` |
| `WeightingSpec.method` | `none`, `label_uniqueness`, `time_decay`, `uniqueness_and_time_decay` |
| `SearchSpec.method` | `grid`, `random` |
| `SearchSpec.scoring` | `cs_rank_ic`, `cs_ic`, `r2`, `neg_mae`, `accuracy`, `auc` |
| `EstimatorSpec.calibration` | `none`, `isotonic`, `sigmoid` |
| `PredictionTransformSpec.method` | `sign`, `cross_sectional_rank`, `cross_sectional_zscore`, `top_bottom_quantile` |
| `PredictionTransformSpec.rebalance_frequency` | `daily`, `weekly`, `monthly` |

## Limits

| name | value | meaning |
|---|---|---|
| `MAX_LAG` | 60 | deepest single lag, in bars |
| `MAX_LAGS_PER_FEATURE` | 20 | lags one feature may request |
| `MAX_EXPANDED_COLUMNS` | 400 | ceiling on the expanded panel |
