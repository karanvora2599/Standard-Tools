# Survey: `data/`, `audit/`, `mcp/` and the top-level modules

Slice: `src/standard_quant_tools/data/*` (19 files, 6,976 lines), `src/standard_quant_tools/audit/*` (15 files, 1,838 lines), `src/standard_quant_tools/mcp/*` (9 files, 2,819 lines), and the 13 top-level modules (2,244 lines). About 13,900 lines read in full. Exposure was determined by grepping the ten tool files (`agent/runtimes/<rt>/tools.py`, `modeling/agent/tools.py`, `modeling/agent/feature_tools.py`), `cli.py`, `mcp/server.py` and `mcp/resources.py` for imports and call sites. Nothing in the repository was modified.

## Legend for the `exposed by` column

| tag | meaning |
|---|---|
| `T:` | reached by a named LLM tool (runtime in parentheses). "indirect" means the tool reaches it through another function without letting the caller choose it. |
| `INT` | internal plumbing every tool passes through (cache, retry, hashing, validation). Reached, but never a decision the agent makes. |
| `CLI` | reachable only through the `sqt` console script. |
| `MCP-res` / `MCP-prompt` / `MCP-srv` | reachable as an MCP resource / prompt / server-side machinery of `sqt-mcp`. |
| `PROG` | programmatic only. Nothing on the tool surface, the CLI or the MCP surface reaches it. |
| `DEAD` | no caller anywhere in `src/`. |

## Headline findings (details in sections 5 and 8)

1. **The data runtime cannot choose a provider.** Every data-runtime tool calls `DataFactory.get_provider()` with no `source`, which is hard-wired to yfinance (`data/factory.py:19`, `agent/runtimes/data/tools.py:189,294,325,361,379`, `data/models.py:513`). `fetch_tick_tape` and `fetch_quote_panel` therefore always refuse outside a test that patches the factory: yfinance has no tick feed. The only places a provider is selectable are three meta tools, the portfolio tick tools (`_tick_provider(source)`), and `build_model_dataset` (`DatasetSpec.provider`).
2. **`request_id` never reaches the caller.** `_run_and_record` mints it and returns only the result dict; no dispatcher, no runtime and not the MCP server surfaces it. `explain_decision`, `replay_decision` and `compare_decisions` therefore need an id the agent can only obtain by reading the audit files out of band. There is no tool that lists or searches decision records.
3. **Four provider capabilities have no fetch tool and no reference kind that a provider can publish into:** `get_order_book` (Databento, L2), `get_order_events` (Databento, MBO), `get_point_in_time_records` (Polygon, filings stamped with `filing_date`) and the Databento `to_raw_symbol` mapping. `Documentation/26_data.md` still says "no shipped provider serves depth", which stopped being true when `DatabentoProvider` landed.
4. **Databento fetches are invisible to the audit trail.** `DatabentoProvider` never calls `audit.record_data_access`, so a decision record for a call that read Databento bars has an empty `data_sources` and `replay_decision` can never return `data_changed` for it. The same is true of every provider's `get_trades`/`get_quotes` and Polygon's PIT records: only `get_ohlcv` is recorded.
5. **`feature_lab` records cannot be replayed or pre-validated.** `audit.replay._resolve_tool` and `meta.validate_tool_call` search `agent.tools._TOOL_DISPATCH` (the eight non-modeling runtimes) plus `MODELING_TOOL_DISPATCH`; `FEATURE_TOOL_DISPATCH` is in neither, so the nine feature_lab tools' records fail with "Unknown tool".
6. **The MCP server docstring claims it sets a request-id context per call; it does not** (`mcp/server.py:28-31` vs `call_tool`, which only dispatches). `sqt://audit/{request_id}` is served, but the client is never told which id its call produced.

---

## 1. Inventory: `data/`

### `data/factory.py`, `data/metadata.py`, `data/_retry.py`, `data/quality.py`

| module | function/class | purpose | inputs -> outputs | exposed by |
|---|---|---|---|---|
| factory.py | `DataFactory.get_provider(source="yfinance", api_key, host, port)` | Construct a provider by name; databento imported lazily | `str` -> `DataProvider` | T: every fetching tool, **always with no `source`** in data/research/backtest/portfolio (yfinance). `source` selectable only in `describe_data_capabilities`, `describe_temporal_contract`, `compare_data_sources` (meta), `get_trade_profile` and `_tick_provider` callers (portfolio), one portfolio tool at L1297, and `build_model_dataset` via `DatasetSpec.provider` (modeling). |
| metadata.py | `DataSetMetadata` | Provider's honest self-report: adjusted, survivorship_free, point_in_time, frequency, timezone | pydantic model | T: `get_dataset_metadata` (data), `describe_data_capabilities` (meta), `get_data_quality_report` (research) |
| _retry.py | `retry(times=3, delay=1, backoff=2)` | Retry transient provider failures; never retries InvalidSymbol/DataNotFound/NonRetryableAPIError or non-API QuantErrors | decorator | INT (yfinance, polygon, bloomberg fetches) |
| quality.py | `detect_missing_bars(df)` | Weekday gaps in a DatetimeIndex (no holiday calendar) | df -> `[{date, weekday}]` | T: `get_data_quality_report` (research) |
| quality.py | `detect_stale_prices(df, n=3)` | Runs of n identical closes | df -> `[{start,end,price,run_length}]` | T: `get_data_quality_report` |
| quality.py | `detect_price_jumps(df, threshold=0.15)` | Single-bar close moves over threshold | df -> `[{date, pct_change}]` | T: `get_data_quality_report` |

### `data/base.py`

| module | function/class | purpose | inputs -> outputs | exposed by |
|---|---|---|---|---|
| base.py | `TickerInfo` | Company reference record | pydantic | T: `get_stock_fundamentals` (research), `get_capacity_report` (portfolio) |
| base.py | `FinancialRatios` (+ `definition_notes`) | Ratios in one canonical unit; declares definition departures | pydantic | T: `fetch_financial_ratios` (data), `get_stock_fundamentals`, `run_screener` (research), `compare_data_sources` (meta) |
| base.py | `DataProvider.SUPPORTED_INTERVALS` | Declared interval vocabulary | frozenset or None | T: `describe_data_capabilities` (meta). Only YFinance and Polygon define it; Bloomberg and Databento report `None` although each has a private table (`_PERIODICITY`, `BAR_SCHEMAS`). |
| base.py | `get_ohlcv(symbol, start, end, interval)` | Bars with an INCLUSIVE end contract enforced by `trim_to_inclusive_end` | -> OHLCV df | T: `fetch_ohlcv` (data) and ~50 tools in research/backtest/portfolio |
| base.py | `get_ohlcv_async(...)` | Same, thread-executor with contextvars copied | -> df | T: `fetch_ohlcv_panel`, `fetch_returns_panel` (data) via `portfolio.fetch_ohlcv_panel_sync`; portfolio panel tools |
| base.py | `get_ticker_info(symbol)` | Reference data | -> `TickerInfo` | T: `get_stock_fundamentals` (research), `get_capacity_report` (portfolio) |
| base.py | `get_financial_ratios(symbol)` | Ratios | -> `FinancialRatios` | T: `fetch_financial_ratios` (data), `get_stock_fundamentals`, `run_screener` (research), `compare_data_sources` (meta) |
| base.py | `get_metadata(symbol, interval)` | Guarantees | -> `DataSetMetadata` | T: `get_dataset_metadata` (data), `describe_data_capabilities` (meta), `get_data_quality_report` (research), `build_model_dataset` (modeling, provenance note) |
| base.py | `get_temporal_contract(frame_kind="bars")` | What the provider can say about WHEN facts became knowable; base returns `price_contract` for bars and an UNSUPPORTED contract for everything else | -> `TemporalContract` | T: `describe_temporal_contract` (meta); `build_model_dataset` (modeling, `_provider_contract`) |
| base.py | `get_trades(symbol, start, end, limit)` | Tick tape (base raises NotImplementedError) | -> df[price,size,...] | T: `fetch_tick_tape` (data; **provider fixed to yfinance, so it always refuses**); portfolio `get_trade_profile` and `_fetch_ticks` callers with `source` |
| base.py | `get_quotes(symbol, start, end, limit)` | Top-of-book (base raises) | -> df[bid_price,bid_size,ask_price,ask_size] | T: `fetch_quote_panel` (data; **always refuses**); portfolio `_fetch_ticks` |
| base.py | `get_order_book(symbol, start, end, levels=5, limit)` | L2 depth in `ORDER_BOOK_COLUMNS` (base raises; Databento implements) | -> df | **PROG.** No tool. `Documentation/26_data.md` and `data/tools.py` header say a fetch tool was withheld because no provider served depth; one now does. |
| base.py | `get_order_events(symbol, start, end, limit)` | Market-by-order in `ORDER_EVENT_COLUMNS` (base raises; Databento implements) | -> df | **PROG.** No tool. |
| base.py | `get_point_in_time_records(symbols, frame_kind, fields, start, end)` | One row per VERSION of a fact with `entity`, `event_time`, `available_time` (base raises; Polygon implements `fundamentals`) | -> PIT df | T: **indirect only** via `build_model_dataset` (modeling) when the spec names a PIT feature and `provider='polygon'` (`modeling/dataset/pit_features.py:125`). No tool fetches and publishes an `event_panel` reference; `join_point_in_time`/`validate_pit_records` take records INLINE (5,000-row cap). |
| base.py | `ORDER_BOOK_COLUMNS`, `ORDER_EVENT_COLUMNS` | Canonical column contracts | tuples | INT (restated by `external.KIND_COLUMNS`) |

### `data/yfinance_provider.py`

| module | function/class | purpose | inputs -> outputs | exposed by |
|---|---|---|---|---|
| yfinance_provider.py | `YFinanceProvider.get_ohlcv` / `_fetch_ohlcv_uncached` | Session cache -> Parquet cache -> `yf.Ticker.history(auto_adjust=True)`; exclusive-end corrected; `audit.record_data_access` on every path | -> df | T: default provider for every fetching tool |
| yfinance_provider.py | `get_ohlcv_async`, `get_ticker_info`, `get_financial_ratios` (debtToEquity %->ratio, `implausible_value_warnings` logged), `get_metadata` (suffix -> IANA tz) | Provider contract | | T: as above |
| yfinance_provider.py | `_EXCHANGE_SUFFIX_TIMEZONES` | 19 Yahoo suffixes -> tz | dict | INT |

### `data/polygon_provider.py`

| module | function/class | purpose | inputs -> outputs | exposed by |
|---|---|---|---|---|
| polygon_provider.py | `_resolve_polygon_api_key(api_key)` | `SQT_POLYGON_API_KEY` via `load_env` | -> str | INT |
| polygon_provider.py | `_polygon_get(path, params, key)` | One GET; 404 -> DataNotFound, 401/403 -> NonRetryable, 429 -> APIError | -> json | INT |
| polygon_provider.py | `_normalize_ticker`, `_tick_range_ns`, `_require_tick_access` | URL-safe ticker; half-open ns range; re-raise 403 as a plan problem | | INT |
| polygon_provider.py | `_parse_ticks`, `_parse_aggs`, `_parse_ticker_info`, `_financial_value`, `_parse_financial_ratios` | Pure parsers (ns timestamps; ms for aggs; ratios derived from latest filing + market cap, `debt_to_equity` = liabilities/equity with a declared note) | | INT |
| polygon_provider.py | `_polygon_pages(path, params, key)` | Follow `next_url` | -> iterator | INT (**only the PIT path paginates; `get_ohlcv` is one page of 50,000 with a log-only warning**) |
| polygon_provider.py | `_parse_financials_records(results, entity, fields)` | Filings -> PIT rows; drops filings lacking `filing_date`, count kept in `frame.attrs` | -> df | INT |
| polygon_provider.py | `PolygonProvider.get_ohlcv` / `_get_ohlcv_uncached` | Aggregates with the shared two-tier cache and audit hook | -> df | T: when `source='polygon'` is selectable (see factory row) |
| polygon_provider.py | `get_ohlcv_async`, `get_ticker_info`, `get_financial_ratios`, `get_metadata`, `_fetch_ticker_details` | Contract | | T: `compare_data_sources` (meta), `build_model_dataset(provider='polygon')` |
| polygon_provider.py | `get_point_in_time_records(symbols, 'fundamentals', ['income_statement.revenues', ...], start, end)` | Quarterly filings, paginated, sorted by `available_time` | -> PIT df with `n_dropped_without_available_time` in attrs | T: indirect via `build_model_dataset` only |
| polygon_provider.py | `get_temporal_contract('fundamentals')` | Both timestamps present, `revisions='unknown'` until measured | -> contract | T: `describe_temporal_contract(source='polygon')` |
| polygon_provider.py | `get_trades`, `get_quotes` | v3 endpoints, one page <= 50k, ns timestamps | -> df | T: portfolio tick tools with `source='polygon'` |

### `data/bloomberg_provider.py`

| module | function/class | purpose | inputs -> outputs | exposed by |
|---|---|---|---|---|
| bloomberg_provider.py | `HAS_BLPAPI`, `_require_blpapi`, `_resolve_bloomberg_config(host, port)`, `_to_bloomberg_ticker`, `_bloomberg_timezone`, `_to_bbg_date` | Optional SDK guard; env config; "AAPL" -> "AAPL US Equity"; yellow key -> tz | | INT |
| bloomberg_provider.py | `_parse_historical_bars`, `_parse_ticker_info`, `_parse_financial_ratios` | Pure parsers; every rate field %->fraction; `market_cap` scale declared unknowable | | INT |
| bloomberg_provider.py | `_drain_historical_response`, `_drain_reference_response` | blpapi event loops | | INT |
| bloomberg_provider.py | `BloombergProvider.get_ohlcv` (daily/weekly/monthly only; **no session or Parquet cache**), `get_ohlcv_async`, `get_ticker_info`, `get_financial_ratios`, `_fetch_reference_fields`, `get_metadata` | Contract | | T: only where `source='bloomberg'` is selectable (meta describe/compare tools; modeling spec). No `SUPPORTED_INTERVALS`, no tick methods. |

### `data/databento_provider.py`

| module | function/class | purpose | inputs -> outputs | exposed by |
|---|---|---|---|---|
| databento_provider.py | `BAR_SCHEMAS` | `1s/1m/1h/1d` -> ohlcv schema | dict | PROG (not surfaced: no `SUPPORTED_INTERVALS`) |
| databento_provider.py | `_to_utc(value, end_of_day)` | Inclusive-end -> half-open UTC | | INT |
| databento_provider.py | `DatabentoProvider._get_client`, `_is_denial`, `_available_range`, `_bar_datasets`, `_range`, `_get_range`, `_fetch` | Lazy client (key from `DATABENTO_API_KEY` only); entitlement-denial memory; dataset preference EQUS.MINI -> XNAS.BASIC -> XNAS.ITCH; clamp to published range; daily finalization walk-back | | INT |
| databento_provider.py | `to_raw_symbol(symbol)` | `BRK.B`/`BRK-B` -> `BRKB` | staticmethod | PROG |
| databento_provider.py | `get_ohlcv(...)` | **Unadjusted** bars; tz-aware UTC index (every other provider strips tz); no cache; **no `audit.record_data_access`**; no `trim_to_inclusive_end` | -> df | T: `build_model_dataset(provider='databento')`, portfolio tick tools with `source='databento'` |
| databento_provider.py | `get_trades`, `get_quotes` | via `normalize_trades`/`normalize_quotes`, `limit` = head | -> df indexed by timestamp | T: portfolio `_tick_provider('databento')` |
| databento_provider.py | `get_order_book(symbol, start, end, levels<=10, limit)` | mbp-10 -> `normalize_book`; the first depth implementation | -> book df | **PROG** |
| databento_provider.py | `get_order_events(symbol, start, end, limit)` | mbo -> `normalize_mbo` | -> order-event df | **PROG** |
| databento_provider.py | `get_ticker_info` (stub), `get_financial_ratios` (refuses), `get_metadata` (adjusted=False, survivorship_free=True, tz UTC) | Contract | | T: `describe_data_capabilities(source='databento')` works because the constructor is lazy; but `_PROVIDER_CLASSES` in meta omits databento so a constructor failure would `KeyError` |

### `data/databento.py` (pure normalizers)

| module | function/class | purpose | inputs -> outputs | exposed by |
|---|---|---|---|---|
| databento.py | `FIXED_PRICE_SCALE`, `UNDEF_PRICE`, `UNDEF_ORDER_SIZE`, `UNDEF_TIMESTAMP`, `F_*` flag bits, `FLAG_MEANINGS`, `DATASET_*`, `CONSOLIDATED_START`, `SCHEMA_KINDS`, `PASSTHROUGH`, `PRICE_SCALES`, `TIMESTAMP_SOURCES` | Vendor wire facts | constants | `SCHEMA_KINDS` and `FLAG_MEANINGS` are read by no tool (vocabulary not askable) |
| databento.py | `looks_like_databento(columns)` | Recognise raw export spelling | -> bool | T: `prepare_vendor_extract` (data); `register_external_dataset` via `check_schema` hint |
| databento.py | `book_depth(columns)` | Complete `bid_px_NN` levels | -> int | T: `prepare_vendor_extract` |
| databento.py | `_decide_price_scale(frame, cols, requested)` | dtype-first fixed-point detection with a magnitude cross-check | -> (factor, note) | INT (also `DatabentoProvider._to_ohlcv`) |
| databento.py | `_mask_sentinels`, `_resolve_timestamp` | int64-max -> NaN BEFORE scaling; `ts_recv` vs `ts_event` choice reported | | INT |
| databento.py | `level_is_empty(frame, level)`, `split_empty_levels(empty, depth)` | Both-sides-empty rule; trailing vs inner empties | | T: `prepare_vendor_extract` |
| databento.py | `normalize_book(frame, price_scale, timestamp, levels, keep_empty_levels)` | mbp-10 -> `order_book_panel` (+ `*_count_*`, passthrough) | -> (df, notes) | T: `prepare_vendor_extract(kind='order_book_panel')`; `DatabentoProvider.get_order_book` (unexposed) |
| databento.py | `normalize_quotes`, `normalize_trades`, `normalize_mbo` | -> `quote_panel`, `tick_tape`, `order_event_panel` | -> (df, notes) | T: `prepare_vendor_extract`; provider methods |
| databento.py | `flag_warnings(flags)` | Vendor's own `F_MAYBE_BAD_BOOK`/`F_BAD_TS_RECV`/snapshot counts | -> notes | T: only inside the normalizers, so only through `prepare_vendor_extract`. **`validate_external_dataset` never reads a `flags` column on a registered book.** |

### `data/external.py`

| module | function/class | purpose | inputs -> outputs | exposed by |
|---|---|---|---|---|
| external.py | `FORMATS`, `KIND_COLUMNS`, `KIND_DESCRIPTIONS`, `DEFAULT_BATCH_ROWS`, `DEFAULT_SCAN_LIMIT` | Contract per external kind | constants | `KIND_DESCRIPTIONS` surfaced only inside refusal text |
| external.py | `_pyarrow_dataset`, `_infer_format`, `_suffix`, `_files` | Format detection (Parquet/CSV only, one compression suffix) | | INT |
| external.py | `resolve_path(path)` | Expand and require existence; deliberately NOT sandboxed to `SQT_RUNS_DIR` | -> Path | T: `register_external_dataset`, `prepare_vendor_extract` |
| external.py | `fingerprint(path)` | sha256 over (relative name, size, mtime) — not a content hash | -> hex | T: `register_external_dataset`, `describe_external_dataset` (`changed_since_registration`) |
| external.py | `total_bytes(path)` | | -> int | T: `describe_external_dataset` |
| external.py | `ExternalDataset` (frozen): `.scanner(columns, batch_rows)`, `.batches(columns, batch_rows)`, `.head(n, columns)` | Handle that stays on disk; column projection; bounded head | | T: `batches` -> `validate_external_dataset`, `prepare_vendor_extract`; `head` -> `describe_external_dataset` preview. **Column projection and any window other than the leading rows are reachable by no tool.** |
| external.py | `open_dataset(path, fmt)` | pyarrow dataset | | INT |
| external.py | `inspect(path, kind, fmt, known_rows, count_rows)` | Schema + stats from footers (CSV row count is a scan) | -> `ExternalDataset` | T: `register_external_dataset` (via `handoff.publish_external`), `prepare_vendor_extract` |
| external.py | `book_levels(columns)` | Complete `bid_price_N` levels, library spelling | -> int | T: `describe/validate_external_dataset`, `prepare_vendor_extract`; `get_order_book_metrics` (microstructure) |
| external.py | `required_columns(kind)`, `check_schema(kind, columns)` | Column contract; names the Databento normalizer in the refusal | -> problems | T: `register_external_dataset`, `validate_external_dataset`, `prepare_vendor_extract` |

### `data/external_validation.py`

| module | function/class | purpose | inputs -> outputs | exposed by |
|---|---|---|---|---|
| external_validation.py | `CROSSED_BLOCKING_FRACTION` (0.05) | Crossed-book blocking threshold | const | INT (not tunable) |
| external_validation.py | `ExternalValidationReport` (`.coverage()`) | Verdict + counts | dataclass | T: `validate_external_dataset` |
| external_validation.py | `_Checker` (`check_order`, `feed`, `_check_order_book_panel`, `_check_event_panel` (delegates to `validate_pit_frame`, publication-lag stats), `_check_tick_tape`, `_check_order_event_panel`, `_check_quote_panel`) | Per-kind batch checks | | INT |
| external_validation.py | `validate_external(handle, kind, scan_limit, batch_rows)` | Bounded scan, schema first | -> report | T: `validate_external_dataset` (`batch_rows` not exposed) |

### `data/_cache.py`

| module | function/class | purpose | inputs -> outputs | exposed by |
|---|---|---|---|---|
| _cache.py | `_session_cache_get/_set` | Locked TTLCache(100, 1h) | | INT |
| _cache.py | `_norm_date(d)` | Validated `YYYY-MM-DD` | | INT (also Polygon PIT bounds) |
| _cache.py | `is_intraday_interval(interval)` | sub-daily test | -> bool | INT |
| _cache.py | `inclusive_end_timestamp(end, interval)`, `trim_to_inclusive_end(df, end, interval)` | The inclusive-end contract, enforced by construction | | INT (yfinance, polygon, bloomberg; **not databento**) |
| _cache.py | `_norm_cache_bound`, `_normalize_ohlcv_index(df, interval)` | Intraday -> UTC-naive; daily -> local date | | INT |
| _cache.py | `_parquet_path`, `_safe_parquet_path(symbol, start, end, interval, provider)` | Containment-checked cache filename, format version `v3` | | INT |
| _cache.py | `_is_historical(end)`, `_write_parquet_atomic(path, df)` | Disk-cache eligibility; atomic write | | INT |
| _cache.py | `_CACHE_ROOT`, `_CACHE_FORMAT_VERSION` | `SQT_CACHE_DIR` | | T: `describe_data_capabilities.cache_dir` reports the path only. **No function lists, sizes, ages or evicts cache entries**; the module's own docstring says a symbol with a recent corporate action needs a manual bypass and offers no way to do it. |

### `data/temporal.py`, `data/bundle.py`

| module | function/class | purpose | inputs -> outputs | exposed by |
|---|---|---|---|---|
| temporal.py | `FRAME_KINDS`, `REVISION_ENCODINGS` | Vocabularies (`versioned/snapshot/none/unknown`) | tuples | INT; meanings not askable by a tool |
| temporal.py | `TemporalContract` (`.pit_safe`, `.reproduces_history`, `.why_not_pit_safe()`, `.caveats()`) | Self-report of event/available time and revision encoding | pydantic | T: `describe_temporal_contract` (meta), `infer_temporal_contract`, `build/describe/validate_data_bundle` (data) |
| temporal.py | `require_pit(contract, purpose)` | Refuse by name before any work | raises | PROG: `DataBundle.frame(require_pit=True)`, modeling `pit_features` only |
| temporal.py | `contract_for_frame(frame, source, frame_kind, entity_scoped)` | Infer from columns; `versioned` only when a fact has >1 `available_time` | -> contract | T: `infer_temporal_contract`; `build_data_bundle` (inference path) |
| temporal.py | `price_contract(source)` | Bars: knowable at close | -> contract | T: `describe_temporal_contract(frame_kind='bars')` |
| bundle.py | `DataBundle.add(frame_kind, frame, contract=None, source, entity_scoped)` | Pair frame with contract; refuses duplicates and unknown kinds | | T: `build_data_bundle` — **always the inference path**: the tool never passes a provider's own contract, so `revisions` is always `unknown`/`versioned` and never `none` for bars (the docstring says passing the provider's contract "is better whenever you have it"; the tool never has it). |
| bundle.py | `.kinds`, `.contract(kind)`, `.frame(kind, require_pit)`, `.describe()`, `.pit_safe`, `.reproduces_history`, `.warnings()` | Reading | | `describe` -> `build/describe_data_bundle`; `frame(require_pit=True)` PROG |
| bundle.py | `validate_bundle(bundle, require_pit=True)` | Verdict with blocking reasons | -> dict | T: `validate_data_bundle` |

### `data/comparison.py`, `data/ratios.py`, `data/continuous.py`

| module | function/class | purpose | inputs -> outputs | exposed by |
|---|---|---|---|---|
| comparison.py | `NOISE_THRESHOLD` (0.01), `SCALE_CONSISTENCY` (0.05) | Classification thresholds | consts | INT (not tunable from tools) |
| comparison.py | `_finite_pairs`, `classify_divergence(pairs, noise)` | `agree` / `scale` (constant ratio) / `definition` / `no_overlap` | -> dict | T: via `compare_ratio_sources` |
| comparison.py | `compare_ratio_sources(left, right, left_name, right_name, fields)` | Field-by-field, with worst examples and declared notes | -> report | T: `compare_ratio_frames` (data), `compare_data_sources` (meta) |
| ratios.py | `CANONICAL_UNITS`, `FIELD_DEFINITIONS` | The unit and formula of every ratio field | dicts | **PROG.** No tool can answer "what unit is `dividend_yield` in". |
| ratios.py | `percent_to_fraction(value)`, `ratio_field(ratios, field)` | Adapter helpers | | INT |
| ratios.py | `implausible_value_warnings(ratios)` | Fraction fields > 3.0 flagged, never corrected | -> [str] | T: `fetch_financial_ratios`, `validate_financial_ratios` (data) |
| continuous.py | `ROLL_RULES`, `ADJUSTMENTS` | Rule/adjustment descriptions | dicts | Only echoed inside result warnings; no offline listing |
| continuous.py | `build_continuous_futures(contracts, roll_rule, adjustment, days_before_expiry)` | Research series + tradeable contract map, monotone roll, backward adjustment | -> dict | T: `build_continuous_futures_series` (data) |
| continuous.py | `_parse`, `_choose_active`, `_roll_dates`, `_adjust` | | | INT |

---

## 2. Inventory: `audit/`

| module | function/class | purpose | inputs -> outputs | exposed by |
|---|---|---|---|---|
| models.py | `DecisionRecord` | One tool call: inputs, `data_sources`, hashes (`output_hash`, `output_hash_normalized`), provenance, chain fields | pydantic | T: `explain_decision` (meta); MCP-res `sqt://audit/{request_id}`; CLI `report` |
| models.py | `ReplayResult` | Replay verdict | dataclass | T: `replay_decision`; CLI `replay` |
| hashing.py | `hash_dataframe(df)` | Schema+values fingerprint, 16 hex | -> str | INT (every recorded bar fetch; modeling lineage) |
| hashing.py | `hash_payload(obj)`, `_canonical_default` | Canonical-JSON fingerprint | -> str | INT (chain, replay) |
| provenance.py | `_cpp_available`, `_git_sha`, `_package_version`, `_strategy_source_hash(model)` | Best-effort provenance | | INT; surfaced as `explain_decision` fields; model registry manifests |
| storage.py | `AuditStorageBackend` (Protocol), `LocalFilesystemBackend` | Pluggable I/O for the writer only | | PROG (injected via `AuditWriter(backend=)`; verify/retention/export read the filesystem directly, as the module admits) |
| context.py | `new_request_id()` | uuid4 hex | | INT — **never returned to the caller** |
| context.py | `RequestIdFilter`, `configure_logging(level, log_file)` | Request-id-correlated logging | -> Handler | PROG |
| context.py | `record_data_access(symbol, start, end, interval, source, content_hash)` | Append a data source to the open record | | INT: yfinance/polygon/bloomberg `get_ohlcv` only. **Not** databento; **not** any `get_trades`/`get_quotes`/`get_order_book`/PIT records. |
| paths.py | `_audit_enabled()`, `_audit_dir()` (XDG state dir; legacy `~/.cache` kept with a warning), `_iter_day_files`, `_acquire_lock`/`_release_lock`, `_GENESIS_HASH`, `_INDEX_FILENAME`, `_DAY_FILE_RE` | Location, discovery, advisory locking | | `_audit_dir` -> T: `verify_audit_integrity` (meta); rest INT |
| writer.py | `AuditWriter(audit_dir, backend).write(record)` and chain helpers (`_last_record_hash_in_file` fails closed on a corrupt tail, `_bootstrap_new_day`, `_chain_head_before`) | Hash-chained, fsync'd JSONL + chain index | -> Path | INT (every tool call) |
| dispatch.py | `_audit_fail_closed()` (`SQT_AUDIT_FAIL_CLOSED`), `_run_and_record(tool_name, fn, model)` | The recording core | -> result dict | INT: every dispatcher in all ten runtimes |
| verify.py | `verify_audit_log_integrity(path, expected_prev_hash)` | One day file | -> problems | T: `verify_audit_integrity(date)` (meta); CLI `verify --file` |
| verify.py | `verify_audit_trail_integrity(audit_dir)` | Chain index + every day file seeded from the index | -> problems | T: `verify_audit_integrity()` (meta); CLI `verify` |
| retention.py | `hold_day(date, audit_dir, reason)`, `release_hold`, `is_held` | Legal-hold sidecars | | **CLI only** (`sqt hold` / `release-hold`); `is_held` has no tool or CLI read |
| retention.py | `gc_candidates(audit_dir, retention_days)`, `gc(..., dry_run=True)` | Retention deletion (never automatic) | -> [dates] | **CLI only** (`sqt gc`, `--confirm`). The meta runtime states destructive retention is deliberately absent. |
| retention.py | `seal_day(date)` | chmod read-only (not WORM) | -> Path | CLI only (`sqt seal`) |
| export.py | `export_bundle(start, end, out_path, audit_dir)` | Zip of day files, chain index, manifest, standalone verifier, README | -> Path | T: `export_audit_bundle` (meta, write-once, contained); CLI `export`. **Checkpoint `.json/.sig` sidecars are not included**, and `_EXPORT_README` says signing is "planned but not yet implemented". |
| redaction.py | `_redact_fields()` (`SQT_AUDIT_REDACT_FIELDS`), `_placeholder_for` (salted via `SQT_AUDIT_REDACT_SALT`), `_redact`, `redact_text` | Field redaction of `input` and `error_message` | | INT (every record). Not queryable: no tool says which fields a record had redacted, though `replay` refuses such records. |
| signing.py | `HAS_CRYPTOGRAPHY`, `generate_keypair()` | Dev keypair | -> (priv, pub) | CLI only (`sqt keygen`) |
| signing.py | `checkpoint_and_sign(date, audit_dir, key_path, signer)` | Ed25519 checkpoint over final record hash + index hash | -> Path | CLI only (`sqt anchor`) |
| signing.py | `verify_checkpoint_signature(date, public_key_path, audit_dir)` | Public-key verify and re-derive | -> bool | T: `verify_audit_integrity(public_key_path)` (meta); CLI `verify --checkpoint` |
| signing.py | `_load_signer`, `_derive_checkpoint_content` | | | INT |
| replay.py | `_resolve_tool(tool_name)` | Look up in `agent.tools._TOOL_DISPATCH` + `MODELING_TOOL_DISPATCH` | | INT — **feature_lab tools not found** |
| replay.py | `normalize_identifiers(obj)`, `_has_volatile_identifiers`, `_redacted_input_fields` | `ds_`/`mdl_` id normalisation; redaction detection | | INT |
| replay.py | `verify_replay(record)` | Re-run; compare data hashes and output hash; failed originals are first-class | -> `ReplayResult` | T: `replay_decision` (meta); CLI `replay` |

---

## 3. Inventory: `mcp/`

| module | function/class | purpose | inputs -> outputs | exposed by |
|---|---|---|---|---|
| schemas.py | `dereference(schema)`, `CircularSchemaError` | Inline `$ref`/`$defs` | -> schema | INT (catalog, every advertised schema) |
| schemas.py | `contains_ref(schema)` | | -> bool | tests only |
| schemas.py | `property_names(schema)` | Recursive property scan -> `openWorldHint` | -> [str] | INT (catalog) |
| schemas.py | `schema_bytes(schema)` | Serialized size | -> int | INT -> T: `estimate_tool_cost` (meta) via `ToolEntry.cost_bytes`; MCP-res `sqt://catalog/categories` |
| progress.py | `progress_token(ctx)`, `report_liveness(ctx, label, interval)`, `_pulse` | Elapsed-seconds heartbeat, only when the client sent a `progressToken`, no fabricated total | | MCP-srv (`call_tool`) |
| prompts.py | `PromptArg`, `WorkflowPrompt.build(values)`, `_get` | Prompt template machinery | | MCP-prompt |
| prompts.py | `PROMPTS`: `screen_and_backtest`, `factor_research_note`, `pair_trade_study`, `build_and_validate_model`, `risk_review`; `BY_NAME`, `names()` | The five workflow prompts | | MCP-prompt (`prompts/list`, `prompts/get`) |
| prompts.py | `required_categories(prompt)`, `_PROMPT_CATEGORIES` | Warn when a prompt's categories are not served | | MCP-srv `get_prompt`. `build_and_validate_model` declares `("modeling",)` but its step 4 (feature report) lives in `feature_lab`. No prompt covers the data runtime, external datasets, PIT joins or the audit/provenance workflow. |
| http.py | `default_allowed_hosts(host, port)`, `MCPEndpoint`, `BearerAuth` (constant-time), `build_app(config, server, tool_count)` (`/healthz`, `/mcp`), `serve_http` | Streamable-HTTP transport, DNS-rebinding protection, bearer gate | | MCP-srv (`sqt-mcp --transport http`) |
| resources.py | `CATALOG_FEATURES/CAPABILITIES/CATEGORIES`, `TEMPLATES` (result, artifact, model, dataset, audit) | Resource surface | | MCP-res list |
| resources.py | `_summarize(payload, limit)`, `store_result(tool, payload, limit)` | Oversized results persisted under `SQT_RUNS_DIR/mcpres-<sha>`; `_truncated` names omitted fields | -> (payload, uri) | MCP-srv (`call_tool`, `--inline-limit`) |
| resources.py | `load_result(result_id)`, `read(uri)`, `_require`, `_read_catalog`, `_read_model`, `_read_dataset`, `_read_audit` (`cli.find_record`) | URI resolution, sandboxed via `modeling.artifacts.run_dir` | -> dict | MCP-res |
| resources.py | `_read_artifact(run_id, name)` | Whole Parquet artifact as JSON records | -> dict | MCP-res `sqt://artifact/{run}/{name}` — **unbounded**, unlike `describe_artifact`/`read_reference` which cap rows |
| catalog.py | `ToolEntry` (`.idempotent`, `.cost_bytes(include_output_schema)`) | One exposed tool | dataclass | T: `describe_tool`, `estimate_tool_cost` (meta) |
| catalog.py | `_output_schema`, `_reads_market_data`, `_entries_for_registry` | Derive output schema, `openWorldHint`, `persists_artifact` (`*_uri` result field) | | INT |
| catalog.py | `build_catalog()` | All 211 tools keyed by name, three registries | -> dict | T: `describe_tool`, `estimate_tool_cost`; MCP-srv; MCP-res categories |
| catalog.py | `select(catalog, categories)`, `select_runtimes(catalog, runtimes)`, `categories_for_runtimes(runtimes)` | Scoping | | MCP-srv / config; `estimate_tool_cost` |
| catalog.py | `DETAIL_MODES`, `DEFAULT_DETAIL_BUDGET` (32,768), `SCHEMA_FETCH_TOOL`, `thin_description`, `thin_schema`, `plan_detail(entries, mode, budget)` | `--tool-detail auto/thin`; thins most expensive first; never thins `describe_tool` | | MCP-srv |
| catalog.py | `_split_runtimes(raw)` | Parse `--runtime` | | **DEAD** — duplicate of `config._split_runtimes`, which is the one called |
| catalog.py | `runtime_costs`, `category_costs` | Budget tables | | MCP `--print-budget`; MCP-res categories |
| catalog.py | `dispatch_for(entry)` | The owning runtime's dispatcher | | MCP-srv `call_tool` |
| server.py | `LONG_RUNNING` (`scan_pairs`, `run_backtest_optimization`), `SCALES_WITH_INPUT`, `_annotations` (read_only=True for all, idempotent from `persists_artifact`, open_world from schema), `_to_mcp_tool` | Advertisement | | MCP-srv |
| server.py | `StandardToolsServer` (`list_tools`, `_refusal` (unknown / other runtime / long-running / other category), `call_tool` (worker thread + heartbeat + inline-limit), `list_resources`, `list_resource_templates`, `read_resource`, `list_prompts`, `get_prompt`, `context_bytes`) | Protocol handlers | | MCP-srv. **Docstring claims a request-id context is set per call; nothing does.** |
| server.py | `_error`, `build_server(config)`, `_serve_stdio`, `_budget_table`, `print_budget`, `main` | Entry | | `sqt-mcp` |
| config.py | `DEFAULT_INLINE_LIMIT` (4096), `DEFAULT_HEARTBEAT_SECONDS`, `CONTEXT_CEILING_BYTES=None`, `LOOPBACK_HOSTS`, `TOKEN_ENV_VAR`, `is_loopback`, `ServerConfig` | Configuration | | MCP |
| config.py | `_split_runtimes`, `_split_categories`, `build_parser`, `_resolve_dir` (write-probe), `_resolve_transport` (fail-fast token/host rules), `_resolve_scope` (runtime x category nesting), `_owner_of_category`, `_owners_of`, `resolve(argv)`, `check_context_budget`, `report` | Fail-fast startup | | `sqt-mcp` |

**MCP resources served:** 3 static (`sqt://catalog/{categories,features,capabilities}`) + 5 templates (`sqt://result/{id}`, `sqt://artifact/{run_id}/{name}`, `sqt://model/{id}`, `sqt://dataset/{id}`, `sqt://audit/{request_id}`). **Prompts served:** 5. No resource exposes the audit trail's day list, a run's artifact list, an external registration, or a data bundle manifest.

---

## 4. Inventory: top-level modules

| module | function/class | purpose | inputs -> outputs | exposed by |
|---|---|---|---|---|
| `__init__.py` | `__version__`, `DISABLE_NATIVE_ENV`, `native_disabled()` | `SQT_DISABLE_NATIVE=1` makes `_sqt_core` unimportable | | INT (`explain_decision.package_version`, `cpp_available`) |
| `_compat.py` | `HAS_POLARS`, `is_series_like`, `is_dataframe_like`, `is_empty`, `to_clean_numpy`, `require_polars` | Optional Polars detection | | INT (`validation.py`, `analysis.hurst`) |
| `_jsonsafe.py` | `sanitize_for_json(obj)` | Non-finite -> None, recursively (tuples/sets/numpy included) | | INT (every dispatch; MCP resources) |
| `_resampling.py` | `block_indices(n, block_size, rng, target)` | Moving-block bootstrap indices by broadcast, bit-identical to the old loop | -> ndarray | INT (`get_bootstrap_interval`, Monte Carlo, robustness, overfitting) |
| `_runspath.py` | `RUNS_DIR_ENV`, `runs_dir()`, `validate_identifier(value, field)`, `resolve_within_runs_dir(path)` | The one path-traversal guard | | INT (every publish / artifact write) |
| `_special.py` | `norm_cdf`, `norm_pdf`, `norm_cdf_array`, `norm_ppf` (raises at 0/1), `betacf`, `betainc`, `f_sf` (returns 1.0 on degenerate dof) | scipy-free special functions, one copy | | INT |
| `artifact_store.py` | `HASH_HEX_CHARS`, `validate_key(key)`, `hash_stream`, `hash_bytes`, `write_bytes_atomically(path, data)` | Key grammar `<run_id>/<filename>`; atomic write | | INT (backtest/modeling artifacts) |
| `artifact_store.py` | `ArtifactStore` (Protocol), `LocalArtifactStore(root).put/get/exists/list/hash/uri` | Local store | | T: `inspect_model` (modeling) via `registry.package.verify_model_package`. **`.list(prefix)` is reachable by no tool: nothing enumerates what a run directory holds.** |
| `artifact_store.py` | `fsspec_available`, `require_fsspec`, `FsspecArtifactStore(url)`, `store_from_url(url)` | Object-storage target for mirroring a verified package | | **PROG** (no tool, no CLI) |
| `cli.py` | `_iter_records(audit_dir)`, `find_record(request_id, audit_dir)` | Scan day files; lookup by id | -> record | T: `explain_decision`, `replay_decision`, `compare_decisions` (meta); MCP-res `sqt://audit`. `_iter_records` (the scan) has no tool: nothing lists or filters records. |
| `cli.py` | `cmd_report`, `_format_replay`, `_replay_exit_code`, `_replay`, `cmd_replay` | `sqt report` / `sqt replay` (exit 0/1/2) | | CLI (`replay` also T: `replay_decision` via `verify_replay`) |
| `cli.py` | `cmd_compare(a, b)` | Text diff of two records | -> str | T: `compare_decisions` (meta) embeds it; CLI |
| `cli.py` | `cmd_verify(file, audit_dir)`, `_format_verify` | | | CLI (T equivalent: `verify_audit_integrity`) |
| `cli.py` | `cmd_hold`, `cmd_release_hold`, `cmd_gc(confirm, retention_days)`, `cmd_seal` | Retention operations | | **CLI only** |
| `cli.py` | `cmd_export` | | | CLI (T: `export_audit_bundle`) |
| `cli.py` | `cmd_keygen(out_dir)`, `cmd_anchor(date, key_path)`, `cmd_verify_checkpoint(date, pubkey)` | Signing | | `keygen`/`anchor` **CLI only**; verify also T |
| `cli.py` | `main(argv)` | `sqt` entry | -> exit code | CLI |
| `config.py` | `load_env(dotenv_path)` | Idempotent `.env` load, never overrides real env | -> bool | INT |
| `constants.py` | `TRADING_DAYS_PER_YEAR` (252), `EULER_MASCHERONI` | | | INT |
| `error.py` | `QuantError`, `DataProviderError`, `DataNotFoundError`, `InvalidSymbolError`, `APIError`, `NonRetryableAPIError`, `CalculationError`, `ValidationError(QuantError, ValueError)`, `BacktestError`, `AuditIntegrityError` | Exception hierarchy | | INT (MCP `call_tool` returns `QuantError` text verbatim) |
| `numeric_contract.py` | `require_finite_series`, `require_finite_series_frame`, `require_positive_price_series`, `require_positive_start_level`, `require_aligned`, `require_positive_int`, `require_finite_scalar`, `require_periods_per_year`, `require_finite_covariance` | The numerical input contract, memo-aware | raise or return | INT (boundary of metrics/backtest/portfolio/delta_one) |
| `validation.py` | `memoized_input_checks()`, `_memo_seen`, `_memo_record` | Skip repeat checks in a batch scope | ctx manager | INT (`build_model_dataset`) |
| `validation.py` | `validate_dataframe(required_columns)` | Decorator | | **no application** (documented as kept on purpose) |
| `validation.py` | `validate_series(allow_empty, allow_nan)`, `_check_series_values`, `require_finite_array`, `last_finite(series, name, minimum)` | Series contract; last finite reading with an actionable error | | INT (indicators, metrics, research/portfolio tools) |

---

## 5. Exposure counts

Counting the rows above at the granularity of a public function, method, class or load-bearing constant group (private helpers folded into the public thing that wraps them):

| package | items inventoried | tool-reachable (direct or indirect) | INT plumbing (reached by every call, no decision) | CLI / MCP only | PROG / DEAD (not reachable) |
|---|---:|---:|---:|---:|---:|
| `data/` | 118 | 52 | 52 | 0 | 14 |
| `audit/` | 46 | 12 | 22 | 9 | 3 |
| `mcp/` | 58 | 6 | 8 | 43 | 1 |
| top-level | 71 | 9 | 47 | 12 | 3 |
| **total** | **293** | **79** | **129** | **64** | **21** |

The 21 not-reachable items that carry real capability (the rest are constants): `DataProvider.get_order_book`, `get_order_events` (Databento implementations), `get_point_in_time_records` as a standalone fetch, `DatabentoProvider.to_raw_symbol`, `BAR_SCHEMAS`, `SCHEMA_KINDS`, `FLAG_MEANINGS`, `CANONICAL_UNITS`, `FIELD_DEFINITIONS`, `REVISION_ENCODINGS` meanings, `ExternalDataset.scanner/batches(columns=...)` projection, `DataBundle.frame(require_pit=True)`, `temporal.require_pit`, `_cache` inspection (no function exists), `AuditStorageBackend`, `configure_logging`, `cli._iter_records` as a query, `FsspecArtifactStore`/`store_from_url`, `LocalArtifactStore.list`, `catalog._split_runtimes` (dead), `schemas.contains_ref` (tests only).

CLI-only capability (9 in audit + the `sqt` commands): `hold_day`, `release_hold`, `is_held`, `gc_candidates`, `gc`, `seal_day`, `generate_keypair`, `checkpoint_and_sign`, `cmd_report`. The meta runtime says destructive retention is absent on purpose; the read-only halves (`is_held`, `gc_candidates` dry-run, "is this day anchored") are absent too, and those are decisions.

---

## 6. Unexposed capability, grouped by theme

### Theme A — Depth and order-flow FETCHING (Databento)
`DatabentoProvider.get_order_book` (mbp-10, up to 10 levels, `normalize_book` with sentinel masking and trailing-empty-level drop) and `get_order_events` (mbo, `normalize_mbo`) are implemented, tested against the column contracts `analysis/order_book.py` and `analysis/order_events.py` read, and reachable by nothing. The data runtime's own header gives the historical reason ("a fetch tool here would have to answer for every provider") and then the same runtime already ships `fetch_tick_tape`, which refuses on every provider but two. The documented alternative (`register_external_dataset`) requires the caller to have pulled the book themselves, outside the audit trail.

### Theme B — Point-in-time records as a publishable frame
`PolygonProvider.get_point_in_time_records` produces exactly the `event_panel` schema (`entity`, `event_time`, `available_time`, fields) that `join_point_in_time`, `validate_pit_records` and `validate_external_dataset(kind='event_panel')` consume, but it can only be reached from inside `build_model_dataset`. There is no way to fetch a filing history, publish it as an `event_panel` reference, look at its publication lags, and then join it — the join tool takes at most 5,000 inline rows.

### Theme C — Provider selection and provider-aware provenance
Every data-runtime tool is yfinance. `describe_data_capabilities` can describe polygon/bloomberg/databento, but `fetch_ohlcv`, `fetch_tick_tape`, `fetch_quote_panel`, `get_dataset_metadata` cannot then use them. `build_data_bundle` never receives a provider contract (always inference). Databento bars carry no `record_data_access` entry, so replay cannot see them.

### Theme D — Decision-record discovery
`request_id` is never returned. `cli._iter_records` is the only scan and it is unfiltered. An agent cannot ask "which calls in this session read AAPL", "what did the last `run_backtest_compact` cost", "which records are unreplayable because of redaction", or "which days are unanchored". `is_held`, `gc_candidates` (dry run) and checkpoint presence are all read-only and all invisible.

### Theme E — Vocabulary that decides numbers but cannot be asked for
`CANONICAL_UNITS` / `FIELD_DEFINITIONS` (what `debt_to_equity` means here), `REVISION_ENCODINGS` (what `snapshot` implies), `ROLL_RULES` / `ADJUSTMENTS` (what a ratio-adjusted series preserves), `SCHEMA_KINDS` (which Databento schema yields which kind), `FLAG_MEANINGS`, `KIND_DESCRIPTIONS`. Every one is echoed in a warning after the fact; none is askable before the call. `list_strategies` and `list_stress_scenarios` are the precedent the meta runtime already set for exactly this.

### Theme F — External datasets: reading a window, not just the head
`ExternalDataset.head(n, columns)` and `.batches(columns=...)` support column projection; `describe_external_dataset` exposes only leading rows of every column. There is no bounded "rows around timestamp T, these columns" read, and `validate_external_dataset` ignores the vendor `flags` column that `flag_warnings` interprets.

### Theme G — Cache state
The Parquet cache can silently serve a pre-split adjusted history (`_cache.py` says so). Nothing can list a symbol's cache entries, their format version, age, or whether a live fetch would now hash differently. `replay_decision` answers a related question only for a past call.

### Theme H — Artifact store and package mirroring
`LocalArtifactStore.list(prefix)` would answer "what did run X leave behind"; `FsspecArtifactStore` / `store_from_url` would let a verified model package be mirrored to object storage. Neither has a tool or CLI.

---

## 7. Proposed tools

Ordered by expected value. Effort: S (< 1 day, existing functions composed), M (1-3 days, a new result model and tests), L (> 3 days, contract work).

### 7.1 `list_decisions` — meta (`provenance`) — **S**
- **Inputs:** `tool_name?`, `symbol?`, `start_date?`, `end_date?` (day files), `status?` (`ok`/`error`), `runtime?`, `limit=50`, `newest_first=true`.
- **Outputs:** rows of `{request_id, timestamp_utc, tool_name, status, duration_ms, n_data_sources, symbols, output_hash, redacted_fields, replayable}` plus `n_matched`, `truncated`, `days_scanned`.
- **Backing:** `cli._iter_records`, `paths._iter_day_files`, `replay._redacted_input_fields`, `catalog.build_catalog` (runtime of each tool).
- **Why a decision:** every other provenance tool needs a `request_id` the agent has never been given. Choosing WHICH past call to explain, replay or compare is the decision; today it cannot be made from inside a session at all. `replayable=False` (redacted input, or a feature_lab record — see 8.4) tells the agent not to spend a replay on it.
- **Warnings the tool must state:** the scan is bounded by `limit` and by the day range, so absence from the result is not absence from the log; records are read as data (an `input` field can contain anything a caller typed); `symbols` comes from `data_sources`, which only bar fetches populate, so a call that read Databento bars, ticks, quotes or PIT records shows none.
- Also fixes finding 2 cheaply if paired with returning `request_id` in every dispatch result (`_run_and_record` already has it; `agent/runtimes/__init__.py` docstring says results cross by `request_id`, which is currently impossible).

### 7.2 `fetch_order_book` — data — **M**
- **Inputs:** `symbol`, `start_date`, `end_date`, `levels=5`, `limit`, `run_id`, `name`, `source='databento'` (the only implementer; refused by name for any other value using the same `_overrides` probe `describe_data_capabilities` uses).
- **Outputs:** `FetchResult` of kind `order_book_panel` (`handoff.KINDS` already has it) plus `levels_kept`, `n_snapshots`, `notes` (price-scale and timestamp choice from `normalize_book`), `flag_warnings` counts, `truncated`.
- **Backing:** `DatabentoProvider.get_order_book` -> `normalize_book` -> `flag_warnings`; `external.book_levels`; `handoff.publish`. Add `audit.record_data_access` (see 8.2) so the fetch appears in the decision record.
- **Why a decision:** whether the book is deep enough to run `depth_slope` at all (one level -> touch only), and whether the venue flagged the book as inconsistent, are things the agent decides on before spending a `get_order_book_metrics` call; today it cannot obtain a book inside the audit trail at all.
- **Warnings:** DEPTH IS AGGREGATED — queue position is not in it (the base docstring's wording); `limit` truncates, never samples, so every intensity understates; `timestamp` came from `ts_recv` unless asked otherwise, and a latency study wants `ts_event`; MBO is a different call (`fetch_order_events`); the published copy is a second materialisation, so for a session of depth use `prepare_vendor_extract` + `register_external_dataset` instead (the tool should refuse above a row threshold and say so).
- **Instead of a new tool:** could be a `kind` parameter on `fetch_tick_tape` (`trades|quotes|book|orders`) — but the result shapes and warnings differ enough that four thin tools read better than one polymorphic one, which is the standard the runtime already applied to tape vs quotes.

### 7.3 `fetch_order_events` — data — **M**
- Same shape as 7.2 for `get_order_events` -> `order_event_panel`. Warning set adds: one record per order event, so an active name is two orders of magnitude larger than the same window of depth; `limit` here means the FIRST N events, which biases every lifetime and cancellation-rate measure toward the open; `action` letters are the venue's own (`A/C/M/F/T/R`) and `analysis/order_events.ACTION_MEANINGS` is the dictionary.

### 7.4 `fetch_point_in_time_records` — data — **M**
- **Inputs:** `symbols`, `frame_kind='fundamentals'`, `fields` (`<statement>.<key>` paths), `start_date`, `end_date`, `run_id`, `name`, `source='polygon'`.
- **Outputs:** `FetchResult` of kind `event_panel` plus `n_records`, `n_entities`, `n_dropped_without_available_time` (from `frame.attrs`), publication-lag stats (mean/min/max days — the same arithmetic `external_validation._check_event_panel` does), `observed_revisions` (facts with >1 `available_time`, via `contract_for_frame`), and the provider's `TemporalContract` caveats.
- **Backing:** `provider.get_temporal_contract(frame_kind)` -> `temporal.require_pit` (refuse before fetching) -> `provider.get_point_in_time_records` -> `modeling.dataset.point_in_time.validate_pit_frame` -> `contract_for_frame` -> `handoff.publish`.
- **Why a decision:** whether to build a dataset on this source at all. `describe_temporal_contract` says Polygon's `revisions` is `unknown` "until `observed_revisions` on a pulled history upgrades it" — and there is no tool that pulls a history. This is the tool that upgrades it, and it also feeds `join_point_in_time` by reference instead of 5,000 inline rows.
- **Warnings:** a zero lag on every row means the extract copied one column into the other; `revisions` stays `unknown` unless a second version was actually observed, and `unknown` is treated as `snapshot`; `start_date`/`end_date` bound EVENT time, so widen `start_date` by the staleness the join will accept; Polygon paginates here, so a wide universe is many requests against a rate-limited plan.

### 7.5 `describe_audit_trail` — meta (`provenance`) — **S**
- **Inputs:** none (optional `date`).
- **Outputs:** `audit_dir`, `enabled`, `fail_closed`, `redacted_fields` (names only), `n_days`, `first_day`, `last_day`, per-day `{date, records, held, hold_reason, sealed (read-only bit), checkpoint_signed, in_chain_index}`, `retention_days` (env), `gc_candidates` (dry run), `unanchored_days_past_retention`, `chain_index_present`.
- **Backing:** `paths._audit_dir/_audit_enabled/_iter_day_files/_INDEX_FILENAME`, `retention.is_held/gc_candidates` (dry run only), `dispatch._audit_fail_closed`, `redaction._redact_fields`, checkpoint sidecar existence (`<date>.checkpoint.json/.sig`), `os.stat` mode for sealed.
- **Why a decision:** `gc.__doc__` itself says "export a bundle for anything you may need to produce later BEFORE running gc" — that decision needs the list of days that are past retention and not held, which only this tool would give. It also answers "is the trail even being written" (`SQT_AUDIT_ENABLED=0`) before an agent relies on `explain_decision`.
- **Warnings:** read-only by construction (no hold/gc/seal — those stay CLI, per the runtime's stated policy); `sealed` is a chmod bit, not WORM; `gc_candidates` is what `sqt gc` WOULD delete, computed against today's clock; a signed checkpoint proves the day as of signing, not now (`verify_audit_integrity` with a key re-derives).

### 7.6 `describe_ratio_definitions` — data (or meta `discovery`) — **S**
- **Inputs:** `fields?`, `provider?`.
- **Outputs:** per field `{canonical_unit, definition, provider_conversion, provider_definition_note}`, drawn from `ratios.CANONICAL_UNITS`, `ratios.FIELD_DEFINITIONS`, and the per-provider adapter facts already written in code comments (yfinance `debtToEquity` is a percentage; Polygon `debt_to_equity` is liabilities/equity; Bloomberg `CUR_MKT_CAP` scale is terminal-dependent; Polygon `forward_pe`/`dividend_yield` always None).
- **Why a decision:** `run_screener` thresholds (`debt_equity_max=2.0`) are numbers in a unit; the module docstring records that the wrong unit admitted "essentially the entire universe". Offline, zero cost, same shape as `list_strategies`.
- **Warnings:** a declared definition difference is not convertible; `compare_data_sources` measures what this declares.

### 7.7 `read_external_window` — data — **S/M**
- **Inputs:** `ref`, `columns?`, `start?`/`end?` (timestamp bounds) or `head`/`tail`, `max_rows=256`.
- **Outputs:** bounded rows, `rows_scanned`, `truncated`, `changed_since_registration`.
- **Backing:** `ExternalDataset.scanner(columns=...)` with a pyarrow filter on the time column, `handoff.describe` fingerprint check.
- **Why a decision:** `validate_external_dataset` reports "3 crossed books in 9 million rows" and the agent cannot look at those three. `describe_external_dataset` gives leading rows of sixty columns; the decision "is the bid/ask transposition real or a locked market" needs the touch columns around the flagged time.
- **Warnings:** bounded; reads the bytes on disk NOW (fingerprint may have moved); a filter on an unsorted file scans the whole file up to the cap.

### 7.8 `check_cache_freshness` — meta (`discovery`) or data — **M**
- **Inputs:** `symbol`, `start_date`, `end_date`, `interval='1d'`, `refetch=false`.
- **Outputs:** `cache_path`, `present`, `format_version`, `written_at`, `cached_hash`; with `refetch=true`: `live_hash`, `matches`, `n_rows_changed`, first differing date.
- **Backing:** `_cache._safe_parquet_path`, `_CACHE_FORMAT_VERSION`, `hash_dataframe`, provider `_fetch_ohlcv_uncached` with the disk read bypassed (needs a small `bypass_cache` flag on the providers, or deleting-then-refetching under a lock).
- **Why a decision:** `_cache.py` documents that adjusted history goes stale after a split and the only remedy is manual eviction; an agent about to backtest a name with a recent action has to decide whether to trust the cache, and today it cannot even see that one exists.
- **Warnings:** a hash mismatch says the provider restated values, not which side is right; the disk cache is shared across sessions and providers keyed by name; `refetch=true` spends a network call and (on Polygon free tier) request budget.

### 7.9 `list_run_artifacts` — meta (`discovery`) — **S**
- **Inputs:** `run_id?` (prefix), `kind?`, `limit`.
- **Outputs:** keys, sizes, published kind and producer from the handoff sidecar, `sqt://` reference where one exists.
- **Backing:** `artifact_store.LocalArtifactStore.list`, `handoff._read_sidecar`, `_runspath.runs_dir`.
- **Why a decision:** a second agent, or the same one after a context reset, has to choose what to build on; `describe_reference` and `describe_artifact` require the name already known. This is the discovery tool for the interconnect the runtimes docstring describes.
- **Warnings:** listing is by directory, so an artifact written outside `publish` has no kind; an `mcpres-*` run is a stored MCP result, not a research artifact.

### 7.10 `explain_vendor_extract` (rename/extend) — data — **S**
Not a new tool: `prepare_vendor_extract(dry_run=true)` already reports scale and timestamp judgements. Two additions make it the pre-flight it is meant to be: report `flag_warnings` counts over the sampled batch, and report `SCHEMA_KINDS` when the columns identify a Databento schema (`mbp-10` -> `order_book_panel`), so the agent does not have to guess `kind`.

### 7.11 MCP prompts — **S** each
Three workflows whose ORDER is the value and which no prompt covers: `data_provenance_check` (describe_data_capabilities -> describe_temporal_contract -> get_dataset_metadata -> build/validate_data_bundle before any model), `microstructure_from_extract` (prepare_vendor_extract dry run -> convert -> register -> validate -> get_order_book_metrics), and `reproduce_decision` (list_decisions -> explain_decision -> replay_decision -> compare_decisions -> export_audit_bundle). Each should declare its categories so `get_prompt` can warn.

### Existing tools that should gain a parameter instead
| tool | parameter | why |
|---|---|---|
| `fetch_ohlcv`, `fetch_ohlcv_panel`, `fetch_returns_panel`, `fetch_tick_tape`, `fetch_quote_panel`, `get_dataset_metadata`, `fetch_financial_ratios` (data) | `source: str = "yfinance"` | The single biggest gap in this slice; the portfolio runtime already has the `_tick_provider(source)` pattern, and `describe_data_capabilities` already names the sources. Without it the two tick tools are unusable. |
| `describe_data_capabilities` (meta) | result fields `order_book`, `order_events`, `point_in_time_records` (with the frame kinds served), `supported_intervals` for Bloomberg/Databento | Probe with the same `_overrides` test it already uses for `get_trades`; add `SUPPORTED_INTERVALS` to `BloombergProvider` (`_PERIODICITY`) and `DatabentoProvider` (`BAR_SCHEMAS`). Add `'databento'` to `_PROVIDER_CLASSES` and to the `source` description on `DataCapabilitiesInput`/`TemporalContractInput`. |
| `build_data_bundle` (data) | `source` per frame may name a PROVIDER, in which case pass `provider.get_temporal_contract(frame_kind)` instead of inferring | The bundle docstring asks for exactly this and the tool never does it. |
| `validate_external_dataset` (data) | read `flags` when present and fold `flag_warnings` into `warnings`; expose `batch_rows` | The vendor's own bad-book flag is the one check the library cannot invent. |
| `export_audit_bundle` (meta) and `export_bundle` | `include_checkpoints=true` | Without the `.checkpoint.json/.sig` sidecars an exported bundle cannot be signature-verified, which contradicts the purpose of anchoring. Also fix the README text. |
| `describe_external_dataset` (data) | `columns`, `tail` | Cheap, uses `ExternalDataset.head(columns=...)`; halfway to 7.7. |
| `replay_decision`, `validate_tool_call` (meta) | none; fix `_resolve_tool` and `every = {...}` to include `FEATURE_TOOL_DISPATCH` | Finding 5. |
| every dispatcher | return `request_id` alongside the result (or in a `_provenance` key) | Finding 2. `_run_and_record` already has it. |

---

## 8. Dead code, duplicates, and docstrings that promise more than the code

1. **`mcp/catalog._split_runtimes`** is dead: `mcp/config.py` defines and calls its own identical copy (`config.py:508`). Delete one.
2. **`DatabentoProvider` fetches are never recorded** (`data/databento_provider.py` has no `audit.record_data_access`), while the audit package docstring says every call "captur[es] the market data it pulled (with content hashes)". Also: no `trim_to_inclusive_end`, tz-aware UTC index where every other provider normalises to tz-naive (`_normalize_ohlcv_index`), and no session/disk cache. `_cache.py`'s docstring says the cache was "extracted so BloombergProvider and PolygonProvider can reuse" it; **Bloomberg imports only `trim_to_inclusive_end` and has no cache either.**
3. **Stale refusal text.** `DataProvider.get_trades` / `get_quotes` say "Only PolygonProvider does" — Databento does too. `Documentation/26_data.md` ("no shipped provider serves depth") and the `data/tools.py` header contradict `databento_provider.py`'s own header ("This is the provider that serves one"). `audit/export._EXPORT_README` says "a signed-checkpoint feature is planned but not yet implemented"; `audit/signing.py` implements it and `sqt anchor` ships.
4. **`audit.replay._resolve_tool` and `meta.validate_tool_call` cannot see `feature_lab`** (`agent.tools._TOOL_DISPATCH` is built from eight runtimes; `FEATURE_TOOL_DISPATCH` lives in `modeling/agent/feature_tools.py`). Nine tools produce records that are recorded and unreplayable; the replay docstring's claim to "cover BOTH agent surfaces" predates the third.
5. **`mcp/server.py` header:** "The server sets a request-id context per call so a record ties back to the client conversation" — `call_tool` sets nothing; `_run_and_record` mints a fresh id that the client never learns.
6. **`mcp/resources._read_artifact` is unbounded** (`frame.reset_index().to_dict(orient="records")` for the whole file) in a module whose header is about keeping bulk values out of the context; `describe_artifact` and `read_reference` cap at 64 rows.
7. **`build_data_bundle` always infers** (`DataBundle.add(... source=entry.source)` with no contract), so `describe_data_bundle` reports `revisions='unknown'` for bars a provider would declare `none`; the bundle docstring says the provider's contract "is better whenever you have it".
8. **`meta._PROVIDER_CLASSES` omits `databento`**; `DataCapabilitiesInput.source` and `TemporalContractInput.source` descriptions list three providers although the factory accepts four.
9. **`PolygonProvider.get_ohlcv` truncates silently at 50,000 bars** (log warning only); `_polygon_pages` exists but only the PIT path uses it. An intraday `fetch_ohlcv` over a long range publishes a shorter panel with no warning in the result.
10. **`flag_warnings` is computed and then dropped on the fetch path**: `DatabentoProvider.get_order_book/get_order_events` log WARNING notes and discard them; `prepare_vendor_extract` keeps them; `validate_external_dataset` never computes them.
11. **Two spellings of "how many levels":** `databento.book_depth` (vendor `bid_px_NN`) and `external.book_levels` (library `bid_price_N`). Deliberate, but both are named as if they were the same thing; a docstring cross-reference would help.
12. **`prompts._PROMPT_CATEGORIES["build_and_validate_model"] = ("modeling",)`** while step 4 of the prompt calls for the feature report, which is a `feature_lab` tool; a server started with `--runtime modeling` gets no warning and the model improvises the step the prompt says not to.
13. **`validation.validate_dataframe`** has no application (self-documented; kept for its docstring). **`schemas.contains_ref`** is used only by tests.
14. **`YFinanceProvider.get_ohlcv_async` / Polygon / Bloomberg use `asyncio.get_event_loop()`**, deprecated outside a running loop; `DatabentoProvider` uses `asyncio.to_thread` and skips the `contextvars.copy_context()` the others do on purpose (the audit contextvars are explicitly why they copy it — moot only because Databento records nothing).
15. **`audit/storage.py`** is honest that only the writer is backend-routed; the `__init__` docstring's "pluggable storage backends" reads as if verify/retention/export were too.
16. **`ExternalValidationReport.rows_total`** comes from `handle.rows`, which for CSV is the row count at REGISTRATION (`known_rows`); after the file changes, `coverage()` is a fraction of a stale denominator. The tool already warns the file changed; the number should be nulled in that case.

---

## Appendix — where each data-runtime tool actually goes

| tool (data) | provider method | provider | recorded in audit `data_sources`? |
|---|---|---|---|
| `fetch_ohlcv` | `get_ohlcv` | yfinance (fixed) | yes |
| `fetch_ohlcv_panel`, `fetch_returns_panel` | `get_ohlcv_async` via `portfolio.fetch_ohlcv_panel_sync` | yfinance (fixed) | yes |
| `fetch_tick_tape` | `get_trades` | yfinance (fixed) -> **always refuses** | n/a |
| `fetch_quote_panel` | `get_quotes` | yfinance (fixed) -> **always refuses** | n/a |
| `fetch_financial_ratios` | `get_financial_ratios` | yfinance (fixed) | no (ratios are never recorded) |
| `get_dataset_metadata` | `get_metadata` | yfinance (fixed) | n/a |
| `infer_temporal_contract` | `contract_for_frame` | none | n/a |
| `build/describe/validate_data_bundle` | `DataBundle`, `validate_bundle` | none (inference) | n/a |
| `compare_ratio_frames`, `validate_financial_ratios` | `comparison`, `ratios` | none | n/a |
| `build_continuous_futures_series` | `continuous` | none | n/a |
| `register/describe/validate_external_dataset`, `prepare_vendor_extract` | `external`, `external_validation`, `databento.normalize_*` | none | n/a |
