"""
`cointegration_test` reads a pair on one index as arrays and its half-life
gate off the spread's values, and returns the bits the pandas path
returned (see the CHANGELOG entry of 2026-10-04).

At 500 bars the Python layer was about 1.0 ms of a 1.1 ms test, twelve
times the native call: two label lookups aligning the pair, and a
half-life gate that put the spread back into a Series to align it with its
own lag, and dropped its missing values twice. Each shortcut is
held here to the path it replaced, written out below from pandas, numpy
and statsmodels as it ran before, on the inputs where the two could part:
equal and unequal indexes, index names, time zones and dtypes, dates out of
order or repeated, gaps, non-finite values, constant and affine pairs, and
magnitudes at the edges of the double range. The whole result is compared
by its bits, errors by type and message.
"""

from __future__ import annotations

import contextlib
import math

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.adfvalues import mackinnoncrit

from standard_quant_tools.analysis import cointegration as co
from standard_quant_tools.error import ValidationError
from standard_quant_tools.metrics.risk_metrics import has_no_dispersion
from standard_quant_tools.validation import require_finite_array


@contextlib.contextmanager
def _one_blas_thread():
    """The reference on one BLAS thread, as the library computes it: above
    10,000 terms a dot product's last bits follow the thread count."""
    try:
        import scipy.linalg  # noqa: F401
        from threadpoolctl import threadpool_limits
    except ImportError:  # pragma: no cover - threadpoolctl is a dependency
        yield
        return
    with threadpool_limits(limits=1, user_api="blas"):
        yield


def _bits(value):
    if isinstance(value, dict):
        return {k: _bits(v) for k, v in value.items()}
    if isinstance(value, float):
        return ("float", np.float64(value).tobytes())
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int):
        return ("int", value)
    return value


def _outcome(call):
    try:
        return ("returned", _bits(call()))
    except Exception as exc:  # noqa: BLE001 - compared below
        return ("raised", type(exc).__name__, str(exc))


# ── the pandas path, as it ran before ─────────────────────────────────────────


def _reference_half_life_statistics(spread: pd.Series, fitted_residual: bool):
    values = spread.dropna().to_numpy(dtype=float)
    require_finite_array(values, "spread", "half_life_statistics")
    n = int(values.size) - 1
    nan = float("nan")
    result = {
        "half_life": float("inf"),
        "ar_coefficient": nan,
        "t_statistic": nan,
        "critical_value": nan,
        "mean_reverting": False,
        "n_obs": max(n, 0),
    }
    if n < 3 or has_no_dispersion(values):
        return result
    y = np.diff(values)
    x = values[:-1]
    design = np.column_stack([np.ones(n), x])
    with _one_blas_thread():
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        residual = y - design @ beta
        dof = n - 2
        s2 = float(residual @ residual) / dof if dof > 0 else nan
        x_centred = x - x.mean()
        sxx = float(x_centred @ x_centred)
    se = math.sqrt(s2 / sxx) if sxx > 0 and s2 == s2 else nan
    ar_coeff = float(beta[1])
    t_stat = ar_coeff / se if se == se and se > 0 else nan
    critical = float(
        mackinnoncrit(N=2 if fitted_residual else 1, regression="c", nobs=n)[1]
    )
    hl = co.half_life(spread.dropna())
    result.update(
        {
            "half_life": float(hl),
            "ar_coefficient": ar_coeff,
            "t_statistic": float(t_stat),
            "critical_value": critical,
            "mean_reverting": bool(
                t_stat == t_stat and t_stat < critical and math.isfinite(hl)
            ),
        }
    )
    return result


def _reference_cointegration_test(series_a, series_b, autolag="aic"):
    if autolag.lower() not in ("aic", "bic"):
        raise ValidationError(f"autolag must be 'aic' or 'bic', got {autolag!r}")
    common_idx = series_a.index.intersection(series_b.index)
    a_vals = series_a.loc[common_idx].to_numpy(dtype=float)
    b_vals = series_b.loc[common_idx].to_numpy(dtype=float)
    require_finite_array(a_vals, "series_a", "cointegration_test")
    require_finite_array(b_vals, "series_b", "cointegration_test")
    n = len(a_vals)
    if n < co._MIN_COINT_OBS:
        raise ValidationError(
            f"cointegration_test: {n} aligned observation(s); a cointegration "
            f"verdict needs at least {co._MIN_COINT_OBS}. Check that the two "
            "series share dates, or widen the date range."
        )
    with _one_blas_thread():
        reason = co._degenerate_pair_reason(a_vals, b_vals)
    if reason is not None:
        raise ValidationError(f"cointegration_test: {reason}")
    raw = co._cpp_core.engle_granger(a_vals, b_vals, -1, autolag.lower() != "bic")
    spread = pd.Series(
        a_vals - float(raw["intercept"]) - float(raw["hedge_ratio"]) * b_vals,
        index=common_idx,
    )
    stats = _reference_half_life_statistics(spread, True)
    return {
        "cointegrated": bool(raw["cointegrated"]),
        "hedge_ratio": float(raw["hedge_ratio"]),
        "adf_statistic": float(raw["adf_statistic"]),
        "p_value": float(raw["p_value"]),
        "critical_values": {
            "1%": float(raw["cv_1pct"]),
            "5%": float(raw["cv_5pct"]),
            "10%": float(raw["cv_10pct"]),
        },
        "half_life_days": float(raw["half_life"]),
        "half_life_mean_reverting": stats["mean_reverting"],
        "half_life_t_statistic": stats["t_statistic"],
        "half_life_critical_value": stats["critical_value"],
        "n_obs": int(raw["n_obs"]),
    }


# ── inputs ────────────────────────────────────────────────────────────────────


def _walk(rng, n):
    return 100.0 + np.cumsum(rng.normal(size=n))


def _pairs():
    """(name, series_a, series_b): every way two series can line up."""
    rng = np.random.default_rng(2026)
    n = 300
    a, b = _walk(rng, n), _walk(rng, n)
    idx = pd.date_range("2010-01-01", periods=n, freq="B")
    utc = idx.tz_localize("UTC")
    reverting = np.zeros(n)
    for t in range(1, n):
        reverting[t] = 0.8 * reverting[t - 1] + rng.normal()
    frame = pd.DataFrame(np.column_stack([a, b, a + b]), index=idx)
    yield "one index object", pd.Series(a, index=idx), pd.Series(b, index=idx)
    yield "equal copies", pd.Series(a, index=idx), pd.Series(b, index=idx.copy())
    yield "cointegrated", pd.Series(1.3 * b + 20 + reverting, index=idx), pd.Series(
        b, index=idx
    )
    yield "names differ", pd.Series(a, index=idx.rename("x")), pd.Series(
        b, index=idx.rename("y")
    )
    yield "b has gaps", pd.Series(a, index=idx), pd.Series(b, index=idx).drop(idx[5:40])
    yield "a starts later", pd.Series(a, index=idx).iloc[3:], pd.Series(b, index=idx)
    yield "b reversed", pd.Series(a, index=idx), pd.Series(b, index=idx)[::-1]
    yield "both shuffled", pd.Series(a, index=idx).sample(
        frac=1.0, random_state=4
    ), pd.Series(b, index=idx).sample(frac=1.0, random_state=5)
    yield "UTC against New York", pd.Series(a, index=utc), pd.Series(
        b, index=utc.tz_convert("America/New_York")
    )
    yield "both UTC", pd.Series(a, index=utc), pd.Series(b, index=utc)
    yield "int against float labels", pd.Series(
        a, index=pd.Index(np.arange(n), dtype="int64")
    ), pd.Series(b, index=pd.Index(np.arange(n), dtype="float64"))
    yield "range index", pd.Series(a), pd.Series(b)
    yield "string labels", pd.Series(
        a, index=pd.Index([f"d{i}" for i in range(n)], dtype=object)
    ), pd.Series(b, index=pd.Index([f"d{i}" for i in range(n)], dtype=object))
    yield "dates repeated in both", pd.Series(
        a, index=idx[:150].append(idx[:150])
    ), pd.Series(b, index=idx[:150].append(idx[:150]))
    yield "a date repeated in a", pd.Series(
        a, index=idx[:-1].append(idx[:1])
    ), pd.Series(b, index=idx)
    yield "integer prices", pd.Series(
        np.round(a * 100).astype(np.int64), index=idx
    ), pd.Series(np.round(b * 100).astype(np.int64), index=idx)
    yield "object prices", pd.Series(list(a), index=idx, dtype=object), pd.Series(
        list(b), index=idx, dtype=object
    )
    yield "float32 prices", pd.Series(a.astype(np.float32), index=idx), pd.Series(
        b.astype(np.float32), index=idx
    )
    yield "nullable floats", pd.Series(a, index=idx, dtype="Float64"), pd.Series(
        b, index=idx, dtype="Float64"
    )
    yield "columns of one frame", frame[0], frame[1]
    yield "a NaN", pd.Series(np.where(np.arange(n) == 9, np.nan, a), index=idx), (
        pd.Series(b, index=idx)
    )
    yield "an infinity", pd.Series(a, index=idx), pd.Series(
        np.where(np.arange(n) == 9, np.inf, b), index=idx
    )
    yield "too short", pd.Series(a[:19], index=idx[:19]), pd.Series(
        b[:19], index=idx[:19]
    )
    yield "no shared date", pd.Series(a[:100], index=idx[:100]), pd.Series(
        b[100:], index=idx[100:]
    )
    yield "constant b", pd.Series(a, index=idx), pd.Series(np.full(n, 42.0), index=idx)
    yield "affine", pd.Series(2.5 * b + 3.0, index=idx), pd.Series(b, index=idx)
    yield "near the top of the range", pd.Series(a * 1e150, index=idx), pd.Series(
        b * 1e150, index=idx
    )
    yield "overflowing", pd.Series(a * 1e306, index=idx), pd.Series(
        -b * 1e306, index=idx
    )
    yield "near the bottom of the range", pd.Series(a * 1e-160, index=idx), pd.Series(
        b * 1e-160, index=idx
    )
    for length in (20, 21, 500, 2000, 12_000):
        r = np.random.default_rng(length)
        dates = pd.bdate_range("1980-01-01", periods=length)
        yield f"{length} bars", pd.Series(_walk(r, length), index=dates), pd.Series(
            _walk(r, length), index=dates
        )


PAIRS = list(_pairs())


@pytest.mark.skipif(not co.HAS_CPP, reason="the native Engle-Granger test")
@pytest.mark.parametrize("autolag", ["aic", "bic", "AIC", "neither"])
@pytest.mark.parametrize("name,series_a,series_b", PAIRS, ids=[p[0] for p in PAIRS])
def test_cointegration_test_is_the_pandas_path(name, series_a, series_b, autolag):
    """The whole result, or the error, of the pandas path: aligned by the
    intersection and two lookups, the spread a Series, the gate through
    it. Repeated dates kept their refusal (the spread's length is not the
    dates'), and a pair on one index its numbers."""
    expected = _outcome(
        lambda: _reference_cointegration_test(series_a, series_b, autolag)
    )
    got = _outcome(lambda: co.cointegration_test(series_a, series_b, autolag=autolag))
    assert got == expected


@pytest.mark.parametrize("name,series_a,series_b", PAIRS, ids=[p[0] for p in PAIRS])
def test_the_alignment_is_the_intersection_and_two_lookups(name, series_a, series_b):
    """The arrays `_aligned_pair` returns are the lookups' bits, contiguous
    like theirs, and never a view of the caller's data."""

    def lookups():
        common = series_a.index.intersection(series_b.index)
        return (
            common,
            series_a.loc[common].to_numpy(dtype=float),
            series_b.loc[common].to_numpy(dtype=float),
        )

    expected = _outcome(lambda: [v.tobytes() for v in lookups()[1:]])
    got = _outcome(
        lambda: [v.tobytes() for v in co._aligned_pair(series_a, series_b)[1:]]
    )
    assert got == expected
    if expected[0] == "returned":
        index, a_vals, b_vals = co._aligned_pair(series_a, series_b)
        assert list(index) == list(lookups()[0])
        for values, series in ((a_vals, series_a), (b_vals, series_b)):
            assert values.flags["C_CONTIGUOUS"]
            assert not np.shares_memory(values, series.to_numpy())


def _spreads():
    rng = np.random.default_rng(31)
    for length in (1, 2, 3, 4, 5, 21, 250, 2000, 10_002, 20_000):
        walk = np.cumsum(rng.normal(size=length))
        yield f"walk {length}", walk
        reverting = np.zeros(length)
        for t in range(1, length):
            reverting[t] = 0.9 * reverting[t - 1] + rng.normal()
        yield f"reverting {length}", reverting
    gappy = np.cumsum(rng.normal(size=400))
    gappy[[3, 50, 51, 52, 399]] = np.nan
    yield "gaps", gappy
    yield "constant", np.full(100, 12.3456)
    yield "constant but for one ulp", np.append(
        np.full(99, 12.3456), np.nextafter(12.3456, 13)
    )
    yield "zeros", np.zeros(50)
    yield "huge", np.cumsum(rng.normal(size=300)) * 1e150
    yield "past the band", np.cumsum(rng.normal(size=300)) * 1e101
    yield "tiny", np.cumsum(rng.normal(size=300)) * 1e-160
    yield "an infinity", np.append(np.cumsum(rng.normal(size=50)), np.inf)
    yield "overflowing differences", np.array([1e308, -1e308] * 30)


SPREADS = list(_spreads())


@pytest.mark.parametrize("fitted", [True, False])
@pytest.mark.parametrize("name,values", SPREADS, ids=[s[0] for s in SPREADS])
def test_the_half_life_statistics_are_the_pandas_path(name, values, fitted):
    """`half_life_statistics` of a Series, and the gate read off the array
    `cointegration_test` holds, both give what the function gave through
    pandas: same keys, same bits, same refusals."""
    spread = pd.Series(values, index=pd.bdate_range("1960-01-01", periods=len(values)))
    expected = _outcome(lambda: _reference_half_life_statistics(spread, fitted))
    assert _outcome(
        lambda: co.half_life_statistics(spread, fitted_residual=fitted)
    ) == (expected)
    if fitted:
        gate = _outcome(lambda: co._half_life_gate(np.asarray(values, dtype=float)))
        if expected[0] == "returned":
            reference = expected[1]
            assert gate == (
                "returned",
                {
                    "half_life_mean_reverting": reference["mean_reverting"],
                    "half_life_t_statistic": reference["t_statistic"],
                    "half_life_critical_value": reference["critical_value"],
                },
            )
        else:
            assert gate == expected


@pytest.mark.parametrize("name,values", SPREADS, ids=[s[0] for s in SPREADS])
def test_a_half_life_from_values_is_the_series_half_life(name, values):
    """With no gap and one row per date the lag alignment keeps every row
    where it is, so the half-life of the finite values the gate holds is
    the Series'."""
    finite = np.asarray(values, dtype=float)
    finite = finite[~np.isnan(finite)]
    if not np.all(np.isfinite(finite)):
        pytest.skip("the gate refuses a non-finite spread before this")
    spread = pd.Series(finite, index=pd.bdate_range("1960-01-01", periods=len(finite)))
    assert _outcome(lambda: co._half_life_of(finite)) == _outcome(
        lambda: co.half_life(spread)
    )


def test_the_dispersion_shortcut_is_has_no_dispersion():
    """`_no_dispersion` skips the standard deviation only where it cannot
    change `has_no_dispersion`'s answer: finite values between 1e-100 and
    1e100. Checked on constants, values one ulp apart, ranges either side
    of the 1e-12 line, and magnitudes across the whole double range."""
    rng = np.random.default_rng(0)
    cases = []
    for exponent in list(range(-320, 309, 11)) + [-101, -100, -99, 99, 100, 101]:
        scale = 10.0**exponent
        for size in (2, 4, 17, 300):
            cases.append(np.full(size, scale))
            one_ulp = np.full(size, scale)
            one_ulp[-1] = np.nextafter(scale, np.inf)
            cases.append(one_ulp)
            for noise in (1e-13, 1e-12, 1e-11):
                cases.append(scale * (1 + rng.normal(0, noise, size)))
            cases.append(scale * rng.normal(size=size))
            with_zero = np.full(size, scale)
            with_zero[0] = 0.0
            cases.append(with_zero)
    for size in (2, 50):
        cases += [
            np.zeros(size),
            np.full(size, 5e-324),
            np.array([1e308, -1e308] * size),
        ]
    for _ in range(3000):
        size = int(rng.integers(2, 40))
        scale = 10.0 ** rng.uniform(-110, 110)
        noise = 10.0 ** rng.uniform(-18, 0)
        cases.append(scale * (rng.uniform(-1, 1) + noise * rng.normal(size=size)))
    answers = set()
    for values in cases:
        values = np.asarray(values, dtype=float)
        if not np.all(np.isfinite(values)):
            continue
        expected = has_no_dispersion(values)
        assert co._no_dispersion(values) == expected, values[:4]
        answers.add(expected)
    assert answers == {True, False}


def test_the_cached_critical_value_is_mackinnons():
    for n_series in (1, 2):
        for nobs in (3, 4, 19, 20, 250, 499, 1999, 12_000):
            assert co._mackinnon_5pct(n_series, nobs) == float(
                mackinnoncrit(N=n_series, regression="c", nobs=nobs)[1]
            )


def test_a_screen_flags_what_the_predicate_flags_pair_by_pair():
    """`_degenerate_pairs` works out each series' half of the predicate
    once; every flag is still `_degenerate_pair_reason`'s, including for a
    duplicated column name, where a lookup selects two columns."""
    rng = np.random.default_rng(5)
    n = 200
    columns = {f"W{i}": _walk(rng, n) for i in range(5)}
    columns["DUP"] = 2.0 * columns["W1"] + 1.0
    columns["FLAT"] = np.full(n, 7.0)
    frame = pd.DataFrame(columns)
    names = list(frame.columns)
    pairs = [(a, b) for a in names for b in names if a != b]
    expected = [
        co._degenerate_pair_reason(
            frame[a].to_numpy(dtype=float), frame[b].to_numpy(dtype=float)
        )
        is not None
        for a, b in pairs
    ]
    assert co._degenerate_pairs(frame, pairs) == expected
    assert any(expected) and not all(expected)

    doubled = pd.concat([frame[["W0", "W2"]], frame[["W2"]]], axis=1)
    pairs = [("W0", "W2"), ("W2", "W0")]
    expected = [
        _outcome(
            lambda a=a, b=b: co._degenerate_pair_reason(
                doubled[a].to_numpy(dtype=float), doubled[b].to_numpy(dtype=float)
            )
            is not None
        )
        for a, b in pairs
    ]
    got = [_outcome(lambda p=p: co._degenerate_pairs(doubled, [p])[0]) for p in pairs]
    assert got == expected


@pytest.mark.skipif(not co.HAS_CPP, reason="the native batch Engle-Granger test")
def test_the_pair_scanner_reads_the_batch_by_column(monkeypatch):
    """`scan_pairs` reads the batch scan's columns rather than a Series per
    row. On a universe with planted pairs, a duplicated listing and a flat
    series, its answer is the one the per-pair loop gives -- the batch and
    the single test are the same native kernel, and every pair here shares
    one index -- but for the words of a refused pair's reason, which the
    two paths have always phrased differently."""
    from standard_quant_tools.agent.models import PairScannerInput
    from standard_quant_tools.agent.runtimes.research import tools as research
    from standard_quant_tools.data.factory import DataFactory

    rng = np.random.default_rng(37)
    n = 400
    dates = pd.bdate_range("2018-01-01", periods=n)
    closes = {f"W{i:02d}": _walk(rng, n) + 50 for i in range(9)}
    for name, base in (("P1", "W00"), ("P2", "W03")):
        reverting = np.zeros(n)
        for t in range(1, n):
            reverting[t] = 0.85 * reverting[t - 1] + rng.normal()
        closes[name] = 1.3 * closes[base] + 20.0 + reverting
    closes["DUP"] = 2.0 * closes["W05"] + 1.0
    closes["FLAT"] = np.full(n, 7.0)
    frames = {t: pd.DataFrame({"Close": c}, index=dates) for t, c in closes.items()}

    class Provider:
        def get_ohlcv(self, symbol, *args, **kwargs):
            return frames[symbol]

    monkeypatch.setattr(DataFactory, "get_provider", lambda *a, **k: Provider())
    inp = PairScannerInput(
        tickers=list(closes),
        start_date="2018-01-01",
        end_date="2030-01-01",
        min_half_life=0.5,
        max_half_life=10_000,
        multiple_testing="none",
        max_pairs=1_000,
    )
    batched = research.scan_pairs(inp).model_dump()
    assert batched["n_pairs_cointegrated"] >= 2

    import standard_quant_tools.analysis.cointegration as module

    def unavailable(*args, **kwargs):
        raise RuntimeError("batch path disabled for this test")

    monkeypatch.setattr(module, "scan_cointegrated_pairs", unavailable)
    looped = research.scan_pairs(inp).model_dump()

    def pairs_refused(result):
        return [(f["symbol_a"], f["symbol_b"]) for f in result.pop("failed_pairs")]

    assert pairs_refused(looped) == pairs_refused(batched)
    assert looped == batched
