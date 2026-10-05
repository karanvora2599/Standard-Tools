"""
`single_threaded_blas()`: the library's own covariance-sized linear algebra
on one BLAS thread, and the answers that then stop depending on the machine
(see the CHANGELOG entries of 2026-10-02, 2026-10-04 and 2026-10-04).

Two halves. The limit itself: applied inside, the caller's setting back
after, nested and concurrent users all on one thread with the setting in
force before the first one restored after the last, a failure never
reaching the caller, `SQT_BLAS_THREADS` read and refused by name, and
nothing at all without threadpoolctl. Then the results: every function that
runs its products and factorizations under the limit returns the same bits
whatever BLAS thread count its caller set, and those bits are the
one-thread ones.

The products that build a matrix -- np.cov's, Ledoit-Wolf's, the EWMA one,
PCA's factor returns, the network features' and the lead-lag correlations
-- kept the caller's threads until 2026-10-04. Their last bits followed the
thread count on the CI runners' OpenBLAS (the sample and Ledoit-Wolf
covariances, where these tests failed) or on OpenBLAS 0.3.27 and 0.3.31 on
a 16-thread Windows machine (the rest), and every output built from them
did too.

Until the CHANGELOG entry of 2026-10-04 some of the library's own linear
algebra still ran on the caller's threads, and these outputs followed the
thread count under OpenBLAS 0.3.27 on that machine: `pca_whiten`'s fit and
projection (20,000 rows of 30 features), the VIFs and `collinear` block of
`redundancy_report` (60 features), the half-life t-statistic of
`half_life_statistics` and `cointegration_test` (dot products of more than
10,000 terms) and `book_metrics`' depth slope. The factor regression's and
the ADF statistic's least squares and Gram products gave the same bits there
at every limit; they run under it too, since a product of the same kind,
np.cov's, followed the thread count on the CI runners.
"""

from __future__ import annotations

import random
import sys
import threading
import time

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools import _blas
from standard_quant_tools._blas import single_threaded_blas
from standard_quant_tools.analysis.cointegration import (
    cointegration_test,
    half_life_statistics,
)
from standard_quant_tools.analysis.correlation import diversification_ratio
from standard_quant_tools.analysis.diagnostics import lead_lag_matrix
from standard_quant_tools.analysis.multi_factor import multi_factor_regression
from standard_quant_tools.analysis.order_book import book_metrics
from standard_quant_tools.analysis.pca import factor_contributions, pca_returns
from standard_quant_tools.analysis.stationarity import run_stationarity_tests
from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.analysis.feature_report import redundancy_report
from standard_quant_tools.modeling.features import network
from standard_quant_tools.modeling.preprocessing.base import FoldContext
from standard_quant_tools.modeling.preprocessing.steps import PCAWhiten
from standard_quant_tools.portfolio import construction
from standard_quant_tools.portfolio.covariance import estimate_covariance
from standard_quant_tools.portfolio.optimize import (
    annualized_mean_cov,
    black_litterman,
    frontier_stats,
)
from standard_quant_tools.portfolio.portfolio import portfolio_metrics

threadpoolctl = pytest.importorskip("threadpoolctl")
from threadpoolctl import threadpool_limits  # noqa: E402


def _threads():
    """Each controlled BLAS library's thread count right now."""
    return [
        library.get_num_threads() for library in _blas._get_controller().lib_controllers
    ]


@pytest.fixture
def four_threads():
    """The caller's BLAS setting during a test: up to four threads, so that
    one thread inside the limit is observable. Skipped where BLAS cannot run
    more than one."""
    if _blas._get_controller() is None:
        pytest.skip("no controllable BLAS in this environment")
    with threadpool_limits(limits=4, user_api="blas"):
        outside = _threads()
        if max(outside) < 2:
            pytest.skip("BLAS runs one thread here, so a limit of one is invisible")
        yield outside


# ── the limit ────────────────────────────────────────────────────────────


class TestTheLimit:
    def test_it_holds_only_the_blas_libraries(self):
        controller = _blas._get_controller()
        if controller is None:
            pytest.skip("no controllable BLAS in this environment")
        assert controller.lib_controllers
        assert {lib.user_api for lib in controller.lib_controllers} == {"blas"}

    def test_one_thread_inside_and_the_callers_setting_after(self, four_threads):
        with single_threaded_blas():
            assert _threads() == [1] * len(four_threads)
        assert _threads() == four_threads
        assert _blas._users == 0 and _blas._saved is None

    def test_nested_users_restore_only_at_the_outermost_exit(self, four_threads):
        with single_threaded_blas():
            with single_threaded_blas():
                assert _threads() == [1] * len(four_threads)
                assert _blas._users == 2
            assert _threads() == [1] * len(four_threads)
        assert _threads() == four_threads

    def test_an_exception_inside_still_restores(self, four_threads):
        with pytest.raises(RuntimeError, match="inside"):
            with single_threaded_blas():
                raise RuntimeError("inside")
        assert _threads() == four_threads
        assert _blas._users == 0

    def test_many_threads_all_run_on_one_and_the_original_returns(self, four_threads):
        """Twelve threads entering and leaving, nested half the time, with
        jittered sleeps between: every one of them always sees one thread,
        their users overlap, and the caller's four come back after the
        last. With a plain set-and-restore per user, an early leaver would
        put the threads back while another was still inside."""
        workers = 12
        barrier = threading.Barrier(workers)
        seen, overlap, errors = set(), [], []

        def run(seed):
            rng = random.Random(seed)
            try:
                barrier.wait()
                for _ in range(40):
                    with single_threaded_blas():
                        seen.add(tuple(_threads()))
                        overlap.append(_blas._users)
                        if rng.random() < 0.5:
                            with single_threaded_blas():
                                seen.add(tuple(_threads()))
                        time.sleep(rng.random() * 2e-4)
                        seen.add(tuple(_threads()))
            except Exception as exc:  # noqa: BLE001 - reported below
                errors.append(exc)

        threads = [threading.Thread(target=run, args=(i,)) for i in range(workers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert not errors
        assert seen == {(1,) * len(four_threads)}
        assert max(overlap) > 1
        assert _threads() == four_threads
        assert _blas._users == 0 and _blas._saved is None

    def test_concurrent_answers_are_the_one_thread_answer(self, four_threads):
        """The point of counting users: a factorization on any thread, at
        any moment, gives the one-thread bits rather than bits that depend
        on whether another user happened to be inside."""
        rng = np.random.default_rng(3)
        a = rng.normal(size=(470, 235))
        matrix = a.T @ a / 470
        with threadpool_limits(limits=1, user_api="blas"):
            expected = np.linalg.eigh(matrix)[0]
        results, errors = [], []

        def run():
            try:
                for _ in range(3):
                    with single_threaded_blas():
                        results.append(np.linalg.eigh(matrix)[0])
            except Exception as exc:  # noqa: BLE001 - reported below
                errors.append(exc)

        threads = [threading.Thread(target=run) for _ in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert not errors and len(results) == 18
        for values in results:
            assert values.tobytes() == expected.tobytes()

    def test_a_failing_library_never_fails_the_caller(self, monkeypatch):
        calls = []

        class Library:
            def __init__(self, name, fail_on_set):
                self.name, self.fail_on_set, self.threads = name, fail_on_set, 8

            def get_num_threads(self):
                return self.threads

            def set_num_threads(self, n):
                calls.append((self.name, n))
                if self.fail_on_set and n == 1:
                    raise OSError("refused")
                self.threads = n

        class Controller:
            lib_controllers = [Library("first", False), Library("second", True)]

        monkeypatch.setattr(_blas, "_controller", Controller())
        ran = []
        with single_threaded_blas():
            ran.append(True)
        assert ran == [True]
        # The first was set, the second refused, and both were put back to
        # what they had: nothing is left half-limited.
        assert calls == [("first", 1), ("second", 1), ("first", 8), ("second", 8)]
        assert [lib.threads for lib in Controller.lib_controllers] == [8, 8]
        assert _blas._users == 0 and _blas._saved is None

    def test_without_threadpoolctl_it_does_nothing(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "threadpoolctl", None)
        monkeypatch.setattr(_blas, "_controller", None)
        monkeypatch.setattr(_blas, "_unavailable", False)
        with single_threaded_blas():
            matrix = np.array([[2.0, 1.0], [1.0, 2.0]])
            values = np.linalg.eigvalsh(matrix)
        assert values.tolist() == pytest.approx([1.0, 3.0])
        assert _blas._unavailable and _blas._controller is None
        assert _blas._users == 0
        # And the library's own callers work unchanged.
        frame = pd.DataFrame(matrix, index=["A", "B"], columns=["A", "B"])
        repaired, notes = construction._repair_psd(frame, "test")
        assert repaired is frame and notes == []

    def test_a_fork_starts_with_no_users_and_a_free_lock(self, monkeypatch):
        """A forked child inherits the counter and the lock but not the
        threads that would release them; the at-fork hook resets both."""
        monkeypatch.setattr(_blas, "_lock", threading.Lock())
        monkeypatch.setattr(_blas, "_users", 3)
        monkeypatch.setattr(_blas, "_saved", [])
        held = _blas._lock
        held.acquire()
        try:
            _blas._after_fork_in_child()
            assert _blas._lock is not held and not _blas._lock.locked()
            assert _blas._users == 0 and _blas._saved is None
        finally:
            held.release()


class TestTheOverride:
    def test_unset_and_blank_are_one(self, monkeypatch):
        monkeypatch.delenv("SQT_BLAS_THREADS", raising=False)
        assert _blas.blas_thread_limit() == 1
        monkeypatch.setenv("SQT_BLAS_THREADS", "  ")
        assert _blas.blas_thread_limit() == 1

    def test_a_number_of_threads_is_used(self, monkeypatch, four_threads):
        monkeypatch.setenv("SQT_BLAS_THREADS", "2")
        with single_threaded_blas():
            assert _threads() == [min(2, n) for n in four_threads]
        assert _threads() == four_threads

    def test_zero_leaves_blas_alone(self, monkeypatch, four_threads):
        monkeypatch.setenv("SQT_BLAS_THREADS", "0")
        assert _blas.blas_thread_limit() is None
        with single_threaded_blas():
            assert _threads() == four_threads
        assert _threads() == four_threads

    @pytest.mark.parametrize("value", ["one", "1.5", "-1"])
    def test_anything_else_is_refused_by_name(self, monkeypatch, value):
        monkeypatch.setenv("SQT_BLAS_THREADS", value)
        with pytest.raises(ValidationError, match="SQT_BLAS_THREADS"):
            with single_threaded_blas():
                pass  # pragma: no cover - never entered
        assert _blas._users == 0


# ── the results ──────────────────────────────────────────────────────────


def _factor_returns(n_obs, n_assets, seed=61, k=5):
    rng = np.random.default_rng(seed)
    factors = rng.normal(0, 0.01, (n_obs, k))
    betas = rng.normal(0.8, 0.4, (k, n_assets)) / k
    values = factors @ betas + rng.normal(0, 0.015, (n_obs, n_assets))
    return pd.DataFrame(
        values,
        columns=[f"A{i:03d}" for i in range(n_assets)],
        index=pd.bdate_range("2015-01-02", periods=n_obs),
    )


def _pairwise_covariance(n_assets, seed=63):
    """A covariance over a ragged panel, estimated pair by pair: slightly
    indefinite, so `_repair_psd` has to repair it."""
    rng = np.random.default_rng(seed)
    values = rng.normal(0, 0.01, (300, n_assets))
    values[rng.random(values.shape) < 0.35] = np.nan
    frame = pd.DataFrame(values, columns=[f"A{i:03d}" for i in range(n_assets)])
    return frame.cov(min_periods=30)


def _bits(value):
    """A result as a comparable structure, every float by its bits."""
    if isinstance(value, dict):
        return {k: _bits(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_bits(v) for v in value]
    if isinstance(value, pd.DataFrame):
        return (list(value.index), list(value.columns), value.to_numpy().tobytes())
    if isinstance(value, pd.Series):
        return (list(value.index), value.to_numpy().tobytes())
    if isinstance(value, np.ndarray):
        return value.tobytes()
    if isinstance(value, float):
        return np.float64(value).tobytes()
    return value


def _matrix(result):
    """estimate_covariance's nested-dict matrix as an array, in its order."""
    assets = result["assets"]
    return np.array([[result["matrix"][row][col] for col in assets] for row in assets])


N = 235
RETURNS = _factor_returns(1260, N)
COVARIANCE = RETURNS.cov() * 252
RAGGED = _pairwise_covariance(N)
WEIGHTS = {name: 1.0 / N for name in COVARIANCE.columns}
#: A wide window for the power iteration, and one for the network
#: features' four products, whose last bits followed the thread count only
#: at four threads and up (sixteen under OpenBLAS 0.3.31) on the machine
#: that measured them: hence the caller's default among the limits below.
WIDE = _factor_returns(252, 500, seed=67)
NETWORK = _factor_returns(126, 1000, seed=71)
VIEWS = np.zeros((2, N))
VIEWS[0, 0], VIEWS[0, 1], VIEWS[1, 2] = 1.0, -1.0, 1.0


def _latent_features(n_rows, n_features, seed, latent):
    """A feature matrix driven by a few latent factors, built without a
    matrix product so its own bits do not depend on the BLAS."""
    rng = np.random.default_rng(seed)
    factors = rng.normal(size=(n_rows, latent))
    loadings = rng.normal(size=(latent, n_features))
    values = rng.normal(scale=0.7, size=(n_rows, n_features))
    for j in range(latent):
        values += factors[:, [j]] * loadings[[j], :]
    return values


def _pca_whiten():
    """A fit and its projection, as a walk-forward fold or the full-panel
    refit runs them."""
    frame = pd.DataFrame(
        _latent_features(20_000, 30, 73, 4), columns=[f"f{i}" for i in range(30)]
    )
    context = FoldContext(dates=np.zeros(len(frame), dtype="datetime64[ns]"))
    step = PCAWhiten(n_components=5, whiten=True)
    state = step.fit(frame, context)
    return [state, step.transform(frame, state, context)]


FEATURES = pd.DataFrame(
    _latent_features(600, 60, 79, 10), columns=[f"x{i:02d}" for i in range(60)]
)


def _walks(n, seed):
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("1990-01-01", periods=n)
    a = pd.Series(100.0 + np.cumsum(rng.normal(size=n)), index=index)
    b = pd.Series(100.0 + np.cumsum(rng.normal(size=n)), index=index)
    return a, b


#: Longer than 10,000 bars, where OpenBLAS splits a dot product across
#: threads: an intraday history.
LONG_A, LONG_B = _walks(12_000, 83)
SPREAD = pd.Series(
    np.cumsum(np.random.default_rng(89).normal(size=20_000)) * 0.05
    + np.random.default_rng(97).normal(size=20_000)
)


def _book(snapshots, levels, seed):
    """Snapshots of a depth book: 600 of ten levels a side give the depth
    slope 12,000 points."""
    rng = np.random.default_rng(seed)
    mid = 100.0 + np.cumsum(rng.normal(0, 0.01, snapshots))
    columns = {}
    for i in range(levels):
        columns[f"bid_price_{i}"] = (
            mid - 0.01 * (i + 0.5) - rng.uniform(0, 2e-3, snapshots)
        )
        columns[f"bid_size_{i}"] = rng.uniform(100, 1000, snapshots) * (1 + i)
        columns[f"ask_price_{i}"] = (
            mid + 0.01 * (i + 0.5) + rng.uniform(0, 2e-3, snapshots)
        )
        columns[f"ask_size_{i}"] = rng.uniform(100, 1000, snapshots) * (1 + i)
    return pd.DataFrame(columns)


BOOK = _book(600, 10, 101)

CALLS = {
    "_repair_psd": lambda: construction._repair_psd(RAGGED, "test"),
    "risk_parity": lambda: construction.risk_parity(RAGGED, max_iterations=200),
    "max_diversification": lambda: construction.max_diversification(COVARIANCE),
    "max_diversification_ragged": lambda: construction.max_diversification(RAGGED),
    "marginal_risk_contribution": lambda: construction.marginal_risk_contribution(
        WEIGHTS, RAGGED
    ),
    "portfolio_scenarios": lambda: construction.portfolio_scenarios(
        WEIGHTS, {"down": {"A000": -0.2, "A001": -0.1}}, covariance=RAGGED
    ),
    # The whole of each: the matrix, its eigenvalues and condition number,
    # the factor returns, the weights.
    "estimate_covariance_sample": lambda: estimate_covariance(RETURNS, method="sample"),
    "estimate_covariance_ledoit_wolf": lambda: estimate_covariance(RETURNS),
    "estimate_covariance_ewma": lambda: estimate_covariance(RETURNS, method="ewma"),
    "estimate_covariance_ewma_shrunk": lambda: estimate_covariance(
        RETURNS, method="ewma_shrunk"
    ),
    "annualized_mean_cov": lambda: annualized_mean_cov(RETURNS, 252),
    "frontier_stats": lambda: frontier_stats(*annualized_mean_cov(RETURNS, 252)),
    "black_litterman": lambda: black_litterman(
        COVARIANCE.to_numpy(), np.full(N, 1.0 / N), VIEWS, np.array([0.02, 0.05])
    ),
    "hierarchical_risk_parity": lambda: construction.hierarchical_risk_parity(RETURNS),
    "portfolio_metrics": lambda: portfolio_metrics(RETURNS, np.full(N, 1.0 / N)),
    "diversification_ratio": lambda: diversification_ratio(RETURNS),
    "pca_returns": lambda: pca_returns(RETURNS),
    "pca_returns_five": lambda: pca_returns(RETURNS, n_components=5),
    "pca_returns_power_iteration": lambda: pca_returns(
        WIDE, n_components=1, method="power_iteration"
    ),
    "factor_contributions": lambda: factor_contributions(RETURNS, 3),
    "network_correlation": lambda: network._pairwise_correlation(NETWORK, 20),
    "lead_lag_matrix": lambda: lead_lag_matrix(
        RETURNS.iloc[:, :120], min_correlation=0.05
    ),
    # The CHANGELOG entry of 2026-10-04: the last of the library's own BLAS
    # work, each whole output.
    "pca_whiten": _pca_whiten,
    "redundancy_report": lambda: redundancy_report(FEATURES, list(FEATURES.columns)),
    "multi_factor_regression": lambda: multi_factor_regression(
        RETURNS["A000"], RETURNS[["A001", "A002", "A003", "A004", "A005"]]
    ),
    "run_stationarity_tests": lambda: run_stationarity_tests(
        LONG_A.iloc[:5_000], lags=10
    ),
    "cointegration_test_long": lambda: cointegration_test(LONG_A, LONG_B),
    "half_life_statistics_long": lambda: half_life_statistics(
        SPREAD, fitted_residual=True
    ),
    "book_metrics": lambda: book_metrics(BOOK),
}


class TestTheAnswerDoesNotDependOnTheCallersThreads:
    @pytest.mark.parametrize("name", sorted(CALLS))
    def test_one_two_four_and_the_default_give_the_same_bits(self, name):
        call = CALLS[name]
        with threadpool_limits(limits=1, user_api="blas"):
            one = _bits(call())
        for limit in (2, 4):
            with threadpool_limits(limits=limit, user_api="blas"):
                assert _bits(call()) == one, f"{limit} threads"
        assert _bits(call()) == one, "the default"

    @pytest.mark.parametrize("method", ["sample", "ledoit_wolf", "ewma", "ewma_shrunk"])
    def test_a_covariance_reports_the_one_thread_eigenvalues_of_its_matrix(
        self, method
    ):
        """What each call reports about its own matrix: at any caller
        setting, the smallest eigenvalue and the condition number are the
        one-thread ones of the matrix that call returned."""
        for limit in (1, 2, 4, None):
            if limit is None:
                result = estimate_covariance(RETURNS, method=method)
            else:
                with threadpool_limits(limits=limit, user_api="blas"):
                    result = estimate_covariance(RETURNS, method=method)
            with threadpool_limits(limits=1, user_api="blas"):
                eigenvalues = np.linalg.eigvalsh(_matrix(result))
            assert result["smallest_eigenvalue"] == float(eigenvalues.min())
            assert result["condition_number"] == float(
                eigenvalues.max() / eigenvalues.min()
            )

    def test_the_sample_matrix_is_the_one_thread_product(self):
        """A known answer for the bits: at the caller's default the sample
        matrix is np.cov's product computed on one BLAS thread, annualized,
        and so is the optimizers' DataFrame.cov() one. On the CI runners'
        OpenBLAS the product at the default differed from it in the last
        bits until the CHANGELOG entry of 2026-10-04. The returns are read
        as estimate_covariance reads them, through dropna's copy, whose
        layout decides the mean's last bits."""
        values = RETURNS.dropna(how="all", axis=1).dropna().to_numpy(dtype=float)
        with threadpool_limits(limits=1, user_api="blas"):
            expected = np.cov(values, rowvar=False, ddof=1) * 252
            frame_cov = RETURNS.cov().to_numpy(dtype=float) * 252
        sample = _matrix(estimate_covariance(RETURNS, method="sample"))
        assert sample.tobytes() == expected.tobytes()
        assert annualized_mean_cov(RETURNS, 252)[1].tobytes() == frame_cov.tobytes()

    def test_the_factor_returns_are_the_one_thread_product(self):
        """The same for PCA: the factor returns are the centred, scaled
        returns times the loadings, multiplied on one BLAS thread. Under
        OpenBLAS 0.3.27 and 0.3.31 at sixteen threads that product differed
        from the one-thread one in the last bits."""
        result = pca_returns(RETURNS)
        values = RETURNS.dropna().to_numpy(dtype=float)
        centred = values - values.mean(axis=0)
        scaled = centred / centred.std(axis=0, ddof=1)
        with threadpool_limits(limits=1, user_api="blas"):
            expected = scaled @ result["loadings"].to_numpy()
        assert result["factor_returns"].to_numpy().tobytes() == expected.tobytes()

    def test_the_repair_is_the_one_thread_repair(self):
        """A known answer for the bits: the repaired matrix is exactly what
        the eigenvalue clipping gives on one BLAS thread."""
        matrix = RAGGED.to_numpy()
        with threadpool_limits(limits=1, user_api="blas"):
            values, vectors = np.linalg.eigh(matrix)
            floor = construction._PSD_TOLERANCE * values.max()
            expected = (vectors * np.maximum(values, floor)) @ vectors.T
        expected = (expected + expected.T) / 2.0
        repaired, notes = construction._repair_psd(RAGGED, "test")
        assert len(notes) == 1 and "not positive semi-definite" in notes[0]
        assert repaired.to_numpy().tobytes() == expected.tobytes()

    def test_a_psd_matrix_comes_back_as_itself(self):
        """The null case: nothing to repair, the same object, no note."""
        frame, notes = construction._repair_psd(COVARIANCE, "test")
        assert frame is COVARIANCE and notes == []
