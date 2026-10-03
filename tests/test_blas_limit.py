"""
`single_threaded_blas()`: the library's own small linear algebra on one BLAS
thread, and the answers that then stop depending on the machine (see the
CHANGELOG entry of 2026-10-02).

Two halves. The limit itself: applied inside, the caller's setting back
after, nested and concurrent users all on one thread with the setting in
force before the first one restored after the last, a failure never
reaching the caller, `SQT_BLAS_THREADS` read and refused by name, and
nothing at all without threadpoolctl. Then the results: every function that
runs its factorizations under the limit returns the same bits whatever BLAS
thread count its caller set, and those bits are the one-thread ones.
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
from standard_quant_tools.analysis.pca import pca_returns
from standard_quant_tools.error import ValidationError
from standard_quant_tools.portfolio import construction
from standard_quant_tools.portfolio.covariance import estimate_covariance

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
    if isinstance(value, float):
        return np.float64(value).tobytes()
    return value


N = 235
RETURNS = _factor_returns(1260, N)
COVARIANCE = RETURNS.cov() * 252
RAGGED = _pairwise_covariance(N)
WEIGHTS = {name: 1.0 / N for name in COVARIANCE.columns}

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
    "estimate_covariance_sample": lambda: estimate_covariance(RETURNS, method="sample"),
    "estimate_covariance_ledoit_wolf": lambda: estimate_covariance(RETURNS),
    # Not the factor returns: that product keeps its threads (see pca.py).
    "pca_returns": lambda: {
        k: v for k, v in pca_returns(RETURNS).items() if k != "factor_returns"
    },
}


class TestTheAnswerDoesNotDependOnTheCallersThreads:
    @pytest.mark.parametrize("name", sorted(CALLS))
    def test_one_two_and_the_default_give_the_same_bits(self, name):
        call = CALLS[name]
        with threadpool_limits(limits=1, user_api="blas"):
            one = _bits(call())
        with threadpool_limits(limits=2, user_api="blas"):
            two = _bits(call())
        default = _bits(call())
        assert two == one
        assert default == one

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
