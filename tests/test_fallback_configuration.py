"""
The configuration this package could not run, and the surveys it broke.

Seventeen modules each decide `HAS_CPP` for themselves by probing
`_sqt_core`. That per-symbol design is right -- a kernel added later falls
back on its own rather than all-or-nothing -- but it meant the NO-EXTENSION
configuration could not be executed. Every fallback was reachable only by
monkeypatching one module's flag inside one test, so roughly half the
C++-adjacent code had no end-to-end coverage at all.

That is also what made this codebase easy to survey wrongly. Instrument
which functions execute on a machine where the extension is present and
every fallback reports as dead -- and a reachability analysis did exactly
that, then recommended deleting two of them. `n_workers`'s process pool and
eight of ten `@njit` kernels are not dead code; they are the `else:` arm of
`if HAS_CPP`.

`SQT_DISABLE_NATIVE=1` closes that. It is checked in a SUBPROCESS here
because the flags are read at import time, so flipping the variable inside a
running interpreter proves nothing about what a fresh one would do.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

#: Every module that makes its own HAS_CPP decision.
NATIVE_AWARE_MODULES = (
    "agent.runtimes._shared",
    "analysis.cointegration",
    "analysis.garch",
    "analysis.hurst",
    "analysis.multi_factor",
    "analysis.regression",
    "backtest.engine",
    "backtest.monte_carlo",
    "backtest.portfolio_engine",
    "backtest.strategies",
    "indicators.momentum",
    "indicators.panel",
    "indicators.trend",
    "indicators.volatility",
    "modeling.features.transforms",
    "modeling.validation.metrics",
    "modeling.validation.weights",
)


def _run(script: str, disable: bool) -> str:
    """Run a snippet in a fresh interpreter, with the switch on or off."""
    env = dict(os.environ)
    env["SQT_RUNS_DIR"] = env.get("SQT_RUNS_DIR", "")
    if disable:
        env["SQT_DISABLE_NATIVE"] = "1"
    else:
        env.pop("SQT_DISABLE_NATIVE", None)
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    return completed.stdout.strip()


_FLAGS = """
    import importlib
    mods = {mods!r}
    on = []
    for name in mods:
        module = importlib.import_module("standard_quant_tools." + name)
        if getattr(module, "HAS_CPP", False):
            on.append(name)
    import json
    print(json.dumps({{"total": len(mods), "still_native": on}}))
"""


class TestTheSwitchReachesEveryModule:
    def test_all_of_them_fall_back_together(self):
        """One name made unimportable flips all seventeen, because they all
        import the same one. No module needed changing."""
        import json

        got = json.loads(_run(_FLAGS.format(mods=NATIVE_AWARE_MODULES), disable=True))
        assert got["total"] == len(NATIVE_AWARE_MODULES)
        assert not got[
            "still_native"
        ], f"these ignored the switch: {got['still_native']}"

    def test_the_default_is_still_native(self):
        """The switch must be opt-in. If this ever reports the fallback by
        default, every published performance number is wrong."""
        import json

        pytest.importorskip("standard_quant_tools._sqt_core")
        got = json.loads(_run(_FLAGS.format(mods=NATIVE_AWARE_MODULES), disable=False))
        assert got["still_native"], "nothing is using the extension without the switch"

    def test_the_module_list_here_is_the_whole_list(self):
        """A new module that probes `_sqt_core` and is not listed above would
        be untested in the configuration this file exists to cover."""
        import pathlib

        root = (
            pathlib.Path(__file__).resolve().parents[1] / "src" / "standard_quant_tools"
        )
        found = set()
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            if "HAS_CPP = " in text:
                rel = path.relative_to(root).with_suffix("")
                found.add(str(rel).replace("\\", ".").replace("/", "."))
        assert found == set(NATIVE_AWARE_MODULES), (
            f"missing from the list: {sorted(found - set(NATIVE_AWARE_MODULES))}; "
            f"listed but gone: {sorted(set(NATIVE_AWARE_MODULES) - found)}"
        )


class TestTheFallbackActuallyComputes:
    """Flipping a flag is not the same as the other path working."""

    def test_a_backtest_runs_and_agrees_with_the_native_one(self):
        script = """
            import numpy as np, pandas as pd, json
            from standard_quant_tools.backtest.engine import run_strategy, HAS_CPP
            n = 400
            idx = pd.bdate_range("2022-01-03", periods=n)
            close = 100*np.exp(np.cumsum(np.random.default_rng(0).normal(0,0.01,n)))
            df = pd.DataFrame({"Open":close,"High":close*1.01,"Low":close*0.99,
                               "Close":close,"Volume":1e6}, index=idx)
            sig = pd.Series((np.arange(n) % 7 < 3).astype(float), index=idx)
            out = run_strategy(df, sig)
            print(json.dumps({"has_cpp": HAS_CPP,
                              "sharpe": round(float(out["sharpe_ratio"]), 9),
                              "total": round(float(out["total_return"]), 9)}))
        """
        import json

        fallback = json.loads(_run(script, disable=True))
        native = json.loads(_run(script, disable=False))
        assert fallback["has_cpp"] is False
        assert native["has_cpp"] is True
        assert fallback["sharpe"] == pytest.approx(native["sharpe"], rel=1e-9)
        assert fallback["total"] == pytest.approx(native["total"], rel=1e-9)

    def test_the_uniqueness_weights_agree(self):
        """The kernel whose row-count guard was removed. Both paths must
        still produce the same weights at a size that used to be gated."""
        script = """
            import numpy as np, pandas as pd, json
            from standard_quant_tools.modeling.validation import weights as w
            days = pd.date_range("2015-01-01", periods=60, freq="B")
            dates = np.repeat(days.to_numpy("datetime64[ns]"), 20)
            ents = np.tile(np.array([f"T{i:02d}" for i in range(20)], dtype=object), 60)
            out = w.label_uniqueness_weights(dates, dates + np.timedelta64(5,"D"), ents)
            print(json.dumps({"has_cpp": w.HAS_CPP,
                              "checksum": round(float(out.sum()), 9),
                              "n": int(out.size)}))
        """
        import json

        fallback = json.loads(_run(script, disable=True))
        native = json.loads(_run(script, disable=False))
        assert fallback["has_cpp"] is False and native["has_cpp"] is True
        assert fallback["n"] == native["n"]
        assert fallback["checksum"] == pytest.approx(native["checksum"], rel=1e-9)


class TestTheSwitchIsReadableFromCode:
    def test_the_helper_reports_it(self):
        import standard_quant_tools as sqt

        assert sqt.native_disabled() is False
        assert sqt.DISABLE_NATIVE_ENV == "SQT_DISABLE_NATIVE"

    def test_only_explicit_truthy_values_count(self):
        """An empty or unset variable must not disable anything, or a stray
        export in a shell profile silently halves the package's speed."""
        script = """
            import standard_quant_tools as sqt
            print(sqt.native_disabled())
        """
        env_cases = {"": "False", "0": "False", "no": "False", "1": "True"}
        for value, expected in env_cases.items():
            env = dict(os.environ)
            env["SQT_DISABLE_NATIVE"] = value
            done = subprocess.run(
                [sys.executable, "-c", textwrap.dedent(script)],
                capture_output=True,
                text=True,
                env=env,
                timeout=120,
            )
            assert done.returncode == 0, done.stderr[-1500:]
            assert done.stdout.strip() == expected, f"{value!r} -> {done.stdout!r}"
