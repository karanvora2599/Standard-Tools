"""
`OMP_WAIT_POLICY=PASSIVE`, set by importing the package (see the CHANGELOG
entry of 2026-10-02).

The OpenMP runtime reads the variable once, when it loads, and it loads with
the compiled extension during `import standard_quant_tools` -- so every check
here runs in a fresh interpreter: the one running the tests imported the
package long ago. Checked: the variable is set when absent or blank, a
caller's value is left alone, a runtime loaded before the package (by
scikit-learn) is detected and said once at debug level rather than warned
about, and the policy takes effect -- measured as the process's CPU time in
the 100 ms after a parallel region, which spinning workers fill and sleeping
ones do not. That is a count of busy CPUs, not a timing, so a loaded machine
moves it little: 0.0 under PASSIVE against 6 to 14 when the workers spin.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import textwrap

import pytest


def _run(script: str, **environment: "str | None") -> dict:
    """Run a snippet in a fresh interpreter with OMP_WAIT_POLICY unset
    unless given; return the JSON its last line prints."""
    env = dict(os.environ)
    env.pop("OMP_WAIT_POLICY", None)
    for name, value in environment.items():
        if value is None:
            env.pop(name, None)
        else:
            env[name] = value
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    return json.loads(completed.stdout.strip().splitlines()[-1])


_REPORT = """
    import json, os
    import standard_quant_tools as sqt
    print(json.dumps({
        "value": os.environ.get("OMP_WAIT_POLICY"),
        "outcome": sqt._OMP_WAIT_POLICY_DEFAULT,
    }))
"""

windows_only = pytest.mark.skipif(
    sys.platform != "win32", reason="vcomp140.dll is the MSVC OpenMP runtime"
)


class TestTheVariable:
    def test_absent_is_set_to_passive(self):
        assert _run(_REPORT) == {"value": "PASSIVE", "outcome": "set"}

    def test_blank_counts_as_unset(self):
        assert _run(_REPORT, OMP_WAIT_POLICY="  ") == {
            "value": "PASSIVE",
            "outcome": "set",
        }

    @pytest.mark.parametrize("value", ["ACTIVE", "passive", "active "])
    def test_a_value_in_the_environment_is_left_alone(self, value):
        assert _run(_REPORT, OMP_WAIT_POLICY=value) == {
            "value": value,
            "outcome": "caller",
        }

    def test_a_value_set_in_python_before_the_import_is_left_alone(self):
        script = (
            'import os\nos.environ["OMP_WAIT_POLICY"] = "ACTIVE"\n'
            + textwrap.dedent(_REPORT)
        )
        assert _run(script) == {"value": "ACTIVE", "outcome": "caller"}


@windows_only
class TestARuntimeLoadedFirst:
    def test_it_is_said_once_at_debug_level_and_never_warned(self):
        script = """
            import ctypes, json, logging, warnings
            records = []

            class Keep(logging.Handler):
                def emit(self, record):
                    records.append([record.levelno, record.getMessage()])

            logger = logging.getLogger("standard_quant_tools")
            logger.addHandler(Keep())
            logger.setLevel(logging.DEBUG)
            import sklearn.ensemble  # noqa: F401  (loads its own vcomp140.dll)

            kernel32 = ctypes.WinDLL("kernel32")
            kernel32.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
            kernel32.GetModuleHandleW.restype = ctypes.c_void_p
            preloaded = bool(kernel32.GetModuleHandleW("vcomp140.dll"))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                import standard_quant_tools as sqt
            print(json.dumps({
                "preloaded": preloaded,
                "outcome": sqt._OMP_WAIT_POLICY_DEFAULT,
                "records": [r for r in records if "OMP_WAIT_POLICY" in r[1]],
                "warnings": [str(w.message) for w in caught if "OMP" in str(w.message)],
            }))
        """
        result = _run(script)
        if not result["preloaded"]:
            pytest.skip("scikit-learn did not load vcomp140.dll in this environment")
        assert result["outcome"] == "too_late"
        assert len(result["records"]) == 1
        level, message = result["records"][0]
        assert level == logging.DEBUG
        assert "already loaded" in message and "Results are unaffected" in message
        assert result["warnings"] == []

    def test_a_fresh_import_says_nothing(self):
        script = """
            import json, logging
            records = []

            class Keep(logging.Handler):
                def emit(self, record):
                    records.append(record.getMessage())

            logger = logging.getLogger("standard_quant_tools")
            logger.addHandler(Keep())
            logger.setLevel(logging.DEBUG)
            import standard_quant_tools as sqt
            print(json.dumps({
                "outcome": sqt._OMP_WAIT_POLICY_DEFAULT,
                "said": [m for m in records if "OMP_WAIT_POLICY" in m],
            }))
        """
        assert _run(script) == {"outcome": "set", "said": []}


_BUSY_AFTER_A_REGION = """
    import json, time
    import numpy as np, pandas as pd
    import standard_quant_tools as sqt
    from standard_quant_tools.analysis.hurst import rolling_hurst

    series = pd.Series(np.random.default_rng(1).normal(0, 0.01, 5000))

    def busy_cpus_after_a_region():
        rolling_hurst(series, 252, 1, "dfa", 10)
        rolling_hurst(series, 252, 1, "dfa", 10)
        cpu, wall = time.process_time(), time.perf_counter()
        time.sleep(0.1)
        return (time.process_time() - cpu) / (time.perf_counter() - wall)

    values = []
    for _ in range(5):
        values.append(busy_cpus_after_a_region())
    print(json.dumps({
        "native": sqt.native_build_status().used,
        "outcome": sqt._OMP_WAIT_POLICY_DEFAULT,
        "busy": sorted(values)[2],
    }))
"""


@windows_only
class TestThePolicyTakesEffect:
    def test_the_workers_sleep_after_a_region(self):
        """The median of five: busy CPUs in the 100 ms after a 16-thread
        rolling Hurst. ACTIVE, the caller's own choice, is the control that
        shows the measurement can see a spin at all."""
        if (os.cpu_count() or 1) < 4:
            pytest.skip("too few CPUs for a spin to be told from noise")
        passive = _run(_BUSY_AFTER_A_REGION, SQT_NUM_THREADS=None)
        if not passive["native"]:
            pytest.skip("the compiled extension is not in use, so no region runs")
        active = _run(
            _BUSY_AFTER_A_REGION, OMP_WAIT_POLICY="ACTIVE", SQT_NUM_THREADS=None
        )
        assert passive["outcome"] == "set" and active["outcome"] == "caller"
        if active["busy"] < 2.0:
            pytest.skip(
                f"no spin to measure: ACTIVE kept {active['busy']:.2f} CPUs busy"
            )
        assert passive["busy"] < 1.0
        assert passive["busy"] < active["busy"] / 4
