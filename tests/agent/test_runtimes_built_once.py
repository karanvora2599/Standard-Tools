"""
The runtime registry is built once per process, not once per thread.

`all_runtimes()` built lazily with no lock, so threads arriving together
each ran the whole build -- nine package imports and every Runtime -- and
each got a dict of its own: `resolve(name)` returned different objects to
different threads. Measured in a fresh process, because the point is the
cold start.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

_PROBE = textwrap.dedent("""
    import threading

    from standard_quant_tools.agent import runtimes

    barrier = threading.Barrier(8)
    seen = []

    def build():
        barrier.wait()
        seen.append(runtimes.all_runtimes())

    threads = [threading.Thread(target=build) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    print(len(seen), len({id(registry) for registry in seen}))
    print(runtimes.resolve("data") is seen[0]["data"])
    print(runtimes.all_runtimes() is runtimes.all_runtimes())
    """)


def test_threads_arriving_together_get_one_registry():
    """Planted: eight threads through a barrier got eight dicts. Null: a
    later sequential call is the same object."""
    completed = subprocess.run(
        [sys.executable, "-c", _PROBE],
        env=dict(os.environ),
        capture_output=True,
        text=True,
        timeout=600,
        check=True,
    )
    lines = completed.stdout.split()
    assert lines[:2] == ["8", "1"], completed.stdout
    assert lines[2:] == ["True", "True"], completed.stdout
