"""
Mutation testing: does the suite NOTICE when the code is wrong?

A passing test suite proves the tests run. It does not prove they would
catch a defect, and those are different claims. This breaks one specific
thing at a time and reruns the tests that should care; a mutation that
SURVIVES marks a test that is decorative.

RUN IT WITH:  python scripts/mutation_testing.py
              python scripts/mutation_testing.py --filter granger

WHAT IT HAS ALREADY CAUGHT. Three mutations survived the first run of this
catalogue, and two were real:

  - The Granger false-positive test asserted `rate < 0.15`, and the
    UNCORRECTED rate is 12-15%. It slipped under the very bar it existed
    to enforce.
  - The seasonality test asserted `p_value_corrected >= p_value_raw`, which
    stays true when the mutation changes only the FLAG. The corrected
    number is still reported; it just stops being the one that decides.

Both are now pinned by a test that searches for a case where the correction
changes the ANSWER and asserts the flag follows it. Neither can pass with
the correction removed.

THE MUTATIONS ARE DELIBERATE, not random. Each is a change somebody could
plausibly make -- a correction dropped as redundant, a sign flipped, a
guard removed, a permutation swapped for a resample. Random character
damage would be caught by any test and would prove nothing.

ADDING ONE: append to `MUTATIONS`. If the anchor no longer matches, the run
reports it as SKIPPED rather than silently passing -- a mutation that does
not apply is not a mutation that was survived, and conflating the two is
how a mutation suite quietly stops testing anything.

A RUN LEAVES THE TREE AS IT FOUND IT, byte for byte and with the same
modification time. Each file is put back from the bytes read before it was
mutated, not re-encoded from decoded text: a text round trip rewrote an LF
working copy as CRLF on Windows (and a CRLF one as LF elsewhere), git then
listed every mutated file as modified with no diff to show, and the next
run refused to start.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "standard_quant_tools"


@dataclass(frozen=True)
class Mutation:
    """One deliberate defect, and the tests that should notice it."""

    name: str
    path: Path
    old: str
    new: str
    tests: str


MUTATIONS: List[Mutation] = [
    Mutation(
        "roll: delete the significance gate",
        SRC / "analysis/microstructure_estimators.py",
        "        bool(covariance < -2.0 * covariance_se) if math.isfinite(covariance) else None",
        "        True if math.isfinite(covariance) else None",
        "tests/analysis/test_microstructure_estimators.py",
    ),
    Mutation(
        "granger: drop the Bonferroni correction",
        SRC / "analysis/structure.py",
        'corrected = min(1.0, best["p_value"] * len(results))',
        'corrected = best["p_value"]',
        "tests/analysis/test_structure.py",
    ),
    Mutation(
        "seasonality: drop the Bonferroni correction",
        SRC / "analysis/diagnostics.py",
        '"significant_after_correction": bool(raw_p * n_tests < 0.05),',
        '"significant_after_correction": bool(raw_p < 0.05),',
        "tests/analysis/test_diagnostics.py",
    ),
    Mutation(
        # Named "skip purging entirely" while the purge had one side; the
        # side after a test block now has its own line and mutation below.
        "cpcv: skip the purge before a test block",
        SRC / "backtesting/overfitting.py",
        "        dropped = seen[window_end] > seen\n",
        "        dropped = np.zeros(n_observations, dtype=bool)\n",
        "tests/backtesting/test_overfitting.py",
    ),
    Mutation(
        # The one-sided purge this splitter shipped with: rows inside a
        # test block's label reach after it are kept unless the embargo
        # happens to cover them.
        "cpcv: skip the purge after a test block",
        SRC / "backtesting/overfitting.py",
        "        dropped |= prior > prior[window_start]\n",
        "        dropped |= np.zeros(n_observations, dtype=bool)\n",
        "tests/backtesting/test_overfitting.py",
    ),
    Mutation(
        "monte carlo: resample with replacement instead of permuting",
        SRC / "backtesting/trade_analysis.py",
        "_path_stats(rng.permutation(array))",
        "_path_stats(rng.choice(array, size=array.size, replace=True))",
        "tests/backtesting/test_trade_analysis.py",
    ),
    Mutation(
        # Found by an audit that applied it out-of-repo: it survived 2,149
        # tests across 71 live calls. Optimizer weights are invariant to a
        # uniform scaling of the covariance, and every magnitude assertion
        # elsewhere is computed from that same covariance, so nothing
        # noticed every portfolio volatility coming back 15.87x too small.
        "covariance: annualize by sqrt(ppy) instead of ppy",
        SRC / "portfolio/optimize.py",
        "    cov = returns_df.cov().to_numpy(dtype=float) * periods_per_year",
        "    cov = returns_df.cov().to_numpy(dtype=float) * np.sqrt(periods_per_year)",
        "tests/test_single_definition.py",
    ),
    Mutation(
        # Also a proven survivor: 1,300 tests. Every deflated-Sharpe test
        # asserts an ordering or a 0..1 range and a monotone transform
        # preserves both. `kurtosis` is the RAW fourth moment here, so
        # normal is 3.0 and the correction subtracts 1.0, not 3.0.
        "deflated sharpe: switch to the excess-kurtosis convention",
        SRC / "backtest/robustness.py",
        "((kurtosis - 1.0) / 4.0)",
        "((kurtosis - 3.0) / 4.0)",
        "tests/test_single_definition.py",
    ),
    Mutation(
        # Clamping a non-positive block_size UP to 1 returns an IID
        # resample under the name of a block bootstrap.
        "bootstrap: clamp a non-positive block_size instead of refusing",
        SRC / "_resampling.py",
        '    block_size = require_positive_int(block_size, "block_size", "block_indices")',
        "    block_size = max(int(block_size), 1)",
        "tests/test_single_definition.py",
    ),
    Mutation(
        # Moved here from analysis/inference.py, which used to hold its
        # own copy of the block index construction. Setting `span` to 1
        # is the same mutation the old anchor made: every block becomes
        # one observation, so the resample is IID and the serial
        # correlation the block bootstrap exists to preserve is gone.
        #
        # The anchor carries the moving-block start line because the
        # circular draw beside it opens with the same `span` line, and an
        # anchor that matches twice mutates nothing.
        "bootstrap: force an IID resample regardless of block_size",
        SRC / "_resampling.py",
        "    span = min(block_size, n)\n"
        "    n_blocks = int(math.ceil(target / span))\n"
        "    starts = rng.integers(0, n - span + 1, n_blocks)",
        "    span = 1\n"
        "    n_blocks = int(math.ceil(target / span))\n"
        "    starts = rng.integers(0, n - span + 1, n_blocks)",
        "tests/analysis/test_inference.py",
    ),
    Mutation(
        # The circular draw with its starts confined as the moving draw's
        # are: the wrap then never happens and the edges are under-drawn.
        "resampling: a circular block draw that never wraps",
        SRC / "_resampling.py",
        "    starts = rng.integers(0, n, n_blocks)",
        "    starts = rng.integers(0, n - span + 1, n_blocks)",
        "tests/backtesting/test_overfitting.py",
    ),
    Mutation(
        # The reality check's own summed draw, with its starts confined to
        # [0, n - block]: the moving draw again, whose edges are under-drawn
        # and whose null is off centre.
        "reality check: draw block starts without wrapping the series",
        SRC / "backtesting/overfitting.py",
        "        n_starts = n\n",
        "        n_starts = n - block_size + 1\n",
        "tests/backtesting/test_overfitting.py",
    ),
    Mutation(
        # Drop the sqrt(n / (n - b)) widening as a cosmetic factor near 1.
        # It is 1.095 at 120 days with 20-day blocks, and without it White's
        # best of three rejected 37 of 400 true nulls at a nominal 5%.
        #
        # The factor moved to `_resampling.circular_block_scale` when the
        # bootstrap intervals took it too, so the mutation follows it there
        # and each of its three users' tests has to notice.
        "reality check: drop the block-bootstrap variance scale",
        SRC / "_resampling.py",
        "    return math.sqrt(n / (n - block_size)) if block_size < n else 1.0\n",
        "    return 1.0\n",
        "tests/backtesting/test_overfitting.py tests/analysis/test_inference.py "
        "tests/backtest/test_robustness.py",
    ),
    Mutation(
        # The alternating series everywhere, as `compare_distributions` had
        # it: it does not converge near zero and gave two identical samples
        # a p-value of 0.0.
        "kolmogorov tail: use the alternating series below the switch too",
        SRC / "_special.py",
        "    if x < 1.18:\n        y = math.exp(-(math.pi**2) / (8.0 * x * x))",
        "    if x < 0.0:\n        y = math.exp(-(math.pi**2) / (8.0 * x * x))",
        "tests/analysis/test_inference.py tests/test_single_definition.py",
    ),
    Mutation(
        # compare_distributions back on its own 100-term alternating sum.
        "compare distributions: sum the alternating series for the KS p-value",
        SRC / "analysis/inference.py",
        "    p_value = kolmogorov_sf(lam)\n",
        "    p_value = min(1.0, max(0.0, 2.0 * sum(\n"
        "        (-1) ** (k - 1) * math.exp(-2.0 * k * k * lam * lam)\n"
        "        for k in range(1, 101)\n"
        "    )))\n",
        "tests/analysis/test_inference.py",
    ),
    Mutation(
        # The count taken from the list after it is cut to ten, as it was.
        "volatility smile: count the listed violations, not all of them",
        SRC / "analysis/derivatives.py",
        '        "n_arbitrage_violations": len(every_violation),\n',
        '        "n_arbitrage_violations": len(violations),\n',
        "tests/analysis/test_derivatives.py",
    ),
    Mutation(
        # run_cointegration_test returning a random walk's half-life with
        # nothing said about the test that rejects it.
        "cointegration tool: say nothing when the half-life's test fails",
        SRC / "agent/runtimes/research/tools.py",
        "    if not reverting and math.isfinite(hl):\n",
        "    if not reverting and hl < 0:\n",
        "tests/agent/test_agent_tools.py",
    ),
    Mutation(
        # scan_pairs ranking on a half-life without its test, as before.
        "scan pairs: keep a half-life its own test rejects",
        SRC / "agent/runtimes/research/tools.py",
        '            if not reversion["mean_reverting"]:\n',
        '            if reversion["n_obs"] < 0:\n',
        "tests/agent/test_new_agent_tools.py",
    ),
    Mutation(
        # Change points by level only, as before: a volatility jump with no
        # change in level is no change point at all.
        "change points: skip the variance search",
        SRC / "analysis/structure.py",
        "    variance_breaks = _variance_breaks(values, array, breaks, max_breaks, min_segment)\n",
        "    variance_breaks = []\n",
        "tests/analysis/test_structure.py",
    ),
    Mutation(
        # The squares' variance as normal tails give it (Inclan and Tiao):
        # t(4) noise with a constant variance was flagged 62 times in 100.
        "change points: assume normal tails in the variance search",
        SRC / "analysis/structure.py",
        "    omega = float(deviation @ deviation) / n\n",
        "    omega = 2.0 * float(squares.mean()) ** 2\n",
        "tests/analysis/test_structure.py",
    ),
    Mutation(
        # The squares treated as independent: short-lived volatility
        # states were flagged 41 times in 100 against 15.
        "change points: drop the long-run variance of the squares",
        SRC / "analysis/structure.py",
        "    for lag in range(1, min(bandwidth, n - 1) + 1):\n",
        "    for lag in range(1, 1):\n",
        "tests/analysis/test_structure.py",
    ),
    Mutation(
        # The break placed at the CUSUM's own argmax: 515 for a jump at 500,
        # and the true change's tail then read as a second break at 495.
        "change points: place a variance break at the CUSUM's argmax",
        SRC / "analysis/structure.py",
        "    return int(positions[int(np.argmin(cost))]), statistic, kolmogorov_sf(statistic)\n",
        "    return (\n"
        "        int(positions[int(np.argmax(cusum[positions - 1]))]),\n"
        "        statistic,\n"
        "        kolmogorov_sf(statistic),\n"
        "    )\n",
        "tests/analysis/test_structure.py",
    ),
    Mutation(
        # The variance searched on the series about its overall mean: a
        # step in the level is then a jump in the squares.
        "change points: search the variance about the overall mean",
        SRC / "analysis/structure.py",
        "        residuals[lo:hi] = array[lo:hi] - array[lo:hi].mean()\n",
        "        residuals[lo:hi] = array[lo:hi] - array.mean()\n",
        "tests/analysis/test_structure.py",
    ),
    Mutation(
        # The concentration as it was: the largest share in absolute value,
        # whichever regime it belongs to, under the best regime's name.
        "regime stratified: report the largest absolute share as the best's",
        SRC / "backtesting/overfitting.py",
        '    concentration = concentrated["share_of_pnl"] if concentrated else None\n',
        "    concentration = (\n"
        '        max(abs(r["share_of_pnl"]) for r in rows) if concentrated else None\n'
        "    )\n",
        "tests/backtesting/test_overfitting.py",
    ),
    Mutation(
        # A loss attributed to the best regime, which is the one that lost
        # least of it.
        "regime stratified: attribute a loss to the best regime",
        SRC / "backtesting/overfitting.py",
        '        concentrated, concentration_of = rows[-1], "loss"\n',
        '        concentrated, concentration_of = rows[0], "loss"\n',
        "tests/backtesting/test_overfitting.py",
    ),
    Mutation(
        # The interval's draw back to moving blocks: the first and last
        # block - 1 observations are under-drawn and the draws centre off
        # the sample's statistic. A day carrying the whole sum then
        # averaged 7% of the sample mean across the draws.
        "bootstrap interval: draw moving blocks instead of circular ones",
        SRC / "analysis/inference.py",
        "_draw_resamples(array, block_size, hi - lo, rng, circular=True)",
        "_draw_resamples(array, block_size, hi - lo, rng, circular=False)",
        "tests/analysis/test_inference.py",
    ),
    Mutation(
        # The widening dropped at the interval's own call site, which the
        # helper's mutation above cannot reach: 60 observations in 20-day
        # blocks then covered the mean 80% of the time at a nominal 95%.
        "bootstrap interval: drop the widening of the draws",
        SRC / "analysis/inference.py",
        "        usable = centre + scale * (usable - centre)\n",
        "        usable = centre + (usable - centre)\n",
        "tests/analysis/test_inference.py",
    ),
    Mutation(
        # A widened win rate left unbounded: two losing days in sixty put
        # the upper bound past 1.
        "bootstrap interval: let a widened draw leave the statistic's range",
        SRC / "analysis/inference.py",
        "            usable = np.clip(usable, *_STATISTIC_BOUNDS[statistic])\n",
        "            usable = usable\n",
        "tests/analysis/test_inference.py",
    ),
    Mutation(
        "robustness CI: draw moving blocks instead of circular ones",
        SRC / "backtest/robustness.py",
        "    draw = circular_block_indices if circular else block_indices\n",
        "    draw = block_indices\n",
        "tests/backtest/test_robustness.py",
    ),
    Mutation(
        "robustness CI: drop the widening of the draws",
        SRC / "backtest/robustness.py",
        "        boot_metrics = centre + scale * (boot_metrics - centre)\n",
        "        boot_metrics = centre + (boot_metrics - centre)\n",
        "tests/backtest/test_robustness.py",
    ),
    Mutation(
        # Moved here from backtesting/overfitting.py, where this check used
        # to be inlined. It is now the shared guard six call sites use, so
        # reverting it here tests all of them at once rather than one.
        "sharpe: revert the relative zero-dispersion check",
        SRC / "metrics/risk_metrics.py",
        "    scale = float(np.nanmax(np.abs(array))) if array.size else 0.0\n"
        "    return scale > 0 and float(np.ptp(array)) <= scale * DISPERSION_RTOL",
        "    return False",
        "tests/metrics/test_metrics.py",
    ),
    Mutation(
        "normality: revert the relative zero-dispersion check",
        SRC / "analysis/inference.py",
        "    scale = float(np.max(np.abs(array)))\n"
        "    if std <= 0 or (scale > 0 and float(np.ptp(array)) <= scale * 1e-12):",
        "    if std <= 0:",
        "tests/analysis/test_final_seven.py",
    ),
    Mutation(
        "risk parity: report convergence unconditionally",
        SRC / "portfolio/construction.py",
        "        if np.max(np.abs(weights - previous)) < tolerance:\n"
        "            converged = True\n            break",
        "        converged = True\n        break",
        "tests/portfolio/test_construction.py",
    ),
    Mutation(
        "parity: drop the dividend growth term",
        SRC / "analysis/derivatives.py",
        "    right = spot * growth - strike * discount",
        "    right = spot - strike * discount",
        "tests/analysis/test_derivatives.py",
    ),
    Mutation(
        "vanna: drop the division by vol",
        SRC / "analysis/derivatives.py",
        "    vanna_raw = -growth * pdf_d1 * d2 / vol",
        "    vanna_raw = -growth * pdf_d1 * d2",
        "tests/analysis/test_derivatives.py",
    ),
    Mutation(
        "runs test: flip the clustering sign",
        SRC / "backtesting/trade_analysis.py",
        "    clustered = bool(math.isfinite(z) and z < -1.96)\n"
        "    alternating = bool(math.isfinite(z) and z > 1.96)",
        "    clustered = bool(math.isfinite(z) and z > 1.96)\n"
        "    alternating = bool(math.isfinite(z) and z < -1.96)",
        "tests/backtesting/test_trade_analysis.py",
    ),
    Mutation(
        "shortfall: ignore the buy/sell direction",
        SRC / "analysis/microstructure_estimators.py",
        '    direction = 1.0 if side == "buy" else -1.0',
        "    direction = 1.0",
        "tests/analysis/test_final_seven.py",
    ),
    Mutation(
        "marginal risk: use weight instead of the covariance product",
        SRC / "portfolio/construction.py",
        "    marginal = matrix @ ordered / volatility",
        "    marginal = ordered / volatility",
        "tests/analysis/test_final_seven.py",
    ),
    Mutation(
        "decay test: compare overlapping halves of the rolling series",
        SRC / "analysis/diagnostics.py",
        "    split = len(values) // 2\n    first = values.to_numpy()[:split]\n"
        "    second = values.to_numpy()[split:]",
        "    split = len(values) // 2\n    first = values.to_numpy()[:split]\n"
        "    second = values.to_numpy()[:split]",
        "tests/analysis/test_diagnostics.py",
    ),
    Mutation(
        "last_finite: return NaN instead of refusing an empty series",
        SRC / "validation.py",
        "    if finite.size < max(1, minimum):",
        "    if False:",
        "tests/surface/test_adversarial_inputs.py -m slow",
    ),
    Mutation(
        "options: remove the magnitude bound that stops exp() overflowing",
        SRC / "analysis/options.py",
        "        if not math.isfinite(numeric) or abs(numeric) > high:",
        "        if False:",
        "tests/surface/test_adversarial_inputs.py -m slow",
    ),
    Mutation(
        "runtime: stop refusing a tool from another runtime",
        SRC / "agent/runtimes/__init__.py",
        "        if tool_name not in self.dispatch_table:\n"
        "            raise ValueError(self._out_of_scope_message(tool_name))",
        "        if tool_name not in self.dispatch_table:\n"
        "            from standard_quant_tools.agent.tools import _TOOL_DISPATCH\n"
        "            fn, model_cls = _TOOL_DISPATCH[tool_name]\n"
        "            return sanitize_for_json(\n"
        "                _run_and_record(tool_name, fn, model_cls(**arguments))\n"
        "            )",
        "tests/surface/test_invariants.py",
    ),
    Mutation(
        # The fixed 9.0 back as the default: calibrated for independent
        # noise, it fired on 40 of 40 no-shift AR(0.9) bases.
        "basis dislocation: a fixed threshold whatever the basis's memory",
        SRC / "delta_one/basis.py",
        "        calibrate_threshold=calibrate,\n",
        "        calibrate_threshold=False,\n",
        "tests/delta_one/test_a_persistent_basis_is_not_a_shift.py",
    ),
    Mutation(
        # Memory read off the whole series, shift included: a step looks
        # like persistence and raises the null that judges it.
        "basis dislocation: measure the memory on the whole series",
        SRC / "delta_one/basis.py",
        '        memory_window="reference",\n',
        '        memory_window="series",\n',
        "tests/delta_one/test_a_persistent_basis_is_not_a_shift.py",
    ),
    Mutation(
        # The warm-up's marginal sd back as the scale: no-shift AR(0.9)
        # streams alerted 20 of 20 times against it.
        "spread monitor: standardize by the marginal sd",
        SRC / "delta_one/streaming.py",
        '        z = (value - mean) / state["baseline_long_run_std"]\n',
        "        z = (value - mean) / std\n",
        "tests/delta_one/test_a_persistent_basis_is_not_a_shift.py",
    ),
]


def run_tests(target: str) -> bool:
    """True when the named tests pass."""
    arguments = target.split()
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            *arguments,
            "-x",
            "-q",
            "--no-header",
            "-p",
            "no:cacheprovider",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    return result.returncode == 0


def _git(*arguments: str, root: Path = ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *arguments], cwd=root, capture_output=True, text=True, timeout=120
    )


def uncommitted_change(path: Path, root: Path = ROOT) -> str:
    """
    Why `git checkout --` could not give this file back as it is now, or ""
    when it could.

    CONTENT, NOT STAT. `git status` lists a file whose size changed without
    comparing what it holds, so a file rewritten with other line endings --
    which is what this tool's restore used to do -- reads as modified while
    `git diff` shows nothing. That is not work `git checkout --` would
    lose. These three are: a file git does not track, a change staged but
    not committed, and a working copy whose content (after the line-ending
    rules in `.gitattributes` and `core.autocrlf`) differs from the index.
    """
    relative = path.relative_to(root).as_posix()
    if _git("ls-files", "--error-unmatch", "--", relative, root=root).returncode:
        return "not tracked by git"
    if _git("diff", "--cached", "--quiet", "--", relative, root=root).returncode:
        return "staged, not committed"
    if _git("diff", "--quiet", "--", relative, root=root).returncode:
        return "modified"
    return ""


def require_clean_tree(paths: List[Path], root: Path = ROOT) -> None:
    """
    Refuse to start with uncommitted changes in a file about to be mutated.

    THIS IS THE SAFETY PROPERTY THAT MATTERS. Restoration happens in a
    `finally`, and a `finally` does not run when the process is KILLED --
    by a CI timeout, by Ctrl-C on some platforms, by an OOM. This tool then
    leaves a deliberate defect on disk that looks like ordinary source.

    It happened: a ten-minute shell timeout killed a run mid-mutation and
    left `if False:` in place of a bounds check in options.py.

    Requiring a clean tree makes that recoverable unconditionally, because
    `git checkout --` restores the original whatever state the process died
    in. `--restore` does exactly that, and is what to run after an
    interrupted session. "Clean" is judged on content; see
    `uncommitted_change`.
    """
    dirty = []
    for path in sorted({p for p in paths}):
        reason = uncommitted_change(path, root)
        if reason:
            dirty.append(f"{path.relative_to(root).as_posix()} ({reason})")
    if dirty:
        raise SystemExit(
            "Refusing to run: these files have uncommitted changes and are "
            f"about to be mutated -- {dirty}.\n"
            "Restoration relies on `git checkout --`, which would discard "
            "your work. Commit or stash first."
        )


@contextmanager
def applied(mutation: Mutation) -> Iterator[bool]:
    """
    Apply `mutation` for the body of the block, then put the file back.

    Yields False, and writes nothing, when the anchor does not match exactly
    once. The anchors are written with "\\n"; in a working copy checked out
    with CRLF they are matched, and the replacement written, with CRLF, so
    only the anchor's bytes change while the mutation is live.

    THE RESTORE WRITES THE ORIGINAL BYTES and then the original access and
    modification times, so the file is what it was before the run in every
    respect `git status` reads but the inode change time. It runs in a
    `finally`, so a failing or interrupted test run still restores; a killed
    process does not, which is what `require_clean_tree` and `--restore`
    are for.
    """
    path = mutation.path
    original = path.read_bytes()
    before = path.stat()
    text = original.decode("utf-8")
    newline = "\r\n" if "\r\n" in text else "\n"
    old = mutation.old.replace("\n", newline)
    if text.count(old) != 1:
        yield False
        return
    mutated = text.replace(old, mutation.new.replace("\n", newline), 1)
    try:
        path.write_bytes(mutated.encode("utf-8"))
        yield True
    finally:
        path.write_bytes(original)
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument(
        "--filter", default="", help="only run mutations whose name contains this"
    )
    parser.add_argument(
        "--list", action="store_true", help="list the catalogue and exit"
    )
    parser.add_argument(
        "--restore",
        action="store_true",
        help="git-restore every file this tool mutates, then exit. Run this "
        "after an interrupted session.",
    )
    options = parser.parse_args()

    selected = [m for m in MUTATIONS if options.filter.lower() in m.name.lower()]
    targets = [m.path for m in MUTATIONS]

    if options.restore:
        for path in {p.relative_to(ROOT).as_posix() for p in targets}:
            _git("checkout", "--", path)
        (ROOT / "scripts" / ".mutation_active").unlink(missing_ok=True)
        print(f"restored {len({p for p in targets})} file(s) from git")
        return 0

    if options.list:
        for mutation in selected:
            print(f"  {mutation.name}  ->  {mutation.tests}")
        return 0

    require_clean_tree([m.path for m in selected])

    # A marker, so a `git add -A` running in another shell while this is
    # live is at least diagnosable afterwards. It is not a lock -- nothing
    # here can stop another process committing a mutated file, which is why
    # `tests/surface/test_invariants.py` also refuses constant conditions in
    # the source. That happened: `if False:` reached a commit.
    marker = ROOT / "scripts" / ".mutation_active"
    marker.write_text(
        "A mutation is applied to the source RIGHT NOW. Do not commit.\n"
        "If this file exists and no run is in progress, a run was killed:\n"
        "  python scripts/mutation_testing.py --restore\n",
        encoding="utf-8",
    )

    survivors: List[str] = []
    skipped: List[str] = []
    for mutation in selected:
        # Restored in a `finally` inside `applied`, AND recoverable through
        # git if the process is killed before the `finally` can run -- which
        # is why `require_clean_tree` refuses to start otherwise. A mutated
        # source that looks committed is the one way this tool could do real
        # harm, and it has happened once.
        with applied(mutation) as live:
            if live:
                still_passing = run_tests(mutation.tests)
        if not live:
            found = mutation.path.read_text(encoding="utf-8").count(mutation.old)
            skipped.append(f"{mutation.name} (anchor matched {found})")
            print(f"  SKIP      {mutation.name}", flush=True)
            continue
        if still_passing:
            survivors.append(mutation.name)
            print(f"  SURVIVED  {mutation.name}", flush=True)
        else:
            print(f"  killed    {mutation.name}", flush=True)

    marker.unlink(missing_ok=True)
    killed = len(selected) - len(survivors) - len(skipped)
    print(f"\n{killed} killed, {len(survivors)} survived, {len(skipped)} skipped")
    if survivors:
        print("\nSURVIVORS — the tests covering these would not notice the defect:")
        for name in survivors:
            print(f"  - {name}")
    if skipped:
        print("\nSKIPPED — the anchor drifted; the mutation tested nothing:")
        for name in skipped:
            print(f"  - {name}")
    return 1 if survivors or skipped else 0


if __name__ == "__main__":
    raise SystemExit(main())
