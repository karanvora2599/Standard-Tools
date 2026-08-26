"""
Mutation testing: does the suite NOTICE when the code is wrong?

A passing test suite proves the tests run. It does not prove they would
catch a defect, and those are different claims. This breaks one specific
thing at a time and reruns the tests that should care; a mutation that
SURVIVES marks a test that is decorative.

RUN IT WITH:  python Development/mutation_testing.py
              python Development/mutation_testing.py --filter granger

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
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

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
        "cpcv: skip purging entirely",
        SRC / "backtesting/overfitting.py",
        "            label_window = range(i, min(i + label_horizon + 1, n_observations))\n"
        "            if any(j in test_set for j in label_window):\n"
        "                purged += 1\n                continue",
        "            pass",
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
        "bootstrap: force an IID resample regardless of block_size",
        SRC / "analysis/inference.py",
        "    if block_size <= 1:\n        return rng.integers(0, n, n)",
        "    return rng.integers(0, n, n)\n    if block_size <= 1:\n"
        "        return rng.integers(0, n, n)",
        "tests/analysis/test_inference.py",
    ),
    Mutation(
        "sharpe: revert the relative zero-dispersion check",
        SRC / "backtesting/overfitting.py",
        "    scale = float(np.max(np.abs(values)))\n"
        "    if scale > 0 and float(np.ptp(values)) <= scale * 1e-12:\n"
        '        return float("nan")',
        "    pass",
        "tests/backtesting/test_overfitting.py",
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


def _git(*arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *arguments], cwd=ROOT, capture_output=True, text=True, timeout=120
    )


def require_clean_tree(paths: List[Path]) -> None:
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
    interrupted session.
    """
    dirty = []
    for path in {p for p in paths}:
        relative = path.relative_to(ROOT).as_posix()
        result = _git("status", "--porcelain", "--", relative)
        if result.stdout.strip():
            dirty.append(relative)
    if dirty:
        raise SystemExit(
            "Refusing to run: these files have uncommitted changes and are "
            f"about to be mutated -- {dirty}.\n"
            "Restoration relies on `git checkout --`, which would discard "
            "your work. Commit or stash first."
        )


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
        print(f"restored {len({p for p in targets})} file(s) from git")
        return 0

    if options.list:
        for mutation in selected:
            print(f"  {mutation.name}  ->  {mutation.tests}")
        return 0

    require_clean_tree([m.path for m in selected])

    survivors: List[str] = []
    skipped: List[str] = []
    for mutation in selected:
        original = mutation.path.read_text(encoding="utf-8")
        if original.count(mutation.old) != 1:
            skipped.append(
                f"{mutation.name} (anchor matched {original.count(mutation.old)})"
            )
            print(f"  SKIP      {mutation.name}", flush=True)
            continue
        mutation.path.write_text(
            original.replace(mutation.old, mutation.new, 1), encoding="utf-8"
        )
        try:
            still_passing = run_tests(mutation.tests)
        finally:
            # Restored in a `finally`, AND recoverable through git if the
            # process is killed before the `finally` can run -- which is why
            # `require_clean_tree` refuses to start otherwise. A mutated
            # source that looks committed is the one way this tool could do
            # real harm, and it has happened once.
            mutation.path.write_text(original, encoding="utf-8")
        if still_passing:
            survivors.append(mutation.name)
            print(f"  SURVIVED  {mutation.name}", flush=True)
        else:
            print(f"  killed    {mutation.name}", flush=True)

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
