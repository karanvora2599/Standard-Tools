"""
`validate_series` must not care how it was called.

THE BUG THIS EXISTS FOR. The decorator iterated `args` and never
`kwargs.values()`, so every guarantee it makes evaporated the moment a
caller used keyword arguments. Measured before the fix:

    sharpe_ratio(all-NaN)      positional REFUSED   keyword -> nan
    sortino_ratio(all-NaN)     positional REFUSED   keyword -> nan
    obv(empty, empty)          positional REFUSED   keyword -> IndexError

Nineteen functions wear the decorator and every one of them had the hole.

It matters more here than it would in most libraries. Keyword calls are
normal, readable Python, and they are specifically what an agent
constructing a call from a JSON schema produces — so the guarantee was
missing on exactly the caller this library exists to serve.

WHY IT WENT UNNOTICED. Every existing test called positionally. That is the
blind spot a suite develops when it is written alongside the code rather
than against the contract: the tests exercise the way the author happened to
write the calls, and a second way through the same door goes unopened. So
the test below does not check three functions someone thought of — it
enumerates every function carrying the decorator and calls each one both
ways.
"""

import inspect

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.error import ValidationError


def _decorated_functions():
    """
    Every function wearing `@validate_series`, found by walking the package.

    Enumerated rather than listed, so a function added tomorrow is covered
    without anybody remembering to add it here -- which is the same reason
    the bug survived: the list of things somebody thought to test is always
    shorter than the list of things that exist.
    """
    import importlib
    import pkgutil

    import standard_quant_tools

    found = {}
    for module_info in pkgutil.walk_packages(
        standard_quant_tools.__path__, "standard_quant_tools."
    ):
        try:
            module = importlib.import_module(module_info.name)
        except Exception:  # pragma: no cover - optional deps, SDK-gated modules
            continue
        for name, obj in vars(module).items():
            if name.startswith("_") or not callable(obj):
                continue
            if getattr(obj, "__wrapped__", None) is None:
                continue
            try:
                source = inspect.getsource(obj.__wrapped__)
            except (OSError, TypeError):  # pragma: no cover
                continue
            # The decorator is applied by name, so the wrapped source does
            # not carry it -- look at the definition site instead.
            try:
                lines, start = inspect.getsourcelines(obj)
            except (OSError, TypeError):  # pragma: no cover
                continue
            if any("@validate_series" in line for line in lines[:4]):
                found.setdefault(f"{module_info.name}.{name}", obj)
    return found


DECORATED = _decorated_functions()


class TestTheDecoratorIsFoundAtAll:
    def test_the_walk_finds_a_meaningful_number_of_functions(self):
        """If this drops to zero the tests below pass vacuously, which is
        worse than failing."""
        assert len(DECORATED) >= 10, (
            f"only {len(DECORATED)} decorated functions were found; the "
            "enumeration has broken and every test below is now vacuous"
        )

    def test_it_finds_the_ones_the_bug_was_measured_on(self):
        names = {n.rsplit(".", 1)[-1] for n in DECORATED}
        assert {"sharpe_ratio", "sortino_ratio", "obv"} <= names


class TestPositionalAndKeywordAgree:
    """The invariant. How a function was called cannot change whether its
    input was valid."""

    @staticmethod
    def _outcome(fn, series, keyword):
        """REFUSED, or whatever came back. Both are answers; disagreeing
        between call styles is not."""
        try:
            if keyword:
                first = list(inspect.signature(fn).parameters)[0]
                fn(**{first: series})
            else:
                fn(series)
            return "returned"
        except ValidationError:
            return "REFUSED"
        except Exception as exc:  # noqa: BLE001 - the point is what it was
            return type(exc).__name__

    @pytest.mark.parametrize("name", sorted(DECORATED))
    def test_an_empty_series_is_treated_the_same_either_way(self, name):
        fn = DECORATED[name]
        empty = pd.Series([], dtype=float)
        positional = self._outcome(fn, empty, keyword=False)
        keyword = self._outcome(fn, empty, keyword=True)
        assert positional == keyword, (
            f"{name} answered {positional!r} positionally and {keyword!r} by "
            "keyword on the same empty input. The decorator is inspecting "
            "only one of the two."
        )

    @pytest.mark.parametrize("name", sorted(DECORATED))
    def test_an_all_nan_series_is_treated_the_same_either_way(self, name):
        fn = DECORATED[name]
        nans = pd.Series([np.nan] * 40)
        positional = self._outcome(fn, nans, keyword=False)
        keyword = self._outcome(fn, nans, keyword=True)
        assert positional == keyword, (
            f"{name} answered {positional!r} positionally and {keyword!r} by "
            "keyword on the same all-NaN input"
        )

    @pytest.mark.parametrize("name", sorted(DECORATED))
    def test_a_series_containing_an_infinity_is_treated_the_same_either_way(self, name):
        """The docstring's own worst example: max_drawdown on a series with
        one inf returned a plausible-looking -1.70 rather than failing."""
        fn = DECORATED[name]
        infected = pd.Series([0.01] * 20 + [np.inf] + [0.01] * 19)
        positional = self._outcome(fn, infected, keyword=False)
        keyword = self._outcome(fn, infected, keyword=True)
        assert positional == keyword, (
            f"{name} answered {positional!r} positionally and {keyword!r} by "
            "keyword on the same series containing an infinity"
        )


class TestTheGuaranteesThemselves:
    """That the two call styles agree is necessary and not sufficient --
    they could agree on being wrong. These check the outcome."""

    @pytest.mark.parametrize("keyword", [False, True])
    def test_an_all_nan_return_series_is_refused(self, keyword):
        from standard_quant_tools.metrics import sharpe_ratio

        nans = pd.Series([np.nan] * 40)
        with pytest.raises(ValidationError):
            sharpe_ratio(returns=nans) if keyword else sharpe_ratio(nans)

    @pytest.mark.parametrize("keyword", [False, True])
    def test_an_infinity_is_refused(self, keyword):
        """An infinity is not a measurement -- it is a division that should
        not have happened upstream -- and it does not stay visible."""
        from standard_quant_tools.metrics import max_drawdown

        infected = pd.Series([0.01] * 20 + [np.inf] + [0.01] * 19)
        with pytest.raises(ValidationError, match="infinite"):
            max_drawdown(returns=infected) if keyword else max_drawdown(infected)

    @pytest.mark.parametrize("keyword", [False, True])
    def test_partial_nan_is_allowed(self, keyword):
        """The deliberate default. Warm-up windows, a mid-sample listing and
        a benchmark on a different holiday calendar all produce legitimate
        gaps; making those fatal would break correct code to catch a problem
        it has already handled."""
        from standard_quant_tools.metrics import sharpe_ratio

        partial = pd.Series(
            [np.nan] * 5 + list(np.random.default_rng(0).normal(0.001, 0.01, 60))
        )
        result = sharpe_ratio(returns=partial) if keyword else sharpe_ratio(partial)
        assert np.isfinite(result)

    def test_a_valid_series_is_untouched_either_way(self):
        from standard_quant_tools.metrics import sharpe_ratio

        good = pd.Series(np.random.default_rng(1).normal(0.001, 0.01, 200))
        assert sharpe_ratio(good) == pytest.approx(sharpe_ratio(returns=good))
