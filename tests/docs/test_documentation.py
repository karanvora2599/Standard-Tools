"""
The documentation is checked, not trusted.

Documentation rots in a way code does not: nothing fails when it goes
stale. Before these tests existed, 85 of 157 tools appeared in no document
at all, the README advertised 95 tools across six runtimes when there were
157 across eight, and three separate guides quoted a tool count that had
been wrong for months. None of that broke a single test, because nothing
was checking.

WHAT IS ENFORCED HERE:

- **The tool index is generated, and current.** It is rendered from the
  live registry and compared exactly -- every name, description and
  argument list -- with each runtime's schema cost allowed 1 KB of slack,
  so adding a tool without regenerating fails in the same commit that
  added it.
- **Every tool appears somewhere.** A tool nobody can find is a tool nobody
  uses, however good it is.
- **Every count that appears in prose is the real one.** A number in a
  sentence is a claim, and this library's whole position is that claims get
  measured.
- **Every internal link resolves.** A link to a renamed file is a dead end
  a reader hits and an author never does.

WHAT IS NOT ENFORCED: whether the prose is any good. These tests catch
staleness and absence, not quality, and passing them is a floor rather than
a standard.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
DOCS = ROOT / "Documentation"
README = ROOT / "README.md"
GENERATOR = ROOT / "scripts" / "generate_tool_index.py"
INDEX = DOCS / "20_tool_index.md"


@pytest.fixture(scope="module")
def runtimes():
    from standard_quant_tools.agent.runtimes import all_runtimes

    return all_runtimes()


@pytest.fixture(scope="module")
def all_tools(runtimes):
    return {t for rt in runtimes.values() for t in rt.dispatch_table}


@pytest.fixture(scope="module")
def doc_text():
    """Every markdown file in the project, concatenated."""
    parts = [README.read_text(encoding="utf-8", errors="ignore")]
    parts += [p.read_text(encoding="utf-8", errors="ignore") for p in DOCS.glob("*.md")]
    return "\n".join(parts)


@pytest.fixture(scope="module")
def index_generator():
    """`scripts/generate_tool_index.py`, imported into this process."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("generate_tool_index", GENERATOR)
    assert spec and spec.loader, f"could not load {GENERATOR}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestTheToolIndexIsGenerated:
    def test_the_index_matches_the_live_registry(self, index_generator):
        """
        Render and compare. This is the mechanism that makes the other
        documentation tests unnecessary for the index itself: it cannot be
        stale, because staleness is a test failure rather than a thing
        someone has to notice.

        Rendered in this process, from the package these tests import, and
        never written to disk. The generator used to run as a subprocess and
        rewrite the file in place: under an interpreter whose editable
        install points at another checkout, that subprocess imported the
        other checkout's registry and compared it with this tree's index.
        Each runtime's schema cost may differ by `KB_TOLERANCE` (see the
        CHANGELOG entry of 2026-10-04); nothing else may.
        """
        problems = index_generator.drift(
            INDEX.read_text(encoding="utf-8"), index_generator.render()
        )
        assert not problems, (
            "Documentation/20_tool_index.md is out of date with the tool "
            "registry. Run `python scripts/generate_tool_index.py` and "
            "commit the result -- the index is generated, not written.\n"
            + "\n".join(problems)
        )

    def test_a_stale_description_fails_the_check(self, index_generator):
        """The slack is in the schema costs only: one word of one description
        that no longer matches the registry is drift."""
        current = INDEX.read_text(encoding="utf-8")
        heading = re.search(r"^#### `([a-z_0-9]+)`\n\n(\S+)", current, re.MULTILINE)
        assert heading, "the index has no tool entries"
        stale = current.replace(heading.group(0), heading.group(0) + " formerly", 1)
        problems = index_generator.drift(stale, current)
        assert problems and "formerly" in problems[0]

    def test_a_changed_argument_list_fails_the_check(self, index_generator):
        current = INDEX.read_text(encoding="utf-8")
        stale = current.replace("**Optional:** `", "**Optional:** `retired`, `", 1)
        assert stale != current
        assert index_generator.drift(stale, current)

    @pytest.mark.parametrize("delta", [-1, 1])
    def test_a_one_kb_difference_in_a_schema_cost_passes(self, index_generator, delta):
        """What a different pydantic release can do to a runtime that sits
        near half a kilobyte, with no tool changed."""
        current = INDEX.read_text(encoding="utf-8")
        shifted, count = re.subn(
            r"^(\| `modeling` \| \d+ \| )(\d+)( KB \|)",
            lambda m: f"{m.group(1)}{int(m.group(2)) + delta}{m.group(3)}",
            current,
            flags=re.MULTILINE,
        )
        assert count == 1, "the runtimes table has no `modeling` row"
        assert index_generator.drift(shifted, current) == []

    @pytest.mark.parametrize("delta", [-2, 2])
    def test_a_larger_schema_cost_difference_fails(self, index_generator, delta):
        current = INDEX.read_text(encoding="utf-8")
        shifted = re.sub(
            r"^(\| `modeling` \| \d+ \| )(\d+)( KB \|)",
            lambda m: f"{m.group(1)}{int(m.group(2)) + delta}{m.group(3)}",
            current,
            flags=re.MULTILINE,
        )
        problems = index_generator.drift(shifted, current)
        assert len(problems) == 1 and "`modeling` schema cost" in problems[0]

    def test_the_generator_writes_lf_and_check_writes_nothing(
        self, index_generator, tmp_path, monkeypatch
    ):
        """`.gitattributes` stores the index with LF; a CRLF write on Windows
        read as a modification in `git status`. `--check` reports drift
        through its exit status and leaves the file alone."""
        text = INDEX.read_text(encoding="utf-8")
        monkeypatch.setattr(index_generator, "render", lambda: text)
        written = tmp_path / "index.md"
        assert index_generator.main(["--output", str(written)]) == 0
        assert written.read_bytes() == text.encode("utf-8")
        assert b"\r" not in written.read_bytes()

        assert index_generator.main(["--check", "--output", str(written)]) == 0
        stale = text.replace("#### `", "#### `x", 1)
        written.write_bytes(stale.encode("utf-8"))
        assert index_generator.main(["--check", "--output", str(written)]) == 1
        assert written.read_bytes() == stale.encode("utf-8")

    def test_every_tool_appears_in_the_index(self, all_tools):
        text = INDEX.read_text(encoding="utf-8")
        missing = sorted(t for t in all_tools if f"#### `{t}`" not in text)
        assert not missing, f"absent from the generated index: {missing}"

    def test_the_index_holds_no_tool_that_does_not_exist(self, all_tools):
        """The other direction: a renamed tool leaves a ghost entry, and a
        reader calling it gets an error the index caused."""
        text = INDEX.read_text(encoding="utf-8")
        listed = set(re.findall(r"^#### `([a-z_0-9]+)`$", text, re.MULTILINE))
        ghosts = sorted(listed - all_tools)
        assert not ghosts, f"documented but no longer exist: {ghosts}"

    def test_every_runtime_has_a_section(self, runtimes):
        text = INDEX.read_text(encoding="utf-8")
        missing = [n for n in runtimes if f"## `{n}` —" not in text]
        assert not missing, f"runtimes with no section: {missing}"


class TestEveryToolIsDiscoverable:
    def test_no_tool_is_undocumented(self, all_tools, doc_text):
        """
        A tool nobody can find is a tool nobody uses. This was 85 of 157
        before the index existed.
        """
        missing = sorted(t for t in all_tools if t not in doc_text)
        assert not missing, (
            f"{len(missing)} tools appear in no documentation at all: "
            f"{missing[:15]}"
        )

    def test_every_runtime_is_named_in_the_readme(self, runtimes):
        text = README.read_text(encoding="utf-8")
        missing = [n for n in runtimes if f"`{n}`" not in text]
        assert not missing, f"runtimes absent from the README: {missing}"


class TestCountsInProseAreReal:
    """
    A number in a sentence is a claim. This library's whole position is
    that claims get measured, and a README asserting a tool count it does
    not have is the cheapest possible violation of it.
    """

    def test_the_readme_tool_count_is_current(self, all_tools):
        text = README.read_text(encoding="utf-8")
        stated = re.findall(r"\*\*(\d+) LLM-callable tools\*\*", text)
        assert stated, "the README no longer states a tool count"
        assert int(stated[0]) == len(
            all_tools
        ), f"README says {stated[0]} tools, the registry has {len(all_tools)}"

    def test_the_readme_runtime_count_is_current(self, runtimes):
        text = README.read_text(encoding="utf-8")
        words = {
            4: "four",
            5: "five",
            6: "six",
            7: "seven",
            8: "eight",
            9: "nine",
            10: "ten",
        }
        expected = words.get(len(runtimes))
        assert expected, f"no word for {len(runtimes)} runtimes; extend the map"
        assert f"**{expected} parallel runtimes**" in text, (
            f"the README does not say '{expected} parallel runtimes' for the "
            f"{len(runtimes)} that exist"
        )

    def test_every_whole_surface_count_is_current(self, all_tools):
        """
        The rot that actually happened: three guides each quoting a
        whole-surface count from whenever they were last touched. Matched
        only on phrasings that clearly mean the WHOLE surface -- a sentence
        about one category holding twelve tools is a different claim, and
        is checked separately below.
        """
        stale = []
        for doc in sorted(DOCS.glob("*.md")) + [README]:
            text = doc.read_text(encoding="utf-8", errors="ignore")
            for phrase in re.findall(
                r"(?:The|Every one of the|Serving all) (\d{2,3}) tools", text
            ):
                if int(phrase) not in (len(all_tools), len(all_tools) - 2):
                    stale.append(f"{doc.name}: 'the {phrase} tools'")
        assert not stale, (
            f"stale whole-surface counts; the registry has {len(all_tools)} "
            f"({len(all_tools) - 2} served by default): {stale}"
        )

    def test_every_runtime_count_quoted_in_a_command_is_current(self, runtimes):
        """
        18_mcp.md shows `sqt-mcp --runtime X  # N tools, K KB` lines. Each
        is a claim a reader will act on, and each was between 20% and 80%
        wrong before this test existed.
        """
        wrong = []
        for doc in sorted(DOCS.glob("*.md")) + [README]:
            text = doc.read_text(encoding="utf-8", errors="ignore")
            for name, count in re.findall(
                r"--runtime ([a-z_]+)\s+#\s*(\d+) tools", text
            ):
                if name in runtimes and int(count) != len(runtimes[name]):
                    wrong.append(
                        f"{doc.name}: --runtime {name} says {count}, "
                        f"actual {len(runtimes[name])}"
                    )
        assert not wrong, wrong

    def test_every_serves_n_tools_message_is_current(self, runtimes):
        """The refusal message quoted in the docs is one a reader compares
        against what they actually see."""
        wrong = []
        for doc in sorted(DOCS.glob("*.md")):
            text = doc.read_text(encoding="utf-8", errors="ignore")
            for count, name in re.findall(
                r"serves (\d+) tools from the ([a-z_]+) runtime", text
            ):
                if name in runtimes and int(count) != len(runtimes[name]):
                    wrong.append(
                        f"{doc.name}: says {count} for {name}, "
                        f"actual {len(runtimes[name])}"
                    )
        assert not wrong, wrong

    #: The documents whose opening sentence is a claim about ONE runtime's
    #: size. Not every "N tools" opener is: `23_inference.md` and
    #: `24_overfitting.md` describe a SUBSET of `research` and `backtest`,
    #: and `05_portfolio.md`/`12_options.md` say "Three tools"/"Two tools"
    #: mid-document about a handful. Those are checked by the
    #: self-consistency guard below instead, which needs no map.
    RUNTIME_DOCS = {
        "21_derivatives.md": "derivatives",
        "22_microstructure.md": "microstructure",
        "26_data.md": "data",
        "27_meta.md": "meta",
        "28_delta_one.md": "delta_one",
    }

    #: Enough to name any runtime this library will plausibly have. A count
    #: past twenty means the map needs extending, and the test says so
    #: rather than passing silently.
    NUMBER_WORDS = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
        "thirteen": 13,
        "fourteen": 14,
        "fifteen": 15,
        "sixteen": 16,
        "seventeen": 17,
        "eighteen": 18,
        "nineteen": 19,
        "twenty": 20,
        "twenty-one": 21,
        "twenty-two": 22,
        "twenty-three": 23,
        "twenty-four": 24,
        "twenty-five": 25,
        "twenty-six": 26,
        "twenty-seven": 27,
        "twenty-eight": 28,
        "twenty-nine": 29,
        "thirty": 30,
        "thirty-one": 31,
        "thirty-two": 32,
        "thirty-three": 33,
        "thirty-four": 34,
        "thirty-five": 35,
        "thirty-six": 36,
        "thirty-seven": 37,
        "thirty-eight": 38,
        "thirty-nine": 39,
        "forty": 40,
        "forty-one": 41,
        "forty-two": 42,
        "forty-three": 43,
        "forty-four": 44,
        "forty-five": 45,
    }

    def test_every_spelled_out_runtime_count_is_current(self, runtimes):
        r"""
        THE HOLE EVERY OTHER GUARD IN THIS CLASS LEFT OPEN.

        The rest match `(\d+)`. Every runtime guide opens by SPELLING its
        count -- "Eighteen tools for the instruments that move one-for-one"
        -- so a runtime could grow and its own front page keep the old
        number indefinitely. Measured when this was written: five of these
        documents were wrong at once, `22_microstructure.md` saying fifteen
        for sixteen and `23_inference.md` saying twenty over a 19-row
        table, and the suite was green on all of them.
        """
        stale = []
        for name, runtime in self.RUNTIME_DOCS.items():
            text = (DOCS / name).read_text(encoding="utf-8")
            match = re.search(r"^([A-Z][a-z]+(?:-[a-z]+)?) tools\b", text, re.M)
            if match is None:
                stale.append(f"{name}: no spelled-out opening count")
                continue
            word = match.group(1).lower()
            stated = self.NUMBER_WORDS.get(word)
            if stated is None:
                stale.append(f"{name}: no entry for {word!r}; extend NUMBER_WORDS")
                continue
            actual = len(runtimes[runtime])
            if stated != actual:
                stale.append(
                    f"{name}: opens '{match.group(1)} tools' for the "
                    f"{runtime} runtime, which has {actual}"
                )
        assert not stale, stale

    def test_a_runtime_guide_lists_every_tool_it_owns(self, runtimes):
        """
        Stronger than the count, and it catches what the count cannot: a
        guide can say the right NUMBER while omitting a tool and listing
        something else. Both real failures here were of that shape --
        `get_order_book_metrics` and `build_continuous_futures_series` were
        each absent from their own runtime's table while the opener was
        merely off by one, so fixing the number alone left the tool
        undocumented.
        """
        missing = []
        for name, runtime in self.RUNTIME_DOCS.items():
            text = (DOCS / name).read_text(encoding="utf-8")
            listed = set(re.findall(r"^\|\s*`([a-z_][a-z0-9_]*)`\s*\|", text, re.M))
            for tool in sorted(set(runtimes[runtime].tool_names) - listed):
                missing.append(f"{name}: no table row for `{tool}`")
        assert not missing, missing

    #: Guides that cover a SUBSET of a runtime, so their opening count is a
    #: claim about their own table rather than about the runtime.
    #: `23_inference.md` documents 19 of `research`'s 42; `24_overfitting.md`
    #: 11 of `backtest`'s 35. The check is therefore self-consistency, which
    #: needs no expected number maintained anywhere.
    SUBSET_DOCS = ("23_inference.md", "24_overfitting.md")

    def test_a_subset_guide_opens_with_the_size_of_its_own_table(self, all_tools):
        """
        `23_inference.md` opened "Twenty tools" above a table of nineteen.
        Nothing checked it, because the runtime it draws from has 42 and no
        guard compares a document against ITSELF.
        """
        wrong = []
        for name in self.SUBSET_DOCS:
            text = (DOCS / name).read_text(encoding="utf-8")
            match = re.search(r"^([A-Z][a-z]+(?:-[a-z]+)?) tools", text, re.M)
            assert match, f"{name}: no spelled-out opening count"
            stated = self.NUMBER_WORDS.get(match.group(1).lower())
            assert stated, f"{name}: extend NUMBER_WORDS for {match.group(1)!r}"
            rows = {
                tool
                for tool in re.findall(r"^\|\s*`([a-z_][a-z0-9_]*)`\s*\|", text, re.M)
                if tool in all_tools
            }
            if stated != len(rows):
                wrong.append(
                    f"{name}: opens '{match.group(1)} tools' but its table "
                    f"lists {len(rows)}"
                )
        assert not wrong, wrong

    @pytest.mark.parametrize("source", ["19_runtimes.md", "README.md"])
    def test_the_runtime_table_counts_match(self, runtimes, source):
        """
        Both files carry a per-runtime table, and every row is a claim about
        a number that changes whenever a tool is added.

        The README was NOT checked here until it had drifted on three rows
        at once -- `data` said 14 against 17, `modeling` 17 against 20,
        `microstructure` 16 against 17. One table being guarded and an
        identical one beside it not being guarded is how that happens.
        """
        path = README if source == "README.md" else DOCS / source
        text = path.read_text(encoding="utf-8")
        wrong = []
        for name, rt in runtimes.items():
            row = re.search(rf"^\| `{name}` \| (\d+) \|", text, re.MULTILINE)
            if row is None:
                wrong.append(f"{name}: no row")
            elif int(row.group(1)) != len(rt):
                wrong.append(f"{name}: doc says {row.group(1)}, actual {len(rt)}")
        assert not wrong, f"{source}: {wrong}"


class TestLinksResolve:
    def test_every_relative_markdown_link_points_at_a_real_file(self):
        """
        A link to a renamed file is a dead end a reader hits and an author
        never does.
        """
        broken = []
        for source in [README, *sorted(DOCS.glob("*.md"))]:
            text = source.read_text(encoding="utf-8", errors="ignore")
            for target in re.findall(r"\]\(([^)#:]+\.md)(?:#[^)]*)?\)", text):
                resolved = (source.parent / target).resolve()
                if not resolved.exists():
                    broken.append(f"{source.name} -> {target}")
        assert not broken, f"broken links: {broken}"

    def test_the_readme_documentation_table_lists_every_guide(self):
        """A guide nobody links to is a guide nobody reads."""
        text = README.read_text(encoding="utf-8")
        missing = [
            p.name
            for p in sorted(DOCS.glob("*.md"))
            if f"Documentation/{p.name}" not in text
        ]
        assert not missing, f"guides absent from the README's table: {missing}"


class TestTheDescriptionsAreUsable:
    """
    The descriptions in the index are the strings a model reads when
    choosing a tool. These are the floor they have to clear.
    """

    def test_no_description_is_a_bare_restatement_of_the_name(self, runtimes):
        thin = []
        for rt in runtimes.values():
            for name, description, _model in rt.tool_defs:
                if len(description) < 60:
                    thin.append(f"{name} ({len(description)} chars)")
        assert not thin, (
            "descriptions too short to choose between this many tools -- say when "
            f"to reach for it and how it fails: {thin}"
        )

    def test_every_description_is_one_paragraph_of_prose(self, runtimes):
        """A description that is a bullet list or contains a newline renders
        unpredictably across function-calling clients."""
        bad = [
            name
            for rt in runtimes.values()
            for name, description, _ in rt.tool_defs
            if "\n" in description
        ]
        assert not bad, f"descriptions containing newlines: {bad}"

    def test_no_description_promises_a_data_source_the_library_lacks(self, runtimes):
        """
        The derivatives tools take quotes as arguments because there is no
        options provider. A description implying otherwise would send an
        agent looking for a chain-fetching tool that does not exist.
        """
        from standard_quant_tools.agent.runtimes import resolve

        for name, description, _ in resolve("derivatives").tool_defs:
            lowered = description.lower()
            assert "fetch the chain" not in lowered
            assert "download the option" not in lowered


class TestNoStaleWholeSurfaceCountSurvivesInAnyPhrasing:
    """
    THE ROT THIS EXISTS TO CATCH, and why the previous guard missed it.

    `test_every_whole_surface_count_is_current` matched three phrasings --
    "The N tools", "Every one of the N tools", "Serving all N tools". The
    docs also say "returns 174 LLM-callable tools", "all 174 tools", "the
    174-tool surface" and "index of all 200 tools", and every one of those
    rotted undetected across several releases while the guard passed.

    This checks the SHAPE instead of the phrasing: any three-digit count of
    tools anywhere in the documentation must be one of the two real surface
    sizes. That works because every scoped count is two digits -- the
    largest runtime is `research` at 42 -- so a three-digit number in this
    library is always a claim about the whole surface or about the analysis
    facade, and there is no third option to confuse it with.

    Genuine exceptions are declared with a reason rather than pattern-matched
    away, the same discipline `EXPECTED_UNSYNTHESIZABLE` applies, so the
    list cannot quietly become a place stale numbers go to hide.
    """

    #: (file, number) -> why this three-digit count is not a surface claim.
    ALLOWED = {
        ("25_testing.md", 157): "history: the surface size BEFORE tests/docs "
        "existed, in a sentence that says so",
    }

    def test_every_three_digit_tool_count_is_a_real_surface(self, all_tools):
        from standard_quant_tools.agent.tools import TOOL_CATEGORY

        real = {len(all_tools), len(TOOL_CATEGORY)}
        stale = []
        for doc in sorted(DOCS.glob("*.md")) + [README]:
            text = doc.read_text(encoding="utf-8", errors="ignore")
            for match in re.finditer(r"(\d{3})[ -]tools?\b", text):
                value = int(match.group(1))
                if value in real:
                    continue
                if (doc.name, value) in self.ALLOWED:
                    continue
                line = text[: match.start()].count("\n") + 1
                stale.append(f"{doc.name}:{line} says {value}")
        assert not stale, (
            f"the surface is {sorted(real)} tools (whole, then analysis "
            f"facade); these say otherwise: {stale}"
        )

    def test_the_allowlist_holds_no_entry_that_is_now_a_real_count(self, all_tools):
        """An exemption that became true is an exemption nobody needs."""
        from standard_quant_tools.agent.tools import TOOL_CATEGORY

        real = {len(all_tools), len(TOOL_CATEGORY)}
        stale = [key for key in self.ALLOWED if key[1] in real]
        assert not stale, f"remove these, they are real counts now: {stale}"

    def test_the_allowlist_is_still_needed(self):
        """A declared exception whose text has gone is dead weight."""
        unused = []
        for (name, value), reason in self.ALLOWED.items():
            doc = DOCS / name if (DOCS / name).exists() else README
            text = doc.read_text(encoding="utf-8", errors="ignore")
            if not re.search(rf"{value}[ -]tools?\b", text):
                unused.append(f"{name}:{value} ({reason})")
        assert not unused, f"no longer present, so delete: {unused}"


MODELING_GENERATOR = ROOT / "scripts" / "generate_modeling_reference.py"
MODELING_REFERENCE = DOCS / "29_modeling_reference.md"


class TestTheModelingReferenceIsGenerated:
    """
    The modeling guide carried a hand-written feature table that said 21
    entries when the registry held 23. The catalog is generated now, on
    the same terms as the tool index: render and compare, so a feature,
    estimator or target added without regenerating fails in its own commit.
    """

    def test_the_reference_matches_the_live_registries(self):
        """Rendered in this process, from the package these tests import,
        and never written to disk, as the tool index is (see the CHANGELOG
        entry of 2026-10-04)."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "generate_modeling_reference", MODELING_GENERATOR
        )
        assert spec and spec.loader, f"could not load {MODELING_GENERATOR}"
        generator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(generator)
        on_disk = MODELING_REFERENCE.read_text(encoding="utf-8")
        assert on_disk == generator.render(), (
            "Documentation/29_modeling_reference.md is out of date with the "
            "modeling registries. Run "
            "`python scripts/generate_modeling_reference.py` and commit "
            "the result -- the reference is generated, not written."
        )

    def test_every_feature_and_target_is_in_it(self):
        from standard_quant_tools.modeling.features.registry import FEATURE_REGISTRY
        from standard_quant_tools.modeling.specs import TARGET_KINDS

        text = MODELING_REFERENCE.read_text(encoding="utf-8")
        missing = [
            f"`{name}`"
            for name in [*FEATURE_REGISTRY, *TARGET_KINDS]
            if f"`{name}`" not in text
        ]
        assert not missing, f"absent from the generated reference: {missing}"

    def test_every_optional_estimator_is_in_it_whether_or_not_installed(self):
        """The document is the same on every machine, so it must name the
        estimators this machine may not have."""
        from standard_quant_tools.modeling.estimators.boosting import (
            OPTIONAL_ESTIMATORS,
        )

        text = MODELING_REFERENCE.read_text(encoding="utf-8")
        missing = [
            name for _task, name in OPTIONAL_ESTIMATORS if f"`{name}`" not in text
        ]
        assert not missing, missing


class TestNoCountIsHardcodedInTheModelingSource:
    """
    Seven docstrings under `modeling/` quoted a 6-tool, 46-tool or 16-tool
    surface after the counts had moved, and nothing checked them because
    the count guards above read the documentation, not the source. A count
    in a docstring is a claim that rots on exactly the same schedule; the
    fix is to say "the modeling surface" and let the generated index carry
    the number.
    """

    SOURCES = (
        ROOT / "src" / "standard_quant_tools" / "modeling",
        ROOT / "Multi_Agent_Implementation" / "worker_agents.py",
    )
    PATTERN = re.compile(
        r"\b(?:\d+|five|six|eight|sixteen|seventeen|twenty)-(?:tool|entry)\b",
        re.IGNORECASE,
    )

    def test_no_n_tool_or_n_entry_phrase_survives(self):
        stale = []
        for source in self.SOURCES:
            files = [source] if source.is_file() else sorted(source.rglob("*.py"))
            for path in files:
                text = path.read_text(encoding="utf-8", errors="ignore")
                for match in self.PATTERN.finditer(text):
                    line = text[: match.start()].count("\n") + 1
                    stale.append(f"{path.relative_to(ROOT)}:{line} {match.group(0)!r}")
        assert not stale, f"counts hardcoded in source: {stale}"
