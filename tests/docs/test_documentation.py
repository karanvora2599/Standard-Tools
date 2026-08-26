"""
The documentation is checked, not trusted.

Documentation rots in a way code does not: nothing fails when it goes
stale. Before these tests existed, 85 of 157 tools appeared in no document
at all, the README advertised 95 tools across six runtimes when there were
157 across eight, and three separate guides quoted a tool count that had
been wrong for months. None of that broke a single test, because nothing
was checking.

WHAT IS ENFORCED HERE:

- **The tool index is generated, and current.** It is rebuilt from the live
  registry and compared byte-for-byte, so adding a tool without
  regenerating fails in the same commit that added it.
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
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
DOCS = ROOT / "Documentation"
README = ROOT / "README.md"
GENERATOR = ROOT / "Development" / "generate_tool_index.py"
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


class TestTheToolIndexIsGenerated:
    def test_the_index_matches_the_live_registry(self):
        """
        Regenerate and compare. This is the mechanism that makes the other
        documentation tests unnecessary for the index itself: it cannot be
        stale, because staleness is a test failure rather than a thing
        someone has to notice.
        """
        before = INDEX.read_text(encoding="utf-8")
        result = subprocess.run(
            [sys.executable, str(GENERATOR)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode == 0, result.stderr
        after = INDEX.read_text(encoding="utf-8")
        assert before == after, (
            "Documentation/20_tool_index.md is out of date with the tool "
            "registry. Run `python Development/generate_tool_index.py` and "
            "commit the result -- the index is generated, not written."
        )

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

    def test_the_runtime_table_counts_match(self, runtimes):
        """19_runtimes.md carries a per-runtime table, and every row is a
        claim about a number that changes whenever a tool is added."""
        text = (DOCS / "19_runtimes.md").read_text(encoding="utf-8")
        wrong = []
        for name, rt in runtimes.items():
            row = re.search(rf"^\| `{name}` \| (\d+) \|", text, re.MULTILINE)
            if row is None:
                wrong.append(f"{name}: no row")
            elif int(row.group(1)) != len(rt):
                wrong.append(f"{name}: doc says {row.group(1)}, actual {len(rt)}")
        assert not wrong, wrong


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
