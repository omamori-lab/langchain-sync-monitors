"""Every docs page has the shape the site's layout depends on.

The docs standard asks these things of each page, checked on the page as the
site renders it:

- exactly one level-1 heading, the page's title;
- the page opens with its title, one opening paragraph and then the contents
  list that ``[TOC]`` renders, which the site shows as a numbered box;
- every heading sits at the top level of the page, never inside an admonition,
  a list, a quote, a footnote or a raw HTML block: the site's stylesheet
  numbers only the h2 and h3 at the top level, while the contents lists list
  every heading, so a nested one puts the two numberings out of step;
- no number typed into a heading, since the stylesheet numbers the sections
  itself;
- a page that cites a source ends with a ``## References`` section, under
  which the site prints that page's bibliography;
- no citation inside code: mkdocs-bibtex [@mkdocsbibtex2025] rewrites every
  bracketed ``@key`` in a page's source, code included, so a code sample that
  shows one is corrupted and the page gains a bibliography.

Two kinds of page need no contents list. The API reference is generated from
the docstrings, and its right-hand contents list already indexes every class
and function; a ``[TOC]`` would repeat that whole index above the first entry.
A section index page, such as ``how-to/index.md``, is itself the list of its
section's pages, and most have no sections of their own to list.

A page is read as the site reads it, with the libraries the site uses rather
than a parser of our own. MkDocs strips the front matter with its own
``get_data`` [@mkdocs2024], which keeps a ``---`` block that YAML does not load
as a mapping. Python-Markdown renders the rest with the extensions and settings
of ``mkdocs.yml``, read with MkDocs' own config loader. Two things of the full
build are left out: MkDocs' private treeprocessors, which only collect anchors
and the title, and the plugins, so a ``:::`` directive stays as text. The
checks read the rendered HTML with the standard library's ``html.parser``, and
find citations with mkdocs-bibtex's own ``CitationBlock``, so a citation counts
exactly when the plugin would rewrite it, ``[see @key, p. 3]`` included.

A typed number is a leading number or Roman numeral followed by ``.``, ``)`` or
``:`` and a space (``## 2. Options``, ``## 2) Options``, ``## 2.1. Options``,
``## II. Options``), a number in parentheses (``## (2) Options``), or "Step"
and a number at the start (``## Step 1: the tools``). The rule reads the
heading's rendered text, so bold, code or a link around the number changes
nothing. It is narrow on purpose, and the trade-off is:

- ``## 2.1 Thresholds`` passes, the price of passing ``## 3.5 Sonnet as a
  monitor``, ``## 0.1 threshold`` and ``## 1:1 mapping``;
- ``## Stage 2 of a cascade``, ``## Section 2.1 of the paper`` and ``## Phase 2
  trials`` pass, since there the number names a thing rather than a section;
- Roman numerals are I, V and X only, so ``## CLI: the command line`` passes,
  but ``## X: the format`` is flagged.

The renderer keeps no line numbers. To name a heading's line, the check appends
a marker to each source line that holds the heading's words and renders again:
the line whose marker lands in the heading is its line. A citation inside code
is found the same way. A message says when no line takes the marker, as for a
heading written in raw HTML.

Rendering needs the docs dependency group. Without it, as in the dev-only CI
jobs, the tests that render are skipped; ``scripts/check.sh`` sets
``REQUIRE_DOCS_GROUP=1``, which turns that skip into a failure. The plans under
``docs/plans/`` are working notes, not pages.

Every page complies, so the page check fails the run on the first page that
breaks the standard, and its message lists every violation.
"""

from __future__ import annotations

import functools
import importlib
import os
import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from types import ModuleType
from typing import override

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIRECTORY = REPOSITORY_ROOT / "docs"
NOTES_DIRECTORY = DOCS_DIRECTORY / "plans"
CONFIG_PATH = REPOSITORY_ROOT / "mkdocs.yml"
CHECK_SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "check.sh"
REQUIRE_DOCS_GROUP_VARIABLE = "REQUIRE_DOCS_GROUP"
REFERENCES_TITLE = "References"
PAGES_WITHOUT_CONTENTS_LIST = frozenset(
    {
        "reference/api.md",
        "tutorials/index.md",
        "how-to/index.md",
        "reference/index.md",
        "explanation/index.md",
    },
)
HEADING_TAGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})
VOID_TAGS = frozenset(
    {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "wbr"},
)
PERMALINK_CLASS = "headerlink"
CONTENTS_CLASS = "toc"
TYPED_NUMBER_PATTERN = re.compile(
    r"(?:\d+(?:\.\d+)*|[IVX]+)[.):]\s|\(\d+\)\s|(?i:step)\s+\d",
)
# Plain letters, so appending it to a line never changes how the line parses.
LINE_MARKER = "qzxlinemarkerqzx"


@dataclass(slots=True, kw_only=True)
class Element:
    """One element of a rendered page: its tag, its classes, and its children in order."""

    tag: str
    classes: frozenset[str]
    children: list[Element | str] = field(default_factory=list)


@dataclass(frozen=True, slots=True, kw_only=True)
class Heading:
    """One rendered heading: its level, its text, and the top-level element that holds it.

    `container` is empty for a heading at the top level of the page, and names
    the enclosing element otherwise, such as ``blockquote`` or
    ``div.admonition.note``.
    """

    level: int
    text: str
    container: str

    @property
    def nested(self) -> bool:
        """Tell whether the heading sits inside another element."""
        return bool(self.container)


@dataclass(frozen=True, slots=True, kw_only=True)
class PageSource:
    """The Markdown the renderer sees, and the page line its first line is on.

    `first_line` is None when the body cannot be matched back to the page.
    """

    body: str
    first_line: int | None


@dataclass(frozen=True, slots=True, kw_only=True)
class MarkdownSettings:
    """The Markdown extensions ``mkdocs.yml`` names, with their settings, as MkDocs loads them."""

    extensions: list[str]
    extension_configs: dict[str, dict[str, object]]


class HtmlTreeBuilder(HTMLParser):
    """Build a tree of elements from HTML, tolerating the unclosed tags raw HTML may leave."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = Element(tag="body", classes=frozenset())
        self.open_elements: list[Element] = [self.root]

    @override
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        element = build_element(tag, attributes=attrs)
        self.open_elements[-1].children.append(element)
        if tag not in VOID_TAGS:
            self.open_elements.append(element)

    @override
    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.open_elements[-1].children.append(build_element(tag, attributes=attrs))

    @override
    def handle_endtag(self, tag: str) -> None:
        for depth in range(len(self.open_elements) - 1, 0, -1):
            if self.open_elements[depth].tag == tag:
                del self.open_elements[depth:]
                return

    @override
    def handle_data(self, data: str) -> None:
        self.open_elements[-1].children.append(data)


def build_element(tag: str, *, attributes: list[tuple[str, str | None]]) -> Element:
    """Return an empty element with the classes its attributes give it."""
    classes = next((value or "" for name, value in attributes if name == "class"), "")
    return Element(tag=tag, classes=frozenset(classes.split()))


def parse_html(html: str) -> Element:
    """Return the root of the tree the HTML of a page's content builds."""
    builder = HtmlTreeBuilder()
    builder.feed(html)
    builder.close()
    return builder.root


def read_child_elements(parent: Element) -> list[Element]:
    """Return an element's child elements, in order, leaving out its text."""
    return [child for child in parent.children if isinstance(child, Element)]


def iterate_elements(parent: Element) -> Iterator[Element]:
    """Yield every element below `parent`, in document order."""
    for child in read_child_elements(parent):
        yield child
        yield from iterate_elements(child)


def read_text(element: Element) -> str:
    """Return the text of an element, leaving out the permalink anchor the toc extension adds."""
    parts = [
        child if isinstance(child, str) else read_text(child)
        for child in element.children
        if isinstance(child, str) or PERMALINK_CLASS not in child.classes
    ]
    return "".join(parts).strip()


def describe_element(element: Element) -> str:
    """Name an element by its tag and classes, as a CSS selector does: ``div.admonition.note``."""
    return ".".join([element.tag, *sorted(element.classes)])


def build_heading(element: Element, *, container: str) -> Heading:
    """Return the heading an ``h1`` to ``h6`` element renders."""
    return Heading(level=int(element.tag[1]), text=read_text(element), container=container)


def read_headings(root: Element) -> list[Heading]:
    """Return every heading of a rendered page, in document order."""
    headings: list[Heading] = []
    for top_level in read_child_elements(root):
        if top_level.tag in HEADING_TAGS:
            headings.append(build_heading(top_level, container=""))
            continue
        container = describe_element(top_level)
        headings.extend(
            build_heading(element, container=container)
            for element in iterate_elements(top_level)
            if element.tag in HEADING_TAGS
        )
    return headings


def has_contents_list_after_opening(root: Element) -> bool:
    """Tell whether a page opens with its title, one paragraph, and then the contents list."""
    opening = read_child_elements(root)[:3]
    tags = [element.tag for element in opening]
    return tags == ["h1", "p", "div"] and CONTENTS_CLASS in opening[-1].classes


def describe_opening(root: Element) -> str:
    """Name the first three elements a page opens with, such as ``<h1>, <p>, <p>``."""
    opening = read_child_elements(root)[:3]
    return ", ".join(f"<{element.tag}>" for element in opening) or "nothing"


def is_typed_number(text: str) -> bool:
    """Tell whether a heading's text starts with a number someone typed."""
    return TYPED_NUMBER_PATTERN.match(text) is not None


def ends_with_references(headings: list[Heading]) -> bool:
    """Tell whether the page's last top-level h2 section is the References section."""
    sections = [heading.text for heading in headings if heading.level == 2 and not heading.nested]
    return bool(sections) and sections[-1] == REFERENCES_TITLE


def read_code_texts(root: Element) -> list[str]:
    """Return the text of every code element of a rendered page, inline code included."""
    return [read_text(element) for element in iterate_elements(root) if element.tag == "code"]


def normalise_words(text: str) -> str:
    """Return only the letters and digits of a text, in lower case, to compare it loosely."""
    return "".join(character for character in text.casefold() if character.isalnum())


def import_docs_module(name: str) -> ModuleType:
    """Import a module of the docs dependency group, skipping the test when it is missing.

    With ``REQUIRE_DOCS_GROUP=1`` set, as ``scripts/check.sh`` sets it, a
    missing module fails the test instead.
    """
    if os.environ.get(REQUIRE_DOCS_GROUP_VARIABLE) == "1":
        return importlib.import_module(name)
    return pytest.importorskip(name)


@functools.cache
def load_markdown_settings() -> MarkdownSettings:
    """Return the site's Markdown extensions and settings, read by MkDocs from ``mkdocs.yml``."""
    config_module = import_docs_module("mkdocs.config")
    config = config_module.load_config(config_file=str(CONFIG_PATH))
    return MarkdownSettings(
        extensions=list(config["markdown_extensions"]),
        extension_configs=dict(config["mdx_configs"] or {}),
    )


def render_markdown(text: str) -> Element:
    """Render Markdown as the site does, and return the root of its HTML."""
    settings = load_markdown_settings()
    python_markdown = import_docs_module("markdown")
    renderer = python_markdown.Markdown(
        extensions=settings.extensions,
        extension_configs=settings.extension_configs,
    )
    return parse_html(renderer.convert(text))


def read_page_source(text: str) -> PageSource:
    """Strip a page's front matter as MkDocs does, keeping track of where its body starts."""
    meta = import_docs_module("mkdocs.utils.meta")
    body, _ = meta.get_data(text)
    if not text.endswith(body):
        return PageSource(body=body, first_line=None)
    return PageSource(body=body, first_line=text[: len(text) - len(body)].count("\n") + 1)


def find_citations(text: str) -> list[str]:
    """Return each citation mkdocs-bibtex finds in Markdown or code, as its bracketed text."""
    citation = import_docs_module("mkdocs_bibtex.citation")
    return [f"[{block.raw}]" for block in citation.CitationBlock.from_markdown(text)]


def mark_line(lines: list[str], *, number: int, marked: str) -> str:
    """Return the Markdown with line `number` (counted from zero) replaced by `marked`."""
    return "\n".join([*lines[:number], marked, *lines[number + 1 :]])


def find_heading_line(source: PageSource, *, index: int, heading: Heading) -> int | None:
    """Return the page line of the heading at `index` in render order, or None if none takes it."""
    words = normalise_words(heading.text)
    if source.first_line is None or not words:
        return None
    lines = source.body.split("\n")
    for number, line in enumerate(lines):
        if words not in normalise_words(line):
            continue
        marked = render_markdown(mark_line(lines, number=number, marked=f"{line} {LINE_MARKER}"))
        rendered = read_headings(marked)
        if index < len(rendered) and LINE_MARKER in rendered[index].text:
            return source.first_line + number
    return None


def find_citation_lines_in_code(source: PageSource, *, citation: str) -> list[int]:
    """Return the page lines where `citation` sits inside code."""
    if source.first_line is None:
        return []
    lines = source.body.split("\n")
    found: list[int] = []
    tagged = f"{citation}{LINE_MARKER}"
    for number, line in enumerate(lines):
        if citation not in line:
            continue
        marked = mark_line(lines, number=number, marked=line.replace(citation, tagged, 1))
        if any(tagged in text for text in read_code_texts(render_markdown(marked))):
            found.append(source.first_line + number)
    return found


def describe_line(number: int | None) -> str:
    """Name a page line, or say that the check could not find it."""
    return f"line {number}" if number is not None else "a line the check could not find"


def find_violations(text: str, *, page: str) -> list[str]:
    """Return what a page, named by its path under ``docs/``, lacks or breaks."""
    source = read_page_source(text)
    root = render_markdown(source.body)
    headings = read_headings(root)
    lines = [
        find_heading_line(source, index=index, heading=heading)
        if heading.level == 1 or heading.nested or is_typed_number(heading.text)
        else None
        for index, heading in enumerate(headings)
    ]
    violations: list[str] = []
    titles = [
        (line, heading) for line, heading in zip(lines, headings, strict=True) if heading.level == 1
    ]
    if not titles:
        violations.append(f"{page}: no level-1 heading, but a page has exactly one, its title")
    elif len(titles) > 1:
        found = "; ".join(f"{describe_line(line)} {heading.text!r}" for line, heading in titles)
        violations.append(
            f"{page}: {len(titles)} level-1 headings, but a page has exactly one, "
            f"its title: {found}",
        )
    if page not in PAGES_WITHOUT_CONTENTS_LIST and not has_contents_list_after_opening(root):
        violations.append(
            f"{page}: no [TOC] right after the title and the opening paragraph; "
            f"the page opens with {describe_opening(root)}",
        )
    violations.extend(
        f"{page}: the heading on {describe_line(line)} is inside <{heading.container}>, where "
        f"the site's section numbers skip it: {heading.text!r}"
        for line, heading in zip(lines, headings, strict=True)
        if heading.nested
    )
    violations.extend(
        f"{page}: the heading on {describe_line(line)} starts with a typed number: {heading.text!r}"
        for line, heading in zip(lines, headings, strict=True)
        if is_typed_number(heading.text)
    )
    code_citations = dict.fromkeys(
        citation for code in read_code_texts(root) for citation in find_citations(code)
    )
    for citation in code_citations:
        citation_lines = find_citation_lines_in_code(source, citation=citation) or [None]
        violations.extend(
            f"{page}: {describe_line(line)} cites {citation} inside code, where "
            "mkdocs-bibtex still rewrites it into a footnote"
            for line in citation_lines
        )
    if find_citations(source.body) and not ends_with_references(headings):
        violations.append(f"{page}: cites a source, so its last section must be ## References")
    return violations


def read_docs_pages() -> list[Path]:
    """Return every Markdown page of the site, the plans left out, in path order."""
    return sorted(
        path for path in DOCS_DIRECTORY.rglob("*.md") if NOTES_DIRECTORY not in path.parents
    )


def build_page(body: str) -> str:
    """Return a page with a title, an opening line and the contents marker, then `body`."""
    return f"# Use a thing\n\nThis guide sets up a thing.\n\n[TOC]\n\n{body}"


def render_headings(text: str) -> list[tuple[int, str, bool]]:
    """Render Markdown as the site does and return each heading's level, text and nesting."""
    return [
        (heading.level, heading.text, heading.nested)
        for heading in read_headings(render_markdown(text))
    ]


# The page check ---------------------------------------------------------------


def test_the_docs_hold_pages_to_check() -> None:
    # Act
    pages = {str(path.relative_to(DOCS_DIRECTORY)) for path in read_docs_pages()}

    # Assert
    assert {"index.md", "reference/api.md", "how-to/use-auto-mode.md"} <= pages
    assert not any(page.startswith("plans/") for page in pages)


def test_every_docs_page_meets_the_docs_standard() -> None:
    # Arrange
    pages = read_docs_pages()

    # Act
    violations = [
        violation
        for path in pages
        for violation in find_violations(
            path.read_text(encoding="utf-8"),
            page=str(path.relative_to(DOCS_DIRECTORY)),
        )
    ]

    # Assert
    assert violations == [], "Pages that break the docs standard:\n" + "\n".join(violations)


def test_the_page_check_is_never_expected_to_fail() -> None:
    # Arrange: every page complies, so a violation must fail the run, not pass as expected.
    marks = vars(test_every_docs_page_meets_the_docs_standard).get("pytestmark", [])

    # Act
    names = {mark.name for mark in marks}

    # Assert
    assert not names & {"xfail", "skip", "skipif"}


# The docs group: skipped where it is missing, required by the gate -------------


def read_missing_module_outcome() -> str:
    """Import a module that is missing, and say whether that skipped or failed the test."""
    try:
        import_docs_module("missing_docs_module")
    except ImportError:
        return "failed"
    except pytest.skip.Exception:
        return "skipped"
    return "imported"


@pytest.mark.parametrize(("required", "outcome"), [(None, "skipped"), ("1", "failed")])
def test_a_missing_docs_module_skips_unless_the_gate_requires_it(
    monkeypatch: pytest.MonkeyPatch,
    required: str | None,
    outcome: str,
) -> None:
    # Arrange
    monkeypatch.delenv(REQUIRE_DOCS_GROUP_VARIABLE, raising=False)
    if required is not None:
        monkeypatch.setenv(REQUIRE_DOCS_GROUP_VARIABLE, required)
    monkeypatch.setitem(sys.modules, "missing_docs_module", None)

    # Act
    found = read_missing_module_outcome()

    # Assert
    assert found == outcome


def test_the_gate_script_requires_the_docs_group_for_the_unit_tests() -> None:
    # Arrange
    script = CHECK_SCRIPT_PATH.read_text(encoding="utf-8")

    # Act
    test_lines = [line for line in script.splitlines() if "pytest" in line and "uv run" in line]

    # Assert
    assert test_lines, "scripts/check.sh no longer runs pytest"
    assert all(f"{REQUIRE_DOCS_GROUP_VARIABLE}=1" in line for line in test_lines)
    assert all("--group docs" in line for line in test_lines)


# Reading rendered HTML: no docs group needed -----------------------------------


def test_the_tree_keeps_nesting_and_tolerates_unclosed_tags() -> None:
    # Arrange: <br> and <hr> are void, and the raw <div> is never closed.
    html = '<p>One<br>two</p>\n<hr><div class="note warning"><h2>In</h2>\n<p>Tail &amp; more'

    # Act
    root = parse_html(html)

    # Assert
    paragraph, rule, block = read_child_elements(root)
    assert (rule.tag, rule.children) == ("hr", [])
    assert read_text(paragraph) == "Onetwo"
    assert describe_element(block) == "div.note.warning"
    assert [element.tag for element in read_child_elements(block)] == ["h2", "p"]
    assert read_text(block) == "In\nTail & more"


def test_heading_text_leaves_out_the_permalink() -> None:
    # Arrange
    html = '<h2 id="a"><strong>2.</strong> Options<a class="headerlink" href="#a">&para;</a></h2>'

    # Act
    headings = read_headings(parse_html(html))

    # Assert
    assert headings == [Heading(level=2, text="2. Options", container="")]


def test_a_heading_inside_any_element_is_nested_and_names_its_container() -> None:
    # Arrange
    html = (
        "<h1>Title</h1><blockquote><h1>Quoted</h1></blockquote>"
        '<div class="admonition note"><p class="admonition-title">Note</p><h2>In</h2></div>'
        "<ol><li><h3>Listed</h3></li></ol>"
        '<div class="footnote"><ol><li><h2>Noted</h2></li></ol></div>'
    )

    # Act
    headings = read_headings(parse_html(html))

    # Assert
    assert headings == [
        Heading(level=1, text="Title", container=""),
        Heading(level=1, text="Quoted", container="blockquote"),
        Heading(level=2, text="In", container="div.admonition.note"),
        Heading(level=3, text="Listed", container="ol"),
        Heading(level=2, text="Noted", container="div.footnote"),
    ]


@pytest.mark.parametrize(
    ("html", "expected"),
    [
        ('<h1>T</h1><p>Opening.</p><div class="toc"><ul></ul></div><h2>A</h2>', True),
        ('<h1>T</h1><p>Opening.</p><p>More.</p><div class="toc"><ul></ul></div>', False),
        ('<hr><h2>key: x</h2><h1>T</h1><p>Opening.</p><div class="toc"></div>', False),
        ('<h1>T</h1><p>Opening.</p><div class="admonition"></div>', False),
        ('<p>Opening.</p><h1>T</h1><div class="toc"></div>', False),
        ('<h1>T</h1><h2>A</h2><div class="toc"></div>', False),
        ("", False),
    ],
)
def test_the_contents_list_must_follow_the_title_and_opening(html: str, expected: bool) -> None:
    # Act
    found = has_contents_list_after_opening(parse_html(html))

    # Assert
    assert found is expected


@pytest.mark.parametrize(
    "text",
    [
        "2. Options",
        "2) Options",
        "3: Options",
        "2.1. Thresholds",
        "10. Options",
        "II. Options",
        "IV) Options",
        "X: the format",
        "(2) Options",
        "Step 1: the tools",
        "step 2 the tools",
        "\N{FULLWIDTH DIGIT TWO}. Options",
    ],
)
def test_a_typed_number_is_found(text: str) -> None:
    # Act
    typed = is_typed_number(text)

    # Assert
    assert typed


@pytest.mark.parametrize(
    "text",
    [
        "429 responses from the provider",
        "2 Options",
        "2.1 Thresholds",
        "3.5 Sonnet as a monitor",
        "0.1 threshold",
        "1:1 mapping of calls to records",
        "2.Options",
        "Stage 2 of a cascade",
        "Section 2.1 of the paper",
        "Phase 2 trials",
        "Part 2 of the protocol",
        "Step-by-step setup",
        "Step one: the tools",
        "CLI: the command line",
        "XML: the format",
        "II Options",
        "A. Options",
        "Options",
    ],
)
def test_a_heading_that_starts_with_a_number_as_a_word_is_not_typed(text: str) -> None:
    # Act
    typed = is_typed_number(text)

    # Assert
    assert not typed


def test_the_last_top_level_section_must_be_the_references() -> None:
    # Arrange
    in_order = [
        Heading(level=2, text="How it works", container=""),
        Heading(level=2, text="References", container=""),
        Heading(level=3, text="Books", container=""),
        Heading(level=2, text="Aside", container="div.admonition"),
    ]
    out_of_order = [*in_order, Heading(level=2, text="Later", container="")]

    # Act
    results = (ends_with_references(in_order), ends_with_references(out_of_order))

    # Assert
    assert results == (True, False)


# Rendering Markdown as the site does: needs the docs group ---------------------


def test_the_renderer_uses_the_extensions_that_mkdocs_yml_names() -> None:
    # Act
    settings = load_markdown_settings()

    # Assert
    assert {"admonition", "footnotes", "toc", "pymdownx.superfences"} <= set(settings.extensions)
    assert "pymdownx.details" not in settings.extensions
    assert "pymdownx.tabbed" not in settings.extensions
    assert settings.extension_configs["toc"]["toc_depth"] == "2-3"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Text.\n\n- item\n\n    ## In a list\n", [(2, "In a list", True)]),
        ("Text.\n\n1. item\n\n    ## In a list\n", [(2, "In a list", True)]),
        ("- item\n\n\t## In a list\n", [(2, "In a list", True)]),
        ("Text.\n\n> ## In a quote\n", [(2, "In a quote", True)]),
        ("Text.\n\n> # Second title\n", [(1, "Second title", True)]),
        ("Text[^1].\n\n[^1]: A note.\n\n    ## In a footnote\n", [(2, "In a footnote", True)]),
        ("!!!note\n    ## Inside\n", [(2, "Inside", True)]),
        ("!!! note\n\n    ## Inside\n\n## After\n", [(2, "Inside", True), (2, "After", False)]),
        ("<div>\n<h2>In a div</h2>\n</div>\n", [(2, "In a div", True)]),
        ("Text.\n\n<!--\n# Commented out\n-->\n", []),
        ("<div>\n\n## Raw text in a div\n\n</div>\n", []),
        ("Text.\n\n    ```\n    code\n\n## Real heading\n", [(2, "Real heading", False)]),
        ("Text.\n\n```\n## After an unclosed fence\n", [(2, "After an unclosed fence", False)]),
        ("```python `x`\n## After a broken fence\n```\n", [(2, "After a broken fence", False)]),
        ("```python\n# a comment\n## 2. not a heading\n```\n", []),
        ("Para\ntext\n---\nNext\n====\n", [(1, "Next", False)]),
        ("Title\n=====\n\nSection\n-------\n", [(1, "Title", False), (2, "Section", False)]),
        ("??? note\n\n    ## Not a block on this site\n", []),
        ('=== "Tab"\n\n    ## Not a block on this site\n', []),
    ],
)
def test_headings_are_read_as_the_site_renders_them(
    text: str,
    expected: list[tuple[int, str, bool]],
) -> None:
    # Act
    headings = render_headings(text)

    # Assert
    assert headings == expected


@pytest.mark.parametrize(
    ("text", "body", "first_line"),
    [
        ("---\ntitle: A thing\n---\n\n# T\n", "# T\n", 5),
        ("title: A thing\n\n# T\n", "# T\n", 3),
        ("---\n- a\n- b\n---\n\n# T\n", "---\n- a\n- b\n---\n\n# T\n", 1),
        ("---\nkey: [unclosed\n---\n\n# T\n", "---\nkey: [unclosed\n---\n\n# T\n", 1),
        ("---\n\n# T\n\nText.\n\n---\n", "---\n\n# T\n\nText.\n\n---\n", 1),
    ],
)
def test_front_matter_is_stripped_only_as_mkdocs_strips_it(
    text: str,
    body: str,
    first_line: int,
) -> None:
    # Act
    source = read_page_source(text)

    # Assert
    assert source == PageSource(body=body, first_line=first_line)


def test_a_page_in_the_standard_shape_has_no_violations() -> None:
    # Arrange: front matter, a setext title, a two-line opening, the marker, a
    # cited source, a heading that starts with a number as a word, and a note.
    text = (
        "---\nheading_numbers: false\n---\n\nUse a thing\n===========\n\nThis guide sets up a "
        "thing,\nin two lines.\n\n[TOC]\n\n## How it works\n\nIt follows a paper [@key].\n\n"
        "```python\n# Arrange: a comment, not a heading\n```\n\n"
        "!!! note\n\n    A note, with no heading in it.\n\n"
        "## 429 responses from the provider\n\n## References\n"
    )

    # Act
    violations = find_violations(text, page="how-to/use-a-thing.md")

    # Assert
    assert violations == []


def test_the_api_reference_needs_no_contents_list() -> None:
    # Arrange
    text = "# API\n\nThis reference is generated from the docstrings.\n\n::: package\n"

    # Act
    violations = find_violations(text, page="reference/api.md")

    # Assert
    assert violations == []


@pytest.mark.parametrize(
    ("text", "opening"),
    [
        ("# Use a thing\n\nThis guide.\n\nA second paragraph.\n\n[TOC]\n", "<h1>, <p>, <p>"),
        ("# Use a thing\n\nThis guide.\n\n```\n[TOC]\n```\n", "<h1>, <p>, <div>"),
        ("---\n\n# Use a thing\n\nThis guide.\n\n[TOC]\n\n---\n", "<hr>, <h1>, <p>"),
        (
            "---\nkey: [unclosed\n---\n\n# Use a thing\n\nThis guide.\n\n[TOC]\n",
            "<hr>, <h2>, <h1>",
        ),
    ],
)
def test_a_page_that_does_not_open_with_its_contents_list_is_reported(
    text: str,
    opening: str,
) -> None:
    # Act
    violations = find_violations(text, page="how-to/use-a-thing.md")

    # Assert
    assert (
        "how-to/use-a-thing.md: no [TOC] right after the title and the opening paragraph; "
        f"the page opens with {opening}"
    ) in violations


def test_a_wrapped_line_that_starts_with_a_hash_is_a_second_title() -> None:
    # Arrange: the stray title in explanation/design.md, where "(issue" ends a line.
    text = build_page(
        "## Forks\n\nForks are not supported yet (issue\n#35), and the helper raises.\n",
    )

    # Act
    violations = find_violations(text, page="explanation/design.md")

    # Assert
    assert violations == [
        "explanation/design.md: 2 level-1 headings, but a page has exactly one, its title: "
        "line 1 'Use a thing'; line 10 '35), and the helper raises.'",
    ]


def test_a_title_in_a_quote_is_a_second_title_and_nested() -> None:
    # Arrange
    text = build_page("## Section\n\n> # Second title\n")

    # Act
    violations = find_violations(text, page="how-to/use-a-thing.md")

    # Assert
    assert violations == [
        "how-to/use-a-thing.md: 2 level-1 headings, but a page has exactly one, its title: "
        "line 1 'Use a thing'; line 9 'Second title'",
        "how-to/use-a-thing.md: the heading on line 9 is inside <blockquote>, where the "
        "site's section numbers skip it: 'Second title'",
    ]


def test_a_page_without_a_title_is_reported() -> None:
    # Arrange
    text = "## API\n\nThis reference is generated from the docstrings.\n"

    # Act
    violations = find_violations(text, page="reference/api.md")

    # Assert
    assert violations == [
        "reference/api.md: no level-1 heading, but a page has exactly one, its title",
    ]


def test_a_heading_inside_a_list_is_reported_with_its_line() -> None:
    # Arrange
    text = build_page("## Section\n\n1. Install it.\n\n    ### Options\n")

    # Act
    violations = find_violations(text, page="how-to/use-a-thing.md")

    # Assert
    assert violations == [
        "how-to/use-a-thing.md: the heading on line 11 is inside <ol>, where the site's "
        "section numbers skip it: 'Options'",
    ]


def test_a_heading_in_raw_html_is_reported_without_a_line() -> None:
    # Arrange
    text = build_page("## Section\n\n<div>\n<h3>Raw</h3>\n</div>\n")

    # Act
    violations = find_violations(text, page="how-to/use-a-thing.md")

    # Assert
    assert violations == [
        "how-to/use-a-thing.md: the heading on a line the check could not find is inside "
        "<div>, where the site's section numbers skip it: 'Raw'",
    ]


def test_a_code_sample_of_a_heading_does_not_take_the_heading_line() -> None:
    # Arrange: the sample shows the same text as the heading, earlier on the page.
    text = build_page("## Example\n\n```markdown\n## 2. Options\n```\n\n## 2. Options\n")

    # Act
    violations = find_violations(text, page="how-to/use-a-thing.md")

    # Assert
    assert violations == [
        "how-to/use-a-thing.md: the heading on line 13 starts with a typed number: '2. Options'",
    ]


@pytest.mark.parametrize(
    "heading",
    ["## 2. Options", "## **2.** Options", "## `2.` Options", "## [2. Options](#options)"],
)
def test_a_typed_number_is_reported_whatever_wraps_it(heading: str) -> None:
    # Arrange
    text = build_page(f"{heading}\n")

    # Act
    violations = find_violations(text, page="how-to/use-a-thing.md")

    # Assert
    assert violations == [
        "how-to/use-a-thing.md: the heading on line 7 starts with a typed number: '2. Options'",
    ]


def test_a_setext_typed_number_is_reported_on_its_text_line() -> None:
    # Arrange
    text = build_page("2. Options\n----------\n")

    # Act
    violations = find_violations(text, page="how-to/use-a-thing.md")

    # Assert
    assert violations == [
        "how-to/use-a-thing.md: the heading on line 7 starts with a typed number: '2. Options'",
    ]


def test_a_heading_that_names_a_stage_is_not_reported() -> None:
    # Arrange
    text = build_page("## Stage 2 of a cascade\n\n## 3.5 Sonnet as a monitor\n")

    # Act
    violations = find_violations(text, page="how-to/use-a-thing.md")

    # Assert
    assert violations == []


def test_a_citation_inside_fenced_code_is_reported_and_needs_references() -> None:
    # Arrange: mkdocs-bibtex rewrites it, and appends a bibliography to the page.
    text = build_page('## Example\n\n```python\nnote = "[@key]"\n```\n')

    # Act
    violations = find_violations(text, page="how-to/use-a-thing.md")

    # Assert
    assert violations == [
        "how-to/use-a-thing.md: line 10 cites [@key] inside code, where mkdocs-bibtex "
        "still rewrites it into a footnote",
        "how-to/use-a-thing.md: cites a source, so its last section must be ## References",
    ]


def test_a_citation_inside_inline_code_is_reported() -> None:
    # Arrange
    text = build_page("## Example\n\nWrite `[@key]` to cite.\n\n## References\n")

    # Act
    violations = find_violations(text, page="how-to/use-a-thing.md")

    # Assert
    assert violations == [
        "how-to/use-a-thing.md: line 9 cites [@key] inside code, where mkdocs-bibtex "
        "still rewrites it into a footnote",
    ]


@pytest.mark.parametrize(
    "citation",
    ["[@key]", "[see @key, p. 3]", "[@first; @second]", "<!-- [@key] -->"],
)
def test_every_citation_the_plugin_rewrites_needs_references(citation: str) -> None:
    # Arrange
    text = build_page(f"## How it works\n\nIt follows a paper {citation}.\n")

    # Act
    violations = find_violations(text, page="how-to/use-a-thing.md")

    # Assert
    assert violations == [
        "how-to/use-a-thing.md: cites a source, so its last section must be ## References",
    ]


@pytest.mark.parametrize(
    "text",
    ["Write to [team@example.com] for help.", "See [the guide](guide.md).", "A decorator: @key."],
)
def test_text_the_plugin_leaves_alone_needs_no_references(text: str) -> None:
    # Arrange
    page = build_page(f"## How it works\n\n{text}\n")

    # Act
    violations = find_violations(page, page="how-to/use-a-thing.md")

    # Assert
    assert violations == []


def test_a_page_that_cites_without_a_final_references_section_is_reported() -> None:
    # Arrange: the References section exists but is not the last one.
    text = build_page("## References\n\n## How it works\n\nIt follows a paper [@key].\n")

    # Act
    violations = find_violations(text, page="how-to/use-a-thing.md")

    # Assert
    assert violations == [
        "how-to/use-a-thing.md: cites a source, so its last section must be ## References",
    ]
