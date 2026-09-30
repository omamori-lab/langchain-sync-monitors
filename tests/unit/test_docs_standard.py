"""Every docs page has the shape the site's layout depends on.

The docs standard asks three things of each page, checked here in its Markdown:

- a contents list: ``[TOC]`` on its own line, right after the title and the
  opening paragraph, which the site renders as a numbered contents box;
- no number typed into a heading: no heading starts with a digit, as in
  ``## 2. Options``, or with a word and a number, as in ``## Step 1: the
  tools``, since the site's stylesheet numbers the h2 and h3 sections itself;
- a page that cites a source with ``[@key]`` ends with a ``## References``
  section, under which the site prints that page's bibliography.

Two kinds of page need no contents list. The API reference is generated from
the docstrings, and its right-hand contents list already indexes every class
and function; a ``[TOC]`` would repeat that whole index above the first entry.
A section index page, such as ``how-to/index.md``, is itself the list of its
section's pages, and most have no sections of their own to list.

Headings are read as Python-Markdown, the site's renderer, reads them: any line
that starts with ``#`` outside a fenced code block, with or without a space
after the hashes. The plans under ``docs/plans/`` are working notes, not pages.

The pages do not comply yet, so the page check is expected to fail until the
docs pass brings every page in line and removes its ``xfail`` mark. Run
``pytest tests/unit/test_docs_standard.py --runxfail`` to list what is left.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIRECTORY = REPOSITORY_ROOT / "docs"
NOTES_DIRECTORY = DOCS_DIRECTORY / "plans"
CONTENTS_MARKER = "[TOC]"
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
FRONT_MATTER_PATTERN = re.compile(r"\A---\n.*?\n---\n", flags=re.DOTALL)
FENCE_PATTERN = re.compile(r"[ \t]*(?P<fence>`{3,}|~{3,})")
HEADING_PATTERN = re.compile(r"(?P<hashes>#{1,6})(?P<title>.*?)#*[ \t]*$")
TYPED_NUMBER_PATTERN = re.compile(
    r"\d|(?:step|part|section|chapter|stage|phase)\s+\d",
    flags=re.IGNORECASE,
)
CITATION_PATTERN = re.compile(r"\[@[^\]\n]+\]")


@dataclass(frozen=True, slots=True, kw_only=True)
class Heading:
    """One heading of a page: its level, its text and the line it is on."""

    level: int
    title: str
    line_number: int


def read_docs_pages() -> list[Path]:
    """Return every Markdown page of the site, the plans left out, in path order."""
    return sorted(
        path for path in DOCS_DIRECTORY.rglob("*.md") if NOTES_DIRECTORY not in path.parents
    )


def remove_front_matter(text: str) -> str:
    """Return a page's Markdown without its YAML front matter, if it has any."""
    return FRONT_MATTER_PATTERN.sub("", text, count=1)


def is_closing_fence(line: str, *, fence: str) -> bool:
    """Tell whether a line closes a code block opened with `fence`."""
    closing = re.compile(rf"[ \t]*{re.escape(fence[0])}{{{len(fence)},}}[ \t]*")
    return closing.fullmatch(line) is not None


def read_headings(text: str) -> list[Heading]:
    """Return the headings Python-Markdown finds in a page, skipping fenced code."""
    headings: list[Heading] = []
    open_fence: str | None = None
    for line_number, line in enumerate(text.splitlines(), start=1):
        if open_fence is not None:
            if is_closing_fence(line, fence=open_fence):
                open_fence = None
            continue
        fence = FENCE_PATTERN.match(line)
        if fence is not None:
            open_fence = fence.group("fence")
            continue
        heading = HEADING_PATTERN.match(line)
        if heading is not None:
            level = len(heading.group("hashes"))
            title = heading.group("title").strip()
            headings.append(Heading(level=level, title=title, line_number=line_number))
    return headings


def read_blocks(text: str) -> list[list[str]]:
    """Split a page into its blocks: runs of non-blank lines, each line stripped."""
    blocks: list[list[str]] = [[]]
    for line in text.splitlines():
        if line.strip():
            blocks[-1].append(line.strip())
        elif blocks[-1]:
            blocks.append([])
    return [block for block in blocks if block]


def has_contents_list_after_opening(text: str) -> bool:
    """Tell whether ``[TOC]`` is the block right after the title and the opening paragraph."""
    blocks = read_blocks(text)
    if not blocks or not blocks[0][0].startswith("# "):
        return False
    # The opening paragraph is the rest of the title's block, or the next block.
    marker_index = 1 if len(blocks[0]) > 1 else 2
    return len(blocks) > marker_index and blocks[marker_index] == [CONTENTS_MARKER]


def find_typed_numbers(headings: list[Heading]) -> list[Heading]:
    """Return the headings whose text starts with a number someone typed."""
    return [heading for heading in headings if TYPED_NUMBER_PATTERN.match(heading.title)]


def ends_with_references(headings: list[Heading]) -> bool:
    """Tell whether the page's last h2 section is the References section."""
    sections = [heading.title for heading in headings if heading.level == 2]
    return bool(sections) and sections[-1] == REFERENCES_TITLE


def find_violations(text: str, *, page: str) -> list[str]:
    """Return what a page, named by its path under ``docs/``, lacks or breaks."""
    body = remove_front_matter(text)
    headings = read_headings(body)
    violations: list[str] = []
    if page not in PAGES_WITHOUT_CONTENTS_LIST and not has_contents_list_after_opening(body):
        violations.append(f"{page}: no {CONTENTS_MARKER} line right after the opening paragraph")
    violations.extend(
        f"{page}: the heading on line {heading.line_number} starts with a typed number: "
        f"{heading.title!r}"
        for heading in find_typed_numbers(headings)
    )
    if CITATION_PATTERN.search(body) and not ends_with_references(headings):
        violations.append(f"{page}: cites a source, so its last section must be ## References")
    return violations


def test_the_docs_hold_pages_to_check() -> None:
    # Act
    pages = {str(path.relative_to(DOCS_DIRECTORY)) for path in read_docs_pages()}

    # Assert
    assert {"index.md", "reference/api.md", "how-to/use-auto-mode.md"} <= pages
    assert not any(page.startswith("plans/") for page in pages)


@pytest.mark.xfail(strict=True, reason="enabled by the docs pass")
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


def test_a_page_in_the_standard_shape_has_no_violations() -> None:
    # Arrange: front matter, a title, a two-line opening, the marker, and a cited source.
    text = (
        "---\nheading_numbers: false\n---\n\n# Use a thing\n\nThis guide sets up a thing,\n"
        "in two lines.\n\n[TOC]\n\n## How it works\n\nIt follows a paper [@key].\n\n"
        "```python\n# Arrange: a comment, not a heading\n```\n\n## References\n"
    )

    # Act
    violations = find_violations(text, page="how-to/use-a-thing.md")

    # Assert
    assert violations == []


def test_a_contents_marker_after_a_second_paragraph_is_reported() -> None:
    # Arrange
    text = "# Use a thing\n\nThis guide sets up a thing.\n\nA second paragraph.\n\n[TOC]\n"

    # Act
    violations = find_violations(text, page="how-to/use-a-thing.md")

    # Assert
    assert violations == ["how-to/use-a-thing.md: no [TOC] line right after the opening paragraph"]


def test_the_api_reference_needs_no_contents_marker() -> None:
    # Arrange
    text = "# API\n\nThis reference is generated from the docstrings.\n\n::: package\n"

    # Act
    violations = find_violations(text, page="reference/api.md")

    # Assert
    assert violations == []


@pytest.mark.parametrize(
    "line",
    ["## 2. Options", "### 2.1 Thresholds", "## Step 1: the tools", "#35), and the helper"],
)
def test_a_heading_with_a_typed_number_is_reported(line: str) -> None:
    # Arrange
    text = f"# Use a thing\n\nThis guide sets up a thing.\n\n[TOC]\n\n{line}\n"

    # Act
    violations = find_violations(text, page="how-to/use-a-thing.md")

    # Assert
    assert len(violations) == 1
    assert "starts with a typed number" in violations[0]


def test_a_heading_that_mentions_a_number_later_is_not_reported() -> None:
    # Arrange
    text = "# Use a thing\n\nThis guide sets up a thing.\n\n[TOC]\n\n## Limits of 3 and 20\n"

    # Act
    violations = find_violations(text, page="how-to/use-a-thing.md")

    # Assert
    assert violations == []


def test_a_page_that_cites_without_a_final_references_section_is_reported() -> None:
    # Arrange: the References section exists but is not the last one.
    text = (
        "# Use a thing\n\nThis guide sets up a thing.\n\n[TOC]\n\n"
        "## References\n\n## How it works\n\nIt follows a paper [@key].\n"
    )

    # Act
    violations = find_violations(text, page="how-to/use-a-thing.md")

    # Assert
    assert violations == [
        "how-to/use-a-thing.md: cites a source, so its last section must be ## References",
    ]
