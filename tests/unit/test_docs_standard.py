"""Every docs page has the shape the site's layout depends on.

The docs standard asks these things of each page, checked here in its Markdown:

- exactly one level-1 heading, the page's title;
- a contents list: ``[TOC]`` on its own line, right after the title and the
  opening paragraph, which the site renders as a numbered contents box;
- no heading inside an admonition, a tab or a details block (``!!!``,
  ``???``, ``???+`` or ``===``): the site's stylesheet numbers only the
  headings at the top level of a page, and the contents lists list every
  heading, so one inside a block would put the two numberings out of step;
- no number typed into a heading, as in ``## 2. Options``, ``## **2.**
  Options``, ``## II. Options`` or ``## Step 1: the tools``, since the
  stylesheet numbers the h2 and h3 sections itself. A leading number counts as
  typed when a ``.``, ``)`` or ``:`` follows it, or when a word such as "Step"
  comes before it, so ``## 429 responses from the provider`` is fine;
- a page that cites a source with ``[@key]`` outside a code block ends with a
  ``## References`` section, under which the site prints that page's
  bibliography.

Two kinds of page need no contents list. The API reference is generated from
the docstrings, and its right-hand contents list already indexes every class
and function; a ``[TOC]`` would repeat that whole index above the first entry.
A section index page, such as ``how-to/index.md``, is itself the list of its
section's pages, and most have no sections of their own to list.

Headings are read as Python-Markdown, the site's renderer, reads them, outside
fenced code blocks and the front matter that MkDocs strips:

- a line that starts with ``#``, with or without a space after the hashes, so
  a wrapped line that happens to start with ``#35)`` is an h1;
- a setext heading: the first line of a paragraph, underlined with ``=`` for
  an h1 or ``-`` for an h2;
- either form, indented, inside an admonition, a tab or a details block.

Headings inside a list, a quote or raw HTML are not read. When Python-Markdown
is installed, a test checks these rules against its own output. The plans under
``docs/plans/`` are working notes, not pages.

The pages do not comply yet, so the page check is expected to fail until the
docs pass brings every page in line and removes its ``xfail`` mark. Run
``pytest tests/unit/test_docs_standard.py --runxfail`` to list what is left.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

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
# Python-Markdown expands a tab to four spaces, and indents a block's content by four.
INDENT_WIDTH = 4
# MkDocs' own patterns for front matter [@mkdocs2024]: a YAML block, or else
# MultiMarkdown-style "key: value" lines, so a page's front matter is exactly
# what the site strips before it renders the page.
FRONT_MATTER_PATTERN = re.compile(
    r"\A-{3}[ \t]*\n(.*?\n)(?:\.{3}|-{3})[ \t]*\n",
    flags=re.DOTALL,
)
META_LINE_PATTERN = re.compile(r"[ ]{0,3}[A-Za-z0-9_-]+:")
META_CONTINUATION_PATTERN = re.compile(r"[ ]{4}|\t")
FENCE_PATTERN = re.compile(r"[ \t]*(?P<fence>`{3,}|~{3,})")
ATX_HEADING_PATTERN = re.compile(r"(?P<hashes>#{1,6})(?P<title>.*?)#*[ \t]*$")
SETEXT_UNDERLINE_PATTERN = re.compile(r"(?P<rule>=+|-+) *")
BLOCK_OPENER_PATTERN = re.compile(r"(?:!!!|\?\?\?\+?|===[+!]?)[ \t]+\S")
# Emphasis or code around a typed number, as in **2.** or `2.`.
MARKUP = r"[*_`]*"
TYPED_NUMBER_PATTERN = re.compile(
    rf"{MARKUP}(?:\d+|[IVX]+){MARKUP}[.):]"
    rf"|{MARKUP}(?i:step|part|section|chapter|stage|phase)\s+\d",
)
CITATION_PATTERN = re.compile(r"\[@[^\]\n]+\]")
# The site's Markdown extensions, with the two that make ??? and === blocks.
RENDERER_EXTENSIONS = (
    "admonition",
    "attr_list",
    "footnotes",
    "toc",
    "pymdownx.superfences",
    "pymdownx.details",
    "pymdownx.tabbed",
)
HEADING_TAGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})


@dataclass(frozen=True, slots=True, kw_only=True)
class SourceLine:
    """One line of a page, its tabs expanded as Python-Markdown expands them."""

    number: int
    text: str
    in_code: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class Heading:
    """One heading of a page: its level, its text, its line, and whether a block holds it.

    `nested` is true for a heading inside an admonition, a tab or a details block.
    """

    level: int
    title: str
    line_number: int
    nested: bool


def read_docs_pages() -> list[Path]:
    """Return every Markdown page of the site, the plans left out, in path order."""
    return sorted(
        path for path in DOCS_DIRECTORY.rglob("*.md") if NOTES_DIRECTORY not in path.parents
    )


def measure_front_matter(text: str) -> int:
    """Return how many characters at the start of a page MkDocs reads as front matter.

    That is a YAML block between ``---`` lines, or else a run of ``key: value``
    lines, with indented lines continuing a value, up to the first blank line
    or the first line of another kind [@mkdocs2024].
    """
    yaml_block = FRONT_MATTER_PATTERN.match(text)
    if yaml_block is not None:
        return yaml_block.end()
    end = 0
    has_key = False
    for line in text.splitlines(keepends=True):
        if not line.strip():
            break
        if META_LINE_PATTERN.match(line):
            has_key = True
        elif not (has_key and META_CONTINUATION_PATTERN.match(line)):
            break
        end += len(line)
    return end


def blank_front_matter(text: str) -> str:
    """Return a page's Markdown with its front matter blanked, keeping every line number."""
    end = measure_front_matter(text)
    return "\n" * text[:end].count("\n") + text[end:]


def is_closing_fence(line: str, *, fence: str) -> bool:
    """Tell whether a line closes a code block opened with `fence`."""
    closing = re.compile(rf"[ \t]*{re.escape(fence[0])}{{{len(fence)},}}[ \t]*")
    return closing.fullmatch(line) is not None


def read_source_lines(text: str) -> list[SourceLine]:
    """Split a page into lines, front matter blanked, marking the lines of fenced code."""
    lines: list[SourceLine] = []
    open_fence: str | None = None
    for number, raw_line in enumerate(blank_front_matter(text).splitlines(), start=1):
        line = raw_line.expandtabs(INDENT_WIDTH)
        if open_fence is not None:
            if is_closing_fence(line, fence=open_fence):
                open_fence = None
            lines.append(SourceLine(number=number, text=line, in_code=True))
            continue
        fence = FENCE_PATTERN.match(line)
        if fence is not None:
            open_fence = fence.group("fence")
        lines.append(SourceLine(number=number, text=line, in_code=fence is not None))
    return lines


def measure_indent(line: str) -> int:
    """Return how many spaces a line, its tabs already expanded, starts with."""
    return len(line) - len(line.lstrip(" "))


def read_setext_level(line: SourceLine, *, block_indent: int) -> int | None:
    """Return 1 or 2 when a line underlines the line above it as a heading, and None otherwise."""
    if line.in_code or measure_indent(line.text) != block_indent:
        return None
    underline = SETEXT_UNDERLINE_PATTERN.fullmatch(line.text[block_indent:])
    if underline is None:
        return None
    return 1 if underline.group("rule").startswith("=") else 2


def read_headings(text: str) -> list[Heading]:
    """Return the headings Python-Markdown finds in a page, outside fenced code and front matter.

    A block (an admonition, a tab or a details block) holds the lines indented
    past its opening line; a heading is one of its lines, set at the block's own
    indent. A setext heading is the first line of a paragraph: one after a blank
    line, a heading, or the start of a block.
    """
    lines = read_source_lines(text)
    headings: list[Heading] = []
    block_indents: list[int] = []
    starts_paragraph = True
    index = 0
    while index < len(lines):
        line = lines[index]
        index += 1
        if line.in_code:
            starts_paragraph = False
            continue
        if not line.text.strip():
            starts_paragraph = True
            continue
        indent = measure_indent(line.text)
        while block_indents and indent < block_indents[-1]:
            block_indents.pop()
            starts_paragraph = True
        block_indent = block_indents[-1] if block_indents else 0
        content = line.text[block_indent:]
        if indent == block_indent and BLOCK_OPENER_PATTERN.match(content):
            block_indents.append(block_indent + INDENT_WIDTH)
            starts_paragraph = True
            continue
        nested = bool(block_indents)
        heading = ATX_HEADING_PATTERN.match(content) if indent == block_indent else None
        if heading is not None:
            headings.append(
                Heading(
                    level=len(heading.group("hashes")),
                    title=heading.group("title").strip(),
                    line_number=line.number,
                    nested=nested,
                ),
            )
            starts_paragraph = True
            continue
        can_be_setext = starts_paragraph and indent < block_indent + INDENT_WIDTH
        level = (
            read_setext_level(lines[index], block_indent=block_indent)
            if can_be_setext and index < len(lines)
            else None
        )
        if level is not None:
            headings.append(
                Heading(level=level, title=content.strip(), line_number=line.number, nested=nested),
            )
            index += 1
            starts_paragraph = True
            continue
        starts_paragraph = False
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
    blocks = read_blocks(blank_front_matter(text))
    if not blocks or not blocks[0][0].startswith("# "):
        return False
    # The opening paragraph is the rest of the title's block, or the next block.
    marker_index = 1 if len(blocks[0]) > 1 else 2
    return len(blocks) > marker_index and blocks[marker_index] == [CONTENTS_MARKER]


def has_citation(text: str) -> bool:
    """Tell whether a page cites a source with ``[@key]`` outside fenced code and front matter."""
    return any(
        CITATION_PATTERN.search(line.text) for line in read_source_lines(text) if not line.in_code
    )


def find_titles(headings: list[Heading]) -> list[Heading]:
    """Return the level-1 headings: a page has one, its title."""
    return [heading for heading in headings if heading.level == 1]


def find_nested_headings(headings: list[Heading]) -> list[Heading]:
    """Return the headings inside an admonition, a tab or a details block."""
    return [heading for heading in headings if heading.nested]


def find_typed_numbers(headings: list[Heading]) -> list[Heading]:
    """Return the headings whose text starts with a number someone typed."""
    return [heading for heading in headings if TYPED_NUMBER_PATTERN.match(heading.title)]


def ends_with_references(headings: list[Heading]) -> bool:
    """Tell whether the page's last top-level h2 section is the References section."""
    sections = [heading.title for heading in headings if heading.level == 2 and not heading.nested]
    return bool(sections) and sections[-1] == REFERENCES_TITLE


def render_title_count(titles: list[Heading], *, page: str) -> str:
    """Say how a page breaks the one-title rule, naming each level-1 heading it has."""
    if not titles:
        return f"{page}: no level-1 heading, but a page has exactly one, its title"
    found = "; ".join(f"line {title.line_number} {title.title!r}" for title in titles)
    return f"{page}: {len(titles)} level-1 headings, but a page has exactly one, its title: {found}"


def find_violations(text: str, *, page: str) -> list[str]:
    """Return what a page, named by its path under ``docs/``, lacks or breaks."""
    headings = read_headings(text)
    violations: list[str] = []
    titles = find_titles(headings)
    if len(titles) != 1:
        violations.append(render_title_count(titles, page=page))
    if page not in PAGES_WITHOUT_CONTENTS_LIST and not has_contents_list_after_opening(text):
        violations.append(f"{page}: no {CONTENTS_MARKER} line right after the opening paragraph")
    violations.extend(
        f"{page}: the heading on line {heading.line_number} is inside an admonition, tab or "
        f"details block, where the site's section numbers skip it: {heading.title!r}"
        for heading in find_nested_headings(headings)
    )
    violations.extend(
        f"{page}: the heading on line {heading.line_number} starts with a typed number: "
        f"{heading.title!r}"
        for heading in find_typed_numbers(headings)
    )
    if has_citation(text) and not ends_with_references(headings):
        violations.append(f"{page}: cites a source, so its last section must be ## References")
    return violations


def read_rendered_headings(text: str) -> list[tuple[int, str, bool]]:
    """Return each heading Python-Markdown renders: its level, its text, whether it is nested."""
    markdown = pytest.importorskip("markdown")
    html = markdown.markdown(text, extensions=list(RENDERER_EXTENSIONS))
    root = ElementTree.fromstring(f"<root>{html}</root>")
    top_level = set(root)
    return [
        (int(element.tag[1]), "".join(element.itertext()).strip(), element not in top_level)
        for element in root.iter()
        if element.tag in HEADING_TAGS
    ]


def build_page(body: str) -> str:
    """Return a page with a title, an opening line and the contents marker, then `body`."""
    return f"# Use a thing\n\nThis guide sets up a thing.\n\n[TOC]\n\n{body}"


def test_the_docs_hold_pages_to_check() -> None:
    # Act
    pages = {str(path.relative_to(DOCS_DIRECTORY)) for path in read_docs_pages()}

    # Assert
    assert {"index.md", "reference/api.md", "how-to/use-auto-mode.md"} <= pages
    assert not any(page.startswith("plans/") for page in pages)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="enabled by the docs pass")
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


def test_the_page_check_expects_only_a_failed_assertion() -> None:
    # Arrange: a crash in the checker, such as a page it cannot decode, must fail the run.
    marks = vars(test_every_docs_page_meets_the_docs_standard)["pytestmark"]

    # Act
    xfail = next(mark for mark in marks if mark.name == "xfail")

    # Assert
    assert xfail.kwargs["strict"] is True
    assert xfail.kwargs["raises"] is AssertionError


def test_a_page_in_the_standard_shape_has_no_violations() -> None:
    # Arrange: front matter, a title, a two-line opening, the marker, a cited source,
    # a heading that starts with a number as a word, and a note with no heading in it.
    text = (
        "---\nheading_numbers: false\n---\n\n# Use a thing\n\nThis guide sets up a thing,\n"
        "in two lines.\n\n[TOC]\n\n## How it works\n\nIt follows a paper [@key].\n\n"
        "```python\n# Arrange: a comment, not a heading\n```\n\n"
        "!!! note\n\n    A note, with no heading in it.\n\n"
        "## 429 responses from the provider\n\n## References\n"
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
    ("line", "level", "title"),
    [
        ("## Options", 2, "Options"),
        ("##Options", 2, "Options"),
        ("### Closing hashes ###", 3, "Closing hashes"),
        ("#35), and the helper raises.", 1, "35), and the helper raises."),
        ("#fork mode, which is new.", 1, "fork mode, which is new."),
        ("#\tA tab after the hash", 1, "A tab after the hash"),
    ],
)
def test_a_line_that_starts_with_a_hash_is_a_heading(line: str, level: int, title: str) -> None:
    # Act
    headings = read_headings(f"Some text.\n\n{line}\n")

    # Assert
    assert headings == [Heading(level=level, title=title, line_number=3, nested=False)]


def test_a_wrapped_line_that_starts_with_a_hash_is_an_h1() -> None:
    # Arrange: the stray title in explanation/design.md, where "(issue" ends a line.
    text = "Forks are not supported yet (issue\n#35), and the helper raises.\n"

    # Act
    headings = read_headings(text)

    # Assert
    assert headings == [
        Heading(level=1, title="35), and the helper raises.", line_number=2, nested=False),
    ]


@pytest.mark.parametrize(
    "text",
    [
        " # One space before the hash\n",
        "Some text.\n\n    # indented code\n",
        "Some text,\n    # a continued line\n",
        "A sentence that ends in a hash #\n",
    ],
)
def test_a_hash_that_does_not_start_a_line_is_no_heading(text: str) -> None:
    # Act
    headings = read_headings(text)

    # Assert
    assert headings == []


@pytest.mark.parametrize(
    "text",
    [
        "```python\n# a comment\n## 2. not a heading\n```\n",
        "~~~\n## 2. not a heading\n~~~\n",
        "````\n```\n## 2. inside the outer fence\n```\n````\n",
        "!!! note\n\n    ```python\n    # a comment in a note\n    ```\n",
    ],
)
def test_a_heading_inside_fenced_code_is_ignored(text: str) -> None:
    # Act
    headings = read_headings(text)

    # Assert
    assert headings == []


def test_a_short_fence_does_not_close_a_longer_one() -> None:
    # Arrange: the inner ``` lines are the code's own text, and only ```` closes the block.
    text = "````\n```\n# inside\n```\n````\n\n## After the code\n"

    # Act
    headings = read_headings(text)

    # Assert
    assert headings == [Heading(level=2, title="After the code", line_number=7, nested=False)]


@pytest.mark.parametrize(
    ("text", "level", "title"),
    [
        ("Title\n=====\n", 1, "Title"),
        ("Title\n=\n", 1, "Title"),
        ("Title\n=== \n", 1, "Title"),
        ("Section\n-------\n", 2, "Section"),
        ("2. Options\n----------\n", 2, "2. Options"),
        ("- an item\n---\n", 2, "- an item"),
    ],
)
def test_an_underlined_line_is_a_setext_heading(text: str, level: int, title: str) -> None:
    # Act
    headings = read_headings(text)

    # Assert
    assert headings == [Heading(level=level, title=title, line_number=1, nested=False)]


@pytest.mark.parametrize(
    "text",
    [
        "Line one\nline two\n=====\n",
        "Text\n  ===\n",
        "Text\n===x\n",
        "Text\n= =\n",
        "Text\n\n---\n",
        "```\ncode\n```\nText\n===\n",
    ],
)
def test_an_underline_that_makes_no_setext_heading_is_ignored(text: str) -> None:
    # Act
    headings = read_headings(text)

    # Assert
    assert headings == []


def test_a_setext_heading_can_follow_a_hash_heading_directly() -> None:
    # Arrange: a hash heading ends its paragraph, so the next line starts one.
    text = "# Title\nSubtitle\n========\n"

    # Act
    headings = read_headings(text)

    # Assert
    assert headings == [
        Heading(level=1, title="Title", line_number=1, nested=False),
        Heading(level=1, title="Subtitle", line_number=2, nested=False),
    ]


@pytest.mark.parametrize(
    "front_matter",
    [
        "---\ntitle: A thing\n---\n",
        "---\ntitle: A thing\n...\n",
        "--- \nheading_numbers: false\n---\n",
        "title: A thing\nsummary: one\n    and two\n",
    ],
)
def test_front_matter_is_no_heading_and_keeps_line_numbers(front_matter: str) -> None:
    # Arrange
    text = f"{front_matter}\n# Use a thing\n"

    # Act
    headings = read_headings(text)

    # Assert
    line_number = front_matter.count("\n") + 2
    assert headings == [
        Heading(level=1, title="Use a thing", line_number=line_number, nested=False),
    ]


@pytest.mark.parametrize(
    "text",
    [
        "---\ntitle: A thing\n---\n\n# T\n",
        "---\ntitle: A thing\n...\n\n# T\n",
        "---\n---\n\n# T\n",
        "title: A thing\n\n# T\n",
        "Summary: one\n    and two\n\n# T\n",
        "# T\n\nkey: value\n",
        "https://example.com\n\n# T\n",
    ],
)
def test_front_matter_is_read_as_mkdocs_reads_it(text: str) -> None:
    # Arrange
    meta = pytest.importorskip("mkdocs.utils.meta")
    expected, _ = meta.get_data(text)

    # Act
    body = blank_front_matter(text)

    # Assert
    assert body.lstrip("\n") == expected
    assert body.count("\n") == text.count("\n")


def test_two_rules_with_nothing_between_are_no_front_matter() -> None:
    # Arrange: MkDocs keeps "---\n---", and Python-Markdown renders it as an h2.
    text = "---\n---\n\n# Use a thing\n"

    # Act
    headings = read_headings(text)

    # Assert
    assert headings == [
        Heading(level=2, title="---", line_number=1, nested=False),
        Heading(level=1, title="Use a thing", line_number=4, nested=False),
    ]


@pytest.mark.parametrize(
    "text",
    [
        "!!! note\n\n    ## Inside\n",
        '!!! note "A title"\n    ## Inside\n',
        "??? note\n\n    ## Inside\n",
        '???+ note "A title"\n    ## Inside\n',
        '=== "A tab"\n\n    ## Inside\n',
        '===+ "A tab"\n\n    ## Inside\n',
        "!!! note\n    Inside\n    ------\n",
        "!!! note\n    !!! tip\n        ## Inside\n",
        "!!! note\n\n\t## Inside\n",
    ],
)
def test_a_heading_inside_an_admonition_tab_or_details_block_is_nested(text: str) -> None:
    # Act
    headings = read_headings(text)

    # Assert
    assert [(heading.level, heading.title, heading.nested) for heading in headings] == [
        (2, "Inside", True),
    ]


def test_a_heading_after_a_block_ends_is_at_the_top_level() -> None:
    # Arrange
    text = "!!! note\n\n    Some text.\n\n## After\n\n!!! tip\n    Text\nStray\n=====\n"

    # Act
    headings = read_headings(text)

    # Assert
    assert headings == [
        Heading(level=2, title="After", line_number=5, nested=False),
        Heading(level=1, title="Stray", line_number=9, nested=False),
    ]


def test_indented_code_inside_a_block_holds_no_heading() -> None:
    # Arrange: eight spaces inside a note is a code block, not a heading.
    text = "!!! note\n\n        ## code in a note\n"

    # Act
    headings = read_headings(text)

    # Assert
    assert headings == []


@pytest.mark.parametrize(
    "text",
    [
        "## 2. Options\n",
        "### 2.1 Thresholds\n",
        "## 1) Options\n",
        "## 3: Options\n",
        "## **2.** Options\n",
        "## `2.` Options\n",
        "## *2*. Options\n",
        "## II. Options\n",
        "## IV) Options\n",
        "## Step 1: the tools\n",
        "## step 2 the tools\n",
        "## **Part 3**: the tools\n",
        "#35), and the helper raises.\n",
        "2. Options\n----------\n",
    ],
)
def test_a_heading_with_a_typed_number_is_found(text: str) -> None:
    # Act
    typed = find_typed_numbers(read_headings(text))

    # Assert
    assert len(typed) == 1


@pytest.mark.parametrize(
    "text",
    [
        "## 429 responses from the provider\n",
        "## 3 ways to calibrate a monitor\n",
        "## Limits of 3 and 20\n",
        "## XML: the format\n",
        "## Stepping back\n",
        "## Options\n",
    ],
)
def test_a_heading_that_starts_with_a_number_as_a_word_is_not_found(text: str) -> None:
    # Act
    typed = find_typed_numbers(read_headings(text))

    # Assert
    assert typed == []


def test_a_page_with_a_second_title_is_reported() -> None:
    # Arrange: a wrapped line that starts with "#35)" renders as a second h1.
    text = build_page(
        "## Forks\n\nForks are not supported yet (issue\n#35), and the helper raises.\n",
    )

    # Act
    violations = find_violations(text, page="explanation/design.md")

    # Assert
    assert violations == [
        "explanation/design.md: 2 level-1 headings, but a page has exactly one, its title: "
        "line 1 'Use a thing'; line 10 '35), and the helper raises.'",
        "explanation/design.md: the heading on line 10 starts with a typed number: "
        "'35), and the helper raises.'",
    ]


def test_a_setext_title_counts_as_a_second_title() -> None:
    # Arrange
    text = build_page("## Section\n\nA stray title\n=============\n")

    # Act
    violations = find_violations(text, page="how-to/use-a-thing.md")

    # Assert
    assert violations == [
        "how-to/use-a-thing.md: 2 level-1 headings, but a page has exactly one, its title: "
        "line 1 'Use a thing'; line 9 'A stray title'",
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


def test_a_heading_inside_an_admonition_is_reported() -> None:
    # Arrange
    text = build_page("## Section\n\n!!! note\n\n    ### Inside the note\n")

    # Act
    violations = find_violations(text, page="how-to/use-a-thing.md")

    # Assert
    assert violations == [
        "how-to/use-a-thing.md: the heading on line 11 is inside an admonition, tab or details "
        "block, where the site's section numbers skip it: 'Inside the note'",
    ]


def test_a_citation_inside_fenced_code_needs_no_references_section() -> None:
    # Arrange
    text = build_page('## Example\n\n```python\nnote = "[@key]"\n```\n')

    # Act
    violations = find_violations(text, page="how-to/use-a-thing.md")

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


@pytest.mark.parametrize(
    "text",
    [
        "Forks are not supported yet (issue\n#35), and the helper raises.\n",
        "See the\n#fork mode, which is new.\n",
        "# Title\n\n## Section\n\n### Closing hashes ###\n",
        "Title\n=====\n\nSection\n-------\n",
        "# Title\nSubtitle\n========\n",
        "- an item\n---\n",
        "Line one\nline two\n=====\n",
        "Text\n  ===\n",
        "```\ncode\n```\nText\n===\n",
        " # One space before the hash\n",
        "Some text.\n\n    # indented code\n",
        "```python\n# a comment\n```\n\n~~~\n## tilde\n~~~\n",
        "````\n```\n## inside\n```\n````\n\n## After the code\n",
        "!!! note\n\n    ## Inside\n\n## After\n",
        "??? note\n\n    ## Inside\n",
        '???+ note "A title"\n    ## Inside\n',
        '=== "A tab"\n\n    ## Inside\n',
        "!!! note\n    Inside\n    ------\n",
        "!!! note\n    !!! tip\n        ## Inside\n",
        "!!! note\n\n        ## code in a note\n",
        "!!! tip\n    Text\nStray\n=====\n",
    ],
)
def test_the_checker_reads_headings_as_python_markdown_does(text: str) -> None:
    # Arrange
    expected = read_rendered_headings(text)

    # Act
    headings = read_headings(text)

    # Assert
    assert [(heading.level, heading.title, heading.nested) for heading in headings] == expected
