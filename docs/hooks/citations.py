"""Turn the citations in rendered docstrings into footnotes, as on every other page.

mkdocs-bibtex [@mkdocsbibtex2025] rewrites the bracketed ``@key`` citations of
a page's Markdown before MkDocs converts it, but mkdocstrings
[@mkdocstrings2026] renders docstrings during that conversion, so the
citations in the API reference's docstrings would reach the page as literal
text. This MkDocs hook [@mkdocs2024] runs on each converted page. It gives
every citation left in the page's text the footnote markup Python-Markdown
gives the citations of the other pages, numbered in order of first use, and
appends the footnotes, each with the text mkdocs-bibtex formats from
``docs/references.bib``. Only the page's text is rewritten: a citation inside
a tag or one of its attributes, inside code, or inside an element whose content
HTML reads as plain text stays as it is. A key missing from the bibliography,
or a page that already has footnotes of its own, is logged as a warning, which
fails a strict build.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Collection
from html.parser import HTMLParser
from typing import override

import markdown
from mkdocs.config.defaults import MkDocsConfig
from mkdocs_bibtex.citation import Citation

# MkDocs counts the warnings of the loggers under "mkdocs" towards --strict.
logger = logging.getLogger("mkdocs.hooks.citations")

CITATION_PATTERN = re.compile(r"\[(@[\w:.-]+(?:;\s*@[\w:.-]+)*)\]")
KEY_PATTERN = re.compile(r"@([\w:.-]+)")
NEWLINE_PATTERN = re.compile("\n")
# Code, and the elements whose content HTML reads as plain text rather than markup.
VERBATIM_TAGS = frozenset({"code", "pre", "script", "style", "textarea", "title"})
FOOTNOTE_LIST_START = '<div class="footnote">'


class TextSpanParser(HTMLParser):
    """Record where each run of a page's text outside the verbatim tags starts and ends.

    With ``convert_charrefs=False`` the standard library's parser hands
    ``handle_data`` each run of text exactly as written, and ``getpos`` gives
    the line and column where that run starts, so the run is the slice of the
    page from there. Tags with their attributes, comments and character
    references reach other handlers, so no recorded span holds any of them.
    The parser ends a run only at a ``<`` or an ``&``, neither of which a
    citation holds, so no citation straddles two runs.
    """

    def __init__(self, html: str) -> None:
        super().__init__(convert_charrefs=False)
        self.line_starts = [0, *(match.end() for match in NEWLINE_PATTERN.finditer(html))]
        self.verbatim_depth = 0
        self.spans: list[tuple[int, int]] = []

    @override
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Count one more open verbatim element."""
        if tag in VERBATIM_TAGS:
            self.verbatim_depth += 1

    @override
    def handle_endtag(self, tag: str) -> None:
        """Count one fewer open verbatim element, ignoring an end tag nothing opened."""
        if tag in VERBATIM_TAGS and self.verbatim_depth:
            self.verbatim_depth -= 1

    @override
    def handle_data(self, data: str) -> None:
        """Record the span of a run of text, unless it lies inside a verbatim element."""
        if self.verbatim_depth:
            return
        line, column = self.getpos()
        start = self.line_starts[line - 1] + column
        self.spans.append((start, start + len(data)))


def find_text_spans(html: str) -> list[tuple[int, int]]:
    """Return the start and end of each run of the page's text outside the verbatim tags."""
    parser = TextSpanParser(html)
    parser.feed(html)
    parser.close()
    return parser.spans


def build_reference_id(key: str, *, use: int) -> str:
    """Return the id of the `use`-th citation of `key`, as Python-Markdown numbers repeats."""
    return f"fnref:{key}" if use == 1 else f"fnref{use}:{key}"


def render_reference(key: str, *, uses: dict[str, int]) -> str:
    """Return the superscript for one more citation of `key`, and count it in `uses`."""
    uses[key] = uses.get(key, 0) + 1
    number = list(uses).index(key) + 1
    reference_id = build_reference_id(key, use=uses[key])
    return f'<sup id="{reference_id}"><a class="footnote-ref" href="#fn:{key}">{number}</a></sup>'


def render_footnotes(uses: dict[str, int], *, render_entry: Callable[[str], str]) -> str:
    """Return the footnote list, one entry per key, with a link back to each citation."""
    items = []
    for number, (key, count) in enumerate(uses.items(), start=1):
        back_links = "".join(
            f'<a class="footnote-backref" href="#{build_reference_id(key, use=use)}" '
            f'title="Jump back to footnote {number} in the text">&#8617;</a>'
            for use in range(1, count + 1)
        )
        items.append(f'<li id="fn:{key}">\n<p>{render_entry(key)}&#160;{back_links}</p>\n</li>')
    entries = "\n".join(items)
    return f"\n{FOOTNOTE_LIST_START}\n<hr />\n<ol>\n{entries}\n</ol>\n</div>"


def render_citations(
    html: str,
    *,
    known_keys: Collection[str],
    render_entry: Callable[[str], str],
) -> str:
    """Return the page with the citations in its text as footnotes, or unchanged without any."""
    uses: dict[str, int] = {}

    def render_citation_group(match: re.Match[str]) -> str:
        keys = KEY_PATTERN.findall(match.group(1))
        unknown = [key for key in keys if key not in known_keys]
        if unknown:
            logger.warning("A docstring cites %s, which docs/references.bib lacks.", unknown)
            return match.group(0)
        return "".join(render_reference(key, uses=uses) for key in keys)

    pieces = []
    position = 0
    for start, end in find_text_spans(html):
        pieces.append(html[position:start])
        pieces.append(CITATION_PATTERN.sub(render_citation_group, html[start:end]))
        position = end
    pieces.append(html[position:])
    if not uses:
        return html
    if FOOTNOTE_LIST_START in html:
        logger.warning("A page cites sources in both its Markdown and its docstrings.")
    return "".join(pieces) + render_footnotes(uses, render_entry=render_entry)


def render_entry_html(entry_markdown: str) -> str:
    """Return a bibliography entry's Markdown as inline HTML, its links included."""
    return markdown.markdown(entry_markdown).removeprefix("<p>").removesuffix("</p>")


def on_page_content(html: str, /, *, config: MkDocsConfig, **_: object) -> str:
    """Turn the citations left in a converted page into footnotes; MkDocs calls it by name."""
    registry = config.plugins["bibtex"].registry
    return render_citations(
        html,
        known_keys=registry.bib_data.entries,
        render_entry=lambda key: render_entry_html(registry.reference_text(Citation(key=key))),
    )
