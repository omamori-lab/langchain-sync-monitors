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

A citation inside a link's text or a button also stays as it is, and is
logged as a warning, so a strict build fails until the author moves it. Its
footnote reference is itself a link, and HTML forbids a link inside another
(section 4.5.1) or inside a button (section 4.10.6) [@whatwg2026html]. The
hook does not move the reference to after the element instead: the element's
text can run on past the words a citation supports, which would part the
reference from them, and only the docstring's author knows where it belongs.
A link here is an ``a`` element, or an ``autoref`` element, the
cross-reference that mkdocs-autorefs [@mkdocsautorefs2026] turns into an
``a`` element after this hook runs. An end tag closes only the most recent
open element of its own name, and one with no such element open is ignored,
so a stray end tag ends neither code nor a link's text early. A self-closing
``a``, ``button`` or ``autoref`` tag opens its element everywhere: HTML ignores
the slash on an element that is not void (section 13.2.2) [@whatwg2026html],
and mkdocs-autorefs reads an ``autoref`` up to the next end tag
[@mkdocsautorefs2026]. Any other self-closing tag closes its element at once.
Inside SVG or MathML the slash does close an ``a``, but the hook does not track
where that content starts and ends, since HTML's rules apply again inside an
SVG ``foreignObject``. Treating every self-closing link as open costs at most
a false warning there, which fails a strict build loudly, and never a footnote
reference silently nested in a link.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Collection
from dataclasses import dataclass
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
# Elements whose content may hold no link: links, the cross-references that
# mkdocs-autorefs turns into links after this hook, and buttons.
LINK_FREE_TAGS = frozenset({"a", "autoref", "button"})
FOOTNOTE_LIST_START = '<div class="footnote">'


@dataclass(frozen=True, slots=True)
class TextSpan:
    """Where a run of a page's text starts and ends, and whether it lies where no link may go."""

    start: int
    end: int
    is_link_free: bool


class TextSpanParser(HTMLParser):
    """Record where each run of a page's text outside the verbatim tags starts and ends.

    With ``convert_charrefs=False`` the standard library's parser hands
    ``handle_data`` each run of text exactly as written, and ``getpos`` gives
    the line and column where that run starts, so the run is the slice of the
    page from there. Tags with their attributes, comments and character
    references reach other handlers, so no recorded span holds any of them.
    The parser ends a run only at a ``<`` or an ``&``, neither of which a
    citation holds, so no citation straddles two runs. It keeps the names of
    the open verbatim elements, so it skips the runs inside them, and the names
    of the open elements whose content may hold no link, inside verbatim ones
    as well, so each run knows whether it lies where no link may go.
    """

    def __init__(self, html: str) -> None:
        super().__init__(convert_charrefs=False)
        self.line_starts = [0, *(match.end() for match in NEWLINE_PATTERN.finditer(html))]
        self.open_verbatim_tags: list[str] = []
        self.open_link_free_tags: list[str] = []
        self.spans: list[TextSpan] = []

    @override
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Note an open verbatim or link-free element."""
        if tag in VERBATIM_TAGS:
            self.open_verbatim_tags.append(tag)
        if tag in LINK_FREE_TAGS:
            self.open_link_free_tags.append(tag)

    @override
    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Open a self-closing element, and close it again unless it may hold no link."""
        self.handle_starttag(tag, attrs)
        if tag not in LINK_FREE_TAGS:
            self.handle_endtag(tag)

    @override
    def handle_endtag(self, tag: str) -> None:
        """Close the most recent open element of this name, unless none is open."""
        remove_most_recent(tag, open_tags=self.open_verbatim_tags)
        remove_most_recent(tag, open_tags=self.open_link_free_tags)

    @override
    def handle_data(self, data: str) -> None:
        """Record the span of a run of text, unless it lies inside a verbatim element."""
        if self.open_verbatim_tags:
            return
        line, column = self.getpos()
        start = self.line_starts[line - 1] + column
        is_link_free = bool(self.open_link_free_tags)
        span = TextSpan(start=start, end=start + len(data), is_link_free=is_link_free)
        self.spans.append(span)


def remove_most_recent(tag: str, *, open_tags: list[str]) -> None:
    """Remove the most recently opened `tag` from `open_tags`, unless none is open."""
    for index in range(len(open_tags) - 1, -1, -1):
        if open_tags[index] == tag:
            del open_tags[index]
            return


def find_text_spans(html: str) -> list[TextSpan]:
    """Return each run of the page's text outside the verbatim tags."""
    parser = TextSpanParser(html)
    parser.feed(html)
    parser.close()
    return parser.spans


def keep_citation_in_link_free_text(match: re.Match[str]) -> str:
    """Return, as written, a citation where no link may go, and warn, which fails a strict build."""
    logger.warning(
        "A docstring cites %s inside a link or a button, which may not hold its footnote "
        "reference, itself a link; move the citation out of it.",
        match.group(0),
    )
    return match.group(0)


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
    for span in find_text_spans(html):
        render = keep_citation_in_link_free_text if span.is_link_free else render_citation_group
        pieces.append(html[position : span.start])
        pieces.append(CITATION_PATTERN.sub(render, html[span.start : span.end]))
        position = span.end
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
