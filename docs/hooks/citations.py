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
a browser reads as plain text stays as it is. A key missing from the
bibliography is logged as a warning, which fails a strict build, and so is a
page that already has footnotes of its own, whose docstring citations then
stay as written.

The hook finds the citations with the standard library's HTML parser, which is
fast but does not read markup as a browser does. It differs on some comments,
on the content of a script and on end tags that a browser ignores, so it can
take for text a citation that a browser reads inside code, a link or a button.
So the hook then reads the page it wrote with html5lib [@html5lib2020], which
implements the HTML parsing algorithm browsers follow (section 13.2)
[@whatwg2026html], with scripting on, as in a browser that runs the theme's
scripts. It keeps a footnote reference only where a browser finds it, outside
every link, button, code element and element whose content a browser reads as
plain text. A footnote reference is itself a link, and HTML forbids a link
inside another (section 4.5.1) or inside a button (section 4.10.6). A
cross-reference is still an ``autoref`` element here, and mkdocs-autorefs
[@mkdocsautorefs2026] turns it into a link after this hook runs, so the hook
reads each one as a link, found with mkdocs-autorefs's own pattern.
html5lib's last release dates from 2020 and does not implement the template
element (section 4.12.3): it reads one as an ordinary element, where a browser
keeps its content out of the page and ignores an end tag in it that closes an
element outside. So a page that holds a template stays as written, with a
warning.

Any other citation stays as written, with a warning naming the page, so a
strict build fails until someone moves the citation or fixes the markup around
it. That covers a citation whose reference a browser would drop or put where
no link may go, and one that a browser shows as text where the hook found no
citation, as when it is written with character references or follows markup
the standard library's parser reads otherwise. The hook does not move a
reference out of a link: the link's text can run on past the words a citation
supports, and only the docstring's author knows where it belongs. Last, the
page with its footnotes must read as the page as written, with each reference
in place of its citation and the footnote list at the end; if it does not, the
hook warns and returns the page as written.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Callable, Collection, Iterator
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from typing import override
from xml.dom.minidom import DocumentFragment, Element, Node, Text

import html5lib
import markdown
from mkdocs.config.defaults import MkDocsConfig
from mkdocs.structure.pages import Page
from mkdocs_autorefs import AUTOREF_RE
from mkdocs_bibtex.citation import Citation

# MkDocs counts the warnings of the loggers under "mkdocs" towards --strict.
logger = logging.getLogger("mkdocs.hooks.citations")

CITATION_PATTERN = re.compile(r"\[(@[\w:.-]+(?:;\s*@[\w:.-]+)*)\]")
KEY_PATTERN = re.compile(r"@([\w:.-]+)")
NEWLINE_PATTERN = re.compile("\n")
# A template start tag, which html5lib reads as an ordinary element and a browser does not.
TEMPLATE_PATTERN = re.compile(r"<template(?=[\s/>]|$)", re.IGNORECASE)
# Code, and the elements whose content a browser that runs scripts reads as plain text.
VERBATIM_TAGS = frozenset({"code", "noscript", "pre", "script", "style", "textarea", "title"})
# The elements that may not hold a footnote reference, itself a link: links, the
# cross-references that mkdocs-autorefs turns into links, buttons and verbatim elements.
REFERENCE_FREE_TAGS = frozenset({"a", "autoref", "button", *VERBATIM_TAGS})
FOOTNOTE_LIST_START = '<div class="footnote">'
# html5lib's walk through a parsed page: its start tags, end tags, text and comments.
TREE_WALKER = html5lib.getTreeWalker("dom")


@dataclass(frozen=True, slots=True)
class PageCitation:
    """A citation in a page's text: where it starts and ends, how it reads and the keys it cites."""

    start: int
    end: int
    text: str
    keys: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FootnoteReference:
    """A footnote reference the hook wrote, and the citation it stands for.

    ``stands_for`` is the citation's text for the first reference of a citation,
    and empty for the others, so the references of a citation together stand
    for its text once.
    """

    citation: int
    reference_id: str
    markup: str
    stands_for: str


@dataclass(frozen=True, slots=True)
class Rendering:
    """A page with some of its citations as footnote references, and its footnote list.

    ``tree`` is the page with its footnote list as a browser reads it, which
    `is_read_as_original` turns back into the page as written.
    """

    html: str
    references: list[FootnoteReference]
    footnotes: str
    tree: DocumentFragment


class CitationParser(HTMLParser):
    """Record each citation in a page's text outside the verbatim tags, where it starts and ends.

    With ``convert_charrefs=False`` the standard library's parser hands
    ``handle_data`` each run of text exactly as written, and ``getpos`` gives
    the line and column where that run starts, so the run is the slice of the
    page from there. Tags with their attributes, comments and character
    references reach other handlers, so no recorded citation lies in any of
    them. The parser ends a run only at a ``<`` or an ``&``, neither of which a
    citation holds, so no citation straddles two runs.
    """

    def __init__(self, html: str) -> None:
        super().__init__(convert_charrefs=False)
        self.line_starts = [0, *(match.end() for match in NEWLINE_PATTERN.finditer(html))]
        self.verbatim_depth = 0
        self.citations: list[PageCitation] = []

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
        """Record each citation in a run of text, unless the run lies inside a verbatim element."""
        if self.verbatim_depth:
            return
        line, column = self.getpos()
        run_start = self.line_starts[line - 1] + column
        for match in CITATION_PATTERN.finditer(data):
            citation = PageCitation(
                start=run_start + match.start(),
                end=run_start + match.end(),
                text=match.group(0),
                keys=tuple(KEY_PATTERN.findall(match.group(1))),
            )
            self.citations.append(citation)


def find_citations(html: str) -> list[PageCitation]:
    """Return each citation in the page's text outside the verbatim tags, in page order."""
    parser = CitationParser(html)
    parser.feed(html)
    parser.close()
    return parser.citations


def find_unknown_citations(
    citations: list[PageCitation],
    *,
    known_keys: Collection[str],
    page_path: str,
) -> set[int]:
    """Return the citations that name a key the bibliography lacks, and warn about each."""
    unknown_citations = set()
    for index, citation in enumerate(citations):
        unknown = [key for key in citation.keys if key not in known_keys]
        if unknown:
            logger.warning(
                "%s: a docstring cites %s, which docs/references.bib lacks.",
                page_path,
                unknown,
            )
            unknown_citations.add(index)
    return unknown_citations


def build_reference_id(key: str, *, use: int) -> str:
    """Return the id of the `use`-th citation of `key`, as Python-Markdown numbers repeats."""
    return f"fnref:{key}" if use == 1 else f"fnref{use}:{key}"


def render_reference(
    key: str,
    *,
    uses: dict[str, int],
    citation: int,
    stands_for: str,
) -> FootnoteReference:
    """Return one more footnote reference to `key`, and count it in `uses`."""
    uses[key] = uses.get(key, 0) + 1
    number = list(uses).index(key) + 1
    reference_id = build_reference_id(key, use=uses[key])
    markup = f'<sup id="{reference_id}"><a class="footnote-ref" href="#fn:{key}">{number}</a></sup>'
    return FootnoteReference(
        citation=citation,
        reference_id=reference_id,
        markup=markup,
        stands_for=stands_for,
    )


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


def render_citation_references(
    citation: PageCitation,
    *,
    index: int,
    uses: dict[str, int],
) -> list[FootnoteReference]:
    """Return a citation's footnote references, one per key, the first standing for its text."""
    return [
        render_reference(key, uses=uses, citation=index, stands_for="" if order else citation.text)
        for order, key in enumerate(citation.keys)
    ]


def render_references(
    html: str,
    *,
    citations: list[PageCitation],
    kept: Collection[int],
    render_entry: Callable[[str], str],
) -> Rendering:
    """Return the page with each citation outside `kept` as footnote references."""
    uses: dict[str, int] = {}
    references: list[FootnoteReference] = []
    pieces = []
    position = 0
    for index, citation in enumerate(citations):
        if index in kept:
            continue
        citation_references = render_citation_references(citation, index=index, uses=uses)
        references.extend(citation_references)
        pieces.append(html[position : citation.start])
        pieces.extend(reference.markup for reference in citation_references)
        position = citation.end
    pieces.append(html[position:])
    rendered_html = "".join(pieces)
    footnotes = render_footnotes(uses, render_entry=render_entry) if uses else ""
    return Rendering(
        html=rendered_html,
        references=references,
        footnotes=footnotes,
        tree=parse_page(rendered_html + footnotes),
    )


def link_cross_references(html: str) -> str:
    """Return the page with each cross-reference turned into a link, as mkdocs-autorefs will."""
    return AUTOREF_RE.sub(lambda match: f"<a>{match.group('title')}</a>", html)


def parse_page(html: str) -> DocumentFragment:
    """Return the page as a browser reads it once its cross-references are links."""
    # Scripting on, as in a browser that runs the theme's scripts: noscript content is then text.
    fragment = html5lib.parseFragment(
        link_cross_references(html),
        container="div",
        treebuilder="dom",
        scripting=True,
    )
    # Join adjacent text, so two pages that read the same walk the same.
    fragment.normalize()
    return fragment


def iterate_nodes(root: Node) -> Iterator[Node]:
    """Yield every node under `root`, in page order."""
    pending = list(reversed(root.childNodes))
    while pending:
        node = pending.pop()
        yield node
        pending.extend(reversed(node.childNodes))


def read_ancestor_names(node: Node) -> set[str]:
    """Return the names of the elements that hold `node`."""
    names = set()
    parent = node.parentNode
    while isinstance(parent, Element):
        names.add(parent.localName)
        parent = parent.parentNode
    return names


def find_reference_elements(tree: DocumentFragment) -> dict[str, list[Element]]:
    """Map the id of each ``sup`` element in the page to the elements that carry it."""
    elements: dict[str, list[Element]] = {}
    for node in iterate_nodes(tree):
        if isinstance(node, Element) and node.localName == "sup":
            elements.setdefault(node.getAttribute("id"), []).append(node)
    return elements


def is_read_as_written(node: Node | None, *, markup: str) -> bool:
    """Tell whether `node` reads in the page as the last node of `markup` reads on its own."""
    expected = parse_page(markup).lastChild
    return node is not None and list(TREE_WALKER(node)) == list(TREE_WALKER(expected))


def is_placed_where_a_link_may_go(elements: list[Element]) -> bool:
    """Tell whether a reference's id names one element, outside every element that may hold no link.

    A browser moves a footnote link out of its ``sup`` only to close another
    link that holds the ``sup``, so the ``sup``'s ancestors show it; any other
    change to the page is for `is_read_as_original` to find.
    """
    return len(elements) == 1 and not read_ancestor_names(elements[0]) & REFERENCE_FREE_TAGS


def find_misplaced_citations(rendering: Rendering) -> set[int]:
    """Return the citations with a reference that a browser drops or puts where no link may go."""
    elements = find_reference_elements(rendering.tree)
    return {
        reference.citation
        for reference in rendering.references
        if not is_placed_where_a_link_may_go(elements.get(reference.reference_id, []))
    }


def render_placed_references(
    html: str,
    *,
    citations: list[PageCitation],
    kept: frozenset[int],
    render_entry: Callable[[str], str],
    page_path: str,
) -> Rendering:
    """Return the page with each reference a browser keeps as written, and warn about the rest.

    Taking a citation out can change how a browser reads the ones after it, as
    when a footnote reference closed the link it sat in, so the page is written
    and read again until every reference left reads as written. Each round
    parses the whole page, so n citations in one link take n + 1 rounds; only a
    page that already fails a strict build pays for more than one.
    """
    while True:
        rendering = render_references(
            html,
            citations=citations,
            kept=kept,
            render_entry=render_entry,
        )
        misplaced = find_misplaced_citations(rendering)
        if not misplaced:
            return rendering
        for index in sorted(misplaced):
            logger.warning(
                "%s: a docstring cites %s inside a link, a button or code, or in markup a "
                "browser reads otherwise, where its footnote reference, itself a link, cannot "
                "go; it stays as written.",
                page_path,
                citations[index].text,
            )
        kept |= misplaced


def find_citations_shown_as_text(tree: DocumentFragment) -> Counter[str]:
    """Count the citations a browser shows as text: outside comments, tags and verbatim elements."""
    shown: Counter[str] = Counter()
    for node in iterate_nodes(tree):
        if isinstance(node, Text) and not read_ancestor_names(node) & VERBATIM_TAGS:
            shown.update(match.group(0) for match in CITATION_PATTERN.finditer(node.data))
    return shown


def warn_about_citations_not_found(
    original: DocumentFragment,
    *,
    citations: list[PageCitation],
    page_path: str,
) -> None:
    """Warn about each citation a browser shows as text that the hook did not find in the text."""
    found = Counter(citation.text for citation in citations)
    for text in find_citations_shown_as_text(original) - found:
        logger.warning(
            "%s: a browser shows %s as text, but the hook did not find it in the page's text, "
            "as when it is written with character references or follows markup the hook's "
            "parser reads otherwise, so it stays as written; check how it is written and the "
            "markup before it.",
            page_path,
            text,
        )


def replace_with_text(element: Element, *, text: str) -> None:
    """Put `text` in the page in place of `element`."""
    parent = element.parentNode
    document = element.ownerDocument
    if isinstance(parent, Element | DocumentFragment) and document is not None:
        parent.replaceChild(document.createTextNode(text), element)


def is_read_as_original(rendering: Rendering, *, original: DocumentFragment) -> bool:
    """Tell whether the page with its footnotes reads as the page as written, but for them.

    The footnote list must close the page as written, and the page must read as
    the original once each reference is back to the text of its citation. Each
    reference's id must name one element already, as `find_misplaced_citations`
    checks. This puts the text back in the rendering's tree, so nothing reads
    the tree after it.
    """
    tree = rendering.tree
    if not is_read_as_written(tree.lastChild, markup=rendering.footnotes):
        return False
    elements = find_reference_elements(tree)
    for reference in rendering.references:
        replace_with_text(elements[reference.reference_id][0], text=reference.stands_for)
    tree.normalize()
    return list(TREE_WALKER(tree)) == list(TREE_WALKER(original))


def find_reason_to_keep_citations(html: str, *, citations: list[PageCitation]) -> str | None:
    """Return why every citation on the page stays as written, or None if the hook can place them.

    A page with a template stays as written even without a citation the hook
    found, since html5lib cannot tell which citations a browser shows on it.
    """
    if TEMPLATE_PATTERN.search(html):
        return (
            "it holds a template element, which html5lib cannot read as a browser does, so its "
            "citations stay as written."
        )
    if citations and FOOTNOTE_LIST_START in html:
        return (
            "it has footnotes of its own, so the citations in its docstrings stay as written; "
            "cite sources in its Markdown or in its docstrings, not both."
        )
    return None


def render_citations(
    html: str,
    *,
    known_keys: Collection[str],
    render_entry: Callable[[str], str],
    page_path: str,
) -> str:
    """Return the page with the citations in its text as footnotes, or unchanged without any."""
    if CITATION_PATTERN.search(unescape(html)) is None:
        return html
    citations = find_citations(html)
    unknown = find_unknown_citations(citations, known_keys=known_keys, page_path=page_path)
    reason = find_reason_to_keep_citations(html, citations=citations)
    if reason is not None:
        logger.warning("%s: %s", page_path, reason)
        return html
    rendering = render_placed_references(
        html,
        citations=citations,
        kept=frozenset(unknown),
        render_entry=render_entry,
        page_path=page_path,
    )
    original = parse_page(html + rendering.footnotes)
    warn_about_citations_not_found(original, citations=citations, page_path=page_path)
    if not rendering.references:
        return html
    if not is_read_as_original(rendering, original=original):
        logger.warning(
            "%s: its footnotes would change how a browser reads the markup around them, so "
            "its citations stay as written; check the markup around each one.",
            page_path,
        )
        return html
    return rendering.html + rendering.footnotes


def render_entry_html(entry_markdown: str) -> str:
    """Return a bibliography entry's Markdown as inline HTML, its links included."""
    return markdown.markdown(entry_markdown).removeprefix("<p>").removesuffix("</p>")


def on_page_content(html: str, /, *, page: Page, config: MkDocsConfig, **_: object) -> str:
    """Turn the citations left in a converted page into footnotes; MkDocs calls it by name."""
    registry = config.plugins["bibtex"].registry
    return render_citations(
        html,
        known_keys=registry.bib_data.entries,
        render_entry=lambda key: render_entry_html(registry.reference_text(Citation(key=key))),
        page_path=page.file.src_uri,
    )
