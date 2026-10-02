"""The docs hook turns the citations in rendered docstrings into footnotes.

mkdocs-bibtex rewrites the citations of a page's Markdown only, so the API
reference's docstrings reach the page with their citations as literal text.
``docs/hooks/citations.py`` gives them the footnote markup of the other pages,
and reads the page it wrote with html5lib, as a browser would, so a footnote
reference never lands inside a link, a button or code. The hook needs the docs
dependency group; without it these tests are skipped, and ``scripts/check.sh``
sets ``REQUIRE_DOCS_GROUP=1``, which makes a missing module fail them instead.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import re
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from xml.dom.minidom import Document, Element, Node, Text

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
HOOK_PATH = REPOSITORY_ROOT / "docs" / "hooks" / "citations.py"
CONFIG_PATH = REPOSITORY_ROOT / "mkdocs.yml"
HOOK_CONFIG_ENTRY = "docs/hooks/citations.py"
HOOK_LOGGER = "mkdocs.hooks.citations"
KNOWN_KEYS = frozenset({"first2026", "second2026"})
PAGE_PATH = "reference/api.md"
FIRST_REFERENCE = (
    '<sup id="fnref:first2026"><a class="footnote-ref" href="#fn:first2026">1</a></sup>'
)
# The elements a footnote link may not sit in, for the tests' own reading of a page.
VERBATIM_NAMES = frozenset({"code", "noscript", "pre", "script", "style", "textarea", "title"})
LINK_FREE_NAMES = frozenset({"a", "autoref", "button", *VERBATIM_NAMES})
CITATION_TEXT_PATTERN = re.compile(r"\[@[^\]]+\]")


def render_test_entry(key: str) -> str:
    """Return a stand-in for a formatted bibliography entry."""
    return f"Entry for {key}."


def render_test_page(page: str, *, hook: ModuleType) -> str:
    """Return the test page with its citations rendered against the test bibliography."""
    return hook.render_citations(
        page,
        known_keys=KNOWN_KEYS,
        render_entry=render_test_entry,
        page_path=PAGE_PATH,
    )


def read_ancestor_names(node: Node) -> list[str]:
    """Return the names of the elements that hold `node` in a parsed test page, nearest first."""
    names = []
    parent = node.parentNode
    while isinstance(parent, Element):
        names.append(parent.localName)
        parent = parent.parentNode
    return names


def parse_test_page(html: str) -> Document:
    """Return the test page as a browser that runs scripts reads it, as a document."""
    html5lib = importlib.import_module("html5lib")
    return html5lib.parse(f"<!DOCTYPE html><body>{html}", treebuilder="dom", scripting=True)


def find_misread_footnote_links(html: str) -> list[str]:
    """Return each footnote link a browser would not keep in its ``sup``, or would nest or drop."""
    document = parse_test_page(html)
    links = [
        link
        for link in document.getElementsByTagName("a")
        if link.getAttribute("class") == "footnote-ref"
    ]
    misread = [
        link.toxml()
        for link in links
        if read_ancestor_names(link)[0] != "sup"
        or LINK_FREE_NAMES.intersection(read_ancestor_names(link))
    ]
    if len(links) != html.count('class="footnote-ref"'):
        misread.append("a footnote link that a browser reads as text")
    return misread


def find_unwarned_citations(html: str, *, messages: list[str]) -> list[str]:
    """Return each citation a browser shows as text in the page that no warning names."""
    shown = []
    for element in parse_test_page(html).getElementsByTagName("*"):
        if VERBATIM_NAMES.intersection([element.localName, *read_ancestor_names(element)]):
            continue
        for child in element.childNodes:
            if isinstance(child, Text):
                shown.extend(CITATION_TEXT_PATTERN.findall(child.data))
    return [citation for citation in shown if not any(citation in text for text in messages)]


@pytest.fixture
def hook(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Load the hook from its file, registered as a module while it runs, as MkDocs does."""
    if os.environ.get("REQUIRE_DOCS_GROUP") != "1":
        pytest.importorskip("mkdocs_bibtex")
    specification = importlib.util.spec_from_file_location("citations_hook", HOOK_PATH)
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    monkeypatch.setitem(sys.modules, specification.name, module)
    specification.loader.exec_module(module)
    return module


def test_a_citation_becomes_a_footnote_reference_with_its_entry(hook: ModuleType) -> None:
    # Arrange
    page = "<p>Auto mode [@first2026] blocks a step.</p>"

    # Act
    rendered = render_test_page(page, hook=hook)

    # Assert
    assert "[@" not in rendered
    assert FIRST_REFERENCE in rendered
    assert '<li id="fn:first2026">\n<p>Entry for first2026.&#160;' in rendered
    assert rendered.endswith("</ol>\n</div>")


def test_a_page_without_citations_is_unchanged(hook: ModuleType) -> None:
    # Arrange
    page = '<p>No sources here.</p>\n<div class="footnote"><ol></ol></div>'

    # Act
    rendered = render_test_page(page, hook=hook)

    # Assert
    assert rendered == page


@pytest.mark.parametrize(
    "page",
    [
        "<p><code>[@first2026]</code></p>\n<pre><code>cite [@second2026]\n</code></pre>",
        '<script>const cite = "[@first2026]";</script>',
        '<style>p::after { content: "[@first2026]"; }</style>',
        "<textarea>[@first2026]</textarea>",
        '<svg><title>A figure after [@first2026]</title><rect width="1" /></svg>',
        '<p title="See [@first2026]">Text</p>',
        '<p title="a > [@first2026]">Text</p>',
        "<p><a href='#' data-cite='[@first2026]'>Text</a></p>",
        "<p><!-- a comment that cites [@first2026] -->Text</p>",
        "<p><noscript>[@first2026]</noscript></p>",
        "<p><noscript><code>[@first2026]</code></noscript></p>",
    ],
    ids=[
        "code",
        "script",
        "style",
        "textarea",
        "title",
        "double-quoted-attribute",
        "attribute-after-a-quoted-angle-bracket",
        "single-quoted-attribute",
        "comment",
        "noscript",
        "code-in-noscript",
    ],
)
def test_a_citation_a_browser_does_not_show_as_text_stays_quietly(
    hook: ModuleType,
    caplog: pytest.LogCaptureFixture,
    page: str,
) -> None:
    # Act
    with caplog.at_level(logging.WARNING, logger=HOOK_LOGGER):
        rendered = render_test_page(page, hook=hook)

    # Assert
    assert rendered == page
    assert caplog.records == []


def test_a_citation_inside_code_stays_while_one_in_its_text_renders(hook: ModuleType) -> None:
    # Arrange
    page = "<p>Write <code>[@first2026]</code> to cite, as [@second2026] does.</p>"

    # Act
    rendered = render_test_page(page, hook=hook)

    # Assert
    assert rendered.startswith(
        "<p>Write <code>[@first2026]</code> to cite, as "
        '<sup id="fnref:second2026"><a class="footnote-ref" href="#fn:second2026">1</a></sup>'
        " does.</p>\n",
    )
    assert '<li id="fn:second2026">' in rendered
    assert "fn:first2026" not in rendered


def test_a_text_citation_renders_in_place_after_markup_it_leaves_alone(hook: ModuleType) -> None:
    # Arrange
    before = (
        '<h2 id="a">A &amp; B</h2>\n\n'
        "<!-- a comment that cites [@second2026] -->\n"
        '<p title="See [@second2026]">Line one,\n'
        "line two &#8212; and &copy;"
    )
    after = "</p>\n"
    page = f"{before}[@first2026]{after}"

    # Act
    rendered = render_test_page(page, hook=hook)

    # Assert
    assert rendered == (
        f"{before}"
        f"{FIRST_REFERENCE}"
        f"{after}"
        "\n"
        '<div class="footnote">\n<hr />\n<ol>\n'
        '<li id="fn:first2026">\n<p>Entry for first2026.&#160;'
        '<a class="footnote-backref" href="#fnref:first2026" '
        'title="Jump back to footnote 1 in the text">&#8617;</a></p>\n</li>\n'
        "</ol>\n</div>"
    )


def test_a_group_and_a_repeat_number_footnotes_by_first_use(hook: ModuleType) -> None:
    # Arrange
    page = "<p>A [@second2026; @first2026] and again [@second2026].</p>"

    # Act
    rendered = render_test_page(page, hook=hook)

    # Assert
    assert '<sup id="fnref:second2026"><a class="footnote-ref" href="#fn:second2026">1</a>' in (
        rendered
    )
    assert '<sup id="fnref:first2026"><a class="footnote-ref" href="#fn:first2026">2</a>' in (
        rendered
    )
    assert '<sup id="fnref2:second2026"><a class="footnote-ref" href="#fn:second2026">1</a>' in (
        rendered
    )
    assert rendered.index('<li id="fn:second2026">') < rendered.index('<li id="fn:first2026">')
    assert rendered.count('href="#fnref2:second2026"') == 1


def test_an_unknown_key_stays_literal_and_warns(
    hook: ModuleType,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange
    page = "<p>A [@missing2026] source.</p>"

    # Act
    with caplog.at_level(logging.WARNING, logger=HOOK_LOGGER):
        rendered = render_test_page(page, hook=hook)

    # Assert
    assert rendered == page
    assert [record.name for record in caplog.records] == [HOOK_LOGGER]
    assert caplog.records[0].getMessage().startswith(f"{PAGE_PATH}: ")
    assert "missing2026" in caplog.records[0].getMessage()


def test_a_citation_in_link_text_stays_as_written_and_warns(
    hook: ModuleType,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange
    page = '<a href="#x">see [@first2026]</a>'

    # Act
    with caplog.at_level(logging.WARNING, logger=HOOK_LOGGER):
        rendered = render_test_page(page, hook=hook)

    # Assert
    assert rendered == page
    assert [record.name for record in caplog.records] == [HOOK_LOGGER]
    assert (
        caplog.records[0]
        .getMessage()
        .startswith(
            f"{PAGE_PATH}: a docstring cites [@first2026] inside a link, a button or code",
        )
    )


@pytest.mark.parametrize(
    "page",
    [
        '<p><a href="#x"><em>see [@first2026]</em></a></p>',
        '<p><a id="x">see [@first2026]</a></p>',
        '<p><autoref identifier="x" optional hover>see [@first2026]</autoref></p>',
        '<p><a href="#x">see</p>\n<p>then [@first2026]</p>',
        '<p>A stray </a> end tag, then <a href="#x">see [@first2026]</a></p>',
        '<p><a id="x"/> then [@first2026]</p>',
        '<p><autoref identifier="x"/> then [@first2026]</p>',
        (
            '<p><autoref identifier="x"/> see</p>\n'
            '<p>[@first2026] <autoref identifier="y">Y</autoref></p>'
        ),
        "<p><button>see [@first2026]</button></p>",
        "<p><button>Copy </a>[@first2026]</button></p>",
        '<svg><a href="#x"/></svg><p><a href="#y">see [@first2026]</a></p>',
        '<svg><foreignObject><a href="#x"/>[@first2026]</a></foreignObject></svg>',
        "<p><code/>[@first2026]</code></p>",
        "<p><pre/>[@first2026]</pre></p>",
        "<p><code>x</pre>[@first2026]</code></p>",
        '<a href="#x"><object></a>[@first2026]</object></a>',
        '<a href="#x"><table><tr><td></a>[@first2026]</td></tr></table></a>',
        '<a href="#x"><marquee></a>[@first2026]</marquee></a>',
        "<button><table><tr><td></button>[@first2026]</td></tr></table></button>",
        "<p><select><option>[@first2026]</option></select></p>",
        '<p><a href="#x">see <noscript></a></noscript>[@first2026]</p>',
    ],
    ids=[
        "inside-an-inline-element-in-a-link",
        "in-a-link-without-an-href",
        "in-a-cross-reference",
        "after-an-unclosed-link",
        "in-a-link-after-a-stray-end-tag",
        "after-a-self-closing-link",
        "after-a-self-closing-cross-reference",
        "between-a-self-closing-cross-reference-and-a-later-end-tag",
        "in-a-button",
        "in-a-button-after-a-stray-link-end-tag",
        "in-a-link-after-a-self-closing-svg-link",
        "after-a-self-closing-link-in-svg-foreign-object",
        "after-a-self-closing-code-tag",
        "after-a-self-closing-pre-tag",
        "in-code-after-a-stray-end-tag",
        "in-a-link-past-an-end-tag-an-object-ignores",
        "in-a-link-past-an-end-tag-a-table-cell-ignores",
        "in-a-link-past-an-end-tag-a-marquee-ignores",
        "in-a-button-past-an-end-tag-a-table-cell-ignores",
        "in-an-option-that-drops-the-reference",
        "in-a-link-past-an-end-tag-in-noscript",
    ],
)
def test_a_citation_where_a_browser_would_not_keep_its_reference_stays_and_warns(
    hook: ModuleType,
    caplog: pytest.LogCaptureFixture,
    page: str,
) -> None:
    # Act
    with caplog.at_level(logging.WARNING, logger=HOOK_LOGGER):
        rendered = render_test_page(page, hook=hook)

    # Assert
    assert rendered == page
    assert [record.name for record in caplog.records] == [HOOK_LOGGER]
    assert caplog.records[0].getMessage().startswith(f"{PAGE_PATH}: a docstring cites [@first2026]")


@pytest.mark.parametrize(
    "before",
    [
        '<p><a href="#x">see</a> ',
        '<p><autoref identifier="x" optional hover>see</autoref> ',
        "<p>A stray </a> end tag, then ",
        "<p>A stray </code> end tag, then ",
        '<p><code><a href="#x">Type</a></code> ',
        '<p><a id="x"/> see</a> ',
        "<p><button>Copy</button> ",
        "<p><br/> ",
        "<p><code>x</pre></code> ",
        '<svg><a href="#x"/></svg><p>',
        '<math><a href="#x"/></math><p>',
        "<svg><button/></svg><p>",
        '<svg><autoref identifier="x"/></svg><p>',
        "<table><tr><td>x</td></tr></table><p>",
        "<p><noscript><p>x</p></noscript> ",
        "<p><templates>x</templates> ",
    ],
    ids=[
        "a-link",
        "a-cross-reference",
        "a-stray-link-end-tag",
        "a-stray-code-end-tag",
        "a-link-inside-code",
        "a-self-closing-link-closed-later",
        "a-button",
        "a-self-closing-void-element",
        "code-holding-a-stray-end-tag",
        "a-self-closing-svg-link",
        "a-self-closing-mathml-link",
        "a-self-closing-svg-button",
        "a-self-closing-svg-cross-reference-with-no-end-tag-after",
        "a-table",
        "a-noscript",
        "an-element-whose-name-starts-with-template",
    ],
)
def test_a_citation_after_markup_a_browser_closes_renders_without_a_warning(
    hook: ModuleType,
    caplog: pytest.LogCaptureFixture,
    before: str,
) -> None:
    # Arrange
    page = f"{before}[@first2026].</p>"

    # Act
    with caplog.at_level(logging.WARNING, logger=HOOK_LOGGER):
        rendered = render_test_page(page, hook=hook)

    # Assert
    assert rendered.startswith(f"{before}{FIRST_REFERENCE}.</p>\n")
    assert caplog.records == []


@pytest.mark.parametrize(
    "page",
    [
        "<p><!-->[@first2026]</p>",
        "<p><!--->[@first2026]--></p>",
        '<p><!-->[@first2026] <a href="#x">-->[@second2026]</a></p>',
        "<script><!--<script></script>[@first2026]</script>",
        "<p><script/>[@first2026]</script></p>",
    ],
    ids=[
        "after-an-empty-comment",
        "after-an-empty-comment-with-a-dash",
        "around-a-link-the-standard-parser-reads-as-a-comment",
        "in-a-script-after-a-double-escape",
        "after-a-self-closing-script-tag",
    ],
)
def test_a_citation_in_markup_the_parsers_read_differently_never_nests_silently(
    hook: ModuleType,
    caplog: pytest.LogCaptureFixture,
    page: str,
) -> None:
    # Act: how the standard library's parser reads these depends on the Python version.
    with caplog.at_level(logging.WARNING, logger=HOOK_LOGGER):
        rendered = render_test_page(page, hook=hook)

    # Assert
    messages = [record.getMessage() for record in caplog.records]
    assert find_misread_footnote_links(rendered) == []
    assert find_unwarned_citations(rendered, messages=messages) == []


def test_each_citation_in_a_link_warns_though_the_first_reference_would_close_it(
    hook: ModuleType,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange: the first footnote link would close the outer link, so the
    # second reference would read as written until the first stays as written.
    page = '<p><a href="#x">see [@first2026] and [@second2026]</a></p>'

    # Act
    with caplog.at_level(logging.WARNING, logger=HOOK_LOGGER):
        rendered = render_test_page(page, hook=hook)

    # Assert
    assert rendered == page
    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 2
    assert "[@first2026]" in messages[0]
    assert "[@second2026]" in messages[1]


@pytest.mark.parametrize(
    "page",
    [
        '<p>Claim [@first2026].</p><p><a href="#x">see</p>',
        "<p>Claim [@first2026].</p><p><code>open",
        "<svg><text>A [@first2026] B</text></svg><p>After the figure.</p>",
    ],
    ids=[
        "a-link-left-open-at-the-end",
        "code-left-open-at-the-end",
        "svg-text-the-reference-would-end",
    ],
)
def test_a_page_its_footnotes_would_change_stays_as_written_and_warns(
    hook: ModuleType,
    caplog: pytest.LogCaptureFixture,
    page: str,
) -> None:
    # Act
    with caplog.at_level(logging.WARNING, logger=HOOK_LOGGER):
        rendered = render_test_page(page, hook=hook)

    # Assert
    assert rendered == page
    assert [record.getMessage() for record in caplog.records] == [
        f"{PAGE_PATH}: its footnotes would change how a browser reads the markup around them, "
        "so its citations stay as written; check the markup around each one.",
    ]


@pytest.mark.parametrize(
    "page",
    [
        # The table cell closes the code in a browser, but the standard
        # library's parser waits for its end tag.
        "<table><tr><td><code>x</td></tr></table><p>[@first2026]</p>",
        "<p>Write &#91;@first2026&#93; to cite.</p>",
        "<p>Write &#91;@first2026&#93; to cite, as [@second2026] does.</p>",
        "<p>Write &lsqb;@first2026&rsqb; to cite.</p>",
        "<p>Write [@first&#50;026] to cite.</p>",
    ],
    ids=[
        "after-code-a-table-cell-closes",
        "in-character-references",
        "in-character-references-beside-a-citation",
        "in-named-character-references",
        "with-a-character-reference-in-its-key",
    ],
)
def test_a_citation_a_browser_shows_that_the_hook_did_not_find_stays_and_warns(
    hook: ModuleType,
    caplog: pytest.LogCaptureFixture,
    page: str,
) -> None:
    # Act
    with caplog.at_level(logging.WARNING, logger=HOOK_LOGGER):
        rendered = render_test_page(page, hook=hook)

    # Assert
    assert "fn:first2026" not in rendered
    assert rendered.startswith(page.removesuffix("[@second2026] does.</p>"))
    assert [record.getMessage() for record in caplog.records] == [
        f"{PAGE_PATH}: a browser shows [@first2026] as text, but the hook did not find it in "
        "the page's text, as when it is written with character references or follows markup "
        "the hook's parser reads otherwise, so it stays as written; check how it is written "
        "and the markup before it.",
    ]


@pytest.mark.parametrize(
    "page",
    [
        '<p><a href="#x">see <template></a></template>[@first2026]</p>',
        '<p><a href="#x">see <TEMPLATE></a></TEMPLATE>[@first2026]</p>',
        "<template><p>[@first2026]</p></template><p>[@second2026]</p>",
    ],
    ids=[
        "in-a-link-past-an-end-tag-a-template-ignores",
        "in-a-link-past-an-end-tag-an-upper-case-template-ignores",
        "in-template-content-a-browser-does-not-show",
    ],
)
def test_a_page_that_holds_a_template_stays_as_written_and_warns(
    hook: ModuleType,
    caplog: pytest.LogCaptureFixture,
    page: str,
) -> None:
    # Act: html5lib reads a template as an ordinary element, so no oracle here.
    with caplog.at_level(logging.WARNING, logger=HOOK_LOGGER):
        rendered = render_test_page(page, hook=hook)

    # Assert
    assert rendered == page
    assert [record.getMessage() for record in caplog.records] == [
        f"{PAGE_PATH}: it holds a template element, which html5lib cannot read as a browser "
        "does, so its citations stay as written.",
    ]


def test_a_citation_outside_a_link_renders_while_one_inside_stays(
    hook: ModuleType,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange
    page = '<p><a href="#x">see [@second2026]</a> and [@first2026].</p>'

    # Act
    with caplog.at_level(logging.WARNING, logger=HOOK_LOGGER):
        rendered = render_test_page(page, hook=hook)

    # Assert
    assert rendered.startswith(
        f'<p><a href="#x">see [@second2026]</a> and {FIRST_REFERENCE}.</p>\n',
    )
    assert "fn:second2026" not in rendered
    assert [record.name for record in caplog.records] == [HOOK_LOGGER]


def test_an_unknown_key_in_link_text_warns_once_about_the_key(
    hook: ModuleType,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange
    page = '<p><a href="#x">see [@missing2026]</a></p>'

    # Act
    with caplog.at_level(logging.WARNING, logger=HOOK_LOGGER):
        rendered = render_test_page(page, hook=hook)

    # Assert
    assert rendered == page
    assert [record.name for record in caplog.records] == [HOOK_LOGGER]
    assert "missing2026" in caplog.records[0].getMessage()


@pytest.mark.parametrize(
    "page",
    [
        '<p>A [@first2026] source.</p>\n<div class="footnote"><ol></ol></div>',
        (
            '<p>A Markdown citation<sup id="fnref:first2026"><a class="footnote-ref" '
            'href="#fn:first2026">1</a></sup> and a docstring one [@first2026].</p>\n'
            '<div class="footnote">\n<hr />\n<ol>\n<li id="fn:first2026">\n<p>Entry.</p>\n'
            "</li>\n</ol>\n</div>"
        ),
    ],
    ids=["another-key", "the-same-key"],
)
def test_a_page_with_footnotes_of_its_own_keeps_its_citations_and_warns(
    hook: ModuleType,
    caplog: pytest.LogCaptureFixture,
    page: str,
) -> None:
    # Act
    with caplog.at_level(logging.WARNING, logger=HOOK_LOGGER):
        rendered = render_test_page(page, hook=hook)

    # Assert
    assert rendered == page
    assert [record.getMessage() for record in caplog.records] == [
        f"{PAGE_PATH}: it has footnotes of its own, so the citations in its docstrings stay as "
        "written; cite sources in its Markdown or in its docstrings, not both.",
    ]


def test_a_page_with_footnotes_of_its_own_and_a_citation_in_code_stays_quietly(
    hook: ModuleType,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange
    page = (
        "<p>Write <code>[@first2026]</code> to cite.</p>\n"
        '<div class="footnote"><ol><li id="fn:x"><p>Entry.</p></li></ol></div>'
    )

    # Act
    with caplog.at_level(logging.WARNING, logger=HOOK_LOGGER):
        rendered = render_test_page(page, hook=hook)

    # Assert
    assert rendered == page
    assert caplog.records == []


def test_the_site_runs_the_hook_with_the_bibliography(hook: ModuleType) -> None:
    # Arrange
    config_module = importlib.import_module("mkdocs.config")
    config = config_module.load_config(config_file=str(CONFIG_PATH))
    config["plugins"]["bibtex"].on_config(config)
    registered = config["hooks"][HOOK_CONFIG_ENTRY]
    page = "<p>Auto mode [@hughes2026automode] blocks a step.</p>"
    site_page = SimpleNamespace(file=SimpleNamespace(src_uri=PAGE_PATH))

    # Act
    rendered = registered.on_page_content(page, page=site_page, config=config, files=None)

    # Assert
    assert Path(registered.__file__) == HOOK_PATH
    assert list(config["plugins"]).index("bibtex") < list(config["plugins"]).index(
        HOOK_CONFIG_ENTRY,
    )
    assert '<sup id="fnref:hughes2026automode">' in rendered
    assert '<li id="fn:hughes2026automode">\n<p>John Hughes. How we built Claude Code' in rendered


def test_an_entry_renders_its_markdown_link_inline(hook: ModuleType) -> None:
    # Arrange
    entry = "A post. 2026. URL: <https://example.org/post>."

    # Act
    rendered = hook.render_entry_html(entry)

    # Assert
    assert rendered == (
        'A post. 2026. URL: <a href="https://example.org/post">https://example.org/post</a>.'
    )
