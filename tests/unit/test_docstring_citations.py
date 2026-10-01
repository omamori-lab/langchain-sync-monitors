"""The docs hook turns the citations in rendered docstrings into footnotes.

mkdocs-bibtex rewrites the citations of a page's Markdown only, so the API
reference's docstrings reach the page with their citations as literal text.
``docs/hooks/citations.py`` gives them the footnote markup of the other pages.
The hook needs the docs dependency group; without it these tests are skipped,
and ``scripts/check.sh`` sets ``REQUIRE_DOCS_GROUP=1``, which makes a missing
module fail them instead.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
HOOK_PATH = REPOSITORY_ROOT / "docs" / "hooks" / "citations.py"
CONFIG_PATH = REPOSITORY_ROOT / "mkdocs.yml"
HOOK_CONFIG_ENTRY = "docs/hooks/citations.py"
HOOK_LOGGER = "mkdocs.hooks.citations"
KNOWN_KEYS = frozenset({"first2026", "second2026"})


def render_test_entry(key: str) -> str:
    """Return a stand-in for a formatted bibliography entry."""
    return f"Entry for {key}."


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
    rendered = hook.render_citations(page, known_keys=KNOWN_KEYS, render_entry=render_test_entry)

    # Assert
    assert "[@" not in rendered
    assert (
        '<sup id="fnref:first2026"><a class="footnote-ref" href="#fn:first2026">1</a></sup>'
        in rendered
    )
    assert '<li id="fn:first2026">\n<p>Entry for first2026.&#160;' in rendered
    assert rendered.endswith("</ol>\n</div>")


def test_a_page_without_citations_is_unchanged(hook: ModuleType) -> None:
    # Arrange
    page = '<p>No sources here.</p>\n<div class="footnote"><ol></ol></div>'

    # Act
    rendered = hook.render_citations(page, known_keys=KNOWN_KEYS, render_entry=render_test_entry)

    # Assert
    assert rendered == page


def test_citations_inside_code_stay_as_written(hook: ModuleType) -> None:
    # Arrange
    page = "<p><code>[@first2026]</code></p>\n<pre><code>cite [@second2026]\n</code></pre>"

    # Act
    rendered = hook.render_citations(page, known_keys=KNOWN_KEYS, render_entry=render_test_entry)

    # Assert
    assert rendered == page


@pytest.mark.parametrize(
    "page",
    [
        '<p title="See [@first2026]">Text</p>',
        '<p title="a > [@first2026]">Text</p>',
        "<p><a href='#' data-cite='[@first2026]'>Text</a></p>",
    ],
    ids=["double-quoted", "after-a-quoted-angle-bracket", "single-quoted"],
)
def test_a_citation_inside_an_attribute_stays_as_written(hook: ModuleType, page: str) -> None:
    # Act
    rendered = hook.render_citations(page, known_keys=KNOWN_KEYS, render_entry=render_test_entry)

    # Assert
    assert rendered == page


def test_a_citation_inside_code_stays_while_one_in_its_text_renders(hook: ModuleType) -> None:
    # Arrange
    page = "<p>Write <code>[@first2026]</code> to cite, as [@second2026] does.</p>"

    # Act
    rendered = hook.render_citations(page, known_keys=KNOWN_KEYS, render_entry=render_test_entry)

    # Assert
    assert rendered.startswith(
        "<p>Write <code>[@first2026]</code> to cite, as "
        '<sup id="fnref:second2026"><a class="footnote-ref" href="#fn:second2026">1</a></sup>'
        " does.</p>\n",
    )
    assert '<li id="fn:second2026">' in rendered
    assert "fn:first2026" not in rendered


@pytest.mark.parametrize(
    "page",
    [
        '<script>const cite = "[@first2026]";</script>',
        '<style>p::after { content: "[@first2026]"; }</style>',
        "<textarea>[@first2026]</textarea>",
        '<svg><title>A figure after [@first2026]</title><rect width="1" /></svg>',
    ],
    ids=["script", "style", "textarea", "title"],
)
def test_a_citation_inside_an_element_html_reads_as_plain_text_stays_as_written(
    hook: ModuleType,
    page: str,
) -> None:
    # Act
    rendered = hook.render_citations(page, known_keys=KNOWN_KEYS, render_entry=render_test_entry)

    # Assert
    assert rendered == page


def test_an_end_tag_nothing_opened_leaves_the_text_after_it_cited(hook: ModuleType) -> None:
    # Arrange
    page = "<p>A stray </code> end tag, then [@first2026].</p>"

    # Act
    rendered = hook.render_citations(page, known_keys=KNOWN_KEYS, render_entry=render_test_entry)

    # Assert
    assert rendered.startswith(
        "<p>A stray </code> end tag, then "
        '<sup id="fnref:first2026"><a class="footnote-ref" href="#fn:first2026">1</a></sup>'
        ".</p>\n",
    )


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
    rendered = hook.render_citations(page, known_keys=KNOWN_KEYS, render_entry=render_test_entry)

    # Assert
    assert rendered == (
        f"{before}"
        '<sup id="fnref:first2026"><a class="footnote-ref" href="#fn:first2026">1</a></sup>'
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
    rendered = hook.render_citations(page, known_keys=KNOWN_KEYS, render_entry=render_test_entry)

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
        rendered = hook.render_citations(
            page,
            known_keys=KNOWN_KEYS,
            render_entry=render_test_entry,
        )

    # Assert
    assert rendered == page
    assert [record.name for record in caplog.records] == [HOOK_LOGGER]
    assert "missing2026" in caplog.records[0].getMessage()


def test_a_citation_in_link_text_stays_as_written_and_warns(
    hook: ModuleType,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange
    page = '<a href="#x">see [@first2026]</a>'

    # Act
    with caplog.at_level(logging.WARNING, logger=HOOK_LOGGER):
        rendered = hook.render_citations(
            page,
            known_keys=KNOWN_KEYS,
            render_entry=render_test_entry,
        )

    # Assert
    assert rendered == page
    assert [record.name for record in caplog.records] == [HOOK_LOGGER]
    assert "[@first2026] inside a link or a button" in caplog.records[0].getMessage()


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
        "<p><button>see [@first2026]</button></p>",
    ],
    ids=[
        "inside-an-inline-element",
        "without-an-href",
        "cross-reference",
        "after-an-unclosed-link",
        "after-a-stray-end-tag",
        "after-a-self-closing-link",
        "after-a-self-closing-cross-reference",
        "inside-a-button",
    ],
)
def test_a_citation_where_no_link_may_go_stays_as_written_and_warns(
    hook: ModuleType,
    caplog: pytest.LogCaptureFixture,
    page: str,
) -> None:
    # Act
    with caplog.at_level(logging.WARNING, logger=HOOK_LOGGER):
        rendered = hook.render_citations(
            page,
            known_keys=KNOWN_KEYS,
            render_entry=render_test_entry,
        )

    # Assert
    assert rendered == page
    assert [record.name for record in caplog.records] == [HOOK_LOGGER]


@pytest.mark.parametrize(
    "before",
    [
        '<p><a href="#x">see</a> ',
        '<p><autoref identifier="x" optional hover>see</autoref> ',
        "<p>A stray </a> end tag, then ",
        '<p><code><a href="#x">Type</a></code> ',
        '<p><a id="x"/> see</a> ',
        "<p><button>Copy</button> ",
        "<p><br/> ",
        "<p><code/> ",
    ],
    ids=[
        "a-link",
        "a-cross-reference",
        "a-stray-end-tag",
        "a-link-inside-code",
        "a-self-closing-link-closed-later",
        "a-button",
        "a-self-closing-void-element",
        "a-self-closing-verbatim-element",
    ],
)
def test_a_citation_where_a_link_may_go_renders_without_a_warning(
    hook: ModuleType,
    caplog: pytest.LogCaptureFixture,
    before: str,
) -> None:
    # Arrange
    page = f"{before}[@first2026].</p>"

    # Act
    with caplog.at_level(logging.WARNING, logger=HOOK_LOGGER):
        rendered = hook.render_citations(
            page,
            known_keys=KNOWN_KEYS,
            render_entry=render_test_entry,
        )

    # Assert
    assert rendered.startswith(
        f"{before}"
        '<sup id="fnref:first2026"><a class="footnote-ref" href="#fn:first2026">1</a></sup>'
        ".</p>\n",
    )
    assert caplog.records == []


def test_a_citation_outside_a_link_renders_while_one_inside_stays(
    hook: ModuleType,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange
    page = '<p><a href="#x">see [@second2026]</a> and [@first2026].</p>'

    # Act
    with caplog.at_level(logging.WARNING, logger=HOOK_LOGGER):
        rendered = hook.render_citations(
            page,
            known_keys=KNOWN_KEYS,
            render_entry=render_test_entry,
        )

    # Assert
    assert rendered.startswith(
        '<p><a href="#x">see [@second2026]</a> and '
        '<sup id="fnref:first2026"><a class="footnote-ref" href="#fn:first2026">1</a></sup>'
        ".</p>\n",
    )
    assert "fn:second2026" not in rendered
    assert [record.name for record in caplog.records] == [HOOK_LOGGER]


def test_an_unknown_key_in_link_text_warns_once_about_the_link(
    hook: ModuleType,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange
    page = '<p><a href="#x">see [@missing2026]</a></p>'

    # Act
    with caplog.at_level(logging.WARNING, logger=HOOK_LOGGER):
        rendered = hook.render_citations(
            page,
            known_keys=KNOWN_KEYS,
            render_entry=render_test_entry,
        )

    # Assert
    assert rendered == page
    assert [record.name for record in caplog.records] == [HOOK_LOGGER]
    assert "inside a link or a button" in caplog.records[0].getMessage()


def test_a_page_with_footnotes_of_its_own_warns(
    hook: ModuleType,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange
    page = '<p>A [@first2026] source.</p>\n<div class="footnote"><ol></ol></div>'

    # Act
    with caplog.at_level(logging.WARNING, logger=HOOK_LOGGER):
        hook.render_citations(page, known_keys=KNOWN_KEYS, render_entry=render_test_entry)

    # Assert
    assert [record.name for record in caplog.records] == [HOOK_LOGGER]


def test_the_site_runs_the_hook_with_the_bibliography(hook: ModuleType) -> None:
    # Arrange
    config_module = importlib.import_module("mkdocs.config")
    config = config_module.load_config(config_file=str(CONFIG_PATH))
    config["plugins"]["bibtex"].on_config(config)
    registered = config["hooks"][HOOK_CONFIG_ENTRY]
    page = "<p>Auto mode [@hughes2026automode] blocks a step.</p>"

    # Act
    rendered = registered.on_page_content(page, page=None, config=config, files=None)

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
