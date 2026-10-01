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
def hook() -> ModuleType:
    """Load the hook from its file, as MkDocs does."""
    if os.environ.get("REQUIRE_DOCS_GROUP") != "1":
        pytest.importorskip("mkdocs_bibtex")
    specification = importlib.util.spec_from_file_location("citations_hook", HOOK_PATH)
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
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
