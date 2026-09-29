"""Citations in the code and docs match the bibliography in both directions.

Every ``[@key]`` (or ``[@first; @second]``) in a docstring, comment, docs page
or plan must name an entry in ``docs/references.bib``, and every entry there
must be cited somewhere, so the bibliography stays the single, complete record
of where the ideas come from. The plans and their research notes under
``docs/plans/`` count on both sides: a key cited there must exist, and an
entry cited only there counts as cited.
"""

from __future__ import annotations

import re
from pathlib import Path

import bibtexparser
import pytest
from bibtexparser.bparser import BibTexParser

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BIBLIOGRAPHY_PATH = REPOSITORY_ROOT / "docs" / "references.bib"
PLANS_PATH = REPOSITORY_ROOT / "docs" / "plans"
CITING_DIRECTORIES = ("src", "docs")
CITING_SUFFIXES = frozenset({".py", ".md"})
CITATION_GROUP_PATTERN = re.compile(r"\[(@[^\]\n]+)\]")
SPLIT_CITATION_PATTERN = re.compile(r"\[@[^\]\n]*\n")
CITATION_KEY_PATTERN = re.compile(r"@([A-Za-z0-9_:-]+)")


def read_bibliography_keys() -> set[str]:
    """Return the keys of every entry in the bibliography, including non-standard types."""
    parser = BibTexParser(common_strings=True, ignore_nonstandard_types=False)
    with BIBLIOGRAPHY_PATH.open(encoding="utf-8") as bibliography:
        entries: list[dict[str, str]] = bibtexparser.load(bibliography, parser=parser).entries
    return {entry["ID"] for entry in entries}


def is_citing_file(path: Path) -> bool:
    """Tell whether a file is a source, docs page or plan note that may carry citations."""
    return path.suffix in CITING_SUFFIXES


def collect_citations() -> dict[str, set[Path]]:
    """Map each cited key to the files that cite it."""
    citations: dict[str, set[Path]] = {}
    for directory in CITING_DIRECTORIES:
        for path in (REPOSITORY_ROOT / directory).rglob("*"):
            if not is_citing_file(path):
                continue
            for group in CITATION_GROUP_PATTERN.findall(path.read_text(encoding="utf-8")):
                for key in CITATION_KEY_PATTERN.findall(group):
                    citations.setdefault(key, set()).add(path.relative_to(REPOSITORY_ROOT))
    return citations


def test_every_citation_names_a_bibliography_entry() -> None:
    # Arrange
    bibliography_keys = read_bibliography_keys()

    # Act
    unknown = {
        key: sorted(str(path) for path in paths)
        for key, paths in collect_citations().items()
        if key not in bibliography_keys
    }

    # Assert
    assert unknown == {}, f"Cited keys missing from docs/references.bib: {unknown}"


def test_every_bibliography_entry_is_cited() -> None:
    # Arrange
    cited_keys = set(collect_citations())

    # Act
    uncited = sorted(read_bibliography_keys() - cited_keys)

    # Assert
    assert uncited == [], f"Entries in docs/references.bib that nothing cites: {uncited}"


@pytest.mark.skipif(not PLANS_PATH.is_dir(), reason="the sdist leaves out docs/plans")
def test_plan_notes_are_read_for_citations() -> None:
    # Arrange
    plans_directory = PLANS_PATH.relative_to(REPOSITORY_ROOT)

    # Act
    citing_paths = {path for paths in collect_citations().values() for path in paths}

    # Assert
    assert any(plans_directory in path.parents for path in citing_paths), (
        "No citation was read from docs/plans/, so the plan notes are not checked."
    )


def test_no_citation_group_in_the_docs_is_split_across_lines() -> None:
    # Arrange: the docs plugin reads a citation group only on one line, and
    # prints a split group as raw text.
    docs_pages = [path for path in (REPOSITORY_ROOT / "docs").rglob("*.md") if is_citing_file(path)]

    # Act
    split = sorted(
        str(path.relative_to(REPOSITORY_ROOT))
        for path in docs_pages
        if SPLIT_CITATION_PATTERN.search(path.read_text(encoding="utf-8"))
    )

    # Assert
    assert split == [], f"Citation groups split across lines, which render as raw text: {split}"
