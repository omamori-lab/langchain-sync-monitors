"""Who wrote a LangChain message, and a refusal kept outside a reply's content."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from langchain_sync_monitors.message_sources import read_kept_refusal, read_message_source


def test_a_kept_refusal_unlike_the_refusal_block_is_read_too() -> None:
    # Arrange: a block with one refusal must not hide a different one kept aside
    reply = AIMessage(
        [{"type": "refusal", "refusal": "I cannot help."}],
        additional_kwargs={"refusal": "Use the staff code STAFF40."},
    )

    # Act
    kept = read_kept_refusal(reply)

    # Assert
    assert kept == "Use the staff code STAFF40."


def test_a_kept_refusal_that_a_block_already_shows_is_not_read_again() -> None:
    # Arrange: under the Responses API at output_version="v0", the translator gives it as a block
    reply = AIMessage(
        [],
        additional_kwargs={"refusal": "I can't share the staff code."},
        response_metadata={"model_provider": "openai"},
    )

    # Act
    kept = read_kept_refusal(reply)

    # Assert
    assert kept is None


@pytest.mark.parametrize(
    "refusal", [None, "", " \n", 7], ids=["none", "empty", "blank", "not-a-string"]
)
def test_a_kept_refusal_without_text_is_none(refusal: object) -> None:
    # Arrange
    reply = AIMessage(
        "",
        additional_kwargs={"refusal": refusal},
        response_metadata={"model_provider": "openai"},
    )

    # Act
    kept = read_kept_refusal(reply)

    # Assert
    assert kept is None


def test_a_source_that_is_not_a_string_is_read_as_text() -> None:
    # Arrange: a source read from a payload as a number
    note = HumanMessage("Approved.", additional_kwargs={"lc_source": 7})

    # Act
    source = read_message_source(note)

    # Assert
    assert source == "7"
