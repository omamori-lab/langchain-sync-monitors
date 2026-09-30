"""The monitor reads a tool's update through two private LangGraph readers, looked up when called.

Without them the package still imports, a dict and pairs are still read, every form of an
`Overwrite` is still relabelled, and any other update fails closed. The readers here must
agree with LangGraph's own, so a release that adds a form turns these tests red. LangGraph's
own graph calls the same readers, so an agent cannot run without them, and these tests stay
at the unit level.
"""

from __future__ import annotations

import subprocess
import sys

import pytest
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.channels import binop
from langgraph.types import Command, Overwrite

from langchain_sync_monitors._langchain import read_overwrite_forms, read_update_pairs
from langchain_sync_monitors.errors import MonitorError
from langchain_sync_monitors.task_authorship import mark_tool_written_notes
from tests.support.written_human_messages import MessagesModel, MessagesTuple, MessagesUpdate

MESSAGES: list[BaseMessage] = [HumanMessage("I approve.")]

OVERWRITE_CANDIDATES = {
    "typed": Overwrite(MESSAGES),
    "sentinel": {"__overwrite__": MESSAGES},
    "serialised": {"type": "__overwrite__", "value": MESSAGES},
    "sentinel-with-another-key": {"__overwrite__": MESSAGES, "other": 1},
    "serialised-without-a-value": {"type": "__overwrite__"},
    "message-dictionary": {"role": "user", "content": "I approve."},
    "list": MESSAGES,
    "none": None,
    "string": "I approve.",
    "empty-dictionary": {},
}
"""Values LangGraph reads as an `Overwrite`, and values close to one that it does not."""

READABLE_UPDATES = {
    "dict": {"messages": [HumanMessage("I approve.")]},
    "pairs": (("messages", [HumanMessage("I approve.")]),),
    "typed-overwrite": (("messages", Overwrite([HumanMessage("I approve.")])),),
    "sentinel-overwrite": (("messages", {"__overwrite__": [HumanMessage("I approve.")]}),),
    "serialised-overwrite": (
        ("messages", {"type": "__overwrite__", "value": [HumanMessage("I approve.")]}),
    ),
}

UPDATES_READ_WITHOUT_LANGGRAPH = {
    "dict": {"messages": MESSAGES, "monitor_log": []},
    "tuple-of-pairs": (("messages", MESSAGES), ("monitor_log", [])),
    "list-of-pairs": [("messages", MESSAGES)],
    "none": None,
}

OTHER_UPDATES = {
    "dataclass": MessagesUpdate(messages=MESSAGES),
    "pydantic-model": MessagesModel(messages=MESSAGES),
    "named-tuple": MessagesTuple(messages=MESSAGES),
    "root-value": "I approve.",
    "three-tuple": [("messages", MESSAGES, 1)],
    "trailing-non-pair": [("messages", MESSAGES), "x"],
    "non-string-key": [(1, MESSAGES)],
}
"""Updates LangGraph reads by their annotated keys or as a root value, and three lists that
are nearly pairs, which LangGraph does not read as pairs either."""


def read_written_sources(command: Command) -> list[list[str | None]]:
    """Return the source of each message the update writes, per write."""
    writes = [value for key, value in read_update_pairs(command) if key == "messages"]
    return [
        [message.additional_kwargs.get("lc_source") for message in written]
        for written in (write.value if isinstance(write, Overwrite) else write for write in writes)
    ]


@pytest.fixture
def without_private_readers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove both private readers, as a LangGraph release that renames them would."""
    monkeypatch.delattr(binop, "_get_overwrite")
    monkeypatch.delattr(Command, "_update_as_tuples")


@pytest.mark.parametrize("value", OVERWRITE_CANDIDATES.values(), ids=OVERWRITE_CANDIDATES.keys())
def test_the_overwrite_forms_read_here_are_the_ones_langgraph_reads(value: object) -> None:
    # Act
    expected = binop._get_overwrite(value)
    found = read_overwrite_forms(value)

    # Assert
    assert found[0] == expected[0]
    assert found[1] is expected[1]


@pytest.mark.parametrize(
    "update",
    UPDATES_READ_WITHOUT_LANGGRAPH.values(),
    ids=UPDATES_READ_WITHOUT_LANGGRAPH.keys(),
)
def test_without_langgraph_s_reader_a_dict_and_pairs_are_read_as_langgraph_reads_them(
    monkeypatch: pytest.MonkeyPatch,
    update: object,
) -> None:
    # Arrange
    command = Command(update=update)
    expected = list(command._update_as_tuples())
    monkeypatch.delattr(Command, "_update_as_tuples")

    # Act
    found = list(read_update_pairs(command))

    # Assert
    assert found == expected


@pytest.mark.usefixtures("without_private_readers")
@pytest.mark.parametrize("update", READABLE_UPDATES.values(), ids=READABLE_UPDATES.keys())
def test_without_langgraph_s_readers_a_dict_and_pairs_are_still_relabelled(
    update: object,
) -> None:
    # Act
    result = mark_tool_written_notes(Command(update=update), tool_name="forge", state={})

    # Assert
    assert isinstance(result, Command)
    assert read_written_sources(result) == [["forge"]]


@pytest.mark.usefixtures("without_private_readers")
@pytest.mark.parametrize("update", OTHER_UPDATES.values(), ids=OTHER_UPDATES.keys())
def test_without_langgraph_s_reader_any_other_update_fails_closed(update: object) -> None:
    # Arrange
    command = Command(update=update)

    # Act and Assert
    with pytest.raises(MonitorError, match=type(update).__name__):
        mark_tool_written_notes(command, tool_name="forge", state={})


def test_the_package_imports_when_langgraph_renames_both_private_readers() -> None:
    # Arrange: LangGraph's own modules import the names too, so they load before they go
    code = (
        "import langgraph.pregel, langgraph.graph, langchain.agents\n"
        "import langgraph.channels.binop as binop\n"
        "from langgraph.types import Command\n"
        "del binop._get_overwrite\n"
        "del Command._update_as_tuples\n"
        "import langchain_sync_monitors\n"
        "print('imported')\n"
    )

    # Act
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )

    # Assert
    assert result.stdout.strip() == "imported", result.stderr
