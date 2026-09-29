"""The boundary helper that keeps the protocol's model calls out of the message stream."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import var_child_runnable_config
from langgraph.constants import TAG_NOSTREAM

from langchain_sync_monitors._langchain import hide_model_calls_from_message_stream

NODE_CONFIG = RunnableConfig(tags=["graph:step:1"], metadata={"langgraph_node": "model"})


@pytest.fixture
def node_config() -> Iterator[RunnableConfig]:
    """Stand in for the config LangGraph sets while a model node runs."""
    token = var_child_runnable_config.set(NODE_CONFIG)
    yield NODE_CONFIG
    var_child_runnable_config.reset(token)


def test_the_block_adds_the_no_stream_tag_and_keeps_the_rest(node_config: RunnableConfig) -> None:
    # Act
    with hide_model_calls_from_message_stream():
        inside = var_child_runnable_config.get()

    # Assert
    assert inside == {**node_config, "tags": ["graph:step:1", TAG_NOSTREAM]}


def test_the_node_config_comes_back_even_when_the_block_raises(
    node_config: RunnableConfig,
) -> None:
    # Act
    with pytest.raises(RuntimeError), hide_model_calls_from_message_stream():
        raise RuntimeError

    # Assert
    assert var_child_runnable_config.get() == node_config


def test_outside_a_graph_the_block_holds_only_the_tag() -> None:
    # Act
    with hide_model_calls_from_message_stream():
        inside = var_child_runnable_config.get()

    # Assert
    assert inside == RunnableConfig(tags=[TAG_NOSTREAM])
    assert var_child_runnable_config.get() is None
