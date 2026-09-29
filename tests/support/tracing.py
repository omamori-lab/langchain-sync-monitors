"""A LangChain callback handler that records every run it hears of, as any tracer would.

LangSmith's tracer, Langfuse's handler and `astream_events` all build their
trees from the same callbacks, so a test that passes this handler through
`config={"callbacks": [...]}` sees what they see. `collect_runs()` would not
do: LangGraph reuses a node's callback manager without configuring a new one,
so the collector it registers never reaches the node's runs.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

MONITOR_SPAN_PREFIX = "monitor "


@dataclass(kw_only=True)
class RecordedRun:
    """One run as a tracer receives it: where it sits, what it carried and how it ended."""

    run_id: UUID
    parent_run_id: UUID | None
    name: str
    run_type: str
    inputs: Any
    tags: list[str]
    metadata: dict[str, Any]
    outputs: Any = None
    error: BaseException | None = None
    ended: bool = False
    children: list[RecordedRun] = field(default_factory=list)

    @property
    def is_monitor_span(self) -> bool:
        return self.name.startswith(MONITOR_SPAN_PREFIX)

    def find_children(self, name: str) -> list[RecordedRun]:
        return [child for child in self.children if child.name == name]

    def read_child_names(self) -> list[str]:
        return [child.name for child in self.children]


def read_run_name(serialized: dict[str, Any] | None, keywords: dict[str, Any]) -> str:
    """Name a run as LangSmith and Langfuse do: the name given, else the serialised class."""
    if keywords.get("name"):
        return str(keywords["name"])
    serialized = serialized or {}
    return str(serialized.get("name") or serialized.get("id", ["<unknown>"])[-1])


class RecordingTracer(BaseCallbackHandler):
    """Records the start, end and parent of every chain, model and tool run.

    It runs inline, on the caller's thread or event loop, so runs are
    recorded in the order they start, as a tracer that keeps order would.
    """

    run_inline = True

    def __init__(self) -> None:
        self.runs: dict[UUID, RecordedRun] = {}
        self.order: list[UUID] = []
        self.lock = threading.Lock()

    def remember_start(
        self,
        *,
        run_id: UUID,
        parent_run_id: UUID | None,
        name: str,
        run_type: str,
        inputs: Any,
        keywords: dict[str, Any],
    ) -> None:
        run = RecordedRun(
            run_id=run_id,
            parent_run_id=parent_run_id,
            name=name,
            run_type=run_type,
            inputs=inputs,
            tags=list(keywords.get("tags") or []),
            metadata=dict(keywords.get("metadata") or {}),
        )
        with self.lock:
            self.runs[run_id] = run
            self.order.append(run_id)
            parent = self.runs.get(parent_run_id) if parent_run_id else None
            if parent is not None:
                parent.children.append(run)

    def remember_end(
        self,
        run_id: UUID,
        *,
        outputs: Any = None,
        error: BaseException | None = None,
        inputs: Any = None,
    ) -> None:
        with self.lock:
            run = self.runs[run_id]
            run.outputs = outputs
            run.error = error
            run.ended = True
            if inputs is not None:
                run.inputs = inputs

    def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self.remember_start(
            run_id=run_id,
            parent_run_id=parent_run_id,
            name=read_run_name(serialized, kwargs),
            run_type=kwargs.get("run_type") or "chain",
            inputs=inputs,
            keywords=kwargs,
        )

    def on_chain_end(self, outputs: Any, *, run_id: UUID, **kwargs: Any) -> None:
        self.remember_end(run_id, outputs=outputs, inputs=kwargs.get("inputs"))

    def on_chain_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self.remember_end(run_id, error=error, inputs=kwargs.get("inputs"))

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self.remember_start(
            run_id=run_id,
            parent_run_id=parent_run_id,
            name=read_run_name(serialized, kwargs),
            run_type="chat_model",
            inputs=messages,
            keywords=kwargs,
        )

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self.remember_start(
            run_id=run_id,
            parent_run_id=parent_run_id,
            name=read_run_name(serialized, kwargs),
            run_type="llm",
            inputs=prompts,
            keywords=kwargs,
        )

    def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        self.remember_end(run_id, outputs=response)

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self.remember_end(run_id, error=error)

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self.remember_start(
            run_id=run_id,
            parent_run_id=parent_run_id,
            name=read_run_name(serialized, kwargs),
            run_type="tool",
            inputs=input_str,
            keywords=kwargs,
        )

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        self.remember_end(run_id, outputs=output)

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self.remember_end(run_id, error=error)

    def read_runs(self) -> list[RecordedRun]:
        """Return every run in the order it started."""
        return [self.runs[run_id] for run_id in self.order]

    def find_runs(self, name: str) -> list[RecordedRun]:
        """Return every run with this name, in the order they started."""
        return [run for run in self.read_runs() if run.name == name]

    def find_monitor_spans(self) -> list[RecordedRun]:
        """Return every span the monitor opened, in the order they started."""
        return [run for run in self.read_runs() if run.is_monitor_span]

    def find_parent(self, run: RecordedRun) -> RecordedRun:
        """Return the run's parent, which the tracer must have heard of."""
        assert run.parent_run_id is not None, f"{run.name} has no parent"
        return self.runs[run.parent_run_id]

    def find_ancestor_names(self, run: RecordedRun) -> list[str]:
        """Return the names of the run's ancestors, nearest first."""
        names: list[str] = []
        while run.parent_run_id is not None and run.parent_run_id in self.runs:
            run = self.runs[run.parent_run_id]
            names.append(run.name)
        return names

    def find_unknown_parents(self) -> list[str]:
        """Return the runs whose parent the tracer never saw start, which Langfuse cannot nest."""
        return [
            run.name
            for run in self.read_runs()
            if run.parent_run_id is not None and run.parent_run_id not in self.runs
        ]

    def find_open_runs(self, *, names: Sequence[str] | None = None) -> list[str]:
        """Return the runs that never ended, among those with these names if given."""
        return [
            run.name
            for run in self.read_runs()
            if not run.ended and (names is None or run.name in names)
        ]

    def find_roots(self) -> list[RecordedRun]:
        return [run for run in self.read_runs() if run.parent_run_id is None]

    def render_tree(self) -> str:
        """Render every run as an indented tree, with the monitor spans' tags and outcome."""
        lines: list[str] = []
        for root in self.find_roots():
            render_run(root, depth=0, lines=lines)
        return "\n".join(lines)


def render_run(run: RecordedRun, *, depth: int, lines: list[str]) -> None:
    status = "ok" if run.ended and run.error is None else "open"
    if run.error is not None:
        status = f"error {type(run.error).__name__}"
    detail = f" tags={run.tags}" if run.is_monitor_span else ""
    lines.append(f"{'  ' * depth}{run.name} [{run.run_type}] {status}{detail}")
    for child in run.children:
        render_run(child, depth=depth + 1, lines=lines)
