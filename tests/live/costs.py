"""What a live run spends, read from OpenRouter's own per-response cost, with a hard cap.

Every chat model gets a `CostTracker` callback that reads the `cost` field
ChatOpenRouter copies into each reply's `response_metadata`, and the Decisions
API client gets response hooks that read `usage.cost`. Both add to one
`CostLedger`, which raises `BudgetExceededError` once the spend reaches its
cap, so a run stops before the budget instead of after it.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, TypedDict
from uuid import UUID

import httpx
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult


class CostRole(StrEnum):
    """Who made a model call: the untrusted agent, the trusted model or the monitor."""

    AGENT = "agent"
    TRUSTED = "trusted"
    MONITOR = "monitor"


class BudgetExceededError(RuntimeError):
    """The spend reached the cap; the run stops before it spends more."""


class CostSnapshot(TypedDict):
    """The spend so far, in US dollars, per role and in total, and the calls counted."""

    agent: float
    trusted: float
    monitor: float
    total: float
    calls: int
    calls_without_cost: int


class TokenUsage(TypedDict):
    """The input and output tokens a role's calls used."""

    input: int
    output: int


@dataclass
class CostLedger:
    """The running spend of every model call of one run, shared by all its trackers.

    `cap` is in US dollars. `providers` names the upstream provider that served
    each role's calls, so a pinned provider can be checked.
    """

    cap: float
    costs: dict[CostRole, float] = field(default_factory=lambda: dict.fromkeys(CostRole, 0.0))
    providers: dict[CostRole, set[str]] = field(default_factory=dict)
    tokens: dict[CostRole, TokenUsage] = field(default_factory=dict)
    calls: int = 0
    calls_without_cost: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, *, role: CostRole, cost: float | None, provider: str | None) -> None:
        """Add one call's cost, then raise `BudgetExceededError` when the cap is reached."""
        with self.lock:
            self.calls += 1
            if cost is None:
                self.calls_without_cost += 1
            self.costs[role] += cost or 0.0
            if provider:
                self.providers.setdefault(role, set()).add(provider)
            total = sum(self.costs.values())
        if total >= self.cap:
            message = f"spent ${total:.4f}, at or above the cap of ${self.cap:.4f}"
            raise BudgetExceededError(message)

    def add_tokens(self, *, role: CostRole, input_tokens: int, output_tokens: int) -> None:
        """Add one call's token counts to the role's total."""
        with self.lock:
            usage = self.tokens.setdefault(role, {"input": 0, "output": 0})
            usage["input"] += input_tokens
            usage["output"] += output_tokens

    def list_tokens(self) -> dict[str, TokenUsage]:
        """Return the tokens each role used, by role name."""
        with self.lock:
            return {role.value: {**usage} for role, usage in self.tokens.items()}

    def list_providers(self) -> dict[str, list[str]]:
        """Return the providers that served each role, by role name."""
        with self.lock:
            return {role.value: sorted(names) for role, names in self.providers.items()}

    def take_snapshot(self) -> CostSnapshot:
        """Return the spend so far."""
        with self.lock:
            return {
                "agent": self.costs[CostRole.AGENT],
                "trusted": self.costs[CostRole.TRUSTED],
                "monitor": self.costs[CostRole.MONITOR],
                "total": sum(self.costs.values()),
                "calls": self.calls,
                "calls_without_cost": self.calls_without_cost,
            }


def read_float(mapping: Mapping[str, Any], key: str) -> float | None:
    """Return a number from a mapping as a float, or `None` when it is absent."""
    value = mapping.get(key)
    return float(value) if isinstance(value, int | float) else None


class CostTracker(BaseCallbackHandler):
    """Adds the cost of every reply of one chat model to the ledger.

    `raise_error` lets `BudgetExceededError` stop the run, and `run_inline`
    keeps the handler on the caller's thread under `ainvoke()`.
    """

    raise_error = True
    run_inline = True

    def __init__(self, *, role: CostRole, ledger: CostLedger) -> None:
        """Track the calls of a model playing `role`."""
        self.role = role
        self.ledger = ledger

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Read the cost, the tokens and the provider of each generation in the reply."""
        for generations in response.generations:
            for generation in generations:
                if not isinstance(generation, ChatGeneration):
                    continue
                self.record_tokens(generation)
                metadata = generation.message.response_metadata
                provider = metadata.get("provider")
                self.ledger.add(
                    role=self.role,
                    cost=read_float(metadata, "cost"),
                    provider=provider if isinstance(provider, str) else None,
                )

    def record_tokens(self, generation: ChatGeneration) -> None:
        """Add the generation's token counts, when the provider reported them."""
        message = generation.message
        usage = message.usage_metadata if isinstance(message, AIMessage) else None
        if usage:
            self.ledger.add_tokens(
                role=self.role,
                input_tokens=usage["input_tokens"],
                output_tokens=usage["output_tokens"],
            )


def record_decisions_cost(response: httpx.Response, *, ledger: CostLedger) -> None:
    """Add the cost in a Decisions API response's `usage` to the ledger."""
    try:
        body = response.json()
    except ValueError:
        body = {}
    usage = body.get("usage") if isinstance(body, dict) else None
    provider = body.get("provider") if isinstance(body, dict) else None
    ledger.add(
        role=CostRole.MONITOR,
        cost=read_float(usage, "cost") if isinstance(usage, dict) else None,
        provider=provider if isinstance(provider, str) else None,
    )


DECISIONS_TIMEOUT_SECONDS = 30.0


def build_decisions_client(*, ledger: CostLedger) -> httpx.Client:
    """Return an HTTP client whose response hook records each Decisions API request's cost."""

    def record_cost(response: httpx.Response) -> None:
        response.read()
        record_decisions_cost(response, ledger=ledger)

    return httpx.Client(
        timeout=DECISIONS_TIMEOUT_SECONDS,
        event_hooks={"response": [record_cost]},
    )


def build_async_decisions_client(*, ledger: CostLedger) -> httpx.AsyncClient:
    """Return the async counterpart of `build_decisions_client`.

    `OpenRouterDecisionModel` opens a fresh async client per request unless it
    is given one, so without this client the hook would never run under
    `ainvoke()`. Build it inside the event loop that uses it.
    """

    async def record_cost(response: httpx.Response) -> None:
        await response.aread()
        record_decisions_cost(response, ledger=ledger)

    return httpx.AsyncClient(
        timeout=DECISIONS_TIMEOUT_SECONDS,
        event_hooks={"response": [record_cost]},
    )
