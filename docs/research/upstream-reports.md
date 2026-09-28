# Upstream reports: drafts

This page holds ready-to-file drafts of the upstream bug reports behind issue
#27, one per confirmed bug in `docs/explanation/langchain-findings.md`. It is a
working note: **nothing here has been filed**, and each draft needs the
maintainer's approval before it is posted.

Every reproduction below was run offline on 2026-09-28 with the versions in the
environment block, and printed the "actual" result shown. None needs an API
key or network access.

Before filing, search the target tracker once more for a duplicate. Searches on
2026-09-28 found no existing report for any of these bugs. They did find two
related open issues, listed at the end, where a comment fits better than a new
issue.

Shared environment:

```text
Python 3.12.12, macOS
langchain 1.4.2
langchain-core 1.6.5
langgraph 1.2.12
deepagents 0.7.19
langchain-openrouter 0.2.9
openrouter (Python SDK) 0.10.8
langchain-typesafe 0.0.1a3
pydantic 2.13.5
httpx 0.28.1
```

Compared with the candidate list in #27, the checks added three drafts: the
SDK's missing `native_finish_reason` (split from the `provider` report,
because the cause is in a different repository), LangChain's composed model
handler, and LangGraph's reducer position. The `timeout` unit is already
reported upstream, so it is listed under related issues instead of drafted.

## OpenRouter SDK: unknown `reasoning` keys are dropped, so `enabled: false` never reaches the API

**Repository:** `OpenRouterTeam/python-sdk`

**Title:** `chat.send(reasoning=...)` silently drops `enabled`, `exclude` and `max_tokens`

**Environment:** openrouter 0.10.8, pydantic 2.13.5, httpx 0.28.1, Python 3.12.

**Description.** The reasoning guide documents `enabled`, `exclude`,
`max_tokens`, `effort` and `summary` as keys of the `reasoning` request
parameter. `ChatRequestReasoning` declares only `effort` and `summary`
(`openrouter/components/chatrequest.py:130-141`), and the SDK's `BaseModel`
keeps pydantic's default of ignoring unknown fields
(`openrouter/types/basemodel.py:9-12`). The other keys are dropped without a
warning, so `{"enabled": False}` is sent as `{}`: a request to switch reasoning
off, to exclude reasoning from the response, or to cap its tokens never reaches
OpenRouter. The same happens through `langchain-openrouter`, which passes
`reasoning` straight to `chat.send`.

**Reproduction:**

```python
import json

import httpx
from openrouter import OpenRouter

sent = []


def handler(request: httpx.Request) -> httpx.Response:
    sent.append(json.loads(request.content))
    return httpx.Response(
        200,
        json={
            "id": "gen-1",
            "object": "chat.completion",
            "created": 1,
            "model": "x/y",
            "system_fingerprint": None,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "ok"},
                }
            ],
        },
    )


client = OpenRouter(
    api_key="sk-or-offline",
    client=httpx.Client(transport=httpx.MockTransport(handler)),
)
for reasoning in ({"enabled": False}, {"exclude": True}, {"max_tokens": 2000}, {"effort": "none"}):
    client.chat.send(model="x/y", messages=[{"role": "user", "content": "hi"}], reasoning=reasoning)
    print(reasoning, "->", sent[-1].get("reasoning"))
```

**Expected:** each `reasoning` object is sent as given.

**Actual:**

```text
{'enabled': False} -> {}
{'exclude': True} -> {}
{'max_tokens': 2000} -> {}
{'effort': 'none'} -> {'effort': 'none'}
```

**Suggested fix.** The file is generated ("DO NOT EDIT"), so the fix belongs
in the OpenAPI schema the SDK is generated from: add `enabled` (boolean),
`exclude` (boolean) and `max_tokens` (integer) to the chat request's reasoning
object. Independently, rejecting unknown keys (`extra="forbid"`) on request
models, or at least warning, would turn this class of silent loss into an
error. Until then, `{"effort": "none"}` switches reasoning off and survives.

## OpenRouter SDK: `ChatChoice` drops the documented `native_finish_reason`

**Repository:** `OpenRouterTeam/python-sdk`

**Title:** `ChatChoice` has no `native_finish_reason`, so the documented field is discarded while parsing

**Environment:** openrouter 0.10.8, pydantic 2.13.5, httpx 0.28.1, Python 3.12.

**Description.** OpenRouter's API reference documents `native_finish_reason`
on each non-streaming choice ("the raw finish_reason from the provider"). The
SDK's `ChatChoice` declares `finish_reason`, `index`, `message` and `logprobs`
only (`openrouter/components/chatchoice.py:32-45`), so the field is ignored
when the response is parsed and is not available on the result. Downstream,
`langchain-openrouter` tries to copy it into `response_metadata`
(`langchain_openrouter/chat_models.py:873-876`) and never finds it.

**Reproduction:**

```python
import httpx
from openrouter import OpenRouter


def handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "gen-1",
            "object": "chat.completion",
            "created": 1,
            "model": "x/y",
            "system_fingerprint": None,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "native_finish_reason": "end_turn",
                    "message": {"role": "assistant", "content": "ok"},
                }
            ],
        },
    )


client = OpenRouter(
    api_key="sk-or-offline",
    client=httpx.Client(transport=httpx.MockTransport(handler)),
)
result = client.chat.send(model="x/y", messages=[{"role": "user", "content": "hi"}])
choice = result.choices[0].model_dump()
print(choice)
print("native_finish_reason kept:", "native_finish_reason" in choice)
```

**Expected:** `choice["native_finish_reason"] == "end_turn"`.

**Actual:**

```text
{'finish_reason': 'stop', 'index': 0, 'message': {'role': 'assistant', 'content': 'ok'}}
native_finish_reason kept: False
```

**Suggested fix.** Add `native_finish_reason: string | null` to the choice
schema the SDK is generated from, as the API reference already documents it.

## langchain-openrouter: `provider` is read from a key the SDK result never has

**Repository:** `langchain-ai/langchain` (package `langchain-openrouter`)

**Title:** `ChatOpenRouter` never populates `response_metadata["provider"]` (or `native_finish_reason`)

**Environment:** langchain-openrouter 0.2.9, openrouter 0.10.8, langchain-core
1.6.5, Python 3.12.

**Description.** `_create_chat_result` converts the SDK result with
`model_dump(by_alias=True)` and then reads `response.get("provider")`
(`langchain_openrouter/chat_models.py:830-831`, `855`, `869-870`). The SDK's
`ChatResult` has no `provider` field (`openrouter/components/chatresult.py:43-69`),
so the key is never present after the dump and the branch never runs, even
when the HTTP body carries a top-level `provider`. The same happens to
`native_finish_reason` (`chat_models.py:873-876`), which the SDK drops while
parsing (reported separately to the SDK). OpenRouter reports the serving
provider in the opt-in `openrouter_metadata` object, which `ChatResult` does
declare.

**Reproduction:**

```python
import httpx
import openrouter
from langchain_openrouter import ChatOpenRouter


def handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "gen-1",
            "object": "chat.completion",
            "created": 1,
            "model": "x/y",
            "provider": "SomeProvider",
            "system_fingerprint": "fp",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "native_finish_reason": "end_turn",
                    "message": {"role": "assistant", "content": "ok"},
                }
            ],
        },
    )


model = ChatOpenRouter(model="x/y", api_key="sk-or-offline")
model.client = openrouter.OpenRouter(
    api_key="sk-or-offline",
    client=httpx.Client(transport=httpx.MockTransport(handler)),
)
metadata = model.invoke("hi").response_metadata
print(metadata)
print(
    "provider:",
    metadata.get("provider"),
    "| native_finish_reason:",
    metadata.get("native_finish_reason"),
)
```

**Expected:** `provider: SomeProvider | native_finish_reason: end_turn`.

**Actual:**

```text
{'model_name': 'x/y', 'id': 'gen-1', 'created': 1, 'object': 'chat.completion', 'finish_reason': 'stop', 'model_provider': 'openrouter', 'system_fingerprint': 'fp'}
provider: None | native_finish_reason: None
```

**Suggested fix.** Read the provider from `openrouter_metadata` when the
caller has opted in to router metadata, and otherwise drop the dead branch, or
ask the SDK to declare `provider`. Add a unit test that feeds a full response
through the real SDK parsing, not a pre-built dictionary, so fields the SDK
drops are caught.

## langchain-openrouter: `ChatOpenRouter(n=...)` fails on every call

**Repository:** `langchain-ai/langchain` (package `langchain-openrouter`)

**Title:** `ChatOpenRouter(n=2)` raises `TypeError: Chat.send() got an unexpected keyword argument 'n'`

**Environment:** langchain-openrouter 0.2.9, openrouter 0.10.8, Python 3.12.

**Description.** `n` is a validated field (`chat_models.py:267`, `ge=1`), and
`_default_params` adds it to the request when it is above 1
(`chat_models.py:800-801`). None of the SDK's `Chat.send` overloads accepts `n`
(`openrouter/chat.py:20`, `178`, `335`), so construction succeeds and every
call fails.

**Reproduction:**

```python
import httpx
import openrouter
from langchain_openrouter import ChatOpenRouter


def handler(request: httpx.Request) -> httpx.Response:
    raise AssertionError("no request should be needed to show the failure")


model = ChatOpenRouter(model="x/y", api_key="sk-or-offline", n=3)  # accepted
model.client = openrouter.OpenRouter(
    api_key="sk-or-offline",
    client=httpx.Client(transport=httpx.MockTransport(handler)),
)
model.invoke("hi")
```

**Expected:** either three generations, or a clear error at construction that
`n` is not supported.

**Actual:**

```text
File ".../langchain_openrouter/chat_models.py", line 572, in _generate
    response = self.client.chat.send(messages=message_dicts, **params)
TypeError: Chat.send() got an unexpected keyword argument 'n'
```

**Suggested fix.** Until the SDK accepts `n`, reject `n > 1` in a model
validator with a message that says so (or remove the field), and add a test
that calls the model with `n=2`.

## langchain-core: `content_blocks` ignores OpenRouter's `reasoning_details`

**Repository:** `langchain-ai/langchain` (packages `langchain-core` and
`langchain-openrouter`)

**Title:** `AIMessage.content_blocks` drops reasoning that `ChatOpenRouter` stores only in `reasoning_details`

**Environment:** langchain-core 1.6.5, langchain-openrouter 0.2.9, Python 3.12.

**Description.** `ChatOpenRouter` keeps OpenRouter's structured reasoning in
`additional_kwargs["reasoning_details"]` and tags messages with
`model_provider="openrouter"` (`langchain_openrouter/chat_models.py:1370-1371`,
`1388`). langchain-core has no block translator for `openrouter`
(`langchain_core/messages/block_translators/`), so `content_blocks` falls back
to best-effort parsing (`langchain_core/messages/ai.py:258-303`), and its
reasoning helper reads only `additional_kwargs["reasoning_content"]`
(`langchain_core/messages/base.py:24-44`). When a reply's reasoning arrives
only as `reasoning_details`, for example a `reasoning.summary` entry from an
OpenAI reasoning model with the plain `reasoning` field empty, the standard
blocks contain no reasoning at all.

**Reproduction:**

```python
from langchain_core.messages import AIMessage

# The shape ChatOpenRouter produces when a reply carries only a reasoning summary.
message = AIMessage(
    content="ok",
    additional_kwargs={
        "reasoning_details": [
            {
                "type": "reasoning.summary",
                "summary": "I checked the request first.",
                "format": "openai-responses-v1",
                "index": 0,
            }
        ]
    },
    response_metadata={"model_provider": "openrouter"},
)
print(message.content_blocks)
```

**Expected:** a `{"type": "reasoning", "reasoning": "I checked the request first."}`
block before the text block.

**Actual:**

```text
[{'type': 'text', 'text': 'ok'}]
```

**Suggested fix.** Register an `openrouter` block translator (in
`langchain-openrouter`, or in core next to the others) that turns
`reasoning.text` entries into reasoning blocks from `text`, `reasoning.summary`
entries into reasoning blocks from `summary`, and leaves `reasoning.encrypted`
entries out of the text or carries them as provider-specific blocks.

## langchain-typesafe: `AutoModeMiddleware` never applies its default criteria, and documents a missing `threshold`

**Repository:** `langchain-ai/langchain` (package `langchain-typesafe`)

**Title:** `AutoModeMiddleware`: default `NoulCriteria` are never used, and the documented `threshold` parameter does not exist

**Environment:** langchain-typesafe 0.0.1a3, langchain 1.4.2, Python 3.12.

**Description.** Two problems in the same constructor
(`langchain_typesafe/experimental/middleware/auto_mode.py`):

- `_AutoModeConfig.criteria` defaults to the built-in risky and safe
  descriptions (`auto_mode.py:61-71`), but `__init__` defaults `criteria` to
  `None` and passes it explicitly to `model_validate` (`auto_mode.py:147`,
  `160-166`), which overrides the model default. The defaults are never sent.
  The class docstring also gives `None` its own meaning, "classify without
  outcome criteria" (`auto_mode.py:114-115`), which the default then always
  selects.
- The class docstring says "Calls below `threshold` execute normally"
  (`auto_mode.py:89-90`) and `__init__` documents a `ValidationError` for
  invalid "threshold configuration" (`auto_mode.py:157-158`), but there is no
  `threshold` parameter; the value is the constant `_PROBABILITY_THRESHOLD =
  0.5` (`auto_mode.py:39`).

**Reproduction:**

```python
import os

os.environ.setdefault("TYPESAFE_API_KEY", "offline")  # no request is made

from langchain_typesafe.experimental.middleware import AutoModeMiddleware
from langchain_typesafe.experimental.middleware.auto_mode import _AutoModeConfig

middleware = AutoModeMiddleware(tools=["send_email"])
print("criteria used:", middleware.config.criteria)
print("criteria defined:", _AutoModeConfig.model_fields["criteria"].default)

AutoModeMiddleware(tools=["send_email"], threshold=0.8)
```

**Expected:** the built-in criteria are used when none are given, and
`threshold=0.8` is accepted, as the docstring describes.

**Actual:**

```text
criteria used: None
criteria defined: true='Execution could cause harm, exceed authorization, expose sensitive data, or create an external side effect.' false='Execution is low risk, reversible, and clearly authorized by the user.'
TypeError: AutoModeMiddleware.__init__() got an unexpected keyword argument 'threshold'
```

**Suggested fix.** Use a sentinel default for `criteria` (or build the
validation dictionary without the key when it is not given), so the model
default applies and an explicit `None` keeps its documented meaning. Either add
a validated `threshold: float = 0.5` parameter, or remove the `threshold`
wording from the docstrings. Related: an open issue asks to inject the
classifier (see below).

## langchain: the composed model handler applies the commands of the wrong call

**Repository:** `langchain-ai/langchain` (package `langchain`)

**Title:** `wrap_model_call`: when an outer middleware calls `handler` several times, inner `ExtendedModelResponse` commands are taken from the last call (or from every call), not from the response returned

**Environment:** langchain 1.4.2, langchain-core 1.6.5, langgraph 1.2.12,
Python 3.12.

**Description.** `compose_two` (`langchain/agents/factory.py:313-345`, async
`404-436`) keeps one `accumulated_commands` list per outer call. Every call of
the inner handler clears it and refills it with that call's commands, and the
list is attached to whatever the outer layer returns. The middleware docs say
the handler "can be called multiple times for retry logic"
(`langchain/agents/middleware/types.py:513`, `524`); for a retry that returns
the last response this is correct. But an outer middleware that returns an
earlier response (a best-of-n, a fallback to the first attempt) commits the
last call's commands, and one that calls the handler concurrently commits the
commands of every call. With a last-value key, the concurrent case raises
`InvalidUpdateError`.

**Reproduction:**

```python
import asyncio
import itertools
import operator
from typing import Annotated

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, AgentState, ExtendedModelResponse
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.types import Command
from typing_extensions import NotRequired

counter = itertools.count(1)


class NumberedModel(GenericFakeChatModel):
    """Answers "sample N", with N counting up on every call."""

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        number = next(counter)
        message = AIMessage(content=f"sample {number}", id=f"ai-{number}")
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        await asyncio.sleep(0)
        return self._generate(messages)


class InnerState(AgentState):
    seen: NotRequired[Annotated[list[str], operator.add]]


class Inner(AgentMiddleware):
    """Records which response it saw, through a Command."""

    state_schema = InnerState

    def wrap_model_call(self, request, handler):
        response = handler(request)
        update = {"seen": [response.result[-1].content]}
        return ExtendedModelResponse(model_response=response, command=Command(update=update))

    async def awrap_model_call(self, request, handler):
        response = await handler(request)
        update = {"seen": [response.result[-1].content]}
        return ExtendedModelResponse(model_response=response, command=Command(update=update))


class Outer(AgentMiddleware):
    """Calls the handler three times and returns the FIRST response."""

    def __init__(self, *, concurrent: bool) -> None:
        super().__init__()
        self.concurrent = concurrent

    def wrap_model_call(self, request, handler):
        responses = [handler(request) for _ in range(3)]
        return responses[0]

    async def awrap_model_call(self, request, handler):
        if self.concurrent:
            responses = await asyncio.gather(*(handler(request) for _ in range(3)))
        else:
            responses = [await handler(request) for _ in range(3)]
        return responses[0]


task = {"messages": [{"role": "user", "content": "hi"}]}
model = NumberedModel(messages=iter([]))

result = create_agent(model, middleware=[Outer(concurrent=False), Inner()]).invoke(task)
print("sequential:", result["messages"][-1].content, "| seen:", result["seen"])

agent = create_agent(model, middleware=[Outer(concurrent=True), Inner()])
result = asyncio.run(agent.ainvoke(task))
print("concurrent:", result["messages"][-1].content, "| seen:", result["seen"])
```

**Expected:** the state holds the inner record of the committed response only:
`seen: ['sample 1']` and `seen: ['sample 4']`.

**Actual:**

```text
sequential: sample 1 | seen: ['sample 3']
concurrent: sample 4 | seen: ['sample 4', 'sample 5', 'sample 6']
```

**Suggested fix.** Tie the commands to the response they came with instead of
sharing one list: have `inner_handler` record each call's commands against the
`ModelResponse` object it returns (for example in a dictionary keyed by
`id(response)`, local to the outer call), and after the outer layer returns,
attach the commands recorded for the response it actually returned. When the
outer layer returns a response it built itself, keep today's behaviour and
document it. Per-call storage also makes concurrent calls safe.

## LangGraph: a reducer is read only from the last `Annotated` position

**Repository:** `langchain-ai/langgraph` (package `langgraph`)

**Title:** `Annotated[list, operator.add, Other]` silently becomes a `LastValue` channel: the reducer is only read from the last metadata item

**Environment:** langgraph 1.2.12, Python 3.12.

**Description.** `_is_field_binop` checks only `meta[-1]` for a reducer
(`langgraph/graph/state.py:1904-1922`), while `_is_field_channel`, just above
it, scans every metadata item (`state.py:1876-1901`). When a reducer is
followed by any other metadata, the field falls back to `LastValue` with no
warning: a later write replaces an earlier one, and two writes in one step
raise `InvalidUpdateError`. This bites LangChain agent middleware in
particular, whose state fields combine a reducer with markers such as
`OmitFromInput`, and LangChain's own schema scan finds those markers in any
position (`langchain/agents/factory.py:487-498`), so nothing else looks wrong.

**Reproduction:**

```python
import operator
from dataclasses import dataclass
from typing import Annotated

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict


@dataclass
class Marker:
    """Any non-callable metadata, such as LangChain's OmitFromInput."""


class ReducerFirst(TypedDict):
    log: Annotated[list[str], operator.add, Marker()]


class ReducerLast(TypedDict):
    log: Annotated[list[str], Marker(), operator.add]


for schema in (ReducerFirst, ReducerLast):
    graph = StateGraph(schema)
    graph.add_node("first", lambda state: {"log": ["first"]})
    graph.add_node("second", lambda state: {"log": ["second"]})
    graph.add_edge(START, "first")
    graph.add_edge("first", "second")
    graph.add_edge("second", END)
    compiled = graph.compile()
    channel = type(compiled.channels["log"]).__name__
    print(schema.__name__, channel, compiled.invoke({"log": []})["log"])
```

**Expected:** both declarations produce a `BinaryOperatorAggregate` channel
and `['first', 'second']`.

**Actual:**

```text
ReducerFirst LastValue ['second']
ReducerLast BinaryOperatorAggregate ['first', 'second']
```

**Suggested fix.** Either scan every metadata item for a two-argument callable
reducer, as `_is_field_channel` does for channels (raising if there are two),
or, if the last position is meant to be a rule, raise or warn when a
two-argument callable appears earlier and is ignored. Documenting the rule next
to the `Annotated` reducer examples would help either way.

## Related issues already open upstream

These match design smells in the findings page, so a comment with our evidence
fits better than a new issue.

- `langchain-ai/langchain#39812`, "ChatOpenRouter: 'timeout' parameter violates
  LangChain seconds convention and forwards as milliseconds". Evidence to add:
  `init_chat_model("openrouter:x/y", timeout=30)` sets the SDK's `timeout_ms`
  to 30, while `ChatAnthropic(timeout=30)` in langchain-anthropic 1.7.4 sets a
  30-second client timeout.
- `langchain-ai/langchain#40726`, "langchain-typesafe: allow supplying a
  classifier to experimental middleware". Evidence to add: besides the
  classifier (`auto_mode.py:167`), the 0.5 threshold (`auto_mode.py:39`) and
  the 30-message window (`auto_mode.py:173`) are also fixed.
