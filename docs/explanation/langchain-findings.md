# LangChain findings

This explanation records the bugs and design smells we met in LangChain,
LangGraph, Deep Agents and two provider integrations while building this
library: what goes wrong, the evidence, and how this library avoids it.

They are notes for maintainers, written in the spirit of a bug report. We read
LangChain's code for ideas rather than as a standard, and these are the places
where we chose a different pattern. Every finding was reproduced offline
against the versions below, with no API keys: request bodies were captured
with an `httpx.MockTransport`, and agents ran on a scripted fake chat model.
Line numbers refer to the installed package files. Draft upstream reports for
the bugs are kept in the repository at `docs/research/upstream-reports.md`;
none has been filed yet.

## Versions checked

| Package | Version |
|---|---|
| Python | 3.12.12 |
| langchain | 1.4.2 |
| langchain-core | 1.6.5 |
| langgraph | 1.2.12 |
| langgraph-prebuilt | 1.1.0 |
| deepagents | 0.7.19 |
| langchain-openrouter | 0.2.9 |
| openrouter (the Python SDK) | 0.10.8 |
| langchain-typesafe | 0.0.1a3 |

## Bugs

A bug here is behaviour that contradicts the package's own documentation or
silently loses what the caller asked for.

### The OpenRouter SDK drops reasoning keys it does not declare

**What goes wrong.** `ChatOpenRouter(reasoning={"enabled": False})` sends
`"reasoning": {}` to OpenRouter. The same happens to `{"exclude": True}` and
`{"max_tokens": 2000}`: only `effort` and `summary` survive. OpenRouter's
reasoning guide lists `enabled`, `exclude` and `max_tokens` as valid keys, so a
request to switch reasoning off, to hide it or to cap it never reaches the
service, which receives an empty reasoning object instead. Nothing warns.

**Evidence.** In the OpenRouter SDK [@openrouter2026pythonsdk],
`ChatRequestReasoning` declares only `effort` and `summary`
(`openrouter/components/chatrequest.py:130-141`), and the SDK's base model
keeps pydantic's default of ignoring unknown fields
(`openrouter/types/basemodel.py:9-12`). langchain-openrouter
[@langchainopenrouter2026] types the option as `dict[str, Any]` and passes it
through unchanged (`langchain_openrouter/chat_models.py:283`, `805-806`); its
docstring names only `effort` and `summary` (`chat_models.py:292-300`), but
nothing enforces that. The request bodies captured through a mock transport:

| `reasoning=` | Sent to OpenRouter |
|---|---|
| `{"enabled": False}` | `{}` |
| `{"exclude": True}` | `{}` |
| `{"max_tokens": 2000}` | `{}` |
| `{"effort": "none"}` | `{"effort": "none"}` |
| `{"effort": "low", "summary": "detailed"}` | unchanged |

**How this library avoids it.** The library never sets reasoning options on a
model you pass in. To switch reasoning off on an OpenRouter model, pass
`reasoning={"effort": "none"}`, which reaches the service intact.

### `ChatOpenRouter(n=...)` fails on every call

**What goes wrong.** `ChatOpenRouter(n=3)` is accepted at construction, then
every call raises `TypeError: Chat.send() got an unexpected keyword argument 'n'`.

**Evidence.** `n` is a validated field
(`langchain_openrouter/chat_models.py:267`), and the request parameters include
it whenever it is above 1 (`chat_models.py:800-801`)
[@langchainopenrouter2026]. None of the SDK's `Chat.send` signatures has an `n`
parameter (`openrouter/chat.py:20`, `178`, `335`) [@openrouter2026pythonsdk].

**How this library avoids it.** Protocols never ask for several choices in one
request. `PendingStep.sample(count=...)` draws each sample with its own model
call, which works with every chat model, and `concurrently=True` lets those
calls overlap under `ainvoke()`.

### Provider and native finish reason never reach `response_metadata`

**What goes wrong.** `ChatOpenRouter` copies `provider` and
`native_finish_reason` into a message's `response_metadata`, but neither ever
appears there, even when the HTTP response carries both.

**Evidence.** langchain-openrouter turns the SDK result into a dictionary with
`model_dump()` and then reads `provider` from its top level and
`native_finish_reason` from each choice
(`langchain_openrouter/chat_models.py:830-831`, `855`, `869-876`)
[@langchainopenrouter2026]. The SDK's `ChatResult` declares no `provider` field
(`openrouter/components/chatresult.py:43-69`) and its `ChatChoice` declares no
`native_finish_reason` field (`openrouter/components/chatchoice.py:32-45`)
[@openrouter2026pythonsdk], so both are discarded while the response is
parsed. A mocked response with `"provider": "FakeProvider"` and
`"native_finish_reason": "end_turn"` produced `response_metadata` with neither
key. OpenRouter's API reference documents `native_finish_reason` on each
choice. It does not document a top-level `provider`; the serving provider is
reported in the opt-in `openrouter_metadata` object, which `ChatResult` does
declare.

**How this library avoids it.** Nothing in the library reads routing metadata.
A `StepRecord` keeps the monitor's scores and the proposals, not which provider
served them.

### `content_blocks` ignores OpenRouter's `reasoning_details`

**What goes wrong.** When an OpenRouter reply carries its reasoning only in
`reasoning_details`, for example as a `reasoning.summary` entry with an empty
`reasoning` field, `AIMessage.content_blocks` returns no reasoning block. Code
that reads reasoning through LangChain's standard content blocks sees none.

**Evidence.** langchain-openrouter stores the field in
`additional_kwargs["reasoning_details"]` and tags the message with
`model_provider="openrouter"` (`langchain_openrouter/chat_models.py:1370-1371`,
`1388`) [@langchainopenrouter2026]. langchain-core [@langchaincore2026] has no
content-block translator for that provider (the translators in
`langchain_core/messages/block_translators/` cover Anthropic, Bedrock, Google,
Groq and OpenAI), so `content_blocks` falls back to best-effort parsing
(`langchain_core/messages/ai.py:258-303`). Its reasoning helper reads only
`additional_kwargs["reasoning_content"]`
(`langchain_core/messages/base.py:24-44`). A reply whose only reasoning was a
summary in `reasoning_details` gave a single text block.

**How this library avoids it.** `transcript.extract_reasoning_text` reads the
standard reasoning blocks first and falls back to the text or summary of each
`reasoning_details` entry, so a monitor whose view includes
`Channel.REASONING` sees the summary.

### TypeSafe `AutoModeMiddleware` ignores its default criteria and documents a missing threshold

**What goes wrong.** Two separate problems in the same constructor.

- The middleware defines default descriptions of a risky and a safe call, but
  never sends them: `AutoModeMiddleware(tools=[...])` asks its question with
  `criteria=None`.
- Its docstring describes a `threshold` ("Calls below `threshold` execute
  normally") and says validation covers the "threshold configuration", but the
  constructor has no such parameter.
  `AutoModeMiddleware(tools=[...], threshold=0.8)` raises `TypeError`, and the
  threshold is fixed at 0.5.

**Evidence.** In langchain-typesafe [@typesafe2026langchain], the
configuration model defaults `criteria` to the built-in descriptions
(`langchain_typesafe/experimental/middleware/auto_mode.py:61-71`). `__init__`
defaults the same argument to `None` and passes it explicitly
(`auto_mode.py:147`, `160-166`), which overrides the model's default. The
docstring gives `None` a separate meaning, "classify without outcome criteria"
(`auto_mode.py:114-115`), so the default and the documented meaning of `None`
conflict. The threshold text is at `auto_mode.py:89-90` and `157-158`; the
value is the constant `_PROBABILITY_THRESHOLD = 0.5` (`auto_mode.py:39`).

**How this library avoids it.** A `Monitor` only scores a step and a
`ControlProtocol` owns the thresholds, so the two cannot be tangled in one
constructor. The decision-model monitor being built takes the TypeSafe client
as a parameter ([#12](https://github.com/Antonio-Tresol/langchain-sync-monitors/issues/12)), and each threshold is a protocol argument whose
default warns until you replace it ([#14](https://github.com/Antonio-Tresol/langchain-sync-monitors/issues/14)).

### The composed model handler shares one command list between calls

**What goes wrong.** When several middlewares wrap the model call, LangChain
composes them so that an outer layer's `handler` runs the inner layers. The
commands the inner layers return are gathered in one list per outer call,
which every handler call clears and refills. The handler is documented as safe
to call more than once for retries, and when the outer layer returns the last
response, the result is correct. When it returns an earlier response, the
state receives the commands of the last call instead. When it runs the calls
concurrently, the commands of every call are applied.

**Evidence.** `compose_two` in langchain [@langchain2026]
(`langchain/agents/factory.py:313-345`, and `404-436` for async). The retry
contract is stated at `langchain/agents/middleware/types.py:513` and `524`. We
reproduced it with an outer middleware that draws three samples and returns the
first, around an inner middleware that records the sample it saw in an
`operator.add` key:

| Run | Committed | Inner records in the state |
|---|---|---|
| `invoke()`, sequential calls | sample 1 | sample 3 |
| `ainvoke()`, sequential calls | sample 4 | sample 6 |
| `ainvoke()`, `asyncio.gather` | sample 7 | samples 7, 8 and 9 |

With a last-value key instead, the concurrent case raises `InvalidUpdateError`.

**How this library avoids it.** The monitor draws several samples and may
commit any of them, so no layer inside it may return commands. In
`create_agent`, middleware listed after the monitor becomes an inner layer, so
the monitor goes last in the list; `AGENTS.md` makes that a project rule. In
Deep Agents the layers inside user middleware only change the request, so the
default stack is safe (check S1, [#3](https://github.com/Antonio-Tresol/langchain-sync-monitors/issues/3)). A warning at construction for any
other middleware placed after the monitor is being built with the middleware
([#17](https://github.com/Antonio-Tresol/langchain-sync-monitors/issues/17)).

### LangGraph reads a reducer only from the last `Annotated` position

**What goes wrong.** In `Annotated[list[X], operator.add, OmitFromInput]`,
LangGraph does not see the reducer, because it looks only at the last metadata
item. The field silently becomes a last-value channel: a later write replaces
an earlier one, and two writes in the same step raise `InvalidUpdateError`.
LangChain's own schema scan still finds `OmitFromInput` in any position, so the
field is hidden from the input schema as intended, and nothing points to the
lost reducer.

**Evidence.** In langgraph [@langgraph2026], `_is_field_binop` checks only
`meta[-1]` (`langgraph/graph/state.py:1904-1922`), while `_is_field_channel`,
just above it, scans every metadata item (`state.py:1876-1901`). LangChain's
`_resolve_schema` also scans every item for `OmitFromSchema`
(`langchain/agents/factory.py:487-498`) [@langchain2026]. Reproduced on a small
state graph and on `create_agent` with a middleware that appends one record per
model call:

| Declaration | Channel | Two writes in sequence | Two writes in parallel | Records after two model calls |
|---|---|---|---|---|
| `Annotated[list[str], operator.add, OmitFromInput]` | `LastValue` | the second only | `InvalidUpdateError` | one |
| `Annotated[list[str], OmitFromInput, operator.add]` | `BinaryOperatorAggregate` | both | both | two |

**How this library avoids it.** `monitor_log` is declared
`Annotated[list[StepRecord], OmitFromInput, operator.add]`, with the reducer
last, and `AGENTS.md` makes the order a project rule. The state schema that
carries it arrives with the middleware ([#17](https://github.com/Antonio-Tresol/langchain-sync-monitors/issues/17)).

## Design smells

A design smell here works as documented but makes a mistake easy to write and
hard to see. We did not copy these patterns.

### Untyped payloads where the shape is known

Middleware hooks return `dict[str, Any] | None`
(`langchain/agents/middleware/types.py:431-492`, `650-663`) and
`ModelRequest.model_settings` is a `dict[str, Any]` (`types.py:83`)
[@langchain2026]. `ToolCallRequest.state` is typed `Any`
(`langgraph/prebuilt/tool_node.py:133-148`) [@langgraphprebuilt2026]. A
misspelt state key or a value of the wrong type passes the type checker.

This library's shared types are frozen dataclasses, `TypedDict`s, enums and
`Literal`s (`contracts.py`), checked by strict mypy, and the gate rejects `Any`
in our signatures. Untyped LangChain surfaces are confined to one boundary
module, which is being built with the middleware ([#17](https://github.com/Antonio-Tresol/langchain-sync-monitors/issues/17)).

### A missing sync or async hook is found only at run time

`AgentMiddleware.wrap_model_call` and `awrap_model_call` have default bodies
that raise `NotImplementedError` (`langchain/agents/middleware/types.py:586-596`,
`638-648`) [@langchain2026]. A middleware that defines only
`wrap_model_call` builds and runs under `invoke()`, then fails under
`ainvoke()`; neither the type checker nor `create_agent` notices.

`Monitor` declares both `evaluate` and `evaluate_sync` abstract, so a monitor
that lacks one cannot be instantiated. Control protocols are written once, as
coroutines, and serve both paths; the [design](design.md#sync-and-async)
explains how.

### An undeclared jump target is silently ignored

A hook that returns `{"jump_to": "end"}` without declaring
`can_jump_to=["end"]` through the `hook_config` decorator gets a plain edge,
and the jump is dropped (`langchain/agents/factory.py:2067-2091`)
[@langchain2026]. No error or warning is raised. In our
reproduction the undeclared version called the model anyway; the declared one
ended the run.

The library does not use `jump_to`. The halt being built ([#15](https://github.com/Antonio-Tresol/langchain-sync-monitors/issues/15)) ends the
run with a final message that has no tool calls, which stops the agent loop by
LangChain's ordinary rule.

### Sync and async bodies copied by hand

`LLMToolSelectorMiddleware.wrap_model_call` and `awrap_model_call` are the same
66 lines apart from `await`
(`langchain/agents/middleware/tool_selection.py:383-448`, `450-515`)
[@langchain2026]. The copies have already drifted in one place: the sync
docstring calls its handler an "Async callback" (`tool_selection.py:392`).

In this library a protocol is one coroutine that talks to a `PendingStep`
(`contracts.py`), and the middleware being built ([#17](https://github.com/Antonio-Tresol/langchain-sync-monitors/issues/17)) gives the pending
step a sync and an async implementation, so each protocol has one copy.

### Options typed as unions of unrelated shapes

`OnParsingFailure` is
`Literal["error", "none", "all"] | list[str] | Callable[[Any], list[str]]`
(`langchain/agents/middleware/tool_selection.py:42`) [@langchain2026]. A reader
has to find the code to learn what each shape means.

This library gives each choice its own enum (`Resampling`,
`FeedbackVisibility`, `SubagentHalt` in `contracts.py`) and uses an abstract
class where behaviour varies (`Fallback`).

### Human-in-the-loop edits a message in place

`HumanInTheLoopMiddleware.after_model` assigns the revised list to
`last_ai_msg.tool_calls` on the message object it took from the state, then
returns that same object as its update
(`langchain/agents/middleware/human_in_the_loop.py:517-525`) [@langchain2026].
The edit is visible to anything else holding the object, whether or not the
update is committed.

This library's records and decisions are frozen dataclasses, and every message
the monitor inserts is a new message with a fresh id (`monitor-<uuid4>`), a
rule in `AGENTS.md`.

### Deep Agents: a hardcoded default model, and replacement by name

With no model, `create_deep_agent` falls back to
`ChatAnthropic(model_name="claude-sonnet-4-6")`
(`deepagents/graph.py:144-152`, `598-614`) [@deepagents2026]. Since 0.5.3 this
raises a `LangChainDeprecationWarning`, and the model becomes required in
1.0.0, so this smell is on its way out.

A user middleware whose `name` matches a built-in one replaces it in place,
with no warning (`deepagents/graph.py:205-239`, applied at `931`); the
documentation of the `middleware` argument does not mention it
(`graph.py:367-397`). In our reproduction a user middleware that happened to be
named `FilesystemMiddleware` removed every file tool (`ls`, `read_file`,
`write_file`, `edit_file`, `glob`, `grep`, `execute` and `delete`), leaving
only `task`, although Deep Agents otherwise protects that middleware from
removal (`graph.py:242-257`).

This library never picks a model: every model is a constructor argument. The
monitor middleware being built names itself `monitor[<agent name>]`, unique
within an agent ([#17](https://github.com/Antonio-Tresol/langchain-sync-monitors/issues/17)), and the helper that adds monitors to subagents
rejects override names that match no subagent ([#18](https://github.com/Antonio-Tresol/langchain-sync-monitors/issues/18)).

### TypeSafe `AutoModeMiddleware` hardcodes its classifier, threshold and window

The question id, the 0.5 threshold, the classifier and the 30-message window
are fixed (`langchain_typesafe/experimental/middleware/auto_mode.py:38-39`,
`167`, `173`) [@typesafe2026langchain]; none can be passed in.

In this library each is a parameter. `MonitorView.most_recent_entries` limits
how much history a monitor reads, and in the parts being built the decision
model is passed to the monitor ([#12](https://github.com/Antonio-Tresol/langchain-sync-monitors/issues/12)) and thresholds belong to the
protocol ([#14](https://github.com/Antonio-Tresol/langchain-sync-monitors/issues/14)).

### `ChatOpenRouter.timeout` is in milliseconds

`ChatOpenRouter` accepts `timeout`, documented as "Timeout for requests in
milliseconds", and passes it on as the SDK's `timeout_ms`
(`langchain_openrouter/chat_models.py:223-224`, `455-456`)
[@langchainopenrouter2026]. `init_chat_model("openrouter:...", timeout=30)`
therefore sets a 30 ms timeout, while `ChatAnthropic(timeout=30)` in
langchain-anthropic 1.7.4 sets 30 seconds. A value copied from another
provider gives a timeout a thousand times shorter than intended.

The library never sets a timeout on a model you give it. If you build a
`ChatOpenRouter` for a monitor or a trusted model, give the timeout in
milliseconds.
