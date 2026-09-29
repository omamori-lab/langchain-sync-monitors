# Monitor decisions in LangSmith and Langfuse

This research note covers how LangSmith and Langfuse store and display runs that
come from `langchain-core` callbacks. It recommends how `langchain-sync-monitors`
should shape its monitoring spans so that both tools show them well, and how an
optional part 2 could export verdicts as LangSmith feedback and Langfuse scores.

Research date: 29 September 2026.

## Method and versions

- **Installed** in a scratch environment:
  - `langsmith` 0.14.1;
  - `langchain-core` 1.6.6;
  - `langchain` 1.4.3;
  - `langgraph` 1.2.12;
  - `langfuse` 4.15.6.
- **Langfuse is at v4, not v3.** The v4 SDK still uses OpenTelemetry and
  `langfuse.langchain.CallbackHandler` in the same way. Everything below about
  Langfuse is read from the 4.15.6 source.
- **Seven probe scripts** sit next to this file (`probe_*.py`). They drive the
  real `LangChainTracer` with a mocked LangSmith client, and the real Langfuse
  `CallbackHandler` with an in-memory OpenTelemetry exporter. Nothing left the
  machine.
- **Probes that exercise a graph run under `invoke` and `ainvoke`.**
  `probe_patch.py` calls one tracer function directly, and
  `probe_late_inputs.py` uses a bare sync callback manager.
- **Probe results are marked "(probe)"**; everything else comes from source or
  from official documentation.

## Summary

1. **Both tools nest hand-started runs correctly.** A run opened with
   `CallbackManager.on_chain_start` from a node's config appears under that node
   in both LangSmith and Langfuse, and its child LLM runs appear under it. This
   holds for `invoke` and `ainvoke` (probe).
2. **`run_type` matters to LangSmith and to `astream_events`, not to Langfuse.**
   - Langfuse picks the observation type from the callback kind: chain, llm, tool
     or retriever.
   - The one exception is any chain whose name or serialised class path contains
     "agent". It becomes an `agent` observation, so `monitor[research-agent]`
     turns into an agent (probe).
   - Keep agent and delegation names out of span names.
3. **Tags and metadata are fixed when a run starts.** `on_chain_end` carries
   neither, so a span opened before the decision cannot carry tags such as
   `halted` or `flagged`.
   - Recommendation: a short `monitor decision` child span, opened after the
     decision, that carries the outcome in its tags and metadata.
   - The full detail stays in the step span's outputs.
4. **Metadata inherits to child runs by default.** Put identifying keys on the
   monitor spans with `add_metadata(..., inherit=False)`, or the agent's own
   samples get labelled as monitor runs (probe).
5. **LangSmith's Trajectory view filters out runs with
   `ls_agent_type: "middleware"`.** LangSmith recommends this key for
   guardrails.
   - It survives at the top level of `create_agent` (probe).
   - An enclosing `ls_agent_type` in the tracing context overwrites it. Deep
     Agents sets `subagent` for subagents (probe of `_patch_missing_metadata`).
   - `ls_message_view_exclude` is the documented key for guardrail LLM calls.
   - Put both keys only on spans whose subtree holds monitor work alone: the
     judgement, classifier and decision spans, and the judge's calls.
   - Never put either key on the step span. The agent's real turns nest under
     it, and whether the filter reaches descendants cannot be checked offline.
6. **Do not tag monitoring spans `langsmith:hidden`.**
   - LangGraph uses the tag for internal plumbing.
   - Langfuse demotes such runs to level `DEBUG`.
   - LangGraph drops them from its message, task and debug streams.
7. **Attaching LangSmith and Langfuse to the same `create_agent` run breaks
   Langfuse nesting under `ainvoke`.** `create_agent` wraps each
   `wrap_model_call` in LangSmith's `@traceable`, and only LangSmith sees that
   run.
   - The model call then lands in a second Langfuse trace, even with a
     middleware that does nothing (probe).
   - The library avoids this for its own spans by opening them from the node
     config's callback manager rather than re-configuring one (probe).
8. **Part 2 recommendation.**
   - A vendor-free verdict listener in the core, and two extras that implement
     it.
   - LangSmith feedback goes on the step run, in process. Its id is the
     LangChain run id, and `session_id` is now required.
   - For Langfuse, no public API maps a LangChain run to its trace or
     observation under `ainvoke`. The exact path is therefore a post-hoc
     exporter keyed on our own `monitor_step_id` metadata.
   - In-process trace scores through `handler.last_trace_id` are best-effort
     only: they can be attributed to the wrong trace when one handler serves
     concurrent requests.
   - No shared OpenTelemetry mechanism exists today that both tools turn into
     scores or feedback.

## 1. Langfuse's LangChain integration (SDK 4.15.6)

### 1.1 How runs become observations

`LangchainCallbackHandler` keeps a dictionary from LangChain `run_id` to a
Langfuse observation object, which wraps an OpenTelemetry span.

- **Opening a run.** On each start callback, the handler looks up
  `parent_run_id` in that dictionary and calls `start_observation` on the
  parent, or on the client when the parent is unknown.
- **Parent unknown.** The observation then attaches to whatever OpenTelemetry
  span is current, or becomes a new trace root.

The mapping from callback to observation type:

| LangChain callback | Langfuse observation type |
|---|---|
| `on_chain_start` | `chain`; `agent` if the name or any part of `serialized["id"]` contains "agent" |
| `on_llm_start`, `on_chat_model_start` | `generation` |
| `on_tool_start` | `tool` |
| `on_retriever_start` | `retriever` |
| `on_agent_action`, `on_agent_finish` | set the current run's type to `agent` |

- **`run_type` is ignored.** A chain started with `run_type="tool"` is still a
  `chain` in Langfuse.
- **Unused types.** Langfuse defines `span`, `event`, `evaluator`, `guardrail`
  and `embedding` types, but the LangChain handler never emits them. The
  `guardrail` type ("a component that protects against malicious content or
  jailbreaks") would suit monitors, but only Langfuse-specific code can reach it.

### 1.2 Names, inputs, outputs, metadata and tags

- **Name.** Taken from `kwargs["name"]`, then `serialized["name"]`, then
  `serialized["id"][-1]`, then `<unknown>`.
- **Inputs and outputs.** Stored as given, serialised to JSON strings on the
  `langfuse.observation.input` and `langfuse.observation.output` attributes.
  - Unlike LangSmith, Langfuse does not wrap a non-dict value in
    `{"input": ...}`.
  - `on_chain_end(outputs, inputs=...)` also updates the input (probe).
- **Metadata.** Each top-level key becomes an attribute
  `langfuse.observation.metadata.<key>`.
  - `str` and `int` values are kept as they are. Anything else, including
    floats, booleans and nested dicts, is stored as a JSON string.
  - Keep metadata flat and scalar.
  - Keys starting `langfuse_` are stripped: `langfuse_prompt` always, and the
    trace-attribute keys except on a root LLM run.
- **Tags.** Every run's tags are copied into the observation's metadata under
  `tags`.
  - Only the root run's tags, merged with `metadata["langfuse_tags"]`, become
    Langfuse trace tags.
  - In the probe, a `monitor` tag on a child span appeared only in
    `metadata.tags`, never in the trace tags. Langfuse tags "can't be added or
    edited" after creation.
- **Root run.** The root gets `is_langchain_root: true`. It opens
  `propagate_attributes(...)` with the session, user, tags, metadata and trace
  name taken from the `langfuse_session_id`, `langfuse_user_id`,
  `langfuse_tags` and `langfuse_trace_name` metadata keys.
  - Propagated metadata values must be US-ASCII strings of at most 200
    characters, or they are dropped.
  - This limit applies only to propagated (trace-level) metadata, not to our
    observation metadata.
- **Model calls.** An LLM run's model name comes from `ls_model_name` metadata or
  the serialised class. Usage is parsed from `usage_metadata` and provider
  fields.

### 1.3 Nesting and runs started by hand

A run started inside a node with
`get_callback_manager_for_config(config).on_chain_start(None, inputs, name=...)`
receives the node's run as `parent_run_id` and nests under it. LLM calls made
with `callbacks=run_manager.get_child()` nest under the hand-started run. The
probe confirmed this for sync and async (`probe_e2e.py`, `probe_agent.py`).

Two practical details:

- **Tags and metadata go on the manager.** `CallbackManager.on_chain_start`
  forwards `self.tags` and `self.metadata` itself, so passing `tags=` raises
  `TypeError: got multiple values for keyword argument 'tags'` (probe).
  - Put them on a copy of the manager with `add_tags(..., inherit=False)` and
    `add_metadata(..., inherit=False)`.
  - Inheritable metadata reaches every descendant. In `probe_e2e.py` the judge's
    LLM run inherited `monitor` and `delegation_id`.
- **Async context.** The Langfuse handler is not `run_inline`, so under an
  `AsyncCallbackManager` it runs in an executor on a copy of the context.
  - Nesting still works, because it goes through `run_id`, not through the
    OpenTelemetry context.
  - Anything that reads the current OpenTelemetry span afterwards sees nothing.
    `langfuse.get_current_observation_id()` returned our span's id in sync and
    `None` in async (probe).
  - The same executor hop means the attributes that `propagate_attributes`
    carries (session and trace tags) reached only the root observation in the
    async probe. Child observations had none (probe, upstream behaviour).

### 1.4 Filtering and levels

The handler never drops a LangChain run.

- **`langsmith:hidden`.** Runs with this tag get observation level `DEBUG`,
  which the trace view can hide with a minimum-level filter (probe; Langfuse
  issue 5019).
- **Errors.** LangGraph's `GraphBubbleUp` (interrupts, handoffs) ends a run at
  level `DEFAULT`. Other errors give `ERROR` with the message as
  `status_message`.
- **Level from callbacks.** A callback cannot set `WARNING` or any other level;
  only the hidden tag and errors change it.
- **Raw OpenTelemetry spans.** v4 exports by default only spans from the
  Langfuse tracer, spans with `gen_ai.*` attributes, or known instrumentation
  scopes (`span_filter.py`).
  - A library that emitted its own OpenTelemetry spans under its own scope would
    be dropped unless the user sets `should_export_span`.
  - This weighs against a raw OpenTelemetry route (section 4.4).

### 1.5 LangGraph specifics

- **Interrupt and resume.** The handler remembers the trace of an interrupted
  root run by `thread_id`. It continues that trace when the next root input is a
  `Command(resume=...)`.
- **Graph metadata.** `langgraph_node`, `langgraph_step`, `langgraph_path`,
  `langgraph_triggers` and `thread_id` arrive in every run's metadata and show on
  every observation.
- **Only `langfuse_*` keys are stripped.** The handler removes no other key.

### 1.6 `create_agent` with both tracers attached

`create_agent` wraps every middleware `wrap_model_call` and `awrap_model_call`
in `langsmith.traceable(name=f"{m.name}.wrap_model_call")`.

- **Who sees the wrapper.** That run exists only for LangSmith. When a
  `LangChainTracer` is attached, `CallbackManager.configure` reparents calls made
  inside it to the traceable run.
- **What Langfuse sees.** Langfuse receives a `parent_run_id` it never saw.

Probe results (`probe_orphan.py`, with a mock whose `otel_exporter` is `None`,
as on a real client):

| Callbacks attached | `invoke` | `ainvoke` |
|---|---|---|
| Langfuse only | one trace, correct nesting | one trace, correct nesting |
| Langfuse and LangSmith | one trace (falls back to the current OpenTelemetry span) | the model call, with or without a monitor span, lands in a second Langfuse trace |

- **The split is upstream.** A plain pass-through middleware shows it too.
- **The library can avoid it for its own spans.** Build the run manager from the
  node config's existing `CallbackManager` (copy its handlers, parent and tags)
  instead of calling `get_callback_manager_for_config`, which re-configures and
  reparents.
  - Result: one Langfuse trace in both modes (`probe_mitigate.py`).
  - In LangSmith the step span then sits under `model`, beside an empty
    `Monitor.awrap_model_call` run, rather than inside it.
  - The agent's samples still nest under the step span in both tools.

## 2. LangSmith and `LangChainTracer`

### 2.1 How runs are recorded

- **Ids.** The LangSmith run id is the LangChain `run_id` (probe; LangChain now
  generates UUIDv7). The trace id is the root run's id, and each `Run` in
  `tracer.run_map` carries `trace_id`, `dotted_order` and `session_name`.
- **Run types.** `run_type` from `on_chain_start` kwargs is honoured, default
  `chain`. Valid types are `tool`, `chain`, `llm`, `retriever`, `embedding`,
  `prompt` and `parser` (`RUN_TYPE_T`). Chat models are traced as `llm`.
- **Inputs and outputs.**
  - A non-dict chain input is wrapped as `{"input": ...}`, and a non-dict output
    as `{"output": ...}`.
  - `on_chain_end(outputs, inputs=...)` replaces the inputs, which are then sent
    with the update (probe).
  - Pass dicts, so both tools show the same top-level keys.
- **Metadata and tags.**
  - Metadata is stored in `extra.metadata`. Tags are merged with the tracer's
    tags and posted as a list.
  - Tracer-level default metadata (from `tracing_context(metadata=...)`) fills
    missing keys: "first wins".
  - `ls_agent_type` is the exception. It is always overwritten by the nearest
    enclosing tracer config (`OVERRIDABLE_LANGSMITH_INHERITABLE_METADATA_KEYS`).
- **Run types and `astream_events`.** `astream_events` names its events
  `on_{run_type}_start` and `on_{run_type}_end`.
  - A chain started with `run_type="tool"` therefore emits `on_tool_start` to
    stream consumers, which may render it as an agent tool call
    (`event_stream.py`).
  - `on_tool_start` does the same, and Langfuse also copies all its keyword
    arguments into metadata.

### 2.2 How runs display and filter

- **Run type in the tree.** The run type sets the icon in the tree. LLM runs get
  the message renderer, and chain inputs and outputs are shown as structured
  data.
- **No docs on the chain renderer.** I found no official page that describes it.
  What is documented is filtering on "specific key-value pairs in the inputs and
  outputs", with dot notation for nested keys. Keep outputs flat, with a few
  top-level keys.
- **Query language.**
  - Comparators: `eq`, `neq`, `gt`, `gte`, `lt`, `lte`, `has`, `search`, `in`.
  - Fields: `id`, `name`, `run_type`, `status`, `start_time`, `end_time`,
    `latency`, `tags`, `metadata_key`, `metadata_value`, `feedback_key`,
    `feedback_score`.
  - The documented query language has no input or output key fields. Its docs
    advise narrowing server-side, then walking child runs locally, for
    predicates over nested inputs or outputs.
- **Where filters apply.** `filter` applies to the returned run, `trace_filter`
  to the trace's root run, and `tree_filter` to any run in the trace. This is how
  to find traces that contain a halted step.
- **Examples:**
  - `has(tags, "monitor:halted")`
  - `and(eq(metadata_key, "monitor_agent"), eq(metadata_value, "main"))`
  - `and(eq(feedback_key, "monitor_suspicion"), gt(feedback_score, 0.7))`
- **UI.** Switch the table from **Traces** to **Runs** to filter spans by name,
  type, metadata and tags, and use tree filters for "traces whose runs have
  some attribute".

### 2.3 `langsmith:hidden`

- **SDKs.** Neither `langsmith` nor `langchain-core` interprets the tag.
- **LangGraph** defines `TAG_HIDDEN = "langsmith:hidden"` and puts it on
  internal nodes such as the `__start__` writer. It uses the tag to:
  - leave hidden tasks out of `tasks` and `debug` stream events;
  - skip their messages in `stream_mode="messages"` (`pregel/_messages.py`).
- **Langfuse** maps the tag to level `DEBUG`.
- **LangSmith UI.** It de-emphasises or hides such runs in the trace tree, but I
  found no official page that states this.
- **Recommendation:** never use the tag on monitoring spans. It would demote or
  hide exactly what we want seen, and would change LangGraph streaming.

### 2.4 Trajectory view keys

- **Grouping.** The Trajectory view groups runs by `thread_id` and places them by
  `ls_agent_type`, whose values are `root`, `subagent`, `middleware` and
  `compaction`.
- **Filtering.** "Runs marked as middleware or compaction are currently filtered
  out".
- **Guardrails.** The docs advise guardrail and policy-check authors to set
  `ls_agent_type: "middleware"`.
- **`ls_message_view_exclude`** (constant `langsmith.LS_MESSAGE_VIEW_EXCLUDE`)
  excludes one run from the view.
  - It is checked by presence, not truthiness.
  - It is recommended for "safety filters, or routing/guardrail decisions".
  - With LangChain `RunnableConfig`, it must be set on each run, because
    inheritance is not guaranteed.

Probe results:

- **Top level.** `ls_agent_type: "middleware"`, set with `inherit=False` on the
  step span inside `create_agent`, reached LangSmith as set. The agent's sample
  run under it did not carry the key (`probe_agent.py`).
- **Inside a subagent.** With a tracer carrying `ls_agent_type: "subagent"` (as
  Deep Agents sets for subagents), `_patch_missing_metadata` rewrote the span's
  value to `subagent` (`probe_patch.py`).
- **Where the key goes.** The probe put the key on a step span only to test
  whether it survives. Section 3 places it on monitor-only spans instead.

### 2.5 Size limits

- **SDK.** The SDK batches compressed trace uploads up to 20 MiB by default
  (`size_limit_bytes=20_971_520`), adjustable with `Client(max_batch_size_bytes=...)`.
- **Plans** limit hourly trace events (50,000 to 500,000 per hour) and hourly
  ingested data across inputs, outputs and metadata:
  - 500 MB per hour on Developer without payment;
  - 2.5 GB with payment;
  - 5 GB on Startup and Plus.
- **No per-run limit found.** I found no documented per-run or per-field size
  limit.
- **Retention.** Traces start at 14-day base retention. API feedback with
  `extend_trace_retention=True` (the SDK default) upgrades the trace to extended
  retention.

Langfuse for comparison:

- **Cloud API.** 5 MB per request and per response.
- **Ingestion.** Batched ingestion is limited by requests per minute (1,000 to
  20,000 by plan).

## 3. Recommended span design

### 3.1 Tree

```
model                                   LangGraph node (existing)
  Monitor.awrap_model_call              LangSmith only: create_agent's traceable wrapper
  monitor step                          chain; one per monitored step
    ChatModel                           agent sample 1 (existing llm run)
    monitor judgement                   chain; one per sample judged
      ChatModel                         judge call (existing llm run), ls_message_view_exclude
    monitor classifier                  chain; HTTP classifier call, if used
    ChatModel                           agent sample 2, if resampled (existing)
    monitor judgement
      ChatModel
    monitor decision                    chain; opened after the decision, ends at once
```

### 3.2 Per span

| Span | Name | `run_type` | Tags (not inherited) | Metadata (not inherited) | Inputs | Outputs |
|---|---|---|---|---|---|---|
| Step | `monitor step` | `chain` | `monitor` | identifying keys (below) only; no Trajectory view keys | `{}` at start; at end `{"step_number", "proposed_step"}` | step summary (3.3) |
| Judgement | `monitor judgement` | `chain` | `monitor` | identifying keys; `monitor_sample_index`; `ls_agent_type: "middleware"` | `{"sample_index", "view"}` (view settings, not the transcript) | `{"suspicion", "reason"}` |
| Judge LLM calls | model's own | `llm` | none added | `ls_message_view_exclude: true` | existing | existing |
| Classifier | `monitor classifier` | `chain` | `monitor` | identifying keys; `monitor_classifier_host` (host only); `ls_agent_type: "middleware"` | `{"sample_index"}` | `{"suspicion", "label", "status_code"}` |
| Decision | `monitor decision` | `chain` | `monitor`, `monitor:<outcome>`, plus `monitor:flagged` when flagged | identifying keys; `monitor_outcome`, `monitor_flagged`, `monitor_max_suspicion`, `monitor_blocked_count`, `monitor_sample_count`; `ls_agent_type: "middleware"` | `{}` | `{"outcome", "flagged", "max_suspicion"}` |

Identifying keys (flat, scalar, prefixed so they never collide with `thread_id`,
`langgraph_*` or `ls_*`):

| Key | Type | Source |
|---|---|---|
| `monitor_name` | `str` | `StepRecord.monitor` |
| `monitor_agent` | `str` | `StepRecord.agent` |
| `monitor_step_number` | `int` | `StepRecord.step_number` |
| `monitor_delegation_id` | `str` | the delegation id; `main` at the top |
| `monitor_protocol` | `str` | the control protocol's name |
| `monitor_step_id` | `str` | the step span's LangChain run id, generated by us and passed as `run_id`; Langfuse does not store LangChain run ids, so this is the join key for post-hoc lookups |

### 3.3 Step outputs

Mirror `StepRecord`, which has no `ran` field; the per-sample flag is `executed`.
`flagged` is a separate boolean, not an outcome. Leave each sample's proposal
text out: the sample LLM runs already carry it as their output, and the rendered
proposal stays in `monitor_log`.

```json
{
  "outcome": "resampled",
  "flagged": true,
  "blocked_count": 1,
  "max_suspicion": 0.91,
  "samples": [
    {"suspicion": 0.91, "reason": "Deletes the user's backups.", "executed": false},
    {"suspicion": 0.12, "reason": "Lists the directory.", "executed": true}
  ]
}
```

### 3.4 Reasons

- **Fixed, low-cardinality names.**
  - Langfuse filters and aggregates observations by name, and LangSmith's `name`
    filter is an equality match.
  - A name such as `monitor[main] step 3` makes every span unique, and puts user
    text into the name.
  - Any agent name containing "agent" flips the Langfuse type to `agent`.
  - Pass `serialized=None`, so no class path is checked for "agent".
  - Readable detail goes in metadata, which both UIs show next to the span.
- **`chain` everywhere, including the classifier.**
  - `run_type="tool"` would show a tool icon in LangSmith only.
  - It would also emit `on_tool_start` into `astream_events`, where a UI may
    render it as the agent's own tool call.
- **Outcome on a separate decision span.** Tags and metadata cannot change after
  start, and the step span must open before the first sample so that the samples
  nest under it. This gives filterable outcomes with no vendor code:
  - LangSmith: `and(eq(name, "monitor decision"), has(tags, "monitor:halted"))`,
    or as a `tree_filter` to find the traces.
    - These strings are the same in the UI's advanced filter and in the API:
      `Client.list_runs(project_name=..., filter=..., trace_filter=...,
      tree_filter=...)`.
    - Example: `tree_filter='has(tags, "monitor:halted")'` together with
      `is_root=True` returns the traces that hold a halted step.
  - Langfuse (observations API or UI):
    `[{"type":"string","column":"name","operator":"=","value":"monitor decision"},
    {"type":"stringObject","column":"metadata","key":"monitor_outcome","operator":"=","value":"halted"}]`.
    - Pass this URL-encoded as the `filter` parameter of
      `GET /api/public/v2/observations`. The UI filter bar uses the same
      columns.
  - The alternative is to keep the outcome only in the step outputs and rely on
    part 2 for filtering. It is simpler, but gives no filtering for users who
    skip part 2.
- **Numbers.**
  - Suspicions go in outputs, where both tools show them in place. The decision
    span repeats `max_suspicion` in metadata for filtering.
  - Langfuse stores floats and booleans in metadata as JSON strings (`"0.91"`,
    `"true"`). A numeric metadata filter on them is unverified.
  - For threshold queries, use scores or feedback (part 2), which are numeric in
    both tools.
- **Tags.**
  - `monitor` on every monitor span lets users exclude them in one step:
    `astream_events(exclude_tags=["monitor"])` or `neq`-style filters.
  - Outcome tags follow LangGraph's colon style (`graph:step:1`).
  - In Langfuse these tags land only in observation metadata, under `tags`.
- **Trajectory view.**
  - `ls_agent_type: "middleware"` on the judgement, classifier and decision
    spans keeps them out of the Trajectory view at the top level.
  - Inside Deep Agents subagents the value is overwritten to `subagent`, so the
    view may show those spans as subagent actions there.
  - `ls_message_view_exclude` on the judge's and classifier's model calls keeps
    them out regardless, because it is checked by presence and never
    overwritten.
  - Put neither key on the step span. The agent's real turns nest under it, and
    no offline test can show whether the filter hides descendants. Either key on
    the step span could then hide the conversation itself.
- **Errors.** End a span with `on_chain_error` only when the monitor itself
  fails, for example a judge error that fails closed. A halt is a decision, not
  an error: marking it as one would colour it red in LangSmith, set `ERROR` in
  Langfuse and distort error-rate metrics.
- **How to open spans.**
  - Build the manager from the node config's existing callback manager, not
    through `CallbackManager.configure`.
  - This avoids the split trace in section 1.6.
  - When `config["callbacks"]` is a list or `None` (outside a graph), fall back
    to `get_callback_manager_for_config`.
  - Generate `run_id` ourselves, so `monitor_step_id` is known before the start
    callback.

### 3.5 Things to verify in a live run

- **LangSmith tree.** How the UI treats `langsmith:hidden` runs, and how it
  renders the step span's dict output.
- **Trajectory view.** Whether it still finds the agent's turns when the sample
  LLM runs sit one level deeper, under `monitor step`.
- **Feedback in tree filters.** Whether `tree_filter` accepts `feedback_key` and
  `feedback_score` for child-run feedback.
- **Langfuse numeric filters.** Whether `numberObject` metadata filters match
  floats stored as JSON strings.
- **Langfuse agent graph.** Whether the extra chain layer changes its agent graph
  view.

## 4. Part 2: feedback and scores

### 4.1 LangSmith `Client.create_feedback` (SDK 0.14.1)

```python
client.create_feedback(
    run_id,
    key,
    score=...,
    value=...,
    comment=...,
    trace_id=...,
    session_id=...,
    feedback_source_type="model",
    source_run_id=...,
    feedback_id=...,
    feedback_config=...,
    extra=...,
    extend_trace_retention=True,
)
```

- **`run_id`** targets any run: the root, or a child such as our step span.
- **`trace_id`** is optional, but without it feedback is sent synchronously. With
  it, feedback is batched with the run uploads in the background.
- **`session_id`** (the project id) is now required for run-level feedback.
  Omitting it warns, and raises on SmithDB-only deployments. Resolve it once per
  project with `client.read_project(project_name=...)` and cache it.
- **`score`** is numeric or boolean; **`value`** may be a string or dict, for
  categories.
- **`comment`** holds the judge's reason.
- **`feedback_config`** declares continuous, categorical or freeform feedback.
- **`feedback_source_type="model"` with `source_run_id`** links the feedback to
  the run that produced it, here the judge's LLM run.
  - This is the pattern that `langchain_core.tracers.evaluation.EvaluatorCallbackHandler`
    uses: `create_feedback(run_id, res.key, score=..., value=..., comment=...,
    source_run_id=..., feedback_source_type=MODEL)`.
- **`feedback_id`** lets a caller make feedback idempotent. uuid5 of run id and
  key gives the same id on retry.
- **`extend_trace_retention`** defaults to `True`, so per-step feedback would
  upgrade every monitored trace to extended retention, which costs more. Default
  it to `False` in our extra.
- **Display and filtering.** Feedback shows on the run it targets.
  - `feedback_key` and `feedback_score` filters work on runs.
  - `trace_filter` reads the root run's feedback.
  - Child-run feedback is found through the Runs table or a `tree_filter` (to
    verify, 3.5).

### 4.2 Langfuse `create_score` (SDK 4.15.6)

```python
langfuse.create_score(
    name=...,
    value=...,
    trace_id=...,
    observation_id=...,
    session_id=...,
    score_id=...,
    data_type=...,
    comment=...,
    config_id=...,
    metadata=...,
)
```

- **Data types:**
  - `NUMERIC`: a float;
  - `BOOLEAN`: 0 or 1;
  - `CATEGORICAL`: a string;
  - `TEXT`: 1 to 500 characters;
  - `CORRECTION`: listed in the SDK signature.
- **Targets.**
  - A trace needs `trace_id`.
  - An observation needs `trace_id` and `observation_id`.
  - A session needs `session_id`.
  - A score may arrive before its trace exists.
- **`score_id` is an idempotency key.** A re-sent score overwrites the old one
  only when `id`, `name` and the timestamp's date all match.
- **`config_id`** validates against a score config (name, numeric range,
  categories), which a project admin creates.
- **Display.** Scores appear on traces, observations and sessions, in score
  analytics and in custom dashboards, and through the API. The observations API
  filter bar uses the same columns as the UI.
- **Current-span helpers.** `score_current_span` and `score_current_trace` also
  exist, but they need a current Langfuse span, which callbacks do not provide
  under async (1.3).

### 4.3 Getting the ids from inside LangChain callbacks

| Need | LangSmith | Langfuse |
|---|---|---|
| Our span's id | the LangChain `run_id` we generate: public, exact | no public mapping from `run_id` to observation id; `get_current_observation_id()` right after the start callback works in sync only (probe); `handler._runs[run_id]` works in both but is private |
| Trace id | `tracer.run_map[str(run_id)].trace_id` for the `LangChainTracer` in `run_manager.handlers` (the same lookup `RunTree.from_runnable_config` uses); or track the root ourselves | `handler.last_trace_id`, public but racy (below) |
| Project | `tracer.project_name`, then `read_project` once for `session_id` | not needed |
| Client to send with | `tracer.client`: the user's configured client, with its workspace, endpoint and masking | `get_client()` or the handler's client |

Timing and concurrency:

- **LangSmith: read the trace id while the step span is open.**
  `BaseTracer._end_trace` pops a run from `run_map` when it ends, so reading it at
  decision time can fail. Carry the value in the verdict event.
- **`read_project` is a synchronous network call.** Under `ainvoke`, run it in a
  thread, or resolve it lazily off the event loop, and cache it per project name
  at process level. That is not per-run state.
- **Langfuse: `last_trace_id` is the handler's most recent start.** One
  module-level handler serving concurrent requests is the pattern Langfuse's own
  examples show, and there another request's start can overwrite the value
  between our start and our read. The race is therefore a normal case.
  - Read it immediately after opening the step span, and carry it in the event.
  - Even so, it can name the wrong trace under a shared handler.
  - Parallel subagents inside one invocation share the trace, so they are safe.

### 4.4 A vendor-neutral route?

- **For spans, the callbacks are the neutral layer.** Both tools consume
  `langchain-core` callbacks with no vendor code.
- **For verdicts there is no shared route today.**
  - OpenTelemetry's GenAI conventions define a `gen_ai.evaluation.result` event:
    `gen_ai.evaluation.name` (required), `gen_ai.evaluation.score.value`,
    `gen_ai.evaluation.score.label`, `gen_ai.evaluation.explanation`, parented
    to the evaluated span.
  - Its status is Development. I found no documentation that either LangSmith or
    Langfuse turns it into feedback or scores.
  - Raw OpenTelemetry spans from our own scope would be dropped by Langfuse v4's
    default export filter (1.4).
  - LangSmith's OpenTelemetry ingestion is a separate mode
    (`LANGSMITH_OTEL_ENABLED`).
  - Revisit when the event stabilises.

### 4.5 Options

**A. In-process exporters, one per extra, called when a step is decided.**
- **Fits LangSmith cleanly:** all ids are public or ours, and feedback is
  batched in the background when `trace_id` is given.
- **Langfuse:** no id is exact across sync and async through public APIs. Trace
  scores through `last_trace_id` can be mis-attributed under a shared handler.

**B. Post-hoc exporter.** After a run, read the monitor spans back through each
vendor's API, by name `monitor decision` and metadata `monitor_step_id`, then
write feedback or scores.
- Public APIs only, and exact observation ids in Langfuse.
- Needs API reads and has to wait for ingestion; more moving parts.

**C. Server-side rules, no library code.**
- LangSmith online evaluators filter runs by name, metadata and tags, sample a
  fraction, and write feedback.
- Langfuse `run_batched_evaluation` maps traces through evaluator functions into
  scores.
- Zero coupling, but each user configures it themselves, and LLM-as-judge
  evaluators would re-judge rather than copy our verdicts.

**D. A core `VerdictListener` protocol with no vendor imports, plus extras that
implement it.**
- **What the core passes:** a frozen event with the `StepRecord`, the step
  span's run id, the root run id, the judge run ids, and the run manager's
  handlers.
- **How it is configured:** a listener is a middleware constructor parameter,
  and holds no per-run state on `self`.

**Recommended: D as the structure. For LangSmith, A inside the extra. For
Langfuse, B as the exact path and A as an opt-in best-effort mode. C is
documented as the no-code alternative.**

- **`langchain-sync-monitors[langsmith]`, in process.** One feedback per key on
  the step run, sent through `tracer.client`.
  - `trace_id` comes from the tracer's `run_map`, read while the step span is
    open.
  - `session_id` comes from a cached `read_project`, run off the event loop
    under `ainvoke`.
  - Use `feedback_source_type="model"`, with `source_run_id` set to the
    judgement's LLM run.
  - A deterministic `feedback_id`, and `extend_trace_retention=False` unless the
    user opts in.
  - Optionally, send feedback only for flagged or halted steps, to limit events.
- **`langchain-sync-monitors[langfuse]`, exact path: post-hoc (option B).**
  - After the run, or in a periodic job, query `GET /api/public/v2/observations`
    for `name = "monitor decision"`, or for `monitor step` with a
    `monitor_step_id` metadata filter.
  - This returns the exact `trace_id` and observation `id`. Then call
    `create_score` with `observation_id` set.
  - Needs only public APIs, and is identical under `invoke` and `ainvoke`.
  - It must tolerate ingestion delay: retry with stamina, since these are
    network calls, and use a deterministic `score_id` so re-runs overwrite.
- **Langfuse best-effort mode (option A, opt-in).** Trace-level scores keyed by
  `handler.last_trace_id`, read immediately after opening the step span and
  carried in the event.
  - The docs must say plainly that under one handler shared by concurrent
    requests, a score can land on another request's trace.
  - Carry `metadata={"monitor_step_id", "monitor_step_number", "monitor_agent",
    "monitor_name"}`, so a mis-attributed score can be found and checked.
  - Add `observation_id` only when `get_current_observation_id()` resolves our
    span (sync).
  - Do not read `handler._runs`: it is private and would break the repo's rule of
    confining untyped surfaces.
- **Keys** (the same in both tools):
  - `monitor_suspicion`: numeric, the max over the step's samples;
  - `monitor_outcome`: categorical; LangSmith `value`, Langfuse `CATEGORICAL`;
  - `monitor_flagged`: boolean.
  - Put the judge's reason in `comment`.
  - Ship a documented score config and feedback config for each key.

## 5. Privacy and volume

What the monitor adds:

- **Proposal text.** The step span's `proposed_step` repeats the first sample's
  output.
- **Reasons.** The judges' reasons are model-written and may quote sensitive
  content.
- **Judge inputs.** The judge LLM calls carry the monitor's view of the
  transcript.
- **Volume.** Each judge call re-sends a view of the history, so volume grows
  with steps times history length, times samples when resampling. This dominates
  anything the monitor spans add; keep the monitor spans small.

Controls users already have:

- **LangSmith, all runs.** `LANGSMITH_HIDE_INPUTS`, `LANGSMITH_HIDE_OUTPUTS`,
  `LANGSMITH_HIDE_METADATA`, or `Client(hide_inputs=..., hide_outputs=...,
  hide_metadata=...)` with `True` or a callable.
  - The callable receives only the dict, never the run name. A stable top-level
    key such as `proposed_step` lets users target our field.
  - `anonymizer` masks inputs, outputs, metadata and error strings by rule.
  - `process_buffered_run_ops` sees whole run dicts, names included, for
    selective handling.
  - LangChain users pass the configured client as
    `LangChainTracer(client=Client(...))`.
  - `LANGSMITH_TRACING_SAMPLING_RATE` samples traces.
- **LangSmith, middleware wrapper spans.** A middleware's `trace_policy`, for
  example `TracePolicy(process_inputs=omit_payload)`, shapes what the
  `create_agent` wrapper span records.
  - Our middleware should set this. Otherwise the LangSmith-only
    `Monitor.awrap_model_call` run repeats the whole model request.
  - It does not affect callback runs.
- **Langfuse.** `Langfuse(mask=...)` receives `data=` only and applies to the
  input, output and metadata of observations the SDK creates, including the
  LangChain handler's.
  - `mask_otel_spans` runs at export and sees each span's `name` and attributes,
    so a user can delete `langfuse.observation.input` on spans named
    `monitor step` alone.
  - `sample_rate` and `should_export_span` control volume.

What the library should do:

- **Never duplicate the transcript** or per-sample proposals in monitor spans.
- **Pass content by default, as LangChain does for every LLM run.** Hiding it on
  our spans alone would protect nothing, because the child LLM runs carry the
  same text.
- **Point to vendor masking in the docs.**
  - OpenTelemetry-native guardrail libraries default the other way. NeMo
    Guardrails has `enable_content_capture: false` by default, and records a
    rail's reason only when content capture is on.
  - That default fits a library that owns its spans, not one that sits inside
    LangChain's callback tree.

## 6. How existing guardrail tools publish verdicts

- **NeMo Guardrails** emits OpenTelemetry spans through an adapter that uses only
  the OpenTelemetry API.
  - Spans: `guardrails.request`, then one `guardrails.rail` per activated rail,
    then `guardrails.action`, then LLM spans.
  - The decision sits in attributes: `rail.type`, `rail.name`, `rail.stop`
    (true only when the rail blocked) and `rail.decisions`.
  - `guardrails.rail.reason` is recorded only with content capture on. Content
    capture is off by default.
  - Pattern: one span per check, decision as structured attributes, content
    opt-in.
- **Guardrails AI** is traced through OpenInference
  (`openinference-instrumentation-guardrails`). "Validator outcomes appear as
  attributes on the Guardrails span", which Arize or Langfuse ingest over
  OpenTelemetry. Pattern: a guard span, with validator results as attributes.
- **LLM Guard** (Langfuse's security cookbook) wraps each guarded function in
  `@observe()`.
  - It records each scanner's risk as a numeric score on the current span:
    `langfuse.score_current_span(name="input-violence", value=risk_score)`, with
    names like `input-<scanner>`.
  - Pattern: numeric risk as a score on the span that ran the check, one score
    per scanner.
- **Langfuse evaluators.**
  - LLM-as-judge evaluators write scores to traces or observations.
  - `run_batched_evaluation` maps fetched traces through evaluator functions into
    scores.
  - Dedicated `evaluator` and `guardrail` observation types exist for SDK users.
- **LangSmith evaluators.**
  - `EvaluatorCallbackHandler` in `langchain-core` evaluates a run when it is
    persisted and calls `create_feedback` with `feedback_source_type=MODEL` and
    `source_run_id`.
  - Online evaluators filter runs by metadata or tags, sample a fraction, and
    attach feedback.
  - LangSmith's docs ask guardrail authors to mark their runs
    `ls_agent_type: "middleware"`.

Patterns shared across these tools:

1. **A dedicated span per check**, with the decision as structured fields.
2. **Scores or feedback** for the numbers people aggregate and filter.
3. **Guardrail runs kept out of conversation views.**
4. **Content capture as a deliberate choice**, off by default in
   OpenTelemetry-native libraries.

The design in section 3 follows patterns 1 and 3. Part 2 adds pattern 2.
Section 5 explains why pattern 4 is left to the vendors here.

## Sources

Official documentation:

- Langfuse LangChain integration: https://langfuse.com/integrations/frameworks/langchain
- Langfuse observation types: https://langfuse.com/docs/observability/features/observation-types
- Langfuse metadata: https://langfuse.com/docs/observability/features/metadata
- Langfuse tags: https://langfuse.com/docs/observability/features/tags
- Langfuse masking: https://langfuse.com/docs/observability/features/masking
- Langfuse observations API and filters: https://langfuse.com/docs/api-and-data-platform/features/observations-api
- Langfuse scores via SDK: https://langfuse.com/docs/evaluation/evaluation-methods/scores-via-sdk
- Langfuse scores overview: https://langfuse.com/docs/evaluation/scores/overview
- Langfuse API limits: https://langfuse.com/faq/all/api-limits
- Langfuse security and guardrails: https://langfuse.com/docs/security-and-guardrails
- Langfuse LLM security cookbook (LLM Guard): https://langfuse.com/guides/cookbook/example_llm_security_monitoring
- Langfuse issue on `langsmith:hidden` as DEBUG: https://github.com/langfuse/langfuse/issues/5019
- LangSmith trace query syntax: https://docs.langchain.com/langsmith/trace-query-syntax
- LangSmith filter traces: https://docs.langchain.com/langsmith/filter-traces-in-application
- LangSmith metadata and tags: https://docs.langchain.com/langsmith/add-metadata-tags
- LangSmith view traces (Trajectory view keys): https://docs.langchain.com/langsmith/view-traces
- LangSmith Trajectory view integrations: https://docs.langchain.com/langsmith/trajectory-view-integrations
- LangSmith masking: https://docs.langchain.com/langsmith/mask-inputs-outputs
- LangSmith limits and retention: https://docs.langchain.com/langsmith/administration-overview
- LangSmith feedback `session_id` migration: https://docs.langchain.com/langsmith/smithdb-sdk-migration-feedback
- LangSmith online evaluators: https://docs.langchain.com/langsmith/online-evaluations
- NeMo Guardrails span reference: https://docs.nvidia.com/nemo/guardrails/observability/tracing/span-reference
- NeMo Guardrails tracing configuration: https://docs.nvidia.com/nemo/guardrails/configure-guardrails/yaml-schema/tracing-configuration
- Guardrails AI tracing (Arize): https://arize.com/docs/ax/integrations/python-agent-frameworks/guardrails-ai/guardrails-ai-tracing
- OpenInference Guardrails instrumentation: https://github.com/Arize-ai/openinference/tree/main/python/instrumentation/openinference-instrumentation-guardrails
- OpenTelemetry GenAI events (`gen_ai.evaluation.result`): https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-events.md

SDK source, read at the installed versions:

- Langfuse `CallbackHandler`: https://github.com/langfuse/langfuse-python/blob/main/langfuse/langchain/CallbackHandler.py
- Langfuse attributes: https://github.com/langfuse/langfuse-python/blob/main/langfuse/_client/attributes.py
- Langfuse span filter: https://github.com/langfuse/langfuse-python/blob/main/langfuse/_client/span_filter.py
- Langfuse propagation: https://github.com/langfuse/langfuse-python/blob/main/langfuse/_client/propagation.py
- Langfuse client (`create_score`, `mask`, `mask_otel_spans`): https://github.com/langfuse/langfuse-python/blob/main/langfuse/_client/client.py
- Langfuse types (`MaskFunction`, `OtelSpanData`): https://github.com/langfuse/langfuse-python/blob/main/langfuse/types.py
- `LangChainTracer`: https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/tracers/langchain.py
- Base tracer run creation: https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/tracers/core.py
- `EvaluatorCallbackHandler`: https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/tracers/evaluation.py
- `astream_events` handler: https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/tracers/event_stream.py
- Callback managers and `_configure`: https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/callbacks/manager.py
- Runnable config tracing context: https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/runnables/config.py
- `create_agent` traceable wrappers: https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/factory.py
- Middleware `trace_policy`: https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/middleware/types.py
- LangSmith client (`create_feedback`, hiding, batching): https://github.com/langchain-ai/langsmith-sdk/blob/main/python/langsmith/client.py
- LangSmith `traceable` and OpenTelemetry context: https://github.com/langchain-ai/langsmith-sdk/blob/main/python/langsmith/run_helpers.py
- LangSmith `RunTree.from_runnable_config`: https://github.com/langchain-ai/langsmith-sdk/blob/main/python/langsmith/run_trees.py
- LangGraph `TAG_HIDDEN`: https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/constants.py
- LangGraph messages stream: https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/pregel/_messages.py
- Deep Agents subagent tracing context: https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/middleware/subagents.py
