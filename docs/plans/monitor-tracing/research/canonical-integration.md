# The monitor's decisions in LangSmith and Langfuse, built from their own concepts: research

This explanation asks how each tracing tool expects a guardrail's or an
evaluator's result to be recorded, checks each answer against the tools' docs,
SDK source and live services, and recommends how the library should present a
monitored step in LangSmith and Langfuse without workarounds.

Research date: 30 September 2026.

Contents:

- [Scope, method and versions](#scope-method-and-versions)
- [Summary](#summary)
- [Open items the earlier note left, now settled](#open-items-the-earlier-note-left-now-settled)
- [Scores and feedback](#scores-and-feedback)
- [Run and observation types](#run-and-observation-types)
- [Hiding and grouping internal calls](#hiding-and-grouping-internal-calls)
- [Stable ids and cross-links](#stable-ids-and-cross-links)
- [Filtering language](#filtering-language)
- [OpenTelemetry](#opentelemetry)
- [What comparable libraries do](#what-comparable-libraries-do)
- [LangSmith, Langfuse and OpenTelemetry side by side](#langsmith-langfuse-and-opentelemetry-side-by-side)
- [Recommended design](#recommended-design)
- [Options where the choice is open](#options-where-the-choice-is-open)
- [Before 0.1.0, and later](#before-010-and-later)
- [Still to check in the tools' interfaces](#still-to-check-in-the-tools-interfaces)
- [Sources](#sources)

## Scope, method and versions

This note builds on [langsmith-and-langfuse.md](langsmith-and-langfuse.md) and
[langchain-callbacks.md](langchain-callbacks.md), and does not repeat them. It
takes as settled their account of how each callback becomes a Langfuse
observation, how Langfuse stringifies metadata, how two tracers together split
a Langfuse trace, and the part 2 options A to D. It adds what the owner asked
after a check against real traces: each tool's own concept for a verdict, and
whether the library can reach it.

- **Read:** `langsmith` 0.14.1 [@langsmithsdk2026], `langfuse` 4.16.0
  [@langfuse2026], `langchain-core` 1.6.6 [@langchaincore2026], `langchain`
  1.4.3 [@langchain2026] and `langgraph` 1.2.12 [@langgraph2026], installed in
  a scratch environment with this branch's library. Langfuse's
  `CallbackHandler.py` is byte for byte the same in 4.15.6 and 4.16.0, so the
  earlier note's account of the handler still holds. The Langfuse server was
  read at `main` commit `0de9ccd` [@langfuseserver2026]. Documentation pages
  were read on 30 September 2026, as Markdown from the tools' own sites.
- **Live:** two probes against the real services, on scripted models, with
  synthetic content, under `invoke()` and `ainvoke()`. The agent looks up a
  made-up fact, posts to a made-up paste site (judged 0.9, flagged) and
  answers; a second agent's first step is halted. In LangSmith project
  `langchain-sync-monitors-docs-check`, the probe's root runs are named after
  `canon-408e439f`; in Langfuse, its traces are named after `canon-2f1364fd`.
  A first Langfuse attempt stopped part-way, after writing traces and
  `monitor_suspicion_during_run` scores whose name prefix was not recorded.
- **Spend:** no model provider was called. The LangSmith probe sent feedback
  with the SDK's default `extend_trace_retention=True`, so by LangSmith's rules
  [@langsmith2026retention] its 4 traces were probably moved to extended
  retention.
- **Marks:** "(live)" is a result read back from a service, "(offline)" a run
  with no service, and "(source)" a reading of SDK or server code. Anything
  else comes from the tools' documentation.

## Summary

1. **A verdict's number belongs in each tool's evaluation object.** LangSmith
   calls it feedback, where the scores from evaluation of a trace or a span are
   stored, and Langfuse calls it a score, which its docs name for guardrail
   results [@langsmith2026feedbackformat; @langfuse2026scores]. Only these
   objects are charted and averaged: LangSmith charts feedback, never metadata values
   [@langsmith2026dashboards], and Langfuse rejects numeric filters on metadata
   (live).
2. **Suspicion is the one number that needs it.** The outcome and the flag are
   known when the span is written. They already filter in both tools through
   tags and string metadata (live), and Langfuse's own rule of thumb puts what
   is known at tracing time in tags, not scores [@langfuse2026scores].
3. **LangSmith can take feedback from inside the run, exactly.** The step
   span's run id is ours, and the trace id is in the tracer's run map while the
   span is open (offline, live). There are two costs:
   - background sending needs `extend_trace_retention=True`, which moves every
     trace that receives feedback to extended, dearer retention
     [@langsmith2026retention];
   - with the extension off, the SDK sends each feedback synchronously, which
     took 0.24 to 1.11 seconds a call (live).
4. **Langfuse gives no public way to know an observation's id during a run.**
   - Under `ainvoke()` no observation is current where the library's code
     runs, and under `invoke()` the current one is whichever the handler
     started last (live).
   - The exact public routes run after ingestion: a Langfuse code evaluator
     on observations named `monitor step` [@langfuse2026codeevaluators], or a
     lookup by `monitor_step_id`, which found all 8 steps exactly (live).
5. **Neither tool's guardrail marker is reachable from callbacks.**
   - Langfuse has `guardrail` and `evaluator` observation types
     [@langfuse2026observationtypes], but its LangChain handler emits only
     `chain`, `agent`, `tool`, `retriever` and `generation` (source).
   - LangSmith has no guardrail run type at all [@langsmith2026querysyntax].
   - Keep `chain`.
6. **Hiding the monitor's own model calls needs a name.**
   - LangSmith cannot negate a tag (live).
   - Langfuse's handler puts `DEBUG` only on chain, tool and retriever runs,
     never on a generation (source, live).
   - A fixed run name on the monitor's calls, such as `monitor call`, makes
     them excludable by name in both tools (offline).
7. **OpenTelemetry offers no common encoding yet.** `gen_ai.evaluation.result`
   is a development-stage log event that neither tool turns into a score or
   feedback, and the guardrail conventions are still open proposals
   [@otel2026genai; @otel2026guardrailproposal].

## Open items the earlier note left, now settled

| Earlier open item | Result |
|---|---|
| Does `tree_filter` accept feedback fields? | Yes: `tree_filter='and(eq(feedback_key, "monitor_suspicion"), gte(feedback_score, 0.5))'` with `is_root=True` returned the 4 traces with a step judged 0.9 (live). |
| Do Langfuse's `numberObject` metadata filters match floats stored as strings? | No, they are refused: `Invalid filter type 'numberObject' for column 'metadata'. Expected filter type 'stringObject'.` (live), although the SDK's docstring for the observations API lists `numberObject` for `metadata` (source). |
| Can feedback be written while the step span is still open? | Yes: feedback sent from the decision span's start callback, before the step run had ended, was read back on the step run, with `feedback_stats` (live). |
| Is `get_current_observation_id()` a way to learn our span's id? | No. Under `invoke()` it returned the decision observation, the last one the handler started, whose parent is the step; under `ainvoke()` it returned `None` (live). The handler attaches each observation to the OpenTelemetry context of the thread its callback runs in (source). |
| Would server-side evaluators have to re-judge? | Not any more: code evaluators in both tools read a span's own outputs and write feedback or scores from them [@langsmith2026codeevaluators; @langfuse2026codeevaluators]. |

The earlier plan also advised `extend_trace_retention=False` for LangSmith
feedback. The SDK sends feedback in the background only when a `trace_id` is
given, the extension is on and LangSmith's OpenTelemetry mode is off; otherwise
it posts it at once, retrying up to ten times (source). That advice therefore
trades cost for latency, as the options below set out.

## Scores and feedback

### LangSmith feedback

- **Shape.** A feedback record has a `key`, a numeric `score`, a `value` for
  categorical labels, a `comment`, and a `feedback_source` whose type is `api`,
  `model` or `app` [@langsmith2026feedbackformat]. Feedback criteria are
  continuous, with a range, or categorical, where the label lands in `value`
  and its mapped number in `score` [@langsmith2026feedbackcriteria].
- **Target.** Feedback may go on any child run, not only the root, and the
  Python client sends it in the background when `trace_id` is given
  [@langsmith2026feedback]. `session_id`, the project's id, is required for
  run feedback; without it the SDK warns now and raises on SmithDB-only
  deployments (source). `create_feedback` first takes `session_id` in
  langsmith 0.6.7, while `langchain-core` 1.6.6 accepts langsmith from 0.3.45
  (source), so an in-process writer needs a floor or a feature check.
- **Written from inside the run.** A callback handler that saw the
  `monitor decision` span start read the step run's `trace_id` from
  `LangChainTracer.run_map` and wrote three feedback keys on the step run
  (live). Code inside a node finds the tracer among the callback manager's
  handlers when tracing is turned on by environment variables, under
  `invoke()` and `ainvoke()` (offline). This is the pattern langchain-core's
  own `EvaluatorCallbackHandler` follows, with `feedback_source_type` set to
  `model` and `source_run_id` naming the evaluator's run (source).
- **Cost and speed.** Feedback sent through the API with the extension on
  moves the whole trace to extended retention, which LangSmith says affects the
  bill [@langsmith2026retention]. The background path returned in under a
  millisecond; the synchronous path took 0.24 to 1.11 seconds per call over 8
  calls (live).
- **Filtering and charts.** `feedback_key` and `feedback_score` are documented
  filter fields, in `filter`, `trace_filter` and `tree_filter`
  [@langsmith2026querysyntax]. `eq(feedback_value, "halted")` also works
  (live) but is not documented. `has(feedback_key, ...)`, which the evaluator
  docs mention, is refused by the runs API (live). Dashboards chart the
  average, minimum and maximum of a feedback key, but offer metadata only as a
  filter or a group [@langsmith2026dashboards].
- **Privacy.** `create_feedback` applies none of the client's hiding or
  anonymising settings (source), so a comment holding the judge's reason would
  be an unmasked copy of it.

### Langfuse scores

- **Shape.** A score has a name, a value, a data type (`NUMERIC`,
  `CATEGORICAL`, `BOOLEAN` or `TEXT`) and an optional comment, and attaches to
  a trace, an observation, a session or a dataset run
  [@langfuse2026scores]. Langfuse names the SDK route as the way for
  "guardrail results" [@langfuse2026scores].
- **Scores or tags.** Langfuse's rule of thumb is to use a tag for what is
  known at tracing time, and a score to measure how good something is
  [@langfuse2026scores].
- **Written during the run.** The server accepted scores sent during the run
  to the exact observation (live), but the ids came from the handler's private
  run map. The documented ways to score a LangChain run are all trace-level: a
  wrapping span with a predefined trace id, `score_current_trace`, or
  `create_score` with a trace id [@langfuse2026langchain]. The docs warn that
  `last_trace_id` needs care when one handler serves concurrent requests
  [@langfuse2026langchain].
- **Written after the run.** Finding each `monitor step` observation by
  `monitor_step_id` returned the exact observation and trace ids for all 8
  steps (live), and scores created on them read back (live).
- **Server-side.** Code evaluators run Python or TypeScript in Langfuse on
  observations a rule selects, read the observation's input, output and
  metadata, and attach their scores to that observation; they exist on every
  plan, and on self-hosted instances once a dispatcher is configured
  [@langfuse2026codeevaluators]. A rule can select by `name` and by `type`,
  `CHAIN` included (source). Observation-level LLM-as-a-judge evaluators are
  the recommended target and trace-level ones are deprecated
  [@langfuse2026llmjudge].
- **Filtering.**
  - The trace list filters on scores (source, live): `scores_avg` with a score
    name as `key`, `score_booleans` and `score_categories`.
  - `scores_avg` with a key behaved as "some score of that name matches", not
    as an average: the traces with steps at 0.1 and 0.9 matched both `<= 0.2`
    and `>= 0.5`, and none matched `>= 0.95` (live).
  - The observations API has no score column (source).
  - The scores API filters by value only with `data_type` given, and refused
    `value_min` without it (live).
  - Score filters join the scores table and may be slow, by the API's own
    notes [@langfuse2026]. Two trace queries timed out without a name filter,
    and passed with one (live).
- **Privacy.** `create_score` applies no `mask` (source), so a score comment is
  an unmasked copy too. A server-side evaluator reads what was ingested, after
  the user's masking.

### Score or metadata

| Need | LangSmith today | Langfuse today | With a first-class suspicion |
|---|---|---|---|
| Halted or flagged steps | Tag filters on `monitor decision` (live) | String metadata filters on `monitor decision` (live) | Unchanged |
| Suspicion above a threshold | Typed metadata, `gte(metadata_value, 0.5)` (live) | Impossible: numeric metadata filters are refused (live) | Feedback or score filters in both |
| Suspicion charted or averaged | Impossible: charts aggregate feedback only [@langsmith2026dashboards] | Impossible without a score [@langfuse2026dashboards] | Both |
| Counts of halted or flagged steps | Count with a tag filter [@langsmith2026dashboards] | Observations counted with a metadata filter [@langfuse2026dashboards] | Unchanged |

So a score or feedback beside the metadata, for suspicion alone, closes every
gap the check against real traces found. Replacing the metadata would break the
filters the guide documents, and gain nothing.

## Run and observation types

- **LangSmith.** Run types are `llm`, `chain`, `tool`, `retriever`,
  `embedding`, `prompt` and `parser`; none is for a guardrail or an evaluator
  [@langsmith2026querysyntax; @langsmith2026otel]. LangSmith's own OpenAI
  Agents integration traces a guardrail span as a `tool` run with a
  `triggered` flag in metadata (source). Its advice for guardrails is
  `ls_agent_type: "middleware"` in metadata [@langsmith2026trajectory], which
  the library already sets.
- **Langfuse.** Observation types include `guardrail`, "a component that
  protects against malicious content or jailbreaks", and `evaluator`
  [@langfuse2026observationtypes]. Framework integrations set types
  automatically, and hand-written code sets them with `as_type`
  [@langfuse2026observationtypes].
- **What the LangChain handler can set.** The handler picks the type from the
  callback kind, and a chain whose name or class path contains "agent" becomes
  `agent`; no metadata key or tag changes it (source). The only routes to
  `guardrail` are the Langfuse SDK's own `start_observation(as_type=...)`, or an
  OpenTelemetry span with `langfuse.observation.type` or OpenInference's
  `openinference.span.kind` set to `GUARDRAIL` [@langfuseserver2026]. Either
  opens an observation outside the LangChain callback tree, so LangSmith would
  not see it and the two tools' trees would differ.
- **Conclusion.** Keep `chain` for all four spans. A `guardrail` observation is
  worth revisiting only if Langfuse's handler gains a hint for it.

## Hiding and grouping internal calls

- **LangSmith.**
  - The Trajectory view hides a run marked `ls_agent_type: "middleware"` or
    carrying `ls_message_view_exclude`, which is checked by presence
    [@langsmith2026metadataparameters]. Excluded runs still appear in the
    trace view, the runs table and metrics.
  - The documented `ls_` keys are `ls_provider`, `ls_model_name`,
    `ls_temperature`, `ls_max_tokens`, `ls_stop`, `ls_invocation_params`,
    `ls_agent_type`, `ls_message_view_exclude` and `ls_is_error_interrupt`,
    with `ls_run_depth` and `ls_method` set by the system
    [@langsmith2026metadataparameters]. None collapses a run in the trace tree.
  - The filter language, as the API takes it, has no negation of tags:
    `neq(tags, "monitor")` is refused as `comparator=NEQ attribute='tags' ...
    not accepted`, and `not(...)` does not parse (live). The interface's
    `is not` operator on tags was not tested.
  - What works is `neq(name, ...)` once per name, and
    `neq(metadata_key, "ls_message_view_exclude")`, which left out the
    monitor's calls (live).
- **Langfuse.**
  - `level` is `DEBUG`, `DEFAULT`, `WARNING` or `ERROR`, a trace can be
    filtered by level, and the LangChain integration sets it automatically
    [@langfuse2026levels].
  - The handler sets `DEBUG` only for a chain, tool or retriever run tagged
    `langsmith:hidden`, never for a model call (source). Live, a chain tagged
    `langsmith:hidden` became a `DEBUG` chain while its model call, tagged the
    same, stayed a `DEFAULT` generation.
  - The observations API and evaluation rules filter `name` with `none of`
    (source), and `none of` on the four span names returned everything else
    (live).
  - A metadata filter still cannot leave out observations that carry a key:
    `does not contain` on `ls_message_view_exclude` returned nothing (live).
  - At export, `should_export_span` can drop spans before they reach Langfuse
    [@langfuse2026]. That removes the calls and their cost data rather than
    hiding them.
- **A name is the one lever both tools share.** Langfuse asks for stable
  observation names, treated like an API, and warns against naming an
  observation after its model, since the model is already an attribute of a
  generation [@langfuse2026bestpractices]. A `run_name` in a chat model call's
  config reaches both handlers as the run's name, under `invoke()` and
  `ainvoke()` (offline). The chat monitors pass their config straight to the
  chat model. A decision model passes it to its classifier, whose run would take
  the name instead.
- **Rejected: `langsmith:hidden`.** It demotes nothing that matters here,
  because the monitor's calls are generations. It also changes LangGraph's
  streams, as the earlier note found.

## Stable ids and cross-links

- **LangSmith.** A run's id is the LangChain run id, and the library already
  generates the step span's id itself, so `monitor_step_id` is the step run's
  id. The trace id is the root run's id [@langsmithsdk2026]. Feedback targets
  that run id exactly.
- **Langfuse.**
  - Trace ids are 32 hex characters and observation ids 16, random by default
    [@langfuse2026traceids].
  - A trace id can be made deterministic with
    `Langfuse.create_trace_id(seed=...)` and handed to the handler as
    `CallbackHandler(trace_context={"trace_id": ...})`. Live, every
    observation of that run landed in the seeded trace.
  - No public route maps a LangChain run id to an observation id, and the
    client's `id_generator` sees no run id (source).
  - So `monitor_step_id` stays the join key. The earlier note's lookup by
    metadata is exact (live).
- **Sessions and users.** Both are the user's to set:
  - LangSmith filters a thread by `session_id` or `thread_id` metadata
    [@langsmith2026querysyntax];
  - Langfuse takes `langfuse_session_id` and `langfuse_user_id` from the root
    run's metadata [@langfuse2026langchain].

  The library needs neither.

## Filtering language

Every expression below was sent to the real API. "Accepted" means that it ran
and returned the expected runs.

| Expression | LangSmith |
|---|---|
| `and(eq(name, "monitor decision"), has(tags, "monitor:halted"))` | Accepted: 2 decisions |
| `and(eq(metadata_key, "monitor_flagged"), eq(metadata_value, true))` | Accepted: 4; the string `"true"` matched 0, so metadata keeps its type |
| `and(eq(metadata_key, "monitor_max_suspicion"), gte(metadata_value, 0.5))` | Accepted: 4 |
| `and(eq(feedback_key, "monitor_suspicion"), gte(feedback_score, 0.5))` | Accepted: 4 step runs; as `tree_filter` on roots, 4 traces |
| `and(eq(feedback_key, "monitor_suspicion"), gte(feedback_score, 0.5), eq(metadata_key, "monitor_name"), eq(metadata_value, "monitor"))` | Accepted: 4; with `"other"`, 0 |
| `and(eq(feedback_key, "monitor_outcome"), eq(feedback_value, "halted"))` | Accepted: 2, though `feedback_value` is undocumented |
| `in(name, ["monitor step", "monitor judgement", "monitor decision"])` | Accepted: 24 spans |
| `neq(name, ...)` three times, joined with `and(...)` | Accepted: every other run |
| `neq(tags, "monitor")` | Refused: `NEQ` on `tags` not accepted |
| `not(has(tags, "monitor"))`, `not(in(name, [...]))` | Refused: the filter does not parse |
| `has(metadata, "ls_message_view_exclude")` | Refused |
| `has(feedback_key, "monitor_suspicion")` | Refused: `HAS` on `feedback_key` not accepted |

| Condition | Langfuse |
|---|---|
| Observations: `stringObject` metadata `monitor_flagged` `=` `"true"` | Accepted: the flagged decisions |
| Observations: `numberObject` metadata `monitor_max_suspicion` `>=` 0.5 | Refused: metadata takes `stringObject` only, though the SDK's docstring lists `numberObject` |
| Observations: `stringOptions` `name` `none of` the span names | Accepted |
| Observations: `stringOptions` `level` `none of` `["DEBUG"]` | Accepted |
| Observations: `stringObject` metadata `ls_message_view_exclude` `does not contain` `"true"` | Accepted, returns nothing |
| Traces: `numberObject` `scores_avg` with key `monitor_suspicion`, `>=` 0.5 | Accepted: the 4 traces, with "some score matches" behaviour |
| Traces: `booleanObject` `score_booleans` with key `monitor_flagged` `=` true | Accepted: 4 traces |
| Traces: `categoryOptions` `score_categories` with key `monitor_outcome`, `any of` `["halted"]` | Accepted: 2 traces |
| Scores v3: `name`, `data_type="NUMERIC"`, `value_min=0.5` | Accepted; refused without `data_type` |

## OpenTelemetry

- **The conventions moved.** The GenAI conventions now live in their own
  repository; in `opentelemetry-semantic-conventions` 0.66b0 every `gen_ai`
  constant is marked as moved [@otel2026genai; @otel2026semconvpython].
- **Evaluation results.** `gen_ai.evaluation.result` is an event, a log record
  rather than a span, at development stability. It carries
  `gen_ai.evaluation.name` (required), `gen_ai.evaluation.score.value`,
  `gen_ai.evaluation.score.label` and `gen_ai.evaluation.explanation`, and
  should be parented to the span it evaluates [@otel2026genai].
- **Guardrails.** No guardrail convention is merged. One open proposal adds a
  `run_guardrail` operation with verdict attributes
  [@otel2026guardrailproposal]. Another adds a decision event for a proposed
  tool call, with `allow`, `deny` and `require_approval` outcomes
  [@otel2026tooldecisionproposal]; that is the nearest match to a step
  monitor.
- **Langfuse over OpenTelemetry.** `langfuse.observation.type` and
  `openinference.span.kind` set the observation type, `guardrail` included,
  and `langfuse.observation.level` sets the level. No attribute or event
  becomes a score, and the server has no logs endpoint that could receive
  `gen_ai.evaluation.result` [@langfuseserver2026; @langfuse2026otel].
- **LangSmith over OpenTelemetry.** `langsmith.span.kind` sets the run type,
  from the same seven values, and `langsmith.metadata.*` and
  `langsmith.span.tags` carry metadata and tags. The documented mapping
  mentions neither feedback nor `gen_ai.evaluation.*` [@langsmith2026otel].
- **Conclusion.** No encoding reaches first-class scores or feedback in either
  tool. Moving the spans to OpenTelemetry would also take them out of the
  LangChain callback tree that both tools already read. Revisit when a
  guardrail or decision convention merges and a tool maps it to scores or
  feedback.

## What comparable libraries do

- **LangChain's own middleware.** PII and human-in-the-loop middleware leave no
  score, feedback or verdict metadata. A PII block raises an error, and
  human-in-the-loop interrupts [@langchain2026].
- **LangSmith evaluators.**
  - `evaluate()` and langchain-core's `EvaluatorCallbackHandler` write feedback
    after the target run, with `feedback_source_type` set to `model` and
    `source_run_id` naming the evaluator's run [@langsmithsdk2026].
  - openevals returns a key, a score and a comment, and writes feedback itself
    only inside its pytest integration [@openevals2026].
  - Online evaluators, LLM-based or code, select runs with the same filter
    language and write feedback under their own name
    [@langsmith2026codeevaluators].
  - A rule whose item type is Runs evaluates each matching run as it arrives
    [@langsmith2026rules].
  - An evaluator can opt out of extending retention when the project's default
    retention is the base tier [@langsmith2026evaluatorretention].
- **Langfuse evaluators.** LLM-as-a-judge and code evaluators target
  observations and write scores after ingestion, with the reasoning as the
  comment [@langfuse2026llmjudge; @langfuse2026codeevaluators].
- **Langfuse's guardrail guidance.** Trace each check, then use scores to track
  and validate the security tools [@langfuse2026guardrails].
- **NeMo Guardrails 0.24.1.**
  - It emits OpenTelemetry spans with the rail's decision in attributes, and
    records the rail's reason only when content capture is on
    [@nemoguardrails2026].
  - Its LangSmith support is LangChain tracing itself [@nemoguardrails2026].
- **Guardrails AI 0.11.0.** It marks its guard and validator spans with
  OpenInference's `GUARDRAIL` kind, which Langfuse shows as `guardrail`
  observations, and puts the verdict in attributes [@guardrailsai2026].
- **OpenAI Agents SDK.** A guardrail span carries a `triggered` flag
  [@openaiagents2026]. LangSmith maps it to a `tool` run marked as
  middleware [@langsmithsdk2026], and OpenInference to a `GUARDRAIL` span
  [@openinference2026].

The pattern is shared:

- the verdict is written during the run as span data;
- scores and feedback come from evaluators, after the run, attached by id,
  with the reasoning as a comment;
- no library writes a guardrail's verdict as a score during the run.

The recommendation below follows that split.

## LangSmith, Langfuse and OpenTelemetry side by side

| Concern | LangSmith | Langfuse | OpenTelemetry |
|---|---|---|---|
| A verdict's number | Feedback: key, score, value, comment, source | Score: name, value, type, comment | `gen_ai.evaluation.result` event, development |
| Written from inside a LangChain run | Yes, exactly, through the tracer's client | Not exactly by public API; trace-level only when the user seeds the trace id | Not consumed by either tool |
| Written after the run | `create_feedback` or an online evaluator | `create_score` after a lookup, or a code or judge evaluator | Not applicable |
| Filters on it | `feedback_key`, `feedback_score` in every filter | Trace list and scores API; not the observations API | None |
| Charts | Average, minimum, maximum | Score analytics and dashboards | None |
| Cost side effect | Extended retention if written with the extension | None documented | None |
| Guardrail kind | None; `ls_agent_type: "middleware"` | `guardrail` type, not settable from callbacks | Open proposals only |
| Hiding internals | Trajectory view keys; `neq(name)` in tables | Level `DEBUG`, not reachable for generations; `name none of` in tables | Not applicable |
| Step id | Run id, chosen by the library | Observation id random; find by `monitor_step_id` | Span id chosen by the tracer |
| Filter on tags | `has` only, no negation | Child tags live in metadata `tags` | Not applicable |
| Metadata types | Kept: numbers and booleans compare as such | Strings only | Typed attributes |

## Recommended design

### What to emit

1. **Keep the spans as they are.** The four span names, their outputs, the
   `monitor_` metadata keys, the decision span's tags and the Trajectory view
   keys stay. They are the record every tracer reads, and every one-line query
   for outcomes and flags already works on them.
2. **Name the monitor's own model calls `monitor call`.** This is a
   `run_name` in the config the monitors already build for their calls. It is
   the only callback-level change that makes the calls excludable in both
   tools, and it follows Langfuse's advice not to name an observation after its
   model. The model stays visible as the generation's model and in
   `ls_model_name`.
3. **Make suspicion a first-class number, beside the metadata.** Emit one
   numeric value per step, `<label>_suspicion`, which is `monitor_suspicion`
   under the default label. Its value is the step's highest suspicion, from 0
   to 1, on the `monitor step` run or observation. A step with no sample
   writes none. The outcome and the flag stay in tags and metadata.

### Where and when

- **LangSmith: during the run, from the library, when the user opts in.**
  - At decision time, find the `LangChainTracer` among the step span's
    handlers.
  - Write feedback on `run_id=monitor_step_id`, with `trace_id` read from the
    tracer's run map while the step span is open.
  - Set `session_id` to the project's id, resolved once per client and project
    off the event loop.
  - Set `feedback_source_type="model"`, and
    `feedback_id=uuid5(monitor_step_id, key)` so that a retry never
    duplicates.
  - Retention and latency are an open choice, D3 below.
- **Langfuse: after ingestion, by a Langfuse code evaluator.**
  - Its rule selects observations named `monitor step` of type `CHAIN`.
  - Its evaluator returns `<label>_suspicion` as a numeric score, read from
    `output.max_suspicion` and `metadata.monitor_name`, and skips a `None`.
  - The library's docs would ship the evaluator's code and the rule's
    filter. It is exact, public, free of latency, and reads the data after the
    user's masking.
- **Both: the same evaluator as a recipe for LangSmith users who prefer
  server-side scoring**, with the evaluator's retention toggle stated.

### Fields and types

| Field | LangSmith | Langfuse |
|---|---|---|
| Name | feedback `key` `<label>_suspicion` | score `name` `<label>_suspicion` |
| Value | `score`, a float from 0 to 1 | `value`, `data_type="NUMERIC"` |
| Target | the `monitor step` run, `run_id=monitor_step_id` | the `monitor step` observation |
| Source | `feedback_source_type="model"` | the evaluator |
| Idempotency | `feedback_id=uuid5(monitor_step_id, key)` | the rule runs when a matching observation arrives [@langfuse2026codeevaluators] |
| Comment | none by default, D6 | none by default, D6 |

### What users configure

- **Spans and the call name:** nothing.
- **LangSmith feedback:** one option on `MonitorMiddleware`, off by default.
  The exact shape follows D3.
- **Langfuse scores:** a one-time code evaluator and rule, pasted from the
  docs.

### The one-line queries

| To find | LangSmith | Langfuse |
|---|---|---|
| Halted steps | `and(eq(name, "monitor decision"), has(tags, "monitor:halted"))` | Observations named `monitor decision` with metadata `monitor_outcome` = `halted` |
| Flagged steps | `and(eq(name, "monitor decision"), has(tags, "monitor:flagged"))` | Observations named `monitor decision` with metadata `monitor_flagged` = `"true"` |
| Suspicion at least 0.5 for label `L` | `and(eq(feedback_key, "L_suspicion"), gte(feedback_score, 0.5))` | Scores named `L_suspicion` with `data_type="NUMERIC"` and `value_min=0.5`; for traces, `scores_avg` with key `L_suspicion` at least 0.5 |
| Everything but the monitor's internals | `and(neq(name, "monitor step"), neq(name, "monitor judgement"), neq(name, "monitor classifier"), neq(name, "monitor decision"), neq(name, "monitor call"))` | Name `none of` the same five names |

Each Langfuse score query should carry a time window, as the live timeouts
showed.

## Options where the choice is open

**D1. Which numbers become first-class.**

| Option | For | Against |
|---|---|---|
| Suspicion only (recommended) | Closes every measured gap; one write per step; follows Langfuse's rule that known categories are tags | Outcome counts stay tag and metadata filters |
| Suspicion, flag and outcome | Score analytics on all three; trace-level Langfuse filters for flags and outcomes | Three writes per step in LangSmith, and a second copy of what tags already hold |

**D2. The route for each tool.**

| Option | For | Against |
|---|---|---|
| In-process for LangSmith, code evaluator for Langfuse (recommended) | Each tool's exact, public route; nothing to set up for LangSmith | Two mechanisms to document |
| Code evaluators in both | No network call from the library; masking applies; retention opt-out per evaluator | Set-up per project; runs after ingestion, so feedback lags |
| In-process in both | No set-up | Langfuse ids are exact only through private state or a user-seeded trace id; otherwise scores can land on another request's trace |
| A post-hoc exporter in a `langfuse` extra | Exact through public APIs; no server feature needed | Needs a scheduled job and ingestion delays; more code to own |
| A helper that installs the evaluator and its rule through the public API | One call instead of a pasted recipe: Langfuse has evaluator and rule endpoints in its SDK (source), and LangSmith's `/api/v1/runs/rules` takes `code_evaluators` and `extend_evaluator_trace_retention` | Writes persistent project configuration; needs a key allowed to; LangSmith's endpoint is outside the SDK's typed client |

**D3. LangSmith retention against latency, for the in-process writer.**

| Option | For | Against |
|---|---|---|
| No extension, sent from one background worker thread (recommended) | The bill does not change; the agent never waits | The library owns a worker and must drain it at exit, as `EvaluatorCallbackHandler` does with its executor |
| The SDK's background batching, with the extension | No thread of the library's own; no latency | Every monitored trace moves to extended retention |
| No extension, sent inline | Simple | 0.24 to 1.11 seconds per write on the agent's path, measured |
| Extension only for flagged or halted steps | Keeps the traces an auditor wants, in line with LangSmith's reasons for upgrades [@langsmith2026retention] | Unflagged steps get no feedback, so suspicion charts are partial |

**D4. Keys per monitor label.** A label-derived name, `<label>_suspicion`, is
recommended. LangSmith names feedback after the evaluator that wrote it
[@langsmith2026codeevaluators], and Langfuse's score APIs filter by name only.
A fixed `monitor_suspicion` with the label in metadata works in one line in
LangSmith (live), but in Langfuse it needs a second query.

**D5. The fixed name for the monitor's calls, and when.** Recommended before
0.1.0, because names become an API once released. It is small: a `run_name`
where the chat monitors build their call config, with tests under `invoke()`
and `ainvoke()`, and a sentence in the tracing guide, which now says the calls
are named after their chat model. A decision model passes its config to its
classifier, so that path needs its own check. The alternative is to keep
model-class names and document that the calls can be hidden only in LangSmith.

**D6. The judge's reason as a comment.** Recommended off by default:
neither SDK masks feedback or score comments (source), and the reason already
sits in the step span's outputs, where the user's masking applies. A code
evaluator may copy it into the comment, since it reads the data after masking.

**D7. The target run.** The `monitor step` span is recommended: its id is ours
in LangSmith, its lookup is exact in Langfuse, and its outputs hold the whole
decision, which is all a Langfuse evaluator can see [@langfuse2026llmjudge].
Per-sample suspicion on each `monitor judgement` span could follow later. The
agent's own model call, the run a classic evaluator would target, has an id
the library never learns.

## Before 0.1.0, and later

| Recommendation | When | Risk |
|---|---|---|
| Add to the tracing guide on `docs/release-pass`: `in(name, [...])` finds every monitor span in one LangSmith filter; Langfuse's observations API takes `none of` on names as its interface does; Langfuse's trace tree cannot hide the monitor's calls, since its handler never sets `DEBUG` on a model call | Before 0.1.0 | Low: documentation of verified behaviour |
| Name the monitor's calls `monitor call` (D5) | Before 0.1.0 if the owner agrees | Low: one config key and tests, but an emitted name changes |
| Settle D1, D4 and D7, so the names are fixed before release even if nothing writes them yet | Before 0.1.0 | None: decisions only |
| Code evaluator recipes for Langfuse and LangSmith | After 0.1.0, once checked in each tool's interface | Low, but unverified |
| In-process LangSmith feedback option (D3) | Later | Medium: a new network call, a worker, a langsmith floor of 0.6.7 |
| Post-hoc Langfuse exporter | Later, only if users ask | Medium: new extra and retries |
| A `guardrail` observation type in Langfuse | Only if Langfuse's handler gains a hint | Not actionable today |
| OpenTelemetry encoding | When a convention merges and a tool maps it | Not actionable today |

## Still to check in the tools' interfaces

These could not be read from docs, source or APIs without creating persistent
configuration in the owner's accounts:

- whether a LangSmith code evaluator can return a comment or a string value,
  and whether its filter selects child runs by name in practice;
- how a Langfuse code evaluator reports a `None` it skips;
- what `ctx.observation.output` holds for a LangChain `CHAIN` observation, an
  object or a JSON string, and whether its metadata values arrive as strings,
  as the handler stores them; the recipe depends on both;
- whether Langfuse's trace view, with a minimum level set, hides a `DEBUG`
  span's `DEFAULT` children.

## Sources

Each key is in `docs/references.bib`. Documentation pages were read on
30 September 2026.

| Key | What was read | Version or commit |
|---|---|---|
| `langsmithsdk2026` | `Client.create_feedback`, `LangChainTracer`, the OpenAI Agents integration | langsmith 0.14.1 |
| `langchaincore2026` | `LangChainTracer`, `EvaluatorCallbackHandler` | langchain-core 1.6.6 |
| `langfuse2026` | `CallbackHandler`, `create_score`, `create_trace_id`, the API client's filter docs | langfuse 4.16.0 |
| `langfuseserver2026` | The OpenTelemetry observation type mapper and level mapping | `main` at `0de9ccd` |
| `langchain2026` | PII and human-in-the-loop middleware | langchain 1.4.3 |
| `langgraph2026` | The hidden tag's effect on streams, via the earlier note | langgraph 1.2.12 |
| `langsmith2026feedback` | https://docs.langchain.com/langsmith/attach-user-feedback | 30 September 2026 |
| `langsmith2026feedbackformat` | https://docs.langchain.com/langsmith/feedback-data-format | 30 September 2026 |
| `langsmith2026feedbackcriteria` | https://docs.langchain.com/langsmith/set-up-feedback-criteria | 30 September 2026 |
| `langsmith2026querysyntax` | https://docs.langchain.com/langsmith/trace-query-syntax | 30 September 2026 |
| `langsmith2026metadataparameters` | https://docs.langchain.com/langsmith/ls-metadata-parameters | 30 September 2026 |
| `langsmith2026retention` | https://docs.langchain.com/langsmith/administration-overview | 30 September 2026 |
| `langsmith2026dashboards` | https://docs.langchain.com/langsmith/dashboards | 30 September 2026 |
| `langsmith2026codeevaluators` | https://docs.langchain.com/langsmith/online-evaluations-code | 30 September 2026 |
| `langsmith2026rules` | https://docs.langchain.com/langsmith/rules | 30 September 2026 |
| `langsmith2026evaluatorretention` | https://docs.langchain.com/langsmith/evaluators | 30 September 2026 |
| `langsmith2026otel` | https://docs.langchain.com/langsmith/trace-with-opentelemetry | docs repo `3af9965` |
| `langsmith2026trajectory` | https://docs.langchain.com/langsmith/trajectory-view-integrations | 30 September 2026 |
| `langfuse2026scores` | https://langfuse.com/docs/evaluation/scores/overview | 30 September 2026 |
| `langfuse2026langchain` | https://langfuse.com/integrations/frameworks/langchain | 30 September 2026 |
| `langfuse2026observationtypes` | https://langfuse.com/docs/observability/features/observation-types | 30 September 2026 |
| `langfuse2026levels` | https://langfuse.com/docs/observability/features/log-levels | 30 September 2026 |
| `langfuse2026traceids` | https://langfuse.com/docs/observability/features/trace-ids-and-distributed-tracing | 30 September 2026 |
| `langfuse2026bestpractices` | https://langfuse.com/docs/observability/best-practices | 30 September 2026 |
| `langfuse2026codeevaluators` | https://langfuse.com/docs/evaluation/evaluation-methods/code-evaluators | 30 September 2026 |
| `langfuse2026llmjudge` | https://langfuse.com/docs/evaluation/evaluation-methods/llm-as-a-judge | 30 September 2026 |
| `langfuse2026dashboards` | https://langfuse.com/docs/metrics/features/custom-dashboards | 30 September 2026 |
| `langfuse2026guardrails` | https://langfuse.com/docs/security-and-guardrails | 30 September 2026 |
| `langfuse2026otel` | https://langfuse.com/integrations/native/opentelemetry | docs repo `a5050cb` |
| `otel2026genai` | https://github.com/open-telemetry/semantic-conventions-genai, `model/gen-ai/events.yaml` and `registry.yaml` | `bcc7f9c` |
| `otel2026semconvpython` | `opentelemetry.semconv._incubating.attributes.gen_ai_attributes` | opentelemetry-semantic-conventions 0.66b0 |
| `otel2026guardrailproposal` | https://github.com/open-telemetry/semantic-conventions-genai/pull/427 | open, head `fffe24a` |
| `otel2026tooldecisionproposal` | https://github.com/open-telemetry/semantic-conventions-genai/pull/535 | open, head `d7a9623` |
| `openevals2026` | `openevals/utils.py` | openevals 0.2.0 |
| `nemoguardrails2026` | `tracing/constants.py` and the LangChain integration page | nemoguardrails 0.24.1 |
| `guardrailsai2026` | `telemetry/guard_tracing.py`, `telemetry/validator_tracing.py` | guardrails-ai 0.11.0 |
| `openaiagents2026` | `tracing/span_data.py`, `GuardrailSpanData` | openai-agents 0.22.3 |
| `openinference2026` | The OpenAI Agents instrumentation's span kinds | openinference-instrumentation-openai-agents 2.5.0 |
