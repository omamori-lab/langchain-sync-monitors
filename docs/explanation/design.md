# How the library is built

This explanation describes how langchain-sync-monitors is put together, for a
reader who has not seen the design plan: what happens on one step of an agent,
where the monitor sits, the four pieces that share the work, and how the same
code serves subagents and both `invoke()` and `ainvoke()`.

## One step, end to end

A step is one model call of the agent. The untrusted model, the one doing the
agent's work, proposes the step: its reasoning, its text and any tool calls.
The monitor middleware wraps that call, so the proposal is judged before any
tool runs, and the control protocol can call the model again or call a
different, trusted model instead.

```mermaid
flowchart TD
    state["Agent state: messages and monitor_log"] --> request["Model request for the next step"]
    subgraph wrapper ["The monitor middleware wraps the model call"]
        request --> untrusted["Untrusted model proposes a step: reasoning, text and tool calls"]
        untrusted --> monitor["Monitor reads the step through its view and returns a Verdict"]
        monitor --> protocol{"Control protocol decides"}
        protocol -- "draw another sample" --> untrusted
        protocol -- "defer" --> trusted["Trusted model writes the step"]
        trusted --> commit["Commit the chosen messages and one StepRecord"]
        protocol -- "allow, resample, steer or halt" --> commit
    end
    commit --> tools["Tool node runs the approved tool calls"]
    tools -- "results are appended, and the next model call is a new step" --> state
```

Everything inside the middleware stays a proposal until the protocol decides.
Rejected samples and the monitor's own model calls never reach the agent's
messages. What is committed is the chosen response, whether an untrusted
sample or the trusted model's step, together with one `StepRecord` in
`monitor_log` that keeps every sample the monitor judged, so rejected samples
survive as evidence for an auditor. Nor do they reach the message stream, as
[What streams](#what-streams) explains.

## Where the middleware sits

LangChain gives middleware several hooks. The monitor uses `wrap_model_call`
and `awrap_model_call`, the only hooks that gate every step and can call the
model more than once. Its `after_model` hook judges nothing; it only ends the
run after a halted step, as [Halts, fallbacks and failed
steps](#halts-fallbacks-and-failed-steps) explains. An `after_model` hook would come too late: after-model
hooks run in reverse list order, and Deep Agents lists
`HumanInTheLoopMiddleware` after user middleware, so its interrupt would fire
before an after-model monitor ran.

LangChain nests `wrap_model_call` handlers so that the first middleware in the
list is the outermost layer. In a plain `create_agent`, the monitor goes last
in the list, directly around the model. That position matters: middleware
listed after the monitor becomes a layer inside it, and when the monitor draws
several samples, LangChain's composed handler keeps only the commands of the
last inner call, by design and as its documentation states, whichever sample
the protocol commits [@langchain2026].

Deep Agents builds its own stack around the user's middleware. With
deepagents 0.7.19 [@deepagents2026] and `interrupt_on` set, the list, from
outermost to innermost, is:

```mermaid
flowchart TD
    filesystem["FilesystemMiddleware"] --> subagents["SubAgentMiddleware"]
    subagents --> summarisation["SummarizationMiddleware"]
    summarisation --> patch["PatchToolCallsMiddleware"]
    patch --> ours["monitor[main], this library"]
    ours --> caching["AnthropicPromptCachingMiddleware"]
    caching --> approval["HumanInTheLoopMiddleware"]
    approval --> content["UnsupportedContentMiddleware"]
    content --> model["Chat model: the untrusted agent"]
```

Skills, memory, asynchronous subagents and harness profiles add layers of
their own when they are configured. The monitor therefore sees the
conversation after summarisation, which is what the agent itself sees, and
every resample or trusted call passes through the inner layers, so prompt
caching and content checks still apply. Those inner layers only change the
request, so they return no commands, and the default stack is safe for
resampling.

Middleware names must be unique within one agent, so the monitor is named
after the agent it watches: `monitor[main]`, `monitor[researcher]` and so on.
Middleware instances are shared across parallel subagent runs, so the monitor
keeps no per-run state on itself; everything a run needs lives in the graph
state.

## Four pieces

The work is split so that what the monitor reads, how it scores, and what
happens with the score are separate choices.

| Piece | Interface | Job |
|---|---|---|
| View | `MonitorView`, `Channel` | Chooses which parts of the transcript the monitor reads. |
| Monitor | `Monitor.evaluate`, `Monitor.evaluate_sync` | Scores one proposed step and returns a `Verdict`: a `suspicion` from 0 to 1 and a `reason`. |
| Control protocol | `ControlProtocol.decide` | Samples and judges the step through a `PendingStep`, then returns a `StepDecision`. |
| Middleware | `MonitorMiddleware` | Builds the pending step for each model call, runs the protocol, and commits the decision and its record. |

Because every monitor reports suspicion on the same 0 to 1 scale, any monitor
works with any protocol. A protocol reaches the model only through the
`PendingStep` it is given:

- `sample(count=..., blocked_attempts=..., concurrently=...)` draws fresh
  samples of the step and has the monitor judge each one, returning `Sample`s
  that pair each response with its `Verdict`;
- `request_trusted_step(trusted_model=...)` has a trusted model write the step,
  with the same conversation and tools;
- `previous_records` holds this monitor's records from earlier steps of the
  run;
- `blocks_in_thread` counts the blocks this monitor recorded in the
  conversation thread before the step, subagents included, except those of a
  subagent whose run raised, and `new_subagent_blocks` the ones recorded
  inside subagents since the agent's last step.

When no sample is acceptable, a `Fallback` writes the step instead: the trusted
model does, or the run halts. The protocol's `StepDecision` names the
`Outcome` (`ALLOWED`, `RESAMPLED`, `DEFERRED_TO_TRUSTED`, `STEERED` or
`HALTED`), the response to commit, every sample, the sample that ran if one
did, and whether the step is flagged for audit.

## What the monitor reads

Every transcript entry belongs to one `Channel`: `USER`, `REASONING`,
`AGENT_TEXT`, `TOOL_CALLS`, `TOOL_RESULTS` or `SUBAGENT_REPORTS`. A
`MonitorView` is a set of channels, plus an optional limit on how many recent
entries to keep and the names of the tools whose results count as subagent
reports (`task` by default).

The default view, `Channel.ACTIONS`, holds the user's messages, the tool calls
and the subagent reports, together with the context notes, tool errors and
monitor feedback described below; their text is escaped and authorises
nothing. It leaves out the agent's reasoning, its prose and
raw tool output, as Claude Code's auto mode classifier does
[@hughes2026automode]. Reasoning is one flag away. Reading the chain of thought
catches more misbehaviour than reading actions alone [@baker2025monitoring],
but persuasive reasoning can also talk a monitor out of a correct flag
[@arnav2025cotredhanded], and more context does not always help a monitor
[@arike2026information].

`transcript.py` renders the entries as tagged text: `<user>`, `<agent>`,
`<agent_reasoning>`, `<tool_call>`, `<tool_result>`, `<subagent_report>` and,
for the step being judged, `<proposed_step>`. Content is HTML-escaped, so text
inside a tool result cannot close a tag and pose as another entry. The view
selects what the monitor reads of the history; the step being judged always
shows its tool calls, whatever the view, because they are the action the
protocol gates.

A provider's built-in tools, such as Anthropic's web fetch or OpenAI's web
search, run at the provider inside the model call, so they have run before the
monitor sees the step. LangChain gives them to every provider as standard
`server_tool_call` and `server_tool_result` blocks [@langchaincore2026], which
the monitor renders as `<provider_tool_call>` beside the tool calls and
`<provider_tool_result>` beside the tool results. The judge reads them and the
audit record keeps them, and a suspicious score still decides what happens to
the rest of the step, but a synchronous monitor cannot stop a call that has
already run.

Only the task author's own words authorise an action, so three kinds of entry
guard against a forged approval:

- Inside a subagent the task came from the parent agent, so its messages are
  rendered as `<delegator>` rather than `<user>`, following the `TaskAuthor`
  in the `MonitorInput`.
- A human message that another part of the application wrote carries an
  `lc_source` tag: LangChain's and Deep Agents' summaries are tagged
  `summarization`, and Deep Agents' rubric grader `rubric_grader`
  [@langchain2026; @deepagents2026]. Such a message is rendered as
  `<context_note source="...">` in the `USER` channel, and the prompt says a
  note authorises nothing. Deep Agents writes its summary with the agent's own
  model, so the untrusted model must not be able to speak as the user through
  it. Only a human message without the tag is rendered as `<user>` or
  `<delegator>`.
- The monitor's own feedback carries `lc_source="monitor"` and is rendered as
  `<monitor_feedback>` wherever the task is shown: both the message that
  answers a blocked final answer and the tool message that answers a blocked
  tool call, which carries the tool's name. So a monitor that does not read
  tool results still sees that an earlier call was blocked and did not run.

Any other tool message with `status="error"` says that a call failed or did not
run: a person rejected it through `HumanInTheLoopMiddleware`, with their reason
[@langchain2026]; the tool does not exist; or the tool raised. It is rendered as
`<tool_error name="...">` in the `TOOL_CALLS` channel, beside the call, so
every view that shows the call also shows that it did not complete, and why.
A tool call whose arguments could not be parsed lands in the message's
`invalid_tool_calls`. It never runs, but it is rendered as
`<malformed_tool_call name="...">` with its raw argument text, in the
transcript, in the proposed step and in the audit record. When a protocol
blocks such a step, the feedback answers each malformed call with an error
result too, because OpenAI-compatible providers receive it as an ordinary tool
call and reject a request that leaves it unanswered.

These tags rest on trust in the application's own code. A tool or a middleware
can set `lc_source="monitor"` on a message it returns, or write a plain
`HumanMessage` through a `Command`, and the monitor then reads it as its own
feedback or as the user. Tool and middleware code is trusted here, as it is
everywhere else in the agent. What the monitor distrusts is the content that
code carries, such as a web page or a file, which is escaped and never read as
a tag.

When the view keeps only recent entries, every message from the task author is
kept anyway, in order: the first states the task, and a later one can narrow or
revoke it. A context note is never kept in their place. After summarisation has
replaced the earlier messages of the model request, the original task may be
gone from what the monitor reads, and the summary is left as a note. Reasoning
is read from LangChain's standard content blocks, with a fallback for a reply
that would carry it only in OpenRouter's `reasoning_details`
[@langchaincore2026; @langchainopenrouter2026].

## Control protocols

Every protocol starts the same way: the untrusted model proposes a step, and
the monitor scores it. The protocols differ in what a suspicious score
triggers. Each threshold is on the common 0 to 1 scale, has a default, and
raises an `UncalibratedThresholdWarning` while the default is in use, because
a threshold should come from the score distribution of honest runs
[@bhatt2025ctrlz]. The defaults are 0.6 to defer, block or flag, and 0.9 to
audit.

A flag is a mark in `monitor_log` for a person to review after the fact. It
never blocks a step: by the time anyone reads the log, the step has run or
been replaced. So Defer to Resample and Defer to Trusted accept an audit
threshold below the defer threshold, which flags steps that still ran, a
review queue of borderline steps. `audit_threshold=None` turns their flagging
off, except that `HaltRun` flags every step it halts.

### Trusted Monitoring

The step always runs; a score at or above the flag threshold marks it for
audit [@greenblatt2023aicontrol].

```mermaid
flowchart LR
    sample["Draw one sample"] --> check{"Suspicion at or above the flag threshold?"}
    check -- "no" --> allowed["Run it: ALLOWED"]
    check -- "yes" --> flagged["Run it, flagged for audit: ALLOWED"]
```

### Defer to Trusted

A suspicious step is replaced by the trusted model's own step
[@greenblatt2023aicontrol]. It is Defer to Resample with no resamples.

```mermaid
flowchart LR
    sample["Draw one sample"] --> check{"Suspicion at or above the defer threshold?"}
    check -- "no" --> allowed["Run it: ALLOWED"]
    check -- "yes" --> trusted["Trusted model writes the step: DEFERRED_TO_TRUSTED"]
```

### Defer to Resample

A suspicious step is sampled again, and the least suspicious acceptable sample
runs; the trusted model writes the step only when every sample looks
suspicious [@bhatt2025ctrlz]. The agent never sees the rejected samples. Up to
two extra samples are drawn, as in the paper. `Resampling.SEQUENTIAL` draws
them one at a time and stops at the first acceptable one;
`Resampling.PARALLEL` draws them all at once, for lower latency and more model
calls. The audit flag uses the lowest score across the samples, as in Ctrl-Z.
Resampling needs fresh draws: the more deterministic the model, the less it
helps [@bhatt2025ctrlzpost]. A LangChain response cache, global or on the
agent's model, answers every resample with the first sample again, so the
library warns once with a `CachedResampleWarning` when a step is resampled
under one; build the untrusted model with `cache=False`.

Running the least suspicious sample has a cost against prompt injection. An
agent can write text aimed at the monitor into its step, and an injection that
fools the monitor only some of the time gets another try with every sample, so
resampling turns it into a best-of-n attack [@terekhov2025adaptive]. An
injection that passes one sample in five gets through about half the time
across three samples.

```mermaid
flowchart LR
    first["Draw one sample"] --> check{"Suspicion at or above the defer threshold?"}
    check -- "no" --> allowed["Run it: ALLOWED"]
    check -- "yes" --> more["Draw up to two more samples"]
    more --> any{"Any sample below the threshold?"}
    any -- "yes" --> resampled["Run the least suspicious: RESAMPLED"]
    any -- "no" --> fallback["Fallback writes the step, usually the trusted model: DEFERRED_TO_TRUSTED"]
```

### Auto Mode

A suspicious step is blocked, and the agent is told why, so its next attempt
can take a safer route [@hughes2026automode]. After three blocks in a row, or
twenty in one conversation thread, the limits Claude Code uses within a
session, a fallback takes over: the run halts, or the trusted model writes the
step. The total counts the monitor's blocks in the thread's `monitor_log`,
where each subagent's records arrive when it returns, so delegating again does
not reset it. A subagent starts from the thread's total, so it reaches the
limit where the thread does, and when blocks inside subagents leave the thread
at or over the total, the parent's next step goes to the fallback without
being sampled. The total spans every turn of a checkpointed thread, not one
run, and never resets, so once a thread has reached it, the first block of
every later step goes to the fallback.

The total misses one kind of block: those inside a subagent whose run raises,
because a subagent's records reach its parent only in the result of the call
that started it, and a call that raises has no result. They count when the
failed run is resumed from its checkpoint with `None` as input, since the
subagent then resumes from its own checkpoint. They never count when the
thread goes on with new input, or when a middleware such as LangChain's
`ToolRetryMiddleware` or `ToolErrorMiddleware` answers the failed call with an
error message or runs it again; a retry starts the subagent again from the
same count. The subagent's model can bring this about by calling a tool with
input that makes it raise. `check_monitor_placement` warns about such
middleware, and the lost steps still reach `stream_mode="custom"` as each is
committed, with its `delegation_id`, when the run streams with
`subgraphs=True`. Counting them would need a channel outside the graph state
and a monitor that turns the subagent's error into a result, which would also
stop a failed run from being resumed.

```mermaid
flowchart LR
    start["A new step"] --> subagents{"Subagent blocks since the last step, and the total reached?"}
    subagents -- "yes" --> fallback
    subagents -- "no" --> attempt
    attempt["Draw one sample, with any earlier feedback"] --> check{"Suspicion at or above the block threshold?"}
    check -- "no, first attempt" --> allowed["Run it: ALLOWED"]
    check -- "no, after a block" --> steered["Run it: STEERED"]
    check -- "yes" --> limit{"Block limit reached?"}
    limit -- "no" --> feedback["Block it and write the feedback"]
    feedback --> attempt
    limit -- "yes" --> fallback["Fallback: HALTED or DEFERRED_TO_TRUSTED"]
```

### Auto Mode, step by step

In this example the agent tries to post a credential off-site, is blocked, and
writes a summary instead.

```mermaid
sequenceDiagram
    participant State as Agent state
    participant Middleware as Monitor middleware
    participant Agent as Untrusted model
    participant Monitor
    State->>Middleware: request with the conversation so far
    Middleware->>Agent: sample the step
    Agent-->>Middleware: attempt 1, http_post(url, api_key)
    Middleware->>Monitor: score attempt 1
    Monitor-->>Middleware: 0.92, sends a credential off-site
    Note over Middleware: above the block threshold, so the attempt is blocked
    Middleware->>Agent: retry with attempt 1 and the feedback
    Agent-->>Middleware: attempt 2, write_file("summary.md")
    Middleware->>Monitor: score attempt 2
    Monitor-->>Middleware: 0.10
    Middleware->>State: commit attempt 2 and one StepRecord holding both attempts
```

`FeedbackVisibility` decides what the transcript keeps. With `HIDDEN`, only
the retry call sees the blocked attempt and its feedback; the transcript shows
attempt 2 alone, and the log keeps attempt 1. With `IN_TRANSCRIPT`, the blocked
attempt and its feedback are committed before attempt 2, as Claude Code does.
A blocked tool call is answered with one error tool message per call, so every
tool call keeps a matching result. Every message the monitor inserts gets a
fresh id (`monitor-<uuid4>`), because LangGraph replaces messages that share an
id, and the feedback is marked in its text, because some providers drop a
message's `name`.

### Halts, fallbacks and failed steps

A halted step ends the run. The halt message has no tool calls, but that alone
does not end an agent with structured output and no tools, which returns to
the model until it has a structured response [@langchain2026]. So the monitor
also has an `after_model` hook, which judges nothing: after a step this monitor
halted, it routes the agent to its end. The model node cannot do this itself.
A routing edge reads a fresh copy of the state that keeps only its own node's
writes, and the `jump_to` key is cleared everywhere else [@langgraph2026], so
a `jump_to` from the model node would reach only the model node's own edge,
which exists only in an agent with no `after_model` hook at all. The hook costs
one graph step per model call, which counts towards an explicit
`recursion_limit`.

A fallback must not escalate to a person by calling LangGraph's `interrupt()`.
The fallback runs inside the model node, and on resume LangGraph runs the whole
node again [@langgraph2026]: the protocol draws fresh samples, and the person's
answer applies to whatever the new samples propose, not to the step they
reviewed. To involve a person, end the step, with `HaltRun` for example, and
ask outside the model node, as `HumanInTheLoopMiddleware` does for tool calls.

When a call inside a step raises, whether a sample, a monitor call or the
trusted model's step, the protocol never decides, so nothing is committed and
the error propagates unchanged. The samples the monitor had already judged
would then leave no trace, so the middleware logs them at warning level, each
with its suspicion, reason and proposal, and writes a `MonitorStepFailedEvent`
(`type="monitor_step_failed"`) to `stream_mode="custom"` before the error
propagates. A middleware outside the monitor that retries failed model calls,
such as LangChain's `ModelRetryMiddleware`, runs the whole step again with
fresh samples, and `monitor_log` then records only the attempt that succeeded;
`check_monitor_placement` warns about such a list.

## Three kinds of monitor model

Any model that can judge a step can be the monitor. The three families return
different things, so each has its own monitor class, and all of them produce
the same `Verdict`. The model is always a parameter: the library never picks
one, and a provider string such as `"openrouter:..."` is resolved with
LangChain's `init_chat_model`.

- **Chat judges** (`LLMMonitor`) are general chat models prompted to reason
  first and then give a score on an anchored scale, 0 to 10 by default, mapped
  onto 0 to 1. Reasoning before the score is the prompt component Apollo
  Research found matters most [@apollo2026monitoringprompt]. A reply is read
  only in that shape: the reasoning, then one score tag, with nothing else
  between them or after the score. The agent writes the step the judge
  reads, so it can aim text at the judge there, such as a planted score tag
  [@terekhov2025adaptive], and a judge may quote it; the
  shape keeps a quoted score from becoming the verdict. A reply in any other
  shape, or one the provider reports as cut off at a length limit, is asked
  for again and then counts as fully suspicious.
- **Guard models** (`GuardModelMonitor`) label a step against a policy you
  write. Llama Guard and its peers score with the probability of the unsafe
  label, read from the label token's log-probabilities
  [@inan2023llamaguard; @zeng2024shieldgemma; @padhi2024graniteguardian]. When
  a provider returns no log-probabilities, the share of several samples
  labelled unsafe is the standard fallback, with gains that level off after a
  handful [@xiong2024uncertainty; @lyu2025sampleconsistency].
  `GuardScoring.AUTO` does the first when it can and draws five samples
  otherwise. Guard models are overconfident, especially under jailbreaks
  [@liu2025guardcalibration], and a hard 0 or 1 label gives a threshold only
  one operating point, so that option warns. The label is read only where the
  guard's format puts it: on the last line, or on the first line when it is
  unsafe and only category codes follow, as in Llama Guard
  [@meta2024llamaguard3format; @meta2025llamaguard4]. A reply with two
  different labels on their own lines, a label elsewhere, or a cut-off reply
  counts as suspicious, for the same reason as a quoted score.
- **Decision models** (`DecisionModelMonitor`) answer yes or no questions with
  a probability and generate no text. Suspicion is the probability of yes.
  Several questions can go in one request, combined by their maximum, so one
  strong hit is not diluted. Jev is served through OpenRouter's Decisions API
  [@openrouter2026decisions] and through TypeSafe's own client
  [@typesafe2026langchain]. Only the answers in a Decisions API response are
  validated, so a change in a field the library does not read, such as
  `usage`, cannot discard a valid answer.

No family adds a score, probability or count to its `reason`. A chat
judge's reason is its own reasoning, passed through as the judge wrote it,
without the score tag; a guard's names its finding without a probability or
a count of replies; and a decision model's names the question it found most
likely without the probability. Auto Mode shows the reason to the agent, and
such a number would tell the agent how close its step came to passing. The
numbers stay in `suspicion`, which the log records keep.

Scores have to be fine-grained. Control protocols set thresholds as
percentiles of honest-run scores, and a score with few distinct values cannot
hit a small audit budget, because the budget falls inside a group of tied
scores [@kaufman2025basharena; @koran2026ensemble]. Three wrappers, each a
monitor itself, address this and stack around any family:

- `RepeatedMonitor` averages several calls of one monitor, as the control
  papers do [@bhatt2025ctrlz; @tracy2026linuxarena; @koran2026ensemble];
- `CalibratedMonitor` reports suspicion as the step's smoothed conformal rank
  among honest-run scores, breaking ties at random
  [@vovk2005algorithmic; @makins2026multiagent], so a threshold of 0.99 flags
  1% of honest steps;
- `CascadeMonitor` runs a fast monitor on every step and sends only the steps
  it finds suspicious to a careful one [@hua2025combining], after the two
  stages of Claude Code's auto mode [@hughes2026automode].

```mermaid
flowchart LR
    step["Proposed step"] --> fast["Fast monitor, for example a decision model"]
    fast -- "below the escalation score" --> verdict["Verdict"]
    fast -- "at or above it" --> careful["Careful monitor, for example a calibrated, repeated chat judge"]
    careful --> verdict
```

## Subagents

Deep Agents does not pass the main agent's middleware to its subagents. A
helper, `monitor_subagents`, adds a monitor to every subagent spec, including
the built-in general-purpose one. Each subagent gets a copy of the main
monitor and protocol by default, named after the subagent and marked with
`TaskAuthor.PARENT_AGENT`, so its monitor reads the delegated task as the
parent agent's words rather than the user's. An override can give a named
subagent a different monitor, protocol or view. Compiled and remote subagents
cannot take injected middleware, so the helper raises an error for them
instead of leaving them unmonitored. Its message says how to monitor one in its
own graph: `MonitorMiddleware(..., agent_name="<subagent name>",
task_author=TaskAuthor.PARENT_AGENT)`. Both settings matter. With the default
`agent_name="main"`, the subagent's records count as the main agent's own, so
the parent's monitor never sees the subagent's halt and `STOP_WHOLE_RUN` does
not fire; with the default task author, the subagent's monitor reads the
delegated task as the user's authorisation.

Forked subagents, `mode="fork"` in Deep Agents, are not supported yet (issue
#35), and the helper raises `ConfigurationError` for them. A fork continues the
parent's conversation and inherits the main agent's middleware, so it runs
under `monitor[main]`. That monitor reads the fork's task, which the parent
agent wrote, as the user's words, and records the fork's steps under the main
agent's name, where they renumber the main agent's steps and hide a halt inside
the fork from `STOP_WHOLE_RUN`. This happens whenever a monitored agent has a
fork, with or without the helper, so a monitored agent should delegate only to
isolated subagents.

```mermaid
flowchart TD
    subgraph parent ["Main agent"]
        mainMonitor["monitor[main], task author: the user"] --> mainModel["Untrusted model"]
        mainModel --> taskTool["The task tool"]
        mainLog[("monitor_log: its own records and the subagents' records")]
    end
    subgraph child ["Subagent: researcher"]
        subMonitor["monitor[researcher], task author: the parent agent"] --> subModel["Untrusted model"]
        subModel --> subTools["Its own tools"]
        subLog[("monitor_log: its own records")]
    end
    taskTool -- "task description, judged first by monitor[main], and the Delegation" --> subMonitor
    subModel -- "final report, read by monitor[main]" --> taskTool
    subLog -. "records, including halts" .-> mainLog
```

`SubagentHalt` decides what a halt inside a subagent does. With
`STOP_SUBAGENT_ONLY`, the default, the parent receives the halt message as the
task's result and carries on under its own monitor, unless the subagent's
blocks spent Auto Mode's thread total: the total belongs to the whole thread,
so the parent's next step then goes to `when_limit_reached`, which halts the
run by default. With `STOP_WHOLE_RUN`, the
parent's monitor finds the subagent's halt record in `monitor_log` and halts
the parent before its next model call. This works because a subagent's
records flow back into the parent's log, which depends on declaring the log
with its reducer in the last position: LangGraph reads a reducer only from the
last position of the annotation [@langgraph2026].

A subagent starts every delegation with an empty log, so its step numbers
start again at 1. Before each tool call, the parent's monitor adds a
`Delegation` to the state the tool sees: the id of the tool call, the name of
the agent that made it, and the blocks each monitor has recorded in the
thread. Deep Agents' `task` tool passes that state on to the subagent, as it
does the rest of the parent's state, and the subagent's monitor stores the id
in each of its records as `delegation_id`. So `(agent, delegation_id,
step_number)` names one step, and an auditor can match a delegation's records
to its `task` call and result in the parent's transcript. This relies on the
model provider giving every tool call in the thread its own id, as providers
do, which LangChain does not check. The state key, `monitor_delegation`, is
left out of the subagent's output, as Deep Agents does for its own
forked-context flag [@deepagents2026], so it never flows back into the parent.
The block counts let the subagent's Auto Mode start from the thread's total.
Subagents that run in parallel do not see each other's blocks, so together
they can pass the total; the parent counts every one of them at its next step.

A second monitor stacked in the same agent finds the delegation for the call
already in place, made by its own agent, and passes it on as it is, so the
blocks are not counted twice. A subagent's monitor finds the delegation it
inherited instead, made by another agent, and replaces it with one that adds
the subagent's own blocks, even when the subagent's tool call reuses the id
of the call that started it. Telling the two apart relies on each agent's
monitor having its own `agent_name`, which `monitor_subagents` gives it.

A subagent receives its delegation as input, so `monitor_delegation` is part of
the input schema of every monitored agent, a plain `create_agent` one
included. Whoever invokes an agent could set it, so the monitor validates the
value with pydantic when it reads it. A value that is not a `Delegation` with
non-negative block counts raises `ConfigurationError` before the step runs. It
is not ignored: a subagent that ignored a bad delegation would count from its
own log alone, which resets the thread's total, and a negative count would
lift the total altogether. Leave the key out when you invoke an agent; the
monitor sets it for each subagent it starts.

## Sync and async

LangChain calls `wrap_model_call` under `invoke()` and `stream()`, and
`awrap_model_call` under `ainvoke()` and `astream()`. Its own middlewares keep
two copies of their logic, one per hook. This library writes each protocol
once, as a coroutine that talks to a pending step, and gives the pending step
two implementations.

```mermaid
flowchart LR
    asyncEntry["ainvoke or astream"] --> asyncHook["awrap_model_call"]
    asyncHook --> asyncStep["Async pending step: awaits the model and the monitor"]
    syncEntry["invoke or stream"] --> syncHook["wrap_model_call"]
    syncHook --> syncStep["Sync pending step: calls them directly"]
    asyncStep --> decide["The protocol's decide coroutine"]
    syncStep --> decide
```

The sync pending step's methods are declared `async` but never wait on
anything, so under `invoke()` the protocol's coroutine can be finished with a
single step, without an event loop. If a protocol ever awaited real
asynchronous work there, the library raises `SynchronousRunError` instead of
hanging. That covers a protocol that suspends, and one that starts asyncio work
where no loop can run it, such as `asyncio.gather` or `asyncio.ensure_future`
outside a running loop. Inside a running loop, as when `invoke()` is called
from a notebook, any task the protocol scheduled is cancelled before it
starts, and once a step is over its pending step refuses to call the model,
so no work a protocol left behind reaches the model after the error. A
timeout, `asyncio.timeout` or `asyncio.wait_for`, cannot limit a pending
step's call under `invoke()`: outside a running loop it raises
`SynchronousRunError`, and inside one it never fires, because the synchronous
call blocks the loop. Use `ainvoke()` for protocols that need asyncio.

Under `invoke()`, samples asked for concurrently are drawn one after another.
Under `ainvoke()` they are drawn in an `asyncio.TaskGroup`, as are the calls
of `RepeatedMonitor`, so a call that fails cancels the others instead of
leaving them to spend tokens, and its own error propagates, not wrapped in an
`ExceptionGroup`. Monitors declare both `evaluate` and `evaluate_sync`, so a
monitor that lacks either cannot be instantiated.

## What streams

A synchronous monitor judges a step before anything uses it, so nothing the
protocol produces reaches `stream_mode="messages"` before the protocol has
decided. The middleware runs the protocol inside a block that adds LangGraph's
`nostream` tag to every model call made there, and the message stream skips
calls with that tag [@langgraph2026]. That covers the untrusted samples, the
trusted model's step and the monitor's own calls, including samples drawn in
parallel and calls inside a subagent's monitor. When the model node returns the
committed step, LangGraph streams it from the node's output [@langgraph2026].

```mermaid
flowchart LR
    samples["Samples, the trusted step and the monitor's calls"] -- "tagged nostream" --> skipped["Skipped by the message stream"]
    protocol["The protocol decides"] --> committed["The committed step"]
    committed -- "returned by the model node" --> consumer["stream_mode=messages"]
```

The cost is live tokens. A monitored agent streams each step whole, once the
protocol commits it, and not token by token as the model writes it.

Rejected samples stay out of the conversation, not out of the record. They
remain in `monitor_log`, which the `values` and `updates` stream modes carry,
in the `MonitorStepEvent` on `stream_mode="custom"`, and in traces.
`astream_events` is not filtered: it reports every model call live, rejected
samples included, so a user interface should read `stream_mode="messages"`
instead. It also reports the monitor's own spans, described in
[Tracing](#tracing), which `exclude_tags=["monitor"]` drops. `astream_log`,
built from the same callbacks, reports them too; the `stream()` modes and
`stream_events(version="v3")` do not. With
`FeedbackVisibility.IN_TRANSCRIPT`, a blocked attempt and its feedback are
committed as part of the conversation, so they stream along with the step that
follows them.

The monitor's own calls also carry LangChain's internal-call metadata, from
`model_calls.build_internal_call_config`, which the experimental
`stream_events(version="v3")` projection uses to drop them [@langchain2026]. A
custom monitor that passes its own `tags` in a call's config replaces the tags
the call would inherit, and that call streams again; such a monitor should add
`langgraph.constants.TAG_NOSTREAM` to its tags.

## Tracing

Any tracer built on LangChain callbacks, LangSmith and Langfuse among them,
shows each monitored step as a small tree of named spans. The model calls stay
where they were, now nested under the step that caused them:

```text
model                           the agent's model node
└── monitor step                one per step
    ├── <untrusted model>       sample 1
    ├── monitor judgement       the monitor's verdict on sample 1, holding either
    │   ├── <judge model>       the monitor's own model calls, or
    │   └── monitor classifier  one Decisions API request
    ├── <untrusted model>       sample 2, if the protocol draws one
    ├── monitor judgement       the verdict on sample 2
    ├── <trusted model>         if a fallback writes the step
    └── monitor decision        opened and closed once the protocol has decided
```

| Span | Inputs | Outputs |
|---|---|---|
| `monitor step` | The step number, and the step first proposed, added when the step ends. | The outcome, whether the step was flagged, the blocked attempts, the highest suspicion, and each sample's suspicion, reason and whether it ran. |
| `monitor judgement` | The sample's number, counted in the order the draws started, and the monitor's class name. | The verdict's suspicion and reason. |
| `monitor classifier` | The decision model and its questions. | The probability of yes to each question. One span covers a request and all its retries. |
| `monitor decision` | None. | The outcome, the flag and the highest suspicion. |

The names are fixed, so both tools can filter and count spans by name, and
what varies goes in metadata under flat keys: `monitor_name`,
`monitor_agent`, `monitor_step_number`, `monitor_protocol`,
`monitor_step_id`, the step span's run id, and, inside a subagent that a
monitored agent started, `monitor_delegation_id`. Every monitor span of a step
carries them, and every one is tagged `monitor`. A span's tags are fixed when
it starts, while the step span must start before its samples, so the outcome
goes on the decision span: it is tagged `monitor:<outcome>`, and
`monitor:flagged` when flagged, and repeats the outcome in its metadata. In LangSmith,
`and(eq(name, "monitor decision"), has(tags, "monitor:halted"))` finds every
halted step; in Langfuse, the name `monitor decision` and the metadata
`monitor_outcome`. The spans below the step span carry
`ls_agent_type: "middleware"`, and the model calls the library's monitors
make carry `ls_message_view_exclude`, which keep them out of LangSmith's
Trajectory view [@langsmith2026trajectory]. The step span carries neither,
because the agent's own samples nest in it.

The spans add no tag or metadata to the LangChain runs inside a step, model
calls included, so no sample is labelled as monitor work. LangSmith's
`traceable` runs are the exception: LangSmith builds each one from its parent
run and copies the parent's metadata into it [@langsmithsdk2026]. So in
LangSmith the hook runs of middleware listed after the monitor, an inner
stacked monitor's hook run and a monitor's own `traceable` code carry the
step's `monitor_` keys. Only dropping the keys from the spans would prevent
that, so a LangSmith filter should name the span as well as its metadata.

The only proposal text the spans add is the step span's proposed step; each
sample's text stays on the model call that drew it. A halt is a decision, and
ends its spans as usual. A span ends with an error only when the step fails: a
model or monitor call that raises, a sample cancelled because another one
failed, or `SynchronousRunError`. With no callback handler attached, no span
is opened. LangGraph attaches one of its own for `stream_mode="messages"`,
which receives the spans and ignores them.

The spans are opened from the running node's own callback manager, not
through `CallbackManager.configure`, so they nest under the model node for
every handler. Configuring a new manager would, with LangSmith tracing on,
hang them under the LangSmith-only run in which `create_agent` wraps each
middleware hook, which can split the trace in Langfuse
[@langchain2026; @langfuse2026]. Code in a monitor that traces itself with
LangSmith's `traceable` decorator nests under its judgement, since LangSmith
takes a `traceable` run's parent from the LangChain config the judgement span
sets [@langsmithsdk2026].

Some runs still name a LangSmith-only hook run as their parent, which
another handler attached beside LangSmith never saw. A tool
call runs inside the monitor's `wrap_tool_call` hook, where the monitor opens
no span, and in a Deep Agent the `task` call that starts a subagent is such a
tool call. The agent's own model calls run inside the hook runs of any
middleware listed after the monitor that wraps the model call, as the ones
Deep Agents adds do. Under `ainvoke()`, Langfuse can put each such run, with
everything inside it, a whole subagent included, in a trace of its own; the
hook runs cause that split, not the spans.
[See the monitor's decisions in LangSmith and Langfuse](../how-to/see-decisions-in-langsmith-and-langfuse.md)
shows how to attach either tool and find the halted and flagged steps.

## The log records

Records pass through checkpointers, so they hold plain values only: strings,
numbers, booleans, `None`, lists and `TypedDict`s, with the outcome stored as
a string. A `StepRecord` names the agent and the monitor, numbers the step,
gives the outcome, whether the step was flagged and how many attempts were
blocked, and lists a `SampleRecord` for every sample: its suspicion, the
monitor's reason, the proposal rendered in full for a human auditor, and
whether it ran. A subagent's records also carry `delegation_id`, the id of the
tool call that started the subagent, because its step numbers start again
with every delegation; the main agent's records have none.
