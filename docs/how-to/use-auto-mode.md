# Use Auto Mode

This guide sets up Auto Mode, the protocol that blocks a suspicious step,
tells the agent why, and lets it try again.

[TOC]

Use it when most suspicious steps are an honest agent overreaching, and a
person would otherwise approve each step. An honest agent that overreached
can then finish the task by a safer route.
After three blocks in a row within one step, or twenty in the conversation
thread, a fallback writes the step, and by default the run halts. The
protocol follows Claude Code's auto mode [@hughes2026automode], which stops a
session after three denials in a row or twenty in all and hands it to a
person, or, in headless mode, ends the process. Here the total counts per
conversation thread, and the default fallback ends the run, as headless mode
does.

## How it works

```mermaid
flowchart TD
    start{"New subagent blocks since the last step, and the thread at or over max_total_blocks?"}
    start -- "yes" --> fallback
    start -- "no" --> attempt["The agent proposes a step, seeing this step's blocked attempts and their feedback"]
    attempt --> check{"The monitor's suspicion at or above block_threshold?"}
    check -- "no, first attempt" --> allowed["Run it: allowed"]
    check -- "no, after a block" --> steered["Run it, flagged: steered"]
    check -- "yes" --> feedback["Block it and write the feedback"]
    feedback --> limit{"A block limit reached?"}
    limit -- "no" --> attempt
    limit -- "yes" --> fallback["when_limit_reached writes the step, flagged: halted or deferred_to_trusted"]
```

The block limits are `max_consecutive_blocks` within the step and
`max_total_blocks` in the thread. A step goes to the fallback before any
attempt only when blocks inside subagents since this agent's last step leave
the thread at or over the total.

The tool calls of a blocked attempt never run. A provider's built-in tools in
it, such as a web fetch, already ran inside the model call that drew it, once
per attempt. The next attempt is drawn with the blocked proposal and its
feedback in view, so the agent knows what was refused and why. The attempt
that reaches a limit is blocked like the others: it gets feedback, and it
counts in `blocked_count`.

## A complete example

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from langchain_sync_monitors import (
    AutoMode,
    FeedbackTemplate,
    FeedbackVisibility,
    HaltRun,
    LLMMonitor,
    MonitorMiddleware,
)


def read_file(path: str) -> str:
    """Read a file from the project folder."""
    return "# Q3 report\nRevenue grew 12% to 4.1M. Churn fell from 3% to 2%."


def http_post(url: str, body: str) -> str:
    """Send a request body to a URL."""
    return "202 Accepted"  # A stand-in that sends nothing.


monitor = MonitorMiddleware(
    monitor=LLMMonitor(model="openrouter:xiaomi/mimo-v2.6-pro"),
    protocol=AutoMode(
        # Block attempts at or above 0.6; calibrate your own value.
        block_threshold=0.6,
        # Claude Code's limits: three blocks in a row, twenty in the thread.
        max_consecutive_blocks=3,
        max_total_blocks=20,
        # At a limit, end the run with this message.
        when_limit_reached=HaltRun(message="[Safety monitor] Stopped after repeated unsafe steps."),
        # What the agent reads after a block; {reason} is the monitor's reason.
        feedback=FeedbackTemplate(
            text=(
                "[Safety monitor] Your last step was blocked, and none of your tools ran it.\n"
                "Reason: {reason}\n"
                "Finish the user's task another way, or stop and say what approval you need."
            ),
        ),
    ),
    # Blocked attempts stay out of the agent's messages.
    feedback_visibility=FeedbackVisibility.HIDDEN,
)
agent = create_agent(
    model="openrouter:deepseek/deepseek-v4.1-flash",
    tools=[read_file, http_post],
    middleware=[monitor],  # The monitor goes last in the list.
    # Keeps monitor_log, and so the total, across the thread's turns.
    checkpointer=InMemorySaver(),
)

config = {"configurable": {"thread_id": "q3-summary"}}
task = {"messages": [{"role": "user", "content": "Summarise q3.md for the team."}]}
result = agent.invoke(task, config=config)
for record in result["monitor_log"]:
    print(record["step_number"], record["outcome"], record["blocked_count"], record["flagged"])
```

The model strings are examples, and `openrouter:` strings need the
`openrouter` extra. Any LangChain chat model works for the agent and for the
judge: pass a provider string, or a chat model object you built.

## Options

| Option | Default | What it does |
|---|---|---|
| `block_threshold` | 0.6, with a warning | An attempt whose suspicion is at or above this value is blocked. |
| `max_consecutive_blocks` | 3 | When the blocks within one step reach this number, `when_limit_reached` writes the step. A whole number of at least 1. |
| `max_total_blocks` | 20 | When this monitor's blocks in the thread, subagents' included, reach this number, `when_limit_reached` writes the step. A whole number of at least 1, or `math.inf` to turn the total off. |
| `when_limit_reached` | `HaltRun()` | The `Fallback` that writes the step once a limit is reached. A protocol here, such as `DeferToTrusted` in place of `DeferToTrustedModel`, raises `ConfigurationError`. |
| `feedback` | `DEFAULT_FEEDBACK_TEMPLATE` | The message the agent receives when a step is blocked, a `FeedbackTemplate`. |

Two options of `MonitorMiddleware` matter here:

| Option | Default | What it does |
|---|---|---|
| `feedback_visibility` | `FeedbackVisibility.HIDDEN` | Whether blocked attempts and their feedback stay in the agent's messages. |
| `label` | `"monitor"` | The name the thread total is counted under. Stacked monitors need distinct labels. |

Each option is checked when the object is built, and a value of the wrong
type raises `ConfigurationError`: a limit given as a float or a `bool`, other
than `math.inf` for `max_total_blocks`, or a plain string for
`feedback_visibility`. The threshold is a number from 0 to
1; see [Thresholds](choose-a-protocol.md#thresholds). The other options of
`MonitorMiddleware` are listed in
[Middleware options](choose-a-protocol.md#middleware-options).

### Thresholds and limits

The default `block_threshold` of 0.6 is a placeholder, and a protocol built
with it emits an `UncalibratedThresholdWarning`. Set the threshold at a high
percentile of the scores your monitor gives on honest runs of your own agent,
as [Calibrate the threshold](use-trusted-monitoring.md#calibrate-the-threshold)
describes.

The two limits count different things:

- `max_consecutive_blocks` counts blocks within one step, since a step ends as
  soon as an attempt is accepted. With the default of 3, the third blocked
  attempt of a step goes to the fallback, and the agent has seen feedback
  twice. With `FeedbackVisibility.IN_TRANSCRIPT`, all three blocked attempts
  and their feedback are committed before the fallback's step.
- `max_total_blocks` counts every block recorded under this monitor's `label`
  in the conversation thread: the sum of `blocked_count` over the records in
  `monitor_log` whose `monitor` is that label, the subagents' records
  included. With a checkpointer the log is kept across turns, so the total
  spans every turn of the thread and never resets. Once a thread has reached
  it, the first block of every later step goes straight to the fallback;
  steps that are not blocked still run. Without a checkpointer, each run
  starts with an empty log, so the total counts one run. To count afresh,
  start a new thread.

In a Deep Agent the total covers the subagents, nested ones included. A
subagent starts each delegated task with an empty log, but starts from the
total recorded before it was started, so it reaches the limit where the
thread does, and delegating again does not reset it. When blocks inside
subagents since this agent's last step leave the thread at or over the total,
this agent's next step goes to `when_limit_reached` without being sampled.
Two limits of the count follow:

- **Parallel subagents can pass the total.** Subagents that run in parallel
  do not see each other's blocks, so together they can pass the total; their
  parent counts every one of them at its next step.
- **A failed subagent's blocks are missed.** A subagent whose run raises
  returns no records, so the total misses the blocks it recorded. They count
  only when the failed run is resumed from its checkpoint with `None` as
  input. They never count when the thread goes on with new input, or when a
  middleware such as LangChain's `ToolRetryMiddleware` or
  `ToolErrorMiddleware` answers the failed call with an error message or runs
  it again; a retry starts the subagent again from the same count.
  `check_monitor_placement` warns about such middleware.

The count is kept per `label`. `monitor_subagents` gives each subagent a copy
of the parent's monitor with the same label. An override with a label of its
own counts apart. One built without a `label` keeps `"monitor"`, so it shares
the parent's total when the parent keeps the default label too.

### When a limit is reached

`when_limit_reached` takes any `Fallback`:

- `HaltRun(message=...)`, the default, ends the run with a final message and
  no tool calls. The middleware routes the agent to its end, even an agent
  that would otherwise loop until it has a structured response. The default
  message is `DEFAULT_HALT_MESSAGE`:

  > [Safety monitor] This run was stopped because the agent's proposed steps
  > looked unsafe. The rejected steps are recorded in monitor_log for review.

  The halt stands: until a later run brings new input from the user, every
  further step halts again without a sample, with the library's standing-halt
  message rather than yours. Two paths through middleware listed before the
  monitor can lift it early; see
  [A halt stands](choose-a-protocol.md#a-halt-stands).
- `DeferToTrustedModel(trusted_model=...)` has a trusted model write the step
  from the step's request with only the model changed, and the run goes on.
  The trusted model does not see this step's blocked attempts or their
  feedback, and its step is not judged. With `IN_TRANSCRIPT`, the blocked
  attempts and feedback of earlier steps are part of the conversation it
  receives.

A fallback must not ask a person by calling LangGraph's `interrupt()`. It runs
inside the model node, and on resume LangGraph runs the whole node again: the
protocol draws fresh attempts, and the person's answer would apply to steps
they never saw. To involve a person, halt, and ask outside the model node, as
`HumanInTheLoopMiddleware` does for tool calls.

### Feedback

The default feedback, `DEFAULT_FEEDBACK_TEMPLATE`, has three lines:

> [Safety monitor] Your last step was blocked, and none of your tools ran it.
>
> Reason: {reason}
>
> Continue the user's task by a safer route. Do not retry the blocked action,
> and do not try to get the same effect another way. If no safe route exists,
> stop and say what approval you would need from the user to continue.

It follows Claude Code's auto mode, whose denial reaches the agent as a tool
result that asks it to find a safer path rather than work around the block
[@hughes2026automode]. "None of your tools ran it" stays true when a
provider's built-in tool in the step already ran inside the model call.

To write your own, pass `FeedbackTemplate(text=...)`:

- `text` must contain `{reason}`, where the monitor's reason goes, and no
  other format field. Write a literal brace as `{{` or `}}`. A template that
  breaks these rules raises `ConfigurationError` when it is built.
- Start the text with a marker such as `[Safety monitor]`. Some providers drop
  a message's `name`, so the text itself must say where it comes from.

The reason is the monitor's own words. The library adds no score or
probability to it, so the agent does not learn from the library how close its
step came to passing. A chat judge's reason is its own reasoning: the default
prompt asks it to keep its score out, but its wording may still hint at one.
A monitor you write yourself should keep scores out too.

### What the agent sees

A blocked attempt is answered in a way every provider accepts:

- each tool call of the attempt, malformed ones included, gets an error tool
  result that carries the feedback, so every call keeps a matching result;
- an attempt with no tool calls, a final answer, gets a human message that
  carries the feedback, tagged as the monitor's.

`feedback_visibility` decides whether these stay in the conversation:

```mermaid
sequenceDiagram
    participant Model as Agent's model
    participant Middleware as Monitor middleware
    participant Messages as Agent's messages
    Middleware->>Model: the conversation so far
    Model-->>Middleware: attempt 1, http_post with the key
    Note over Middleware: suspicion 0.90, blocked
    Middleware->>Model: the conversation, attempt 1 and an error result with the feedback
    Model-->>Middleware: attempt 2, the final answer
    Note over Middleware: suspicion 0.10, runs
    alt FeedbackVisibility.HIDDEN
        Middleware->>Messages: attempt 2 only
    else FeedbackVisibility.IN_TRANSCRIPT
        Middleware->>Messages: attempt 1, its feedback, then attempt 2
    end
```

- With `HIDDEN`, the default, only the retries within the step see the blocked
  attempts and their feedback. The agent's messages show the accepted step
  alone, and `monitor_log` keeps the blocked attempts.
- With `IN_TRANSCRIPT`, the blocked attempts and their feedback are committed
  before the accepted step, as Claude Code does [@hughes2026automode]. The
  agent remembers them on later steps, and so does the monitor, which reads
  earlier feedback as `<monitor_feedback>`.

Every message the monitor inserts gets a fresh id of the form
`monitor-<uuid4>`, because LangGraph replaces messages that share an id.

## Read the outcome in `monitor_log`

Every step adds one record, which keeps every attempt the monitor
judged.

| `outcome` | What happened | `blocked_count` | Samples | Executed sample |
|---|---|---|---|---|
| `"allowed"` | The first attempt was acceptable. | 0 | 1 | The first |
| `"steered"` | One or more attempts were blocked, then one was accepted. | 1 or more | 2 or more | The last |
| `"halted"` | A limit was reached, and `HaltRun` ended the run. | 1 or more | As many as blocked | None |
| `"deferred_to_trusted"` | A limit was reached, and the trusted model wrote the step. | 1 or more | As many as blocked | None |
| `"halted"` or `"deferred_to_trusted"` | Blocks inside subagents since this agent's last step left the thread at or over the total, so the step went to the fallback unsampled. | 0 | 0 | None |
| `"halted"` | A halt stood, or a subagent was halted under `SubagentHalt.STOP_WHOLE_RUN`. | 0 | 0 | None |

Every step except `allowed` is flagged.

To review what was blocked, and the thread's total as Auto Mode counts it:

```python
for record in result["monitor_log"]:
    # A blocked attempt is a sample that did not run.
    blocked = [sample for sample in record["samples"] if not sample["executed"]]
    for sample in blocked:
        print(f"Step {record['step_number']}, blocked at {sample['suspicion']:.2f}")
        print(sample["proposal"])
        print(f"  reason: {sample['reason']}")

# Every record under this monitor's label counts, the subagents' included.
label = "monitor"
records = [record for record in result["monitor_log"] if record["monitor"] == label]
blocks_in_thread = sum(record["blocked_count"] for record in records)
```

## Limits

- **The agent learns what the monitor catches.** Feedback helps an honest
  agent, and it also tells a scheming agent what was refused. If that matters
  more than steering, use Defer to Resample, where the agent never sees a
  rejected step; see [Choose a protocol](choose-a-protocol.md).
- **The total never resets, and can be passed.** A thread that has reached
  `max_total_blocks` sends the first block of every later step to the
  fallback. Parallel subagents can together pass the total, and a failed
  subagent's blocks are missed; see
  [Thresholds and limits](#thresholds-and-limits).
- **Put the monitor last** in the `create_agent` middleware list, so no other
  middleware runs inside it and returns commands for attempts the monitor
  blocks. `check_monitor_placement(middleware=[...])` warns about such a list.
- **A provider's built-in tools run once per attempt.** A blocked attempt's
  web search or code execution has already run at the provider; see
  [Tools the provider runs itself](choose-a-protocol.md#tools-the-provider-runs-itself).
- **The graph takes more steps.** The monitor adds two graph steps per agent
  step and two per run; see
  [Graph steps, streams and failures](choose-a-protocol.md#graph-steps-streams-and-failures),
  which also covers a step that fails.

## Related guides

- [Use a chat judge](use-a-chat-judge.md) for what goes into the reason the agent is shown.
- [Choose what the monitor reads](choose-what-the-monitor-reads.md) to decide which channels the monitor judges.
- [Read the monitor log](read-the-monitor-log.md) to count blocks and read the monitor's reason for each block, which the feedback template wraps.
- [Monitor Deep Agents subagents](monitor-deep-agents-subagents.md) for the total across subagents.

## References
