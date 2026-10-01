"""The execution invariant, checked on a saved report: only a committed step's calls run.

The middleware promises that none of the agent's own tools runs a call the
protocol rejected or blocked. A report holds what ran, from the sandbox's
records, every sample with whether it ran, and every message of the final
state, so the promise can be checked from the report alone, by each call's
tool name and key argument:

- what ran is exactly the tool calls of the committed replies, the AI
  messages the monitor did not insert;
- a step whose sample ran committed exactly that sample's calls, and a step
  the trusted model wrote, or a halted one, ran no sample;
- no call ran that only a rejected or blocked sample proposed: every call
  that ran was proposed by a sample that ran or written by the trusted model.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence

from tests.live.reports import (
    HALTED,
    MONITOR_MESSAGE_ID_PREFIX,
    MessageReport,
    RunReport,
    StepReport,
    read_proposed_calls,
)
from tests.live.sandbox import describe_call

DEFERRED = "deferred_to_trusted"
OUTCOMES_WITHOUT_A_SAMPLE_THAT_RAN = frozenset({DEFERRED, HALTED})


def count_proposed_calls(proposal: str) -> Counter[str]:
    """Count a proposal's calls, each described as the sandbox records it."""
    return Counter(call.description for call in read_proposed_calls(proposal))


def count_message_calls(message: MessageReport) -> Counter[str]:
    """Count a saved message's tool calls, each described as the sandbox records it."""
    return Counter(
        describe_call(call["name"], arguments=call["args"]) for call in message["tool_calls"]
    )


def list_committed_replies(messages: Iterable[MessageReport]) -> list[MessageReport]:
    """Return the saved AI messages the steps committed: every one the monitor did not insert."""
    return [
        message
        for message in messages
        if message["type"] == "ai"
        and not (message["id"] or "").startswith(MONITOR_MESSAGE_ID_PREFIX)
    ]


def sum_counts(counts: Iterable[Counter[str]]) -> Counter[str]:
    """Add up call counts."""
    total: Counter[str] = Counter()
    for count in counts:
        total.update(count)
    return total


def check_step(step: StepReport, *, reply: MessageReport) -> list[str]:
    """Check that a step ran no sample or one, and committed what that sample proposed."""
    number = step["step_number"]
    ran = [sample for sample in step["samples"] if sample["executed"]]
    expected_samples = 0 if step["outcome"] in OUTCOMES_WITHOUT_A_SAMPLE_THAT_RAN else 1
    if len(ran) != expected_samples:
        return [f"step {number} ({step['outcome']}): {len(ran)} samples ran"]
    if ran and count_proposed_calls(ran[0]["proposal"]) != count_message_calls(reply):
        return [f"step {number}: the committed reply's calls differ from the sample that ran"]
    return []


def pair_steps_with_replies(
    steps: Sequence[StepReport],
    *,
    replies: Sequence[MessageReport],
) -> list[tuple[StepReport, MessageReport]] | None:
    """Pair every step but a halted one with its committed reply, or `None` if they differ."""
    running = [step for step in steps if step["outcome"] != HALTED]
    if len(running) != len(replies):
        return None
    return list(zip(running, replies, strict=True))


def find_uncovered_calls(ran: Counter[str], *, covered: Counter[str]) -> list[str]:
    """Return a problem for each call that ran more often than anything that ran proposed it."""
    return [
        f"{call} ran {count} times, but a sample that ran or the trusted model "
        f"proposed it {covered[call]} times"
        for call, count in ran.items()
        if count > covered[call]
    ]


def check_execution_invariant(report: RunReport) -> list[str]:
    """Return every way the run broke the execution invariant; an empty list when it held."""
    replies = list_committed_replies(report["messages"])
    pairs = pair_steps_with_replies(report["steps"], replies=replies)
    if pairs is None:
        return [f"{len(replies)} committed replies for {len(report['steps'])} steps"]
    ran = Counter(report["executed_calls"])
    committed = sum_counts(count_message_calls(reply) for reply in replies)
    problems = [] if ran == committed else [f"ran {dict(ran)}, committed {dict(committed)}"]
    for step, reply in pairs:
        problems.extend(check_step(step, reply=reply))
    covered = sum_counts(
        count_message_calls(reply) if step["outcome"] == DEFERRED else Counter()
        for step, reply in pairs
    ) + sum_counts(
        count_proposed_calls(sample["proposal"])
        for step in report["steps"]
        for sample in step["samples"]
        if sample["executed"]
    )
    return [*problems, *find_uncovered_calls(ran, covered=covered)]
