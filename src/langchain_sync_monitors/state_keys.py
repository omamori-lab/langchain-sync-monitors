"""The monitor's state keys, and `MONITOR_STATE_KEYS`, the ones only the monitor writes.

`MonitorState` declares a field under each. `task_authorship` drops a tool's
writes to every key in `MONITOR_STATE_KEYS`, which is why these live apart
from the modules that write them: those modules import `task_authorship`.
`MONITOR_STATE_KEYS` also holds `monitor_delegation`, defined in `_langchain`
beside the reader that parses it; `monitor_log`, also defined there, is the
one monitor key a tool may write.
"""

from langchain_sync_monitors._langchain import MONITOR_DELEGATION_KEY

TASK_MESSAGES_KEY = "monitor_task_messages"
"""The state key that holds the ids of the human messages that arrived as a run's input."""

SEEN_HUMAN_MESSAGES_KEY = "monitor_seen_human_messages"
"""The state key that holds the ids of every untagged human message the monitor has seen."""

RUN_OPEN_KEY = "monitor_run_open"
"""The state key that is true from the start of a run until the run reaches its end."""

RUN_INPUTS_KEY = "monitor_run_inputs"
"""The state key that holds the text of every human message a run received as its input."""

REWRITTEN_INPUTS_KEY = "monitor_rewritten_inputs"
"""The state key that holds the ids of the seen human messages a tool wrote a message under."""

INPUTS_AT_HALT_KEY = "monitor_inputs_at_halt"
"""The state key that holds, for each monitor, how many run inputs the thread had at its halt."""

MONITOR_STATE_KEYS = frozenset(
    {
        TASK_MESSAGES_KEY,
        SEEN_HUMAN_MESSAGES_KEY,
        RUN_OPEN_KEY,
        RUN_INPUTS_KEY,
        REWRITTEN_INPUTS_KEY,
        INPUTS_AT_HALT_KEY,
        MONITOR_DELEGATION_KEY,
    },
)
"""The state keys only the monitor writes: every key it adds to the agent's state but
`monitor_log`, which Deep Agents' `task` tool returns from a subagent [@deepagents2026]."""
