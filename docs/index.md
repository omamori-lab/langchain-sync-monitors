# langchain-sync-monitors

This page introduces the library and points to the rest of the documentation.

langchain-sync-monitors adds control monitors to LangChain agents and Deep
Agents. A monitor judges every step an agent proposes before anything runs. A
control protocol decides what happens with that judgement: the step runs, the
agent is sampled again, a trusted model writes the step instead, the agent is
told to change course, or the run stops.

The library is under construction. The sections below fill in as each part
lands.

- [Tutorials](tutorials/index.md) walk through a first monitored agent.
- [How-to guides](how-to/index.md) solve one task each, such as monitoring
  every subagent of a Deep Agent.
- [Reference](reference/index.md) lists the public API.
- [Explanation](explanation/index.md) covers the design and where each idea
  comes from.
