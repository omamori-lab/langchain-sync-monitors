# langchain-sync-monitors

This page introduces the library and points to the rest of the documentation.

langchain-sync-monitors adds control monitors to LangChain agents and Deep
Agents. A monitor judges every step an agent proposes before anything runs. A
control protocol decides what happens with that judgement: the step runs, the
agent is sampled again, a trusted model writes the step instead, the agent is
told to change course, or the run stops.

```python
from langchain.agents import create_agent
from langchain_sync_monitors import (
    DeferToResample,
    DeferToTrustedModel,
    LLMMonitor,
    MonitorMiddleware,
)

middleware = MonitorMiddleware(
    monitor=LLMMonitor(model="openrouter:xiaomi/mimo-v2.6-pro"),
    protocol=DeferToResample(
        fallback=DeferToTrustedModel(trusted_model="openrouter:xiaomi/mimo-v2.6-flash"),
        defer_threshold=0.6,
        audit_threshold=0.9,
    ),
)
agent = create_agent(
    "openrouter:deepseek/deepseek-v4.1-flash", tools=tools, middleware=[middleware]
)
result = agent.invoke({"messages": [{"role": "user", "content": "Summarise q3.md."}]})
print(result["monitor_log"])
```

Every model is a parameter: the untrusted agent, the monitor and the trusted
fallback are whichever models you pass in. Thresholds default to 0.6 and 0.9
and warn until you set values calibrated on honest runs of your own agent.

- [Tutorials](tutorials/index.md) walk through a first monitored agent.
- [How-to guides](how-to/index.md) solve one task each, such as monitoring
  every subagent of a Deep Agent.
- [Reference](reference/index.md) lists the public API.
- [Explanation](explanation/index.md) covers the design and where each idea
  comes from.
