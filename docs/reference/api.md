---
heading_numbers: false
---

# API

This reference is generated from the docstrings of the public package, every
name `langchain_sync_monitors` exports, grouped by what it is for.

## Middleware

::: langchain_sync_monitors.MonitorMiddleware
    options:
      members: false

::: langchain_sync_monitors.check_monitor_placement
    options:
      members: false

::: langchain_sync_monitors.monitor_subagents
    options:
      members: false

::: langchain_sync_monitors.MonitorState
    options:
      members:
        - monitor_log
        - monitor_delegation
        - monitor_task_messages
        - monitor_seen_human_messages
        - monitor_run_inputs
        - monitor_rewritten_inputs
        - monitor_run_open
        - monitor_inputs_at_halt
        - monitor_subagent_returns

## Protocols

::: langchain_sync_monitors.ControlProtocol
    options:
      members:
        - decide

::: langchain_sync_monitors.TrustedMonitoring
    options:
      members: false

::: langchain_sync_monitors.DeferToTrusted
    options:
      members: false

::: langchain_sync_monitors.DeferToResample
    options:
      members: false

::: langchain_sync_monitors.Resampling
    options:
      members:
        - SEQUENTIAL
        - PARALLEL

::: langchain_sync_monitors.AutoMode
    options:
      members: false

::: langchain_sync_monitors.FeedbackTemplate
    options:
      members:
        - text
        - render

::: langchain_sync_monitors.DEFAULT_FEEDBACK_TEMPLATE
    options:
      members: false

::: langchain_sync_monitors.FeedbackVisibility
    options:
      members:
        - HIDDEN
        - IN_TRANSCRIPT

::: langchain_sync_monitors.SubagentHalt
    options:
      members:
        - STOP_SUBAGENT_ONLY
        - STOP_WHOLE_RUN

## Fallbacks

::: langchain_sync_monitors.Fallback
    options:
      members:
        - take_over

::: langchain_sync_monitors.HaltRun
    options:
      members: false

::: langchain_sync_monitors.DEFAULT_HALT_MESSAGE
    options:
      members: false

::: langchain_sync_monitors.DeferToTrustedModel
    options:
      members: false

## Thresholds

::: langchain_sync_monitors.DefaultThreshold
    options:
      members: false

::: langchain_sync_monitors.ThresholdStatistic
    options:
      members:
        - SINGLE_SAMPLE_SCORE
        - LOWEST_SAMPLE_SCORE

::: langchain_sync_monitors.resolve_threshold
    options:
      members: false

## Monitors

::: langchain_sync_monitors.Monitor
    options:
      members:
        - evaluate
        - evaluate_sync

::: langchain_sync_monitors.MonitorInput
    options:
      members:
        - history
        - proposal
        - task_author

::: langchain_sync_monitors.Verdict
    options:
      members:
        - suspicion
        - reason

::: langchain_sync_monitors.ChatModelMonitor
    options:
      members:
        - build_verdict_plan

::: langchain_sync_monitors.LLMMonitor
    options:
      members: false

::: langchain_sync_monitors.DEFAULT_MONITOR_PROMPT
    options:
      members: false

::: langchain_sync_monitors.GuardModelMonitor
    options:
      members: false

::: langchain_sync_monitors.GuardScoring
    options:
      members:
        - AUTO
        - LOG_PROBABILITIES
        - SAMPLE_FRACTION
        - HARD_LABEL

::: langchain_sync_monitors.DecisionModelMonitor
    options:
      members: false

::: langchain_sync_monitors.DecisionModel
    options:
      members:
        - estimate_probabilities
        - estimate_probabilities_sync

::: langchain_sync_monitors.OpenRouterDecisionModel
    options:
      members: false

::: langchain_sync_monitors.TypeSafeDecisionModel
    options:
      members: false

::: langchain_sync_monitors.YesNoQuestion
    options:
      members:
        - key
        - text
        - yes_means
        - no_means

::: langchain_sync_monitors.Aggregation
    options:
      members:
        - MAX
        - MIN
        - MEAN

::: langchain_sync_monitors.IS_THE_STEP_SUSPICIOUS
    options:
      members: false

::: langchain_sync_monitors.ATTACK_CRITERIA
    options:
      members: false

## Wrappers

::: langchain_sync_monitors.RepeatedMonitor
    options:
      members: false

::: langchain_sync_monitors.CalibratedMonitor
    options:
      members: false

::: langchain_sync_monitors.CascadeMonitor
    options:
      members: false

## Views

::: langchain_sync_monitors.MonitorView
    options:
      members:
        - channels
        - most_recent_entries
        - delegation_tools

::: langchain_sync_monitors.Channel
    options:
      members:
        - USER
        - REASONING
        - AGENT_TEXT
        - TOOL_CALLS
        - TOOL_RESULTS
        - SUBAGENT_REPORTS

::: langchain_sync_monitors.TaskAuthor
    options:
      members:
        - USER
        - PARENT_AGENT

## Pending steps and decisions

::: langchain_sync_monitors.PendingStep
    options:
      members:
        - sample
        - request_trusted_step
        - previous_records
        - blocks_in_thread
        - new_subagent_blocks

::: langchain_sync_monitors.SampleDrawing
    options:
      members:
        - SEQUENTIAL
        - CONCURRENT

::: langchain_sync_monitors.Sample
    options:
      members:
        - response
        - proposal
        - verdict

::: langchain_sync_monitors.BlockedAttempt
    options:
      members:
        - proposal
        - feedback

::: langchain_sync_monitors.StepDecision
    options:
      members:
        - outcome
        - response
        - samples
        - executed_sample
        - flagged
        - blocked_attempts

::: langchain_sync_monitors.Outcome
    options:
      members:
        - ALLOWED
        - RESAMPLED
        - DEFERRED_TO_TRUSTED
        - STEERED
        - HALTED

## Records and events

::: langchain_sync_monitors.StepRecord
    options:
      members:
        - agent
        - monitor
        - step_number
        - outcome
        - flagged
        - blocked_count
        - samples
        - delegation_id

::: langchain_sync_monitors.SampleRecord
    options:
      members:
        - suspicion
        - reason
        - proposal
        - executed

::: langchain_sync_monitors.Delegation
    options:
      members:
        - tool_call_id
        - delegating_agent
        - blocks_before

::: langchain_sync_monitors.MonitorStepEvent
    options:
      members:
        - type
        - record

::: langchain_sync_monitors.MonitorStepFailedEvent
    options:
      members:
        - type
        - agent
        - monitor
        - step_number
        - error
        - samples
        - delegation_id

## Errors and warnings

::: langchain_sync_monitors.MonitorError
    options:
      members: false

::: langchain_sync_monitors.ConfigurationError
    options:
      members: false

::: langchain_sync_monitors.MissingExtraError
    options:
      members: false

::: langchain_sync_monitors.SynchronousRunError
    options:
      members: false

::: langchain_sync_monitors.InvalidSuspicionError
    options:
      members: false

::: langchain_sync_monitors.UncalibratedThresholdWarning
    options:
      members: false

::: langchain_sync_monitors.CachedResampleWarning
    options:
      members: false

::: langchain_sync_monitors.ServerToolWarning
    options:
      members: false

::: langchain_sync_monitors.MonitorPlacementWarning
    options:
      members: false

::: langchain_sync_monitors.HardLabelWarning
    options:
      members: false
