# Suspicion scores through the tools' public endpoints: addendum

This note records why the library writes the scores itself, which public
endpoints and fields it uses for each tool, what was checked against the live
services, and where the approved plan was corrected while building it.

Research and checks: 30 September and 1 October 2026.

Contents:

- [The decision](#the-decision)
- [LangSmith](#langsmith)
- [Langfuse](#langfuse)
- [The worker](#the-worker)
- [The live check](#the-live-check)
- [Corrections to the plan](#corrections-to-the-plan)
- [Not yet checked](#not-yet-checked)

## The decision

- **Rejected: a Langfuse code evaluator.** `canonical-integration.md`
  recommended one for Langfuse. The owner rejected it: every user would have
  to install an evaluator and a rule in each project, and the library could
  not keep them in step with its own spans.
- **Chosen: an opt-in writer in the library.** `MonitorMiddleware` takes
  `export_scores`, a set of `Tracer` members, empty by default. When a tool is
  named, each committed step's highest suspicion goes to it as
  `<label>_suspicion`, on the `monitor step` span, from one background worker
  per process.
- **Owner decisions kept.** D3: no trace-retention extension, so no change to
  the LangSmith bill. D6: the judge's reason is never sent, since neither SDK
  masks feedback or score comments; only numbers and ids leave the process.
- **No SDK.** Both writers call the public HTTP APIs with httpx and retry with
  stamina, so the library takes no new dependency and no SDK floor. The
  LangSmith SDK's writer would have needed langsmith 0.6.7 or later.

## LangSmith

- **Endpoint.** `POST {LANGSMITH_ENDPOINT}/feedback`, the path LangSmith's
  SDK posts to, relative to the endpoint (langsmith 0.14.1,
  `Client.create_feedback`). The OpenAPI document lists it as
  `/api/v1/feedback`; a self-hosted `LANGSMITH_ENDPOINT` ends in `/api/v1`,
  and the cloud endpoint serves the SDK's path. Checked live on the cloud.
- **Fields sent.**

  | Field | Value |
  |---|---|
  | `id` | `uuid5` of the tool, project, step id and score name, so a retry never duplicates |
  | `run_id` | The step span's run id, which is `monitor_step_id` |
  | `session_id` | The project's id. Required since the SmithDB migration; found once per project with `GET {endpoint}/sessions?name=...` |
  | `key` | `<label>_suspicion` |
  | `score` | The step's highest suspicion |
  | `feedback_source` | `{"type": "model"}`, as langchain-core's evaluator callback writes it |
  | `extend_trace_retention` | `false`, sent explicitly: the endpoint's default is `true`, and LangSmith extends a trace's retention when an API call passes `true` |

- **Not needed: `trace_id` and `start_time`.** The live feedback landed on the
  right runs without either. `start_time` is a lookup key under SmithDB, and
  only the tracer knows its exact value, so it is left out.
- **Answers.** `409` is read as written, `404` as a run not ingested yet,
  which the SDK also retries, and `429` pauses every call to LangSmith for its
  `Retry-After`. A repeat post with the same id returned `200` live and left
  one feedback, an upsert, so the `409` rule is a precaution.
- **Rate limits.** `POST /feedbacks*` allows 5,000 requests a minute per key,
  and the SDK itself posts one request per feedback when the extension is
  off.

## Langfuse

- **Finding the step.** `GET /api/public/v2/observations`, the only real-time
  read path, with `fields=core,basic,metadata`.
  - One query per window serves every waiting step: name `monitor step` and
    `startTime` at or after the earliest waiting step's start, less 5 seconds.
    The ids are matched in the library.
  - A multi-value filter on the metadata key does not work: a
    `categoryOptions` `any of` filter on `metadata` returned `400`, "Expected
    filter type 'stringObject'" (live).
  - The step's start comes from its id, a version 7 UUID made before the span
    starts, so no timestamp needs to travel with the score.
  - Pages follow `meta.cursor`, at most 5 a window.
- **Writing the scores.** One `POST /api/public/ingestion` per window, with a
  `score-create` event per score.
  - Langfuse's own SDK 4.16.0 sends scores this way, in batches
    (`ScoreIngestionConsumer`).
  - The endpoint is deprecated, but keeps taking `score-create` events after
    it stops taking any other on 16 November 2026.
  - `POST /api/public/scores`, which the docs prefer, takes one score per
    request from the general rate limit, 30 requests a minute on the Hobby
    plan. The lookups spend that same bucket.
  - The body is the score's fixed `id`, `traceId`, `observationId`, `name`,
    `value`, `dataType` `NUMERIC`, and the observation's `environment`. Each
    event gets a new envelope id.
  - The answer is `207`, read event by event: a `4xx` refuses the score, and
    a `429` or `5xx` leaves it waiting.
- **Idempotency.** An event re-sent live with the same score id and a new
  envelope id left one score.
- **Credentials.** `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and
  `LANGFUSE_BASE_URL` before `LANGFUSE_HOST`, as the SDK reads them.

## The worker

- **Putting a score.** A monitor puts the score on a queue and never waits or
  raises. A `LangChainTracer` among the step span's handlers sends to
  LangSmith, in the tracer's project. A handler whose class comes from the
  `langfuse` package sends to Langfuse. The library never imports `langfuse`.
- **Windows.** The worker wakes every 10 seconds and hands each tool all its
  waiting scores at once.
  - A score that cannot be written yet waits for the next window.
  - It is dropped, with a warning, after 300 seconds.
  - A `429` holds every call to that tool until its `Retry-After`.
  - A request is retried twice at most, within 10 seconds, on a transport
    failure or a `5xx`.
- **At exit.** The `atexit` hook drains for up to 30 seconds, one window
  every 10 seconds, then logs and drops what is left. The thread is a daemon,
  so a drain that overruns never keeps the process alive.

## The live check

- **Where.** LangSmith project `langchain-sync-monitors-docs-check`, and the
  owner's Langfuse US cloud project.
- **How.** Each run ran in a child process that exited right after its run,
  so the exit drain wrote whatever the windows had not.
- **Two kinds of run, synthetic content only.**
  - A scripted model and a keyword judge: a flagged step judged 0.9, then a
    step judged 0.1.
  - Real cheap OpenRouter models from the live harness:
    `deepseek/deepseek-v4.1-flash` as the agent and `xiaomi/mimo-v2.6-pro` as
    an `LLMMonitor` judge. Both steps were judged 0.0.

| Check | Result |
|---|---|
| LangSmith feedback, both modes, both kinds of run | One `monitor_suspicion` feedback on each of the 8 `monitor step` runs, with the right value, source `model` |
| Langfuse scores, both modes, both kinds of run | One `monitor_suspicion` score on each of the 8 `monitor step` observations, matched by `monitor_step_id`, with the right value, environment `default` |
| When LangSmith took the feedback | About 1.5 seconds after the run, in the drain's first window, with no `404` |
| When Langfuse scores were written, after the run | 12.5 seconds (scripted, `invoke`); 25 seconds (scripted, `ainvoke`); 19 and 22 seconds (real models) |
| A LangSmith feedback posted again with its id | `200`, and still one feedback |
| A Langfuse score sent again with its id | `207` with success, and still one score |
| A run that failed with `GraphRecursionError` part way | Its three steps' scores were written in both tools, during the run and at exit |
| Spend | $0.0026 measured for the two real-model runs; the failed first attempt was not metered and made about 7 calls at about $0.0003 each |

## Corrections to the plan

- **Retention.** The pseudocode says `feedback_source_type="model"` avoids the
  retention extension. It does not: `feedback_source` only labels who wrote
  the feedback. `extend_trace_retention: false` avoids the extension.
- **No metadata `any of`.** The plan's `metadata_is_any_of` lookup does not
  exist in the API. The lookup filters by name and start time instead, and
  matches the ids in the library.
- **Where the scores are posted.** They go through `/api/public/ingestion`
  as `score-create` events, not `POST /api/public/scores`, for the rate-limit
  reason above.
- **No LangSmith SDK or extra.** The plan expected the LangSmith writer to
  need langsmith 0.6.7 or later as an extra. The library posts directly, so it
  needs neither.
- **A project lookup.** LangSmith now requires `session_id`, so the writer
  looks up each project's id once.

## Not yet checked

- A LangSmith `404` for a run not ingested yet: the cloud accepted every
  feedback at once.
- LangSmith's OpenTelemetry mode (`LANGSMITH_OTEL_ENABLED`), self-hosted
  LangSmith and Langfuse, and Langfuse's EU region.
- A tracer built with its own `Client(api_url=...)`, or a Langfuse handler
  built with its own `public_key`: the writer uses the environment's
  credentials, so such a run's scores go where the variables point.
- What each tool's interface shows for the scores: the checks read them back
  through the APIs.
