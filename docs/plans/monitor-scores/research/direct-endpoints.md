# Suspicion scores through the tools' public endpoints: addendum

This note records why the library writes the scores itself, which public
endpoints and fields it uses for each tool, what was checked against the live
services, and where the approved plan was corrected while building it.

Research and checks: 30 September and 1 October 2026.

Contents:

- [The decision](#the-decision)
- [LangSmith](#langsmith)
- [Langfuse](#langfuse)
- [Where the scores go](#where-the-scores-go)
- [The worker](#the-worker)
- [The live check](#the-live-check)
- [The adversarial review](#the-adversarial-review)
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
  the LangSmith bill. The tier is not visible through the API, so this rests
  on sending `extend_trace_retention: false` and on LangSmith's retention
  docs. D6: the judge's reason is never sent, since neither SDK
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
  - One query per window serves every waiting step: name `monitor step`, and
    `startTime` from the earliest waiting step's start less 5 seconds to the
    latest one's plus 5 seconds. The ids are matched in the library. Live,
    the bounded query returned exactly the 4 steps it was built for.
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
  - The owner accepted the ingestion endpoint for this reason on
    1 October 2026.
  - The body is the score's fixed `id`, `traceId`, `observationId`, `name`,
    `value`, `dataType` `NUMERIC`, and the observation's `environment`. Each
    event gets a new envelope id.
  - The answer is `207`, read event by event: a `4xx` refuses the score, and
    a `429` or `5xx` leaves it waiting.
- **Idempotency.** An event re-sent live with the same score id and a new
  envelope id left one score.
- **Credentials.** `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and
  `LANGFUSE_BASE_URL` before `LANGFUSE_HOST`, as the SDK reads them.

## Where the scores go

The owner's rule, 1 October 2026: scores go where the traces go, never to
another project or host.

- **LangSmith follows the tracer.**
  - The feedback goes to the `LangChainTracer`'s project, through its client's
    endpoint, key and workspace. These are read from the client's public
    `api_url`, `api_key` and `workspace_id` (langsmith 0.14.1).
  - Only a client that gives both a URL and a key is read. Any other, such as
    a test double, falls back to the environment whole, as a client built
    with no settings reads it. A key is never sent to another client's
    endpoint.
  - The sender keeps one HTTP client per connection, and looks up each
    project's id once per connection.
- **Langfuse cannot be read, and needs not be.**
  - Langfuse 4.16.0 exposes no public accessor for a handler's keys or host.
    The handler keeps its client in `_langfuse_client`, and the client keeps
    its keys and base URL private.
  - The writer uses the environment's keys. It writes a score only on an
    observation that its own lookup, in the project those keys reach, found
    for the step's exact `monitor_step_id`.
  - A handler built with other keys, or another host, traces to a project
    that lookup never sees, so nothing is written there, or anywhere.
  - The scores wait, are given up after 300 seconds or dropped at the end of
    the drain, and the worker says once per process what usually causes it.
  - The cost to such a run: its Langfuse scores are lost, and at exit the
    drain runs its full 30 seconds.
  - An early skip would need a private read or extra calls to Langfuse's
    SDK, which the library does not make.
- **The build check.** It still needs `LANGSMITH_API_KEY`, since a tracer's
  own client is known only during a run. A user whose only LangSmith key
  lives in an explicit `Client` gets `ConfigurationError`.
- **Replicas.** With `LANGSMITH_RUNS_ENDPOINTS`, feedback goes only to the
  client's `api_url`.

## The worker

- **Putting a score.** A monitor puts the score on a queue and never waits or
  raises. A `LangChainTracer` among the step span's handlers sends to
  LangSmith, as the section above says. A handler whose class comes from the
  `langfuse` package sends to Langfuse. The library never imports `langfuse`.
- **Windows.** The worker wakes every 10 seconds and hands each tool all its
  waiting scores at once, Langfuse first, each tool in a `try` of its own.
  - A score that cannot be written yet waits for the next window.
  - It is dropped, with a warning, after 300 seconds.
  - A `429` holds every call to that tool until its `Retry-After`, read up to
    an hour and held at most 300 seconds.
  - A request is sent at most twice, so retried once, within 10 seconds, on a
    transport failure or a `5xx`.
  - A sender that cannot be built, for any reason, drops its tool's scores,
    with one warning, and the other tool and the give-up go on.
- **LangSmith posts.** Up to 6 posts are in flight at once, on daemon threads
  started with the sender, and one send starts no new batch after 5 seconds.
  - LangSmith has no batch endpoint for this feedback: `POST /runs/multipart`
    takes feedback parts only with the trace's id, which the library does not
    know.
  - `concurrent.futures` does not fit: its threads are not daemons, and its
    exit hook runs before `atexit` and refuses work after it. No thread can
    start during interpreter shutdown, so a sender first built during the
    exit drain posts one at a time.
  - The sender keeps a client for at most 8 connections, the least recently
    used closed first.
- **Langfuse lookups.** Pages come newest first, at most 3 per lookup.
  - Steps whose scores have waited under 60 seconds are looked up every
    window, over their own window. Older ones are looked up apart, together,
    at most once a minute, so that one step Langfuse never ingests cannot
    keep every window five minutes wide.
  - One process asks the general rate limit, which the whole organisation
    shares, for at most 21 lookups a minute, 7 when each fits a page, and
    twice the fresh ones during the drain. The writes go to the ingestion
    bucket.
- **Forks.** A child forked from the process forgets the parent's worker,
  its exit hook and its lock, and starts its own. Queuing a score takes no
  lock once the worker runs. A `multiprocessing` child started by fork
  leaves through `os._exit`, so its waiting scores are dropped.
- **At exit.** The `atexit` hook drains for up to 30 seconds, one window
  every 5 seconds, then logs and drops what is left. The thread is a daemon,
  so a drain that overruns never keeps the process alive.
  - The drain's window is half the normal one, since a step Langfuse has just
    ingested is found one window after it appears. The owner chose 5 seconds
    and kept the 30-second limit on 1 October 2026.
  - A process that exits right after its last step may still drop Langfuse
    scores that wait.
  - A tool paused past the drain's end has its scores dropped at once.
- **When waiting scores are lost.** When the process ends without running
  `atexit`: `os._exit`, a `multiprocessing` fork child, SIGKILL, SIGTERM
  without a handler, and a killed Jupyter kernel.

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
| When Langfuse scores were written, after the run, with 10-second drain windows | 12.5 seconds (scripted, `invoke`); 25 seconds (scripted, `ainvoke`); 19 and 22 seconds (real models) |
| The same with 5-second drain windows, scripted, 1 October 2026 | 13.6 seconds (`invoke`) and 15.5 seconds (`ainvoke`); each process exited at once after. The feedback went through the LangSmith client's public attributes, one per step |
| A LangSmith feedback posted again with its id | `200`, and still one feedback |
| A Langfuse score sent again with its id | `207` with success, and still one score |
| A run that failed with `GraphRecursionError` part way | Its three steps' scores were written in both tools, during the run and at exit |
| The tracing guide's suspicion queries | LangSmith runs filtered with `and(eq(feedback_key, "monitor_suspicion"), gte(feedback_score, 0.5))`, and Langfuse's v3 scores with `name`, `dataType` `NUMERIC` and `valueMin` 0.5, each returned exactly the two steps judged 0.9 |
| Spend | $0.0026 measured for the two real-model runs; the failed first attempt was not metered and made about 7 calls at about $0.0003 each |

## Corrections to the plan

- **Retention.** The pseudocode says `feedback_source_type="model"` avoids the
  retention extension. It does not: `feedback_source` only labels who wrote
  the feedback. `extend_trace_retention: false` avoids the extension.
- **No metadata `any of`.** The plan's `metadata_is_any_of` lookup does not
  exist in the API. The lookup filters by name and a start-time window
  instead, and matches the ids in the library.
- **Where the scores are posted.** They go through `/api/public/ingestion`
  as `score-create` events, not `POST /api/public/scores`, for the rate-limit
  reason above.
- **No LangSmith SDK or extra.** The plan expected the LangSmith writer to
  need langsmith 0.6.7 or later as an extra. The library posts directly, so it
  needs neither.
- **A project lookup.** LangSmith now requires `session_id`, so the writer
  looks up each project's id once.

## The adversarial review

An independent review of the pull request, on 1 October 2026, confirmed the
core contract live: every step had exactly one score, with the right value,
in both tools, and the request bodies carried no text. It also found:

- **LangSmith's OpenTelemetry mode loses every score.** There the run id is
  derived from the span id, so it never equals `monitor_step_id`, yet
  LangSmith answers `200` and the worker logged the score as written. The
  writer now skips a client whose public `tracing_mode` is `otel`, with one
  warning. `hybrid` still sends runs with their own ids.
- **LangSmith accepts feedback on a run it never ingested,** with `200`, so a
  run its sampling rate dropped, or one traced with tracing turned off, loses
  its score without a sign. The writer skips a run whose
  `tracing_context(enabled=False)` turns tracing off; sampling cannot be seen.
- **`LANGFUSE_TRACING_ENABLED=false`** made every exit wait the full drain.
  The writer now reads it as the SDK does.
- **A sender that failed to build**, as with a malformed `LANGFUSE_HOST`,
  stopped every window; a fork while another thread held the start lock
  deadlocked the child's step; one LangSmith request per round trip capped
  the posts at about 250 a minute; one never-ingested Langfuse step kept the
  lookup window five minutes wide; and a long `Retry-After` held the drain
  to its end. Each is fixed as the sections above describe.

## Not yet checked

- A LangSmith `404` for a run not ingested yet: the cloud accepted every
  feedback at once, even on run ids it never ingested.
- Self-hosted LangSmith and Langfuse, and Langfuse's EU region.
- Feedback through a LangSmith client other than the environment's, against
  the live service: the tests check the routing with a stand-in client.
- What each tool's interface shows for the scores: the checks read them back
  through the APIs.
