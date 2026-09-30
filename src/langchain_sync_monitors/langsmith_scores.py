"""Suspicion scores as LangSmith feedback on each step's run, through LangSmith's REST API."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Final, Literal, TypedDict

import httpx
from pydantic import BaseModel, SecretStr, TypeAdapter, ValidationError

from langchain_sync_monitors.monitors.openrouter_decisions import check_key_characters
from langchain_sync_monitors.score_requests import (
    REQUEST_TIMEOUT_SECONDS,
    is_rate_limited,
    read_environment_value,
    read_pause_seconds,
    send_request,
)
from langchain_sync_monitors.scores import DeliveryReport, LangSmithCredentials, PendingScore

logger = logging.getLogger(__name__)

LANGSMITH_ENDPOINT: Final = "https://api.smith.langchain.com"
"""LangSmith's default endpoint, as its SDK has it when `LANGSMITH_ENDPOINT` is unset."""

type ClientFactory = Callable[[LangSmithCredentials], httpx.Client]
"""Builds the HTTP client that reaches LangSmith through one connection."""


def read_langsmith_credentials() -> LangSmithCredentials | None:
    """Read LangSmith's credentials from the variables its SDK reads, or None without a key.

    The key is `LANGSMITH_API_KEY`, or `LANGCHAIN_API_KEY`; the endpoint
    `LANGSMITH_ENDPOINT`, or `LANGCHAIN_ENDPOINT`; the workspace
    `LANGSMITH_WORKSPACE_ID`, or `LANGCHAIN_WORKSPACE_ID` [@langsmithsdk2026].
    """
    api_key = read_environment_value("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY")
    if api_key is None:
        return None
    endpoint = read_environment_value("LANGSMITH_ENDPOINT", "LANGCHAIN_ENDPOINT")
    return LangSmithCredentials(
        api_key=SecretStr(api_key),
        endpoint=(endpoint or LANGSMITH_ENDPOINT).rstrip("/"),
        workspace_id=read_environment_value("LANGSMITH_WORKSPACE_ID", "LANGCHAIN_WORKSPACE_ID"),
    )


def read_text_attribute(owner: object, *, name: str) -> str | None:
    """Return a public attribute that holds non-blank text, trimmed, or None for anything else."""
    value = getattr(owner, name, None)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def read_client_connection(client: object) -> LangSmithCredentials | None:
    """Return the connection a tracer's LangSmith client sends through, else the environment's.

    LangSmith's `Client` exposes `api_url`, `api_key` and `workspace_id`
    [@langsmithsdk2026], so the feedback goes where the trace goes. Only a
    client that gives both a URL and a key is read, so that no key is sent to
    another client's endpoint; any other falls back to the environment, as a
    client built with no settings reads it. A key no HTTP header may carry
    raises `ConfigurationError`, naming no part of it.
    """
    api_url = read_text_attribute(client, name="api_url")
    api_key = read_text_attribute(client, name="api_key")
    if api_url is None or api_key is None:
        return read_langsmith_credentials()
    check_key_characters(api_key, source="the LangSmith tracer's API key")
    return LangSmithCredentials(
        api_key=SecretStr(api_key),
        endpoint=api_url.rstrip("/"),
        workspace_id=read_text_attribute(client, name="workspace_id"),
    )


def build_langsmith_headers(connection: LangSmithCredentials) -> dict[str, str]:
    """Return the headers that authenticate with the key, and name the workspace when one is set."""
    headers = {"x-api-key": connection.api_key.get_secret_value()}
    if connection.workspace_id is not None:
        headers["X-Tenant-Id"] = connection.workspace_id
    return headers


def build_langsmith_client(connection: LangSmithCredentials) -> httpx.Client:
    """Return a client for the connection's endpoint, with its authentication headers."""
    return httpx.Client(
        base_url=connection.endpoint,
        headers=build_langsmith_headers(connection),
        timeout=REQUEST_TIMEOUT_SECONDS,
    )


class FeedbackSource(TypedDict):
    """Who wrote a LangSmith feedback: a model, as for an evaluator's judgement."""

    type: Literal["model"]


class LangSmithFeedback(TypedDict):
    """The body of LangSmith's create-feedback request, as the library sends it."""

    id: str
    run_id: str
    session_id: str
    key: str
    score: float
    feedback_source: FeedbackSource
    extend_trace_retention: bool


class LangSmithProject(BaseModel):
    """The part of a LangSmith project the sender reads: its id and name."""

    id: str
    name: str | None = None


PROJECTS_ADAPTER = TypeAdapter(list[LangSmithProject])
"""Validates the list of projects `GET {endpoint}/sessions` returns [@pydantic2026]."""


def build_feedback(score: PendingScore, *, project_id: str) -> LangSmithFeedback:
    """Return the feedback that writes the score on its step's run."""
    return LangSmithFeedback(
        id=str(score.score_id),
        run_id=str(score.step_id),
        session_id=project_id,
        key=score.name,
        score=score.value,
        feedback_source=FeedbackSource(type="model"),
        extend_trace_retention=False,
    )


def record_feedback_answer(
    report: DeliveryReport,
    *,
    score: PendingScore,
    response: httpx.Response,
) -> None:
    """Record the score by LangSmith's answer: written, waiting for its run, or refused.

    A `409` means feedback with the score's fixed id exists already. A `404`
    means the run is not ingested yet, which LangSmith's SDK retries for the
    same reason [@langsmithsdk2026].
    """
    status = response.status_code
    if response.is_success or status == httpx.codes.CONFLICT:
        report.written.append(score)
    elif status == httpx.codes.NOT_FOUND:
        report.waiting.append(score)
    else:
        report.refused.append(score)
        report.refusal = f"LangSmith answered HTTP {status}"


def read_project_id(response: httpx.Response, *, name: str) -> str | None:
    """Return the id of the project with this name in a projects answer, or None, logged."""
    if not response.is_success:
        logger.warning(
            "score export: LangSmith answered the project lookup with HTTP %d",
            response.status_code,
        )
        return None
    try:
        projects = PROJECTS_ADAPTER.validate_json(response.content)
    except ValidationError:
        logger.warning("score export: LangSmith's project lookup answered in an unknown shape")
        return None
    return next((project.id for project in projects if project.name == name), None)


def group_by_connection(
    scores: Sequence[PendingScore],
) -> dict[LangSmithCredentials | None, list[PendingScore]]:
    """Return the scores by the connection each one is sent through, in their order."""
    groups: dict[LangSmithCredentials | None, list[PendingScore]] = {}
    for score in scores:
        groups.setdefault(score.connection, []).append(score)
    return groups


class LangSmithFeedbackSender:
    """Writes each score as feedback on its step's run, one request per score.

    Each score goes through its own connection: the endpoint, key and
    workspace of the LangSmith client its tracer sends the run through, so
    the feedback lands where the trace does. The sender keeps one HTTP client
    per connection, built by `build_client`. The step's run id is the
    library's own, so no lookup finds it. Each score is one
    `POST {endpoint}/feedback`, the path LangSmith's SDK posts to, relative
    to the endpoint [@langsmithsdk2026; @langsmith2026api], with:

    - `id`, the score's fixed id, so a retry never adds a second feedback;
    - `run_id`, `key` and `score`: the step, `<label>_suspicion` and the value;
    - `session_id`, the id of the project the run was traced to, which the
      endpoint requires [@langsmith2026smithdbfeedback]. It is found once per
      connection and project through `GET {endpoint}/sessions?name=...`;
    - `feedback_source` of type `model`, as langchain-core's evaluator
      callback writes it [@langchaincore2026];
    - `extend_trace_retention` false. LangSmith's retention docs say that
      feedback sent with it true moves the whole trace to extended retention,
      which costs more, and the endpoint's default is true
      [@langsmith2026retention].

    With the extension off, LangSmith's SDK also sends one request per
    feedback [@langsmithsdk2026]. No text, the judge's reason included,
    leaves the process.
    """

    def __init__(self, *, build_client: ClientFactory = build_langsmith_client) -> None:
        self.build_client = build_client
        self.clients: dict[LangSmithCredentials, httpx.Client] = {}
        self.project_ids: dict[tuple[LangSmithCredentials, str], str] = {}

    def send(self, scores: Sequence[PendingScore]) -> DeliveryReport:
        """Write the scores, connection by connection, until one asks for a pause."""
        report = DeliveryReport()
        for connection, group in group_by_connection(scores).items():
            if connection is None:
                report.refused.extend(group)
                report.refusal = "the score names no LangSmith connection"
            elif report.pause_seconds is not None:
                report.waiting.extend(group)
            else:
                self.send_through(connection, scores=group, report=report)
        return report

    def send_through(
        self,
        connection: LangSmithCredentials,
        *,
        scores: Sequence[PendingScore],
        report: DeliveryReport,
    ) -> None:
        """Write each score whose project is known; stop at a failed request or a `429`."""
        report.pause_seconds = self.find_project_ids(
            connection, projects={score.project or "" for score in scores}
        )
        if report.pause_seconds is not None:
            report.waiting.extend(scores)
            return
        for position, score in enumerate(scores):
            project_id = self.project_ids.get((connection, score.project or ""))
            if project_id is None:
                report.waiting.append(score)
                continue
            response = self.post_feedback(score, connection=connection, project_id=project_id)
            if response is None or is_rate_limited(response):
                # The rest wait too: the service is unreachable, or asks for a pause.
                report.waiting.extend(scores[position:])
                report.pause_seconds = None if response is None else read_pause_seconds(response)
                return
            record_feedback_answer(report, score=score, response=response)

    def read_client(self, connection: LangSmithCredentials) -> httpx.Client:
        """Return the connection's HTTP client, built the first time it is needed."""
        if connection not in self.clients:
            self.clients[connection] = self.build_client(connection)
        return self.clients[connection]

    def post_feedback(
        self,
        score: PendingScore,
        *,
        connection: LangSmithCredentials,
        project_id: str,
    ) -> httpx.Response | None:
        """Post the score's feedback, retried on transient failures; None when it still failed."""
        client = self.read_client(connection)
        body = build_feedback(score, project_id=project_id)
        return send_request(client, request=client.build_request("POST", "/feedback", json=body))

    def find_project_ids(
        self,
        connection: LangSmithCredentials,
        *,
        projects: set[str],
    ) -> float | None:
        """Look up the id of each project not yet known; return the pause a `429` asks for."""
        client = self.read_client(connection)
        for project in sorted(projects):
            if (connection, project) in self.project_ids:
                continue
            request = client.build_request("GET", "/sessions", params={"name": project, "limit": 1})
            response = send_request(client, request=request)
            if response is None:
                continue
            if is_rate_limited(response):
                return read_pause_seconds(response)
            project_id = read_project_id(response, name=project)
            if project_id is not None:
                self.project_ids[(connection, project)] = project_id
        return None

    def close(self) -> None:
        """Close every HTTP client the sender built."""
        for client in self.clients.values():
            client.close()
