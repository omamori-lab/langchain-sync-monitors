# Security policy

This page explains how to report a vulnerability in langchain-sync-monitors,
what counts as one, and what happens after you report it.

## Supported versions

The library is at 0.x. Security fixes go into the latest release only.

## Reporting a vulnerability

Please report a vulnerability privately, not in a public issue, discussion or
pull request. Use GitHub's private vulnerability reporting:

1. Open the repository's **Security** tab.
2. Choose **Report a vulnerability**.
3. Describe the problem and the versions it affects, with the smallest
   reproduction you can share: the monitor, protocol and view you configured,
   and the step or transcript that gets through. Leave out API keys and other
   secrets.

The form is also at
<https://github.com/omamori-lab/langchain-sync-monitors/security/advisories/new>.

## Scope

The library exists to stop a step before it runs, so a way around that is the
main thing to report. In scope:

- a way for an agent to get a step past a monitor or protocol that should have
  stopped it: a step that runs without being judged, a blocked, deferred or
  halted step that runs anyway, or a verdict the step itself can set, such as
  a score planted in the step's text;
- content in the transcript that changes how the monitor's view is rendered or
  how its reply is read, beyond fooling the monitor model itself;
- a failure that opens instead of closing: an error or an unreadable monitor
  reply that lets a step through;
- a subagent that runs without the monitor its parent configured, or an Auto
  Mode limit that an agent can reset;
- data reaching a stream, a log or a trace that the documentation says it does
  not reach, such as a rejected sample in `stream_mode="messages"`;
- a weakness in the release process or the published package.

Out of scope:

- a monitor model that scores a harmful step too low. Monitors are models and
  make mistakes; if the default prompt or view could do better, open an
  ordinary issue;
- vulnerabilities in LangChain, LangGraph, Deep Agents or a model provider.
  Report them to that project; tell us here if the library should work around
  one;
- attacks that need control of the application's own code or configuration,
  such as changing the middleware list.

## What to expect

We reply in the private advisory, confirm the problem with you, and agree on a
fix and a date to disclose it. The fix ships in a new release, and the
published advisory credits you unless you prefer otherwise. Please do not
disclose the problem publicly until the fix is released.
