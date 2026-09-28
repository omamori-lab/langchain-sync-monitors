# Changelog

Notable changes to langchain-sync-monitors. The format follows the spirit of
[Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/).

The public API is the set of names exported from `langchain_sync_monitors`
(monitors, control protocols, the middleware and their constructor keywords)
and the shape of the `monitor_log` records. A release that could turn a working
integration into a failing one is a breaking change.

## [Unreleased]

### Added

- Project scaffold: packaging, the gate script, CI, pre-commit hooks, the
  documentation skeleton and the bibliography with its citation test.
- Two explanation pages: how the library is built, with diagrams of a
  monitored step, the protocols, auto mode and subagents; and the LangChain
  findings, the upstream bugs and design smells we met, each with its evidence
  and how this library avoids it.
