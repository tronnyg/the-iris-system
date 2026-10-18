# ADR-0021: Web UI on the standard library with SSE

- **Status:** Accepted
- **Date:** 2026-10-17
- **Supersedes:** none
- **Related:** [architecture/10](../architecture/10-web-ui.md)

## Context

The user wanted a full-screen, futuristic (JARVIS-like, Gruvbox) interface instead of the terminal, without
adding a web stack, and without exposing the session to other web pages in the browser.

## Decision

A `ThreadingHTTPServer` on 127.0.0.1 serves static files, a Server-Sent Events stream from the session's
event bus and a command endpoint. Every API call needs a per-session token; Host and Origin are checked;
a strict CSP forbids inline scripts. The page is plain HTML/CSS/JS; the centrepiece is an ASCII neural
"eye" drawn on a canvas (30 fps cap, paused when hidden).

## Consequences

* No dependencies, no build step; works offline.
* Publishing never blocks the audio threads (bounded queues, oldest event dropped).

## Alternatives considered

* A framework (FastAPI, React): new dependencies for one local page. WebSockets: SSE is enough one way.

## Evidence

Commits 1117b68, 1ff0418, 8d29aa1, 3f8ec2d and the neural-core commits.
