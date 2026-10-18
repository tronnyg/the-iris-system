# ADR-0005: One virtualenv per voice engine, JSON-lines workers

- **Status:** Accepted
- **Date:** 2026-10-17
- **Supersedes:** none
- **Related:** [architecture/04](../architecture/04-speaking.md#6-tts-workers)

## Context

Kokoro and Chatterbox need different torch versions (Chatterbox pins torch 2.6.0; both need CUDA 12.6
wheels for Pascal), and neither should be imported by the core (faster-whisper, audio, the web server).
VRAM must be fully returned when a voice is not used.

## Decision

Each engine has its own venv (`.venvs/kokoro`, `.venvs/chatterbox`) and runs as a child process
`python -m iris.tts.worker NAME`, speaking JSON lines over stdin/stdout (load, synth with the audio
written to a .npy file, ping, quit). The `iris` package stays import-light so the workers can import it.
An engine is one module with a fixed contract.

## Consequences

* Adding an engine is one module plus an install function; nothing else changes.
* Killing the worker frees its VRAM; a crashing engine cannot take the session down.
* Small per-chunk overhead (a file write and read); negligible next to synthesis time.

## Alternatives considered

* One shared venv: dependency conflicts. A TTS HTTP server: more moving parts for one client.

## Evidence

`iris/tts/worker.py`, `iris/tts/client.py`, `tests/test_worker.py`.
