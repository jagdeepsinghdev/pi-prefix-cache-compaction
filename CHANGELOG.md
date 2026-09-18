# Changelog

## 0.2.0 — 2026-09-17

First public release.

- Compaction re-sends the last captured provider request (Anthropic Messages wire format) plus one summarize instruction, so the server's prefix cache covers the whole history and only the instruction is prefilled.
- Post-compaction warm-up: a 1-token request with the new context (Pi's own converter, captured system prompt / tools / thinking settings) so the next turn starts from cache.
- The summary request may only override `max_tokens` and `stream`; every field that can reach the rendered prompt (`system`, `tools`, `tool_choice`, `messages`, `thinking`, `reasoning`, `reasoning_effort`, `output_config`, `chat_template_kwargs`, `model`, `mm_processor_kwargs`, `documents`) is copied verbatim and verified by `assertPrefixPreserved` before sending. A mismatch falls back to Pi's default instead of silently paying a cold re-prefill.
- No option to change thinking for the summary, by design: the toggle differs per model family (Qwen3 `enable_thinking`, DeepSeek-V3.1/Granite `thinking`, Gemma 4 `reasoning_effort`, Holo2 `thinking:false`) and vLLM derives `enable_thinking` from `reasoning_effort`, so changing it is never cache-safe in general.
- Captures are tied to the model (provider/baseUrl/id fingerprint): switching models mid-session falls back to Pi's default until the new model sends a real turn.
- After a Pi-default compaction the capture is marked stale and is never used for a summary again until a real turn re-anchors it; the warm-up still runs.
- Falls back to Pi's default on overflow recovery, missing capture, too little window room, tool use, a stream cut mid-summary (no stop reason and no `message_stop`), or any HTTP/stream error. A compaction whose signal is already aborted cancels cleanly.
- Summary request uses `node:http`, not `fetch`, so a response idle for more than 300 s (long prefill or thinking on a local GPU) is not aborted.
- Compaction usage (`input`/`output`/`cacheRead`/`cacheWrite`, `totalTokens`, zero `cost`) is returned so it counts toward Pi's session totals.
- Captures are released on session shutdown; captured headers are shallow-copied.
- `/prefix-compaction` status command (re-reads config); global and project JSON config; malformed config files warn instead of being silently ignored.
- Smoke test (`scripts/rpc-smoke.mjs`) runs Pi in RPC mode against a real server, with `--thinking <level>` and `--no-warmup`.
