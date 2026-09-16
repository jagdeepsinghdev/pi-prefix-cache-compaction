# Changelog

## 0.2.0 (unreleased)

- The summary request may now only override `max_tokens`, `stream` and `stream_options`; every field that can reach the rendered prompt (`system`, `tools`, `tool_choice`, `messages`, `thinking`, `reasoning`, `reasoning_effort`, `output_config`, `chat_template_kwargs`, `model`, `mm_processor_kwargs`, `documents`) is copied verbatim and verified by `assertPrefixPreserved` before sending. A mismatch falls back to Pi's default instead of silently paying a cold re-prefill.
- Removed the `disableThinking` option: the thinking toggle differs per model family (Qwen3 `enable_thinking`, DeepSeek-V3.1/Granite `thinking`, Gemma 4 `reasoning_effort`, Holo2 `thinking:false`) and vLLM derives `enable_thinking` from `reasoning_effort`, so changing it is never cache-safe in general.
- Warm-up copies the same fields from the captured request, so it warms the prefix the next real turn will send.
- Fix: a compaction whose signal was already aborted crashed the process (an `'error'` listener was registered after the abort path); it now cancels cleanly.
- Captures are tied to the model (provider/baseUrl/id fingerprint): switching models mid-session falls back to Pi's default until the new model sends a real turn.
- After falling back to Pi's default compaction the capture is marked stale and is never used for a summary again until a real turn re-anchors it. The post-compaction warm-up still runs and warms the new prefix in that case.
- Captures are released on session shutdown, and captured headers are shallow-copied so later extensions' in-place header mutations cannot change them.
- A summary stream that ends without a stop reason (server closed the connection mid-summary) now falls back to Pi's default instead of keeping a truncated checkpoint.
- The compaction result now carries the summary request's usage, so it counts toward session totals.
- `isCapturable` also filters requests whose messages wrap the history in `<conversation>` tags, so Pi's fallback summarizer is recognized even if its system prompt wording ever changes.
- `/prefix-compaction` re-reads the config files, and unparseable config files produce a warning instead of being silently ignored.

## 0.1.1 (unreleased)

- Fix: keep the captured thinking settings in the summary and warm-up requests. Forcing thinking off changed the start of the system prompt in Qwen-style templates, so thinking-on sessions missed the prefix cache entirely (245k tokens re-prefilled, then "terminated"). `disableThinking` now defaults to `false`.
- Fix: summary request uses `node:http`; `fetch` aborted responses idle for 300 s ("terminated").
- `maxSummaryTokens` default 12000 → 16000 (thinking tokens share the budget; window room still caps it).
- Smoke test: `--thinking <level>`.

## 0.1.0 (unreleased)

- Compaction re-sends the captured provider request plus a summarize instruction, thinking disabled, so the server prefix cache covers the history.
- Post-compaction warm-up (1-token request with the new context).
- Fallback to Pi's default compaction on overflow, missing capture, low room, tool use, truncation, or errors.
- `/prefix-compaction` status command; global and project JSON config.
