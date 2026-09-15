# Changelog

## 0.2.0 (unreleased)

- The summary request may now only override `max_tokens`, `stream` and `stream_options`; every field that can reach the rendered prompt (`system`, `tools`, `tool_choice`, `messages`, `thinking`, `reasoning`, `reasoning_effort`, `output_config`, `chat_template_kwargs`, `model`, `mm_processor_kwargs`, `documents`) is copied verbatim and verified by `assertPrefixPreserved` before sending. A mismatch falls back to Pi's default instead of silently paying a cold re-prefill.
- Removed the `disableThinking` option: the thinking toggle differs per model family (Qwen3 `enable_thinking`, DeepSeek-V3.1/Granite `thinking`, Gemma 4 `reasoning_effort`, Holo2 `thinking:false`) and vLLM derives `enable_thinking` from `reasoning_effort`, so changing it is never cache-safe in general.
- Warm-up copies the same fields from the captured request, so it warms the prefix the next real turn will send.

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
