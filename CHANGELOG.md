# Changelog

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
