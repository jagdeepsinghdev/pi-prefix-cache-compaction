# Changelog

## Unreleased

- **Feature: `models` config** restricts the extension to specific models (`id` or `provider/id`; empty = any), for gateways that serve several models from one base URL, e.g. local Qwen through the same proxy as hosted Claude/GPT models that should keep Pi's default compaction.

## 0.5.0 — 2026-10-15

0.4.0 was not published to npm; upgrading from 0.3.x gets both. See README, "Upgrading from 0.3.x".


- **Background compaction** (`background`, experimental, off by default; `backgroundAt`, default 0.7). After the first turn past `backgroundAt` × context window, the captured request is summarized in the background while the session keeps working. When Pi auto-compacts at its threshold, that summary is returned immediately and every entry after the summarized request is kept verbatim. Not used when summary + kept tail would exceed 60 % of the window; `/compact <instructions>` always takes the normal path.
  - **Pre-warm.** The post-compaction context (system, summary, kept entries) is warmed as soon as the summary is ready and extended after every turn, so the first request after the compaction starts from cache. If a pre-warm is in flight when Pi compacts, the compaction waits for it; there is no warm-up after the compaction in this mode, unless nothing was pre-warmed and Pi is idle. Found live: the first pre-release version warmed *after* the compaction, in parallel with the next request, and that request then prefilled ~92k tokens cold (51 s in a real session).
  - The summary is stored without the file-list suffix in this mode (it would change the pre-warmed prefix).
  - `scripts/background-smoke.mjs` on vLLM Qwen3.8-27B, Pi's threshold auto-compaction at ~49k tokens, compaction start to first reply: 86.2 s with `background: false`, 24.4 s without pre-warm, **7.2 s** with it (3.0 s compaction + 4.2 s reply, 88 % from cache). DeepSeek, both wire formats: compaction 3–4 ms. Planted facts survived in every run.
  - **Fix (found live): Pi crashed after `/reload`.** Background work that finished after a reload or session switch updated the pill through the stale ctx; Pi throws on that, and the exception escaped a fire-and-forget promise and exited the process. All UI and notification calls are now guarded, and no background promise can reject unhandled. Regression test: `scripts/stale-ctx-smoke.mjs` (compact, replace the session while the background work runs, check Pi still answers): exit code 1 before, alive after.
  - `PI_PREFIX_CACHE_DUMP=<dir>` writes every real request and warm-up body to `<dir>` for diffing.
  - Live session (Qwen3.8-27B, xhigh, 262k window): background summary of 184,358 tokens (182,400 from prefix cache, 5,180 out) in 120 s; live turns during it kept a 98–99 % cache hit. Pi's auto-compaction at 231,076 tokens took 0.1 s and kept 63 entries verbatim. With pre-warm: compaction at 234,417 tokens, first request after it 95,394 tokens with 97 % from cache (the size the last pre-warm covered).
- **Status pill** in Pi's footer (`statusBar`, on by default): spinner with elapsed time and an approximate output count during the summary, background summary and warm-up; green or amber result afterwards. Colors from the active theme.
- **Notifications** for each background step: start, ready (seconds, tokens in, from cache, out), still running when Pi compacts, out of date, failed.
- `/prefix-compaction` shows background summaries started / used and the last error.
- Tests: 39 unit tests (new: `backgroundAt` clamping, summary + kept-tail fit).

## 0.4.0 — 2026-10-03

- **Fix: compaction right after resume fell back to Pi's default** ("no captured request yet in this process"). New `persist` option, on by default: the last captured request per session is also written to `~/.pi/agent/cache/pi-prefix-cache-compaction/` (debounced, flushed on shutdown, credential headers stripped, 14-day pruning). A compaction before the first turn of a resumed process replays that request, which is the prefix the server still has cached. The file is dropped after every compaction. Verified live with the new `scripts/resume-smoke.mjs` against vLLM Qwen3.8-27B: resumed process compacted 103,023 tokens through the extension in 16 s with 97,600 prompt tokens from the prefix cache; the same run with `persist: false` produced the old fallback.
- **Fix: a stalled summary stream hung "Auto-compacting" forever.** The summary request had no timeout at all (`setTimeout(0)`). Seen behind LiteLLM: the upstream stream ended at 109 s but the proxy kept Pi's socket open for 25+ minutes. It now gives up after 600 s with no bytes (long enough for a cold 230k-token prefill) and Pi's default runs.
- **Shorter summaries.** Both summary prompts now ask for about 1,000–1,500 words and to drop detail no longer needed. On a local GPU the summary is generated at ~20–50 tok/s, and real sessions were producing 15–27k-character summaries (3–9k tokens incl. reasoning), i.e. 1–7 minutes per compaction. The prompt is appended after the cached prefix, so the cache hit is unaffected.
- **Experimental: incremental compaction** (`incremental`, off by default, Pi ≥ 0.99). From the second compaction on, the previous summary plus an update instruction is appended so the model merges the new delta instead of re-summarizing everything; the merged summary is carried forward in the compaction entry's `details`. Mechanism proven by unit tests (fake-server two-compaction end-to-end, byte-identical prefix) and against Pi 0.99.1's real `prepareCompaction()`, and now live with the new config-driven runner (`scripts/experiment.mjs experiments/incremental.json`) (Pi 0.99.2, vLLM Qwen3.8-27B on 2× RTX 3090, 3 cycles, ~52–59k-token contexts, thinking off). All four checks passed: every compaction went through the extension; compactions 2–3 were really incremental (notice + `details.previousSummary`); all three planted facts survived (fact 1 through three summaries) in both modes; and incremental was faster over compactions 2–3:

  | compaction | baseline | incremental |
  |---|---|---|
  | 1 (identical in both modes) | 33 s, 2,523 tok | 33 s, 2,417 tok |
  | 2 | 52 s, 4,058 tok | **23 s, 1,501 tok** |
  | 3 | 33 s, 2,595 tok | 50 s, 3,912 tok |
  | **2–3 total** | **85 s, 6,653 tok** | **73 s, 5,413 tok (−14 % time, −19 % tokens)** |

  A second run with the config-driven runner went the other way (incremental 56 s / 3,949 tok vs baseline 39 s / 2,802 tok over compactions 2–3), and even the identical compaction 1 varied 34 s → 70 s between runs. Run-to-run variance in summary length is larger than the effect, so **the speed benefit is unproven**; correctness (no lost facts in its runs, real incremental wiring) is shown. It stays opt-in until `experiments/incremental.json` passes with repeats.
- Tests: 37 unit tests (persist round-trip and credential stripping, persist default, idle-timeout abort); new live scripts `resume-smoke.mjs` and the config-driven experiment runner `experiment.mjs` with `experiments/*.json`.

## 0.3.0 — 2026-09-20

- **Fix: compaction 401 on authenticated endpoints** ([#2](https://github.com/jagdeepsinghdev/pi-prefix-cache-compaction/issues/2)). `before_provider_headers` runs before Pi hands the API key to the SDK, so the captured header map never contained a credential unless `authHeader: true` was set. The summary request now resolves auth at request time via `ctx.modelRegistry.getApiKeyAndHeaders` and attaches the key the way the SDK does (`x-api-key` for Anthropic Messages, `Authorization: Bearer` for OpenAI and OAuth-style tokens), layered over the captured routing headers; a resolved `baseUrl` is preferred over `model.baseUrl`. A missing credential is reported as `auth: …` instead of an HTTP 401.
- **Fix: warm-up never worked against authenticated endpoints.** It guessed a Bearer token from the captured headers and fell back to the literal `"local"`. It now runs through `ctx.modelRegistry.streamSimple` on Pi >= 0.86, so Pi supplies the credential; on Pi 0.85, whose registry facade cannot stream, it resolves auth via `getApiKeyAndHeaders` and calls pi-ai's `completeSimple` with the real key. The `"local"` default is gone, the last warm-up error is shown by `/prefix-compaction`, and `PI_PREFIX_CACHE_DEBUG=1` prints it to stderr.
- **Feature: OpenAI Chat Completions support** ([#1](https://github.com/jagdeepsinghdev/pi-prefix-cache-compaction/issues/1)). Models with `"api": "openai-completions"` are now handled: the summarize instruction is appended as a string-content user message, the output cap is set in whichever field the captured request used (`max_tokens` or `max_completion_tokens`), the request goes to `<baseUrl>/chat/completions`, and a `chat.completion.chunk` stream parser handles text, `reasoning_content`/`reasoning`, `tool_calls`, `finish_reason` (`length` and `tool_calls` fall back) and usage (`prompt_tokens_details.cached_tokens`, `prompt_cache_hit_tokens`, `cached_tokens`). `enable_thinking`, `chat_template_args` and `response_format` joined the prompt-affecting fields. Pi's own summarizer is still filtered out when its system prompt arrives as a `system`/`developer` message.
- The "Compaction done" notice now reports how many prompt tokens the server served from its prefix cache, so a cache miss is visible.
- Tests: fake-server end-to-end tests reproduce the 401 with a keyless capture and verify both wire formats; 28 tests total. Verified live on Pi 0.85.1 and 0.86.0 against: a llama.cpp server with `--api-key` (401 without the key), both `/v1/messages` and `/v1/chat/completions`, ~26,220 of ~26,225 prompt tokens from cache; a self-hosted vLLM Qwen3.8 endpoint over Anthropic Messages (Bearer-only auth) and OpenAI Chat Completions with thinking on; and DeepSeek's official API on both wire formats with thinking on and off, ~56,190 of ~56,350 prompt tokens from cache.
- Smoke script: new `--cycles N` runs N compact-then-continue rounds in one session and verifies two planted facts survive after each. Four cycles on DeepSeek, both wire formats, thinking high: every compaction went through the extension with a full prefix-cache hit and the facts stayed intact.

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
