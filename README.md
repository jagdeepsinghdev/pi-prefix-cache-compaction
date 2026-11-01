# pi-prefix-cache-compaction

Faster [Pi](https://github.com/earendil-works/pi) compaction on any model server with automatic prefix caching, self-hosted (vLLM, SGLang, llama.cpp, oMLX) or hosted (e.g. DeepSeek), by reusing the server's **prefix cache** instead of re-reading the whole conversation. Works with the two wire formats Pi custom providers use: `anthropic-messages` and `openai-completions`.

Optional background mode (0.5.0) prepares the summary and the post-compaction cache before Pi compacts. Measured on a local 27B model, Pi's auto-compaction then took 0.1–3 s instead of minutes; when the background work is not finished yet, Pi waits for the remainder (see [Limits](#limits)).

## The problem

Pi's built-in compaction serializes the history into one new user message under a different system prompt. To the server that is a brand-new prompt: the prefix cache misses completely and the whole history (typically 100k–190k tokens) is prefilled from scratch, then the summary is generated with your normal thinking level. On hosted APIs that costs seconds. On local GPUs it costs minutes, every time the context fills.

Measured on 2× RTX 3090 (vLLM, Qwen3.8-27B W4A16, TP=2), across 132 default compactions:

| | Pi default (median) | This extension |
|---|---|---|
| Tokens re-read cold per compaction | 142,106 | ~0 (prefix cache hit) |
| Compaction time | 229 s (p90 489 s) | **86 s** at 215k context |
| First turn after compaction | 69 s | **35 s** without warm-up; **1.8 s** with warm-up (47k-token smoke test, thinking xhigh) |

On a hosted API the win is smaller in seconds but the cache hit is the same: against DeepSeek's API a 56k-token compaction took 3 s with 56,192 of 56,216 prompt tokens served from cache (see [Verified against](#verified-against)).

## How it works

1. **Capture.** Every real provider request is kept in memory (`before_provider_request`), per session.
2. **Compact from cache.** On `session_before_compact` the extension re-sends that exact request (same system prompt, tools, messages, thinking settings, routing headers) with one summarize instruction appended. Credentials are resolved at request time through Pi's model registry, exactly as for a real turn, and attached the way the SDK would (`x-api-key` for Anthropic Messages, `Authorization: Bearer` for OpenAI). The prefix is byte-for-byte what the server just processed, so only the instruction is prefilled; the time left is generating the summary.
3. **Warm-up.** After compaction, it sends a 1-token request through Pi's model registry with the new context (built by Pi's own converter, with the captured system prompt, tools and thinking settings), so the summary and kept messages are already cached when you send the next turn.
4. **Background mode (optional, `background: true`).** Steps 2 and 3 still cost time while Pi waits. In background mode the summary is made earlier, after the first turn past `backgroundAt` (70 % of the window), while the session keeps running. When Pi auto-compacts at its threshold, the extension returns that summary at once and keeps every entry after the summarized request verbatim.
   The context Pi will have after that compaction (system, summary, kept entries) is known as soon as the summary exists, and only grows at the end. So it is pre-warmed then, and extended after every turn; each extension prefills only the new turns. When Pi compacts, the first request finds it cached. If a pre-warm is still running at that moment, the compaction waits for it: a request that races an in-flight pre-warm of the same prefix prefills everything itself (measured on vLLM: 32.6 s racing vs 3.3 s after the pre-warm finished, 30k-token context).

```
at 70 %:  [ system | history up to the summary point | ... ]   summary made in the background
at Pi's threshold:
before:   [ system | history up to the summary point | everything after it ]
after:    [ system | summary                         | everything after it ]   (summary already made, context already cached)
```

**The prompt is never modified, only extended.** The summary request may set the output cap (`max_tokens` or `max_completion_tokens`, whichever the captured request used) and the streaming flags; every field that can reach the rendered prompt is copied verbatim and checked before sending (`assertPrefixPreserved`), so a mistake fails fast into Pi's default instead of silently costing a cold re-prefill.

That matters most for thinking, because the toggle differs per model family and servers derive one field from another:

| Model family | Toggle | Default |
|---|---|---|
| Qwen3 / Qwen3.x | `chat_template_kwargs.enable_thinking` (or top-level `enable_thinking` on OpenAI endpoints) | on |
| DeepSeek-V3.1, IBM Granite 3.2 | `chat_template_kwargs.thinking` | off |
| Gemma 4 | `enable_thinking` or `reasoning_effort` | off |
| Holo2 | `thinking: false` disables | on |
| DeepSeek R1, GLM-4.5, MiniMax-M2, ERNIE, Hunyuan, Cohere Command A | parser-specific | varies |

vLLM also injects `enable_thinking` from `reasoning_effort` (`low`/`medium`/`high` → true, `none` → false). And templates like Qwen3.x render the effort text at the *start* of the system prompt, so flipping any of these re-prefills the entire conversation: an earlier build that forced thinking off turned a 245k-token compaction into a 5-minute cold read that then timed out. The summary therefore runs at the session's own thinking level, and the prompt simply asks for brief reasoning.

The summary request uses `node:http` rather than `fetch`, because `fetch` gives up on a response that sends no bytes for 300 s, which a long prefill or thinking phase on a local GPU can do. It is still bounded: after 600 s with no bytes at all (a proxy that stopped relaying without closing the connection) it gives up and Pi's default runs, instead of leaving "Auto-compacting" spinning forever.

Anything unexpected makes it step aside and Pi's default compaction runs: overflow recovery, no captured request yet (a fresh session, or `persist` turned off), a capture that belongs to a different model or to a pre-fallback-compaction history, too little room left in the window, the model trying to call a tool, a summary cut off mid-stream, or any HTTP/stream error.

## Verified against

Each row is a full run of `scripts/rpc-smoke.mjs` (real turns, compaction through this extension, warm-up, one turn after) on the 0.3.0 code. "From cache" is what the server's own usage report said about the summary request.

| Server | Wire format | Pi | Thinking | Prompt tokens from cache |
|---|---|---|---|---|
| llama.cpp `--api-key` (401 without key), Qwen3-4B | `anthropic-messages` | 0.86.0 | off | 26,222 / 26,226 |
| llama.cpp, same server | `openai-completions` | 0.86.0 | off | 26,323 / 26,327 |
| vLLM, Qwen3.8-27B, Bearer-only proxy | `anthropic-messages` | 0.86.0 | off | hit (vLLM omits the count by default) |
| vLLM, same | `openai-completions` | 0.86.0 | high | hit (60k tokens compacted in 19 s) |
| DeepSeek API (`/anthropic`) | `anthropic-messages` | 0.85.1, 0.86.0 | off, high | 56,192 / 56,216 |
| DeepSeek API (`/v1`) | `openai-completions` | 0.85.1, 0.86.0 | off, high | 56,192 / 56,313 |
| oMLX (Apple Silicon, MPS), Qwen3.8-27B AWQ | `openai-completions` | 0.87.0 | off | 18,432 / 18,981 |
| oMLX, same stack, live coding session | `openai-completions` | 0.87.0 | adaptive | 131,072 / 135,414 |
| LiteLLM proxy → local vLLM, Qwen3.8-27B (2× RTX 3090, TP=2) | `anthropic-messages` | 0.99.1 | off | 3 cycles, 51.6–56.9k prompt (proxy strips the cache count) |

The two oMLX rows were run with Pi 0.87.0 in RPC mode in an isolated `PI_CODING_AGENT_DIR` against oMLX 0.6.4 (engines mlx 0.32.0 locally and 0.31.3 on the second Apple Silicon host): the first is a scripted smoke (compaction at `reason: "threshold"` went through the extension, warm-up included), the second a real `/compact` in an everyday long-running session on `qwen3.8-flash` with thinking `adaptive` — 96.8% of the 135k-token summary request was served from cache. oMLX specifics worth knowing:

- Prefix caching is **on by default, no flag** (hot RAM + an SSD tier). Cache blocks are chain-hashed 2,048-token units, which is why the cache counts above are exact multiples of 2,048 with only the tail cold — and why an appended instruction keeps the hit.
- Cached blocks **survive model unload/reload** (and a server restart): the first compaction after a reload still hits the SSD tier.
- `usage.prompt_tokens_details.cached_tokens` is reported on the OpenAI wire, including over SSE, on both engine versions — no extra server flag is needed to see the cache count.
- For block-level verification oMLX exposes `/admin/api/cache/probe` (and `cache/stats` / `cache/efficiency`), which uses the scheduler's own block hashing.

Multi-compaction sessions: `--cycles 4` on DeepSeek, both wire formats, thinking high. All eight compactions went through the extension with a full cache hit (44,672 to 68,864 prompt tokens served from cache per request), and two planted facts survived every summary-of-a-summary.

Not verified: SGLang (needs an NVIDIA GPU; not run here). Not implemented: `openai-responses`, a different payload and stream format, which is what Pi's built-in hosted OpenAI provider uses.

Not verified on oMLX: multi-compaction `--cycles` runs (single compactions, incl. one live long session, are covered above).

## Related work

The same idea, appending the summarize instruction to the last real request so the prefix stays cached, exists for other wire formats:

- [pisceslailai/deepseek-kvcache](https://github.com/pisceslailai/deepseek-kvcache): DeepSeek's hosted API (OpenAI wire format). Technique credit for this package.
- [yuan-/pi-kvc](https://github.com/yuan-/pi-kvc): llama.cpp / LM Studio over `openai-completions`, with a manual `/kvc` command.

Neither is on npm. This package covers both `anthropic-messages` and `openai-completions`, and adds the model-bound and stale-capture checks, the prefix-preservation assertion, request-time credential resolution and the post-compaction warm-up.

## Requirements

- Pi coding-agent ≥ 0.85 (tested on 0.85.1, 0.86.0, 0.87.0, 0.99.1 and 1.1.0), Node ≥ 22.19. Background mode and the status pill were tested on 1.1.0.
- Caveat: a proxy in between (e.g. LiteLLM) may strip `usage.prompt_tokens_details`/`cache_creation` from responses — the extension still works (the server-side prefix cache is unaffected), but the "served from prefix cache" count in the notices will be absent.
- A custom provider with `"api": "anthropic-messages"` or `"api": "openai-completions"` pointing at a server with automatic prefix caching (vLLM `--enable-prefix-caching`, SGLang RadixAttention, llama.cpp `--cache-reuse`, oMLX by default). Authenticated endpoints work with the key configured as `apiKey` in `models.json`; `authHeader: true` is not required.
- A chat template that renders earlier turns the same whether or not a new user message follows. Templates that strip earlier reasoning after a new user message (for example Qwen3 with `preserve_thinking` off) still work, with a smaller cache hit.

## Install

```bash
pi install npm:pi-prefix-cache-compaction
# or try it for one run
pi -e npm:pi-prefix-cache-compaction
```

Then `/reload` open sessions. Check with `/prefix-compaction`.

### Upgrading from 0.3.x

```bash
pi update --extension npm:pi-prefix-cache-compaction
```

Then `/reload` open sessions (or restart them). Existing config files keep working. What changes without any config edit:

| Since | Change | To opt out |
|---|---|---|
| 0.4.0 | The last request per session is kept on disk in `~/.pi/agent/cache/pi-prefix-cache-compaction/` (no credentials, pruned after 14 days), so a compaction right after resume still hits the cache. | `"persist": false` |
| 0.4.0 | Summaries are shorter (prompt asks for ~1,000–1,500 words). | — |
| 0.4.0 | A summary stream that sends nothing for 600 s is abandoned and Pi's default runs (was: hung forever). | — |
| 0.5.0 | A status pill in Pi's footer while a compaction, warm-up or background summary runs. | `"statusBar": false` |

Background mode is off by default. To turn it on, add it to `~/.pi/agent/pi-prefix-cache-compaction.json` (all projects) or `<project>/.pi/pi-prefix-cache-compaction.json` (one project), then `/reload`:

```json
{ "background": true }
```

It starts working at the first turn past 70 % of the context window (`backgroundAt`); see [How it works](#how-it-works), step 4. `"incremental": true` is the other opt-in (experimental, Pi ≥ 0.99).

## Configuration

Optional JSON, project overrides global:

- `~/.pi/agent/pi-prefix-cache-compaction.json`
- `<project>/.pi/pi-prefix-cache-compaction.json`

```json
{
  "enabled": true,
  "providers": [],
  "baseUrlIncludes": [],
  "baseUrlExcludes": ["api.anthropic.com"],
  "models": [],
  "maxSummaryTokens": 16000,
  "minSummaryTokens": 4000,
  "promptOverheadTokens": 3000,
  "warmup": true,
  "notify": true,
  "statusBar": true,
  "incremental": false,
  "persist": true,
  "background": false,
  "backgroundAt": 0.7
}
```

- `providers` / `baseUrlIncludes`: restrict to specific providers or endpoints (empty = every `anthropic-messages` or `openai-completions` model not excluded).
- `models`: restrict to specific models, as `id` or `provider/id` (empty = any), for providers that serve several models from one endpoint.
- `baseUrlExcludes`: hosted Anthropic is excluded by default because its prompt caching needs explicit `cache_control` breakpoints, which this extension does not manage. Hosted APIs with automatic prefix caching work as-is (DeepSeek verified on both wire formats).
- There is **no option to change thinking for the summary**, by design — see below.
- The summary budget is `min(maxSummaryTokens, contextWindow − tokensBefore − promptOverheadTokens)`; below `minSummaryTokens` Pi's default runs instead.
- `persist` (0.4.0, on by default): the last captured request per session is also kept on disk in `~/.pi/agent/cache/pi-prefix-cache-compaction/` (credentials are never written; files older than 14 days are pruned). A compaction that fires right after `pi -c` / resume, before any new turn, then replays the request the previous process sent — the prefix the server still has cached — instead of falling back to Pi's default. The file is dropped after every compaction, since it then describes pre-compaction history.
- `incremental` (0.4.0, **experimental**, off by default): on Pi ≥ 0.99, from the second compaction on Pi hands the extension a `previousSummary` plus a message window that excludes the pre-compaction history. When enabled, the summary request appends that previous summary and an update instruction so the model merges the new delta into the existing summary instead of re-summarizing everything, and the merged result is carried into the next compaction. Verified live with `scripts/experiment.mjs experiments/incremental.json`: it merges correctly and kept every planted fact, but two single runs disagreed on speed (−14 % and +44 % summary time), so a speed benefit is not shown and it stays opt-in.

- `statusBar` (on by default): live pill in Pi's footer, see [What you see](#what-you-see).
- `background` (0.5.0, **experimental**, off by default) and `backgroundAt` (default `0.7`): see [How it works](#how-it-works), step 4.
  - The summary starts after the first turn whose request is at or above `backgroundAt` × context window (clamped to 0.3–0.95). One job per session.
  - If Pi compacts while the summary is still streaming, it waits only for the rest.
  - The summary is not used if summary + kept tail would exceed 60 % of the window (Pi would compact again right away); a fresh one is started after the next turn, or the normal path runs.
  - Plain `/compact` uses a ready summary too; `/compact <instructions>` always takes the normal path.
  - In this mode the summary is stored without the file-list suffix the normal path appends: the file list is only known at compaction time, and appending it would change the pre-warmed summary message.
  - Measured with `scripts/background-smoke.mjs` on vLLM Qwen3.8-27B (2× RTX 3090, TP=2, thinking off, 64k window, Pi's threshold auto-compaction at ~49k tokens), time from compaction start to the first reply after it:

    | | compaction | first reply after | total |
    |---|---|---|---|
    | `background: false` | 83.5 s | 2.7 s | 86.2 s |
    | `background: true`, no pre-warm (0.5.0 pre-release) | 0.002 s | 24.4 s | 24.4 s |
    | `background: true` | 3.0 s (finishing the last pre-warm) | 4.2 s, 88 % of the prompt from cache | **7.2 s** |

    Planted facts survived in every run.

## What you see

```
Compaction: reusing cached prefix (215,275 tokens)
Compaction done in 86s (4461 tokens out, 214,900 prompt tokens served from prefix cache)
Context re-warmed in Ns; next turn starts from cache
```

With `background` on, in a real session (Qwen3.8-27B, thinking xhigh, 262k window):

![Background summary ready at 70 % of the window; the footer pill shows compaction ready](https://raw.githubusercontent.com/jagdeepsinghdev/pi-prefix-cache-compaction/master/docs/background-compaction.png)

The full sequence (from a `background-smoke.mjs` run, 64k window):

```
Background compaction: started at 25,238 / 64,000 tokens (39%); you can keep working
Background compaction: summary ready in 14s (29,614 tokens in, 27,200 from prefix cache, 870 out); ready for the next compaction
Compaction: used background summary (53,156 tokens; kept everything after the summarized request)
Context re-warmed in 27s; next turn starts from cache
```

Footer pill (`statusBar`), colors from the active Pi theme:

| State | Pill |
|---|---|
| summary request waiting on prefill | `⠹ prefilling 215k from cache · 12s` |
| summary streaming (output count is chars / 4) | `⠼ summarizing 215k · ~1.2k out · 41s` |
| background summary running (grey) | `⠏ pre-summarizing in background · ~3.6k out · 90s` |
| background summary ready (stays until used) | `● compaction ready · summary made in 120s` |
| warm-up | `⠋ warming next turn · 3s` |
| done (green, clears after 8 s) | `✓ compacted in 0.1s · context pre-warmed` / `✓ compacted · warm in 3s` |
| fallback (amber) | `⚠ prefix-cache skipped: <reason>` |

`"statusBar": false` turns it off. Print mode never shows it; RPC clients receive `setStatus` UI requests.

The cache count comes from the server's usage report: DeepSeek, llama.cpp and oMLX include it; vLLM only when started with `--enable-prompt-tokens-details` (the compaction still hits the cache without it, the number is just omitted). On vLLM's **Anthropic** endpoint (`/v1/messages`) the count is gated on the same flag too: vLLM's adapter only emits `cache_read_input_tokens`/`cache_creation_input_tokens` when `usage.prompt_tokens_details` is present (vllm/entrypoints/anthropic/serving.py), and that object is built only with the flag — so behind a LiteLLM-style proxy on the anthropic wire the count is invisible unless the flag is set. If the endpoint rejects the request you get one line and Pi's default runs, e.g. `Prefix-cache compaction skipped (HTTP 401 ...); using Pi default`. A missing credential shows up as `auth: No API key found for "<provider>"` instead of an HTTP error.

## Limits

- Anthropic Messages and OpenAI Chat Completions only; `openai-responses` and other wire formats fall through to Pi's default. Pi's built-in hosted OpenAI provider uses `openai-responses`, so hosted GPT models are not covered. The OpenAI Chat Completions path is new in 0.3.0 (see [Verified against](#verified-against)); any malformed request falls back to Pi's default rather than failing the compaction.
- With `persist` off the capture lives in memory only: the first compaction after starting, resuming or `/reload`, before any turn is sent, uses Pi's default. With `persist` on (default) this applies only to a brand-new session.
- Switching the model mid-session invalidates the capture (different model means a different cache): the next compaction uses Pi's default until the new model sends a real turn.
- After falling back to Pi's default compaction, the capture is marked stale until a real turn re-anchors it, so two consecutive compactions without a turn in between use the default both times. The warm-up still runs after such a default compaction.
- The summary covers the whole captured request, including the recent messages Pi keeps verbatim, so it can repeat a little of that tail.
- The warm-up uses Pi's `convertToLlm`, not other extensions' `context` transforms; if you use such extensions the warm-up may only partly hit.
- A user message containing the literal `<conversation>` tag is not captured (same text as Pi's own summarizer requests); the previous turn stays the anchor instead.
- Background mode does not always remove the wait. Pi still waits when it compacts before the background summary is finished (it waits for the rest; at high thinking levels a summary takes minutes) or while the last pre-warm is running (seconds; capped at 90 s). It falls back to the normal path when the summary failed, when summary + kept tail would exceed 60 % of the window, and for `/compact <instructions>`. If the context grows from 70 % to Pi's threshold faster than the summary is written (e.g. several large file reads in a row), lower `backgroundAt`.
- Background mode replays the session's own thinking level (changing it would change the prefix). At high thinking levels the background summary takes minutes, not seconds (120 s at 184k tokens, xhigh, Qwen3.8-27B); if the context reaches Pi's threshold first, Pi waits for the remainder.
- Background mode runs the summary alongside your live turns on the same server. On a single local GPU pair those turns decode somewhat slower until it finishes; their prefix-cache hit is unaffected (98–99 % measured during a background summary).
- The warm-up goes through `ctx.modelRegistry.streamSimple` (Pi ≥ 0.86) or pi-ai's `completeSimple` with auth from `getApiKeyAndHeaders` (Pi 0.85), so the credential is resolved at request time either way. A failed warm-up is silent in the UI (the next turn simply prefills normally); `/prefix-compaction` shows the last warm-up error, and `PI_PREFIX_CACHE_DEBUG=1` prints it to stderr.

## Development

```bash
npm test                     # unit tests (node --test, no Pi needed)
node scripts/rpc-smoke.mjs --provider <id> --model <id> [--thinking xhigh] [--no-warmup] [--rows 700] [--cycles 1]
node scripts/resume-smoke.mjs --provider <id> --model <id> [--persist false] [--base-url URL --api-key KEY --model-id NAME]
node scripts/experiment.mjs experiments/incremental.json [--repeats N] [--dry-run]
node scripts/background-smoke.mjs --provider <id> --model <id> [--window 64000] [--rows 500] [--no-background]
node scripts/stale-ctx-smoke.mjs --provider <id> --model <id>
```

The smoke test runs Pi in RPC mode in an isolated agent directory against your real server, compacts a ~45k-token session (`--rows` scales it; 300 is comfortable for a laptop llama.cpp) and times the first turn after; `--cycles N` repeats compact-then-continue N times in one session and checks two planted facts after each. `experiment.mjs` runs config-driven live A/B experiments (see [`experiments/README.md`](experiments/README.md)): each arm is any overlay of this extension's config, every arm runs the same scripted session (plant a fact per cycle, push it out of the verbatim tail, compact, finally recall every fact), arms interleave over `repeats`, and the exit code is gated on the checks you choose — every compaction used the extension, incremental wiring, no lost facts, and a minimum speed gain of candidate over baseline. `resume-smoke.mjs` checks `persist`: one process runs a short session and quits, a second resumes it with `--continue` and compacts before any turn; it passes only if that compaction went through the extension (or, with `--persist false`, only if it fell back). `--base-url` points it straight at one inference server so it does not take a slot on a shared proxy. `background-smoke.mjs` shrinks the model's context window in a temp `models.json` so Pi's own threshold auto-compaction fires after a few file reads, then reports how long Pi was blocked and whether planted facts survived; `--no-background` runs the same scenario as the baseline. `stale-ctx-smoke.mjs` replaces the session while background work is still running and checks Pi survives (regression test for a crash after `/reload`). `PI_BIN` selects another Pi binary, `PI_CODING_AGENT_DIR` another `models.json`.

## License

MIT
