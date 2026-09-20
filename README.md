# pi-prefix-cache-compaction

Faster [Pi](https://github.com/earendil-works/pi) compaction on any model server with automatic prefix caching, self-hosted (vLLM, SGLang, llama.cpp) or hosted (e.g. DeepSeek), by reusing the server's **prefix cache** instead of re-reading the whole conversation. Works with the two wire formats Pi custom providers use: `anthropic-messages` and `openai-completions`.

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

The summary request uses `node:http` rather than `fetch`, because `fetch` gives up on a response that sends no bytes for 300 s, which a long prefill or thinking phase on a local GPU can do.

Anything unexpected makes it step aside and Pi's default compaction runs: overflow recovery, no captured request yet (e.g. right after `/reload`), a capture that belongs to a different model or to a pre-fallback-compaction history, too little room left in the window, the model trying to call a tool, a summary cut off mid-stream, or any HTTP/stream error.

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

Multi-compaction sessions: `--cycles 4` on DeepSeek, both wire formats, thinking high. All eight compactions went through the extension with a full cache hit (44,672 to 68,864 prompt tokens served from cache per request), and two planted facts survived every summary-of-a-summary.

Not verified: SGLang (needs an NVIDIA GPU; not run here). Not implemented: `openai-responses`, a different payload and stream format, which is what Pi's built-in hosted OpenAI provider uses.

## Related work

The same idea, appending the summarize instruction to the last real request so the prefix stays cached, exists for other wire formats:

- [pisceslailai/deepseek-kvcache](https://github.com/pisceslailai/deepseek-kvcache): DeepSeek's hosted API (OpenAI wire format). Technique credit for this package.
- [yuan-/pi-kvc](https://github.com/yuan-/pi-kvc): llama.cpp / LM Studio over `openai-completions`, with a manual `/kvc` command.

Neither is on npm. This package covers both `anthropic-messages` and `openai-completions`, and adds the model-bound and stale-capture checks, the prefix-preservation assertion, request-time credential resolution and the post-compaction warm-up.

## Requirements

- Pi coding-agent ≥ 0.85 (tested on 0.85.1 and 0.86.0), Node ≥ 22.19.
- A custom provider with `"api": "anthropic-messages"` or `"api": "openai-completions"` pointing at a server with automatic prefix caching (vLLM `--enable-prefix-caching`, SGLang RadixAttention, llama.cpp `--cache-reuse`). Authenticated endpoints work with the key configured as `apiKey` in `models.json`; `authHeader: true` is not required.
- A chat template that renders earlier turns the same whether or not a new user message follows. Templates that strip earlier reasoning after a new user message (for example Qwen3 with `preserve_thinking` off) still work, with a smaller cache hit.

## Install

```bash
pi install npm:pi-prefix-cache-compaction
# or try it for one run
pi -e npm:pi-prefix-cache-compaction
```

Then `/reload` open sessions. Check with `/prefix-compaction`.

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
  "maxSummaryTokens": 16000,
  "minSummaryTokens": 4000,
  "promptOverheadTokens": 3000,
  "warmup": true,
  "notify": true
}
```

- `providers` / `baseUrlIncludes`: restrict to specific providers or endpoints (empty = every `anthropic-messages` or `openai-completions` model not excluded).
- `baseUrlExcludes`: hosted Anthropic is excluded by default because its prompt caching needs explicit `cache_control` breakpoints, which this extension does not manage. Hosted APIs with automatic prefix caching work as-is (DeepSeek verified on both wire formats).
- There is **no option to change thinking for the summary**, by design — see below.
- The summary budget is `min(maxSummaryTokens, contextWindow − tokensBefore − promptOverheadTokens)`; below `minSummaryTokens` Pi's default runs instead.

## What you see

```
Compaction: reusing cached prefix (215,275 tokens)
Compaction done in 86s (4461 tokens out, 214,900 prompt tokens served from prefix cache)
Context re-warmed in Ns; next turn starts from cache
```

The cache count comes from the server's usage report: DeepSeek and llama.cpp include it; vLLM only when started with `--enable-prompt-tokens-details` (the compaction still hits the cache without it, the number is just omitted). If the endpoint rejects the request you get one line and Pi's default runs, e.g. `Prefix-cache compaction skipped (HTTP 401 ...); using Pi default`. A missing credential shows up as `auth: No API key found for "<provider>"` instead of an HTTP error.

## Limits

- Anthropic Messages and OpenAI Chat Completions only; `openai-responses` and other wire formats fall through to Pi's default. Pi's built-in hosted OpenAI provider uses `openai-responses`, so hosted GPT models are not covered. The OpenAI Chat Completions path is new in 0.3.0 (see [Verified against](#verified-against)); any malformed request falls back to Pi's default rather than failing the compaction.
- The capture lives in memory: the first compaction after starting or `/reload`, before any turn is sent, uses Pi's default.
- Switching the model mid-session invalidates the capture (different model means a different cache): the next compaction uses Pi's default until the new model sends a real turn.
- After falling back to Pi's default compaction, the capture is marked stale until a real turn re-anchors it, so two consecutive compactions without a turn in between use the default both times. The warm-up still runs after such a default compaction.
- The summary covers the whole captured request, including the recent messages Pi keeps verbatim, so it can repeat a little of that tail.
- The warm-up uses Pi's `convertToLlm`, not other extensions' `context` transforms; if you use such extensions the warm-up may only partly hit.
- A user message containing the literal `<conversation>` tag is not captured (same text as Pi's own summarizer requests); the previous turn stays the anchor instead.
- The warm-up goes through `ctx.modelRegistry.streamSimple` (Pi ≥ 0.86) or pi-ai's `completeSimple` with auth from `getApiKeyAndHeaders` (Pi 0.85), so the credential is resolved at request time either way. A failed warm-up is silent in the UI (the next turn simply prefills normally); `/prefix-compaction` shows the last warm-up error, and `PI_PREFIX_CACHE_DEBUG=1` prints it to stderr.

## Development

```bash
npm test                     # unit tests (node --test, no Pi needed)
node scripts/rpc-smoke.mjs --provider <id> --model <id> [--thinking xhigh] [--no-warmup] [--rows 700] [--cycles 1]
```

The smoke test runs Pi in RPC mode in an isolated agent directory against your real server, compacts a ~45k-token session (`--rows` scales it; 300 is comfortable for a laptop llama.cpp) and times the first turn after; `--cycles N` repeats compact-then-continue N times in one session and checks two planted facts after each. `PI_BIN` selects another Pi binary, `PI_CODING_AGENT_DIR` another `models.json`.

## License

MIT
