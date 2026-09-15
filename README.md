# pi-prefix-cache-compaction

Faster [Pi](https://github.com/earendil-works/pi) compaction on self-hosted models (vLLM, SGLang, llama.cpp) by reusing the server's **prefix cache** instead of re-reading the whole conversation.

## The problem

Pi's built-in compaction serializes the history into one new user message under a different system prompt. To the server that is a brand-new prompt: the prefix cache misses completely and the whole history (typically 100k–190k tokens) is prefilled from scratch, then the summary is generated with your normal thinking level. On hosted APIs that costs seconds. On local GPUs it costs minutes, every time the context fills.

Measured on 2× RTX 3090 (vLLM, Qwen3.8-27B W4A16, TP=2), across 132 default compactions:

| | Pi default (median) | This extension |
|---|---|---|
| Tokens re-read cold per compaction | 142,106 | ~0 (prefix cache hit) |
| Compaction time | 229 s (p90 489 s) | **86 s** at 215k context |
| First turn after compaction | 69 s | **35 s** without warm-up; **1.8 s** with warm-up (47k-token smoke test, thinking xhigh) |

## How it works

1. **Capture.** Every real provider request is kept in memory (`before_provider_request`), per session.
2. **Compact from cache.** On `session_before_compact` the extension re-sends that exact request (same system prompt, tools, messages, thinking settings, auth/routing headers) with one summarize instruction appended. The prefix is byte-for-byte what the server just processed, so only the instruction is prefilled; the time left is generating the summary.
3. **Warm-up.** After compaction, it sends a 1-token request with the new context (built by Pi's own converter, with the captured system prompt, tools and thinking settings), so the summary and kept messages are already cached when you send the next turn.

**The prompt is never modified, only extended.** The summary request may set `max_tokens`, `stream` and `stream_options`; every field that can reach the rendered prompt is copied verbatim and checked before sending (`assertPrefixPreserved`), so a mistake fails fast into Pi's default instead of silently costing a cold re-prefill.

That matters most for thinking, because the toggle differs per model family and servers derive one field from another:

| Model family | Toggle | Default |
|---|---|---|
| Qwen3 / Qwen3.x | `chat_template_kwargs.enable_thinking` | on |
| DeepSeek-V3.1, IBM Granite 3.2 | `chat_template_kwargs.thinking` | off |
| Gemma 4 | `enable_thinking` or `reasoning_effort` | off |
| Holo2 | `thinking: false` disables | on |
| DeepSeek R1, GLM-4.5, MiniMax-M2, ERNIE, Hunyuan, Cohere Command A | parser-specific | varies |

vLLM also injects `enable_thinking` from `reasoning_effort` (`low`/`medium`/`high` → true, `none` → false). And templates like Qwen3.x render the effort text at the *start* of the system prompt, so flipping any of these re-prefills the entire conversation: an earlier build that forced thinking off turned a 245k-token compaction into a 5-minute cold read that then timed out. The summary therefore runs at the session's own thinking level, and the prompt simply asks for brief reasoning.

The summary request uses `node:http` rather than `fetch`, because `fetch` gives up on a response that sends no bytes for 300 s, which a long prefill or thinking phase on a local GPU can do.

Anything unexpected makes it step aside and Pi's default compaction runs: overflow recovery, no captured request yet (e.g. right after `/reload`), too little room left in the window, the model trying to call a tool, a truncated summary, or any HTTP/stream error.

Technique credit: [pisceslailai/deepseek-kvcache](https://github.com/pisceslailai/deepseek-kvcache) (DeepSeek, OpenAI wire format). This package applies it to the Anthropic Messages API used by Pi custom providers.

## Requirements

- Pi coding-agent ≥ 0.85, Node ≥ 22.19.
- A custom provider with `"api": "anthropic-messages"` pointing at a server with automatic prefix caching (vLLM `--enable-prefix-caching`, SGLang RadixAttention, llama.cpp `--cache-reuse`).
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

- `providers` / `baseUrlIncludes`: restrict to specific providers or endpoints (empty = every `anthropic-messages` model not excluded).
- `baseUrlExcludes`: hosted Anthropic is excluded by default; it has its own caching rules.
- There is **no option to change thinking for the summary**, by design — see below.
- The summary budget is `min(maxSummaryTokens, contextWindow − tokensBefore − promptOverheadTokens)`; below `minSummaryTokens` Pi's default runs instead.

## What you see

```
Compaction: reusing cached prefix (215,275 tokens)
Compaction done in 86s (4461 tokens out)
Context re-warmed in Ns; next turn starts from cache
```

## Limits

- Anthropic Messages wire format only (not `openai-completions` yet).
- The capture lives in memory: the first compaction after starting or `/reload`, before any turn is sent, uses Pi's default.
- The summary covers the whole captured request, including the recent messages Pi keeps verbatim, so it can repeat a little of that tail.
- The warm-up uses Pi's `convertToLlm`, not other extensions' `context` transforms; if you use such extensions the warm-up may only partly hit.

## Development

```bash
npm test                     # unit tests (node --test, no Pi needed)
node scripts/rpc-smoke.mjs --provider <id> --model <id> [--thinking xhigh] [--no-warmup]
```

The smoke test runs Pi in RPC mode in an isolated agent directory against your real server, compacts a ~60k-token session and times the first turn after.

## License

MIT
