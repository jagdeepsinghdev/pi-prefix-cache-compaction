/**
 * Pure logic for pi-prefix-cache-compaction. No Pi imports, so it is unit-testable
 * with plain `node --test`.
 */
import { request as httpRequest } from "node:http";
import { request as httpsRequest } from "node:https";

export interface Config {
	enabled: boolean;
	/** Apply only to these provider ids (empty = any provider that passes the other checks). */
	providers: string[];
	/** Apply only when the model baseUrl contains one of these (empty = any). */
	baseUrlIncludes: string[];
	/** Never apply when the baseUrl contains one of these (hosted APIs have their own caching). */
	baseUrlExcludes: string[];
	maxSummaryTokens: number;
	minSummaryTokens: number;
	/** Tokens reserved for the appended instruction and chat-template overhead. */
	promptOverheadTokens: number;
	/** After compaction, send a 1-token request with the new context so the next turn starts warm. */
	warmup: boolean;
	notify: boolean;
}

export const DEFAULT_CONFIG: Config = {
	enabled: true,
	providers: [],
	baseUrlIncludes: [],
	baseUrlExcludes: ["api.anthropic.com"],
	maxSummaryTokens: 16_000,
	minSummaryTokens: 4_000,
	promptOverheadTokens: 3_000,
	warmup: true,
	notify: true,
};

export function mergeConfig(...layers: Array<Partial<Config> | undefined>): Config {
	const out: Config = { ...DEFAULT_CONFIG };
	for (const layer of layers) {
		if (!layer || typeof layer !== "object") continue;
		for (const [k, v] of Object.entries(layer)) {
			if (!(k in DEFAULT_CONFIG) || v === undefined) continue;
			const def = (DEFAULT_CONFIG as any)[k];
			if (Array.isArray(def) ? Array.isArray(v) : typeof v === typeof def) (out as any)[k] = v;
		}
	}
	return out;
}

export interface ModelLike {
	provider?: string;
	id?: string;
	api?: string;
	baseUrl?: string;
	contextWindow?: number;
	/** Static headers from models.json; the SDK sends them on every request. */
	headers?: Record<string, string>;
}

/** Wire formats whose payload and stream shape this extension knows how to extend. */
export type WireApi = "anthropic-messages" | "openai-completions";

export function wireApi(model: ModelLike | undefined): WireApi | undefined {
	const api = model?.api;
	return api === "anthropic-messages" || api === "openai-completions" ? api : undefined;
}

/**
 * Stable fingerprint of the model a request was captured for. A capture is only reusable
 * for the same model: different model = different prefix cache, and the captured `model`
 * field is copied verbatim, so the server would silently serve the old model.
 */
export function modelKey(m: ModelLike | undefined): string | undefined {
	if (!m) return undefined;
	return `${m.provider ?? ""}|${m.baseUrl ?? ""}|${m.id ?? ""}`;
}

/** Anthropic Messages and OpenAI Chat Completions: the two wire formats the payload surgery understands. */
export function appliesTo(model: ModelLike | undefined, cfg: Config): boolean {
	if (!cfg.enabled || !model) return false;
	if (!wireApi(model)) return false;
	const url = String(model.baseUrl ?? "");
	if (cfg.baseUrlExcludes.some((s) => s && url.includes(s))) return false;
	if (cfg.providers.length && !cfg.providers.includes(String(model.provider))) return false;
	if (cfg.baseUrlIncludes.length && !cfg.baseUrlIncludes.some((s) => url.includes(s))) return false;
	return true;
}

export const SUMMARIZER_SYSTEM_MARKER = "context summarization assistant";

/**
 * Request fields that can change the rendered prompt, so the summary request must carry
 * them EXACTLY as captured. Thinking/effort is the subtle one: the toggle differs per
 * model family (Qwen3 `enable_thinking`, DeepSeek-V3.1 / Granite `thinking`, Gemma 4
 * `reasoning_effort` or `enable_thinking`, Holo2 `thinking:false`), vLLM derives
 * `enable_thinking` from `reasoning_effort` (low/medium/high -> true, none -> false),
 * and templates like Qwen3.x render the effort text at the START of the system prompt —
 * so flipping any of them re-prefills the whole conversation.
 *
 * OpenAI Chat Completions adds its own spellings: `enable_thinking` (Qwen via vLLM),
 * `chat_template_args` (Baseten), `response_format` (grammar-constrained output).
 *
 * Everything else (max_tokens, stream, sampling, metadata) never reaches the prompt.
 */
export const PROMPT_AFFECTING_KEYS = [
	"system",
	"tools",
	"tool_choice",
	"messages",
	"thinking",
	"reasoning",
	"reasoning_effort",
	"enable_thinking",
	"output_config",
	"chat_template_kwargs",
	"chat_template_args",
	"response_format",
	"model",
	"mm_processor_kwargs",
	"documents",
] as const;

/** Fields the summary request is allowed to set; anything else is copied verbatim. */
export const SUMMARY_OVERRIDES = ["max_tokens", "max_completion_tokens", "stream", "stream_options"] as const;

export class PrefixChangedError extends Error {}

/**
 * Belt and braces: refuse to send a request whose prompt-affecting fields differ from the
 * captured turn (other than the one appended message). A silent mismatch is not a wrong
 * answer, it is a full cold re-prefill — minutes on a local GPU.
 */
export function assertPrefixPreserved(captured: Record<string, any>, body: Record<string, any>, appended: number): void {
	for (const key of PROMPT_AFFECTING_KEYS) {
		if (key === "messages") continue;
		if (JSON.stringify(captured[key]) !== JSON.stringify(body[key])) {
			throw new PrefixChangedError(`${key} differs from the captured request; that would miss the prefix cache`);
		}
	}
	const cm = captured.messages ?? [];
	const bm = body.messages ?? [];
	if (bm.length !== cm.length + appended) throw new PrefixChangedError("message count changed beyond the appended instruction");
	for (let i = 0; i < cm.length; i++) {
		if (JSON.stringify(cm[i]) !== JSON.stringify(bm[i])) throw new PrefixChangedError(`message ${i} changed; that would miss the prefix cache`);
	}
}

function contentText(content: unknown): string {
	if (typeof content === "string") return content;
	if (Array.isArray(content)) return content.map((b) => (typeof b?.text === "string" ? b.text : "")).join("\n");
	return "";
}

function isSystemRole(m: any): boolean {
	return m?.role === "system" || m?.role === "developer";
}

/**
 * The system prompt text, wherever the wire format puts it: Anthropic's top-level `system`
 * (string or blocks) or OpenAI's leading `system` / `developer` messages.
 */
export function systemText(payload: Record<string, any>): string {
	const parts = [contentText(payload?.system)];
	if (Array.isArray(payload?.messages)) for (const m of payload.messages) if (isSystemRole(m)) parts.push(contentText(m.content));
	return parts.filter(Boolean).join("\n");
}

/**
 * Pi's fallback summarizer wraps the serialized history in these tags
 * (compaction.js: `<conversation>...`), independent of its system prompt wording.
 */
export const CONVERSATION_TAG = "<conversation>";

function hasConversationTag(content: unknown): boolean {
	if (typeof content === "string") return content.includes(CONVERSATION_TAG);
	if (Array.isArray(content)) return content.some((b) => typeof b?.text === "string" && b.text.includes(CONVERSATION_TAG));
	return false;
}

/**
 * A payload worth capturing: a real conversation turn, not Pi's own fallback summarizer.
 * Two filters: the summarizer system prompt wording, and the `<conversation>` tag its
 * requests carry. The tag is only checked on a payload with a SINGLE non-system message,
 * which is what Pi's summarizer sends (buildSummarizationContext: one user message; on
 * OpenAI the system prompt is a `system` message in front of it). Scanning every message
 * would disable capture for a whole session as soon as any turn quotes the literal tag —
 * and a session that discusses this extension does exactly that.
 */
export function isCapturable(payload: unknown): payload is Record<string, any> {
	if (!payload || typeof payload !== "object") return false;
	const p = payload as Record<string, any>;
	if (!Array.isArray(p.messages) || p.messages.length === 0) return false;
	if (systemText(p).includes(SUMMARIZER_SYSTEM_MARKER)) return false;
	const conversation = p.messages.filter((m: any) => !isSystemRole(m));
	if (conversation.length === 0) return false;
	return !(conversation.length === 1 && hasConversationTag(conversation[0]?.content));
}

export function summaryTokenBudget(contextWindow: number, tokensBefore: number, cfg: Config): number | undefined {
	const room = contextWindow - tokensBefore - cfg.promptOverheadTokens;
	const budget = Math.min(cfg.maxSummaryTokens, room);
	return budget >= cfg.minSummaryTokens ? budget : undefined;
}

// Pi's SUMMARIZATION_PROMPT (core/compaction/compaction.js), adapted: the history is the
// conversation above rather than a <conversation> block, and may begin with an earlier checkpoint.
export const SUMMARY_PROMPT = `STOP. Do not continue the task and do not call any tools.

Everything above is this session so far. Create a structured context checkpoint summary that another LLM will use to continue the work. If the conversation starts with an earlier checkpoint summary, merge it in: PRESERVE everything still relevant, move finished items to Done, update Next Steps.

Use this EXACT format:

## Goal
[What is the user trying to accomplish? Can be multiple items if the session covers different tasks.]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned by user]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Current work]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [Ordered list of what should happen next]

## Critical Context
- [Any data, examples, or references needed to continue]
- [Or "(none)" if not applicable]

Keep each section concise. Preserve exact file paths, function names, and error messages. Keep any reasoning brief. Output ONLY the summary.`;

/**
 * The captured request, unchanged up to its last message, plus one instruction.
 * Identical prefix => the server's prefix cache covers everything but the instruction.
 * The summary therefore runs at whatever thinking level the session itself uses; there is
 * deliberately no option to change it (see PROMPT_AFFECTING_KEYS).
 */
export function buildSummaryBody(captured: Record<string, any>, maxTokens: number, api: WireApi = "anthropic-messages"): Record<string, any> {
	const instruction =
		api === "openai-completions" ? { role: "user", content: SUMMARY_PROMPT } : { role: "user", content: [{ type: "text", text: SUMMARY_PROMPT }] };
	const body: Record<string, any> = { ...captured, messages: [...captured.messages, instruction], stream: true };
	setMaxTokens(body, captured, maxTokens);
	assertPrefixPreserved(captured, body, 1);
	return body;
}

/**
 * Set the output cap in the field the captured request used. OpenAI has two spellings
 * (`max_tokens`, `max_completion_tokens`) and Pi picks per provider; sending the one the
 * server has not seen risks a 400, sending both risks a "mutually exclusive" error.
 */
export function setMaxTokens(body: Record<string, any>, captured: Record<string, any>, n: number): void {
	const field = "max_completion_tokens" in captured && !("max_tokens" in captured) ? "max_completion_tokens" : "max_tokens";
	delete body.max_tokens;
	delete body.max_completion_tokens;
	body[field] = n;
}

/** What Pi's model registry resolves at request time (`ctx.modelRegistry.getApiKeyAndHeaders`). */
export interface ResolvedAuth {
	apiKey?: string;
	headers?: Record<string, string>;
	baseUrl?: string;
}

/** Copy string headers (lower-cased), dropping hop-by-hop ones node:http sets itself. */
function copyHeaders(into: Record<string, string>, from: Record<string, unknown> | undefined): void {
	for (const [k, v] of Object.entries(from ?? {})) {
		if (typeof v !== "string") continue;
		const lower = k.toLowerCase();
		if (lower === "content-length" || lower === "host" || lower === "accept-encoding") continue;
		into[lower] = v;
	}
}

/**
 * Headers for the summary request, layered the way the SDK layers them on a real turn:
 * static model headers, then the captured map (routing / attribution / anything another
 * extension added), then Pi's freshly resolved auth headers, then the API key.
 *
 * The key is the important part. `before_provider_headers` runs BEFORE Pi hands the key to
 * the SDK as `apiKey`, so the captured map never contains it (issue #2): replaying the map
 * alone gets a 401 from any authenticated endpoint. The SDKs place it as `x-api-key`
 * (Anthropic, or `Authorization: Bearer` for OAuth tokens) and `Authorization: Bearer`
 * (OpenAI); an explicit auth header already present (models.json `authHeader: true`) wins.
 */
export function requestHeaders(
	api: WireApi,
	captured: Record<string, unknown> | undefined,
	auth: ResolvedAuth | undefined,
	extra: Record<string, string> = {},
	modelHeaders?: Record<string, string>,
): Record<string, string> {
	const out: Record<string, string> = {};
	copyHeaders(out, modelHeaders);
	copyHeaders(out, captured);
	copyHeaders(out, auth?.headers);
	const key = auth?.apiKey;
	if (key && !out.authorization && !out["x-api-key"]) {
		if (api === "openai-completions" || key.startsWith("sk-ant-oat")) out.authorization = `Bearer ${key}`;
		else out["x-api-key"] = key;
	}
	out["content-type"] = "application/json";
	return { ...out, ...extra };
}

/**
 * Where the SDK would POST. Anthropic's client appends `/v1/messages` to the base URL (with
 * a `/v1` base tolerated); OpenAI's appends `/chat/completions` to a base that already
 * ends in `/v1`, which is how Pi custom providers configure it.
 */
export function endpointUrl(baseUrl: string | undefined, api: WireApi): string {
	const base = String(baseUrl ?? "").replace(/\/+$/, "");
	if (api === "openai-completions") return `${base}/chat/completions`;
	return base.endsWith("/v1") ? `${base}/messages` : `${base}/v1/messages`;
}

export interface SseResult {
	text: string;
	stopReason: string;
	usage: Record<string, number>;
}

/**
 * Structurally identical to pi-ai's Usage, kept local so core.ts has no Pi imports.
 * `totalTokens` and `cost` are REQUIRED there: Pi feeds compaction usage into
 * `addUsageToTotals`, which reads `usage.cost.total` — omit it and session stats throw.
 */
export interface UsageLike {
	input: number;
	output: number;
	cacheRead: number;
	cacheWrite: number;
	cacheWrite1h?: number;
	reasoning?: number;
	totalTokens: number;
	cost: { input: number; output: number; cacheRead: number; cacheWrite: number; total: number };
}

/** Anthropic usage fields → Pi's normalized Usage, for CompactionResult.usage. */
export function toPiUsage(u: Record<string, number>): UsageLike {
	const input = u.input_tokens ?? 0;
	const output = u.output_tokens ?? 0;
	const cacheRead = u.cache_read_input_tokens ?? 0;
	const cacheWrite = u.cache_creation_input_tokens ?? 0;
	return {
		input,
		output,
		cacheRead,
		cacheWrite,
		totalTokens: input + output + cacheRead + cacheWrite,
		// Self-hosted endpoints bill nothing; Pi still requires the shape.
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
	};
}

export class SummaryError extends Error {}

/** Incremental SSE parser for one wire format; `push` chunks, then `finish`. */
export interface StreamCollector {
	text: string;
	thinkingChars: number;
	push(chunk: string): void;
	finish(): SseResult;
}

export function createCollector(api: WireApi): StreamCollector {
	return api === "openai-completions" ? new ChatCompletionsCollector() : new SseCollector();
}

/**
 * Split a chunked SSE byte stream into `data:` payloads. Frames end at a blank line; a
 * trailing partial frame stays buffered until the next push (or `flush`).
 */
class SseFrames {
	private buf = "";

	push(chunk: string, onData: (data: string) => void): void {
		this.buf += chunk.replace(/\r\n/g, "\n");
		let idx: number;
		while ((idx = this.buf.indexOf("\n\n")) >= 0) {
			this.frame(this.buf.slice(0, idx), onData);
			this.buf = this.buf.slice(idx + 2);
		}
	}

	flush(onData: (data: string) => void): void {
		if (this.buf.trim()) this.frame(this.buf, onData);
		this.buf = "";
	}

	private frame(frame: string, onData: (data: string) => void): void {
		for (const line of frame.split("\n")) {
			if (!line.startsWith("data:")) continue;
			const data = line.slice(5).trim();
			if (data) onData(data);
		}
	}
}

/**
 * OpenAI Chat Completions stream (`chat.completion.chunk` events, terminated by `[DONE]`).
 * Usage is normalized to the Anthropic field names so `toPiUsage` and the notifications
 * need only one shape. Throws SummaryError on tool calls or a stream error object.
 */
export class ChatCompletionsCollector implements StreamCollector {
	private frames = new SseFrames();
	text = "";
	thinkingChars = 0;
	finishReason = "";
	/** Saw the terminal `[DONE]` sentinel. */
	complete = false;
	usage: Record<string, number> = {};

	push(chunk: string): void {
		this.frames.push(chunk, (d) => this.data(d));
	}

	finish(): SseResult {
		this.frames.flush((d) => this.data(d));
		if (this.finishReason === "length") throw new SummaryError("summary hit the token cap");
		if (this.finishReason === "tool_calls" || this.finishReason === "function_call") throw new SummaryError("model tried to call a tool");
		if (!this.text.trim()) throw new SummaryError("empty summary");
		if (!this.finishReason && !this.complete) throw new SummaryError("stream ended mid-summary (no finish_reason, no [DONE])");
		return { text: this.text.trim(), stopReason: this.finishReason, usage: this.usage };
	}

	private data(data: string): void {
		if (data === "[DONE]") return void (this.complete = true);
		let ev: any;
		try {
			ev = JSON.parse(data);
		} catch {
			return;
		}
		if (ev?.error) throw new SummaryError(`stream error: ${JSON.stringify(ev.error).slice(0, 200)}`);
		if (ev?.usage) this.usage = normalizeChatUsage(ev.usage);
		const choice = Array.isArray(ev?.choices) ? ev.choices[0] : undefined;
		if (!choice) return;
		if (!ev.usage && choice.usage) this.usage = normalizeChatUsage(choice.usage); // Moonshot puts it here
		if (choice.finish_reason) this.finishReason = String(choice.finish_reason);
		const delta = choice.delta ?? {};
		if (Array.isArray(delta.tool_calls) && delta.tool_calls.length) throw new SummaryError("model tried to call a tool");
		if (typeof delta.content === "string") this.text += delta.content;
		for (const f of ["reasoning_content", "reasoning", "reasoning_text"]) {
			if (typeof delta[f] === "string" && delta[f].length) {
				this.thinkingChars += delta[f].length;
				break;
			}
		}
	}
}

/** OpenAI usage → Anthropic field names (same placement rules as pi-ai's parseChunkUsage). */
export function normalizeChatUsage(u: Record<string, any>): Record<string, number> {
	const prompt = u.prompt_tokens || 0;
	const cacheRead = u.prompt_tokens_details?.cached_tokens ?? u.prompt_cache_hit_tokens ?? u.cached_tokens ?? 0;
	const cacheWrite = u.prompt_tokens_details?.cache_write_tokens || 0;
	return {
		input_tokens: Math.max(0, prompt - cacheRead - cacheWrite),
		output_tokens: u.completion_tokens || 0,
		cache_read_input_tokens: cacheRead,
		cache_creation_input_tokens: cacheWrite,
	};
}

/** Incremental Anthropic SSE parser. Throws SummaryError on tool use or stream error. */
export class SseCollector implements StreamCollector {
	private frames = new SseFrames();
	text = "";
	thinkingChars = 0;
	stopReason = "";
	/** Saw the terminal `message_stop` event: the server finished, whatever it reported. */
	complete = false;
	usage: Record<string, number> = {};

	push(chunk: string): void {
		this.frames.push(chunk, (d) => this.data(d));
	}

	finish(): SseResult {
		this.frames.flush((d) => this.data(d));
		if (this.stopReason === "max_tokens") throw new SummaryError("summary hit the token cap");
		if (!this.text.trim()) throw new SummaryError("empty summary");
		// Reject only a stream that was cut off mid-summary: no stop reason AND no terminal
		// `message_stop`. Accepting one would store a truncated checkpoint. A server that
		// ends cleanly without reporting a stop reason is fine — requiring the field would
		// make every compaction fall back on such a server.
		if (!this.stopReason && !this.complete) throw new SummaryError("stream ended mid-summary (no stop reason, no message_stop)");
		return { text: this.text.trim(), stopReason: this.stopReason, usage: this.usage };
	}

	private data(data: string): void {
		if (data === "[DONE]") return;
		let ev: any;
		try {
			ev = JSON.parse(data);
		} catch {
			return;
		}
		switch (ev.type) {
			case "content_block_start":
				if (ev.content_block?.type === "tool_use") throw new SummaryError("model tried to call a tool");
				break;
			case "content_block_delta":
				if (ev.delta?.type === "text_delta") this.text += ev.delta.text;
				else if (ev.delta?.type === "thinking_delta") this.thinkingChars += String(ev.delta.thinking ?? "").length;
				break;
			case "message_start":
				Object.assign(this.usage, ev.message?.usage ?? {});
				break;
			case "message_delta":
				this.stopReason = ev.delta?.stop_reason ?? this.stopReason;
				Object.assign(this.usage, ev.usage ?? {});
				break;
			case "message_stop":
				this.complete = true;
				break;
			case "error":
				throw new SummaryError(`stream error: ${JSON.stringify(ev.error).slice(0, 200)}`);
		}
	}
}

export function fileListSuffix(fileOps: { read?: Iterable<string>; edited?: Iterable<string>; written?: Iterable<string> } | undefined): string {
	if (!fileOps) return "";
	const modified = new Set<string>([...(fileOps.edited ?? []), ...(fileOps.written ?? [])]);
	const read = [...(fileOps.read ?? [])].filter((f) => !modified.has(f)).sort();
	const parts: string[] = [];
	if (read.length) parts.push(`<read-files>\n${read.join("\n")}\n</read-files>`);
	if (modified.size) parts.push(`<modified-files>\n${[...modified].sort().join("\n")}\n</modified-files>`);
	return parts.length ? `\n\n${parts.join("\n\n")}` : "";
}

/**
 * Warm-up body: messages come from Pi's own converter (so they match the next real turn);
 * everything else (system, tools, sampling fields) is the captured request verbatim.
 */
export function buildWarmupBody(captured: Record<string, any>, built: Record<string, any>): Record<string, any> {
	// Everything except `messages` comes from the captured turn, so the warmed prefix is the
	// one the next real turn will send. A 1-token cap is fine: the server stops at max_tokens.
	const body: Record<string, any> = { ...captured, messages: built.messages, stream: built.stream ?? true };
	setMaxTokens(body, captured, 1);
	for (const key of PROMPT_AFFECTING_KEYS) {
		if (key !== "messages") body[key] = captured[key];
	}
	return body;
}

export class HttpStatusError extends Error {}

/**
 * POST and collect an SSE stream with node:http, not fetch: fetch (undici) aborts a
 * response that sends no bytes for 300 s ("terminated"), which is exactly what a long
 * prefill or thinking phase on a local server looks like. Aborts only on `signal`.
 */
export function streamMessages(url: string, headers: Record<string, string>, body: string, sse: StreamCollector, signal: AbortSignal) {
	return new Promise<SseResult>((resolve, reject) => {
		const u = new URL(url);
		const send = u.protocol === "https:" ? httpsRequest : httpRequest;
		const req = send(u, { method: "POST", headers: { ...headers, "content-length": Buffer.byteLength(body) } }, (res) => {
			const status = res.statusCode ?? 0;
			let errBody = "";
			res.setEncoding("utf8");
			res.on("data", (chunk: string) => {
				if (status >= 400) return void (errBody += chunk);
				try {
					sse.push(chunk);
				} catch (err) {
					req.destroy(err as Error);
				}
			});
			res.on("end", () => {
				if (status >= 400) return reject(new HttpStatusError(`HTTP ${status} ${errBody.slice(0, 160)}`));
				try {
					resolve(sse.finish());
				} catch (err) {
					reject(err);
				}
			});
			res.on("error", reject);
		});
		req.setTimeout(0);
		const onAbort = () => req.destroy(new Error("aborted"));
		// The error listener must exist before the pre-abort check: req.destroy(err) emits
		// 'error' asynchronously, and with no listener that is an uncaught exception that
		// kills the process instead of rejecting this promise (graceful cancel).
		req.on("error", reject);
		signal.addEventListener("abort", onAbort, { once: true });
		req.on("close", () => signal.removeEventListener("abort", onAbort));
		if (signal.aborted) return onAbort();
		req.end(body);
	});
}
