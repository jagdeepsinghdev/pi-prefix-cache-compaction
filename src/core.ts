/**
 * Pure logic for pi-prefix-cache-compaction. No Pi imports, so it is unit-testable
 * with plain `node --test`.
 */

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

/** Only the Anthropic Messages wire format is supported (that is what the payload surgery assumes). */
export function appliesTo(model: ModelLike | undefined, cfg: Config): boolean {
	if (!cfg.enabled || !model) return false;
	if (model.api !== "anthropic-messages") return false;
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
	"output_config",
	"chat_template_kwargs",
	"model",
	"mm_processor_kwargs",
	"documents",
] as const;

/** Fields the summary request is allowed to set; anything else is copied verbatim. */
export const SUMMARY_OVERRIDES = ["max_tokens", "stream", "stream_options"] as const;

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

export function systemText(payload: Record<string, any>): string {
	const s = payload?.system;
	if (typeof s === "string") return s;
	if (Array.isArray(s)) return s.map((b) => (typeof b?.text === "string" ? b.text : "")).join("\n");
	return "";
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
 * requests always carry. A genuine user message containing the literal tag merely skips
 * one capture (the previous turn stays the anchor); it can never capture a summarizer.
 */
export function isCapturable(payload: unknown): payload is Record<string, any> {
	if (!payload || typeof payload !== "object") return false;
	const p = payload as Record<string, any>;
	if (!Array.isArray(p.messages) || p.messages.length === 0) return false;
	if (systemText(p).includes(SUMMARIZER_SYSTEM_MARKER)) return false;
	return !p.messages.some((m) => hasConversationTag(m?.content));
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
export function buildSummaryBody(captured: Record<string, any>, maxTokens: number): Record<string, any> {
	const body: Record<string, any> = {
		...captured,
		messages: [...captured.messages, { role: "user", content: [{ type: "text", text: SUMMARY_PROMPT }] }],
		max_tokens: maxTokens,
		stream: true,
	};
	assertPrefixPreserved(captured, body, 1);
	return body;
}

/** Copy string headers, force JSON, and keep auth/routing headers from the captured request. */
export function requestHeaders(captured: Record<string, unknown> | undefined, extra: Record<string, string> = {}) {
	const out: Record<string, string> = {};
	for (const [k, v] of Object.entries(captured ?? {})) {
		if (typeof v !== "string") continue;
		const lower = k.toLowerCase();
		if (lower === "content-length" || lower === "host" || lower === "accept-encoding") continue;
		out[lower] = v;
	}
	out["content-type"] = "application/json";
	return { ...out, ...extra };
}

export function messagesUrl(baseUrl: string | undefined): string {
	const base = String(baseUrl ?? "").replace(/\/+$/, "");
	return base.endsWith("/v1") ? `${base}/messages` : `${base}/v1/messages`;
}

export interface SseResult {
	text: string;
	stopReason: string;
	usage: Record<string, number>;
}

/** Structurally identical to pi-ai's Usage, kept local so core.ts has no Pi imports. */
export interface UsageLike {
	input: number;
	output: number;
	cacheRead: number;
	cacheWrite: number;
	cacheWrite1h?: number;
	reasoning?: number;
}

/** Anthropic usage fields → Pi's normalized Usage, for CompactionResult.usage. */
export function toPiUsage(u: Record<string, number>): UsageLike {
	return {
		input: u.input_tokens ?? 0,
		output: u.output_tokens ?? 0,
		cacheRead: u.cache_read_input_tokens ?? 0,
		cacheWrite: u.cache_creation_input_tokens ?? 0,
	};
}

export class SummaryError extends Error {}

/** Incremental Anthropic SSE parser. Throws SummaryError on tool use or stream error. */
export class SseCollector {
	private buf = "";
	text = "";
	thinkingChars = 0;
	stopReason = "";
	usage: Record<string, number> = {};

	push(chunk: string): void {
		this.buf += chunk.replace(/\r\n/g, "\n");
		let idx: number;
		while ((idx = this.buf.indexOf("\n\n")) >= 0) {
			this.frame(this.buf.slice(0, idx));
			this.buf = this.buf.slice(idx + 2);
		}
	}

	finish(): SseResult {
		if (this.buf.trim()) this.frame(this.buf);
		this.buf = "";
		if (this.stopReason === "max_tokens") throw new SummaryError("summary hit the token cap");
		if (!this.text.trim()) throw new SummaryError("empty summary");
		// A stream that carried text but never a stop reason was cut off mid-summary
		// (server closed the socket); accept it and you keep a truncated checkpoint.
		if (!this.stopReason) throw new SummaryError("stream ended without a stop reason");
		return { text: this.text.trim(), stopReason: this.stopReason, usage: this.usage };
	}

	private frame(frame: string): void {
		for (const line of frame.split("\n")) {
			if (!line.startsWith("data:")) continue;
			const data = line.slice(5).trim();
			if (!data || data === "[DONE]") continue;
			let ev: any;
			try {
				ev = JSON.parse(data);
			} catch {
				continue;
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
				case "error":
					throw new SummaryError(`stream error: ${JSON.stringify(ev.error).slice(0, 200)}`);
			}
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
	const body: Record<string, any> = { ...captured, messages: built.messages, max_tokens: 1, stream: built.stream ?? true };
	for (const key of PROMPT_AFFECTING_KEYS) {
		if (key !== "messages") body[key] = captured[key];
	}
	return body;
}
