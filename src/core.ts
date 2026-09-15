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
	/**
	 * Force thinking off for the summary request. Default false: many chat templates
	 * (Qwen3.x among them) put the reasoning-effort text at the START of the system
	 * prompt, so changing thinking changes the whole prefix and the cache misses.
	 * Only enable for templates where thinking affects the generation prompt alone.
	 */
	disableThinking: boolean;
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
	disableThinking: false,
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
	api?: string;
	baseUrl?: string;
	contextWindow?: number;
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

export function systemText(payload: Record<string, any>): string {
	const s = payload?.system;
	if (typeof s === "string") return s;
	if (Array.isArray(s)) return s.map((b) => (typeof b?.text === "string" ? b.text : "")).join("\n");
	return "";
}

/** A payload worth capturing: a real conversation turn, not Pi's own fallback summarizer. */
export function isCapturable(payload: unknown): payload is Record<string, any> {
	if (!payload || typeof payload !== "object") return false;
	const p = payload as Record<string, any>;
	if (!Array.isArray(p.messages) || p.messages.length === 0) return false;
	return !systemText(p).includes(SUMMARIZER_SYSTEM_MARKER);
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
 * Thinking fields are kept as captured unless cfg.disableThinking (see Config).
 */
export function buildSummaryBody(captured: Record<string, any>, maxTokens: number, cfg: Config): Record<string, any> {
	const body: Record<string, any> = {
		...captured,
		messages: [...captured.messages, { role: "user", content: [{ type: "text", text: SUMMARY_PROMPT }] }],
		max_tokens: maxTokens,
		stream: true,
	};
	if (cfg.disableThinking) {
		body.thinking = { type: "disabled" };
		delete body.output_config;
		delete body.reasoning_effort;
		// vLLM ignores Anthropic thinking.type for Qwen-style templates; this is what it honors.
		body.chat_template_kwargs = { ...(captured.chat_template_kwargs ?? {}), enable_thinking: false };
		delete body.chat_template_kwargs.reasoning_effort;
	}
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
	// Thinking fields stay as captured: in templates that render reasoning effort into the
	// system prompt, changing them would warm a different prefix than the next real turn.
	// A 1-token request is fine: the server stops at max_tokens, thinking or not.
	return { ...captured, messages: built.messages, max_tokens: 1, stream: built.stream ?? true };
}
