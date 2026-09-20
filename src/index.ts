/**
 * pi-prefix-cache-compaction
 *
 * Pi's default compaction serializes the history into ONE new user message under a
 * different system prompt, so a server-side prefix cache (vLLM, SGLang, llama.cpp)
 * misses completely and the whole history is prefilled again. On local GPUs that is
 * minutes per compaction.
 *
 * This extension instead:
 *  1. captures each real provider request (before_provider_request),
 *  2. on compaction re-sends that exact request plus one summarize instruction,
 *     thinking settings untouched -> the prefix is served from cache (Qwen-style
 *     templates render reasoning effort into the system prompt, so changing thinking
 *     would change the whole prefix),
 *  3. after compaction, optionally sends a 1-token warm-up with the new context so the
 *     next turn does not prefill the summary + kept messages cold.
 *
 * Works for the Anthropic Messages and OpenAI Chat Completions wire formats.
 * Anything unexpected returns nothing, and Pi runs its default compaction.
 * Technique credit: pisceslailai/deepseek-kvcache (MIT), adapted to Anthropic Messages.
 */
import { existsSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { buildSessionContext, convertToLlm } from "@earendil-works/pi-coding-agent";
import { completeSimple } from "@earendil-works/pi-ai/compat";
import {
	appliesTo,
	buildSummaryBody,
	buildWarmupBody,
	type Config,
	createCollector,
	endpointUrl,
	fileListSuffix,
	isCapturable,
	mergeConfig,
	modelKey,
	requestHeaders,
	type ResolvedAuth,
	streamMessages,
	summaryTokenBudget,
	toPiUsage,
	wireApi,
} from "./core.ts";

const CONFIG_NAME = "pi-prefix-cache-compaction.json";

function readJsonFile(path: string, problems: string[]): Partial<Config> | undefined {
	if (!existsSync(path)) return undefined;
	try {
		return JSON.parse(readFileSync(path, "utf8"));
	} catch (err) {
		problems.push(`${path}: ${(err as Error).message}`);
		return undefined;
	}
}

function loadConfig(cwd: string): { config: Config; problems: string[] } {
	const problems: string[] = [];
	const agentDir = process.env.PI_CODING_AGENT_DIR ?? join(homedir(), ".pi", "agent");
	const config = mergeConfig(readJsonFile(join(agentDir, CONFIG_NAME), problems), readJsonFile(join(cwd, ".pi", CONFIG_NAME), problems));
	return { config, problems };
}

type Capture = {
	payload?: Record<string, any>;
	headers?: Record<string, unknown>;
	/** modelKey() of the model the payload was captured for (undefined = legacy capture). */
	modelKey?: string;
	/** True after a Pi-default compaction invalidated the prefix; a new real turn clears it. */
	stale?: boolean;
};

const stats = { compactions: 0, fallbacks: 0, warmups: 0, lastSeconds: 0, lastReason: "", lastWarmupError: "" };

/**
 * Request-time auth, the same way Pi resolves it for a real turn. `before_provider_headers`
 * sees Pi's header map BEFORE the API key is attached (the key goes to the SDK as `apiKey`),
 * so the captured headers alone are keyless against any authenticated endpoint (issue #2).
 * Older Pi builds without `getApiKeyAndHeaders` fall back to the captured map only.
 */
async function resolveAuth(ctx: ExtensionContext): Promise<{ auth?: ResolvedAuth; error?: string }> {
	const registry = ctx.modelRegistry as { getApiKeyAndHeaders?: (m: any) => Promise<any> } | undefined;
	if (!ctx.model || typeof registry?.getApiKeyAndHeaders !== "function") return {};
	try {
		const r = await registry.getApiKeyAndHeaders(ctx.model);
		if (!r?.ok) return { error: r?.error ?? "auth resolution failed" };
		return { auth: { apiKey: r.apiKey, headers: r.headers, baseUrl: r.baseUrl } };
	} catch (err) {
		return { error: (err as Error).message };
	}
}

export default function prefixCacheCompaction(pi: ExtensionAPI) {
	const captures = new Map<string, Capture>(); // Pi session id -> last real request
	let config: Config | undefined;
	const cfg = (ctx: ExtensionContext) => (config ??= loadConfig(ctx.cwd).config);
	const notify = (ctx: ExtensionContext, msg: string, kind: "info" | "warning" = "info") => {
		if (cfg(ctx).notify) ctx.ui.notify(msg, kind);
	};
	const fallback = (ctx: ExtensionContext, reason: string) => {
		stats.fallbacks++;
		stats.lastReason = reason;
		notify(ctx, `Prefix-cache compaction skipped (${reason}); using Pi default`, "warning");
	};

	pi.on("session_start", async (_event, ctx) => {
		const { config: c, problems } = loadConfig(ctx.cwd);
		config = c;
		if (problems.length && c.notify) ctx.ui.notify(`pi-prefix-cache-compaction: ignoring malformed config (${problems.join("; ")})`, "warning");
	});

	// Release the (potentially megabyte-scale) captured payload when a session goes away.
	pi.on("session_shutdown", (_event, ctx) => {
		captures.delete(ctx.sessionManager.getSessionId());
	});

	pi.on("before_provider_headers", (event, ctx) => {
		if (!appliesTo(ctx.model, cfg(ctx))) return;
		const id = ctx.sessionManager.getSessionId();
		const cap = captures.get(id) ?? {};
		cap.headers = { ...event.headers }; // copy: later extensions may mutate event.headers in place
		captures.set(id, cap);
	});

	pi.on("before_provider_request", (event, ctx) => {
		if (!appliesTo(ctx.model, cfg(ctx)) || !isCapturable(event.payload)) return;
		const id = ctx.sessionManager.getSessionId();
		const cap = captures.get(id) ?? {};
		cap.payload = structuredClone(event.payload as Record<string, any>);
		cap.modelKey = modelKey(ctx.model);
		cap.stale = false; // this real turn re-anchors the prefix
		captures.set(id, cap);
	});

	pi.on("session_before_compact", async (event, ctx) => {
		const c = cfg(ctx);
		const api = wireApi(ctx.model);
		if (!api || !appliesTo(ctx.model, c)) return;
		if (event.reason === "overflow") return; // context already over the window; nothing to reuse
		const id = ctx.sessionManager.getSessionId();
		const cap = captures.get(id);
		if (!cap?.payload) return fallback(ctx, "no captured request yet in this process");
		if (cap.modelKey !== modelKey(ctx.model)) return fallback(ctx, "last captured request was from a different model");
		if (cap.stale) return fallback(ctx, "capture is stale after an earlier default compaction");

		const { preparation, signal } = event;
		const budget = summaryTokenBudget(ctx.model?.contextWindow ?? 0, preparation.tokensBefore, c);
		if (!budget) return fallback(ctx, "not enough room left in the context window");

		const { auth, error } = await resolveAuth(ctx);
		if (error) return fallback(ctx, `auth: ${error}`);

		const t0 = Date.now();
		notify(ctx, `Compaction: reusing cached prefix (${preparation.tokensBefore.toLocaleString()} tokens)`);
		try {
			const { text, usage } = await streamMessages(
				endpointUrl(auth?.baseUrl ?? ctx.model?.baseUrl, api),
				requestHeaders(api, cap.headers, auth, { "x-session-id": id }, ctx.model?.headers),
				JSON.stringify(buildSummaryBody(cap.payload, budget, api)),
				createCollector(api),
				signal,
			);
			const secs = Math.round((Date.now() - t0) / 1000);
			stats.compactions++;
			stats.lastSeconds = secs;
			const cached = usage.cache_read_input_tokens;
			notify(ctx, `Compaction done in ${secs}s (${usage.output_tokens ?? "?"} tokens out${cached ? `, ${cached.toLocaleString()} prompt tokens served from prefix cache` : ""})`);
			return {
				compaction: {
					summary: text + fileListSuffix(preparation.fileOps),
					firstKeptEntryId: preparation.firstKeptEntryId,
					tokensBefore: preparation.tokensBefore,
					usage: toPiUsage(usage),
				},
			};
		} catch (err) {
			if (signal.aborted) return { cancel: true };
			return fallback(ctx, (err as Error).message);
		}
	});

	// Warm the new prefix (summary + kept messages) so the next turn is not a cold prefill.
	pi.on("session_compact", async (event, ctx) => {
		const c = cfg(ctx);
		const id = ctx.sessionManager.getSessionId();
		const cap = captures.get(id);
		// A Pi-default compaction means the captured prefix no longer describes the session.
		// Keep it only to feed the warm-up below; never for a summary again until a real turn
		// re-anchors it (before_provider_request clears `stale`).
		if (cap && !event.fromExtension) cap.stale = true;
		if (!c.warmup || !appliesTo(ctx.model, c) || event.willRetry) return;
		if (!cap?.payload || !ctx.model) return;
		if (cap.modelKey !== modelKey(ctx.model)) return; // warmed prefix would not match the new model
		try {
			const session = buildSessionContext(ctx.sessionManager.getEntries(), ctx.sessionManager.getLeafId());
			const messages = convertToLlm(session.messages);
			// The captured map only contributes routing / attribution headers; the credential
			// is resolved at request time, never replayed from the capture (issue #2).
			const headers = requestHeaders(wireApi(ctx.model)!, cap.headers, undefined, { "x-session-id": id });
			delete headers["content-type"];
			const context = { systemPrompt: "", messages, tools: [] };
			const onPayload = (built: unknown) => buildWarmupBody(cap.payload!, built as Record<string, any>);
			const t0 = Date.now();
			const registry = ctx.modelRegistry as unknown as { streamSimple?: (m: any, c: any, o: any) => { result(): Promise<any> } };
			let result: { stopReason?: string; errorMessage?: string };
			if (typeof registry.streamSimple === "function") {
				// Pi >= 0.86: the registry streams with request-time auth exactly like a real turn.
				result = await registry.streamSimple(ctx.model, context, { maxTokens: 1, headers, onPayload }).result();
			} else {
				// Pi 0.85: the registry facade cannot stream. Resolve auth through it and call the
				// same pi-ai entrypoint Pi 0.85 uses for its own compaction.
				const { auth, error } = await resolveAuth(ctx);
				if (error) throw new Error(`auth: ${error}`);
				const model = auth?.baseUrl ? { ...ctx.model, baseUrl: auth.baseUrl } : ctx.model;
				result = await completeSimple(model, context, {
					maxTokens: 1,
					headers: { ...headers, ...(auth?.headers ?? {}) },
					apiKey: auth?.apiKey,
					onPayload,
				});
			}
			if (result.stopReason === "error" || result.stopReason === "aborted") {
				throw new Error(result.errorMessage ?? `stop reason ${result.stopReason}`);
			}
			stats.warmups++;
			stats.lastWarmupError = "";
			notify(ctx, `Context re-warmed in ${Math.round((Date.now() - t0) / 1000)}s; next turn starts from cache`);
		} catch (err) {
			// Warm-up is best effort; the next turn simply prefills normally. The reason is
			// kept for /prefix-compaction so a silently failing warm-up is diagnosable.
			stats.lastWarmupError = (err as Error).message ?? String(err);
			if (process.env.PI_PREFIX_CACHE_DEBUG) console.error(`[pi-prefix-cache-compaction] warm-up failed: ${stats.lastWarmupError}`);
		}
	});

	pi.registerCommand("prefix-compaction", {
		description: "pi-prefix-cache-compaction: status and config",
		handler: async (_args, ctx) => {
			const { config: c, problems } = loadConfig(ctx.cwd); // fresh read so edits are visible
			config = c;
			ctx.ui.notify(
				[
					`applies to current model: ${appliesTo(ctx.model, c)} (api: ${ctx.model?.api ?? "none"})`,
					`cached compactions: ${stats.compactions} (last ${stats.lastSeconds}s), fallbacks: ${stats.fallbacks}${stats.lastReason ? ` (last: ${stats.lastReason})` : ""}, warm-ups: ${stats.warmups}${stats.lastWarmupError ? ` (last warm-up error: ${stats.lastWarmupError})` : ""}`,
					`config: ${JSON.stringify(c)}`,
					...problems.map((p) => `config problem: ${p}`),
				].join("\n"),
				"info",
			);
		},
	});
}
