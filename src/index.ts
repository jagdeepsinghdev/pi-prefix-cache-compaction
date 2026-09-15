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
 *     thinking disabled -> the prefix is served from cache,
 *  3. after compaction, optionally sends a 1-token warm-up with the new context so the
 *     next turn does not prefill the summary + kept messages cold.
 *
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
	fileListSuffix,
	isCapturable,
	mergeConfig,
	messagesUrl,
	requestHeaders,
	SseCollector,
	summaryTokenBudget,
} from "./core.ts";

const CONFIG_NAME = "pi-prefix-cache-compaction.json";

function readJson(path: string): Partial<Config> | undefined {
	try {
		return existsSync(path) ? JSON.parse(readFileSync(path, "utf8")) : undefined;
	} catch {
		return undefined;
	}
}

function loadConfig(cwd: string): Config {
	const agentDir = process.env.PI_CODING_AGENT_DIR ?? join(homedir(), ".pi", "agent");
	return mergeConfig(readJson(join(agentDir, CONFIG_NAME)), readJson(join(cwd, ".pi", CONFIG_NAME)));
}

type Capture = { payload?: Record<string, any>; headers?: Record<string, unknown> };

const stats = { compactions: 0, fallbacks: 0, warmups: 0, lastSeconds: 0, lastReason: "" };

export default function prefixCacheCompaction(pi: ExtensionAPI) {
	const captures = new Map<string, Capture>(); // Pi session id -> last real request
	let config: Config | undefined;
	const cfg = (ctx: ExtensionContext) => (config ??= loadConfig(ctx.cwd));
	const notify = (ctx: ExtensionContext, msg: string, kind: "info" | "warning" = "info") => {
		if (cfg(ctx).notify) ctx.ui.notify(msg, kind);
	};
	const fallback = (ctx: ExtensionContext, reason: string) => {
		stats.fallbacks++;
		stats.lastReason = reason;
		notify(ctx, `Prefix-cache compaction skipped (${reason}); using Pi default`, "warning");
	};

	pi.on("session_start", async (_event, ctx) => {
		config = loadConfig(ctx.cwd);
	});

	pi.on("before_provider_headers", (event, ctx) => {
		if (!appliesTo(ctx.model, cfg(ctx))) return;
		const id = ctx.sessionManager.getSessionId();
		const cap = captures.get(id) ?? {};
		cap.headers = event.headers as Record<string, unknown>; // copied at use time
		captures.set(id, cap);
	});

	pi.on("before_provider_request", (event, ctx) => {
		if (!appliesTo(ctx.model, cfg(ctx)) || !isCapturable(event.payload)) return;
		const id = ctx.sessionManager.getSessionId();
		const cap = captures.get(id) ?? {};
		cap.payload = structuredClone(event.payload as Record<string, any>);
		captures.set(id, cap);
	});

	pi.on("session_before_compact", async (event, ctx) => {
		const c = cfg(ctx);
		if (!appliesTo(ctx.model, c)) return;
		if (event.reason === "overflow") return; // context already over the window; nothing to reuse
		const id = ctx.sessionManager.getSessionId();
		const cap = captures.get(id);
		if (!cap?.payload) return fallback(ctx, "no captured request yet in this process");

		const { preparation, signal } = event;
		const budget = summaryTokenBudget(ctx.model?.contextWindow ?? 0, preparation.tokensBefore, c);
		if (!budget) return fallback(ctx, "not enough room left in the context window");

		const t0 = Date.now();
		notify(ctx, `Compaction: reusing cached prefix (${preparation.tokensBefore.toLocaleString()} tokens), thinking off`);
		try {
			const res = await fetch(messagesUrl(ctx.model?.baseUrl), {
				method: "POST",
				headers: requestHeaders(cap.headers, { "x-session-id": id }),
				body: JSON.stringify(buildSummaryBody(cap.payload, budget, c)),
				signal,
			});
			if (!res.ok || !res.body) {
				const detail = (await res.text().catch(() => "")).slice(0, 160);
				return fallback(ctx, `HTTP ${res.status} ${detail}`);
			}
			const sse = new SseCollector();
			const decoder = new TextDecoder();
			for await (const chunk of res.body as unknown as AsyncIterable<Uint8Array>) {
				sse.push(decoder.decode(chunk, { stream: true }));
			}
			const { text, usage } = sse.finish();
			const secs = Math.round((Date.now() - t0) / 1000);
			stats.compactions++;
			stats.lastSeconds = secs;
			notify(ctx, `Compaction done in ${secs}s (${usage.output_tokens ?? "?"} tokens out)`);
			return {
				compaction: {
					summary: text + fileListSuffix(preparation.fileOps),
					firstKeptEntryId: preparation.firstKeptEntryId,
					tokensBefore: preparation.tokensBefore,
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
		if (!c.warmup || !appliesTo(ctx.model, c) || event.willRetry) return;
		const id = ctx.sessionManager.getSessionId();
		const cap = captures.get(id);
		if (!cap?.payload || !ctx.model) return;
		try {
			const session = buildSessionContext(ctx.sessionManager.getEntries(), ctx.sessionManager.getLeafId());
			const messages = convertToLlm(session.messages);
			const headers = requestHeaders(cap.headers, { "x-session-id": id });
			const t0 = Date.now();
			await completeSimple(
				ctx.model,
				{ systemPrompt: "", messages, tools: [] },
				{
					maxTokens: 1,
					headers,
					apiKey: bearer(headers) ?? "local",
					onPayload: (built) => buildWarmupBody(cap.payload!, built as Record<string, any>),
				},
			);
			stats.warmups++;
			notify(ctx, `Context re-warmed in ${Math.round((Date.now() - t0) / 1000)}s; next turn starts from cache`);
		} catch {
			// Warm-up is best effort; the next turn simply prefills normally.
		}
	});

	pi.registerCommand("prefix-compaction", {
		description: "pi-prefix-cache-compaction: status and config",
		handler: async (_args, ctx) => {
			const c = cfg(ctx);
			ctx.ui.notify(
				[
					`applies to current model: ${appliesTo(ctx.model, c)}`,
					`cached compactions: ${stats.compactions} (last ${stats.lastSeconds}s), fallbacks: ${stats.fallbacks}${stats.lastReason ? ` (last: ${stats.lastReason})` : ""}, warm-ups: ${stats.warmups}`,
					`config: ${JSON.stringify(c)}`,
				].join("\n"),
				"info",
			);
		},
	});
}

function bearer(headers: Record<string, string>): string | undefined {
	const auth = headers.authorization;
	if (auth?.toLowerCase().startsWith("bearer ")) return auth.slice(7);
	return headers["x-api-key"];
}
