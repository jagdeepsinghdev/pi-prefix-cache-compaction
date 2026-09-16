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
 * Anything unexpected returns nothing, and Pi runs its default compaction.
 * Technique credit: pisceslailai/deepseek-kvcache (MIT), adapted to Anthropic Messages.
 */
import { existsSync, readFileSync } from "node:fs";
import { request as httpRequest } from "node:http";
import { request as httpsRequest } from "node:https";
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
	modelKey,
	requestHeaders,
	SseCollector,
	summaryTokenBudget,
	toPiUsage,
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

const stats = { compactions: 0, fallbacks: 0, warmups: 0, lastSeconds: 0, lastReason: "" };

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
		if (!appliesTo(ctx.model, c)) return;
		if (event.reason === "overflow") return; // context already over the window; nothing to reuse
		const id = ctx.sessionManager.getSessionId();
		const cap = captures.get(id);
		if (!cap?.payload) return fallback(ctx, "no captured request yet in this process");
		if (cap.modelKey !== modelKey(ctx.model)) return fallback(ctx, "last captured request was from a different model");
		if (cap.stale) return fallback(ctx, "capture is stale after an earlier default compaction");

		const { preparation, signal } = event;
		const budget = summaryTokenBudget(ctx.model?.contextWindow ?? 0, preparation.tokensBefore, c);
		if (!budget) return fallback(ctx, "not enough room left in the context window");

		const t0 = Date.now();
		notify(ctx, `Compaction: reusing cached prefix (${preparation.tokensBefore.toLocaleString()} tokens)`);
		try {
			const { text, usage } = await streamMessages(
				messagesUrl(ctx.model?.baseUrl),
				requestHeaders(cap.headers, { "x-session-id": id }),
				JSON.stringify(buildSummaryBody(cap.payload, budget)),
				signal,
			);
			const secs = Math.round((Date.now() - t0) / 1000);
			stats.compactions++;
			stats.lastSeconds = secs;
			notify(ctx, `Compaction done in ${secs}s (${usage.output_tokens ?? "?"} tokens out)`);
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
			const { config: c, problems } = loadConfig(ctx.cwd); // fresh read so edits are visible
			config = c;
			ctx.ui.notify(
				[
					`applies to current model: ${appliesTo(ctx.model, c)}`,
					`cached compactions: ${stats.compactions} (last ${stats.lastSeconds}s), fallbacks: ${stats.fallbacks}${stats.lastReason ? ` (last: ${stats.lastReason})` : ""}, warm-ups: ${stats.warmups}`,
					`config: ${JSON.stringify(c)}`,
					...problems.map((p) => `config problem: ${p}`),
				].join("\n"),
				"info",
			);
		},
	});
}

class HttpStatusError extends Error {}

/**
 * POST and collect an Anthropic SSE stream with node:http, not fetch: fetch (undici)
 * aborts a response that sends no bytes for 300 s ("terminated"), which is exactly what
 * a long prefill or thinking phase on a local server looks like. Aborts only on `signal`.
 */
function streamMessages(url: string, headers: Record<string, string>, body: string, signal: AbortSignal) {
	return new Promise<ReturnType<SseCollector["finish"]>>((resolve, reject) => {
		const u = new URL(url);
		const send = u.protocol === "https:" ? httpsRequest : httpRequest;
		const req = send(u, { method: "POST", headers: { ...headers, "content-length": Buffer.byteLength(body) } }, (res) => {
			const status = res.statusCode ?? 0;
			const sse = new SseCollector();
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

function bearer(headers: Record<string, string>): string | undefined {
	const auth = headers.authorization;
	if (auth?.toLowerCase().startsWith("bearer ")) return auth.slice(7);
	return headers["x-api-key"];
}
