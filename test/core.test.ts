import assert from "node:assert/strict";
import { test } from "node:test";
import {
	appliesTo,
	buildSummaryBody,
	buildWarmupBody,
	ChatCompletionsCollector,
	createCollector,
	DEFAULT_CONFIG,
	endpointUrl,
	fileListSuffix,
	isCapturable,
	mergeConfig,
	modelKey,
	normalizeChatUsage,
	requestHeaders,
	assertPrefixPreserved,
	PrefixChangedError,
	SseCollector,
	SummaryError,
	SUMMARY_PROMPT,
	summaryTokenBudget,
	systemText,
	toPiUsage,
	wireApi,
} from "../src/core.ts";

const local = { provider: "qwen-local", api: "anthropic-messages", baseUrl: "http://127.0.0.1:18770", contextWindow: 262_144 };

test("mergeConfig: later layers win, wrong types and unknown keys ignored", () => {
	const c = mergeConfig({ maxSummaryTokens: 8000, providers: ["a"] }, { maxSummaryTokens: "x" as any, bogus: 1 } as any, {
		warmup: false,
	});
	assert.equal(c.maxSummaryTokens, 8000);
	assert.deepEqual(c.providers, ["a"]);
	assert.equal(c.warmup, false);
	assert.equal((c as any).bogus, undefined);
});

test("appliesTo: anthropic-messages and openai-completions local servers by default", () => {
	assert.equal(appliesTo(local, DEFAULT_CONFIG), true);
	assert.equal(appliesTo({ ...local, api: "openai-completions" }, DEFAULT_CONFIG), true);
	assert.equal(appliesTo({ ...local, api: "openai-responses" }, DEFAULT_CONFIG), false);
	assert.equal(wireApi({ api: "openai-completions" }), "openai-completions");
	assert.equal(wireApi({ api: "openai-responses" }), undefined);
	assert.equal(appliesTo({ ...local, baseUrl: "https://api.anthropic.com" }, DEFAULT_CONFIG), false);
	assert.equal(appliesTo(undefined, DEFAULT_CONFIG), false);
	assert.equal(appliesTo(local, mergeConfig({ enabled: false })), false);
	assert.equal(appliesTo(local, mergeConfig({ providers: ["other"] })), false);
	assert.equal(appliesTo(local, mergeConfig({ baseUrlIncludes: [":18770"] })), true);
	assert.equal(appliesTo(local, mergeConfig({ baseUrlIncludes: [":9999"] })), false);
});

test("isCapturable: real turns yes, Pi's fallback summarizer and empty payloads no", () => {
	assert.equal(isCapturable({ system: "You are pi", messages: [{ role: "user", content: "hi" }] }), true);
	assert.equal(
		isCapturable({ system: [{ type: "text", text: "You are a context summarization assistant." }], messages: [{}] }),
		false,
	);
	assert.equal(isCapturable({ messages: [] }), false);
	// Pi's summarizer sends exactly one user message wrapping the history in <conversation>
	assert.equal(isCapturable({ system: "You are pi", messages: [{ role: "user", content: "<conversation>\nx\n</conversation>" }] }), false);
	// but a real multi-turn chat that merely QUOTES the tag must still be captured,
	// otherwise one such turn disables the extension for the rest of the session
	assert.equal(
		isCapturable({
			system: "You are pi",
			messages: [
				{ role: "user", content: "what does <conversation> mean in the summarizer?" },
				{ role: "assistant", content: "it wraps the history" },
			],
		}),
		true,
	);
	assert.equal(isCapturable(null), false);
});

test("isCapturable: <conversation>-tagged requests are filtered even with an unknown system prompt", () => {
	const sys = "Some totally different system prompt";
	assert.equal(
		isCapturable({ system: sys, messages: [{ role: "user", content: "<conversation>\n[User]: hi\n</conversation>\n\nSummarize." }] }),
		false,
	);
	assert.equal(
		isCapturable({ system: sys, messages: [{ role: "user", content: [{ type: "text", text: "<conversation>\nx\n</conversation>" }] }] }),
		false,
	);
	// a normal message that merely mentions the word stays capturable
	assert.equal(isCapturable({ system: sys, messages: [{ role: "user", content: "write a conversation serializer" }] }), true);
	// …while one containing the literal tag skips this one capture (previous turn stays the anchor)
	assert.equal(isCapturable({ system: sys, messages: [{ role: "user", content: "write a <conversation> serializer" }] }), false);
});

test("modelKey: stable per model identity, distinct per provider/baseUrl/id", () => {
	const a = { provider: "qwen-local", id: "qwen3.8-27b", baseUrl: "http://127.0.0.1:18770" };
	assert.equal(modelKey(a), modelKey({ ...a }));
	assert.notEqual(modelKey(a), modelKey({ ...a, id: "other" }));
	assert.notEqual(modelKey(a), modelKey({ ...a, provider: "other-local" }));
	assert.notEqual(modelKey(a), modelKey({ ...a, baseUrl: "http://127.0.0.1:9999" }));
	assert.equal(modelKey(undefined), undefined);
});

test("summaryTokenBudget: capped, and refuses when the window is nearly full", () => {
	assert.equal(summaryTokenBudget(262_144, 100_000, DEFAULT_CONFIG), 16_000);
	assert.equal(summaryTokenBudget(262_144, 250_000, DEFAULT_CONFIG), 9_144);
	assert.equal(summaryTokenBudget(262_144, 256_000, DEFAULT_CONFIG), undefined);
	assert.equal(summaryTokenBudget(0, 1, DEFAULT_CONFIG), undefined);
});

test("buildSummaryBody: prefix untouched, one instruction appended, thinking as captured", () => {
	const captured = {
		model: "m",
		system: [{ type: "text", text: "sys" }],
		tools: [{ name: "read" }],
		messages: [
			{ role: "user", content: "a" },
			{ role: "assistant", content: "b" },
		],
		thinking: { type: "adaptive" },
		output_config: { effort: "xhigh" },
		stream: true,
		max_tokens: 32000,
	};
	const snapshot = structuredClone(captured);
	const body = buildSummaryBody(captured, 9000);
	assert.deepEqual(captured, snapshot, "captured payload must not be mutated");
	assert.deepEqual(body.messages.slice(0, 2), captured.messages);
	assert.equal(body.messages.length, 3);
	assert.equal(body.messages[2].content[0].text, SUMMARY_PROMPT);
	assert.deepEqual(body.system, captured.system);
	assert.deepEqual(body.tools, captured.tools);
	assert.equal(body.max_tokens, 9000);
	// default keeps thinking exactly as captured: Qwen-style templates render reasoning
	// effort at the start of the system prompt, so changing it would miss the cache
	assert.deepEqual(body.thinking, { type: "adaptive" });
	assert.deepEqual(body.output_config, { effort: "xhigh" });
	assert.equal(body.chat_template_kwargs, undefined);

});

test("assertPrefixPreserved: catches every field that could re-render the prompt", () => {
	const captured = {
		system: "s",
		tools: [{ name: "read" }],
		messages: [{ role: "user", content: "a" }],
		thinking: { type: "adaptive" },
		output_config: { effort: "xhigh" },
		chat_template_kwargs: { enable_thinking: true },
		reasoning_effort: "high",
		model: "m",
	};
	const ok = buildSummaryBody(captured, 9000);
	assert.doesNotThrow(() => assertPrefixPreserved(captured, ok, 1));
	// sampling / streaming knobs never reach the prompt
	assert.doesNotThrow(() => assertPrefixPreserved(captured, { ...ok, temperature: 0, stream: false, max_tokens: 5 }, 1));

	const breaks: Array<[string, Record<string, unknown>]> = [
		["thinking off (Anthropic)", { thinking: { type: "disabled" } }],
		["enable_thinking (Qwen3)", { chat_template_kwargs: { enable_thinking: false } }],
		["thinking flag (DeepSeek-V3.1 / Granite)", { chat_template_kwargs: { thinking: true } }],
		["reasoning_effort (Gemma 4, vLLM injects enable_thinking)", { reasoning_effort: "none" }],
		["effort (Pi output_config)", { output_config: { effort: "low" } }],
		["system prompt", { system: "other" }],
		["tools", { tools: [] }],
		["tool_choice", { tool_choice: { type: "none" } }],
		["model", { model: "other" }],
	];
	for (const [label, patch] of breaks) {
		assert.throws(() => assertPrefixPreserved(captured, { ...ok, ...patch }, 1), PrefixChangedError, label);
	}
	assert.throws(() => assertPrefixPreserved(captured, { ...ok, messages: [{ role: "user", content: "changed" }, ok.messages[1]] }, 1), /message 0 changed/);
	assert.throws(() => assertPrefixPreserved(captured, { ...ok, messages: [...ok.messages, { role: "user", content: "extra" }] }, 1), /message count/);
});

test("buildWarmupBody: new messages, captured system/tools/thinking, 1 token", () => {
	const captured = { system: "S", tools: [{ name: "t" }], messages: [{ role: "user", content: "old" }], output_config: { effort: "x" }, thinking: { type: "adaptive" } };
	const body = buildWarmupBody(captured, {
		messages: [{ role: "user", content: "new" }],
		system: "WRONG",
		thinking: { type: "disabled" },
		chat_template_kwargs: { enable_thinking: false },
		stream: true,
	});
	assert.equal(body.system, "S");
	assert.deepEqual(body.tools, [{ name: "t" }]);
	assert.deepEqual(body.messages, [{ role: "user", content: "new" }]);
	assert.equal(body.max_tokens, 1);
	assert.deepEqual(body.output_config, { effort: "x" }, "thinking config must not change the prefix");
	assert.deepEqual(body.thinking, { type: "adaptive" });
	assert.equal(body.chat_template_kwargs, undefined, "builder's thinking kwargs must not leak in");
});

test("requestHeaders: keeps captured routing headers, drops hop headers and non-strings, extra wins", () => {
	const h = requestHeaders("anthropic-messages", { Authorization: "Bearer k", "Content-Length": "9", Host: "x", n: 5 as any }, undefined, { "x-session-id": "s" });
	assert.deepEqual(h, { authorization: "Bearer k", "content-type": "application/json", "x-session-id": "s" });
});

test("requestHeaders: the API key is attached from resolved auth, the way the SDK does it (issue #2)", () => {
	// before_provider_headers never sees the key: Pi hands it to the SDK as `apiKey`. The
	// captured map is keyless, so the summary request must add it from the registry.
	const captured = { "x-pi-attribution": "pi", "x-session-affinity": "abc" };
	const anthropic = requestHeaders("anthropic-messages", captured, { apiKey: "secret" });
	assert.equal(anthropic["x-api-key"], "secret");
	assert.equal(anthropic.authorization, undefined);
	assert.equal(anthropic["x-pi-attribution"], "pi", "routing headers from the capture survive");

	const openai = requestHeaders("openai-completions", captured, { apiKey: "secret" });
	assert.equal(openai.authorization, "Bearer secret");
	assert.equal(openai["x-api-key"], undefined);

	// OAuth-style Anthropic tokens go as Bearer, like the SDK
	assert.equal(requestHeaders("anthropic-messages", {}, { apiKey: "sk-ant-oat01-x" }).authorization, "Bearer sk-ant-oat01-x");

	// models.json `authHeader: true` already materialized Authorization; do not add a second credential
	const withAuthHeader = requestHeaders("anthropic-messages", { authorization: "Bearer secret" }, { apiKey: "secret" });
	assert.equal(withAuthHeader["x-api-key"], undefined);
	assert.equal(withAuthHeader.authorization, "Bearer secret");

	// resolved auth headers win over stale captured values; static model headers sit underneath
	const layered = requestHeaders("openai-completions", { "x-tenant": "old" }, { headers: { "x-tenant": "new" } }, {}, { "x-static": "1", "x-tenant": "static" });
	assert.equal(layered["x-tenant"], "new");
	assert.equal(layered["x-static"], "1");

	// no key at all: nothing is invented ("local" is gone), the server decides
	const keyless = requestHeaders("openai-completions", {}, {});
	assert.equal(keyless.authorization, undefined);
	assert.equal(keyless["x-api-key"], undefined);
});

test("endpointUrl: Anthropic with and without /v1; OpenAI appends /chat/completions to the /v1 base", () => {
	assert.equal(endpointUrl("http://h:1", "anthropic-messages"), "http://h:1/v1/messages");
	assert.equal(endpointUrl("http://h:1/", "anthropic-messages"), "http://h:1/v1/messages");
	assert.equal(endpointUrl("http://h:1/v1", "anthropic-messages"), "http://h:1/v1/messages");
	assert.equal(endpointUrl("http://h:1/v1", "openai-completions"), "http://h:1/v1/chat/completions");
	assert.equal(endpointUrl("http://h:1/v1/", "openai-completions"), "http://h:1/v1/chat/completions");
});

test("isCapturable / systemText: OpenAI payloads carry the system prompt as a message", () => {
	assert.equal(systemText({ messages: [{ role: "system", content: "You are pi" }, { role: "user", content: "hi" }] }), "You are pi");
	assert.equal(systemText({ system: "S", messages: [{ role: "developer", content: [{ type: "text", text: "D" }] }] }), "S\nD");
	// Pi's summarizer on OpenAI: system message + ONE user message wrapping <conversation>
	assert.equal(
		isCapturable({
			messages: [
				{ role: "system", content: "You are a context summarization assistant." },
				{ role: "user", content: "<conversation>\nx\n</conversation>" },
			],
		}),
		false,
	);
	assert.equal(isCapturable({ messages: [{ role: "system", content: "other" }, { role: "user", content: "<conversation>\nx\n</conversation>" }] }), false);
	// a real OpenAI turn is captured; system-only payloads are not
	assert.equal(isCapturable({ messages: [{ role: "system", content: "You are pi" }, { role: "user", content: "hi" }] }), true);
	assert.equal(isCapturable({ messages: [{ role: "system", content: "You are pi" }] }), false);
});

test("buildSummaryBody (openai-completions): string-content instruction, cap in the captured field", () => {
	const captured = {
		model: "m",
		messages: [{ role: "system", content: "sys" }, { role: "user", content: "a" }, { role: "assistant", content: "b" }],
		tools: [{ type: "function", function: { name: "read" } }],
		reasoning_effort: "high",
		chat_template_kwargs: { enable_thinking: true },
		stream: true,
		stream_options: { include_usage: true },
		max_completion_tokens: 32000,
	};
	const snapshot = structuredClone(captured);
	const body = buildSummaryBody(captured, 9000, "openai-completions");
	assert.deepEqual(captured, snapshot, "captured payload must not be mutated");
	assert.deepEqual(body.messages.slice(0, 3), captured.messages);
	assert.deepEqual(body.messages[3], { role: "user", content: SUMMARY_PROMPT });
	assert.equal(body.max_completion_tokens, 9000, "uses the field Pi used for this provider");
	assert.equal(body.max_tokens, undefined, "never both spellings");
	assert.deepEqual(body.stream_options, { include_usage: true });
	assert.equal(body.reasoning_effort, "high");
	assert.deepEqual(body.chat_template_kwargs, { enable_thinking: true });

	const legacy = buildSummaryBody({ model: "m", messages: [{ role: "user", content: "a" }], max_tokens: 100 }, 9000, "openai-completions");
	assert.equal(legacy.max_tokens, 9000);
	assert.equal(legacy.max_completion_tokens, undefined);

	// OpenAI thinking spellings are prefix-affecting too
	assert.throws(() => assertPrefixPreserved(captured, { ...body, enable_thinking: false }, 1), PrefixChangedError);
	assert.throws(() => assertPrefixPreserved(captured, { ...body, response_format: { type: "json_object" } }, 1), PrefixChangedError);
});

test("buildWarmupBody (openai-completions): 1 token in the captured cap field", () => {
	const body = buildWarmupBody({ model: "m", messages: [{ role: "user", content: "old" }], max_completion_tokens: 5000 }, { messages: [{ role: "user", content: "new" }] });
	assert.equal(body.max_completion_tokens, 1);
	assert.equal(body.max_tokens, undefined);
});

const chat = (...chunks: Array<object | string>) => chunks.map((c) => `data: ${typeof c === "string" ? c : JSON.stringify(c)}\n\n`).join("");
const delta = (d: Record<string, unknown>, finish_reason: string | null = null) => ({ id: "c1", object: "chat.completion.chunk", choices: [{ index: 0, delta: d, finish_reason }] });

test("ChatCompletionsCollector: text across chunk splits, reasoning counted, usage normalized", () => {
	const raw = chat(
		delta({ role: "assistant", content: "" }),
		delta({ reasoning_content: "hmm" }),
		delta({ content: "## Goal\n" }),
		delta({ content: "ship it" }),
		delta({}, "stop"),
		{ id: "c1", choices: [], usage: { prompt_tokens: 1000, completion_tokens: 4, prompt_tokens_details: { cached_tokens: 900 } } },
		"[DONE]",
	);
	for (const size of [1, 7, 64, raw.length]) {
		const c = new ChatCompletionsCollector();
		for (let i = 0; i < raw.length; i += size) c.push(raw.slice(i, i + size));
		const r = c.finish();
		assert.equal(r.text, "## Goal\nship it");
		assert.equal(r.stopReason, "stop");
		assert.equal(c.thinkingChars, 3);
		assert.deepEqual(r.usage, { input_tokens: 100, output_tokens: 4, cache_read_input_tokens: 900, cache_creation_input_tokens: 0 });
		assert.equal(toPiUsage(r.usage).totalTokens, 1004);
	}
	assert.ok(createCollector("openai-completions") instanceof ChatCompletionsCollector);
	assert.ok(createCollector("anthropic-messages") instanceof SseCollector);
});

test("ChatCompletionsCollector: clean end without finish_reason is accepted; tool calls, length, cut-off and errors refuse", () => {
	const clean = new ChatCompletionsCollector();
	clean.push(chat(delta({ content: "## Goal" }), "[DONE]"));
	assert.equal(clean.finish().text, "## Goal");

	const tool = new ChatCompletionsCollector();
	assert.throws(() => tool.push(chat(delta({ tool_calls: [{ index: 0, id: "t", function: { name: "read", arguments: "" } }] }))), SummaryError);

	const byReason = new ChatCompletionsCollector();
	byReason.push(chat(delta({ content: "x" }, "tool_calls"), "[DONE]"));
	assert.throws(() => byReason.finish(), /tool/);

	const cap = new ChatCompletionsCollector();
	cap.push(chat(delta({ content: "partial" }, "length"), "[DONE]"));
	assert.throws(() => cap.finish(), /token cap/);

	const cut = new ChatCompletionsCollector();
	cut.push(chat(delta({ content: "## Goal\npartial" })));
	assert.throws(() => cut.finish(), /mid-summary/);

	assert.throws(() => new ChatCompletionsCollector().finish(), /empty/);

	const err = new ChatCompletionsCollector();
	assert.throws(() => err.push(chat({ error: { message: "boom", type: "server_error" } })), /stream error/);
});

test("normalizeChatUsage: DeepSeek and Kimi cache-hit spellings", () => {
	assert.equal(normalizeChatUsage({ prompt_tokens: 10, prompt_cache_hit_tokens: 8 }).cache_read_input_tokens, 8);
	assert.equal(normalizeChatUsage({ prompt_tokens: 10, cached_tokens: 3 }).input_tokens, 7);
	assert.deepEqual(normalizeChatUsage({}), { input_tokens: 0, output_tokens: 0, cache_read_input_tokens: 0, cache_creation_input_tokens: 0 });
});

const sse = (...events: object[]) => events.map((e) => `event: x\ndata: ${JSON.stringify(e)}\n\n`).join("");

test("SseCollector: text across arbitrary chunk splits, usage merged", () => {
	const raw = sse(
		{ type: "message_start", message: { usage: { input_tokens: 10 } } },
		{ type: "content_block_start", content_block: { type: "text" } },
		{ type: "content_block_delta", delta: { type: "text_delta", text: "## Goal\n" } },
		{ type: "content_block_delta", delta: { type: "text_delta", text: "ship it" } },
		{ type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { output_tokens: 4 } },
	);
	for (const size of [1, 7, 64, raw.length]) {
		const c = new SseCollector();
		for (let i = 0; i < raw.length; i += size) c.push(raw.slice(i, i + size));
		const r = c.finish();
		assert.equal(r.text, "## Goal\nship it");
		assert.equal(r.usage.output_tokens, 4);
		assert.equal(r.usage.input_tokens, 10);
	}
});

test("SseCollector: thinking deltas are progress, not summary text", () => {
	const c = new SseCollector();
	c.push(
		sse(
			{ type: "content_block_delta", delta: { type: "thinking_delta", thinking: "hmm" } },
			{ type: "content_block_delta", delta: { type: "text_delta", text: "## Goal" } },
			{ type: "message_delta", delta: { stop_reason: "end_turn" } },
		),
	);
	assert.equal(c.thinkingChars, 3);
	assert.equal(c.finish().text, "## Goal");
});

test("SseCollector: CRLF frames parse", () => {
	const c = new SseCollector();
	c.push(sse({ type: "content_block_delta", delta: { type: "text_delta", text: "ok" } }, { type: "message_delta", delta: { stop_reason: "end_turn" } }).replace(/\n/g, "\r\n"));
	assert.equal(c.finish().text, "ok");
});

test("SseCollector: a clean end without stop_reason is accepted", () => {
	// some servers finish with message_stop but never report stop_reason; requiring the
	// field would make every compaction fall back to Pi's default on such a server
	const c = new SseCollector();
	c.push(sse({ type: "content_block_delta", delta: { type: "text_delta", text: "## Goal" } }, { type: "message_stop" }));
	assert.equal(c.finish().text, "## Goal");
});

test("SseCollector: tool use, truncation, empty, incomplete and error all refuse", () => {
	const toolUse = new SseCollector();
	assert.throws(() => toolUse.push(sse({ type: "content_block_start", content_block: { type: "tool_use" } })), SummaryError);

	const cap = new SseCollector();
	cap.push(sse({ type: "content_block_delta", delta: { type: "text_delta", text: "partial" } }, { type: "message_delta", delta: { stop_reason: "max_tokens" } }));
	assert.throws(() => cap.finish(), /token cap/);

	assert.throws(() => new SseCollector().finish(), /empty/);

	// text arrived but the server closed the connection before message_delta AND without
	// the terminal message_stop: a truncated summary
	const cut = new SseCollector();
	cut.push(sse({ type: "content_block_delta", delta: { type: "text_delta", text: "## Goal\npartial" } }));
	assert.throws(() => cut.finish(), /mid-summary/);

	const err = new SseCollector();
	assert.throws(() => err.push(sse({ type: "error", error: { message: "boom" } })), /stream error/);
});

test("toPiUsage: Anthropic fields map onto Pi's normalized usage, incl. totalTokens and cost", () => {
	const zeroCost = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 };
	assert.deepEqual(
		toPiUsage({ input_tokens: 100, output_tokens: 5, cache_read_input_tokens: 900, cache_creation_input_tokens: 10 }),
		{ input: 100, output: 5, cacheRead: 900, cacheWrite: 10, totalTokens: 1015, cost: zeroCost },
	);
	assert.deepEqual(toPiUsage({}), { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: zeroCost });
	// Pi does `totals.cost += usage.cost.total` on compaction entries (usage-totals.js);
	// a missing cost object throws there, so both fields must always be present.
	const u = toPiUsage({ output_tokens: 3 });
	assert.equal(typeof u.cost.total, "number");
	assert.equal(typeof u.totalTokens, "number");
});

test("fileListSuffix: read-only vs modified, sorted, empty when none", () => {
	assert.equal(fileListSuffix(undefined), "");
	assert.equal(fileListSuffix({ read: [], edited: [], written: [] }), "");
	const s = fileListSuffix({ read: new Set(["b", "a", "c"]), edited: new Set(["c"]), written: new Set(["d"]) });
	assert.equal(s, "\n\n<read-files>\na\nb\n</read-files>\n\n<modified-files>\nc\nd\n</modified-files>");
});

// ---- end-to-end against a fake server: the exact failure from issue #2 and its fix ----
import { createServer } from "node:http";
import type { AddressInfo } from "node:net";
import { HttpStatusError, streamMessages } from "../src/core.ts";

function fakeServer(handler: (req: import("node:http").IncomingMessage, body: string, res: import("node:http").ServerResponse) => void) {
	const server = createServer((req, res) => {
		let body = "";
		req.on("data", (d) => (body += d));
		req.on("end", () => handler(req, body, res));
	});
	return new Promise<{ url: string; close: () => Promise<void> }>((resolve) =>
		server.listen(0, "127.0.0.1", () =>
			resolve({ url: `http://127.0.0.1:${(server.address() as AddressInfo).port}`, close: () => new Promise((r) => server.close(() => r())) }),
		),
	);
}

test("issue #2: a keyless captured header map gets 401; resolved auth attaches the key and the summary streams", async () => {
	const seen: Array<Record<string, string | string[] | undefined>> = [];
	const srv = await fakeServer((req, body, res) => {
		seen.push(req.headers);
		if (req.headers["x-api-key"] !== "secret") {
			res.writeHead(401, { "content-type": "application/json" });
			return res.end(JSON.stringify({ error: "unauthorized: invalid or missing API key" }));
		}
		assert.equal(req.url, "/v1/messages");
		const parsed = JSON.parse(body);
		assert.equal(parsed.messages.at(-1).content[0].text, SUMMARY_PROMPT);
		res.writeHead(200, { "content-type": "text/event-stream" });
		res.end(
			sse(
				{ type: "message_start", message: { usage: { input_tokens: 5, cache_read_input_tokens: 1000 } } },
				{ type: "content_block_delta", delta: { type: "text_delta", text: "## Goal\nfixed" } },
				{ type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { output_tokens: 3 } },
				{ type: "message_stop" },
			),
		);
	});
	try {
		// what before_provider_headers actually hands us: attribution/routing, no credential
		const captured = { "x-pi-attribution": "pi", "user-agent": "pi" };
		const body = JSON.stringify(buildSummaryBody({ model: "m", messages: [{ role: "user", content: "hi" }] }, 500));
		const signal = new AbortController().signal;

		await assert.rejects(
			streamMessages(endpointUrl(srv.url, "anthropic-messages"), requestHeaders("anthropic-messages", captured, undefined), body, new SseCollector(), signal),
			(e: Error) => e instanceof HttpStatusError && /HTTP 401 .*missing API key/.test(e.message),
		);

		const r = await streamMessages(
			endpointUrl(srv.url, "anthropic-messages"),
			requestHeaders("anthropic-messages", captured, { apiKey: "secret" }, { "x-session-id": "s1" }),
			body,
			new SseCollector(),
			signal,
		);
		assert.equal(r.text, "## Goal\nfixed");
		assert.equal(toPiUsage(r.usage).cacheRead, 1000);
		assert.equal(seen[1]["x-pi-attribution"], "pi", "captured routing headers still sent");
		assert.equal(seen[1]["x-session-id"], "s1");
	} finally {
		await srv.close();
	}
});

test("openai-completions end to end: Bearer auth, /chat/completions, chunk stream parsed", async () => {
	const srv = await fakeServer((req, body, res) => {
		if (req.headers.authorization !== "Bearer secret") {
			res.writeHead(401);
			return res.end("nope");
		}
		assert.equal(req.url, "/v1/chat/completions");
		const parsed = JSON.parse(body);
		assert.deepEqual(parsed.messages.at(-1), { role: "user", content: SUMMARY_PROMPT });
		assert.equal(parsed.max_tokens, 500);
		res.writeHead(200, { "content-type": "text/event-stream" });
		res.end(chat(delta({ content: "## Goal\nok" }, "stop"), { id: "c", choices: [], usage: { prompt_tokens: 50, completion_tokens: 2 } }, "[DONE]"));
	});
	try {
		const body = JSON.stringify(buildSummaryBody({ model: "m", messages: [{ role: "user", content: "hi" }], max_tokens: 9 }, 500, "openai-completions"));
		const r = await streamMessages(
			endpointUrl(`${srv.url}/v1`, "openai-completions"),
			requestHeaders("openai-completions", {}, { apiKey: "secret" }),
			body,
			createCollector("openai-completions"),
			new AbortController().signal,
		);
		assert.equal(r.text, "## Goal\nok");
		assert.equal(r.usage.output_tokens, 2);
	} finally {
		await srv.close();
	}
});

test("streamMessages: an already-aborted signal rejects instead of crashing the process", async () => {
	const ac = new AbortController();
	ac.abort();
	await assert.rejects(streamMessages("http://127.0.0.1:9/v1/messages", {}, "{}", new SseCollector(), ac.signal), /aborted/);
});
