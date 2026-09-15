import assert from "node:assert/strict";
import { test } from "node:test";
import {
	appliesTo,
	buildSummaryBody,
	buildWarmupBody,
	DEFAULT_CONFIG,
	fileListSuffix,
	isCapturable,
	mergeConfig,
	messagesUrl,
	requestHeaders,
	SseCollector,
	SummaryError,
	SUMMARY_PROMPT,
	summaryTokenBudget,
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

test("appliesTo: anthropic-messages local servers only by default", () => {
	assert.equal(appliesTo(local, DEFAULT_CONFIG), true);
	assert.equal(appliesTo({ ...local, api: "openai-completions" }, DEFAULT_CONFIG), false);
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
	assert.equal(isCapturable(null), false);
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
	const body = buildSummaryBody(captured, 9000, DEFAULT_CONFIG);
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

	const off = buildSummaryBody(captured, 9000, mergeConfig({ disableThinking: true }));
	assert.deepEqual(off.thinking, { type: "disabled" });
	assert.equal(off.output_config, undefined);
	assert.equal(off.chat_template_kwargs.enable_thinking, false);
});

test("buildWarmupBody: new messages, captured system/tools/thinking, 1 token", () => {
	const captured = { system: "S", tools: [{ name: "t" }], messages: [{ role: "user", content: "old" }], output_config: { effort: "x" }, thinking: { type: "adaptive" } };
	const body = buildWarmupBody(captured, { messages: [{ role: "user", content: "new" }], system: "WRONG", stream: true });
	assert.equal(body.system, "S");
	assert.deepEqual(body.tools, [{ name: "t" }]);
	assert.deepEqual(body.messages, [{ role: "user", content: "new" }]);
	assert.equal(body.max_tokens, 1);
	assert.deepEqual(body.output_config, { effort: "x" }, "thinking config must not change the prefix");
	assert.deepEqual(body.thinking, { type: "adaptive" });
});

test("requestHeaders: keeps auth, drops hop headers and non-strings, extra wins", () => {
	const h = requestHeaders({ Authorization: "Bearer k", "Content-Length": "9", Host: "x", n: 5 as any }, { "x-session-id": "s" });
	assert.deepEqual(h, { authorization: "Bearer k", "content-type": "application/json", "x-session-id": "s" });
});

test("messagesUrl: with and without /v1 and trailing slash", () => {
	assert.equal(messagesUrl("http://h:1"), "http://h:1/v1/messages");
	assert.equal(messagesUrl("http://h:1/"), "http://h:1/v1/messages");
	assert.equal(messagesUrl("http://h:1/v1"), "http://h:1/v1/messages");
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
	c.push(sse({ type: "content_block_delta", delta: { type: "thinking_delta", thinking: "hmm" } }, { type: "content_block_delta", delta: { type: "text_delta", text: "## Goal" } }));
	assert.equal(c.thinkingChars, 3);
	assert.equal(c.finish().text, "## Goal");
});

test("SseCollector: CRLF frames parse", () => {
	const c = new SseCollector();
	c.push(sse({ type: "content_block_delta", delta: { type: "text_delta", text: "ok" } }).replace(/\n/g, "\r\n"));
	assert.equal(c.finish().text, "ok");
});

test("SseCollector: tool use, truncation, empty and error all refuse", () => {
	const toolUse = new SseCollector();
	assert.throws(() => toolUse.push(sse({ type: "content_block_start", content_block: { type: "tool_use" } })), SummaryError);

	const cap = new SseCollector();
	cap.push(sse({ type: "content_block_delta", delta: { type: "text_delta", text: "partial" } }, { type: "message_delta", delta: { stop_reason: "max_tokens" } }));
	assert.throws(() => cap.finish(), /token cap/);

	assert.throws(() => new SseCollector().finish(), /empty/);

	const err = new SseCollector();
	assert.throws(() => err.push(sse({ type: "error", error: { message: "boom" } })), /stream error/);
});

test("fileListSuffix: read-only vs modified, sorted, empty when none", () => {
	assert.equal(fileListSuffix(undefined), "");
	assert.equal(fileListSuffix({ read: [], edited: [], written: [] }), "");
	const s = fileListSuffix({ read: new Set(["b", "a", "c"]), edited: new Set(["c"]), written: new Set(["d"]) });
	assert.equal(s, "\n\n<read-files>\na\nb\n</read-files>\n\n<modified-files>\nc\nd\n</modified-files>");
});
