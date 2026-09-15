#!/usr/bin/env node
/**
 * Live smoke test against a real server. Builds a ~70k-token session with Pi in RPC mode,
 * runs /compact through this extension, then times the first turn after compaction.
 *
 * Isolated: uses a temporary PI_CODING_AGENT_DIR containing only your models.json (and
 * auth.json if present) plus this extension, so other installed extensions do not interfere.
 *
 *   node scripts/rpc-smoke.mjs --provider qwen-local --model qwen3.8-27b [--no-warmup]
 *
 * Exit code 0 only if the compaction came from this extension.
 */
import { spawn } from "node:child_process";
import { copyFileSync, existsSync, mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const args = process.argv.slice(2);
const opt = (name, def) => {
	const i = args.indexOf(`--${name}`);
	return i >= 0 ? args[i + 1] : def;
};
const provider = opt("provider");
const model = opt("model");
const warmup = !args.includes("--no-warmup");
if (!provider || !model) {
	console.error("usage: rpc-smoke.mjs --provider <id> --model <id> [--no-warmup] [--extra-extension <path>]");
	process.exit(2);
}

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const srcAgent = process.env.PI_CODING_AGENT_DIR ?? join(homedir(), ".pi", "agent");
const agent = mkdtempSync(join(tmpdir(), "ppcc-agent-"));
for (const f of ["models.json", "auth.json"]) if (existsSync(join(srcAgent, f))) copyFileSync(join(srcAgent, f), join(agent, f));
writeFileSync(join(agent, "settings.json"), JSON.stringify({ defaultProvider: provider, defaultModel: model }));
writeFileSync(join(agent, "pi-prefix-cache-compaction.json"), JSON.stringify({ warmup }));

const work = mkdtempSync(join(tmpdir(), "ppcc-work-"));
const line = (i) => `record ${i}: service=svc${i % 17} owner=team${i % 5} status=${["ok", "degraded", "down"][i % 3]} note=lorem ipsum dolor sit amet consectetur`;
for (const name of ["a.txt", "b.txt", "c.txt"]) {
	const rows = Array.from({ length: 700 }, (_, i) => line(i));
	if (name === "b.txt") rows.splice(350, 0, "record SPECIAL: beta=release-2026-09 deploy=blue");
	writeFileSync(join(work, name), rows.join("\n"));
}

const piArgs = ["--mode", "rpc", "--provider", provider, "--model", model, "--thinking", "off", "--no-session", "-e", join(root, "src", "index.ts")];
const extra = opt("extra-extension");
if (extra) piArgs.push("-e", resolve(extra));
const pi = spawn(process.env.PI_BIN ?? "pi", piArgs, { cwd: work, env: { ...process.env, PI_CODING_AGENT_DIR: agent } });

const t0 = Date.now();
const el = () => `${((Date.now() - t0) / 1000).toFixed(1)}s`;
let buf = "";
let waiters = [];
const seen = [];
pi.stdout.on("data", (d) => {
	buf += d;
	let i;
	while ((i = buf.indexOf("\n")) >= 0) {
		const raw = buf.slice(0, i);
		buf = buf.slice(i + 1);
		let ev;
		try {
			ev = JSON.parse(raw);
		} catch {
			continue;
		}
		seen.push(ev);
		if (ev.type === "extension_ui_request" && ev.method === "notify") console.log(el(), "notify:", ev.message);
		if (ev.type === "extension_error") console.log(el(), "EXTENSION ERROR:", JSON.stringify(ev).slice(0, 300));
		waiters = waiters.filter((w) => !w(ev));
	}
});
pi.stderr.on("data", (d) => process.stderr.write(d));
const send = (o) => pi.stdin.write(`${JSON.stringify(o)}\n`);
const until = (pred, ms = 600_000) =>
	new Promise((res, rej) => {
		const timer = setTimeout(() => rej(new Error("timeout")), ms);
		waiters.push((ev) => pred(ev) && (clearTimeout(timer), res(ev), true));
	});
const turn = async (message) => {
	const s = Date.now();
	send({ type: "prompt", message });
	await until((e) => e.type === "agent_end");
	return (Date.now() - s) / 1000;
};

try {
	console.log(el(), "turn 1:", (await turn("Read a.txt, b.txt and c.txt fully with your read tool (three reads) and tell me the SPECIAL record. Be brief.")).toFixed(1), "s");
	console.log(el(), "turn 2:", (await turn("Remember: the deploy colour is blue. Reply OK.")).toFixed(1), "s");
	const cs = Date.now();
	send({ id: "c1", type: "compact" });
	const r = await until((e) => e.type === "response" && e.id === "c1");
	const compactS = (Date.now() - cs) / 1000;
	const end = seen.find((e) => e.type === "compaction_end");
	console.log(el(), `compact: ${compactS.toFixed(1)}s success=${r.success}`, r.error ?? "");
	const warmed = (e) => e.type === "extension_ui_request" && /re-warmed/.test(e.message ?? "");
	if (warmup && !seen.some(warmed)) await until(warmed, 180_000).catch(() => console.log(el(), "no warm-up notice"));
	const after = await turn("What is the SPECIAL record and the deploy colour? One line.");
	const last = [...seen].reverse().find((e) => e.type === "agent_end");
	const answer = JSON.stringify(last?.messages?.at(-1)?.content ?? "").slice(0, 200);
	console.log(el(), `first turn after compaction: ${after.toFixed(1)}s  answer: ${answer}`);
	const fromExt = seen.some((e) => e.type === "extension_ui_request" && /Compaction done/.test(e.message ?? ""));
	console.log(JSON.stringify({ warmup, compactSeconds: compactS, firstTurnAfterSeconds: after, fromExtension: fromExt, summaryChars: end?.result?.summary?.length }));
	pi.kill();
	process.exit(fromExt && r.success ? 0 : 1);
} catch (err) {
	console.error("smoke failed:", err.message);
	pi.kill();
	process.exit(1);
}
