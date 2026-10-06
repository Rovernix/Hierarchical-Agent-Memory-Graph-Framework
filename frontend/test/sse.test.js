import test from "node:test";
import assert from "node:assert/strict";
import { consumeEvents } from "../src/api.js";

test("SSE preserves Chinese tokens across byte boundaries and split CRLF frames", async () => {
  const bytes = new TextEncoder().encode(': heartbeat\r\n\r\nevent: token\r\ndata: {"delta":"你好，记忆"}\r\n\r\nevent: done\ndata: {"ok":true}\n\n');
  const body = new ReadableStream({ start(controller) { for (const byte of bytes) controller.enqueue(Uint8Array.of(byte)); controller.close(); } });
  const events = [];
  await consumeEvents(body, (kind, data) => events.push([kind, data]));
  assert.deepEqual(events, [["token", { delta: "你好，记忆" }], ["done", { ok: true }]]);
});

test("SSE consumes trailing frame without separator and cancels malformed streams", async () => {
  const events = [];
  await consumeEvents(new Response('event: graph\ndata: {"nodes":[]}').body, (kind, data) => events.push([kind, data]));
  assert.deepEqual(events, [["graph", { nodes: [] }]]);
  let cancelled = false;
  const broken = new ReadableStream({ start(controller) { controller.enqueue(new TextEncoder().encode('event: token\ndata: nope\n\n')); }, cancel() { cancelled = true; } });
  await assert.rejects(consumeEvents(broken, () => {}), SyntaxError);
  assert.equal(cancelled, true);
});
