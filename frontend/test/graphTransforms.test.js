import assert from "node:assert/strict";
import test from "node:test";
import { memoryCategory, memoryEvidence, relationDetails, temporalDetails, toCytoscapeElements, traceToRoot, visibleMemoryGraph } from "../src/graphTransforms.js";

const node = (id) => ({ node_id: id, summary: `Summary ${id}`, content: `Content ${id}`, type: "event", pool: "working" });
const edge = (source, target, relation = "causal", weight = 0.5, extra = {}) => ({ source, target, relation, weight, label: "follows", status: "active", ...extra });

test("graph adaptation preserves HAMGF metadata and excludes dangling edges", () => {
  const first = Object.freeze({ ...node("M-a"), credibility_source: "agent_inferred", importance: 0.8 });
  const graph = Object.freeze({
    nodes: Object.freeze([first, Object.freeze(node("M-b"))]),
    edges: Object.freeze([Object.freeze(edge("M-a", "M-b", "causal", 0.8, { edge_id: "E-a" })), Object.freeze(edge("M-missing", "M-b"))]),
  });
  const elements = toCytoscapeElements(graph);
  assert.equal(elements.length, 3);
  assert.deepEqual(elements[0].data, { ...first, id: "M-a", label: first.summary });
  assert.equal(elements[2].data.id, "E-a");
  assert.equal(elements[2].data.source, "M-a");
  assert.equal(elements[2].data.weight, 0.8);
  assert.equal(first.id, undefined);
});

test("parallel relationships retain distinct, stable graph identifiers", () => {
  const graph = { nodes: [node("M-a"), node("M-b")], edges: [edge("M-a", "M-b", "causal", 0.5, { key: 0 }), edge("M-a", "M-b", "temporal", 0.5, { key: 1 })] };
  const ids = toCytoscapeElements(graph).slice(2).map((item) => item.data.id);
  assert.equal(new Set(ids).size, 2);
  assert.deepEqual(toCytoscapeElements({ ...graph, edges: [...graph.edges].reverse() }).slice(2).map((item) => item.data.id).reverse(), ids);
});

test("trace prefers causal relationships, then their strongest weight", () => {
  const graph = { nodes: ["M-a", "M-b", "M-c", "M-d"].map(node), edges: [edge("M-a", "M-d", "temporal", 1), edge("M-b", "M-d", "causal", 0.3), edge("M-c", "M-d", "causal", 0.8)] };
  assert.deepEqual(traceToRoot(graph, "M-d"), ["M-c", "M-d"]);
});

test("trace terminates on a cycle without repeating any node", () => {
  const graph = { nodes: ["M-a", "M-b", "M-c"].map(node), edges: [edge("M-a", "M-b"), edge("M-b", "M-c"), edge("M-c", "M-a")] };
  assert.deepEqual(traceToRoot(graph, "M-c"), ["M-a", "M-b", "M-c"]);
});

test("trace excludes replaced relationships, supersession pointers, and dangling endpoints", () => {
  const graph = { nodes: ["M-a", "M-b", "M-c", "M-d"].map(node), edges: [edge("M-a", "M-d", "causal", 1, { status: "superseded" }), edge("M-b", "M-d", "causal", 1, { label: "supersedes" }), edge("M-missing", "M-d", "causal", 1), edge("M-c", "M-d", "temporal", 0.5)] };
  assert.deepEqual(traceToRoot(graph, "M-d"), ["M-c", "M-d"]);
  assert.deepEqual(traceToRoot(graph, "M-missing"), []);
  assert.deepEqual(traceToRoot(null, "M-a"), []);
  assert.deepEqual(toCytoscapeElements(null), []);
});

test("explicit event dates and their exact source evidence survive graph adaptation", () => {
  const evidence = " 10月8日提交论文初稿，10月12日进行答辩。";
  const metadata = Object.freeze({
    kind: "event_memory",
    temporal: Object.freeze({ raw: "10月8日", normalized: "--10-08", precision: "month-day", evidence }),
    provenance: Object.freeze({ method: "explicit_date_parser", evidence }),
    relations: Object.freeze([{ target_node_id: "M-b", relation: "temporal", label: "事件时间先于（--10-08 → --10-12）", evidence, method: "explicit_date_parser" }]),
  });
  const first = { ...node("M-a"), created_at: "2026-10-06T12:00:00Z", metadata };
  const [element] = toCytoscapeElements({ nodes: [first], edges: [] });
  assert.deepEqual(element.data.metadata, metadata);
  const time = temporalDetails(element.data);
  assert.equal(time.normalized, "--10-08");
  assert.equal(time.compact, "10月8日");
  assert.equal(time.display, "10月8日（年份未说明）");
  assert.equal(time.evidence, evidence);
  assert.ok(!time.display.includes("2026"));
});

test("memory creation dates never become inferred event dates", () => {
  assert.equal(temporalDetails({ ...node("M-old"), created_at: "2026-10-06T12:00:00Z" }), null);
  assert.equal(temporalDetails({ ...node("M-empty"), metadata: { temporal: {} } }), null);
  assert.equal(temporalDetails({ metadata: { temporal: { raw: "下周", normalized: "", precision: "time-text" } } }).display, "下周");
});

test("event ordering and historical message ordering stay distinct with real evidence", () => {
  const eventEdge = { ...edge("M-a", "M-b", "temporal"), label: "事件时间先于（--10-08 → --10-12）", temporal_kind: "event_time", evidence: "10月8日提交，10月12日答辩", provenance: { method: "explicit_date_parser" } };
  const event = relationDetails(eventEdge);
  assert.equal(event.kind, "event_time");
  assert.equal(event.evidence, eventEdge.evidence);
  const legacy = edge("M-a", "M-b", "temporal", 1, { label: "next conversation turn" });
  assert.deepEqual(relationDetails(legacy), { label: "对话轮次先后", kind: "conversation_order", evidence: "", method: "", model: "" });
  assert.equal(legacy.label, "next conversation turn");
});

test("relation evidence comes only from a matching stored relation", () => {
  const relation = edge("M-a", "M-b", "semantic", 0.7, { label: "共同涉及论文" });
  const graph = { nodes: [{ ...node("M-a"), metadata: { provenance: { evidence: "unrelated node evidence" }, relations: [{ target_node_id: "M-b", relation: "semantic", label: "共同涉及论文", evidence: "提交论文初稿和修改论文", method: "llm_extraction" }] } }, node("M-b")] };
  assert.equal(relationDetails(relation, graph).evidence, "提交论文初稿和修改论文");
  assert.equal(relationDetails({ ...relation, label: "another relationship" }, graph).evidence, "");
});

test("calendar ordering displays both verbatim evidence strings without changing their stored array", () => {
  const evidence = Object.freeze([" 10月8日提交论文初稿。", "10月12日进行答辩。 "]);
  const relation = Object.freeze({ ...edge("M-a", "M-b", "temporal"), label: "事件时间先于（--10-08 → --10-12）", temporal_kind: "event_time", evidence, provenance: { method: "calendar_order" } });
  assert.equal(relationDetails(relation).evidence, " 10月8日提交论文初稿。\n10月12日进行答辩。 ");
  assert.equal(relationDetails(relation).method, "calendar_order");
  const graph = { nodes: [node("M-a"), node("M-b")], edges: [relation] };
  assert.deepEqual(toCytoscapeElements(graph)[2].data.evidence, evidence);
  assert.equal(Array.isArray(relation.evidence), true);
  const { evidence: _evidence, provenance: _provenance, ...withoutDecoration } = relation;
  const storedGraph = { nodes: [{ ...node("M-a"), metadata: { relations: [{ target_node_id: "M-b", relation: "temporal", label: relation.label, method: "calendar_order", evidence }] } }, node("M-b")] };
  assert.equal(relationDetails(withoutDecoration, storedGraph).evidence, evidence.join("\n"));
});

test("current view hides old versions and their edges without inventing links between remaining memories", () => {
  const graph = { nodes: [{ ...node("M-fact"), status: "active" }, { ...node("M-old"), status: "superseded" }, { ...node("M-archived"), status: "archived" }, { ...node("M-preference"), status: "active" }], edges: [edge("M-fact", "M-old"), edge("M-old", "M-preference"), edge("M-archived", "M-fact"), edge("M-fact", "M-preference", "semantic", 1, { status: "superseded" })] };
  const current = visibleMemoryGraph(graph);
  assert.deepEqual(current.nodes.map((item) => item.node_id), ["M-fact", "M-preference"]);
  assert.deepEqual(current.edges, []);
  assert.equal(toCytoscapeElements(current).length, 2);
  assert.equal(visibleMemoryGraph(graph, true).nodes.length, 4);
  assert.equal(visibleMemoryGraph(graph, true).edges.length, 4);
  assert.equal(graph.nodes.length, 4);
});

test("logical categories and multiple source quotes preserve real evidence, independent of core node type", () => {
  const evidence = Object.freeze([Object.freeze({ message_id: "msg-1", quote: " 我偏好中文回答。" }), Object.freeze({ message_id: "msg-2", quote: "我还是更喜欢中文。 " })]);
  const metadata = Object.freeze({ kind: "logical_memory", category: "preference", logical_key: "response-language", evidence, source_message_ids: ["msg-1", "msg-2"], revisions: [{ summary: "之前偏好英文", content: "我偏好英文回答。", evidence: [{ message_id: "msg-old", quote: "我偏好英文回答。" }], reason: "用户更正" }] });
  const memory = Object.freeze({ ...node("M-pref"), type: "state", metadata });
  assert.equal(memoryCategory(memory), "preference");
  assert.deepEqual(memoryEvidence(memory), evidence);
  const adapted = toCytoscapeElements({ nodes: [memory], edges: [] });
  assert.deepEqual(adapted[0].data.metadata, metadata);
  assert.equal(adapted[0].data.metadata.revisions.length, 1);
  assert.deepEqual(memoryEvidence(node("M-no-evidence")), []);
  assert.equal(visibleMemoryGraph({ nodes: [memory], edges: [] }).nodes.length, 1);
});

test("history view still discards dangling edges and relation-source quotes stay verbatim", () => {
  const related = { ...edge("M-a", "M-b", "semantic"), evidence_sources: [{ message_id: "msg-1", quote: " 项目是 HAMGF。" }, { message_id: "msg-2", quote: "HAMGF 的研究方向是记忆。 " }] };
  const graph = { nodes: [node("M-a"), node("M-b")], edges: [related, edge("M-missing", "M-a")] };
  assert.equal(visibleMemoryGraph(graph, true).edges.length, 1);
  assert.equal(relationDetails(related).evidence, " 项目是 HAMGF。\nHAMGF 的研究方向是记忆。 ");
  assert.deepEqual(visibleMemoryGraph(null), { nodes: [], edges: [] });
});
