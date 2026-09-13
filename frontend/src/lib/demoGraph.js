const node = (node_id, summary, type, status = "active", pool = "working", credibility = 0.9) => ({
  node_id,
  summary,
  content: summary,
  type,
  status,
  pool,
  importance: 0.9,
  timeliness: 0.8,
  credibility,
  credibility_source: "user_confirmed",
  created_at: "2026-09-02T09:00:00+00:00",
  last_reaffirmed_at: null,
  decay_lambda: 0.01,
  embedding: null,
  metadata: {},
  credibility_history: [],
});

const edge = (source, target, relation, label, key, status = "active") => ({
  source,
  target,
  relation,
  label,
  key,
  edge_id: `${source}::${target}::${key}`,
  weight: 0.9,
  status,
  created_at: "2026-09-02T09:00:00+00:00",
});

export const DEMO_GRAPH = {
  revision: 7,
  schema_version: "1.0",
  nodes: [
    node("M-DEMO-001", "初始需求", "event"),
    node("M-DEMO-002", "方案讨论", "decision"),
    node("M-DEMO-A", "方案 A", "decision"),
    node("M-DEMO-B", "方案 B（已推翻）", "decision", "superseded", "working", 0.72),
    node("M-DEMO-FB", "客户反馈 B", "feedback"),
    node("M-DEMO-B2", "修订方案 B2", "decision"),
    node("M-DEMO-DONE", "最终确认", "feedback"),
  ],
  edges: [
    edge("M-DEMO-001", "M-DEMO-002", "causal", "触发讨论", 0),
    edge("M-DEMO-002", "M-DEMO-A", "causal", "分叉自方案讨论", 0),
    edge("M-DEMO-002", "M-DEMO-B", "causal", "分叉自方案讨论", 1),
    edge("M-DEMO-B", "M-DEMO-FB", "causal", "收到反馈", 0),
    edge("M-DEMO-FB", "M-DEMO-B2", "causal", "反馈导致修订", 0),
    edge("M-DEMO-B2", "M-DEMO-B", "semantic", "supersedes", 1),
    edge("M-DEMO-B2", "M-DEMO-DONE", "temporal", "最终确认", 0),
  ],
};
