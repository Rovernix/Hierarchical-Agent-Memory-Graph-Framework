/** Adapt HAMGF's node_id/source/target schema without modifying its records. */
export function toCytoscapeElements(graph) {
  const nodes = (graph?.nodes || []).map((node) => ({
    data: { ...node, id: node.node_id, label: node.summary || node.content },
  }));
  const nodeIds = new Set(nodes.map((node) => node.data.id));
  const edges = (graph?.edges || [])
    .filter((edge) => nodeIds.has(edge.source) && nodeIds.has(edge.target))
    .map((edge, index) => ({
      data: {
        ...edge,
        id: edge.edge_id || `edge:${JSON.stringify([edge.source, edge.target, edge.key ?? index])}`,
      },
    }));
  return [...nodes, ...edges];
}

/** History is optional, but every rendered edge must still have two visible ends. */
export function visibleMemoryGraph(graph, includeHistory = false) {
  const isCurrent = (item) => item.status !== "superseded" && item.status !== "archived";
  const nodes = (graph?.nodes || []).filter((node) => includeHistory || isCurrent(node));
  const nodeIds = new Set(nodes.map((node) => node.node_id));
  const edges = (graph?.edges || []).filter((edge) => nodeIds.has(edge.source) && nodeIds.has(edge.target) && (includeHistory || (isCurrent(edge) && edge.label !== "supersedes")));
  return { ...graph, nodes, edges };
}

export function memoryCategory(node) {
  return node?.metadata?.category || node?.type || "memory";
}

/** Preserve each quote exactly, including repeated evidence from separate messages. */
export function memoryEvidence(node) {
  const evidence = node?.metadata?.evidence;
  if (Array.isArray(evidence)) return evidence.filter((item) => item && typeof item.quote === "string").map((item) => ({ ...item }));
  const legacy = node?.metadata?.provenance?.evidence;
  const quotes = typeof legacy === "string" ? [legacy] : Array.isArray(legacy) ? legacy.filter((item) => typeof item === "string") : [];
  return quotes.map((quote) => ({ message_id: node?.metadata?.source_message_id || "", quote }));
}

/** An event's time is explicit metadata, never the time its memory was created. */
export function temporalDetails(node) {
  const temporal = node?.metadata?.temporal;
  if (!temporal || typeof temporal !== "object") return null;
  const raw = typeof temporal.raw === "string" ? temporal.raw : "";
  const normalized = typeof temporal.normalized === "string" ? temporal.normalized : "";
  if (!raw && !normalized) return null;
  const monthDay = normalized.match(/^--(\d{2})-(\d{2})$/);
  const compact = monthDay ? `${Number(monthDay[1])}月${Number(monthDay[2])}日` : normalized || raw;
  return {
    raw, normalized, precision: temporal.precision || "", compact,
    display: monthDay ? `${compact}（年份未说明）` : normalized || raw,
    evidence: typeof temporal.evidence === "string" ? temporal.evidence : "",
  };
}

/** Relation evidence lives on graph-view edges or on the source event metadata. */
export function relationDetails(edge, graph) {
  const stored = graph?.nodes?.find((node) => node.node_id === edge.source)?.metadata?.relations;
  const match = Array.isArray(stored) ? stored.find((relation) => relation.target_node_id === edge.target && relation.relation === edge.relation && relation.label === edge.label) : null;
  const legacyOrder = edge.label === "next conversation turn" || edge.label === "对话轮次先后";
  const kind = edge.temporal_kind || (legacyOrder ? "conversation_order" : "");
  const label = legacyOrder ? "对话轮次先后" : edge.label === "responds to" ? "回复此消息" : edge.label || ({ causal: "因果", temporal: "时序", semantic: "语义" }[edge.relation] || "关联");
  const provenance = edge.provenance && typeof edge.provenance === "object" ? edge.provenance : {};
  const evidenceText = (value) => typeof value === "string" ? value : Array.isArray(value) && value.every((part) => typeof part === "string") ? value.join("\n") : "";
  return {
    label, kind,
    evidence: evidenceText(edge.evidence) || evidenceText(Array.isArray(edge.evidence_sources) ? edge.evidence_sources.map((item) => item.quote) : null) || evidenceText(match?.evidence),
    method: provenance.method || (typeof edge.provenance === "string" ? edge.provenance : "") || match?.method || "",
    model: provenance.model || "",
  };
}

/** Follow one strongest incoming chain, preferring causal over temporal links. */
export function traceToRoot(graph, nodeId) {
  const nodeIds = new Set((graph?.nodes || []).map((node) => node.node_id));
  if (!nodeIds.has(nodeId)) return [];
  const incoming = new Map();
  for (const edge of graph?.edges || []) {
    if (!nodeIds.has(edge.source) || !nodeIds.has(edge.target)) continue;
    if (edge.status === "superseded" || edge.label === "supersedes") continue;
    const candidates = incoming.get(edge.target) || [];
    candidates.push(edge);
    incoming.set(edge.target, candidates);
  }
  const rank = { causal: 0, temporal: 1, semantic: 2 };
  const weight = (edge) => Number.isFinite(edge.weight) ? edge.weight : 0;
  const path = [nodeId];
  const seen = new Set(path);
  let current = nodeId;
  while (incoming.has(current)) {
    const candidates = incoming.get(current)
      .filter((edge) => !seen.has(edge.source))
      .sort((left, right) => (rank[left.relation] ?? 3) - (rank[right.relation] ?? 3) || weight(right) - weight(left));
    if (!candidates.length) break;
    current = candidates[0].source;
    seen.add(current);
    path.unshift(current);
  }
  return path;
}
