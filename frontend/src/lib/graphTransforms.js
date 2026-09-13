export function toCytoscapeElements(graph, hiddenNodeIds = new Set()) {
  const nodes = (graph?.nodes || [])
    .filter((node) => !hiddenNodeIds.has(node.node_id))
    .map((node) => ({
      data: {
        ...node,
        id: node.node_id,
        label: node.summary || node.content,
      },
    }));
  const visibleIds = new Set(nodes.map((node) => node.data.id));
  const edges = (graph?.edges || [])
    .filter((edge) => visibleIds.has(edge.source) && visibleIds.has(edge.target))
    .map((edge, index) => ({
      data: {
        ...edge,
        id: edge.edge_id || `${edge.source}::${edge.target}::${edge.key ?? index}`,
      },
    }));
  return [...nodes, ...edges];
}

export function traceToRoot(graph, nodeId) {
  const incoming = new Map();
  for (const edge of graph?.edges || []) {
    if (edge.status === "superseded" || edge.label === "supersedes") continue;
    const list = incoming.get(edge.target) || [];
    list.push(edge);
    incoming.set(edge.target, list);
  }
  const path = [nodeId];
  const seen = new Set(path);
  let current = nodeId;
  while (incoming.has(current)) {
    const candidates = incoming
      .get(current)
      .filter((edge) => !seen.has(edge.source))
      .sort((left, right) => relationRank(left.relation) - relationRank(right.relation) || right.weight - left.weight);
    if (!candidates.length) break;
    current = candidates[0].source;
    seen.add(current);
    path.unshift(current);
  }
  return path;
}

export function directBranches(graph, anchorId) {
  const active = (graph?.edges || []).filter(
    (edge) => edge.source === anchorId && edge.status !== "superseded" && edge.label !== "supersedes",
  );
  return active.length > 1 ? active.map((edge) => edge.target) : [];
}

export function branchDescendants(graph, branchRootId) {
  const outgoing = new Map();
  for (const edge of graph?.edges || []) {
    if (edge.status === "superseded" || edge.label === "supersedes") continue;
    const list = outgoing.get(edge.source) || [];
    list.push(edge.target);
    outgoing.set(edge.source, list);
  }
  const descendants = new Set([branchRootId]);
  const queue = [branchRootId];
  while (queue.length) {
    const current = queue.shift();
    for (const next of outgoing.get(current) || []) {
      if (!descendants.has(next)) {
        descendants.add(next);
        queue.push(next);
      }
    }
  }
  return descendants;
}

export function findSupersessionPair(graph, nodeId) {
  const link = (graph?.edges || []).find(
    (edge) => edge.label === "supersedes" && (edge.source === nodeId || edge.target === nodeId),
  );
  if (!link) return null;
  const nodes = new Map((graph?.nodes || []).map((node) => [node.node_id, node]));
  return { replacement: nodes.get(link.source), original: nodes.get(link.target), edge: link };
}

export function graphStats(graph) {
  const nodes = graph?.nodes || [];
  const edges = graph?.edges || [];
  const outgoingCounts = new Map();
  for (const edge of edges) {
    if (edge.status === "superseded" || edge.label === "supersedes") continue;
    outgoingCounts.set(edge.source, (outgoingCounts.get(edge.source) || 0) + 1);
  }
  return {
    nodes: nodes.length,
    edges: edges.length,
    branches: [...outgoingCounts.values()].filter((count) => count > 1).length,
    pending: nodes.filter((node) => node.status === "pending_verification").length + edges.filter((edge) => edge.status === "pending_verification").length,
  };
}

function relationRank(relation) {
  return { causal: 0, temporal: 1, semantic: 2 }[relation] ?? 3;
}
