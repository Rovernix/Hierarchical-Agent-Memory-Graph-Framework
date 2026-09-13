import test from "node:test";
import assert from "node:assert/strict";

import { DEMO_GRAPH } from "../src/lib/demoGraph.js";
import {
  branchDescendants,
  directBranches,
  findSupersessionPair,
  graphStats,
  toCytoscapeElements,
  traceToRoot,
} from "../src/lib/graphTransforms.js";

test("converts the API contract into unique Cytoscape elements", () => {
  const elements = toCytoscapeElements(DEMO_GRAPH);
  assert.equal(elements.length, DEMO_GRAPH.nodes.length + DEMO_GRAPH.edges.length);
  assert.equal(new Set(elements.map((element) => element.data.id)).size, elements.length);
});

test("traces causes, detects branches, and collapses descendants", () => {
  assert.deepEqual(directBranches(DEMO_GRAPH, "M-DEMO-002"), ["M-DEMO-A", "M-DEMO-B"]);
  assert.deepEqual(traceToRoot(DEMO_GRAPH, "M-DEMO-B2"), ["M-DEMO-001", "M-DEMO-002", "M-DEMO-B", "M-DEMO-FB", "M-DEMO-B2"]);
  assert.deepEqual([...branchDescendants(DEMO_GRAPH, "M-DEMO-B")], ["M-DEMO-B", "M-DEMO-FB", "M-DEMO-B2", "M-DEMO-DONE"]);
});

test("builds supersession comparison and aggregate stats", () => {
  const pair = findSupersessionPair(DEMO_GRAPH, "M-DEMO-B");
  assert.equal(pair.original.node_id, "M-DEMO-B");
  assert.equal(pair.replacement.node_id, "M-DEMO-B2");
  assert.deepEqual(graphStats(DEMO_GRAPH), { nodes: 7, edges: 7, branches: 1, pending: 0 });
});
