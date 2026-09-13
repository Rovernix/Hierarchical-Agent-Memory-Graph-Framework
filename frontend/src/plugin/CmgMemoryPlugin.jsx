import cytoscape from "cytoscape";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { createHamgfApi } from "../api/client.js";
import {
  branchDescendants,
  directBranches,
  findSupersessionPair,
  graphStats,
  toCytoscapeElements,
  traceToRoot,
} from "../lib/graphTransforms.js";

const STATUS_NAMES = {
  active: "活跃",
  superseded: "已推翻",
  archived: "已归档",
  pending_verification: "待验证",
};

const TYPE_NAMES = { event: "事件", decision: "决策", state: "状态", feedback: "反馈" };

export function CmgMemoryPlugin({
  baseUrl = "http://127.0.0.1:8000",
  fallbackGraph = null,
  refreshMs = 2000,
  onNodeSelect,
  className = "",
}) {
  const api = useMemo(() => createHamgfApi(baseUrl), [baseUrl]);
  const containerRef = useRef(null);
  const cyRef = useRef(null);
  const [graph, setGraph] = useState(fallbackGraph);
  const [connection, setConnection] = useState(fallbackGraph ? "demo" : "connecting");
  const [error, setError] = useState("");
  const [selectedId, setSelectedId] = useState(null);
  const [traceIds, setTraceIds] = useState(new Set());
  const [searchIds, setSearchIds] = useState(new Set());
  const [collapsedBranches, setCollapsedBranches] = useState(new Set());
  const [showSuperseded, setShowSuperseded] = useState(true);
  const [query, setQuery] = useState("");
  const [busy, setBusy] = useState(false);

  const loadGraph = useCallback(async () => {
    try {
      const fresh = await api.graph();
      setGraph(fresh);
      setConnection("live");
      setError("");
      return fresh;
    } catch (caught) {
      setError(caught.message);
      setConnection((current) => (fallbackGraph && current !== "live" ? "demo" : "offline"));
      if (fallbackGraph) setGraph((current) => current || fallbackGraph);
      return null;
    }
  }, [api, fallbackGraph]);

  useEffect(() => {
    let active = true;
    loadGraph();
    const timer = window.setInterval(async () => {
      if (!active || !graph) return;
      try {
        const updates = await api.events(graph.revision || 0);
        if (active && updates.revision !== graph.revision) await loadGraph();
      } catch {
        if (active) setConnection((current) => (current === "demo" ? current : "offline"));
      }
    }, Math.max(800, refreshMs));
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [api, graph?.revision, loadGraph, refreshMs]);

  const selected = useMemo(
    () => graph?.nodes?.find((node) => node.node_id === selectedId) || null,
    [graph, selectedId],
  );
  const branches = useMemo(() => (selected ? directBranches(graph, selected.node_id) : []), [graph, selected]);
  const comparison = useMemo(
    () => (selected ? findSupersessionPair(graph, selected.node_id) : null),
    [graph, selected],
  );
  const selectedEdges = useMemo(
    () => (graph?.edges || []).filter((edge) => selectedId && [edge.source, edge.target].includes(selectedId)),
    [graph, selectedId],
  );
  const stats = useMemo(() => graphStats(graph), [graph]);

  const hiddenNodeIds = useMemo(() => {
    const hidden = new Set();
    for (const branchId of collapsedBranches) {
      for (const nodeId of branchDescendants(graph, branchId)) hidden.add(nodeId);
    }
    if (!showSuperseded) {
      for (const node of graph?.nodes || []) {
        if (node.status === "superseded") hidden.add(node.node_id);
      }
    }
    return hidden;
  }, [collapsedBranches, graph, showSuperseded]);

  useEffect(() => {
    if (!containerRef.current || !graph) return undefined;
    cyRef.current?.destroy();
    const elements = toCytoscapeElements(graph, hiddenNodeIds).map((element) => {
      const id = element.data.id;
      const isNode = Boolean(element.data.node_id);
      const inTrace = isNode
        ? traceIds.has(id)
        : traceIds.has(element.data.source) && traceIds.has(element.data.target);
      const inSearch = isNode && searchIds.has(id);
      return { ...element, classes: [inTrace ? "trace" : "", inSearch ? "search-hit" : ""].filter(Boolean).join(" ") };
    });
    const cy = cytoscape({
      container: containerRef.current,
      elements,
      minZoom: 0.25,
      maxZoom: 2.2,
      wheelSensitivity: 0.45,
      layout: {
        name: "breadthfirst",
        directed: true,
        padding: 38,
        spacingFactor: 1.35,
        avoidOverlap: true,
        maximal: false,
        transform: (_node, position) => ({ x: position.y, y: position.x }),
      },
      style: [
        {
          selector: "node",
          style: {
            shape: "round-rectangle",
            width: 142,
            height: 54,
            padding: 8,
            "background-color": "#1b1b1b",
            "border-width": 1.5,
            "border-color": "#666666",
            label: "data(label)",
            color: "#eeeeee",
            "font-size": 11,
            "font-weight": 600,
            "text-wrap": "ellipsis",
            "text-max-width": 126,
            "text-valign": "center",
            "text-halign": "center",
            "overlay-opacity": 0,
          },
        },
        { selector: 'node[pool = "working"]', style: { "border-color": "#e5e5e5" } },
        { selector: 'node[pool = "episodic"]', style: { "border-color": "#a8a8a8" } },
        { selector: 'node[pool = "archive"]', style: { "border-color": "#555555", "background-color": "#121212" } },
        { selector: 'node[status = "pending_verification"]', style: { "border-color": "#f2f2f2", "border-style": "dashed", "background-color": "#1f1f1f" } },
        { selector: 'node[status = "superseded"]', style: { "border-color": "#707070", "background-color": "#171717", opacity: 0.68 } },
        { selector: 'node[status = "archived"]', style: { opacity: 0.48 } },
        {
          selector: "edge",
          style: {
            width: 2,
            "curve-style": "bezier",
            "line-color": "#777777",
            "target-arrow-color": "#777777",
            "target-arrow-shape": "triangle",
            "arrow-scale": 0.8,
            label: "data(label)",
            color: "#999999",
            "font-size": 8,
            "text-background-color": "#101010",
            "text-background-opacity": 0.82,
            "text-background-padding": 3,
            "text-rotation": "autorotate",
          },
        },
        { selector: 'edge[relation = "causal"]', style: { "line-style": "solid", "line-color": "#e5e5e5", "target-arrow-color": "#e5e5e5" } },
        { selector: 'edge[relation = "temporal"]', style: { "line-style": "dashed", "line-color": "#c8c8c8", "target-arrow-color": "#c8c8c8" } },
        { selector: 'edge[relation = "semantic"]', style: { "line-style": "dotted", "line-color": "#686868", "target-arrow-color": "#686868" } },
        { selector: 'edge[status = "pending_verification"]', style: { "line-color": "#f2f2f2", "target-arrow-color": "#f2f2f2" } },
        { selector: 'edge[status = "superseded"]', style: { opacity: 0.3 } },
        { selector: ".trace", style: { "border-width": 3, "border-color": "#c8c8c8", "line-color": "#c8c8c8", "target-arrow-color": "#c8c8c8", opacity: 1, "z-index": 20 } },
        { selector: ".search-hit", style: { "underlay-color": "#f2f2f2", "underlay-opacity": 0.24, "underlay-padding": 10 } },
        { selector: ":selected", style: { "border-width": 3, "border-color": "#ffffff" } },
      ],
    });
    cy.ready(() => {
      const renderedCenter = {
        x: containerRef.current.clientWidth / 2,
        y: containerRef.current.clientHeight / 2,
      };
      cy.zoom({ level: Math.min(cy.maxZoom(), cy.zoom() * 1.22), renderedPosition: renderedCenter });
    });
    cy.on("tap", "node", (event) => {
      const nodeId = event.target.id();
      setSelectedId(nodeId);
      onNodeSelect?.(event.target.data());
    });
    cyRef.current = cy;
    return () => cy.destroy();
  }, [graph, hiddenNodeIds, onNodeSelect, searchIds, traceIds]);

  function toggleBranch(branchId) {
    setCollapsedBranches((current) => {
      const next = new Set(current);
      if (next.has(branchId)) next.delete(branchId);
      else next.add(branchId);
      return next;
    });
  }

  function traceSelected() {
    if (!selected) return;
    setTraceIds(new Set(traceToRoot(graph, selected.node_id)));
  }

  async function runSearch(event) {
    event.preventDefault();
    if (!query.trim()) return;
    setBusy(true);
    try {
      if (connection === "live") {
        const result = await api.search(query.trim());
        setSearchIds(new Set(result.node_ids));
        if (result.entry_node_id) setSelectedId(result.entry_node_id);
      } else {
        const lowered = query.trim().toLocaleLowerCase();
        const local = (graph?.nodes || [])
          .filter((node) => `${node.summary} ${node.content}`.toLocaleLowerCase().includes(lowered))
          .map((node) => node.node_id);
        setSearchIds(new Set(local));
        if (local[0]) setSelectedId(local[0]);
      }
    } catch (caught) {
      setError(caught.message);
    } finally {
      setBusy(false);
    }
  }

  async function confirmEdge(edge) {
    if (connection !== "live") return;
    setBusy(true);
    try {
      await api.updateEdge({
        source: edge.source,
        target: edge.target,
        key: edge.key,
        status: "active",
      });
      await loadGraph();
    } catch (caught) {
      setError(caught.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className={`cmg-plugin ${className}`.trim()}>
      <div className="command-row">
        <form className="memory-search" onSubmit={runSearch}>
          <span aria-hidden="true">⌕</span>
          <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="检索记忆链、决策或客户反馈…" aria-label="检索记忆链" />
          <button type="submit" disabled={busy || !query.trim()}>定位链路</button>
        </form>
        <button className={`connection ${connection}`} onClick={loadGraph} title={error || "刷新图数据"}>
          <i /> {connection === "live" ? `实时 · r${graph?.revision || 0}` : connection === "demo" ? "演示数据" : "连接中断"}
        </button>
      </div>

      <div className="stats-strip">
        <Stat label="记忆节点" value={stats.nodes} />
        <Stat label="逻辑关系" value={stats.edges} />
        <Stat label="分叉锚点" value={stats.branches} />
        <Stat label="待验证" value={stats.pending} />
        <div className="legend-block">
          <span><i className="line causal" />因果</span>
          <span><i className="line temporal" />时序</span>
          <span><i className="line semantic" />语义</span>
        </div>
      </div>

      <div className="workspace">
        <section className="graph-panel">
          <div className="panel-head">
            <div><span className="panel-index">01</span><h2>记忆链视图</h2></div>
            <div className="graph-actions">
              <button onClick={() => { setTraceIds(new Set()); setSearchIds(new Set()); cyRef.current?.fit(undefined, 36); }}>重置视图</button>
              <label><input type="checkbox" checked={showSuperseded} onChange={(event) => setShowSuperseded(event.target.checked)} />显示旧链</label>
            </div>
          </div>
          <div className="graph-canvas" ref={containerRef} role="img" aria-label="链式记忆关系图" />
          <div className="canvas-note"><kbd>滚轮</kbd> 缩放 · <kbd>拖拽</kbd> 平移 · 点击节点审计</div>
        </section>

        <aside className="detail-panel">
          <div className="panel-head"><div><span className="panel-index">02</span><h2>节点审计</h2></div></div>
          {!selected ? (
            <div className="empty-detail"><div className="radar" /><p>选择任一节点</p><span>查看原始内容、可信度变化、关系与修正链</span></div>
          ) : (
            <div className="detail-content">
              <div className="node-title-row">
                <span className={`status-dot ${selected.status}`} />
                <div><p>{TYPE_NAMES[selected.type] || selected.type} · {STATUS_NAMES[selected.status] || selected.status}</p><h3>{selected.summary}</h3></div>
              </div>
              <p className="node-content">{selected.content}</p>
              <div className="score-grid">
                <Score label="重要性" value={selected.importance} />
                <Score label="时效性" value={selected.timeliness} />
                <Score label="可信度" value={selected.credibility} />
              </div>
              <dl className="metadata-list"><div><dt>节点 ID</dt><dd>{selected.node_id}</dd></div><div><dt>记忆池</dt><dd>{selected.pool}</dd></div><div><dt>来源</dt><dd>{selected.credibility_source}</dd></div></dl>
              <div className="detail-actions"><button className="primary" onClick={traceSelected}>沿链回溯</button><button onClick={() => setTraceIds(new Set([selected.node_id]))}>仅聚焦此节点</button></div>

              {branches.length > 0 && <DetailSection title={`分叉控制 · ${branches.length} 条`}>
                <div className="branch-list">{branches.map((branchId) => <button key={branchId} onClick={() => toggleBranch(branchId)}><span>{graph.nodes.find((node) => node.node_id === branchId)?.summary || branchId}</span><b>{collapsedBranches.has(branchId) ? "展开" : "折叠"}</b></button>)}</div>
              </DetailSection>}

              <DetailSection title="可信度时间线">
                <div className="timeline">
                  {(selected.credibility_history || []).length ? selected.credibility_history.map((item, index) => <div key={`${item.timestamp}-${index}`}><i /><p>{item.action}<span>{Number(item.new_score).toFixed(2)}</span></p><small>{item.timestamp}</small></div>) : <div><i /><p>当前评分<span>{Number(selected.credibility).toFixed(2)}</span></p><small>{selected.created_at}</small></div>}
                </div>
              </DetailSection>

              {selectedEdges.length > 0 && <DetailSection title={`关系审计 · ${selectedEdges.length}`}>
                <div className="edge-list">{selectedEdges.map((edge) => <div key={edge.edge_id}><span className={`relation ${edge.relation}`}>{edge.relation}</span><p>{edge.source === selected.node_id ? `→ ${edge.target}` : `← ${edge.source}`}<small>{edge.label} · {edge.weight.toFixed(2)}</small></p>{edge.status === "pending_verification" && <button disabled={busy || connection !== "live"} onClick={() => confirmEdge(edge)}>确认</button>}</div>)}</div>
              </DetailSection>}

              {comparison && <DetailSection title="修正链对比">
                <div className="compare-grid"><CompareCard label="旧链 / SUPERSEDED" node={comparison.original} old /><CompareCard label="修正链 / ACTIVE" node={comparison.replacement} /></div>
              </DetailSection>}
            </div>
          )}
        </aside>
      </div>
      {error && <div className="error-toast" role="status">API: {error}</div>}
    </section>
  );
}

function Stat({ label, value }) {
  return <div className="stat"><p>{label}</p><strong>{String(value).padStart(2, "0")}</strong></div>;
}

function Score({ label, value }) {
  const safe = Math.max(0, Math.min(1, Number(value) || 0));
  return <div className="score"><div><span>{label}</span><b>{safe.toFixed(2)}</b></div><i><em style={{ width: `${safe * 100}%` }} /></i></div>;
}

function DetailSection({ title, children }) {
  return <section className="detail-section"><h4>{title}</h4>{children}</section>;
}

function CompareCard({ label, node, old = false }) {
  return <article className={old ? "old" : "new"}><span>{label}</span><strong>{node?.summary || "不可用"}</strong><p>{node?.content}</p></article>;
}
