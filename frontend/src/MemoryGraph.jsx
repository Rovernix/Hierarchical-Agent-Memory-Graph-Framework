import cytoscape from "cytoscape";
import { useEffect, useMemo, useRef, useState } from "react";

import { memoryCategory, memoryEvidence, relationDetails, temporalDetails, toCytoscapeElements, traceToRoot, visibleMemoryGraph } from "./graphTransforms.js";
import "./memory-graph.css";

const EMPTY_IDS = [];
const TYPE_NAMES = { fact: "事实", preference: "偏好", event: "事件", decision: "决策", state: "状态", feedback: "反馈", memory: "记忆" };
const POOL_NAMES = { working: "工作记忆", episodic: "情景记忆", buffer: "缓冲记忆", archive: "长期归档" };
const SOURCE_NAMES = { user_confirmed: "用户确认", agent_inferred: "模型推断", external_fetched: "外部来源" };
const STATUS_NAMES = { active: "活跃", superseded: "已替代", archived: "已归档", pending_verification: "待验证" };
const RELATION_NAMES = { causal: "因果", temporal: "时序", semantic: "语义" };
const METHOD_NAMES = { explicit_fact_parser: "明确事实提取", financial_scope: "收支信息关联", explicit_sequence_parser: "明确先后顺序", explicit_temporal_parser: "明确事件先后", explicit_date_parser: "日期原文解析", explicit_relation_parser: "原文关系解析", llm_extraction: "模型提取", llm_relation_review: "LLM 关系复核", calendar_order: "时间比较", normalized_date_order: "日期先后比较", event_time_order: "日期先后比较" };
const PHASE_NAMES = {
  idle: "待命", connecting: "正在连接", retrieving: "检索记忆", retrieval: "检索记忆",
  recalling: "检索记忆", extracting: "提取逻辑记忆", organizing: "整理历史记忆", generating: "生成回复", streaming: "生成回复", thinking: "正在思考",
  ingesting: "写入记忆", ingestion: "写入记忆", writing: "写入记忆", persisting: "保存记忆", done: "已完成", complete: "已完成", error: "发生错误",
};
const EVENT_NAMES = {
  phase: "处理进度", retrieval: "检索记忆", recall: "召回记忆", graph: "图谱更新",
  node_added: "新增记忆", edge_added: "关联记忆", ingest: "写入记忆", ingested: "写入记忆",
  query: "检索记忆", search: "检索记忆", retrieving: "检索记忆", extracting: "提取逻辑记忆", organizing: "整理历史记忆", extraction: "记忆整理完成", generating: "生成回复",
  writing: "写入记忆", complete: "回复完成", error: "发生错误", done: "回复完成",
};

function Icon({ name, size = 16 }) {
  const paths = {
    close: <path d="m6 6 12 12M18 6 6 18" />,
    expand: <><path d="M8 3H3v5M16 3h5v5M21 16v5h-5M8 21H3v-5" /><path d="m3 3 6 6m12-6-6 6m6 12-6-6M3 21l6-6" /></>,
    collapse: <><path d="M3 9h6V3m6 0v6h6M21 15h-6v6M9 21v-6H3" /></>,
    plus: <path d="M12 5v14M5 12h14" />,
    minus: <path d="M5 12h14" />,
    fit: <><path d="M8 3H3v5m13-5h5v5m0 8v5h-5m-8 0H3v-5" /><rect x="8" y="8" width="8" height="8" rx="1" /></>,
    graph: <><path d="m8 7 8 5m-8 5 8-5M5 9v6" /><circle cx="5" cy="5" r="3" /><circle cx="5" cy="19" r="3" /><circle cx="19" cy="12" r="3" /></>,
    trace: <><path d="M20 7H9a5 5 0 0 0 0 10h4m-8-6L1 7l4-4" /><circle cx="18" cy="17" r="3" /></>,
    activity: <path d="M3 12h4l3-7 4 14 3-7h4" />,
    arrow: <path d="M5 12h14m-5-5 5 5-5 5" />,
  };
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[name] || paths.graph}</svg>;
}

const GRAPH_STYLE = [
  { selector: "node", style: {
    shape: "round-rectangle", width: 180, height: 72, padding: 8,
    "background-color": "#222628", "border-color": "#484d50", "border-width": 1,
    label: "data(displayLabel)", color: "#d4d7da", "font-family": "Segoe UI, Microsoft YaHei, sans-serif",
    "font-size": 11, "font-weight": 400, "text-wrap": "wrap", "text-max-width": 156, "text-overflow-wrap": "anywhere",
    "text-valign": "center", "text-halign": "center", "line-height": 1.5, "overlay-opacity": 0,
    "transition-property": "background-color, border-color", "transition-duration": "160ms",
  } },
  { selector: 'node[type = "decision"]', style: { "border-color": "#747565", "background-color": "#292a26" } },
  { selector: 'node[type = "state"]', style: { "border-color": "#667681", "background-color": "#252a2e" } },
  { selector: 'node[type = "feedback"]', style: { "border-color": "#7e6f78", "background-color": "#2c262b" } },
  { selector: 'node[displayCategory = "fact"]', style: { "border-color": "#7c9290", "background-color": "#242e2d" } },
  { selector: 'node[displayCategory = "preference"]', style: { "border-color": "#908477", "background-color": "#302b26" } },
  { selector: 'node[displayCategory = "event"]', style: { "border-color": "#657b7b", "background-color": "#222a2b" } },
  { selector: 'node[displayCategory = "decision"]', style: { "border-color": "#747565", "background-color": "#292a26" } },
  { selector: 'node[displayCategory = "state"]', style: { "border-color": "#667681", "background-color": "#252a2e" } },
  { selector: 'node[status = "pending_verification"]', style: { "border-style": "dashed" } },
  { selector: 'node[status = "superseded"], node[status = "archived"]', style: { opacity: 0.45 } },
  { selector: "edge", style: {
    width: 1.1, "line-color": "#505759", "target-arrow-color": "#505759", "target-arrow-shape": "triangle",
    "arrow-scale": 0.7, "curve-style": "bezier", label: "data(displayLabel)", color: "#858d8d",
    "font-size": 9, "text-wrap": "ellipsis", "text-max-width": 140, "text-background-color": "#191b1d", "text-background-opacity": 0.94,
    "text-background-padding": 4, "text-rotation": "autorotate", "text-margin-y": -8,
    "overlay-opacity": 0,
  } },
  { selector: 'edge[relation = "temporal"]', style: { "line-style": "dashed" } },
  { selector: 'edge[displayTemporalKind = "event_time"]', style: { "line-color": "#94a6ad", "target-arrow-color": "#94a6ad", color: "#acbbc0", width: 1.6 } },
  { selector: 'edge[displayTemporalKind = "conversation_order"]', style: { "line-color": "#5b635f", "target-arrow-color": "#5b635f", color: "#737e77", "line-style": "dotted" } },
  { selector: 'edge[relation = "semantic"]', style: { "line-style": "dotted" } },
  { selector: 'edge[status = "superseded"]', style: { opacity: 0.3 } },
  { selector: "node.recalled", style: {
    "background-color": "#263630", "border-color": "#93baa4", "border-width": 1.5,
    color: "#dce8e0", "underlay-color": "#93baa4", "underlay-opacity": 0.05, "underlay-padding": 5, opacity: 1,
  } },
  { selector: "edge.recalled, edge.traced", style: { "line-color": "#93baa4", "target-arrow-color": "#93baa4", width: 1.8, opacity: 1 } },
  { selector: "node.traced", style: { "border-color": "#b6cbbf", "border-width": 2, opacity: 1 } },
  { selector: ".muted", style: { opacity: 0.18 } },
  { selector: "node:selected", style: { "border-color": "#e1e9e4", "border-width": 2, "background-color": "#303936", opacity: 1 } },
];

function fitGraph(cy, collection) {
  if (!cy || cy.destroyed() || !cy.nodes().length) return;
  cy.fit(collection || cy.elements(), 42);
  if (cy.zoom() > 1.15) {
    cy.zoom(1.15);
    cy.center(collection || cy.elements());
  }
}

function arrangeGraph(cy, horizontal = false) {
  if (!cy?.nodes().length) return;
  if (!cy.edges().length) {
    const columns = horizontal ? Math.max(1, Math.min(3, Math.floor(cy.width() / 230))) : 1;
    cy.nodes().forEach((node, index) => node.position({ x: (index % columns) * 240, y: Math.floor(index / columns) * 135 }));
    fitGraph(cy);
    return;
  }
  // Semantic links may point back to an earlier event. Keep them visible, but
  // use the chronological/causal direction to place the nodes when available.
  const orderedEdges = cy.edges().filter((edge) => ["temporal", "causal"].includes(edge.data("relation")));
  const layoutElements = cy.nodes().union(orderedEdges.length ? orderedEdges : cy.edges());
  layoutElements.layout({
    name: "breadthfirst", directed: true, spacingFactor: 1.25, avoidOverlap: true,
    fit: false, padding: 40, circle: false, grid: false,
  }).run();
  // Keep compact chains legible; the expanded view runs left to right.
  const rows = new Map();
  cy.nodes().forEach((node) => {
    const depth = Math.round(node.position("y") * 100) / 100;
    const row = rows.get(depth) || [];
    row.push(node);
    rows.set(depth, row);
  });
  [...rows.keys()].sort((a, b) => a - b).forEach((depth, rank) => {
    const row = rows.get(depth).sort((a, b) => a.position("x") - b.position("x"));
    row.forEach((node, index) => {
      const across = index - (row.length - 1) / 2;
      node.position(horizontal ? { x: rank * 275, y: across * 125 } : { x: across * 240, y: rank * 150 });
    });
  });
  fitGraph(cy);
}

function labelForNode(node) {
  const title = String(node.summary || node.content || node.node_id || "").replace(/\s+/g, " ");
  const characters = Array.from(title);
  const time = temporalDetails(node);
  const category = TYPE_NAMES[memoryCategory(node)] || "记忆";
  const heading = time ? `${Array.from(time.compact).slice(0, 15).join("")} · ${category}` : `${category} · ${POOL_NAMES[node.pool] || node.pool || "记忆"}`;
  return `${heading}\n${characters.length > 26 ? `${characters.slice(0, 26).join("")}…` : title}`;
}

function eventMessage(event) {
  const message = event.message || event.detail || event.summary || event.data?.message;
  if (typeof message === "string") return message;
  if (event.kind === "phase" || event.type === "phase") return PHASE_NAMES[event.phase || event.data?.phase] || "更新处理状态";
  const kind = event.kind || event.type || "";
  if (event.data?.node_id) return `${EVENT_NAMES[kind] || "记忆更新"} · ${String(event.data.node_id).slice(0, 12)}`;
  return EVENT_NAMES[kind] || (kind ? kind.replace(/_/g, " ") : "记忆图谱已更新");
}

function eventTime(value) {
  if (!value) return "";
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? "" : parsed.toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

export function MemoryGraph({ graph, activeNodeIds = EMPTY_IDS, phase = "idle", events = EMPTY_IDS, onClose, expanded = false, onExpand }) {
  const containerRef = useRef(null);
  const cyRef = useRef(null);
  const userViewportRef = useRef(false);
  const [tab, setTab] = useState("graph");
  const [selectedId, setSelectedId] = useState("");
  const [traceIds, setTraceIds] = useState(EMPTY_IDS);
  const [focusRecall, setFocusRecall] = useState(false);
  const [zoom, setZoom] = useState(100);
  const [unseenIds, setUnseenIds] = useState(EMPTY_IDS);
  const [showHistory, setShowHistory] = useState(false);
  const visibleGraph = useMemo(() => visibleMemoryGraph(graph, showHistory), [graph, showHistory]);
  const nodes = visibleGraph.nodes;
  const edges = visibleGraph.edges;
  const historyCount = (graph?.nodes || EMPTY_IDS).reduce((count, node) => count + (node.status === "superseded" || node.status === "archived" ? 1 : 0) + (node.metadata?.revisions?.length || 0), 0);
  const selected = nodes.find((node) => node.node_id === selectedId);
  const selectedTime = temporalDetails(selected);
  const selectedProvenance = selected?.metadata?.provenance;
  const selectedEvidence = memoryEvidence(selected);
  const selectedRevisions = Array.isArray(selected?.metadata?.revisions) ? selected.metadata.revisions : EMPTY_IDS;
  const recalledIds = useMemo(() => new Set(activeNodeIds), [activeNodeIds]);
  const recalledCount = nodes.filter((node) => recalledIds.has(node.node_id)).length;
  const selectedEdges = useMemo(() => edges.filter((edge) => edge.source === selectedId || edge.target === selectedId), [edges, selectedId]);
  const busy = !["idle", "done", "complete", "error"].includes(phase);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return undefined;
    const cy = cytoscape({ container, elements: [], layout: { name: "preset" }, style: GRAPH_STYLE,
      minZoom: 0.12, maxZoom: 2.5, wheelSensitivity: 0.25, selectionType: "single", boxSelectionEnabled: false });
    cyRef.current = cy;
    cy.on("tap", "node", (event) => { userViewportRef.current = true; setSelectedId(event.target.id()); });
    cy.on("tap", (event) => { if (event.target === cy) setSelectedId(""); });
    cy.on("zoom", () => setZoom(Math.round(cy.zoom() * 100)));
    cy.on("dragpan scrollzoom pinchzoom dragfree", (event) => { if (event.originalEvent) userViewportRef.current = true; });
    const observer = new ResizeObserver(() => cy.resize());
    observer.observe(container);
    return () => { observer.disconnect(); cy.destroy(); cyRef.current = null; };
  }, []);

  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) return;
    const elements = toCytoscapeElements(visibleGraph).map((element) => {
      const relation = element.data.node_id ? null : relationDetails(element.data, graph);
      return { ...element, data: { ...element.data, displayLabel: relation ? relation.label : labelForNode(element.data), displayTemporalKind: relation?.kind || "", displayCategory: element.data.node_id ? memoryCategory(element.data) : "" } };
    });
    const wanted = new Set(elements.map((element) => element.data.id));
    const hadSharedNodes = cy.nodes().some((node) => wanted.has(node.id()));
    const viewport = { zoom: cy.zoom(), pan: { ...cy.pan() } };
    const addedIds = [];
    cy.batch(() => {
      cy.elements().filter((element) => !wanted.has(element.id())).remove();
      const positions = cy.nodes().map((node) => ({ ...node.position() }));
      for (const element of elements) {
        const existing = cy.getElementById(element.data.id);
        if (existing.length) { existing.data(element.data); continue; }
        if (!element.data.node_id) { cy.add(element); continue; }
        addedIds.push(element.data.id);
        const parentEdge = edges.find((edge) => edge.target === element.data.id && cy.getElementById(edge.source).length);
        const parent = parentEdge ? cy.getElementById(parentEdge.source) : null;
        const position = parent ? { x: parent.position("x") + (expanded ? 275 : 0), y: parent.position("y") + (expanded ? 0 : 150) } : { x: 0, y: positions.length ? Math.max(...positions.map((point) => point.y)) + 150 : 0 };
        while (positions.some((point) => Math.abs(point.x - position.x) < 210 && Math.abs(point.y - position.y) < 110)) {
          if (expanded) position.y += 125;
          else position.x += 240;
        }
        positions.push(position);
        cy.add({ ...element, position });
      }
    });
    if (!hadSharedNodes) userViewportRef.current = false;
    if (cy.nodes().length && !userViewportRef.current) { arrangeGraph(cy, expanded); setUnseenIds(EMPTY_IDS); }
    else {
      cy.zoom(viewport.zoom); cy.pan(viewport.pan);
      if (addedIds.length) setUnseenIds((current) => [...new Set([...current, ...addedIds])].filter((id) => wanted.has(id)));
    }
    if (selectedId && !wanted.has(selectedId)) setSelectedId("");
  }, [visibleGraph]); // Once the reader navigates, new memory keeps their viewport intact.

  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) return;
    const traced = new Set(traceIds);
    cy.batch(() => {
      cy.elements().removeClass("recalled traced muted");
      cy.nodes().forEach((node) => {
        if (recalledIds.has(node.id())) node.addClass("recalled");
        if (traced.has(node.id())) node.addClass("traced");
        if (focusRecall && recalledCount && !recalledIds.has(node.id())) node.addClass("muted");
      });
      cy.edges().forEach((edge) => {
        if (recalledIds.has(edge.source().id()) && recalledIds.has(edge.target().id())) edge.addClass("recalled");
        if (traced.has(edge.source().id()) && traced.has(edge.target().id())) edge.addClass("traced");
        if (focusRecall && recalledCount && !(recalledIds.has(edge.source().id()) && recalledIds.has(edge.target().id()))) edge.addClass("muted");
      });
    });
  }, [visibleGraph, recalledIds, traceIds, focusRecall, recalledCount]);

  useEffect(() => {
    const cy = cyRef.current;
    cy?.resize();
    if (tab === "graph" && !userViewportRef.current) fitGraph(cy);
  }, [tab]);

  useEffect(() => {
    const cy = cyRef.current;
    cy?.resize();
    arrangeGraph(cy, expanded);
    userViewportRef.current = false;
    setUnseenIds(EMPTY_IDS);
  }, [expanded]);

  useEffect(() => {
    // Details reduce the canvas height. Small graphs should remain fully visible.
    const frame = requestAnimationFrame(() => {
      const cy = cyRef.current;
      if (!cy || cy.destroyed()) return;
      cy.resize();
      if (selectedId) {
        const node = cy.getElementById(selectedId);
        if (node.length) {
          cy.stop();
          if (cy.nodes().length <= 4) fitGraph(cy);
          else cy.center(node);
        }
      }
    });
    return () => cancelAnimationFrame(frame);
  }, [selectedId]);

  function selectNode(id) {
    const cy = cyRef.current;
    if (id) userViewportRef.current = true;
    setSelectedId(id);
    cy?.nodes().unselect();
    if (id) cy?.getElementById(id)?.select();
  }

  function changeZoom(factor) {
    const cy = cyRef.current;
    if (!cy) return;
    userViewportRef.current = true;
    cy.zoom({ level: Math.min(cy.maxZoom(), Math.max(cy.minZoom(), cy.zoom() * factor)), renderedPosition: { x: cy.width() / 2, y: cy.height() / 2 } });
  }

  function traceSelected() {
    if (!selectedId) return;
    userViewportRef.current = true;
    const path = traceToRoot(visibleGraph, selectedId);
    setTraceIds(path);
    setFocusRecall(false);
    const ids = new Set(path);
    fitGraph(cyRef.current, cyRef.current?.nodes().filter((node) => ids.has(node.id())));
  }

  function showAllNodes() {
    userViewportRef.current = false;
    arrangeGraph(cyRef.current, expanded);
    setUnseenIds(EMPTY_IDS);
  }

  function changeHistory(value) {
    userViewportRef.current = false;
    setTraceIds(EMPTY_IDS);
    setUnseenIds(EMPTY_IDS);
    setShowHistory(value);
  }

  return <aside className={`memory-panel${expanded ? " is-expanded" : ""}`} aria-label="HAMGF 记忆图谱">
    <header className="memory-header">
      <div className="memory-heading"><span className="memory-heading-icon"><Icon name="graph" size={17} /></span><div><h2>记忆图谱</h2><span>HAMGF · LOGICAL MEMORY</span></div></div>
      <div className="memory-header-actions">
        {onExpand && <button className="memory-icon-button" onClick={onExpand} title={expanded ? "收起大视图" : "展开图谱"} aria-label={expanded ? "收起大视图" : "展开图谱"}><Icon name={expanded ? "collapse" : "expand"} size={15} /></button>}
        {onClose && <button className="memory-icon-button" onClick={onClose} title="关闭记忆图谱" aria-label="关闭记忆图谱"><Icon name="close" size={16} /></button>}
      </div>
    </header>
    <div className="memory-summary"><span><i className={busy ? "is-busy" : ""} />{PHASE_NAMES[phase] || "处理中"}</span><span>{nodes.length} 个节点<span className="memory-separator">/</span>{edges.length} 条关联</span></div>
    <div className="memory-tabs" role="tablist" aria-label="记忆视图">
      <button role="tab" id="memory-graph-tab" aria-selected={tab === "graph"} aria-controls="memory-graph-view" className={tab === "graph" ? "is-active" : ""} onClick={() => setTab("graph")}><Icon name="graph" size={14} />逻辑关系</button>
      <button role="tab" id="memory-events-tab" aria-selected={tab === "events"} aria-controls="memory-events-view" className={tab === "events" ? "is-active" : ""} onClick={() => setTab("events")}><Icon name="activity" size={14} />实时动态{events.length > 0 && <span className="memory-tab-count">{events.length}</span>}</button>
    </div>
    <div id="memory-graph-view" role="tabpanel" aria-labelledby="memory-graph-tab" className="memory-graph-view" hidden={tab !== "graph"}>
      <div className="memory-history-toggle" role="group" aria-label="记忆版本"><button className={!showHistory ? "is-active" : ""} aria-pressed={!showHistory} onClick={() => changeHistory(false)}>当前记忆</button><button className={showHistory ? "is-active" : ""} aria-pressed={showHistory} onClick={() => changeHistory(true)}>含历史版本{historyCount > 0 && <span>{historyCount}</span>}</button></div>
      <div className="memory-legend"><span><i className="memory-dot fact" />事实</span><span><i className="memory-dot preference" />偏好</span><span><i className="memory-dot event" />事件</span><span><i className="memory-dot state" />状态</span><span><i className="memory-dot decision" />决策</span></div>
      <div className="memory-canvas-wrap">
        <div className="memory-canvas" ref={containerRef} role="img" aria-label={`包含 ${nodes.length} 条逻辑记忆和 ${edges.length} 条真实关联。可在下方列表选择节点查看详情。`} />
        {nodes.length === 0 && <div className="memory-empty"><div className="memory-empty-icon"><Icon name="graph" size={28} /></div><h3>这里保存有用的记忆</h3><p>仅保存值得记住的事实与关系；<br />闲聊不会创建节点。</p><span>{historyCount && !showHistory ? "历史记忆可在“含历史版本”中查看" : "尚无逻辑记忆"}</span></div>}
        {nodes.length > 0 && <>
          <div className="memory-canvas-caption"><span>LOGICAL MEMORY GRAPH</span><span>{traceIds.length ? `正在回溯 ${traceIds.length} 条记忆` : edges.length ? "拖动平移 · 滚轮缩放" : "独立记忆 · 暂无已确认关系"}</span></div>
          {unseenIds.length > 0 && <button className="memory-new-nodes" onClick={showAllNodes}><Icon name="fit" size={12} />查看 {unseenIds.length} 个新增节点</button>}
          {recalledCount > 0 && <button className={`memory-recall-badge${focusRecall ? " is-active" : ""}`} onClick={() => setFocusRecall((value) => !value)} aria-pressed={focusRecall}><span />本轮召回 {recalledCount}<span className="memory-recall-action">{focusRecall ? "查看全部" : "聚焦"}</span></button>}
          <div className="memory-graph-controls"><button onClick={() => changeZoom(1 / 1.25)} aria-label="缩小图谱" title="缩小"><Icon name="minus" size={14} /></button><span>{zoom}%</span><button onClick={() => changeZoom(1.25)} aria-label="放大图谱" title="放大"><Icon name="plus" size={14} /></button><i /><button onClick={showAllNodes} aria-label="适应图谱大小" title="适应图谱"><Icon name="fit" size={14} /></button></div>
        </>}
      </div>
      {nodes.length > 0 && <div className="memory-node-navigation"><select aria-label="选择记忆节点" value={selectedId} onChange={(event) => selectNode(event.target.value)}><option value="">选择节点，查看记忆详情</option>{nodes.map((node) => <option key={node.node_id} value={node.node_id}>{TYPE_NAMES[memoryCategory(node)] || "记忆"} · {String(node.summary || node.content || node.node_id).slice(0, 56)}</option>)}</select><button onClick={() => { setTraceIds(EMPTY_IDS); showAllNodes(); }} title="重新排列图谱">重排</button></div>}
      {selected && <section className="memory-detail" aria-label="节点详情">
        <div className="memory-detail-top"><span className={`memory-type-tag ${memoryCategory(selected)}`}>{TYPE_NAMES[memoryCategory(selected)] || "记忆"}</span><span className="memory-detail-pool">{POOL_NAMES[selected.pool] || selected.pool}</span><button className="memory-icon-button" onClick={() => selectNode("")} aria-label="关闭节点详情"><Icon name="close" size={14} /></button></div>
        <h3>{selected.summary || "记忆内容"}</h3>
        {selectedTime && <section className="memory-temporal" aria-label="事件时间"><div className="memory-temporal-heading"><Icon name="activity" size={13} /><strong>事件时间</strong><span>{selectedTime.display}</span></div><dl>{selectedTime.raw && <div><dt>时间原文</dt><dd>{selectedTime.raw}</dd></div>}{selectedTime.normalized && <div><dt>规范日期</dt><dd><code>{selectedTime.normalized}</code></dd></div>}</dl>{selectedTime.evidence && <blockquote>{selectedTime.evidence}</blockquote>}</section>}
        <p>{selected.content || selected.summary || "此节点没有文本内容。"}</p>
        {selectedProvenance?.method && <div className="memory-evidence-source"><span>提取方式</span><strong>{METHOD_NAMES[selectedProvenance.method] || selectedProvenance.method}</strong>{selectedProvenance.model && <span>{selectedProvenance.model}</span>}</div>}
        {selectedEvidence.length > 0 && <section className="memory-evidence-list" aria-label="来源证据"><h4>来源原文 <span>{selectedEvidence.length} 条引用</span></h4>{selectedEvidence.map((item, index) => <blockquote className="memory-source-evidence" key={`${item.message_id || "source"}-${index}`}>{item.quote}</blockquote>)}</section>}
        {showHistory && selectedRevisions.length > 0 && <details className="memory-revisions"><summary>历史版本 <span>{selectedRevisions.length}</span></summary>{selectedRevisions.map((revision, index) => <article key={index}><h4>{revision.summary || "之前的记忆"}</h4><p>{revision.content}</p>{revision.reason && <span className="memory-revision-reason">{revision.reason === "explicit_user_update" ? "用户更新了这条记忆" : revision.reason}</span>}{Array.isArray(revision.evidence) && revision.evidence.map((item, quoteIndex) => typeof item.quote === "string" ? <blockquote className="memory-source-evidence" key={quoteIndex}>{item.quote}</blockquote> : null)}</article>)}</details>}
        <dl className="memory-detail-meta"><div><dt>状态</dt><dd>{STATUS_NAMES[selected.status] || selected.status || "—"}</dd></div>{selected.credibility_source && <div><dt>来源</dt><dd>{SOURCE_NAMES[selected.credibility_source] || selected.credibility_source}</dd></div>}{Number.isFinite(selected.importance) && <div><dt>重要性</dt><dd>{selected.importance.toFixed(2)}</dd></div>}{Number.isFinite(selected.credibility) && <div><dt>可信度</dt><dd>{selected.credibility.toFixed(2)}</dd></div>}<div><dt>关联</dt><dd>{selectedEdges.length}</dd></div></dl>
        {selectedEdges.length > 0 && <div className="memory-detail-relations">{selectedEdges.map((edge, index) => {
          const incoming = edge.target === selectedId;
          const other = nodes.find((node) => node.node_id === (incoming ? edge.source : edge.target));
          const relation = relationDetails(edge, graph);
          return <div className="memory-relation-card" key={edge.edge_id || `${edge.source}-${edge.target}-${index}`}><button title={relation.label} onClick={() => selectNode(incoming ? edge.source : edge.target)}><span>{incoming ? "←" : "→"} {relation.kind === "conversation_order" ? "对话顺序" : relation.kind === "event_time" ? "事件时间" : RELATION_NAMES[edge.relation] || edge.relation}</span><span>{other?.summary || other?.content || (incoming ? edge.source : edge.target)}</span></button><div className="memory-relation-label">{relation.label}{relation.kind === "conversation_order" && <small>仅表示消息顺序</small>}</div>{relation.evidence && <details className="memory-relation-evidence"><summary>关系依据{relation.method && <span>{METHOD_NAMES[relation.method] || relation.method}</span>}</summary><blockquote>{relation.evidence}</blockquote></details>}</div>;
        })}</div>}
        {selectedEdges.length === 0 && <p className="memory-no-relations">这条记忆尚无已确认的关系。</p>}
        <div className="memory-detail-bottom"><button onClick={traceSelected} disabled={selectedEdges.length === 0}><Icon name="trace" size={14} />沿关系回溯</button>{traceIds.length > 0 && <button onClick={() => setTraceIds(EMPTY_IDS)}>清除高亮</button>}<span title={selected.node_id}>{selected.node_id.slice(-8)}</span></div>
      </section>}
      <footer className="memory-footer"><div><span className="memory-line causal" />因果<span className="memory-line temporal" />时序<span className="memory-line semantic" />语义</div><span><i />本轮召回</span></footer>
    </div>
    <div id="memory-events-view" role="tabpanel" aria-labelledby="memory-events-tab" className="memory-events-view" hidden={tab !== "events"}>
      <div className="memory-events-intro">查看记忆的检索与更新过程</div>
      {events.length ? <ol className="memory-events">{events.slice(-50).reverse().map((event, index) => <li key={event.id || `${event.revision || event.timestamp || "event"}-${index}`} className={(event.type || event.kind) === "error" ? "is-error" : ""}><span className="memory-event-marker" /><div><div className="memory-event-heading"><strong>{EVENT_NAMES[event.type || event.kind] || "记忆活动"}</strong><time>{eventTime(event.timestamp || event.created_at)}</time></div><p>{eventMessage(event)}</p>{event.data?.node_ids?.length > 0 && <small>{event.data.node_ids.length} 个记忆节点</small>}</div></li>)}</ol> : <div className="memory-events-empty"><Icon name="activity" size={24} /><h3>等待记忆活动</h3><p>开始对话后，检索和写入过程会在这里实时显示。</p></div>}
    </div>
  </aside>;
}

export default MemoryGraph;
