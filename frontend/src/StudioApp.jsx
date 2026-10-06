import { useEffect, useRef, useState } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { api, streamMessage, streamOperation } from "./api.js";
import { Icon, Mark } from "./Icons.jsx";
import { Settings } from "./Settings.jsx";
import { MemoryGraph } from "./MemoryGraph.jsx";
import { visibleMemoryGraph } from "./graphTransforms.js";

const EMPTY_GRAPH = { nodes: [], edges: [], revision: 0 };
const normalizeMessages = (items = []) => items.map((m) => ({ ...m, retrieval: m.retrieval || m.memory, streaming: m.status === "streaming" }));
const stageLabels = { retrieving: "正在检索记忆", retrieval: "正在检索记忆", generating: "正在生成回答", extracting: "正在提炼事实与关系", reviewing: "正在复核关系", organizing: "正在整理历史记忆", writing: "正在写入记忆", ingestion: "正在写入记忆", complete: "记忆已同步", idle: "等待对话" };
function timeLabel(value) { return value ? new Date(typeof value === "number" && value < 1e12 ? value * 1000 : value).toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" }) : ""; }
function Message({ message, onGraph, onCopy, copied }) {
  const [traceOpen, setTraceOpen] = useState(false);
  const user = message.role === "user";
  const memoryCount = message.extraction?.memory_count ?? message.extraction?.event_count ?? 0;
  const relationCount = (message.extraction?.temporal_edges || 0) + (message.extraction?.semantic_edges || 0) + (message.extraction?.causal_edges || 0);
  return <article className={`message ${user ? "user-message" : "assistant-message"}`}>
    <div className={`avatar ${user ? "user-avatar" : "agent-avatar"}`}>{user ? "你" : <Mark size={23}/>}</div>
    <div className="message-main"><div className="message-heading"><strong>{user ? "你" : "HAMGF"}</strong>{!user && <span className="agent-tag">AGENT</span>}<time>{timeLabel(message.created_at)}</time></div>
      {!user && message.retrieval && <button className={`memory-trace ${traceOpen ? "open" : ""}`} onClick={() => setTraceOpen(!traceOpen)}><Icon name="graph" size={14}/><span>HAMGF 记忆链</span><span className="trace-count">{new Set(message.retrieval.node_ids || []).size} 条已召回</span><Icon name="down" size={13}/></button>}
      {traceOpen && <div className="inline-trace"><div><span className="connection-dot"/>检索 → 上下文 → 回答 → 记忆写入</div><p>{Array.isArray(message.retrieval.narrative) ? message.retrieval.narrative.map((node) => node.summary || node.content).join("\n") || "当前没有可召回的历史记忆。本轮将按内容判断是否需要保存记忆。" : message.retrieval.narrative || "此回答使用了当前会话中检索到的记忆。"}</p><button className="text-button" onClick={onGraph}>在图谱中查看 <Icon name="right" size={13}/></button></div>}
      {message.reasoning && <details className="reasoning"><summary><Icon name="spark" size={14}/>深度思考<Icon name="down" size={13}/></summary><div>{message.reasoning}</div></details>}
      <div className="message-markdown"><Markdown remarkPlugins={[remarkGfm]} components={{ a: (props) => <a {...props} target="_blank" rel="noopener noreferrer"/> }}>{message.content || (message.streaming ? "" : "（未生成回答）")}</Markdown>{message.streaming && <span className="stream-cursor"/>}</div>
      {message.status === "cancelled" && <span className="message-state">已停止生成</span>}
      {message.status === "error" && <span className="message-state">回答未完成</span>}
      {message.status === "truncated" && <span className="message-state">已达到输出上限</span>}
      {message.status === "interrupted" && <span className="message-state">上次生成已中断</span>}
      {!user && message.extraction && (memoryCount > 0 || relationCount > 0) && <button className="memory-write-summary" onClick={onGraph}><Icon name="graph" size={13}/><span>{memoryCount > 0 ? `新增 ${message.extraction.created_count ?? message.extraction.event_count ?? 0} 条记忆` : "关联已有记忆"}</span>{message.extraction.updated_count > 0 && <span>更新 {message.extraction.updated_count} 条</span>}{message.extraction.deduplicated_count > 0 && <span>合并重复 {message.extraction.deduplicated_count} 条</span>}{message.extraction.temporal_edges > 0 && <span>{message.extraction.temporal_edges} 条时序</span>}{message.extraction.semantic_edges > 0 && <span>{message.extraction.semantic_edges} 条语义</span>}{message.extraction.causal_edges > 0 && <span>{message.extraction.causal_edges} 条因果</span>}{message.extraction.relation_review?.created_count > 0 && <span>LLM 复核 {message.extraction.relation_review.created_count} 条</span>}<Icon name="right" size={12}/></button>}
      {!user && message.extraction && memoryCount === 0 && relationCount === 0 && !message.streaming && <span className="memory-skip-note">{message.extraction.status === "partial" ? "本轮未写入记忆 · 聊天记录已保留" : "本轮没有新增记忆"}</span>}
      {!user && (message.extraction?.warnings || []).map((warning, index) => <p className="extraction-warning" key={index}><Icon name="info" size={13}/>{warning}</p>)}
      {!message.streaming && message.content && <div className="message-tools"><button className="icon-button" title="复制内容" aria-label="复制内容" onClick={() => onCopy(message)}><Icon name={copied === message.id ? "check" : "copy"} size={14}/></button>{!user && <button className="icon-button" title="查看记忆图谱" aria-label="查看记忆图谱" onClick={onGraph}><Icon name="graph" size={14}/></button>}</div>}
    </div>
  </article>;
}

export function StudioApp() {
  const [settings, setSettings] = useState(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [conversations, setConversations] = useState([]);
  const [current, setCurrent] = useState(null);
  const [messages, setMessages] = useState([]);
  const [graph, setGraph] = useState(EMPTY_GRAPH);
  const [graphOpen, setGraphOpen] = useState(() => localStorage.getItem("hamgf.graph-open") === "true");
  const [graphExpanded, setGraphExpanded] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [filter, setFilter] = useState("");
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(true);
  const [phase, setPhase] = useState("idle");
  const [activeNodes, setActiveNodes] = useState([]);
  const [events, setEvents] = useState([]);
  const [error, setError] = useState("");
  const [toast, setToast] = useState("");
  const [copied, setCopied] = useState(null);
  const [manage, setManage] = useState(null);
  const abort = useRef(null);
  const activeId = useRef(null);
  const busyRef = useRef(false);
  const requestVersion = useRef(0);
  const textarea = useRef(null);
  const scrollArea = useRef(null);
  const nearBottom = useRef(true);
  const search = useRef(null);

  async function refreshList() { const result = await api("/conversations"); setConversations(result.conversations || []); }
  useEffect(() => {
    let mounted = true;
    Promise.all([api("/settings"), api("/conversations")]).then(([prefs, list]) => { if (mounted) { setSettings(prefs); setConversations(list.conversations || []); } }).catch((e) => { if (mounted) setError(e.message); }).finally(() => { if (mounted) setLoading(false); });
    return () => { mounted = false; abort.current?.abort(); };
  }, []);
  useEffect(() => { localStorage.setItem("hamgf.graph-open", String(graphOpen)); }, [graphOpen]);
  useEffect(() => { if (nearBottom.current) scrollArea.current?.scrollTo({ top: scrollArea.current.scrollHeight, behavior: "instant" }); }, [messages, busy]);
  useEffect(() => { if (toast) { const t = setTimeout(() => setToast(""), 2800); return () => clearTimeout(t); } }, [toast]);
  useEffect(() => {
    const handler = (event) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "n") { event.preventDefault(); newChat(); }
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") { event.preventDefault(); setSidebarOpen(true); setTimeout(() => search.current?.focus(), 0); }
      if ((event.ctrlKey || event.metaKey) && event.key === ",") { event.preventDefault(); if (settings) setSettingsOpen(true); }
    };
    window.addEventListener("keydown", handler); return () => window.removeEventListener("keydown", handler);
  }, [settings]);

  function newChat() {
    if (busyRef.current) return;
    requestVersion.current++; activeId.current = null; setCurrent(null); setMessages([]); setGraph(EMPTY_GRAPH); setActiveNodes([]); setEvents([]); setPhase("idle"); setLoading(false); setError(""); setDraft(""); textarea.current?.focus();
  }
  async function openChat(id) {
    if (busyRef.current) return;
    const version = ++requestVersion.current;
    setLoading(true); setError("");
    try {
      const conversation = await api(`/conversations/${id}`);
      if (version !== requestVersion.current) return;
      activeId.current = id; setCurrent(conversation); setMessages(normalizeMessages(conversation.messages)); setGraph(conversation.graph || EMPTY_GRAPH); setActiveNodes(conversation.graph?.active_node_ids || []); setEvents([]); setPhase("idle"); setDraft(""); nearBottom.current = true;
    } catch (e) { if (version === requestVersion.current) setError(e.message); }
    finally { if (version === requestVersion.current) setLoading(false); }
  }
  async function send(event) {
    event?.preventDefault();
    if (!draft.trim() || busyRef.current || loading) return;
    if (!settings?.has_api_key) { setSettingsOpen(true); return; }
    const text = draft.trim();
    const controller = new AbortController(); abort.current = controller;
    busyRef.current = true; setBusy(true); setError(""); setDraft(""); setPhase("retrieving"); nearBottom.current = true;
    if (textarea.current) textarea.current.style.height = "auto";
    const assistantId = `pending-${Date.now()}`;
    let id = activeId.current;
    let streamStarted = false;
    let retrieval = null;
    try {
      if (!id) {
        const conversation = await api("/conversations", { method: "POST", body: {} });
        id = conversation.id; activeId.current = id; setCurrent(conversation); await refreshList();
      }
      setMessages((old) => [...old, { id: `user-${Date.now()}`, role: "user", content: text, created_at: new Date().toISOString() }, { id: assistantId, role: "assistant", content: "", reasoning: "", streaming: true }]);
      streamStarted = true;
      await streamMessage(id, text, controller.signal, (type, data) => {
        if (type === "token" || type === "reasoning") setMessages((old) => old.map((m) => m.id === assistantId ? { ...m, [type === "token" ? "content" : "reasoning"]: (m[type === "token" ? "content" : "reasoning"] || "") + data.delta } : m));
        if (type === "status") { setPhase(data.stage); setEvents((old) => [...old.slice(-49), { type: data.stage, message: data.message, timestamp: Date.now() }]); }
        if (type === "retrieval") { retrieval = data; setActiveNodes(data.node_ids || []); setMessages((old) => old.map((m) => m.id === assistantId ? { ...m, retrieval: data } : m)); }
        if (type === "extraction") setMessages((old) => old.map((m) => m.id === assistantId ? { ...m, extraction: data.summary || data } : m));
        if (type === "warning") setEvents((old) => [...old.slice(-49), { type: "warning", message: data.message, timestamp: Date.now() }]);
        if (type === "graph") { setGraph(data); if (data.active_node_ids) setActiveNodes(data.active_node_ids); }
        if (type === "done") {
          setMessages((old) => old.map((m) => m.id === assistantId ? { ...m, ...(data.message || {}), retrieval: data.message?.retrieval || retrieval, streaming: false } : m));
          if (data.conversation) setCurrent(data.conversation);
          setPhase("complete");
        }
      });
    } catch (e) {
      if (e.name !== "AbortError") setError(e.message);
      if (!streamStarted) setDraft(text);
      setMessages((old) => old.map((m) => m.id === assistantId ? { ...m, streaming: false, status: e.name === "AbortError" ? "cancelled" : "error" } : m));
    } finally {
      if (id) {
        try {
          let fresh = await api(`/conversations/${id}`);
          for (let attempt = 0; fresh.generating && attempt < 15; attempt++) {
            await new Promise((resolve) => setTimeout(resolve, 100));
            fresh = await api(`/conversations/${id}`);
          }
          setCurrent(fresh); setGraph(fresh.graph || EMPTY_GRAPH);
          if (fresh.messages?.length) setMessages(normalizeMessages(fresh.messages));
          await refreshList();
        } catch { /* Keep streamed content when a final refresh is unavailable. */ }
      }
      busyRef.current = false; setBusy(false); abort.current = null; setPhase("idle"); textarea.current?.focus();
    }
  }
  async function stop() {
    try { if (activeId.current) await api(`/conversations/${activeId.current}/cancel`, { method: "POST", body: {} }); }
    catch (e) { setError(e.message); }
    finally { abort.current?.abort(); }
  }
  async function organizeHistory() {
    const id = activeId.current;
    if (!id || busyRef.current) return;
    if (!settings?.has_api_key) { setSettingsOpen(true); return; }
    const controller = new AbortController(); abort.current = controller;
    busyRef.current = true; setBusy(true); setError(""); setPhase("organizing"); setGraphOpen(true);
    try {
      await streamOperation(`/conversations/${encodeURIComponent(id)}/organize`, {}, controller.signal, (type, data) => {
        if (type === "status") { setPhase(data.stage || "organizing"); setEvents((old) => [...old.slice(-49), { type: data.stage, message: data.message, timestamp: Date.now() }]); }
        if (type === "warning") setEvents((old) => [...old.slice(-49), { type: "warning", message: data.message, timestamp: Date.now() }]);
        if (type === "graph") { setGraph(data); setActiveNodes(data.active_node_ids || []); }
        if (type === "done") setToast(data.summary?.status === "partial" ? "部分记忆尚未整理完成，请查看提示" : "历史记忆已整理");
      });
    } catch (e) { if (e.name !== "AbortError") setError(e.message); }
    finally {
      try {
        let fresh = await api(`/conversations/${id}`);
        for (let attempt = 0; fresh.generating && attempt < 15; attempt++) {
          await new Promise((resolve) => setTimeout(resolve, 100));
          fresh = await api(`/conversations/${id}`);
        }
        setCurrent(fresh); setMessages(normalizeMessages(fresh.messages)); setGraph(fresh.graph || EMPTY_GRAPH); await refreshList();
      } catch { /* The next conversation load retries the refresh. */ }
      busyRef.current = false; setBusy(false); abort.current = null; setPhase("idle");
    }
  }
  async function copy(message) { try { await navigator.clipboard.writeText(message.content); setCopied(message.id); setTimeout(() => setCopied(null), 1800); } catch { setToast("无法自动复制，请选择文本复制。"); } }
  async function updateConversation(event) {
    event.preventDefault();
    try {
      if (manage.kind === "delete") { await api(`/conversations/${manage.id}`, { method: "DELETE" }); if (activeId.current === manage.id) newChat(); }
      else { const updated = await api(`/conversations/${manage.id}`, { method: "PATCH", body: { title: manage.title.trim() } }); if (activeId.current === manage.id) setCurrent((old) => ({ ...old, ...updated })); }
      setManage(null); await refreshList();
    } catch (e) { setError(e.message); setManage(null); }
  }
  function exportChat() {
    const body = `# ${current?.title || "HAMGF 对话"}\n\n` + messages.map((m) => `## ${m.role === "user" ? "你" : "HAMGF"}\n\n${m.content}\n`).join("\n");
    const url = URL.createObjectURL(new Blob([body], { type: "text/markdown;charset=utf-8" }));
    const a = document.createElement("a"); a.href = url; a.download = `${(current?.title || "HAMGF-chat").replace(/[<>:"/\\|?*]/g, "_")}.md`; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  const filtered = conversations.filter((c) => c.title.toLowerCase().includes(filter.toLowerCase()));
  const currentGraph = visibleMemoryGraph(graph);
  return <div className={`studio ${sidebarOpen ? "" : "sidebar-collapsed"} ${graphOpen ? "graph-visible" : ""} ${graphExpanded ? "graph-expanded" : ""}`}>
    <aside className="activity-bar"><button className="app-logo" aria-label="HAMGF Studio 首页" onClick={newChat}><Mark size={28}/></button><div className="activity-main"><button className="activity-button active" title="会话" onClick={() => setSidebarOpen(!sidebarOpen)} aria-label="切换会话侧栏"><Icon name="chat" size={21}/></button><button className={`activity-button ${graphOpen ? "selected" : ""}`} title="记忆图谱" onClick={() => setGraphOpen(!graphOpen)} aria-label="切换记忆图谱"><Icon name="graph" size={21}/></button></div><button className="activity-button settings-activity" title="设置 · Ctrl+," aria-label="打开设置" disabled={!settings} onClick={() => setSettingsOpen(true)}><Icon name="settings" size={21}/></button><div className="profile-dot">H</div></aside>
    <aside className="conversation-sidebar"><div className="sidebar-header"><span>HAMGF <span className="studio-word">Studio</span></span><button className="icon-button" title="收起侧栏" aria-label="收起侧栏" onClick={() => setSidebarOpen(false)}><Icon name="panel" size={17}/></button></div><button className="new-chat-button" disabled={busy} onClick={newChat}><Icon name="plus" size={17}/><span>新对话</span><kbd>Ctrl N</kbd></button><label className="conversation-search"><Icon name="search" size={15}/><input ref={search} value={filter} onChange={(e) => setFilter(e.target.value)} placeholder="搜索对话" aria-label="搜索对话"/><kbd>Ctrl K</kbd></label><div className="sidebar-section-title">对话记录 <span>{conversations.length.toString().padStart(2, "0")}</span></div><nav className="conversation-list" aria-label="会话列表">{filtered.map((c) => <div className={`conversation-item ${current?.id === c.id ? "is-current" : ""}`} key={c.id}><button className="conversation-select" disabled={busy} onClick={() => openChat(c.id)}><Icon name="chat" size={15}/><span>{c.title}</span></button><div className="conversation-actions"><button className="icon-button" disabled={busy} aria-label={`重命名 ${c.title}`} onClick={() => setManage({ kind: "rename", id: c.id, title: c.title })}><Icon name="edit" size={13}/></button><button className="icon-button" disabled={busy} aria-label={`删除 ${c.title}`} onClick={() => setManage({ kind: "delete", id: c.id, title: c.title })}><Icon name="trash" size={13}/></button></div></div>)}{!filtered.length && <p className="no-conversations">{filter ? "没有匹配的对话" : "你的想法，从这里开始。"}</p>}</nav><div className="sidebar-bottom"><div className="workspace-card"><div className="workspace-icon"><Icon name="graph" size={18}/></div><div><strong>本地记忆空间</strong><span>HAMGF · 按会话独立保存</span></div><span className="connection-dot"/></div><button className="provider-status" onClick={() => settings && setSettingsOpen(true)}><span className={`connection-dot ${settings?.has_api_key ? "" : "muted"}`}/><span>{settings?.has_api_key ? "DeepSeek 已配置" : "连接 DeepSeek"}</span><Icon name="right" size={14}/></button></div></aside>
    <main className="chat-workspace"><header className="chat-header"><div className="chat-breadcrumb">{!sidebarOpen && <button className="icon-button" aria-label="展开侧栏" onClick={() => setSidebarOpen(true)}><Icon name="panel" size={17}/></button>}<Icon name="chat" size={15}/><span>工作空间</span><span className="breadcrumb-slash">/</span><strong>{current?.title || "新对话"}</strong></div><div className="chat-header-actions">{messages.length > 0 && <button className="organize-memory-button" aria-label="整理历史记忆" title="从历史消息提炼记忆（使用 DeepSeek API）" disabled={busy} onClick={organizeHistory}><Icon name="spark" size={14}/><span>整理记忆</span></button>}{messages.length > 0 && <button className="icon-button" aria-label="导出对话" title="导出 Markdown" onClick={exportChat}><Icon name="export" size={16}/></button>}<button className={`graph-toggle ${graphOpen ? "is-active" : ""}`} onClick={() => setGraphOpen(!graphOpen)} aria-expanded={graphOpen}><Icon name="graph" size={16}/><span>记忆图谱</span><span className="graph-count">{currentGraph.nodes.length}</span></button></div></header>
      <div className="conversation-scroll" ref={scrollArea} onScroll={(e) => { const t = e.currentTarget; nearBottom.current = t.scrollHeight - t.scrollTop - t.clientHeight < 100; }}>
        {messages.length === 0 ? <section className="welcome"><div className="welcome-eyebrow"><span className="connection-dot"/>A SPACE FOR CONNECTED THINKING</div><div className="welcome-mark"><Mark size={58}/></div><h1>对话，从记忆延伸。</h1><p>一个记得来路的 AI 工作空间。<br/>让想法相连，让上下文一直在场。</p><div className="welcome-suggestions">{[{ icon: "graph", title: "建立一段记忆", desc: "告诉我值得记住的事", prompt: "请记住：" }, { icon: "spark", title: "一起梳理想法", desc: "把零散的思路连接起来", prompt: "我想和你一起梳理一个想法：" }, { icon: "chat", title: "开始自由对话", desc: "从一个问题或灵感出发", prompt: "你好，介绍一下你如何使用 HAMGF 记忆来帮助我。" }].map((item) => <button className="suggestion" key={item.title} onClick={() => { setDraft(item.prompt); textarea.current?.focus(); }}><Icon name={item.icon} size={19}/><strong>{item.title}</strong><span>{item.desc}</span><Icon name="right" size={13}/></button>)}</div><div className="welcome-memory"><Icon name="graph" size={14}/><span>HAMGF 记忆已就绪</span><span className="tiny-divider"/><span>记忆图谱可随时展开</span></div></section> : <div className="messages">{messages.map((message, index) => <Message key={message.id || index} message={message} onGraph={() => setGraphOpen(true)} onCopy={copy} copied={copied}/>)}{busy && <div className="generation-status"><span className="spinner"/>{stageLabels[phase] || "正在处理"}<span>HAMGF × DeepSeek</span></div>}</div>}
      </div>
      <div className="composer-area">{error && <div className="error-banner" role="alert"><Icon name="info" size={16}/><span>{error}</span><button className="icon-button" aria-label="关闭提示" onClick={() => setError("")}><Icon name="close" size={14}/></button></div>}{!loading && settings && !settings.has_api_key && <button className="setup-banner" onClick={() => setSettingsOpen(true)}><Icon name="bolt" size={15}/><span>接入 DeepSeek，让第一段对话开始。</span><strong>设置 API Key <Icon name="right" size={13}/></strong></button>}<form className="composer" onSubmit={send}><textarea ref={textarea} rows={2} aria-label="消息输入框" placeholder={loading ? "正在加载工作空间…" : "发送消息，让想法继续生长…"} value={draft} disabled={loading} onChange={(e) => { setDraft(e.target.value); e.target.style.height = "auto"; e.target.style.height = Math.min(e.target.scrollHeight, 190) + "px"; }} onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); send(); } }}/><div className="composer-toolbar"><div><button type="button" className="model-selector" disabled={!settings || busy} onClick={() => setSettingsOpen(true)}><span className="model-symbol">D</span>{settings?.model || "DeepSeek"}<Icon name="down" size={12}/></button><span className="composer-divider"/><span className="memory-indicator"><Icon name="graph" size={14}/>HAMGF 记忆</span></div>{busy ? <button type="button" className="send-button stop-button" onClick={stop} aria-label="停止生成"><Icon name="stop" size={17}/></button> : <button className="send-button" disabled={!draft.trim() || loading} aria-label="发送消息"><Icon name="arrow" size={18}/></button>}</div></form><div className="composer-caption"><span>Enter 发送 <span>·</span> Shift + Enter 换行</span><span>记忆有迹可循</span></div></div>
    </main>
    {graphOpen && <aside className="memory-rail"><MemoryGraph key={current?.id || "empty"} graph={graph} activeNodeIds={activeNodes} phase={phase} events={events} onClose={() => { setGraphOpen(false); setGraphExpanded(false); }} expanded={graphExpanded} onExpand={() => setGraphExpanded(!graphExpanded)}/></aside>}
    <footer className="status-bar"><div><span className={`connection-dot ${error ? "muted" : ""}`}/><span>{loading ? "正在加载" : "本地工作空间"}</span><span className="status-separator">/</span><span>HAMGF Engine</span></div><div><Icon name="graph" size={12}/><span>{currentGraph.nodes.length} 节点</span><span>{currentGraph.edges.length} 关联</span><span className="status-separator">/</span><span>{busy ? stageLabels[phase] || "处理中" : "就绪"}</span></div></footer>
    {settingsOpen && settings && <Settings settings={settings} onClose={() => setSettingsOpen(false)} onSaved={(value) => { setSettings(value); setToast("设置已保存"); }}/>} 
    {manage && <div className="modal-backdrop" onClick={() => setManage(null)}><form className="confirm-dialog" role="dialog" aria-modal="true" aria-label={manage.kind === "delete" ? "删除对话" : "重命名对话"} onClick={(e) => e.stopPropagation()} onSubmit={updateConversation}><h2>{manage.kind === "delete" ? "删除这段对话？" : "重命名对话"}</h2>{manage.kind === "delete" ? <p>“{manage.title}”及其记忆图谱将从本机删除。</p> : <input autoFocus aria-label="对话名称" required maxLength={120} value={manage.title} onChange={(e) => setManage({ ...manage, title: e.target.value })}/>}<footer><button type="button" className="button secondary" onClick={() => setManage(null)}>取消</button><button className={`button ${manage.kind === "delete" ? "danger" : "primary"}`} disabled={busy || !manage.title.trim()}>{manage.kind === "delete" ? "删除" : "保存"}</button></footer></form></div>}
    {toast && <div className="toast" role="status"><Icon name="check" size={15}/>{toast}</div>}
  </div>;
}
