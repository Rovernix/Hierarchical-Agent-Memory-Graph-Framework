import { useEffect, useRef, useState } from "react";
import { api } from "./api.js";
import { Icon, Mark } from "./Icons.jsx";

export function Settings({ settings, onClose, onSaved }) {
  const [form, setForm] = useState({ ...settings, api_key: "" });
  const [showKey, setShowKey] = useState(false);
  const [busy, setBusy] = useState("");
  const [notice, setNotice] = useState(null);
  const [models, setModels] = useState([]);
  const dialog = useRef(null);
  useEffect(() => {
    dialog.current?.showModal();
    const modal = dialog.current;
    return () => modal?.close();
  }, []);
  const update = (key, value) => setForm((old) => ({ ...old, [key]: value }));
  function payload() {
    const { base_url, model, temperature, max_tokens, retrieval_k, system_prompt, thinking, llm_relation_review, api_key, clear_api_key } = form;
    return { base_url, model, temperature: Number(temperature), max_tokens: Number(max_tokens), retrieval_k: Number(retrieval_k), system_prompt, thinking, llm_relation_review: !!llm_relation_review, ...(api_key ? { api_key } : {}), ...(clear_api_key ? { clear_api_key } : {}) };
  }
  async function action(kind) {
    setBusy(kind); setNotice(null);
    try {
      if (kind === "test") {
        const result = await api("/test-connection", { method: "POST", body: payload() });
        setModels((result.models || []).map((m) => typeof m === "string" ? m : m.id));
        setNotice({ ok: result.ok, message: result.ok ? `连接成功 · ${result.latency_ms ?? "—"} ms` : result.message });
      } else {
        const result = await api("/settings", { method: "PUT", body: payload() });
        onSaved(result); onClose();
      }
    } catch (error) { setNotice({ ok: false, message: error.message }); }
    finally { setBusy(""); }
  }
  return <dialog ref={dialog} className="settings-dialog" aria-labelledby="settings-title" onCancel={(e) => { e.preventDefault(); onClose(); }} onClick={(e) => { if (e.target === dialog.current) onClose(); }}>
    <div className="settings-shell">
      <div className="settings-nav"><div className="settings-brand"><Mark size={27}/><span>HAMGF<span>STUDIO</span></span></div><p>偏好设置</p><div className="settings-nav-item"><Icon name="bolt" size={16}/>模型与连接</div><div className="settings-note"><Icon name="graph" size={19}/><span>让每次对话<br/>成为下一次的起点。</span></div></div>
      <form className="settings-content" onSubmit={(e) => { e.preventDefault(); action("save"); }}>
        <header><div><span className="overline">MODEL PROVIDER</span><h2 id="settings-title">连接你的 DeepSeek</h2><p>为记忆赋予语言，为对话保留上下文。</p></div><button type="button" className="icon-button" aria-label="关闭设置" onClick={onClose}><Icon name="close"/></button></header>
        <div className="settings-scroll">
          <div className="provider-card"><span className="provider-symbol">D</span><div><strong>DeepSeek</strong><span>OpenAI-compatible API</span></div><span className="subtle-badge">API</span></div>
          <label className="field-label">API Key<span>{settings.has_api_key ? `已保存 ${settings.api_key_hint || ""}` : "尚未配置"}</span></label>
          <div className="secret-input"><input aria-label="API Key" type={showKey ? "text" : "password"} value={form.api_key} onChange={(e) => { update("api_key", e.target.value); update("clear_api_key", false); }} placeholder={settings.has_api_key ? "留空以保留已保存的密钥" : "sk-…"} autoComplete="off" spellCheck="false"/><button type="button" className="icon-button" onClick={() => setShowKey(!showKey)} aria-label={showKey ? "隐藏密钥" : "显示密钥"}><Icon name="eye" size={17}/></button></div>
          <p className="field-help">密钥使用 Windows 用户级加密，仅保存在本机。</p>
          {settings.has_api_key && <label className="checkbox-field"><input type="checkbox" checked={!!form.clear_api_key} onChange={(e) => update("clear_api_key", e.target.checked)}/>清除已保存的密钥</label>}
          <label className="field-label" htmlFor="base-url">API 地址</label><input id="base-url" type="url" required value={form.base_url || ""} onChange={(e) => update("base_url", e.target.value)} placeholder="https://api.deepseek.com" spellCheck="false"/>
          <div className="field-row"><div><label className="field-label" htmlFor="model">模型</label><input id="model" list="model-options" required value={form.model || ""} onChange={(e) => update("model", e.target.value)} spellCheck="false"/><datalist id="model-options">{[...new Set(["deepseek-flash", "deepseek-v4-pro", ...models])].map((m) => <option key={m} value={m}/>)}</datalist></div><div><label className="field-label" htmlFor="thinking">深度思考</label><select id="thinking" value={form.thinking || "disabled"} onChange={(e) => update("thinking", e.target.value)}><option value="disabled">关闭</option><option value="enabled">开启</option></select></div></div>
          <label className="checkbox-field relation-review-option"><input type="checkbox" checked={!!form.llm_relation_review} onChange={(e) => update("llm_relation_review", e.target.checked)}/>LLM 辅助关系校验</label>
          <p className="field-help">规则无法确认时，让当前模型复核关系含义与方向。仅凭有据结果连线；每轮最多增加一次 API 请求和相应用量。默认关闭。</p>
          <details className="advanced-settings"><summary>生成与记忆 <Icon name="down" size={14}/></summary><div className="field-row"><div><label className="field-label" htmlFor="temperature">Temperature</label><input id="temperature" type="number" min="0" max="2" step="0.1" value={form.temperature ?? 0.7} onChange={(e) => update("temperature", e.target.value)}/></div><div><label className="field-label" htmlFor="tokens">最大输出 tokens</label><input id="tokens" type="number" min="256" max="65536" step="1" value={form.max_tokens ?? 4096} onChange={(e) => update("max_tokens", e.target.value)}/></div></div><label className="field-label" htmlFor="recall">每轮检索记忆数</label><input id="recall" type="number" min="1" max="30" value={form.retrieval_k ?? 6} onChange={(e) => update("retrieval_k", e.target.value)}/><label className="field-label" htmlFor="system-prompt">系统提示词</label><textarea id="system-prompt" value={form.system_prompt || ""} onChange={(e) => update("system_prompt", e.target.value)} rows={4}/></details>
          {notice && <p role="status" className={`notice ${notice.ok ? "success" : "error"}`}><Icon name={notice.ok ? "check" : "info"} size={16}/>{notice.message}</p>}
        </div>
        <footer><button type="button" className="button secondary" disabled={!!busy} onClick={() => action("test")}><span className={busy === "test" ? "spinner" : "connection-dot"}/>{busy === "test" ? "正在连接…" : "测试连接"}</button><button className="button primary" disabled={!!busy}>{busy === "save" ? "保存中…" : "保存设置"}<Icon name="check" size={15}/></button></footer>
      </form>
    </div>
  </dialog>;
}
