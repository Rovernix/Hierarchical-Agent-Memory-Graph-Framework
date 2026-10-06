const ROOT = "/api/studio";

export async function api(path, options = {}) {
  const response = await fetch(`${ROOT}${path}`, {
    ...options,
    headers: { Accept: "application/json", ...(options.body ? { "Content-Type": "application/json" } : {}), ...options.headers },
    body: options.body ? JSON.stringify(options.body) : undefined,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.message || payload.error || `请求失败 (${response.status})`);
  return payload;
}

// Consume complete SSE frames, even when network chunks split UTF-8 characters.
export async function consumeEvents(body, onEvent) {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let pending = "";
  function flush(final = false) {
    pending = pending.replace(/\r\n/g, "\n");
    let boundary;
    while ((boundary = pending.indexOf("\n\n")) !== -1 || (final && pending.trim())) {
      const frame = boundary === -1 ? pending : pending.slice(0, boundary);
      pending = boundary === -1 ? "" : pending.slice(boundary + 2);
      let event = "message";
      const data = [];
      for (const line of frame.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        if (line.startsWith("data:")) data.push(line.slice(5).replace(/^ /, ""));
      }
      if (data.length && data.join("\n") !== "[DONE]") onEvent(event, JSON.parse(data.join("\n")));
    }
  }
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      pending += decoder.decode(value, { stream: true });
      flush();
    }
    pending += decoder.decode();
    flush(true);
  } catch (error) {
    await reader.cancel().catch(() => {});
    throw error;
  } finally { reader.releaseLock(); }
}

export async function streamOperation(path, payload, signal, onEvent) {
  const response = await fetch(`${ROOT}${path}`, {
    method: "POST", headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify(payload), signal,
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.message || error.error || `请求失败 (${response.status})`);
  }
  let completed = false;
  await consumeEvents(response.body, (type, data) => {
    if (type === "done") completed = true;
    if (type === "error") throw new Error(data.message || "生成中断，请重试。");
    onEvent(type, data);
  });
  if (!completed) throw new Error("连接中断，本次操作可能未完成。");
}

export function streamMessage(id, content, signal, onEvent) {
  return streamOperation(`/conversations/${encodeURIComponent(id)}/messages`, { content }, signal, onEvent);
}
