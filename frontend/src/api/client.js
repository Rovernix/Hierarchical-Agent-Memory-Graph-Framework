export class HamgfApiError extends Error {
  constructor(message, status, payload) {
    super(message);
    this.name = "HamgfApiError";
    this.status = status;
    this.payload = payload;
  }
}

async function request(baseUrl, path, options = {}) {
  const response = await fetch(`${baseUrl.replace(/\/$/, "")}${path}`, {
    headers: { Accept: "application/json", ...(options.body ? { "Content-Type": "application/json" } : {}) },
    ...options,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new HamgfApiError(payload.message || payload.error || `HTTP ${response.status}`, response.status, payload);
  }
  return payload;
}

export function createHamgfApi(baseUrl) {
  return {
    graph: () => request(baseUrl, "/v1/graph"),
    events: (since = 0) => request(baseUrl, `/v1/events?since=${encodeURIComponent(since)}`),
    node: (nodeId) => request(baseUrl, `/v1/nodes/${encodeURIComponent(nodeId)}`),
    audit: (nodeId) => request(baseUrl, `/v1/audit${nodeId ? `?node_id=${encodeURIComponent(nodeId)}` : ""}`),
    search: (query, options = {}) =>
      request(baseUrl, "/v1/search", {
        method: "POST",
        body: JSON.stringify({ query, k: 8, include_superseded: true, ...options }),
      }),
    updateEdge: (payload) =>
      request(baseUrl, "/v1/edges", { method: "PATCH", body: JSON.stringify(payload) }),
  };
}
