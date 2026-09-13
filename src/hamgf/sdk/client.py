from __future__ import annotations

import json
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import ProxyHandler, Request, build_opener


class HamgfSDKError(RuntimeError):
    def __init__(self, message: str, *, status: int | None = None, payload: Any = None) -> None:
        super().__init__(message)
        self.status = status
        self.payload = payload


class HamgfClient:
    """
    Example Usage:

        client = HamgfClient("http://127.0.0.1:8000")
        client.write_memory("客户确认方案 B", importance=0.6767, timeliness=0.9) 恭喜你又发现了一个67彩蛋
        chain = client.search("方案 B")
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        *,
        timeout: float = 10.0,
        api_token: str | None = None,
    ) -> None:
        if not isinstance(base_url, str) or not base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if api_token is not None and (not isinstance(api_token, str) or not api_token):
            raise ValueError("api_token must be a non-empty string or None")
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self.api_token = api_token
        hostname = urlparse(self.base_url).hostname
        self._opener = build_opener(ProxyHandler({})) if hostname in {"127.0.0.1", "localhost", "::1"} else build_opener()

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def graph(self) -> dict[str, Any]:
        return self._request("GET", "/v1/graph")

    def snapshot(self) -> dict[str, Any]:
        return self._request("GET", "/v1/snapshot")

    def events(self, *, since: int = 0) -> dict[str, Any]:
        return self._request("GET", "/v1/events", query={"since": since})

    def get_node(self, node_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/nodes/{node_id}")

    def write_memory(self, content: str, **options: Any) -> dict[str, Any]:
        return self._request("POST", "/v1/memories", {"content": content, **options})

    def search(
        self,
        query: str,
        *,
        k: int = 5,
        include_pending_edges: bool = True,
        include_superseded: bool = False,
        query_embedding: list[float] | tuple[float, ...] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "query": query,
            "k": k,
            "include_pending_edges": include_pending_edges,
            "include_superseded": include_superseded,
        }
        if query_embedding is not None:
            payload["query_embedding"] = list(query_embedding)
        return self._request("POST", "/v1/search", payload)

    def audit(self, node_id: str | None = None) -> dict[str, Any]:
        query = {"node_id": node_id} if node_id is not None else None
        return self._request("GET", "/v1/audit", query=query)

    def update_edge(
        self,
        source: str,
        target: str,
        key: str | int,
        **changes: Any,
    ) -> dict[str, Any]:
        return self._request(
            "PATCH",
            "/v1/edges",
            {"source": source, "target": target, "key": key, **changes},
        )

    def _request(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None = None,
        *,
        query: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        if query:
            url = f"{url}?{urlencode(query)}"
        data = None
        headers = {"Accept": "application/json"}
        if self.api_token is not None:
            headers["Authorization"] = f"Bearer {self.api_token}"
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
            headers["Content-Type"] = "application/json; charset=utf-8"
        request = Request(url, data=data, method=method, headers=headers)
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                raw = response.read()
        except HTTPError as exc:
            raw = exc.read()
            try:
                error_payload = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError):
                error_payload = {"message": raw.decode("utf-8", errors="replace")}
            message = error_payload.get("message") or error_payload.get("error") or str(exc)
            raise HamgfSDKError(message, status=exc.code, payload=error_payload) from exc
        except URLError as exc:
            raise HamgfSDKError(f"unable to reach HAMGF API: {exc.reason}") from exc
        try:
            decoded = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise HamgfSDKError("HAMGF API returned invalid JSON") from exc
        if not isinstance(decoded, dict):
            raise HamgfSDKError("HAMGF API returned a non-object response", payload=decoded)
        return decoded
