from __future__ import annotations

import argparse
import hmac
import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from hamgf.api.service import APIValidationError, MemoryApplication, seed_demo_graph
from hamgf.core.validation import SchemaValidationError


class MemoryHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        application: MemoryApplication,
        *,
        cors_origin: str = "http://127.0.0.1:5173",
        api_token: str | None = None,
    ) -> None:
        self.application = application
        self.cors_origin = cors_origin
        self.api_token = api_token
        super().__init__(server_address, MemoryRequestHandler)


class MemoryRequestHandler(BaseHTTPRequestHandler):
    server: MemoryHTTPServer
    protocol_version = "HTTP/1.1"
    max_body_bytes = 2 * 1024 * 1024

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self._cors_headers()
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PATCH, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        self._dispatch("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch("POST")

    def do_PATCH(self) -> None:  # noqa: N802
        self._dispatch("PATCH")

    def _dispatch(self, method: str) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path).rstrip("/") or "/"
        query = parse_qs(parsed.query)
        if path.startswith("/v1/") and not self._authorized():
            self._json(
                HTTPStatus.UNAUTHORIZED,
                {"error": "unauthorized", "message": "valid bearer token required"},
                extra_headers={"WWW-Authenticate": "Bearer"},
            )
            return
        try:
            if method == "GET" and path == "/health":
                self._json(HTTPStatus.OK, self.server.application.health())
            elif method == "GET" and path == "/v1/graph":
                self._json(HTTPStatus.OK, self.server.application.graph_view())
            elif method == "GET" and path == "/v1/snapshot":
                self._json(HTTPStatus.OK, self.server.application.snapshot())
            elif method == "GET" and path == "/v1/events":
                raw = query.get("since", ["0"])[0]
                try:
                    since = int(raw)
                except ValueError as exc:
                    raise APIValidationError("since must be a non-negative integer") from exc
                self._json(HTTPStatus.OK, self.server.application.events_since(since))
            elif method == "GET" and path.startswith("/v1/nodes/"):
                node_id = path.removeprefix("/v1/nodes/")
                self._json(HTTPStatus.OK, self.server.application.node_view(node_id))
            elif method == "GET" and path == "/v1/audit":
                node_id = query.get("node_id", [None])[0]
                self._json(HTTPStatus.OK, self.server.application.audit(node_id))
            elif method == "POST" and path == "/v1/memories":
                self._json(HTTPStatus.CREATED, self.server.application.write_memory(self._body()))
            elif method == "POST" and path == "/v1/search":
                self._json(HTTPStatus.OK, self.server.application.search(self._body()))
            elif method == "PATCH" and path == "/v1/edges":
                self._json(HTTPStatus.OK, self.server.application.update_edge(self._body()))
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "route_not_found", "path": path})
        except KeyError as exc:
            self._json(HTTPStatus.NOT_FOUND, {"error": "not_found", "message": str(exc)})
        except (APIValidationError, SchemaValidationError, TypeError, ValueError) as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid_request", "message": str(exc)})
        except json.JSONDecodeError as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid_json", "message": str(exc)})

    def _body(self) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length", "0")
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise APIValidationError("invalid Content-Length") from exc
        if length <= 0:
            raise APIValidationError("JSON request body is required")
        if length > self.max_body_bytes:
            raise APIValidationError("request body exceeds 2 MiB")
        payload = json.loads(self.rfile.read(length))
        if not isinstance(payload, dict):
            raise APIValidationError("JSON body must be an object")
        return payload

    def _authorized(self) -> bool:
        expected = self.server.api_token
        if expected is None:
            return True
        scheme, separator, supplied = self.headers.get("Authorization", "").partition(" ")
        return bool(
            separator
            and scheme.casefold() == "bearer"
            and supplied
            and hmac.compare_digest(supplied, expected)
        )

    def _json(
        self,
        status: HTTPStatus,
        payload: Any,
        *,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        for name, value in (extra_headers or {}).items():
            self.send_header(name, value)
        self._cors_headers()
        self.end_headers()
        self.wfile.write(encoded)

    def _cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", self.server.cors_origin)
        self.send_header("Vary", "Origin")

    def log_message(self, format: str, *args: Any) -> None:
        return


def create_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    *,
    application: MemoryApplication | None = None,
    cors_origin: str = "http://127.0.0.1:5173",
    api_token: str | None = None,
) -> MemoryHTTPServer:
    return MemoryHTTPServer(
        (host, port), application or MemoryApplication(), cors_origin=cors_origin,
        api_token=api_token,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the HAMGF Phase 3 REST API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--cors-origin", default="http://127.0.0.1:5173")
    parser.add_argument("--demo", action="store_true", help="seed the canonical CMG demo")
    parser.add_argument(
        "--encrypted-snapshot",
        action="store_true",
        help="encrypt --snapshot with AES-256-GCM using HAMGF_SNAPSHOT_KEY",
    )
    parser.add_argument(
        "--require-auth",
        action="store_true",
        help="protect /v1/* with the bearer token in HAMGF_API_TOKEN",
    )
    parser.add_argument(
        "--neo4j",
        action="store_true",
        help="restore/synchronize the CMG using HAMGF_NEO4J_* environment variables",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    snapshot_store = None
    if args.encrypted_snapshot:
        if args.snapshot is None:
            raise SystemExit("--encrypted-snapshot requires --snapshot")
        from hamgf.persistence import EncryptedSnapshotStore

        snapshot_store = EncryptedSnapshotStore.from_env()
    api_token = None
    if args.require_auth:
        api_token = os.environ.get("HAMGF_API_TOKEN")
        if not api_token:
            raise SystemExit("--require-auth requires HAMGF_API_TOKEN")
    persistence = None
    restored_graph = None
    restored_pool_state = None
    if args.neo4j:
        from hamgf.persistence import Neo4jGraphStore

        persistence = Neo4jGraphStore.from_env()
        persistence.verify_connectivity()
        restored_state = persistence.pull_state()
        restored_graph = restored_state.graph
        restored_pool_state = restored_state.pool_state
    application = MemoryApplication(
        restored_graph if restored_graph is not None and len(restored_graph) else None,
        snapshot_path=args.snapshot,
        persistence=persistence,
        snapshot_store=snapshot_store,
        pool_state=restored_pool_state,
    )
    if persistence is not None and restored_graph is not None and not len(restored_graph):
        application.sync_persistence()
    if args.demo:
        seed_demo_graph(application)
    server = create_server(
        args.host,
        args.port,
        application=application,
        cors_origin=args.cors_origin,
        api_token=api_token,
    )
    print(f"HAMGF API listening on http://{args.host}:{server.server_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if persistence is not None:
            persistence.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
