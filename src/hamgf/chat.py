from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Sequence
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener

from hamgf.adapters import LLMBackendError, OpenAICompatibleBackend
from hamgf.agent import AgentConfig, MemoryGroundedAgent
from hamgf.api import MemoryApplication, create_server


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SNAPSHOT = Path(
    os.path.join(PROJECT_ROOT, "data", "snapshots", "phase4_chat.json")
)


class FrontendDevServer:

    def __init__(
        self,
        frontend_dir: Path,
        *,
        host: str,
        port: int,
        api_url: str,
        open_browser: bool = True,
        startup_timeout: float = 20.0,
    ) -> None:
        self.frontend_dir = frontend_dir
        self.host = host
        self.port = port
        self.api_url = api_url
        self.open_browser = open_browser
        self.startup_timeout = startup_timeout
        self.process: subprocess.Popen[bytes] | None = None

    @property
    def browser_host(self) -> str:
        return "127.0.0.1" if self.host in {"0.0.0.0", "::"} else self.host

    @property
    def url(self) -> str:
        return f"http://{self.browser_host}:{self.port}"

    @property
    def command(self) -> tuple[str, ...]:
        return (
            "npm",
            "run",
            "dev",
            "--",
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--strictPort",
        )

    def start(self) -> str:
        if self.process is not None:
            raise RuntimeError("frontend process is already running")
        if shutil.which("npm") is None:
            raise RuntimeError("npm is required to start the graph visualization")
        if not Path(os.path.join(self.frontend_dir, "package.json")).exists():
            raise RuntimeError(f"frontend project not found: {self.frontend_dir}")
        if not Path(os.path.join(self.frontend_dir, "node_modules")).exists():
            raise RuntimeError("frontend dependencies are missing; run 'cd frontend && npm ci'")

        environment = os.environ.copy()
        environment["VITE_HAMGF_API_URL"] = self.api_url
        self.process = subprocess.Popen(
            self.command,
            cwd=self.frontend_dir,
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            _wait_for_http(self.url, process=self.process, timeout=self.startup_timeout)
        except Exception:
            self.stop()
            raise
        if self.open_browser:
            webbrowser.open_new_tab(self.url)
        return self.url

    def open(self) -> bool:
        return webbrowser.open_new_tab(self.url)

    def stop(self) -> None:
        process = self.process
        self.process = None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def _wait_for_http(
    url: str,
    *,
    process: subprocess.Popen[bytes] | None = None,
    timeout: float = 20.0,
) -> None:
    deadline = time.monotonic() + timeout
    opener = build_opener(ProxyHandler({}))
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError("graph visualization process exited during startup")
        try:
            with opener.open(Request(url, method="GET"), timeout=0.5):
                return
        except (URLError, TimeoutError):
            time.sleep(0.1)
    raise RuntimeError(f"graph visualization did not become ready within {timeout:g}s: {url}")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a memory-grounded chat session with live HAMGF graph visualization."
    )
    parser.add_argument(
        "--llm-base-url",
        default=os.environ.get("HAMGF_LLM_BASE_URL", "http://127.0.0.1:11434/v1"),
        help="OpenAI-compatible API base URL (default: local Ollama compatibility endpoint)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("HAMGF_LLM_MODEL"),
        help="model name; may also be set with HAMGF_LLM_MODEL",
    )
    parser.add_argument(
        "--api-key-env",
        default="HAMGF_LLM_API_KEY",
        help="environment variable containing the optional LLM API key",
    )
    parser.add_argument("--llm-timeout", type=float, default=120.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--retrieval-k", type=int, default=6)
    parser.add_argument("--session-id")
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--api-host", default="127.0.0.1")
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument("--frontend-host", default="127.0.0.1")
    parser.add_argument("--frontend-port", type=int, default=5173)
    parser.add_argument(
        "--frontend-dir",
        type=Path,
        default=Path(os.path.join(PROJECT_ROOT, "frontend")),
    )
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--no-frontend", action="store_true")
    return parser.parse_args(argv)


def _advertised_host(host: str) -> str:
    return "127.0.0.1" if host in {"0.0.0.0", "::"} else host


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.model:
        print(
            "缺少模型名称：请使用 --model，或设置 HAMGF_LLM_MODEL。",
            file=sys.stderr,
        )
        return 2

    api_host = _advertised_host(args.api_host)
    graph_origin = f"http://{_advertised_host(args.frontend_host)}:{args.frontend_port}"
    args.snapshot.parent.mkdir(parents=True, exist_ok=True)
    application = MemoryApplication(snapshot_path=args.snapshot)
    api_server = create_server(
        args.api_host,
        args.api_port,
        application=application,
        cors_origin=graph_origin,
    )
    api_thread = threading.Thread(
        target=api_server.serve_forever,
        name="hamgf-api",
        daemon=True,
    )
    frontend = None
    try:
        api_thread.start()
        actual_api_url = f"http://{api_host}:{api_server.server_port}"
        graph_url = None
        if not args.no_frontend:
            frontend = FrontendDevServer(
                args.frontend_dir.resolve(),
                host=args.frontend_host,
                port=args.frontend_port,
                api_url=actual_api_url,
                open_browser=not args.no_browser,
            )
            graph_url = frontend.start()

        backend = OpenAICompatibleBackend(
            args.llm_base_url,
            args.model,
            api_key=os.environ.get(args.api_key_env),
            timeout=args.llm_timeout,
        )
        agent = MemoryGroundedAgent(
            application,
            backend,
            config=AgentConfig(
                retrieval_k=args.retrieval_k,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            ),
            session_id=args.session_id,
        )
        print("HAMGF Phase 4 推理对话已启动。")
        print(f"API: {actual_api_url}")
        print(f"Snapshot: {args.snapshot.resolve()}")
        if graph_url:
            print(f"Graph: {graph_url}")
        else:
            print("Graph: 未启动（--no-frontend）")
        print(f"Session: {agent.session_id}")
        print("命令：/graph 打开图页，/stats 查看图状态，/quit 退出。")

        while True:
            try:
                query = input("\n你 > ").strip()
            except EOFError:
                print()
                break
            if not query:
                continue
            if query in {"/quit", "/exit", "quit", "exit"}:
                break
            if query == "/graph":
                if frontend is None:
                    print("图页面未启动。")
                elif not frontend.open():
                    print(f"当前环境无法自动拉起浏览器，请手动访问：{frontend.url}")
                continue
            if query == "/stats":
                health = application.health()
                print(
                    f"节点 {health['nodes']} · 边 {health['edges']} · revision {health['revision']}"
                )
                continue
            try:
                turn = agent.ask(query)
            except LLMBackendError as exc:
                suffix = f" (HTTP {exc.status})" if exc.status else ""
                print(f"模型调用失败{suffix}：{exc}")
                continue
            except (RuntimeError, ValueError) as exc:
                print(f"推理失败：{exc}")
                continue
            print(f"\nAgent > {turn.grounded_answer}")
            print(
                f"[检索 {turn.retrieval_ms:.1f}ms · 推理 {turn.inference_ms:.1f}ms · "
                f"写图 {turn.write_ms:.1f}ms · revision {application.revision}]"
            )
            print(f"[已同步节点：{turn.user_node_id} → {turn.assistant_node_id}]")
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"启动失败：{exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n已停止。")
    finally:
        if frontend is not None:
            frontend.stop()
        api_server.shutdown()
        api_server.server_close()
        if api_thread.is_alive():
            api_thread.join(timeout=5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
