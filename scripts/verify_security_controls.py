from __future__ import annotations
import os
import json
import stat
import sys
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(Path(os.path.join(PROJECT_ROOT, 'src')))]

from hamgf.api import MemoryApplication, create_server
from hamgf.core.graph import ChainMemoryGraph
from hamgf.core.nodes import MemoryNode
from hamgf.persistence import EncryptedSnapshotError, EncryptedSnapshotStore, graph_digest


def _status(url: str, token: str | None = None) -> int:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    request = Request(url, headers=headers)
    try:
        with build_opener(ProxyHandler({})).open(request, timeout=3) as response:
            return response.status
    except HTTPError as exc:
        return exc.code


def verify() -> dict:
    controls: dict[str, bool] = {}
    with tempfile.TemporaryDirectory() as directory:
        path = Path(os.path.join(Path(directory), 'sensitive.enc'))
        graph = ChainMemoryGraph()
        marker = "SENSITIVE-SNAPSHOT-MARKER"
        graph.add_node(MemoryNode.create(marker, node_id="M-SECURITY-VERIFY"))
        store = EncryptedSnapshotStore(bytes(range(32)))
        store.save(graph, path)
        encrypted = path.read_bytes()
        restored = store.load(path)
        controls["plaintext_absent"] = marker.encode() not in encrypted
        controls["owner_only_permissions"] = stat.S_IMODE(path.stat().st_mode) == 0o600
        controls["lossless_round_trip"] = graph_digest(restored) == graph_digest(graph)
        try:
            EncryptedSnapshotStore(bytes(reversed(range(32)))).load(path)
        except EncryptedSnapshotError:
            controls["wrong_key_rejected"] = True
        else:
            controls["wrong_key_rejected"] = False
        tampered = bytearray(encrypted)
        tampered[-1] ^= 1
        path.write_bytes(tampered)
        try:
            store.load(path)
        except EncryptedSnapshotError:
            controls["tamper_rejected"] = True
        else:
            controls["tamper_rejected"] = False

    token = "ephemeral-security-verification-token"
    server = create_server("127.0.0.1", 0, application=MemoryApplication(), api_token=token)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        controls["health_public"] = _status(f"{base}/health") == 200
        controls["missing_token_rejected"] = _status(f"{base}/v1/graph") == 401
        controls["wrong_token_rejected"] = _status(f"{base}/v1/graph", "wrong") == 401
        controls["valid_token_accepted"] = _status(f"{base}/v1/graph", token) == 200
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)

    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": "AES-256-GCM + HTTP Bearer",
        "controls": controls,
        "passed_controls": sum(controls.values()),
        "total_controls": len(controls),
        "all_passed": all(controls.values()),
        "secrets_exported": False,
        "limitations": [
            "Bearer authentication is optional and must be enabled explicitly.",
            "Snapshot encryption does not encrypt Neo4j volume blocks; use host storage encryption for that volume.",
            "TLS termination is an external deployment responsibility when leaving loopback interfaces.",
        ],
    }


def export(result: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (Path(os.path.join(output, 'result.json'))).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [name.replace("_", " ") for name in result["controls"]]
    values = [100 if passed else 0 for passed in result["controls"].values()]
    with plt.rc_context({
        "font.family": "DejaVu Serif", "font.size": 8, "axes.titlepad": 22,
        "pdf.fonttype": 42, "savefig.dpi": 300, "axes.spines.top": False,
        "axes.spines.right": False,
    }):
        figure, axis = plt.subplots(figsize=(8.4, 4.6))
        figure.subplots_adjust(left=.10, right=.98, top=.82, bottom=.34)
        axis.bar(labels, values, color=".55", edgecolor="black", linewidth=.6)
        axis.set_ylim(0, 100)
        axis.set_ylabel("Control result (%)")
        axis.set_title("Authenticated snapshot and API security controls")
        axis.tick_params(axis="x", rotation=35)
        for label in axis.get_xticklabels():
            label.set_ha("right")
        axis.grid(axis="y", linestyle=":", alpha=.35)
        axis.set_axisbelow(True)
        for extension in ("pdf", "svg", "png"):
            figure.savefig(Path(os.path.join(output, f'controls.{extension}')), bbox_inches="tight")
        plt.close(figure)


def main() -> int:
    destination = Path(os.path.join(PROJECT_ROOT, 'tests', 'sol', 'security-validation'))
    result = verify()
    export(result, destination)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
