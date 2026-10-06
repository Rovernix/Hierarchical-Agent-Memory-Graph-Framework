"""Run the synced Studio service using the unmodified parent HAMGF core."""
from __future__ import annotations

import _thread
import sys
import threading
from pathlib import Path

frontend = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(frontend.parent / "src"))
sys.path.insert(0, str(frontend))

from backend.studio import main

args = sys.argv[1:]
if "--watch-stdin" in args:
    args.remove("--watch-stdin")

    def watch_owner() -> None:
        sys.stdin.buffer.read()
        _thread.interrupt_main()

    threading.Thread(target=watch_owner, daemon=True).start()

raise SystemExit(main(["--frontend-dir", str(frontend / "dist"), *args]))
