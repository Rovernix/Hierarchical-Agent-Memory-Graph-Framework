from __future__ import annotations
import argparse
import contextlib
import json
import os
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(os.path.join(Path(__file__).resolve().parents[1], 'src'))))
os.environ["MEM0_TELEMETRY"] = "False"
os.environ["GRAPHITI_TELEMETRY_ENABLED"] = "false"
from hamgf.adapters.memory_frameworks import run_framework


class RedactedStream:
    def __init__(self, stream):
        self.stream = stream
        self.secrets = [v for k, v in os.environ.items() if v and len(v) >= 6 and
                        (k.endswith("_API_KEY") or k.endswith("_PASSWORD"))]

    def write(self, text):
        for secret in self.secrets:
            text = text.replace(secret, "[REDACTED]")
        return self.stream.write(text)

    def flush(self):
        return self.stream.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("strategy", choices=("mem0", "graphiti", "memos", "memobase"))
    args = parser.parse_args()
    request = json.load(sys.stdin)
    # Third-party prints must not corrupt the machine-readable result channel.
    with contextlib.redirect_stdout(sys.stderr):
        result = run_framework(args.strategy, request)
    print(json.dumps(result, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.stderr = RedactedStream(sys.stderr)
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException:
        # Some framework clients leave non-daemon retry/executor threads alive
        # after raising. This process is intentionally one invocation, so emit
        # the diagnostic and terminate the isolated worker without hanging the
        # resumable orchestrator.
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
