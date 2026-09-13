from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, Path(os.path.join(ROOT, 'src'))):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from benchmarks.model_config import load_model_config
from hamgf.adapters.llm import LLMBackendError, OpenAICompatibleBackend


def probe_model(
    config_path: Path,
    model_key: str,
    *,
    timeout: float = 30.0,
    max_tokens: int = 1,
    backend_factory: Callable[..., Any] = OpenAICompatibleBackend,
) -> dict[str, Any]:
    """Send one minimal completion and return a credential-redacted result."""

    started_at = datetime.now(timezone.utc).isoformat()
    api_key: str | None = None
    try:
        config = load_model_config(config_path)
        if model_key not in config.api_models:
            raise ValueError(f"configured API model does not exist: {model_key}")
        model = config.api_models[model_key]
        api_key = os.environ.get(model.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"required environment variable is not set: {model.api_key_env}"
            )
        backend = backend_factory(
            model.base_url,
            model.model,
            api_key=api_key,
            extra_body=model.extra_body,
            timeout=timeout,
        )
        backend.complete(
            "Reply with OK.",
            system_prompt="Return only OK.",
            max_tokens=max_tokens,
            temperature=0.0,
        )
        return {
            "checked_at": started_at,
            "model_key": model_key,
            "model": model.model,
            "status": "passed",
            "http_status": 200,
            "max_tokens": max_tokens,
        }
    except LLMBackendError as error:
        return {
            "checked_at": started_at,
            "model_key": model_key,
            "status": "failed",
            "http_status": error.status,
            "error_type": type(error).__name__,
            "error": _redact(str(error), api_key),
            "max_tokens": max_tokens,
        }
    except Exception as error:
        return {
            "checked_at": started_at,
            "model_key": model_key,
            "status": "failed",
            "http_status": None,
            "error_type": type(error).__name__,
            "error": _redact(str(error), api_key),
            "max_tokens": max_tokens,
        }


def _redact(message: str, secret: str | None) -> str:
    if secret and len(secret) >= 6:
        return message.replace(secret, "[REDACTED]")
    return message


def write_result(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(os.path.join(ROOT, 'Config.md')))
    parser.add_argument("--model", default="deepseek")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.path.join(ROOT, 'tests', 'sol', 'phase4-api-preflight', 'deepseek.json')),
    )
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    result = probe_model(
        args.config,
        args.model,
        timeout=args.timeout,
        max_tokens=args.max_tokens,
    )
    write_result(args.output, result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
