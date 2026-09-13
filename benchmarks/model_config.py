"""Safe loader for the root benchmark model configuration Markdown file."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class ApiModelConfig:
    key: str
    base_url: str
    api_key_env: str
    model: str
    extra_body: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class LocalModelConfig:
    key: str
    path: Path


@dataclass(frozen=True, slots=True)
class BenchmarkModelConfig:
    api_models: Mapping[str, ApiModelConfig]
    local_models: Mapping[str, LocalModelConfig]


def load_model_config(path: str | Path) -> BenchmarkModelConfig:
    """Parse literal dict assignments without executing the Markdown code."""

    source_path = Path(path).resolve()
    markdown = source_path.read_text(encoding="utf-8")
    blocks = re.findall(r"```python\s*(.*?)```", markdown, flags=re.DOTALL | re.IGNORECASE)
    if not blocks:
        raise ValueError("model config does not contain a Python code block")
    assignments: dict[str, Any] = {}
    for block in blocks:
        tree = ast.parse(block, filename=str(source_path))
        for node in tree.body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in {"API_MODELS", "LOCAL_MODELS"}:
                assignments[target.id] = ast.literal_eval(node.value)
    api_raw = _mapping(assignments.get("API_MODELS"), "API_MODELS")
    local_raw = _mapping(assignments.get("LOCAL_MODELS"), "LOCAL_MODELS")
    api_models = {
        str(key): _api_model(str(key), _mapping(value, f"API_MODELS[{key!r}]"))
        for key, value in api_raw.items()
    }
    local_models = {
        str(key): _local_model(
            str(key),
            _mapping(value, f"LOCAL_MODELS[{key!r}]"),
            source_path.parent,
        )
        for key, value in local_raw.items()
    }
    if not api_models and not local_models:
        raise ValueError("model config contains no models")
    return BenchmarkModelConfig(api_models=api_models, local_models=local_models)


def _api_model(key: str, data: Mapping[str, Any]) -> ApiModelConfig:
    base_url = _text(data.get("base_url"), f"{key}.base_url")
    if not base_url.startswith(("http://", "https://")):
        raise ValueError(f"{key}.base_url must be HTTP(S)")
    extra_body = data.get("extra_body", {})
    if not isinstance(extra_body, Mapping):
        raise ValueError(f"{key}.extra_body must be a mapping")
    return ApiModelConfig(
        key=key,
        base_url=base_url,
        api_key_env=_text(data.get("api_key_env"), f"{key}.api_key_env"),
        model=_text(data.get("model"), f"{key}.model"),
        extra_body=dict(extra_body),
    )


def _local_model(key: str, data: Mapping[str, Any], root: Path) -> LocalModelConfig:
    configured = Path(_text(data.get("path"), f"{key}.path"))
    resolved = configured if configured.is_absolute() else root / configured
    return LocalModelConfig(key=key, path=resolved.resolve())


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()
