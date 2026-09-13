from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import ProxyHandler, Request, build_opener


class LLMBackendError(RuntimeError):
    """Structured failure raised by an LLM backend adapter."""

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        payload: Any = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.payload = payload


class OpenAICompatibleBackend:
    """Call an OpenAI-compatible chat-completions endpoint.

    The adapter intentionally uses only the Python standard library. It works
    with local servers such as vLLM, llama.cpp or an Ollama compatibility
    endpoint, as well as hosted services that expose the same JSON contract.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        extra_body: Mapping[str, Any] | None = None,
        max_tokens_field: str | None = None,
        timeout: float = 120.0,
        stream: bool = False,
    ) -> None:
        if not isinstance(base_url, str) or not base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        if not isinstance(timeout, (int, float)) or isinstance(timeout, bool):
            raise ValueError("timeout must be a positive finite number")
        if not math.isfinite(float(timeout)) or timeout <= 0:
            raise ValueError("timeout must be a positive finite number")
        if extra_body is not None and not isinstance(extra_body, Mapping):
            raise ValueError("extra_body must be a mapping")
        if extra_body and {"model", "messages"}.intersection(extra_body):
            raise ValueError("extra_body cannot replace model or messages")
        if api_key is not None and not isinstance(api_key, str):
            raise ValueError("api_key must be a string or None")
        normalized_api_key = api_key.strip() if api_key is not None else None
        if normalized_api_key and any(char in normalized_api_key for char in "\r\n"):
            raise ValueError("api_key cannot contain line breaks")
        self.base_url = base_url.rstrip("/")
        self.model = model.strip()
        self.api_key = normalized_api_key or None
        self.extra_body = dict(extra_body or {})
        self.stream = bool(stream)
        self.last_timing: dict[str, Any] = {}
        hostname = urlparse(self.base_url).hostname
        if max_tokens_field is None:
            max_tokens_field = (
                "max_completion_tokens"
                if hostname == "api.openai.com" and self.model.startswith(("gpt-5", "o"))
                else "max_tokens"
            )
        if max_tokens_field not in {"max_tokens", "max_completion_tokens"}:
            raise ValueError("max_tokens_field is not supported")
        self.max_tokens_field = max_tokens_field
        self.timeout = float(timeout)
        self.last_usage: dict[str, Any] = {}
        self._opener = (
            build_opener(ProxyHandler({}))
            if hostname in {"127.0.0.1", "localhost", "::1"}
            else build_opener()
        )

    @property
    def endpoint(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    def complete(self, prompt: str, **options: Any) -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        system_prompt = options.pop(
            "system_prompt",
            "你是一个严格依据已提供证据回答问题的智能体。",
        )
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ValueError("system_prompt must be a non-empty string")

        temperature = options.pop("temperature", 0.0)
        max_tokens = options.pop("max_tokens", None)
        stream = bool(options.pop("stream", self.stream))
        extra_body = options.pop("extra_body", {})
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        }
        if self.max_tokens_field != "max_completion_tokens":
            payload["temperature"] = float(temperature)
        if max_tokens is not None:
            if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
                raise ValueError("max_tokens must be a positive integer")
            payload[self.max_tokens_field] = max_tokens
        if stream:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
        if not isinstance(extra_body, Mapping):
            raise ValueError("extra_body must be a mapping")
        protected = {"model", "messages"}
        if protected.intersection(extra_body):
            raise ValueError("extra_body cannot replace model or messages")
        merged_extra = dict(self.extra_body)
        merged_extra.update(extra_body)
        payload.update(merged_extra)
        payload.update(options)

        try:
            encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ValueError("LLM request options must be JSON serializable") from exc
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json; charset=utf-8",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(self.endpoint, data=encoded, method="POST", headers=headers)
        self.last_usage = {}
        self.last_timing = {}
        started = time.perf_counter()
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                if stream:
                    return self._read_stream(response, started)
                raw = response.read()
        except HTTPError as exc:
            raw = exc.read()
            error_payload = self._decode_error(raw)
            message = self._error_message(error_payload) or str(exc)
            raise LLMBackendError(message, status=exc.code, payload=error_payload) from exc
        except (URLError, TimeoutError) as exc:
            reason = getattr(exc, "reason", exc)
            raise LLMBackendError(f"unable to reach LLM backend: {reason}") from exc
        response_ms = (time.perf_counter() - started) * 1_000

        try:
            decoded = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise LLMBackendError("LLM backend returned invalid JSON") from exc
        if not isinstance(decoded, dict):
            raise LLMBackendError("LLM backend returned a non-object response", payload=decoded)
        self.last_usage = dict(decoded.get("usage") or {})
        self.last_timing = {
            "mode": "non_streaming",
            "ttft_ms": None,
            "response_ms": response_ms,
            "ttft_definition": "unavailable_without_visible_content_stream",
        }
        try:
            content = decoded["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMBackendError(
                "LLM response does not contain choices[0].message.content",
                payload=decoded,
            ) from exc
        text = self._content_text(content)
        if not text:
            raise LLMBackendError("LLM backend returned empty content", payload=decoded)
        return text

    def _read_stream(self, response: Any, started: float) -> str:
        """Consume OpenAI-compatible SSE and measure first visible content."""

        parts: list[str] = []
        usage: dict[str, Any] = {}
        ttft_ms: float | None = None
        last_payload: Any = None
        saw_data = False
        for raw_line in response:
            try:
                line = raw_line.decode("utf-8").strip()
            except UnicodeDecodeError as exc:
                raise LLMBackendError("LLM stream returned invalid UTF-8") from exc
            if not line or line.startswith(":"):
                continue
            if not line.startswith("data:"):
                raise LLMBackendError(
                    "LLM streaming response is not an SSE data stream"
                )
            data = line[5:].strip()
            if data == "[DONE]":
                break
            saw_data = True
            try:
                payload = json.loads(data)
            except json.JSONDecodeError as exc:
                raise LLMBackendError("LLM stream returned invalid JSON") from exc
            last_payload = payload
            if not isinstance(payload, Mapping):
                raise LLMBackendError(
                    "LLM stream returned a non-object event", payload=payload
                )
            if isinstance(payload.get("usage"), Mapping):
                usage = dict(payload["usage"])
            choices = payload.get("choices")
            if not isinstance(choices, list) or not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta") if isinstance(choice, Mapping) else None
            content = self._content_delta(
                delta.get("content") if isinstance(delta, Mapping) else None
            )
            if content:
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - started) * 1_000
                parts.append(content)
        response_ms = (time.perf_counter() - started) * 1_000
        self.last_usage = usage
        self.last_timing = {
            "mode": "streaming",
            "ttft_ms": ttft_ms,
            "response_ms": response_ms,
            "ttft_definition": "request_start_to_first_nonempty_visible_content_delta",
        }
        text = "".join(parts).strip()
        if not saw_data:
            raise LLMBackendError("LLM stream returned no data events")
        if not text:
            raise LLMBackendError(
                "LLM backend returned empty content", payload=last_payload
            )
        return text

    @staticmethod
    def _content_delta(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                str(item.get("text"))
                for item in content
                if isinstance(item, Mapping) and isinstance(item.get("text"), str)
            )
        return ""

    @staticmethod
    def _content_text(content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, Mapping) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            return "\n".join(parts).strip()
        return ""

    @staticmethod
    def _decode_error(raw: bytes) -> Any:
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {"message": raw.decode("utf-8", errors="replace")}

    @staticmethod
    def _error_message(payload: Any) -> str | None:
        if not isinstance(payload, Mapping):
            return None
        error = payload.get("error")
        if isinstance(error, Mapping):
            value = error.get("message")
            return value if isinstance(value, str) else None
        if isinstance(error, str):
            return error
        message = payload.get("message")
        return message if isinstance(message, str) else None


class TransformersLocalBackend:

    def __init__(
        self,
        model_path: str | Path,
        *,
        model_name: str | None = None,
        device_map: str | None = None,
        stream: bool = False
    ) -> None:
        path = Path(model_path).expanduser().resolve()
        if not path.is_dir():
            raise ValueError(f"local model directory does not exist: {path}")
        self.model_path = path
        self.model = model_name or path.name
        self.device_map = device_map
        self.stream = bool(stream)
        self.last_usage: dict[str, Any] = {}
        self.last_timing: dict[str, Any] = {}
        self._tokenizer: Any = None
        self._model: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import (
                AutoConfig,
                AutoModelForCausalLM,
                AutoModelForImageTextToText,
                AutoTokenizer,
            )
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise LLMBackendError(
                "local inference requires the optional transformers package"
            ) from exc
        try:
            config = AutoConfig.from_pretrained(
                self.model_path,
                local_files_only=True,
                trust_remote_code=False
            )
            architectures = tuple(getattr(config, "architectures", ()) or ())
            model_class = (
                AutoModelForImageTextToText
                if any("ConditionalGeneration" in value for value in architectures)
                else AutoModelForCausalLM
            )
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                local_files_only=True,
                trust_remote_code=False
            )
            load_options: dict[str, Any] = {
                "config": config,
                "local_files_only": True,
                "trust_remote_code": False,
                "dtype": "auto"
            }
            if self.device_map is not None:
                load_options["device_map"] = self.device_map
                
            self._model = model_class.from_pretrained(self.model_path, **load_options)
            if self.device_map is None:
                import torch

                self._model.to("cuda" if torch.cuda.is_available() else "cpu")
            self._model.eval()
        except Exception as exc:  # pragma: no cover - model/environment dependent
            self.close()
            raise LLMBackendError(f"unable to load local model {self.model}: {exc}") from exc

    def prepare(self) -> None:
        self._load()

    def complete(self, prompt: str, **options: Any) -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        self.last_usage = {}
        self.last_timing = {}
        self._load()

        import torch

        system_prompt = options.pop(
            "system_prompt",
            "你是一个严格依据已提供证据回答问题的智能体。",
        )
        temperature = float(options.pop("temperature", 0.0))
        max_tokens = options.pop("max_tokens", 1024)
        stream = bool(options.pop("stream", self.stream))
        if options:
            raise ValueError(
                f"unsupported local inference options: {', '.join(sorted(options))}"
            )
        started = time.perf_counter()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        try:
            inputs = self._tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                enable_thinking=False,
            )
        except (TypeError, ValueError):
            messages = [
                {"role": "user", "content": f"{system_prompt}\n\n{prompt}"}
            ]
            inputs = self._tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
        inputs = inputs.to(self._model.device)
        generate_options: dict[str, Any] = {
            "max_new_tokens": int(max_tokens),
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            generate_options["temperature"] = temperature
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        if stream:
            return self._complete_streaming(
                inputs,
                generate_options,
                prompt_tokens=prompt_tokens,
                started=started,
            )
        with torch.inference_mode():
            generated = self._model.generate(**inputs, **generate_options)
        response_ms = (time.perf_counter() - started) * 1_000
        completion = generated[0][prompt_tokens:]
        completion_tokens = int(completion.shape[-1])
        self.last_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        self.last_timing = {
            "mode": "local_non_streaming",
            "ttft_ms": None,
            "response_ms": response_ms,
            "ttft_definition": "unavailable_without_visible_content_stream",
        }
        text = self._tokenizer.decode(completion, skip_special_tokens=True).strip()
        if not text:
            raise LLMBackendError("local model returned empty content")
        return text

    def _complete_streaming(
        self,
        inputs: Any,
        generate_options: Mapping[str, Any],
        *,
        prompt_tokens: int,
        started: float
    ) -> str:

        import torch
        from transformers import TextIteratorStreamer

        streamer = TextIteratorStreamer(
            self._tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )
        generated_box: list[Any] = []
        error_box: list[BaseException] = []

        def generate() -> None:
            try:
                with torch.inference_mode():
                    generated_box.append(
                        self._model.generate(
                            **inputs,
                            **dict(generate_options),
                            streamer=streamer
                        )
                    )
            except BaseException as error:
                error_box.append(error)
                streamer.end()

        worker = threading.Thread(target=generate, daemon=True)
        worker.start()
        parts: list[str] = []
        ttft_ms: float | None = None
        for piece in streamer:
            if not isinstance(piece, str):
                continue
            parts.append(piece)
            if ttft_ms is None and piece.strip():
                ttft_ms = (time.perf_counter() - started) * 1_000
        worker.join()
        response_ms = (time.perf_counter() - started) * 1_000
        if error_box:
            error = error_box[0]
            raise LLMBackendError(
                f"local model generation failed: {error}"
            ) from error
        if not generated_box:
            raise LLMBackendError("local model generation returned no token sequence")
        completion = generated_box[0][0][prompt_tokens:]
        completion_tokens = int(completion.shape[-1])
        self.last_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        self.last_timing = {
            "mode": "local_streaming",
            "ttft_ms": ttft_ms,
            "response_ms": response_ms,
            "ttft_definition": "request_start_to_first_nonempty_visible_content_delta",
        }
        text = "".join(parts).strip()
        if not text:
            raise LLMBackendError("local model returned empty content")
        return text

    def close(self) -> None:
        self._model = None
        self._tokenizer = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
