# Benchmark model configuration example

Copy this file to `Config.md` locally. Replace the intentionally invalid endpoint and model aliases with values supplied by your provider. Export the named environment variables; never put API key values in this file. The loader uses literal Python dictionaries, not executable imports or environment lookups. This file does not authorize remote data transfer or initiate a paid run.

```python
API_MODELS = {
    "deepseek": {
        "base_url": "https://example.invalid/v1",
        "api_key_env": "HAMGF_DEEPSEEK_API_KEY",
        "model": "deepseek-v4-flash",
    },
    "gemini": {
        "base_url": "https://example.invalid/v1",
        "api_key_env": "HAMGF_GEMINI_API_KEY",
        "model": "gemini-3.1-flash-lite",
    },
    "gpt": {
        "base_url": "https://example.invalid/v1",
        "api_key_env": "HAMGF_JUDGE_API_KEY",
        "model": "gpt-5.5",
    },
}
LOCAL_MODELS = {
    "qwen3.6-27b": {"path": "models/Qwen3.6"},
}
```

Qwen is the local reader in the experiment; download authorized weights separately and adjust its path. DeepSeek and Gemini are API readers; the `gpt` entry is the judge. DeepSeek also serves framework extraction. Embedding configuration is separate in the preparation CLI: check `python scripts/prepare_memory_baselines.py --help` and configure the named embedding key/endpoint before running. See the [baseline protocol and PrefEval notes](docs/memory-baselines.md).
