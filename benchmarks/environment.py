"""Small, provider-neutral runtime metadata helper."""

from __future__ import annotations

import os
import platform
from pathlib import Path


def runtime_metadata(**details: str) -> dict[str, str | bool]:
    metadata: dict[str, str | bool] = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "container": Path("/.dockerenv").exists(),
        "ci": bool(os.environ.get("CI")),
    }
    metadata.update(details)
    return metadata
