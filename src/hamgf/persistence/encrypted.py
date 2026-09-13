
from __future__ import annotations

import base64
import json
import os
import tempfile
from pathlib import Path
from typing import Mapping, Any

from hamgf.core.graph import ChainMemoryGraph


class EncryptedSnapshotError(ValueError):
    """Raised when an encrypted snapshot or its key is invalid."""


class EncryptedSnapshotStore:
    """Persist a CMG snapshot with AES-256-GCM and atomic file replacement."""

    MAGIC = b"HAMGF-AESGCM-1\x00"
    NONCE_BYTES = 12
    KEY_BYTES = 32

    def __init__(self, key: bytes) -> None:
        if not isinstance(key, bytes) or len(key) != self.KEY_BYTES:
            raise EncryptedSnapshotError("snapshot key must decode to exactly 32 bytes")
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        except ImportError as exc:  # pragma: no cover - depends on optional install
            raise RuntimeError(
                "encrypted snapshots require the 'security' optional dependency"
            ) from exc
        self._cipher = AESGCM(key)

    @classmethod
    def from_encoded_key(cls, value: str) -> "EncryptedSnapshotStore":
        if not isinstance(value, str) or not value.strip():
            raise EncryptedSnapshotError("encoded snapshot key must be non-empty")
        try:
            key = base64.b64decode(value.strip().encode("ascii"), altchars=b"-_", validate=True)
        except (ValueError, UnicodeEncodeError) as exc:
            raise EncryptedSnapshotError("snapshot key must be URL-safe base64") from exc
        return cls(key)

    @classmethod
    def from_env(cls, name: str = "HAMGF_SNAPSHOT_KEY") -> "EncryptedSnapshotStore":
        value = os.environ.get(name)
        if not value:
            raise EncryptedSnapshotError(f"required snapshot key environment variable is unset: {name}")
        return cls.from_encoded_key(value)

    @staticmethod
    def generate_key() -> str:
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        except ImportError as exc:  # pragma: no cover - depends on optional install
            raise RuntimeError(
                "key generation requires the 'security' optional dependency"
            ) from exc
        return base64.urlsafe_b64encode(AESGCM.generate_key(bit_length=256)).decode("ascii")

    def save(self, graph: ChainMemoryGraph, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        plaintext = json.dumps(
            graph.to_node_link_data(), ensure_ascii=False, allow_nan=False,
            sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
        nonce = os.urandom(self.NONCE_BYTES)
        ciphertext = self._cipher.encrypt(nonce, plaintext, self.MAGIC)
        payload = self.MAGIC + nonce + ciphertext
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        temporary = Path(temporary_name)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as handle:
                descriptor = -1
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
            os.chmod(destination, 0o600)
        except Exception:
            if descriptor >= 0:
                os.close(descriptor)
            temporary.unlink(missing_ok=True)
            raise
        return destination

    def load(self, path: str | Path) -> ChainMemoryGraph:
        source = Path(path)
        payload = source.read_bytes()
        minimum = len(self.MAGIC) + self.NONCE_BYTES + 16
        if len(payload) < minimum or not payload.startswith(self.MAGIC):
            raise EncryptedSnapshotError("invalid encrypted snapshot envelope")
        offset = len(self.MAGIC)
        nonce = payload[offset : offset + self.NONCE_BYTES]
        ciphertext = payload[offset + self.NONCE_BYTES :]
        try:
            plaintext = self._cipher.decrypt(nonce, ciphertext, self.MAGIC)
        except Exception as exc:
            try:
                from cryptography.exceptions import InvalidTag
            except ImportError:  # pragma: no cover
                InvalidTag = ()  # type: ignore[assignment,misc]
            if isinstance(exc, InvalidTag):
                raise EncryptedSnapshotError(
                    "encrypted snapshot authentication failed"
                ) from exc
            raise
        try:
            data: Mapping[str, Any] = json.loads(plaintext)
        except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as exc:
            raise EncryptedSnapshotError("encrypted snapshot payload is invalid") from exc
        if not isinstance(data, dict):
            raise EncryptedSnapshotError("encrypted snapshot payload must be an object")
        return ChainMemoryGraph.from_node_link_data(data)
