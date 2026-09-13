from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from enum import Enum
from typing import TypeVar


class SchemaValidationError(ValueError):
    """Raised when a node or edge violates the public schema."""


EnumT = TypeVar("EnumT", bound=Enum)


def coerce_enum(value: EnumT | str, enum_type: type[EnumT], field: str) -> EnumT:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in enum_type)
        raise SchemaValidationError(f"{field} must be one of: {allowed}") from exc


def score(value: float, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaValidationError(f"{field} must be a number")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise SchemaValidationError(f"{field} must be within [0, 1]")
    return result


def non_negative(value: float, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaValidationError(f"{field} must be a number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise SchemaValidationError(f"{field} must be non-negative")
    return result


def timestamp(value: str | datetime | None, field: str, *, nullable: bool = False) -> str | None:
    if value is None:
        if nullable:
            return None
        return datetime.now(timezone.utc).isoformat()
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise SchemaValidationError(f"{field} must be an ISO-8601 timestamp") from exc
    else:
        raise SchemaValidationError(f"{field} must be an ISO-8601 timestamp")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise SchemaValidationError(f"{field} must include a timezone")
    return parsed.isoformat()


def ensure_json(value: object, field: str) -> None:
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise SchemaValidationError(f"{field} must be JSON serializable") from exc

