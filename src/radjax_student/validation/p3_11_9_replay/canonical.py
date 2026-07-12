"""JAX-free canonical encoders for replay evidence."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any

from radjax_student.checkpoints.npz_codec import mapping_pytree_digest


class ReplayCanonicalError(ValueError):
    """Raised when replay evidence cannot be canonically represented."""


def finite_float_hex(value: Any) -> str:
    """Encode one finite scalar without decimal-rendering ambiguity."""

    if isinstance(value, bool):
        raise ReplayCanonicalError("replay scalar must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ReplayCanonicalError("replay scalar must be a finite number") from exc
    if not math.isfinite(result):
        raise ReplayCanonicalError("replay scalar must be finite")
    return result.hex()


def canonical_json_bytes(value: Any) -> bytes:
    """Encode an already JSON-safe value with one stable representation."""

    try:
        return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
    except (TypeError, ValueError) as exc:
        raise ReplayCanonicalError("replay value is not canonical JSON") from exc


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def parse_canonical_json(data: bytes | str) -> Any:
    """Parse JSON while rejecting duplicate object fields."""

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ReplayCanonicalError(f"duplicate replay field: {key}")
            result[key] = value
        return result

    try:
        return json.loads(data, object_pairs_hook=object_pairs)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        if isinstance(exc, ReplayCanonicalError):
            raise
        raise ReplayCanonicalError("invalid replay JSON") from exc


def canonical_metric_mapping(values: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(values, Mapping):
        raise ReplayCanonicalError("replay metrics must be a mapping")
    result: dict[str, str] = {}
    for name, value in values.items():
        if not isinstance(name, str) or not name:
            raise ReplayCanonicalError("replay metric name must be nonempty")
        result[name] = finite_float_hex(value)
    return {name: result[name] for name in sorted(result)}


__all__ = [
    "ReplayCanonicalError",
    "canonical_digest",
    "canonical_json_bytes",
    "canonical_metric_mapping",
    "finite_float_hex",
    "mapping_pytree_digest",
    "parse_canonical_json",
]
