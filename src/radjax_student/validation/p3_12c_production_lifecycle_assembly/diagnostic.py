"""Actual-observation primitives for the P3.12C literal adversarial matrix."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any


def _digest(value: object) -> str:
    return hashlib.sha256(
        (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()


@dataclass(frozen=True)
class Invocation:
    callable: Callable[..., Any]
    args: tuple[Any, ...] = ()
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    baseline_input: Any = None
    mutated_input: Any = None

    def __post_init__(self) -> None:
        if not callable(self.callable):
            raise TypeError("invocation callable must be callable")
        object.__setattr__(self, "args", tuple(self.args))
        if not isinstance(self.kwargs, Mapping):
            raise TypeError("invocation kwargs must be a mapping")
        object.__setattr__(self, "kwargs", dict(self.kwargs))


@dataclass(frozen=True)
class Observation:
    boundary: str
    exception_type: str | None
    code: str | None
    evidence_digest: str


def observe(invocation: Invocation) -> Observation:
    """Observe only actual callable execution; expected metadata is unavailable."""
    callable_identity = (
        f"{invocation.callable.__module__}.{invocation.callable.__qualname__}"
    )
    try:
        invocation.callable(*invocation.args, **invocation.kwargs)
    except Exception as error:
        code = getattr(error, "code", None)
        return Observation(
            callable_identity,
            type(error).__name__,
            code if isinstance(code, str) else None,
            _digest(
                {
                    "boundary": callable_identity,
                    "type": type(error).__name__,
                    "code": code,
                }
            ),
        )
    return Observation(
        callable_identity, None, None, _digest({"boundary": callable_identity})
    )
