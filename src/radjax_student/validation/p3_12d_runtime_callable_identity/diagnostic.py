"""Actual callable-bound observation for the P3.12D adversarial matrix."""

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
    """An actual public boundary plus independently digestible mutation evidence."""

    callable: Callable[..., Any]
    args: tuple[Any, ...] = ()
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    baseline_input: Any = None
    mutated_input: Any = None

    def __post_init__(self) -> None:
        if not callable(self.callable):
            raise TypeError("invocation callable must be callable")
        object.__setattr__(self, "args", tuple(self.args))
        object.__setattr__(self, "kwargs", dict(self.kwargs))


@dataclass(frozen=True)
class Observation:
    boundary: str
    exception_type: str | None
    code: str | None
    evidence_digest: str


def observe(invocation: Invocation) -> Observation:
    """Observe only an invocation and its actual exception.

    Expected metadata is unavailable at this boundary.
    """
    boundary = f"{invocation.callable.__module__}.{invocation.callable.__qualname__}"
    try:
        result = invocation.callable(*invocation.args, **invocation.kwargs)
    except Exception as error:
        code = getattr(error, "code", None)
        return Observation(
            boundary,
            type(error).__name__,
            code if isinstance(code, str) else None,
            _digest({"boundary": boundary, "type": type(error).__name__, "code": code}),
        )
    execution_result = (
        result[1] if isinstance(result, tuple) and len(result) == 2 else None
    )
    blockers = getattr(execution_result, "blockers", ())
    if getattr(execution_result, "status", None) == "fail" and blockers:
        first = blockers[0]
        code = getattr(first, "code", None)
        return Observation(
            boundary,
            type(execution_result).__name__,
            code if isinstance(code, str) else None,
            _digest(
                {
                    "boundary": boundary,
                    "type": type(execution_result).__name__,
                    "code": code,
                }
            ),
        )
    return Observation(boundary, None, None, _digest({"boundary": boundary}))
