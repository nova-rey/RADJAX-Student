from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, TypeVar, runtime_checkable

from radjax_student.runtime.models import (
    CompilationOptions,
    ExecutionContext,
    PlacementPolicy,
    RuntimeCapabilityProfile,
    RuntimeConfig,
    RuntimeEnvironment,
)

ValueT = TypeVar("ValueT")


@runtime_checkable
class RuntimeBackend(Protocol):
    """Architecture-independent backend contract; no backend is selected here."""

    @property
    def backend_id(self) -> str: ...

    def inspect_environment(self) -> RuntimeEnvironment: ...

    def capability_profile(self) -> RuntimeCapabilityProfile: ...

    def initialize(self, config: RuntimeConfig) -> ExecutionContext: ...

    def place(self, value: ValueT, placement: PlacementPolicy) -> Any: ...

    def compile(
        self,
        function: Callable[..., Any],
        options: CompilationOptions,
    ) -> Callable[..., Any]: ...

    def synchronize(self, value: ValueT) -> ValueT: ...

    def close(self, context: ExecutionContext) -> None: ...
