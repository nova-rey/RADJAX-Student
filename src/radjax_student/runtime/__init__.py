"""Architecture-independent runtime contracts.

P2.1 defines serializable intent, observation, capability, context, error,
protocol, state-envelope, and reporting concepts. It does not select or execute
a backend.
"""

from radjax_student.runtime.errors import (
    RUNTIME_ERROR_CODES,
    RuntimeContractError,
    RuntimeErrorCode,
    RuntimeIssue,
)
from radjax_student.runtime.models import (
    COMPILATION_POLICIES,
    DISTRIBUTED_POLICIES,
    FALLBACK_POLICIES,
    PLACEMENT_POLICIES,
    PRECISION_POLICIES,
    RUNTIME_CAPABILITY_VOCABULARY,
    CompilationOptions,
    CompilationPolicy,
    DeviceDescriptor,
    DeviceInventory,
    DistributedPolicy,
    ExecutionContext,
    FallbackPolicy,
    PlacementPolicy,
    PrecisionPolicy,
    RuntimeCapabilityProfile,
    RuntimeConfig,
    RuntimeEnvironment,
    RuntimeState,
    RuntimeStatus,
)
from radjax_student.runtime.protocols import RuntimeBackend
from radjax_student.runtime.reports import RuntimeReport

__all__ = [
    "COMPILATION_POLICIES",
    "DISTRIBUTED_POLICIES",
    "FALLBACK_POLICIES",
    "PLACEMENT_POLICIES",
    "PRECISION_POLICIES",
    "RUNTIME_CAPABILITY_VOCABULARY",
    "RUNTIME_ERROR_CODES",
    "CompilationOptions",
    "CompilationPolicy",
    "DeviceDescriptor",
    "DeviceInventory",
    "DistributedPolicy",
    "ExecutionContext",
    "FallbackPolicy",
    "PlacementPolicy",
    "PrecisionPolicy",
    "RuntimeBackend",
    "RuntimeCapabilityProfile",
    "RuntimeConfig",
    "RuntimeContractError",
    "RuntimeEnvironment",
    "RuntimeErrorCode",
    "RuntimeIssue",
    "RuntimeReport",
    "RuntimeState",
    "RuntimeStatus",
]
