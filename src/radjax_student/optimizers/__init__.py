"""Architecture-independent optimizer contracts and a test-only SGD backend."""

from radjax_student.optimizers.errors import (
    OPTIMIZER_ERROR_CODES,
    OptimizerContractError,
    OptimizerErrorCode,
    OptimizerIssue,
)
from radjax_student.optimizers.jax import (
    JaxOptimizerState,
    advanced_jax_optimizer_state,
    require_finite_jax_gradients,
    validate_jax_optimizer_state,
)
from radjax_student.optimizers.models import (
    GRADIENT_CLIP_MODES,
    OPTIMIZER_CLAIMS_NOT_MADE,
    OPTIMIZER_CONFIG_SCHEMA_VERSION,
    OPTIMIZER_STATE_ROLES,
    OPTIMIZER_STATE_SCHEMA_VERSION,
    WEIGHT_DECAY_MODES,
    GradientTree,
    OptimizerCapabilityProfile,
    OptimizerConfig,
    OptimizerInitRequest,
    OptimizerInitResult,
    OptimizerState,
    OptimizerStateDescriptor,
    OptimizerUpdateRequest,
    OptimizerUpdateResult,
    ParameterUpdate,
    canonical_optimizer_json,
)
from radjax_student.optimizers.protocols import JaxOptimizerExecution, OptimizerBackend
from radjax_student.optimizers.registry import OptimizerRegistry
from radjax_student.optimizers.sgd import SGD_OPTIMIZER_ID, SgdOptimizer

__all__ = [
    "GRADIENT_CLIP_MODES",
    "OPTIMIZER_CLAIMS_NOT_MADE",
    "OPTIMIZER_CONFIG_SCHEMA_VERSION",
    "OPTIMIZER_ERROR_CODES",
    "OPTIMIZER_STATE_ROLES",
    "OPTIMIZER_STATE_SCHEMA_VERSION",
    "SGD_OPTIMIZER_ID",
    "WEIGHT_DECAY_MODES",
    "GradientTree",
    "JaxOptimizerExecution",
    "JaxOptimizerState",
    "OptimizerBackend",
    "OptimizerCapabilityProfile",
    "OptimizerConfig",
    "OptimizerContractError",
    "OptimizerErrorCode",
    "OptimizerInitRequest",
    "OptimizerInitResult",
    "OptimizerIssue",
    "OptimizerRegistry",
    "OptimizerState",
    "OptimizerStateDescriptor",
    "OptimizerUpdateRequest",
    "OptimizerUpdateResult",
    "ParameterUpdate",
    "SgdOptimizer",
    "advanced_jax_optimizer_state",
    "canonical_optimizer_json",
    "require_finite_jax_gradients",
    "validate_jax_optimizer_state",
]
