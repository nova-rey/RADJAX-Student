"""JAX-free static exports for the RWKV-7 reference plugin."""

from radjax_student.architecture.rwkv7_reference.config import (
    RWKV7_REFERENCE_ARCHITECTURE_ID,
    RWKV7_REFERENCE_ARCHITECTURE_VERSION,
    RWKV7ReferenceConfig,
    configurable_architecture_config,
    language_identities,
    reference_architecture_config,
    validate_reference_config,
    validate_rwkv7_config,
)
from radjax_student.architecture.rwkv7_reference.plugin import RWKV7ReferencePlugin
from radjax_student.architecture.rwkv7_reference.registration import (
    register_rwkv7_reference,
)
from radjax_student.architecture.rwkv7_reference.schema import (
    architecture_metadata,
    carry_descriptor,
    hf_descriptor,
    initialization_parameter_slots,
    parameter_catalog,
    parameter_layout,
    parameter_layout_for_vocabulary,
    pinned_numpy_parameter_order,
)

__all__ = [
    "RWKV7_REFERENCE_ARCHITECTURE_ID",
    "RWKV7_REFERENCE_ARCHITECTURE_VERSION",
    "RWKV7ReferenceConfig",
    "RWKV7ReferencePlugin",
    "configurable_architecture_config",
    "language_identities",
    "architecture_metadata",
    "carry_descriptor",
    "hf_descriptor",
    "initialization_parameter_slots",
    "parameter_catalog",
    "parameter_layout",
    "parameter_layout_for_vocabulary",
    "pinned_numpy_parameter_order",
    "reference_architecture_config",
    "register_rwkv7_reference",
    "validate_reference_config",
    "validate_rwkv7_config",
]
