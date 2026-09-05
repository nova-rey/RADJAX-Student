"""JAX-free static exports for the Mamba-2 reference plugin."""

from radjax_student.architecture.mamba2_reference.config import (
    MAMBA2_REFERENCE_ARCHITECTURE_ID,
    MAMBA2_REFERENCE_ARCHITECTURE_VERSION,
    Mamba2EquationShape,
    Mamba2ReferenceConfig,
    configurable_architecture_config,
    equation_shape,
    language_identities,
    reference_architecture_config,
    reference_config,
    validate_mamba2_config,
    validate_reference_config,
)
from radjax_student.architecture.mamba2_reference.plugin import Mamba2ReferencePlugin
from radjax_student.architecture.mamba2_reference.registration import (
    register_mamba2_reference,
)
from radjax_student.architecture.mamba2_reference.schema import (
    architecture_metadata,
    carry_descriptor,
    hf_descriptor,
    initialization_parameter_slots,
    parameter_catalog,
    parameter_layout,
    parameter_layout_for_vocabulary,
)

__all__ = [
    "MAMBA2_REFERENCE_ARCHITECTURE_ID",
    "MAMBA2_REFERENCE_ARCHITECTURE_VERSION",
    "Mamba2EquationShape",
    "Mamba2ReferenceConfig",
    "Mamba2ReferencePlugin",
    "architecture_metadata",
    "carry_descriptor",
    "configurable_architecture_config",
    "equation_shape",
    "hf_descriptor",
    "initialization_parameter_slots",
    "language_identities",
    "parameter_catalog",
    "parameter_layout",
    "parameter_layout_for_vocabulary",
    "reference_architecture_config",
    "reference_config",
    "register_mamba2_reference",
    "validate_mamba2_config",
    "validate_reference_config",
]
