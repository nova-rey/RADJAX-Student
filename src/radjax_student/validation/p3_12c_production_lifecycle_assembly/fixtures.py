"""Fresh P3.12C production registry fixtures; never an assembly recipe."""

from __future__ import annotations

from radjax_student.architecture import ArchitectureConfig, ArchitectureRegistry
from radjax_student.contracts import ObjectiveConfig
from radjax_student.learning import (
    JaxLearningAssemblyRegistries,
    JaxLearningAssemblyRequest,
    LearningState,
    ObjectiveScope,
    UpdateScope,
)
from radjax_student.objectives import (
    CANONICAL_MSE_IDENTITY,
    build_default_objective_registry,
)
from radjax_student.optimizers import OptimizerConfig, OptimizerRegistry, SgdOptimizer
from radjax_student.runtime import RuntimeConfig, build_default_runtime_registry
from radjax_student.validation.p3_11_9_replay.runner_jax import (
    ARCHITECTURE_ID,
    StatefulLinearJaxArchitecture,
)


def fresh_request_and_registries() -> tuple[
    JaxLearningAssemblyRequest, JaxLearningAssemblyRegistries
]:
    architecture_registry = ArchitectureRegistry()
    architecture_registry.register(StatefulLinearJaxArchitecture(ARCHITECTURE_ID))
    optimizer_registry = OptimizerRegistry()
    optimizer_registry.register(SgdOptimizer())
    request = JaxLearningAssemblyRequest(
        architecture_id=ARCHITECTURE_ID,
        architecture_version=1,
        architecture_config=ArchitectureConfig(
            ARCHITECTURE_ID, vocab_size=8, dtype_intent="float32"
        ),
        objective_identity=CANONICAL_MSE_IDENTITY,
        objective_config=ObjectiveConfig(CANONICAL_MSE_IDENTITY, {"reduction": "mean"}),
        optimizer_id="sgd.v1",
        optimizer_version=1,
        optimizer_config=OptimizerConfig("sgd.v1", learning_rate=0.25),
        runtime_backend_id="jax",
        runtime_implementation_version="p2.9",
        runtime_config=RuntimeConfig(
            backend_id="jax",
            platform_preference="cpu",
            precision_policy="float32",
            placement_policy="single_device",
            compilation_policy="eager",
            distributed_policy="disabled",
            fallback_policy="disallowed",
            seed=17,
        ),
        root_seed=17,
        learning_state=LearningState(
            "p312c",
            active_update_scope=UpdateScope("named_region", "trunk"),
            active_objective_scope=ObjectiveScope(),
        ),
    )
    return request, JaxLearningAssemblyRegistries(
        architecture_registry,
        build_default_objective_registry(),
        optimizer_registry,
        build_default_runtime_registry(),
    )
