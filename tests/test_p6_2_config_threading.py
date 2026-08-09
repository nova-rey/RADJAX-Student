"""P6.2 proves neutral configuration reaches generic JAX execution planning."""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from radjax_student.architecture import (
    ArchitectureCapabilityProfile,
    ArchitectureConfig,
    ArchitectureContractError,
    ArchitectureInitRequest,
    ArchitectureMetadata,
    ForwardResult,
    IntermediateSurfaceDescriptor,
    JaxArchitecturePlugin,
    NamedRegion,
    ParameterCatalog,
    ParameterDescriptor,
)
from radjax_student.architecture.rwkv7_reference import (
    RWKV7_REFERENCE_ARCHITECTURE_ID,
    RWKV7_REFERENCE_ARCHITECTURE_VERSION,
    RWKV7ReferencePlugin,
    configurable_architecture_config,
    parameter_layout,
    reference_architecture_config,
)
from radjax_student.architecture.testing import (
    FAKE_ARCHITECTURE_CAPABILITIES,
    FakeArchitecturePlugin,
)
from radjax_student.contracts import (
    HFSpecialTokenIdentity,
    HFTokenizerIdentity,
    HFVocabularyIdentity,
    LearningBatch,
    ObjectiveConfig,
    ObjectiveScope,
    ParameterTreeLayout,
    ParameterTreeLayoutEntry,
    UpdateScope,
    hf_digest,
)
from radjax_student.learning import (
    JaxLearningAssemblyRegistries,
    JaxLearningAssemblyRequest,
    LearningState,
    assemble_jax_learning_lifecycle,
)
from radjax_student.learning.jax_execution import prepare_jax_execution_plan
from radjax_student.objectives import (
    SPARSE_CROSS_ENTROPY_IDENTITY,
    build_default_objective_registry,
)
from radjax_student.optimizers import (
    OptimizerConfig,
    OptimizerRegistry,
    SgdOptimizer,
)
from radjax_student.runtime import (
    RuntimeConfig,
    build_default_runtime_registry,
)

jax = pytest.importorskip("jax")

pytestmark = pytest.mark.jax


def _config(vocabulary_size: int, sequence_length: int) -> ArchitectureConfig:
    return configurable_architecture_config(
        vocabulary_size,
        sequence_length,
        tokenizer=HFTokenizerIdentity(
            "p6-config-threading-tokenizer",
            "v1",
            hf_digest({"tokenizer": vocabulary_size}),
            hf_digest({"tokenizer_config": vocabulary_size}),
            "synthetic",
            hf_digest({"normalization": "identity"}),
            "synthetic",
        ),
        vocabulary=HFVocabularyIdentity(
            vocabulary_size,
            hf_digest({"vocabulary": vocabulary_size}),
            hf_digest({"token_to_id": vocabulary_size}),
            hf_digest({"added": []}),
            None,
        ),
        special_tokens=HFSpecialTokenIdentity(0, 1, 2, 3, None),
    )


def _registries() -> JaxLearningAssemblyRegistries:
    from radjax_student.architecture import ArchitectureRegistry
    from radjax_student.architecture.rwkv7_reference import register_rwkv7_reference

    architectures = ArchitectureRegistry()
    register_rwkv7_reference(architectures)
    optimizers = OptimizerRegistry()
    optimizers.register(SgdOptimizer())
    return JaxLearningAssemblyRegistries(
        architectures,
        build_default_objective_registry(),
        optimizers,
        build_default_runtime_registry(),
    )


def _assemble(config: ArchitectureConfig):
    return assemble_jax_learning_lifecycle(
        JaxLearningAssemblyRequest(
            architecture_id=RWKV7_REFERENCE_ARCHITECTURE_ID,
            architecture_version=RWKV7_REFERENCE_ARCHITECTURE_VERSION,
            architecture_config=config,
            objective_identity=SPARSE_CROSS_ENTROPY_IDENTITY,
            objective_config=ObjectiveConfig(
                SPARSE_CROSS_ENTROPY_IDENTITY, {"reduction": "mean"}
            ),
            optimizer_id="sgd.v1",
            optimizer_version=1,
            optimizer_config=OptimizerConfig("sgd.v1", learning_rate=0.01),
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
                seed=31,
            ),
            root_seed=31,
            learning_state=LearningState(
                "p6-config-threading",
                active_update_scope=UpdateScope(),
                active_objective_scope=ObjectiveScope(),
            ),
        ),
        registries=_registries(),
    )


def _execute(config: ArchitectureConfig):
    assembled = _assemble(config)
    lifecycle = assembled.loop_executor.lifecycle
    sequence_length = config.sequence_length
    assert sequence_length is not None
    batch = LearningBatch(
        "p6-config-threading",
        inputs={"token_ids": [list(range(sequence_length))]},
        targets={
            "token_ids": [list(range(1, sequence_length)) + [0]],
        },
    )
    return assembled.loop_executor(
        architecture=lifecycle.architecture,
        architecture_config=lifecycle.architecture_config,
        optimizer=lifecycle.optimizer,
        optimizer_config=lifecycle.optimizer_config,
        optimizer_state=lifecycle.optimizer_state,
        learning_state=lifecycle.learning_state,
        objective=lifecycle.objective_selection,
        batch=batch,
    )


@pytest.mark.parametrize("config", (reference_architecture_config(), _config(512, 8)))
def test_neutral_execution_plan_reaches_genuine_rwkv_lifecycle(
    config: ArchitectureConfig,
) -> None:
    """Both frozen and accepted configurable capacities use the same loop seam."""

    execution = _execute(config)

    assert execution.runtime_result.status == "pass"
    assert execution.learning_state.global_step == 1


def test_config_layout_mismatch_fails_closed_before_jax_execution() -> None:
    config = _config(512, 8)
    plugin = RWKV7ReferencePlugin()
    parameters = plugin.initialize_parameters(
        ArchitectureInitRequest(
            config=config,
            runtime_keys_reference="p6-config-threading:31",
            precision_policy="float32",
            runtime_initialization_material=jax.random.key(31),
        )
    ).parameters

    with pytest.raises(ArchitectureContractError) as caught:
        prepare_jax_execution_plan(
            architecture=plugin,
            parameters=parameters,
            parameter_layout=parameter_layout(),
            architecture_config=config,
            objective_scope=ObjectiveScope(),
            update_scope=UpdateScope(),
        )

    assert caught.value.code == "architecture_parameter_catalog_invalid"


class _SecondArchitecture(FakeArchitecturePlugin):
    """A non-RWKV config-aware plugin proving the neutral catalog seam."""

    architecture_id = "test.second_architecture.v1"
    architecture_version = 1

    def __init__(self) -> None:
        self.received_configs: list[ArchitectureConfig | None] = []

    def capability_profile(self) -> ArchitectureCapabilityProfile:
        return ArchitectureCapabilityProfile(
            self.architecture_id,
            self.architecture_version,
            (*FAKE_ARCHITECTURE_CAPABILITIES, "architecture.jax_execution_v1"),
        )

    def describe_parameters(
        self,
        parameters: object | None = None,
        *,
        architecture_config: ArchitectureConfig | None = None,
    ) -> ParameterCatalog:
        del parameters
        self.received_configs.append(architecture_config)
        vocabulary_size = (
            3 if architecture_config is None else architecture_config.vocab_size
        )
        assert vocabulary_size is not None
        return ParameterCatalog(
            self.architecture_id,
            (
                ParameterDescriptor(
                    "model.weight",
                    (vocabulary_size,),
                    "float32",
                    "output_head",
                    ("whole_student",),
                ),
            ),
        )

    def architecture_metadata(self) -> ArchitectureMetadata:
        catalog = self.describe_parameters()
        return ArchitectureMetadata(
            self.architecture_id,
            catalog,
            self.capability_profile(),
            named_regions=(NamedRegion("whole_student", catalog.paths),),
            objective_surfaces=(
                IntermediateSurfaceDescriptor(
                    "final_output", "logits", available_in_training=True
                ),
            ),
        )

    def apply_jax(
        self,
        parameters: Any,
        architecture_state: Any,
        batch: Any,
        *,
        architecture_config: ArchitectureConfig | None = None,
        objective_scope: ObjectiveScope,
        training: bool,
        rng_key: Any | None,
    ) -> ForwardResult:
        del (
            parameters,
            architecture_state,
            batch,
            architecture_config,
            objective_scope,
            training,
            rng_key,
        )
        return ForwardResult(outputs="synthetic")


def test_second_architecture_uses_the_same_config_threading_seam() -> None:
    architecture = _SecondArchitecture()
    config = ArchitectureConfig(
        architecture.architecture_id, vocab_size=7, sequence_length=2
    )
    layout = ParameterTreeLayout(
        architecture.architecture_id,
        (
            ParameterTreeLayoutEntry(
                "model.weight",
                ("model", "weight"),
                (7,),
                "float32",
                "output_head",
                ("whole_student",),
            ),
        ),
    )

    plan = prepare_jax_execution_plan(
        architecture=architecture,
        parameters={"model": {"weight": _Leaf((7,))}},
        parameter_layout=layout,
        architecture_config=config,
        objective_scope=ObjectiveScope(),
        update_scope=UpdateScope(),
    )

    assert isinstance(architecture, JaxArchitecturePlugin)
    assert config in architecture.received_configs
    assert plan.update_selection.selected_parameter_paths == ("model.weight",)


def test_generic_owners_do_not_import_or_branch_on_rwkv() -> None:
    """Config propagation remains neutral rather than architecture-special cased."""

    import radjax_student.learning.assembly as assembly
    import radjax_student.learning.jax_execution as execution
    import radjax_student.steps.jax_step as step

    for owner in (assembly, execution, step):
        assert "rwkv" not in inspect.getsource(owner).lower()


class _Leaf:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape
        self.dtype = "float32"
