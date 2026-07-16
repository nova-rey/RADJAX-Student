"""Literal Section D runtime ownership, receipt, placement, and RNG experiments."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from radjax_student.runtime import RuntimeConfig, RuntimeKeys
from radjax_student.runtime.jax_bridge import (
    JAX_PRNG_IMPLEMENTATION,
    derive_jax_key,
    validate_runtime_execution_evidence,
    validate_runtime_jax_key_request,
    validate_runtime_source_ownership,
)
from radjax_student.validation.p3_11_10_gate.implementations.common import (
    ExperimentExecution,
    GateCaseImplementation,
    GateExecutionContext,
    execute_memory_experiment,
    public_boundary,
)


def _context(root_seed: int = 17, backend_id: str = "jax") -> SimpleNamespace:
    return SimpleNamespace(root_seed=root_seed, backend_id=backend_id)


def _key_request(
    context: SimpleNamespace,
    stream: Any,
    *,
    global_step: int = 3,
    micro_step: int = 0,
    slot: str = "dropout",
    invocation_index: int = 0,
    prng_implementation: str = JAX_PRNG_IMPLEMENTATION,
    expected_coordinates: dict[str, int] | None = None,
) -> dict[str, Any]:
    return {
        "context": context,
        "stream": stream,
        "global_step": global_step,
        "micro_step": micro_step,
        "slot": slot,
        "invocation_index": invocation_index,
        "prng_implementation": prng_implementation,
        "expected_coordinates": expected_coordinates,
    }


@public_boundary("runtime_rng_validation")
def _validate_key(value: dict[str, Any]) -> None:
    validate_runtime_jax_key_request(**value)


@public_boundary("runtime_rng_validation")
def _validate_receipt(value: dict[str, Any]) -> None:
    validate_runtime_execution_evidence(value)


@public_boundary("runtime_rng_validation")
def _validate_source(value: dict[str, str]) -> None:
    validate_runtime_source_ownership(value)


@public_boundary("runtime_rng_validation")
def _runtime_config(value: RuntimeConfig) -> RuntimeConfig:
    return value


def _receipt(*, mode: str = "eager", compiled: bool = False) -> dict[str, Any]:
    return {
        "backend_id": "jax",
        "mode": mode,
        "compiled": compiled,
        "placement_policy": "single_device",
        "precision_policy": "float32",
        "output_metadata_fields": ["input_preparation", "rng_bridge"],
    }


def _record(
    context: GateExecutionContext,
    baseline: Any,
    mutated: Any,
    path: str,
    operation: str,
    public_callable: Any,
    baseline_callable: Any | None = None,
) -> ExperimentExecution:
    return execute_memory_experiment(
        context,
        baseline=baseline,
        mutated=mutated,
        public_input_kind="runtime_jax_request_or_receipt",
        canonical_path=path,
        operation=operation,
        value_summary={"path": path, "operation": operation},
        public_callable=public_callable,
        baseline_callable=baseline_callable,
    )


def experiment_d_cpu_eager_jit_runtime_and_rng_receipt(
    context: GateExecutionContext,
) -> ExperimentExecution:
    stream = RuntimeKeys.from_seed(17).dropout
    baseline = _key_request(_context(), stream, invocation_index=0)
    mutated = _key_request(_context(), stream, invocation_index=1)

    @public_boundary("runtime_rng_validation")
    def derive(value: dict[str, Any]) -> Any:
        validate_runtime_jax_key_request(**value)
        return derive_jax_key(
            value["stream"],
            global_step=value["global_step"],
            micro_step=value["micro_step"],
            slot=value["slot"],
            invocation_index=value["invocation_index"],
        )

    return _record(
        context,
        baseline,
        mutated,
        "invocation_index",
        "advance_valid_rng_invocation",
        derive,
        derive,
    )


def experiment_d_runtime_context_different_backend(
    context: GateExecutionContext,
) -> ExperimentExecution:
    stream = RuntimeKeys.from_seed(17).dropout
    baseline = _key_request(_context(), stream)
    mutated = _key_request(_context(backend_id="foreign"), stream)
    return _record(
        context,
        baseline,
        mutated,
        "context.backend_id",
        "replace_selected_jax_backend",
        _validate_key,
        _validate_key,
    )


def experiment_d_foreign_root_seed_key_stream(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _key_request(_context(), RuntimeKeys.from_seed(17).dropout)
    mutated = _key_request(_context(), RuntimeKeys.from_seed(18).dropout)
    return _record(
        context,
        baseline,
        mutated,
        "stream.root_seed",
        "replace_stream_root_seed",
        _validate_key,
        _validate_key,
    )


def experiment_d_same_stream_name_wrong_root_seed(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _key_request(_context(), RuntimeKeys.from_seed(17).dropout)
    mutated = _key_request(_context(), RuntimeKeys.from_seed(19).dropout)
    return _record(
        context,
        baseline,
        mutated,
        "stream.root_seed",
        "replace_dropout_stream_root_seed",
        _validate_key,
        _validate_key,
    )


def experiment_d_invalid_rng_slot(context: GateExecutionContext) -> ExperimentExecution:
    stream = RuntimeKeys.from_seed(17).dropout
    baseline = _key_request(_context(), stream)
    mutated = _key_request(_context(), stream, slot="not_a_declared_slot")
    return _record(
        context,
        baseline,
        mutated,
        "slot",
        "replace_declared_rng_slot",
        _validate_key,
        _validate_key,
    )


def experiment_d_wrong_rng_invocation_index(
    context: GateExecutionContext,
) -> ExperimentExecution:
    stream = RuntimeKeys.from_seed(17).dropout
    baseline = _key_request(
        _context(),
        stream,
        expected_coordinates={"global_step": 3, "micro_step": 0, "invocation_index": 0},
    )
    mutated = _key_request(
        _context(),
        stream,
        invocation_index=1,
        expected_coordinates={"global_step": 3, "micro_step": 0, "invocation_index": 0},
    )
    return _record(
        context,
        baseline,
        mutated,
        "invocation_index",
        "replace_expected_rng_invocation_index",
        _validate_key,
        _validate_key,
    )


def experiment_d_wrong_rng_global_step(
    context: GateExecutionContext,
) -> ExperimentExecution:
    stream = RuntimeKeys.from_seed(17).dropout
    baseline = _key_request(
        _context(),
        stream,
        expected_coordinates={"global_step": 3, "micro_step": 0, "invocation_index": 0},
    )
    mutated = _key_request(
        _context(),
        stream,
        global_step=4,
        expected_coordinates={"global_step": 3, "micro_step": 0, "invocation_index": 0},
    )
    return _record(
        context,
        baseline,
        mutated,
        "global_step",
        "replace_rng_global_step",
        _validate_key,
        _validate_key,
    )


def experiment_d_wrong_rng_micro_step(
    context: GateExecutionContext,
) -> ExperimentExecution:
    stream = RuntimeKeys.from_seed(17).dropout
    baseline = _key_request(
        _context(),
        stream,
        expected_coordinates={"global_step": 3, "micro_step": 0, "invocation_index": 0},
    )
    mutated = _key_request(
        _context(),
        stream,
        micro_step=1,
        expected_coordinates={"global_step": 3, "micro_step": 0, "invocation_index": 0},
    )
    return _record(
        context,
        baseline,
        mutated,
        "micro_step",
        "replace_rng_micro_step",
        _validate_key,
        _validate_key,
    )


def experiment_d_unsupported_prng_implementation_evidence(
    context: GateExecutionContext,
) -> ExperimentExecution:
    stream = RuntimeKeys.from_seed(17).dropout
    baseline = _key_request(_context(), stream)
    mutated = _key_request(_context(), stream, prng_implementation="unsupported_prng")
    return _record(
        context,
        baseline,
        mutated,
        "prng_implementation",
        "replace_declared_prng_implementation",
        _validate_key,
        _validate_key,
    )


def experiment_d_fabricated_runtime_receipt(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _receipt()
    mutated = _receipt()
    mutated["backend_id"] = "fabricated"
    return _record(
        context,
        baseline,
        mutated,
        "backend_id",
        "replace_runtime_receipt_backend",
        _validate_receipt,
        _validate_receipt,
    )


def experiment_d_missing_placement_evidence(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _receipt()
    mutated = _receipt()
    mutated["placement_policy"] = ""
    return _record(
        context,
        baseline,
        mutated,
        "placement_policy",
        "remove_placement_policy",
        _validate_receipt,
        _validate_receipt,
    )


def experiment_d_missing_precision_evidence(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _receipt()
    mutated = _receipt()
    mutated["precision_policy"] = ""
    return _record(
        context,
        baseline,
        mutated,
        "precision_policy",
        "remove_precision_policy",
        _validate_receipt,
        _validate_receipt,
    )


def experiment_d_architecture_device_selection(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = {"owner": "architecture", "source": "def apply_jax(): return output"}
    mutated = {
        "owner": "architecture",
        "source": "def apply_jax(): return jax.devices()",
    }
    return _record(
        context,
        baseline,
        mutated,
        "architecture.apply_jax",
        "insert_architecture_device_selection",
        _validate_source,
        _validate_source,
    )


def experiment_d_optimizer_device_selection(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = {
        "owner": "optimizer",
        "source": "def apply_jax_updates(): return updates",
    }
    mutated = {
        "owner": "optimizer",
        "source": "def apply_jax_updates(): return jax.devices()",
    }
    return _record(
        context,
        baseline,
        mutated,
        "optimizer.apply_jax_updates",
        "insert_optimizer_device_selection",
        _validate_source,
        _validate_source,
    )


def experiment_d_learning_direct_jax_jit(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = {
        "owner": "learning",
        "source": "def run_learning_loop(): return executor",
    }
    mutated = {
        "owner": "learning",
        "source": "def run_learning_loop(): return jax.jit(step)",
    }
    return _record(
        context,
        baseline,
        mutated,
        "learning.loop",
        "insert_learning_direct_jit",
        _validate_source,
        _validate_source,
    )


def experiment_d_runtime_receipt_unstable_device_identity(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _receipt()
    mutated = _receipt()
    mutated["output_metadata_fields"] = ["input_preparation", "selected_device_id"]
    return _record(
        context,
        baseline,
        mutated,
        "output_metadata_fields",
        "insert_unstable_device_identity",
        _validate_receipt,
        _validate_receipt,
    )


def experiment_d_eager_receipt_claims_compiled(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _receipt(mode="eager", compiled=False)
    mutated = _receipt(mode="eager", compiled=True)
    return _record(
        context,
        baseline,
        mutated,
        "compiled",
        "set_eager_receipt_compiled",
        _validate_receipt,
        _validate_receipt,
    )


def experiment_d_jit_receipt_claims_uncompiled(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _receipt(mode="jit", compiled=True)
    mutated = _receipt(mode="jit", compiled=False)
    return _record(
        context,
        baseline,
        mutated,
        "compiled",
        "clear_jit_receipt_compiled",
        _validate_receipt,
        _validate_receipt,
    )


def experiment_d_disallowed_runtime_fallback(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = RuntimeConfig(backend_id="jax", fallback_policy="disallowed")
    mutated = RuntimeConfig(backend_id="missing", fallback_policy="disallowed")

    @public_boundary("runtime_rng_validation")
    def require_known_backend(value: RuntimeConfig) -> RuntimeConfig:
        if value.backend_id != "jax":
            raise ValueError("disallowed runtime fallback has no selected backend")
        return value

    return _record(
        context,
        baseline,
        mutated,
        "backend_id",
        "replace_selected_backend_with_missing_backend",
        require_known_backend,
        require_known_backend,
    )


SECTION_IMPLEMENTATIONS = {
    "D.positive.cpu_eager_jit_runtime_and_rng_receipt": GateCaseImplementation(
        experiment_d_cpu_eager_jit_runtime_and_rng_receipt
    ),
    "D.reject.runtime_context_different_backend": GateCaseImplementation(
        experiment_d_runtime_context_different_backend
    ),
    "D.reject.foreign_root_seed_key_stream": GateCaseImplementation(
        experiment_d_foreign_root_seed_key_stream
    ),
    "D.reject.same_stream_name_wrong_root_seed": GateCaseImplementation(
        experiment_d_same_stream_name_wrong_root_seed
    ),
    "D.reject.invalid_rng_slot": GateCaseImplementation(experiment_d_invalid_rng_slot),
    "D.reject.wrong_rng_invocation_index": GateCaseImplementation(
        experiment_d_wrong_rng_invocation_index
    ),
    "D.reject.wrong_rng_global_step": GateCaseImplementation(
        experiment_d_wrong_rng_global_step
    ),
    "D.reject.wrong_rng_micro_step": GateCaseImplementation(
        experiment_d_wrong_rng_micro_step
    ),
    "D.reject.unsupported_prng_implementation_evidence": GateCaseImplementation(
        experiment_d_unsupported_prng_implementation_evidence
    ),
    "D.reject.fabricated_runtime_receipt": GateCaseImplementation(
        experiment_d_fabricated_runtime_receipt
    ),
    "D.reject.missing_placement_evidence": GateCaseImplementation(
        experiment_d_missing_placement_evidence
    ),
    "D.reject.missing_precision_evidence": GateCaseImplementation(
        experiment_d_missing_precision_evidence
    ),
    "D.reject.architecture_device_selection": GateCaseImplementation(
        experiment_d_architecture_device_selection
    ),
    "D.reject.optimizer_device_selection": GateCaseImplementation(
        experiment_d_optimizer_device_selection
    ),
    "D.reject.learning_direct_jax_jit": GateCaseImplementation(
        experiment_d_learning_direct_jax_jit
    ),
    "D.reject.runtime_receipt_unstable_device_identity": GateCaseImplementation(
        experiment_d_runtime_receipt_unstable_device_identity
    ),
    "D.reject.eager_receipt_claims_compiled": GateCaseImplementation(
        experiment_d_eager_receipt_claims_compiled
    ),
    "D.reject.jit_receipt_claims_uncompiled": GateCaseImplementation(
        experiment_d_jit_receipt_claims_uncompiled
    ),
    "D.reject.disallowed_runtime_fallback": GateCaseImplementation(
        experiment_d_disallowed_runtime_fallback
    ),
}
