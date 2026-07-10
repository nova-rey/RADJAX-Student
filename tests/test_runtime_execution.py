from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from radjax_student.runtime import (
    CompilationOptions,
    DeviceInventory,
    ExecutionBoundaryError,
    ExecutionContext,
    ExecutionRequest,
    ExecutionResult,
    JaxRuntimeBackend,
    PreparedExecution,
    RuntimeCapabilityProfile,
    RuntimeEnvironment,
    execute_function,
    execute_prepared,
    prepare_execution,
)


def test_eager_execution_prepares_dispatches_synchronizes_and_reports() -> None:
    backend = _FakeExecutionBackend()
    request = _request(mode="eager", synchronize=True)

    output, result = execute_function(
        context=_context(),
        function=lambda x, scale: x * scale + 1,
        request=request,
        backend=backend,
        args=(2, 3),
    )

    assert output == 7
    assert result.status == "pass"
    assert result.mode == "eager"
    assert result.compiled is False
    assert result.dispatched is True
    assert result.synchronized is True
    assert result.compilation_seconds == 0.0
    assert backend.synchronize_calls == 1
    assert result.output_metadata["tree_structure_summary"] == "scalar"


def test_jit_execution_separates_preparation_compilation_and_dispatch() -> None:
    backend = _FakeExecutionBackend()
    request = _request(mode="jit", synchronize=True)

    prepared = prepare_execution(
        context=_context(),
        function=lambda x, scale: x * scale + 1,
        request=request,
        backend=backend,
    )
    output, result = execute_prepared(
        context=_context(),
        prepared=prepared,
        args=(2, 3),
    )

    assert output == 7
    assert prepared.compiled is False
    assert result.compiled is True
    assert backend.compile_calls == 1
    assert backend.dispatch_calls == 1
    assert result.preparation_seconds >= 0
    assert result.compilation_seconds >= 0
    assert result.dispatch_seconds >= 0
    assert result.synchronization_seconds >= 0
    assert result.total_seconds >= result.preparation_seconds


def test_automatic_mode_resolves_to_eager_with_explicit_warning() -> None:
    backend = _FakeExecutionBackend()
    request = _request(mode="automatic")

    prepared = prepare_execution(
        context=_context(),
        function=lambda value: value + 1,
        request=request,
        backend=backend,
    )
    output, result = execute_prepared(
        context=_context(),
        prepared=prepared,
        args=(2,),
    )

    assert prepared.mode == "eager"
    assert output == 3
    assert "execution_automatic_mode_unresolved" in _warning_codes(result)
    assert backend.compile_calls == 0


def test_static_and_donation_policies_validate_against_function_signature() -> None:
    backend = _FakeExecutionBackend()

    def function(x, scale):
        return x * scale + 1

    named = prepare_execution(
        context=_context(),
        function=function,
        request=_request(mode="jit", static_arg_names=("scale",)),
        backend=backend,
    )
    positioned = prepare_execution(
        context=_context(),
        function=function,
        request=_request(mode="jit", static_arg_positions=(1,)),
        backend=backend,
    )

    assert named.mode == positioned.mode == "jit"
    with pytest.raises(
        ExecutionBoundaryError, match="execution_static_argument_invalid"
    ):
        prepare_execution(
            context=_context(),
            function=function,
            request=_request(mode="jit", static_arg_names=("missing",)),
            backend=backend,
        )
    with pytest.raises(ValueError, match="positions must not contain duplicates"):
        CompilationOptions(mode="jit", static_arg_positions=(1, 1))
    with pytest.raises(ValueError, match="names must not overlap"):
        CompilationOptions(
            mode="jit",
            static_arg_names=("scale",),
            donate_arg_names=("scale",),
        )


def test_donation_requires_declared_capability_and_is_never_automatic() -> None:
    backend = _FakeExecutionBackend(capabilities=("execution.eager_v1",))
    request = _request(mode="eager", donate_arg_names=("buffer",))

    output, result = execute_function(
        context=_context(),
        function=lambda buffer: buffer,
        request=request,
        backend=backend,
        args=(1,),
    )

    assert output is None
    assert result.status == "fail"
    assert "execution_capability_missing" in _blocker_codes(result)


@pytest.mark.parametrize(
    ("phase", "code"),
    [
        ("prepare", "execution_preparation_failed"),
        ("compile", "execution_compilation_failed"),
        ("dispatch", "execution_dispatch_failed"),
        ("synchronize", "execution_synchronization_failed"),
    ],
)
def test_execution_phase_failures_are_structured(phase: str, code: str) -> None:
    backend = _FakeExecutionBackend(fail_phase=phase)
    request = _request(mode="jit", synchronize=True)

    output, result = execute_function(
        context=_context(),
        function=lambda value: value + 1,
        request=request,
        backend=backend,
        args=(2,),
    )

    assert output is None
    assert result.status == "fail"
    assert code in _blocker_codes(result)


def test_execution_models_round_trip_without_callable_or_backend_handle() -> None:
    backend = _FakeExecutionBackend()
    request = _request(mode="jit", synchronize=True)
    prepared = prepare_execution(
        context=_context(),
        function=lambda value: value + 1,
        request=request,
        backend=backend,
    )
    _, result = execute_prepared(
        context=_context(),
        prepared=prepared,
        args=(2,),
    )

    assert ExecutionRequest.from_dict(request.to_dict()) == request
    assert PreparedExecution.from_dict(prepared.to_dict()) == prepared
    assert ExecutionResult.from_dict(result.to_dict()) == result
    payload = json.loads(json.dumps(result.to_dict(), sort_keys=True))
    assert payload == result.to_dict()
    assert "function" not in prepared.to_dict()
    assert "handle" not in prepared.to_dict()


def test_unsynchronized_execution_reports_dispatch_without_completion_claim() -> None:
    backend = _FakeExecutionBackend()
    request = _request(mode="eager", synchronize=False)

    _, result = execute_function(
        context=_context(),
        function=lambda value: value + 1,
        request=request,
        backend=backend,
        args=(2,),
    )

    assert result.status == "pass"
    assert result.dispatched is True
    assert result.synchronized is False
    assert "execution_not_synchronized" in _warning_codes(result)


@pytest.mark.skipif(
    importlib.util.find_spec("jax") is None,
    reason="P2.7 real eager and JIT paths require the optional jax extra",
)
@pytest.mark.parametrize("mode", ("eager", "jit"))
def test_real_jax_cpu_execution_boundary(mode: str) -> None:
    backend = JaxRuntimeBackend()
    context = _context(backend_id="jax", capabilities=backend.capability_profile())
    request = _request(mode=mode, synchronize=True)

    output, result = execute_function(
        context=context,
        function=lambda x, scale: x * scale + 1,
        request=request,
        backend=backend,
        args=(2, 3),
    )

    assert output is not None
    assert result.status == "pass"
    assert result.synchronized is True
    assert result.compiled is (mode == "jit")


def test_execution_module_has_no_architecture_training_or_jax_imports() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "radjax_student"
        / "runtime"
        / "execution.py"
    ).read_text(encoding="utf-8")

    for forbidden in (
        "import jax",
        "import numpy",
        "radjax_student.architecture",
        "radjax_student.students",
        "radjax_student.training",
        "radjax_student.artifacts",
        "radjax_contract",
        "socket",
        "urllib",
    ):
        assert forbidden not in source


@dataclass
class _FakeExecutionBackend:
    capabilities: tuple[str, ...] = (
        "compilation.jit_v1",
        "execution.argument_donation_v1",
        "execution.eager_v1",
        "execution.static_arguments_v1",
        "execution.synchronize_v1",
    )
    fail_phase: str | None = None
    backend_id: str = "fake"
    prepare_calls: int = 0
    compile_calls: int = 0
    dispatch_calls: int = 0
    synchronize_calls: int = 0

    def capability_profile(self) -> RuntimeCapabilityProfile:
        return RuntimeCapabilityProfile(
            profile_id="fake.execution.v1",
            backend_id=self.backend_id,
            version=1,
            capabilities=self.capabilities,
        )

    def prepare_runtime_execution(self, context, function, request, mode):
        del context, request
        self.prepare_calls += 1
        if self.fail_phase == "prepare":
            raise RuntimeError("controlled preparation failure")
        return {"function": function, "mode": mode}

    def compile_runtime_execution(self, context, handle, args, kwargs):
        del context, args, kwargs
        if handle["mode"] == "eager":
            return handle, False
        self.compile_calls += 1
        if self.fail_phase == "compile":
            raise RuntimeError("controlled compilation failure")
        return handle, True

    def dispatch_runtime_execution(self, context, handle, args, kwargs):
        del context
        self.dispatch_calls += 1
        if self.fail_phase == "dispatch":
            raise RuntimeError("controlled dispatch failure")
        return handle["function"](*args, **kwargs)

    def synchronize_runtime_execution(self, context, output):
        del context
        self.synchronize_calls += 1
        if self.fail_phase == "synchronize":
            raise RuntimeError("controlled synchronization failure")
        return output


def _request(
    *,
    mode: str,
    synchronize: bool = False,
    static_arg_names: tuple[str, ...] = (),
    static_arg_positions: tuple[int, ...] = (),
    donate_arg_names: tuple[str, ...] = (),
) -> ExecutionRequest:
    return ExecutionRequest(
        request_id=f"request-{mode}",
        function_id="pure.scale_add",
        mode=mode,
        compilation_options=CompilationOptions(
            mode=mode,
            static_arg_names=static_arg_names,
            static_arg_positions=static_arg_positions,
            donate_arg_names=donate_arg_names,
            synchronize_results=synchronize,
        ),
        placement_plan_id="cpu-smoke-intent",
        input_signature={"args": ("scalar", "scalar")},
    )


def _context(
    *,
    backend_id: str = "fake",
    capabilities: RuntimeCapabilityProfile | None = None,
) -> ExecutionContext:
    return ExecutionContext(
        backend_id=backend_id,
        environment=RuntimeEnvironment(
            python_version="3.11.9",
            jax_available=backend_id == "jax",
            process_count=1,
            process_index=0,
            local_device_count=1,
            global_device_count=1,
            distributed_initialized=False,
        ),
        device_inventory=DeviceInventory(
            process_count=1,
            local_device_count=1,
            global_device_count=1,
        ),
        capabilities=(
            _FakeExecutionBackend().capability_profile()
            if capabilities is None
            else capabilities
        ),
        root_seed=0,
        runtime_id=f"{backend_id}-execution-context",
    )


def _blocker_codes(result: ExecutionResult) -> list[str]:
    return [item.code for item in result.blockers]


def _warning_codes(result: ExecutionResult) -> list[str]:
    return [item.code for item in result.warnings]
