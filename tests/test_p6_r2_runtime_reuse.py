from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from radjax_student.runtime import (
    PreparedExecutionReuseCache,
    execute_function,
)
from radjax_student.runtime.callables import bind_runtime_callable
from radjax_student.steps.jax_step import (
    GENERIC_JAX_LEARNING_STEP_DECLARATION,
    execute_jax_learning_step_kernel,
)
from tests.support.rwkv7_learning import assembled, batch, execute, tree_allclose
from tests.test_runtime_execution import (
    _context,
    _FakeExecutionBackend,
    _request,
)


def _generic_request(mode: str = "jit"):
    binding = bind_runtime_callable(
        callable=execute_jax_learning_step_kernel,
        declaration=GENERIC_JAX_LEARNING_STEP_DECLARATION,
    )
    return replace(
        _request(mode=mode),
        function_id=binding.reference.callable_id,
        callable_reference=binding.reference,
    ), binding


def test_same_prepared_specialization_compiles_once_and_reuses_handle() -> None:
    backend = _FakeExecutionBackend()
    request, binding = _generic_request()
    cache = PreparedExecutionReuseCache()

    def function(value, scale):
        return value * scale + 1

    first, first_receipt = execute_function(
        context=_context(),
        callable_binding=binding,
        request=request,
        backend=backend,
        args=(function, 2, 3),
        reuse_cache=cache,
    )
    second, second_receipt = execute_function(
        context=_context(),
        callable_binding=binding,
        request=replace(request, request_id="request-second"),
        backend=backend,
        args=(function, 2, 3),
        reuse_cache=cache,
    )

    assert first == second == 7
    assert first_receipt.status == second_receipt.status == "pass"
    assert first_receipt.compiled is True
    assert second_receipt.compiled is True
    assert (
        first_receipt.prepared_execution_digest
        == second_receipt.prepared_execution_digest
    )
    assert backend.compile_calls == 1
    assert len(cache) == 1


def test_shape_change_gets_distinct_prepared_specialization() -> None:
    backend = _FakeExecutionBackend()
    request, binding = _generic_request()
    cache = PreparedExecutionReuseCache()

    first, first_receipt = execute_function(
        context=_context(),
        callable_binding=binding,
        request=request,
        backend=backend,
        args=(lambda value: value, np.zeros((2,), dtype=np.float32)),
        reuse_cache=cache,
    )
    second, second_receipt = execute_function(
        context=_context(),
        callable_binding=binding,
        request=replace(request, request_id="request-shape"),
        backend=backend,
        args=(lambda value: value, np.zeros((3,), dtype=np.float32)),
        reuse_cache=cache,
    )

    assert first_receipt.status == second_receipt.status == "pass"
    assert (
        first_receipt.prepared_execution_digest
        != second_receipt.prepared_execution_digest
    )
    assert first.shape == (2,)
    assert second.shape == (3,)
    assert backend.compile_calls == 2
    assert len(cache) == 2


def test_static_value_change_gets_distinct_prepared_specialization() -> None:
    backend = _FakeExecutionBackend()
    request, binding = _generic_request()
    request = replace(
        request,
        compilation_options=replace(
            request.compilation_options, static_arg_positions=(2,)
        ),
    )
    cache = PreparedExecutionReuseCache()

    _, first_receipt = execute_function(
        context=_context(),
        callable_binding=binding,
        request=request,
        backend=backend,
        args=(lambda value, scale: value * scale, 2, 3),
        reuse_cache=cache,
    )
    _, second_receipt = execute_function(
        context=_context(),
        callable_binding=binding,
        request=replace(request, request_id="request-static"),
        backend=backend,
        args=(lambda value, scale: value * scale, 2, 4),
        reuse_cache=cache,
    )

    assert (
        first_receipt.prepared_execution_digest
        != second_receipt.prepared_execution_digest
    )
    assert backend.compile_calls == 2
    assert len(cache) == 2


def test_execution_policy_changes_get_distinct_specializations() -> None:
    backend = _FakeExecutionBackend()
    request, binding = _generic_request()
    cache = PreparedExecutionReuseCache()
    args = (lambda value: value, np.zeros((2,), dtype=np.float32))

    _, base = execute_function(
        context=_context(),
        callable_binding=binding,
        request=request,
        backend=backend,
        args=args,
        reuse_cache=cache,
    )
    donation = replace(
        request,
        request_id="request-donation",
        compilation_options=replace(
            request.compilation_options, donate_arg_positions=(1,)
        ),
    )
    placement = replace(request, request_id="request-placement", placement_plan_id="p2")
    options = replace(
        request,
        request_id="request-options",
        compilation_options=replace(
            request.compilation_options, metadata={"compiler_option": "v2"}
        ),
    )
    receipts = [
        execute_function(
            context=_context(),
            callable_binding=binding,
            request=value,
            backend=backend,
            args=args,
            reuse_cache=cache,
        )[1]
        for value in (donation, placement, options)
    ]

    assert all(item.status == "pass" for item in receipts)
    assert (
        len(
            {
                base.prepared_execution_digest,
                *(item.prepared_execution_digest for item in receipts),
            }
        )
        == 4
    )
    assert backend.compile_calls == 4


def test_new_cache_rebuilds_compiled_handle_from_same_identity() -> None:
    backend = _FakeExecutionBackend()
    request, binding = _generic_request()
    args = (lambda value: value, np.zeros((2,), dtype=np.float32))
    first, first_receipt = execute_function(
        context=_context(),
        callable_binding=binding,
        request=request,
        backend=backend,
        args=args,
        reuse_cache=PreparedExecutionReuseCache(),
    )
    second, second_receipt = execute_function(
        context=_context(),
        callable_binding=binding,
        request=replace(request, request_id="request-restored"),
        backend=backend,
        args=args,
        reuse_cache=PreparedExecutionReuseCache(),
    )

    assert np.array_equal(first, second)
    assert (
        first_receipt.prepared_execution_digest
        == second_receipt.prepared_execution_digest
    )
    assert backend.compile_calls == 2


def test_reuse_seam_is_architecture_neutral() -> None:
    import inspect

    from radjax_student.runtime import execution as generic_runtime

    source = inspect.getsource(generic_runtime.execute_function).lower()
    assert "rwkv" not in source
    assert "architecture" not in source


def test_real_rwkv_jit_reuses_compiled_execution_and_preserves_numerics() -> None:
    compiled = assembled("jit")
    first = execute(compiled, batch())
    first_state = compiled.loop_executor.lifecycle
    second = execute(compiled, batch((5, 3, 7, 1)))
    second_state = compiled.loop_executor.lifecycle

    assert first.runtime_result.compiled is True
    assert second.runtime_result.compiled is True
    assert first.runtime_result.prepared_execution_digest == (
        second.runtime_result.prepared_execution_digest
    )
    assert second.runtime_result.compilation_seconds < max(
        first.runtime_result.compilation_seconds / 10.0, 1e-5
    )
    assert second.result.status == "pass"
    assert second_state.learning_state.global_step == 2

    reference = assembled("eager")
    eager_first = execute(reference, batch())
    assert tree_allclose(first.parameters, eager_first.parameters)
    assert tree_allclose(first.architecture_carry, eager_first.architecture_carry)
    assert first.result.loss.loss == pytest.approx(
        eager_first.result.loss.loss, rel=1e-5, abs=2e-5
    )
    assert first_state.learning_state.global_step == 1
