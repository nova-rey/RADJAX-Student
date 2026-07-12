"""P3.11.6 proves runtime-owned JAX keys and complete-pytree placement."""

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

from radjax_student.runtime import (  # noqa: E402
    DeviceInventory,
    ExecutionContext,
    JaxRuntimeBackend,
    RuntimeEnvironment,
    RuntimeKeys,
)
from radjax_student.runtime.jax_bridge import derive_jax_key  # noqa: E402
from radjax_student.runtime.jax_inputs import prepare_jax_inputs  # noqa: E402
from radjax_student.runtime.keys import (  # noqa: E402
    JAX_KEY_BRIDGE_VERSION,
    jax_key_words,
)

pytestmark = pytest.mark.jax


def _context() -> tuple[JaxRuntimeBackend, ExecutionContext]:
    backend = JaxRuntimeBackend()
    context = ExecutionContext(
        backend_id="jax",
        environment=RuntimeEnvironment(
            python_version="3.11",
            jax_available=True,
            local_device_count=1,
            global_device_count=1,
        ),
        device_inventory=DeviceInventory(local_device_count=1, global_device_count=1),
        capabilities=backend.capability_profile(),
        root_seed=7,
        runtime_id="p311-jax-inputs",
        metadata={"selected_device_id": "cpu:0", "placement_policy": "single_device"},
    )
    backend._cpu_contexts[context.runtime_id] = (jax, jax.devices("cpu")[0])
    return backend, context


def test_runtime_key_bridge_is_versioned_deterministic_and_stream_isolated():
    keys = RuntimeKeys.from_seed(9)
    first = jax_key_words(
        keys.dropout, global_step=2, micro_step=1, slot="dropout", invocation_index=3
    )
    assert first == jax_key_words(
        keys.dropout, global_step=2, micro_step=1, slot="dropout", invocation_index=3
    )
    assert first != jax_key_words(
        keys.augmentation,
        global_step=2,
        micro_step=1,
        slot="augmentation",
        invocation_index=3,
    )
    assert JAX_KEY_BRIDGE_VERSION == "runtime_jax_key_bridge.v1"
    assert jnp.array_equal(
        jax.random.key_data(
            derive_jax_key(
                keys.dropout,
                global_step=2,
                micro_step=1,
                slot="dropout",
                invocation_index=3,
            )
        ),
        jnp.asarray(first, dtype=jnp.uint32),
    )


def test_runtime_places_complete_input_pytrees_with_one_precision_policy():
    backend, context = _context()
    prepared = prepare_jax_inputs(
        backend=backend,
        context=context,
        parameters={"w": jnp.asarray((1.0,), dtype=jnp.float16)},
        architecture_carry={"c": jnp.asarray(0.0, dtype=jnp.float16)},
        optimizer_state={"s": jnp.asarray(0.0, dtype=jnp.float16)},
        batch={"x": jnp.asarray((2.0,), dtype=jnp.float16)},
        precision_policy="float32",
    )
    leaves = jax.tree_util.tree_leaves(
        (
            prepared.parameters,
            prepared.architecture_carry,
            prepared.optimizer_state,
            prepared.batch,
        )
    )
    assert all(str(leaf.dtype) == "float32" for leaf in leaves)
    assert prepared.metadata["selected_device_id"] == "cpu:0"
    with pytest.raises(ValueError, match="precision"):
        prepare_jax_inputs(
            backend=backend,
            context=context,
            parameters={},
            architecture_carry={},
            optimizer_state={},
            batch={},
            precision_policy="float8",
        )
