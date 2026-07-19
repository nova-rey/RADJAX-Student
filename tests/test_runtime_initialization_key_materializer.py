"""Focused P4.3 tests for runtime-owned initialization-key materialization."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from radjax_student.runtime.jax_bridge import (
    RuntimeJaxBridgeError,
    materialize_initialization_jax_key,
)

ROOT = Path(__file__).resolve().parents[1]


def test_initialization_key_materializer_is_deterministic_and_seed_isolated() -> None:
    jax = pytest.importorskip("jax")

    first = materialize_initialization_jax_key("runtime_keys.v1:initialization:17")
    repeated = materialize_initialization_jax_key("runtime_keys.v1:initialization:17")
    changed = materialize_initialization_jax_key("runtime_keys.v1:initialization:18")

    assert jax.numpy.array_equal(
        jax.random.key_data(first), jax.random.key_data(repeated)
    )
    assert not jax.numpy.array_equal(
        jax.random.key_data(first), jax.random.key_data(changed)
    )


@pytest.mark.parametrize(
    "reference",
    (
        "runtime_keys.v1:model_initialization:17",
        "runtime_keys.v1:initialization:017",
        "runtime_keys.v1:initialization:-1",
        "runtime_keys.v1:initialization:17:extra",
    ),
)
def test_initialization_key_materializer_rejects_noncanonical_references(
    reference: str,
) -> None:
    with pytest.raises(
        RuntimeJaxBridgeError,
        match="runtime_jax_initialization_reference_invalid",
    ):
        materialize_initialization_jax_key(reference)


def test_runtime_keys_module_remains_jax_free() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import radjax_student.runtime.keys; "
                "assert 'jax' not in sys.modules and 'jaxlib' not in sys.modules"
            ),
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
