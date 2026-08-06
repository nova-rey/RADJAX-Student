"""Measure the narrow P6.1 CPU forward and admission baseline reproducibly."""

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import subprocess
import sys
import time
from hashlib import sha256
from pathlib import Path
from typing import Any


def _identity(value: object) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _language_identities(vocabulary_size: int) -> tuple[Any, Any, Any]:
    from radjax_student.contracts import (
        HFSpecialTokenIdentity,
        HFTokenizerIdentity,
        HFVocabularyIdentity,
    )

    tokenizer = HFTokenizerIdentity(
        "p6_1_probe",
        "v1",
        _identity({"tokenizer": vocabulary_size}),
        _identity({"config": vocabulary_size}),
        "synthetic",
        _identity({"normalization": "none"}),
        "synthetic",
    )
    vocabulary = HFVocabularyIdentity(
        vocabulary_size,
        _identity({"vocabulary": vocabulary_size}),
        _identity({"token_to_id": vocabulary_size}),
        _identity({"added": []}),
        None,
    )
    return tokenizer, vocabulary, HFSpecialTokenIdentity(0, 1, 2, 3, None)


def _synchronized_seconds(callable_: Any) -> float:
    started = time.perf_counter()
    result = callable_()
    result.block_until_ready()
    return time.perf_counter() - started


def _peak_rss_bytes() -> int:
    # macOS reports bytes; Linux reports KiB. The P6.1 authority is macOS CPU.
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _forward_probe(vocabulary_size: int, sequence_length: int) -> dict[str, object]:
    import jax
    import jax.numpy as jnp

    from radjax_student.architecture.models import ArchitectureInitRequest
    from radjax_student.architecture.rwkv7_reference import (
        RWKV7ReferencePlugin,
        configurable_architecture_config,
    )
    from radjax_student.contracts import ObjectiveScope
    from radjax_student.learning.jax_core import JaxBatch

    tokenizer, vocabulary, special_tokens = _language_identities(vocabulary_size)
    config = configurable_architecture_config(
        vocabulary_size,
        sequence_length,
        tokenizer=tokenizer,
        vocabulary=vocabulary,
        special_tokens=special_tokens,
    )
    plugin = RWKV7ReferencePlugin()
    initialized = plugin.initialize_parameters(
        ArchitectureInitRequest(
            config=config,
            runtime_keys_reference="p6_1.baseline:0",
            precision_policy="float32",
            runtime_initialization_material=jax.random.key(0),
        )
    )
    input_ids = jnp.arange(sequence_length, dtype=jnp.int32)[None, :]

    def forward(parameters: Any, tokens: Any) -> Any:
        return plugin.apply_jax(
            parameters,
            initialized.architecture_carry,
            JaxBatch(inputs={"token_ids": tokens}, targets={}),
            architecture_config=config,
            objective_scope=ObjectiveScope(),
            training=False,
            rng_key=None,
        ).outputs

    eager_seconds = _synchronized_seconds(
        lambda: forward(initialized.parameters, input_ids)
    )
    compiled = jax.jit(forward)
    lowered = compiled.lower(initialized.parameters, input_ids)
    started = time.perf_counter()
    executable = lowered.compile()
    compile_seconds = time.perf_counter() - started
    first_seconds = _synchronized_seconds(
        lambda: executable(initialized.parameters, input_ids)
    )
    steady_seconds = tuple(
        _synchronized_seconds(lambda: executable(initialized.parameters, input_ids))
        for _ in range(30)
    )
    try:
        forward(
            initialized.parameters,
            jnp.zeros((2, sequence_length), dtype=jnp.int32),
        )
    except Exception as error:  # The public plugin owns batch rejection.
        batch_two = type(error).__name__
    else:  # pragma: no cover - the baseline must not silently claim B=2 support.
        batch_two = "accepted"
    return {
        "vocabulary_size": vocabulary_size,
        "sequence_length": sequence_length,
        "batch_size_one": True,
        "batch_size_two_result": batch_two,
        "eager_seconds": eager_seconds,
        "lower_compile_seconds": compile_seconds,
        "first_synchronized_seconds": first_seconds,
        "steady_synchronized_seconds": steady_seconds,
        "same_shape_compile_events": 1,
        "output_shape": list(executable(initialized.parameters, input_ids).shape),
        "parameter_bytes": sum(
            int(leaf.nbytes)
            for leaf in jax.tree_util.tree_leaves(initialized.parameters)
        ),
    }


def _materialization_probe(fixture: Path | None) -> dict[str, object]:
    if fixture is None:
        return {"available": False, "reason": "fixture_unavailable"}
    code = """
import json
import resource
import sys
from pathlib import Path
from radjax_student.artifacts.native_v3_v6 import (
    open_native_v3_v6_behavioral_projection,
)
from radjax_student.behavior import materialize_behavioral_batches_v1
fixture = Path(sys.argv[1])
projection = open_native_v3_v6_behavioral_projection(fixture)
batches = materialize_behavioral_batches_v1(projection)
peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
if sys.platform != 'darwin':
    peak *= 1024
print(json.dumps({
    'descriptor_identity': batches.descriptor.identity,
    'peak_rss_bytes': peak,
    'available': True,
}, sort_keys=True))
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(fixture)],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(Path.cwd() / "src")},
    )
    if result.returncode:
        return {"available": False, "reason": "subprocess_failure"}
    return json.loads(result.stdout)


def _finite(value: object) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture", type=Path)
    args = parser.parse_args()
    fixture = args.fixture if args.fixture and args.fixture.is_dir() else None
    payload = {
        "schema_version": "radjax.phase6.p6_1_raw_baseline.v1",
        "runtime": {
            "platform": sys.platform,
            "precision_policy": "float32",
            "python_version": ".".join(map(str, sys.version_info[:3])),
        },
        "forward_probes": [
            _forward_probe(vocabulary_size, sequence_length)
            for vocabulary_size, sequence_length in ((64, 5), (128, 8))
        ],
        "materialization_probe": _materialization_probe(fixture),
        "peak_rss_bytes": _peak_rss_bytes(),
        "nonclaims": [
            "no_generic_behavior_lifecycle",
            "no_accelerator_measurement",
            "no_structural_rwkv_shape_claim",
        ],
    }
    if not _finite(payload):
        raise RuntimeError("P6.1 measurement produced a nonfinite value")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
