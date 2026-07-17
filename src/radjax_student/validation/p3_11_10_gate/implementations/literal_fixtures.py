"""Concrete, inventory-agnostic fixtures used by literal gate experiments."""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from radjax_student.architecture import ArchitectureState
from radjax_student.checkpoints import (
    JaxLearningCheckpointV3,
    load_learning_checkpoint_v3,
    save_learning_checkpoint_v3,
)
from radjax_student.checkpoints.npz_codec import (
    descriptor_digest,
    read_deterministic_npz,
    write_deterministic_npz,
)
from radjax_student.contracts import (
    HFPreservationReference,
    ObjectiveConfig,
    ParameterTreeLayout,
    ParameterTreeLayoutEntry,
    ResolvedObjectiveSelection,
)
from radjax_student.learning import LearningState, ObjectiveScope
from radjax_student.objectives import (
    CANONICAL_MSE_IDENTITY,
    build_default_objective_registry,
)
from radjax_student.optimizers import (
    JaxOptimizerState,
    OptimizerState,
    SgdOptimizer,
)


def checkpoint_layout() -> ParameterTreeLayout:
    return ParameterTreeLayout(
        "literal.checkpoint.architecture.v1",
        (
            ParameterTreeLayoutEntry(
                "trunk.weight", ("trunk", "weight"), (1,), "float32", "weight"
            ),
        ),
    )


def checkpoint_optimizer_state(
    optimizer: SgdOptimizer, step: int = 2
) -> JaxOptimizerState:
    layout = checkpoint_layout()
    return JaxOptimizerState(
        OptimizerState(optimizer.optimizer_id, layout.logical_paths, step=step),
        optimizer.jax_state_descriptor(layout),
        {
            "step": np.asarray(step, dtype=np.int32),
            "per_parameter_steps": {
                "trunk": {"weight": np.asarray(step, dtype=np.int32)}
            },
        },
    )


def checkpoint_objective():
    registry = build_default_objective_registry()
    selection = registry.select(CANONICAL_MSE_IDENTITY)
    config = ObjectiveConfig(CANONICAL_MSE_IDENTITY, {"reduction": "mean"})
    resolved = ResolvedObjectiveSelection(ObjectiveScope(), "final_output")
    descriptor = registry.execution_descriptor(
        selection=selection,
        config=config,
        resolved_selection=resolved,
    )
    return selection, config, resolved, descriptor


def checkpoint_payload(optimizer: SgdOptimizer) -> JaxLearningCheckpointV3:
    layout = checkpoint_layout()
    state = checkpoint_optimizer_state(optimizer)
    config_digest = "literal-architecture-config-digest"
    (
        objective_selection,
        objective_config,
        resolved_objective_selection,
        objective_descriptor,
    ) = checkpoint_objective()
    return JaxLearningCheckpointV3(
        "literal-runtime-reference",
        LearningState("literal-run", global_step=2, optimizer_step=2),
        state,
        {"trunk": {"weight": np.asarray([1.0], dtype=np.float32)}},
        {"count": np.asarray(2, dtype=np.int32)},
        layout,
        ArchitectureState("literal-architecture-state"),
        HFPreservationReference(
            "hf_descriptor.v1",
            "literal-hf-descriptor-digest",
            "literal-model",
            layout.architecture_id,
            "literal-tokenizer",
            8,
            "literal-special-token-digest",
            layout.digest(),
            config_digest,
        ),
        config_digest,
        "literal-parameter-catalog-digest",
        objective_config=objective_config,
        resolved_objective_selection=resolved_objective_selection,
        objective_descriptor=objective_descriptor,
        objective_registry_selection=objective_selection.to_dict(),
    )


def write_valid_checkpoint(directory: Path) -> tuple[SgdOptimizer, ParameterTreeLayout]:
    optimizer = SgdOptimizer()
    save_learning_checkpoint_v3(
        checkpoint_payload(optimizer), directory, optimizer=optimizer
    )
    return optimizer, checkpoint_layout()


def load_valid_checkpoint(
    directory: Path, optimizer: SgdOptimizer, layout: ParameterTreeLayout
) -> JaxLearningCheckpointV3:
    selection, config, resolved, descriptor = checkpoint_objective()
    return load_learning_checkpoint_v3(
        directory,
        optimizer=optimizer,
        parameter_layout=layout,
        runtime_reference="literal-runtime-reference",
        expected_objective_descriptor=descriptor,
        expected_objective_config=config,
        expected_resolved_objective_selection=resolved,
        expected_objective_selection=selection,
    )


def clone_directory(source: Path, destination: Path) -> Path:
    shutil.copytree(source, destination)
    return destination


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def rewrite_integrity(directory: Path) -> None:
    manifest_path = directory / "manifest.json"
    manifest = read_json(manifest_path)
    for name in manifest["files"]:
        payload = (directory / name).read_bytes()
        manifest["hashes"][name] = hashlib.sha256(payload).hexdigest()
        manifest["sizes"][name] = len(payload)
    bare = dict(manifest)
    bare.pop("integrity", None)
    manifest["integrity"] = {
        "algorithm": "sha256",
        "manifest_digest": hashlib.sha256(
            (json.dumps(bare, sort_keys=True, separators=(",", ":")) + "\n").encode(
                "utf-8"
            )
        ).hexdigest(),
    }
    write_json(manifest_path, manifest)


def replace_npz_leaf(
    path: Path, descriptor: dict[str, Any], keypath: tuple[str, ...], value: Any
) -> None:
    tree = read_deterministic_npz(path, descriptor)
    branch: dict[str, Any] = tree
    for part in keypath[:-1]:
        branch = branch[part]
    branch[keypath[-1]] = value
    write_deterministic_npz(path, tree)


def rewrite_npz_members(
    path: Path,
    *,
    additions: dict[str, bytes] | None = None,
    removals: tuple[str, ...] = (),
) -> None:
    """Apply a literal NPZ member mutation without timestamp nondeterminism."""

    additions = {} if additions is None else dict(additions)
    removed = set(removals)
    with zipfile.ZipFile(path, "r") as archive:
        members = {
            name: archive.read(name)
            for name in archive.namelist()
            if name not in removed
        }
    members.update(additions)
    rewritten = path.with_suffix(path.suffix + ".rewritten")
    with zipfile.ZipFile(rewritten, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(members):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, members[name])
    rewritten.replace(path)


def carry_descriptor(directory: Path) -> str:
    return descriptor_digest(read_json(directory / "architecture_carry.json"))


def state_with_envelope_step(
    optimizer: SgdOptimizer, step: int
) -> JaxLearningCheckpointV3:
    state = checkpoint_optimizer_state(optimizer)
    altered = replace(state, envelope=replace(state.envelope, step=step))
    return replace(checkpoint_payload(optimizer), optimizer_state=altered)


__all__ = [
    "carry_descriptor",
    "checkpoint_layout",
    "checkpoint_optimizer_state",
    "checkpoint_payload",
    "clone_directory",
    "load_valid_checkpoint",
    "read_json",
    "rewrite_npz_members",
    "replace_npz_leaf",
    "rewrite_integrity",
    "state_with_envelope_step",
    "write_json",
    "write_valid_checkpoint",
]
