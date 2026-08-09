"""Architecture-owned, checkpoint-neutral mapping carry descriptors."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


def _encode_keypath(keypath: tuple[str, ...]) -> str:
    return (
        "k_"
        + "_".join(f"{len(part.encode()):08x}{part.encode().hex()}" for part in keypath)
        + ".npy"
    )


def describe_mapping_carry(tree: Mapping[str, Any]) -> dict[str, Any]:
    leaves: list[dict[str, Any]] = []

    def visit(value: Any, path: tuple[str, ...]) -> None:
        if not isinstance(value, Mapping) or not value:
            if not path:
                raise ValueError("carry trees must be nonempty mappings")
            leaves.append(
                {
                    "keypath": list(path),
                    "member": _encode_keypath(path),
                    "shape": list(getattr(value, "shape", ())),
                    "dtype": str(getattr(value, "dtype", type(value).__name__)),
                }
            )
            return
        for key in sorted(value):
            if not isinstance(key, str) or not key:
                raise ValueError("carry mapping keys must be nonempty strings")
            visit(value[key], (*path, key))

    visit(tree, ())
    return {
        "schema_version": "jax_pytree_payload.v1",
        "codec": "radjax_deterministic_npz.v1",
        "tree_kind": "mapping_only",
        "leaves": leaves,
    }


def carry_descriptor_digest(descriptor: Mapping[str, Any]) -> str:
    payload = (
        json.dumps(descriptor, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    return hashlib.sha256(payload).hexdigest()


__all__ = ["carry_descriptor_digest", "describe_mapping_carry"]
